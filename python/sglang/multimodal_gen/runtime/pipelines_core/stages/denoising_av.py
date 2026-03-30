import copy
import math
import os
import time
from io import BytesIO

import av
import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _save_intermediate_tensor(file_name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None or not os.environ.get("SAVE_INTERMEDIATE_TENSORS"):
        return
    save_dir = os.environ.get("EXPERIMENTS_DIR", "/tmp")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(save_dir, file_name))


class LTX2AVDenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(self, transformer, scheduler, vae=None, audio_vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.audio_vae = audio_vae

    @staticmethod
    def _get_video_latent_num_frames_for_model(
        batch: Req, server_args: ServerArgs, latents: torch.Tensor
    ) -> int:
        """Return the latent-frame length the DiT model should see.

        - If video latents were time-sharded for SP and are packed as token latents
          ([B, S, D]), the model only sees the local shard and must use the local
          latent-frame count (stored on the batch during SP sharding).
        - Otherwise, fall back to the global latent-frame count inferred from the
          requested output frames and the VAE temporal compression ratio.
        """
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        is_token_latents = isinstance(latents, torch.Tensor) and latents.ndim == 3

        if did_sp_shard and is_token_latents:
            if not hasattr(batch, "sp_video_latent_num_frames"):
                raise ValueError(
                    "SP-sharded LTX2 token latents require `batch.sp_video_latent_num_frames` "
                    "to be set by `LTX2PipelineConfig.shard_latents_for_sp()`."
                )
            return int(batch.sp_video_latent_num_frames)

        pc = server_args.pipeline_config
        return int((batch.num_frames - 1) // int(pc.vae_temporal_compression) + 1)

    @staticmethod
    def _truncate_sp_padded_token_latents(
        batch: Req, latents: torch.Tensor
    ) -> torch.Tensor:
        """Remove token padding introduced by SP time-sharding (if applicable)."""
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard or not (
            isinstance(latents, torch.Tensor) and latents.ndim == 3
        ):
            return latents

        raw_shape = getattr(batch, "raw_latent_shape", None)
        if not (isinstance(raw_shape, tuple) and len(raw_shape) == 3):
            return latents

        orig_s = int(raw_shape[1])
        cur_s = int(latents.shape[1])
        if cur_s == orig_s:
            return latents
        if cur_s < orig_s:
            raise ValueError(
                f"Unexpected gathered token-latents seq_len {cur_s} < original seq_len {orig_s}."
            )
        return latents[:, :orig_s, :].contiguous()

    def _maybe_enable_cache_dit(self, num_inference_steps: int, batch: Req) -> None:
        """Disable cache-dit for TI2V-style requests (image-conditioned), to avoid stale activations.

        NOTE: base denoising stage calls this hook with (num_inference_steps, batch).
        """
        if getattr(self, "_disable_cache_dit_for_request", False):
            return
        return super()._maybe_enable_cache_dit(num_inference_steps, batch)

    @staticmethod
    def _convert_velocity_to_x0(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        step_idx: int,
        scheduler,
    ) -> torch.Tensor:
        return sample - velocity * scheduler.sigmas[step_idx]

    @staticmethod
    def _convert_x0_to_velocity(
        sample: torch.Tensor,
        x0: torch.Tensor,
        step_idx: int,
        scheduler,
    ) -> torch.Tensor:
        return (sample - x0) / scheduler.sigmas[step_idx]

    @staticmethod
    def _resize_center_crop(
        img: PIL.Image.Image, *, width: int, height: int
    ) -> PIL.Image.Image:
        return img.resize((width, height), resample=PIL.Image.Resampling.BILINEAR)

    @staticmethod
    def _apply_video_codec_compression(
        img_array: np.ndarray, crf: int = 33
    ) -> np.ndarray:
        """Encode as a single H.264 frame and decode back to simulate compression artifacts."""
        if crf == 0:
            return img_array
        height, width = img_array.shape[0] // 2 * 2, img_array.shape[1] // 2 * 2
        img_array = img_array[:height, :width]
        buffer = BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height, stream.width = height, width
        frame = av.VideoFrame.from_ndarray(img_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
        container.close()
        buffer.seek(0)
        container = av.open(buffer)
        decoded = next(container.decode(container.streams.video[0]))
        container.close()
        return decoded.to_ndarray(format="rgb24")

    @staticmethod
    def _resize_center_crop_tensor(
        img: PIL.Image.Image,
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        apply_codec_compression: bool = True,
        codec_crf: int = 33,
    ) -> torch.Tensor:
        """Resize, center-crop, and normalize to [1, C, 1, H, W] tensor in [-1, 1]."""
        img_array = np.array(img).astype(np.uint8)[..., :3]
        if apply_codec_compression:
            img_array = LTX2AVDenoisingStage._apply_video_codec_compression(
                img_array, crf=codec_crf
            )
        tensor = (
            torch.from_numpy(img_array.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
        )
        src_h, src_w = tensor.shape[2], tensor.shape[3]
        scale = max(height / src_h, width / src_w)
        new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, left = (new_h - height) // 2, (new_w - width) // 2
        tensor = tensor[:, :, top : top + height, left : left + width]
        return ((tensor / 127.5 - 1.0).to(dtype=dtype)).unsqueeze(2)

    @staticmethod
    def _pil_to_normed_tensor(img: PIL.Image.Image) -> torch.Tensor:
        # PIL -> numpy [0,1] -> torch [B,C,H,W], then [-1,1]
        arr = pil_to_numpy(img)
        t = numpy_to_pt(arr)
        return normalize(t)

    @staticmethod
    def _should_apply_ltx2_ti2v(batch: Req) -> bool:
        """True if we have an image-latent token prefix to condition with.

        SP note: when token latents are time-sharded, only the rank that owns the
        *global* first latent frame should apply TI2V conditioning (rank with start_frame==0).
        """
        if (
            batch.image_latent is None
            or int(getattr(batch, "ltx2_num_image_tokens", 0)) <= 0
        ):
            return False
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard:
            return True
        return int(getattr(batch, "sp_video_start_frame", 0)) == 0

    def _prepare_ltx2_image_latent(self, batch: Req, server_args: ServerArgs) -> None:
        """Encode `batch.image_path` into packed token latents for LTX-2 TI2V."""
        if (
            batch.image_latent is not None
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        ):
            return
        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if batch.image_path is None:
            return
        if batch.width is None or batch.height is None:
            raise ValueError("width/height must be provided for LTX-2 TI2V.")
        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 TI2V.")

        image_path = (
            batch.image_path[0]
            if isinstance(batch.image_path, list)
            else batch.image_path
        )

        img = load_image(image_path)
        batch.condition_image = self._resize_center_crop(
            img, width=int(batch.width), height=int(batch.height)
        )

        latents_device = (
            batch.latents.device
            if isinstance(batch.latents, torch.Tensor)
            else torch.device("cpu")
        )
        encode_dtype = batch.latents.dtype
        original_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=latents_device, dtype=encode_dtype)
        vae_autocast_enabled = (
            original_dtype != torch.float32
        ) and not server_args.disable_autocast

        video_condition = self._resize_center_crop_tensor(
            img,
            width=int(batch.width),
            height=int(batch.height),
            device=latents_device,
            dtype=encode_dtype,
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=original_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(encode_dtype)

            latent_dist: DiagonalGaussianDistribution = self.vae.encode(video_condition)
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if mode == "argmax":
            latent = latent_dist.mode()
        elif mode == "sample":
            if batch.generator is None:
                raise ValueError("Generator must be provided for VAE sampling.")
            latent = latent_dist.sample(batch.generator)
        else:
            raise ValueError(f"Unsupported encode_sample_mode: {mode}")

        # Per-channel normalization: normalized = (x - mean) / std
        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
        latent = (latent - mean) / std

        packed = server_args.pipeline_config.maybe_pack_latents(
            latent, latent.shape[0], batch
        )
        if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
            raise ValueError("Expected packed image latents [B, S0, D].")

        # Fail-fast token count: must match one latent frame's tokens.
        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        latent_h = int(batch.height) // vae_sf
        latent_w = int(batch.width) // vae_sf
        expected_tokens = (latent_h // patch) * (latent_w // patch)
        if int(packed.shape[1]) != int(expected_tokens):
            raise ValueError(
                "LTX-2 conditioning token count mismatch: "
                f"{int(packed.shape[1])=} {int(expected_tokens)=}."
            )

        batch.image_latent = packed
        batch.ltx2_num_image_tokens = int(packed.shape[1])

        if batch.debug:
            logger.info(
                "LTX2 TI2V conditioning prepared: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(batch.image_latent.shape),
                batch.width,
                batch.height,
            )

        self.vae.to(original_dtype)
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
         Run the denoising loop.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with denoised latents.
        """
        # Disable cache-dit for image-conditioned requests (TI2V-style) for correctness/debuggability.
        self._disable_cache_dit_for_request = batch.image_path is not None

        # Prepare variables for the denoising loop

        prepared_vars = self._prepare_denoising_loop(batch, server_args)
        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        image_kwargs = prepared_vars["image_kwargs"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        seq_len = prepared_vars["seq_len"]
        guidance = prepared_vars["guidance"]

        audio_latents = batch.audio_latents
        audio_scheduler = copy.deepcopy(self.scheduler)

        # Prepare TI2V conditioning once (encode image -> patchify tokens).
        self._prepare_ltx2_image_latent(batch, server_args)

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=latents
        )
        latent_height = batch.height // server_args.pipeline_config.vae_scale_factor
        latent_width = batch.width // server_args.pipeline_config.vae_scale_factor

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []
        trajectory_audio_latents: list[torch.Tensor] = []

        # Run denoising loop
        denoising_start_time = time.time()

        # to avoid device-sync caused by timestep comparison
        is_warmup = batch.is_warmup
        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)
        timesteps_cpu = timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]

        do_ti2v = self._should_apply_ltx2_ti2v(batch)
        num_img_tokens = int(getattr(batch, "ltx2_num_image_tokens", 0))
        denoise_mask = None
        clean_latent = None
        if do_ti2v:
            if not (isinstance(latents, torch.Tensor) and latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            latents[:, :num_img_tokens, :] = batch.image_latent[
                :, :num_img_tokens, :
            ].to(device=latents.device, dtype=latents.dtype)
            denoise_mask = torch.ones(
                (latents.shape[0], latents.shape[1], 1),
                device=latents.device,
                dtype=torch.float32,
            )
            denoise_mask[:, :num_img_tokens, :] = 0.0
            clean_latent = latents.detach().clone()
            clean_latent[:, :num_img_tokens, :] = batch.image_latent[
                :, :num_img_tokens, :
            ].to(device=latents.device, dtype=latents.dtype)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{i}",
                        logger=logger,
                        metrics=batch.metrics,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                    ):
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )

                        # Predict noise residual
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)

                        latent_model_input = latents.to(target_dtype)
                        audio_latent_model_input = audio_latents.to(target_dtype)

                        if batch.do_classifier_free_guidance:
                            latent_model_input = torch.cat(
                                [latent_model_input, latent_model_input], dim=0
                            )
                            audio_latent_model_input = torch.cat(
                                [audio_latent_model_input, audio_latent_model_input],
                                dim=0,
                            )

                        latent_num_frames = latent_num_frames_for_model

                        # Audio latent dims
                        if audio_latent_model_input.ndim == 3:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[1]
                            )
                        elif audio_latent_model_input.ndim == 4:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[2]
                            )
                        else:
                            raise ValueError(
                                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
                            )

                        # LTX-2 model can generate coords internally.
                        video_coords = None
                        audio_coords = None

                        timestep = t_device.repeat(int(latent_model_input.shape[0]))
                        if do_ti2v and denoise_mask is not None:
                            timestep_video = timestep.unsqueeze(
                                -1
                            ) * denoise_mask.squeeze(-1)
                        else:
                            timestep_video = timestep
                        timestep_audio = timestep

                        # Conditions
                        encoder_hidden_states = batch.prompt_embeds[0]
                        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                        encoder_attention_mask = batch.prompt_attention_mask
                        if batch.do_classifier_free_guidance:
                            encoder_hidden_states = torch.cat(
                                [
                                    batch.negative_prompt_embeds[0],
                                    encoder_hidden_states,
                                ],
                                dim=0,
                            )
                            audio_encoder_hidden_states = torch.cat(
                                [
                                    batch.negative_audio_prompt_embeds[0],
                                    audio_encoder_hidden_states,
                                ],
                                dim=0,
                            )
                            encoder_attention_mask = torch.cat(
                                [batch.negative_attention_mask, encoder_attention_mask],
                                dim=0,
                            )

                        # Match official diffusers LTX-2 semantics:
                        # run a single batch-2 forward for CFG and apply guidance
                        # directly on the model velocity/noise prediction.
                        with set_forward_context(
                            current_timestep=i, attn_metadata=attn_metadata
                        ):
                            v_video, v_audio = current_model(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                timestep=timestep_video,
                                audio_timestep=timestep_audio,
                                encoder_attention_mask=encoder_attention_mask,
                                audio_encoder_attention_mask=encoder_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=batch.fps,
                                audio_num_frames=audio_num_frames_latent,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                return_latents=False,
                                return_dict=False,
                            )
                        v_video = v_video.float()
                        v_audio = v_audio.float()

                        if (
                            i == 0
                            and batch.extra.get("ltx2_phase") == "stage2"
                            and os.environ.get("SAVE_INTERMEDIATE_TENSORS")
                        ):
                            _save_intermediate_tensor(
                                "sglang_stage2_step0_velocity_video.pt",
                                v_video[1:2] if batch.do_classifier_free_guidance else v_video,
                            )
                            _save_intermediate_tensor(
                                "sglang_stage2_step0_velocity_audio.pt",
                                v_audio[1:2] if batch.do_classifier_free_guidance else v_audio,
                            )

                        if batch.do_classifier_free_guidance:
                            v_video_uncond, v_video_text = v_video.chunk(2)
                            x0_video = self._convert_velocity_to_x0(
                                latents, v_video_text, i, self.scheduler
                            )
                            x0_video_uncond = self._convert_velocity_to_x0(
                                latents, v_video_uncond, i, self.scheduler
                            )
                            x0_video = x0_video + (current_guidance_scale - 1.0) * (
                                x0_video - x0_video_uncond
                            )

                            v_audio_uncond, v_audio_text = v_audio.chunk(2)
                            x0_audio = self._convert_velocity_to_x0(
                                audio_latents, v_audio_text, i, audio_scheduler
                            )
                            x0_audio_uncond = self._convert_velocity_to_x0(
                                audio_latents, v_audio_uncond, i, audio_scheduler
                            )
                            x0_audio = x0_audio + (current_guidance_scale - 1.0) * (
                                x0_audio - x0_audio_uncond
                            )

                            if batch.guidance_rescale > 0.0:
                                x0_video = self._rescale_noise_cfg(
                                    x0_video,
                                    self._convert_velocity_to_x0(
                                        latents, v_video_text, i, self.scheduler
                                    ),
                                    guidance_rescale=batch.guidance_rescale,
                                )
                                x0_audio = self._rescale_noise_cfg(
                                    x0_audio,
                                    self._convert_velocity_to_x0(
                                        audio_latents, v_audio_text, i, audio_scheduler
                                    ),
                                    guidance_rescale=batch.guidance_rescale,
                                )

                            v_video = self._convert_x0_to_velocity(
                                latents, x0_video, i, self.scheduler
                            )
                            v_audio = self._convert_x0_to_velocity(
                                audio_latents, x0_audio, i, audio_scheduler
                            )

                        latents = self.scheduler.step(
                            v_video, t_device, latents, return_dict=False
                        )[0]
                        audio_latents = audio_scheduler.step(
                            v_audio, t_device, audio_latents, return_dict=False
                        )[0]

                        if do_ti2v:
                            latents[:, :num_img_tokens, :] = batch.image_latent[
                                :, :num_img_tokens, :
                            ].to(device=latents.device, dtype=latents.dtype)

                        latents = self.post_forward_for_ti2v_task(
                            batch, server_args, reserved_frames_mask, latents, z
                        )

                        # save trajectory latents if needed
                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)
                            if audio_latents is not None:
                                trajectory_audio_latents.append(audio_latents)

                        # Update progress bar
                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        if not is_warmup:
                            self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0 and not is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )

        batch.audio_latents = audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_audio_latents=trajectory_audio_latents,
            server_args=server_args,
            is_warmup=is_warmup,
        )

        return batch

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        trajectory_audio_latents: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
    ):
        # 1. Handle Trajectory (Video) - Copy from base
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )

        # If SP time-sharding padded whole frames worth of tokens, remove padding
        # after gather and before unpacking.
        latents = self._truncate_sp_padded_token_latents(batch, latents)

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        # 2. Handle Trajectory (Audio) - LTX-2 specific
        if trajectory_audio_latents:
            trajectory_audio_tensor = torch.stack(trajectory_audio_latents, dim=1)
            # We don't have SP support for audio latents yet (or needed?)
            batch.trajectory_audio_latents = trajectory_audio_tensor.cpu()

        # 3. Unpack and Denormalize
        # Call pipeline_config._unpad_and_unpack_latents
        # latents is video latents.
        # batch.audio_latents is audio latents.

        audio_latents = batch.audio_latents

        # NOTE: self.vae and self.audio_vae should be populated via __init__ or manual setting
        if self.vae is None or self.audio_vae is None:
            logger.warning(
                "VAE or Audio VAE not found in DenoisingStage. Skipping unpack and denormalize."
            )
            batch.latents = latents
            batch.audio_latents = audio_latents
            if type(self) is LTX2AVDenoisingStage:
                _save_intermediate_tensor("sglang_stage1_video_latent.pt", latents)
                _save_intermediate_tensor("sglang_stage1_audio_latent.pt", audio_latents)
        else:
            latents, audio_latents = (
                server_args.pipeline_config._unpad_and_unpack_latents(
                    latents, audio_latents, batch, self.vae, self.audio_vae
                )
            )

            batch.latents = latents
            batch.audio_latents = audio_latents

        if isinstance(self.transformer, OffloadableDiTMixin):
            for manager in self.transformer.layerwise_offload_managers:
                manager.release_all()

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs.

        Note: LTX-2 connector stage converts `prompt_embeds`/`negative_prompt_embeds`
        from list-of-tensors to a single tensor (video context) and stores audio
        context separately.
        """

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])

        # LTX-2 may carry prompt embeddings as either a tensor (preferred) or legacy list.
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            lambda x: V.is_tensor(x) or V.list_not_empty(x),
        )

        # Keep base expectation: image_embeds is always a list (may be empty).
        result.add_check("image_embeds", batch.image_embeds, V.is_list)

        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )

        # When CFG is enabled, negative prompt embeddings must exist (tensor or legacy list).
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x),
        )
        return result

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return batch.guidance_scale > 1.0


class LTX2RefinementStage(LTX2AVDenoisingStage):
    def __init__(
        self, transformer, scheduler, distilled_sigmas, vae=None, audio_vae=None
    ):
        super().__init__(transformer, scheduler, vae, audio_vae)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    @staticmethod
    def _normalize_stage2_video_latents(
        latents: torch.Tensor, vae, pipeline_config
    ) -> torch.Tensor:
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)
        if not (
            isinstance(latents_mean, torch.Tensor) and isinstance(latents_std, torch.Tensor)
        ):
            return latents

        scaling_factor = (
            getattr(getattr(vae, "config", None), "scaling_factor", None)
            or getattr(vae, "scaling_factor", None)
            or getattr(pipeline_config.vae_config.arch_config, "scaling_factor", None)
            or 1.0
        )
        latents_mean = latents_mean.to(device=latents.device, dtype=latents.dtype)
        latents_std = latents_std.to(device=latents.device, dtype=latents.dtype)

        if latents.ndim == 3:
            if latents.shape[-1] != latents_mean.numel():
                raise ValueError(
                    f"stage2 video latents last dim {latents.shape[-1]} does not match "
                    f"vae stats {latents_mean.numel()}"
                )
            return (latents - latents_mean.view(1, 1, -1)) * float(
                scaling_factor
            ) / latents_std.view(1, 1, -1)

        if latents.ndim == 5:
            return (latents - latents_mean.view(1, -1, 1, 1, 1)) * float(
                scaling_factor
            ) / latents_std.view(1, -1, 1, 1, 1)

        raise ValueError(
            f"Unsupported stage2 video latent shape for normalization: {latents.shape}"
        )

    @staticmethod
    def _normalize_stage2_audio_latents(latents: torch.Tensor, audio_vae) -> torch.Tensor:
        latents_mean = getattr(audio_vae, "latents_mean", None)
        latents_std = getattr(audio_vae, "latents_std", None)
        if not (
            isinstance(latents_mean, torch.Tensor) and isinstance(latents_std, torch.Tensor)
        ):
            return latents

        latents_mean = latents_mean.to(device=latents.device, dtype=latents.dtype)
        latents_std = latents_std.to(device=latents.device, dtype=latents.dtype)

        if latents.ndim == 3:
            if latents.shape[-1] != latents_mean.numel():
                raise ValueError(
                    f"stage2 audio latents last dim {latents.shape[-1]} does not match "
                    f"audio VAE stats {latents_mean.numel()}"
                )
            return (latents - latents_mean.view(1, 1, -1)) / latents_std.view(1, 1, -1)

        return (latents - latents_mean) / latents_std

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Re-noise stage-1 outputs using the official diffusers semantics:
        # noised = noise_scale * noise + (1 - noise_scale) * latents
        def _create_noised_state(
            latents: torch.Tensor,
            noise_scale: torch.Tensor,
            generator,
        ) -> torch.Tensor:
            noise = None
            if isinstance(generator, (list, tuple)):
                if len(generator) == 0:
                    generator = None
                elif len(generator) == 1:
                    generator = generator[0]
                else:
                    noise = torch.cat(
                        [
                            torch.randn(
                                (1, *latents.shape[1:]),
                                generator=sample_generator,
                                device=latents.device,
                                dtype=latents.dtype,
                            )
                            for sample_generator in generator
                        ],
                        dim=0,
                    )
                    generator = None

            if noise is None and generator is not None:
                noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
            elif noise is None:
                noise = torch.randn_like(latents)

            return noise_scale * noise + (1 - noise_scale) * latents

        noise_scale = self.distilled_sigmas[0].to(batch.latents.device)
        _save_intermediate_tensor("sglang_upscaled_video_latent.pt", batch.latents)
        generator = None
        if batch.seeds is not None and len(batch.seeds) > 0:
            generator = [
                torch.Generator(device=batch.latents.device).manual_seed(int(seed))
                for seed in batch.seeds
            ]
        elif batch.seed is not None:
            generator = torch.Generator(device=batch.latents.device).manual_seed(
                int(batch.seed)
            )
        else:
            generator = batch.generator

        batch.latents = self._normalize_stage2_video_latents(
            batch.latents, self.vae, server_args.pipeline_config
        )
        batch.latents = _create_noised_state(batch.latents, noise_scale, generator)
        _save_intermediate_tensor("sglang_stage2_noised_video_latent.pt", batch.latents)
        if batch.audio_latents is not None:
            batch.audio_latents = self._normalize_stage2_audio_latents(
                batch.audio_latents, self.audio_vae
            )
            batch.audio_latents = _create_noised_state(
                batch.audio_latents, noise_scale, generator
            )
            _save_intermediate_tensor(
                "sglang_stage2_noised_audio_latent.pt", batch.audio_latents
            )

        # 2. Run denoising loop with distilled_sigmas
        # Save original sigmas
        original_sigmas = self.scheduler.sigmas
        original_timesteps = self.scheduler.timesteps
        original_num_inference_steps = self.scheduler.num_inference_steps
        original_batch_num_inference_steps = batch.num_inference_steps
        original_batch_timesteps = batch.timesteps
        original_batch_sigmas = batch.sigmas

        # Match diffusers `retrieve_timesteps(..., sigmas=..., mu=...)` for stage 2.
        if isinstance(batch.latents, torch.Tensor) and batch.latents.ndim == 3:
            video_sequence_length = int(batch.latents.shape[1])
        elif isinstance(batch.latents, torch.Tensor) and batch.latents.ndim == 5:
            video_sequence_length = int(
                batch.latents.shape[2] * batch.latents.shape[3] * batch.latents.shape[4]
            )
        else:
            raise ValueError(
                f"Unsupported stage2 latent shape for LTX-2 refinement: {getattr(batch.latents, 'shape', None)}"
            )

        scheduler_cfg = self.scheduler.config
        base_seq_len = int(getattr(scheduler_cfg, "base_image_seq_len", 1024))
        max_seq_len = int(getattr(scheduler_cfg, "max_image_seq_len", 4096))
        base_shift = float(getattr(scheduler_cfg, "base_shift", 0.95))
        max_shift = float(getattr(scheduler_cfg, "max_shift", 2.05))
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = video_sequence_length * m + b

        self.scheduler.set_timesteps(
            sigmas=self.distilled_sigmas.tolist(),
            device=batch.latents.device,
            mu=mu,
        )
        stage2_num_inference_steps = len(self.scheduler.timesteps)
        self.scheduler.num_inference_steps = stage2_num_inference_steps
        batch.timesteps = self.scheduler.timesteps
        batch.sigmas = (
            self.scheduler.sigmas.tolist()
            if isinstance(self.scheduler.sigmas, torch.Tensor)
            else list(self.scheduler.sigmas)
        )
        batch.num_inference_steps = stage2_num_inference_steps

        # Call parent forward
        try:
            batch = super().forward(batch, server_args)
        finally:
            # Restore original sigmas
            self.scheduler.sigmas = original_sigmas
            self.scheduler.timesteps = original_timesteps
            self.scheduler.num_inference_steps = original_num_inference_steps
            batch.timesteps = original_batch_timesteps
            batch.sigmas = original_batch_sigmas
            batch.num_inference_steps = original_batch_num_inference_steps

        return batch

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return False  # Stage 2 uses simple denoising (no CFG)
