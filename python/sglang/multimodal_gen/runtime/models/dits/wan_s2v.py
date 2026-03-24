# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanS2VTransformer3DModel(torch.nn.Module, OffloadableDiTMixin):
    _aliases = ["WanS2VTransformer3DModel"]
    _sdpa_warned_padding_mask = False
    layer_names = ["blocks"]

    def __init__(
        self,
        noise_model: torch.nn.Module,
        *,
        component_model_path: str,
        config: dict[str, Any],
        config_obj: Any,
        device: torch.device,
        reference_text_encoder: Any | None = None,
        reference_audio_encoder: Any | None = None,
        reference_vae: Any | None = None,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.blocks = getattr(noise_model, "blocks", None)
        self.component_model_path = component_model_path
        self.config = config
        self.device_ = device
        self.param_dtype = config_obj.param_dtype
        self.num_train_timesteps = int(config_obj.num_train_timesteps)
        self.sample_neg_prompt = config_obj.sample_neg_prompt
        self.motion_frames = int(config_obj.transformer.motion_frames)
        self.drop_first_motion = bool(config_obj.drop_first_motion)
        self.fps = int(config_obj.sample_fps)
        self.audio_sample_m = 0
        self.supports_standard_denoising = True
        self.uses_native_components = bool(config.get("use_native_components", True))
        self.reference_text_encoder = reference_text_encoder
        self.reference_audio_encoder = reference_audio_encoder
        self.reference_vae = reference_vae
        self.t5_cpu = bool(config.get("t5_cpu", False))

    @property
    def device(self):
        return self.device_

    @staticmethod
    def _resolve_existing_path(
        component_model_path: str, path_value: str | None
    ) -> str:
        if not path_value:
            raise ValueError(
                "WanS2VTransformer3DModel config is missing a required path"
            )
        path_value = os.path.expanduser(path_value)
        if os.path.isabs(path_value):
            resolved = path_value
        else:
            resolved = os.path.join(component_model_path, path_value)
        if not os.path.exists(resolved):
            raise ValueError(f"Resolved path does not exist: {resolved}")
        return resolved

    @staticmethod
    def _sdpa_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del window_size, deterministic, version
        if q_scale is not None:
            q = q * q_scale
        if softmax_scale is not None:
            q = q * softmax_scale

        if q_lens is not None or k_lens is not None:
            if not WanS2VTransformer3DModel._sdpa_warned_padding_mask:
                logger.warning(
                    "Wan S2V SDPA fallback ignores q_lens/k_lens padding masks; "
                    "this is intended only for compatibility validation."
                )
                WanS2VTransformer3DModel._sdpa_warned_padding_mask = True

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        )
        return out.transpose(1, 2).contiguous()

    @classmethod
    def _patch_attention_backend(cls, config: dict[str, Any]):
        backend = str(config.get("attention_backend", "auto")).lower()
        if backend not in {"auto", "flash", "sdpa"}:
            raise ValueError(f"Unsupported Wan attention backend: {backend}")

        module_attention = importlib.import_module("wan.modules.attention")
        if backend == "flash":
            logger.info("Using official Wan flash attention backend")
            return

        if backend == "auto":
            has_fa2 = getattr(
                importlib.import_module("flash_attn"),
                "flash_attn_varlen_func",
                None,
            )
            has_fa3 = getattr(module_attention, "FLASH_ATTN_3_AVAILABLE", False)
            if has_fa2 or not has_fa3:
                logger.info(
                    "Using official Wan flash attention backend (auto detected)"
                )
                return
            backend = "sdpa"

        logger.info("Patching official Wan attention backend to SDPA fallback")
        module_attention.flash_attention = cls._sdpa_flash_attention
        module_attention.FLASH_ATTN_2_AVAILABLE = False
        module_attention.FLASH_ATTN_3_AVAILABLE = False

        module_model = importlib.import_module("wan.modules.model")
        module_model.flash_attention = cls._sdpa_flash_attention
        for module_name in (
            "wan.modules.s2v.model_s2v",
            "wan.modules.s2v.motioner",
            "wan.distributed.ulysses",
        ):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            setattr(module, "flash_attention", cls._sdpa_flash_attention)

    @classmethod
    def from_component_path(
        cls,
        component_model_path: str,
        server_args,
        config: dict[str, Any],
    ):
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root", "official_code")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root", "checkpoints")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        task_name = str(config.get("wan_task_name", "s2v-14B"))
        module_configs = importlib.import_module("wan.configs")
        module_s2v = importlib.import_module("wan.speech2video")
        cls._patch_attention_backend(config)

        wan_configs = getattr(module_configs, "WAN_CONFIGS", None)
        if not isinstance(wan_configs, dict) or task_name not in wan_configs:
            raise ValueError(
                f"Official Wan config {task_name!r} not found under {code_root}"
            )

        config_obj = wan_configs[task_name]
        local_device = get_local_torch_device()
        device_id = 0 if local_device.index is None else int(local_device.index)
        use_sp = (
            max(
                getattr(server_args, "sp_degree", 1) or 1,
                getattr(server_args, "ulysses_degree", 1) or 1,
            )
            > 1
        )
        engine = module_s2v.WanS2V(
            config=config_obj,
            checkpoint_dir=checkpoint_root,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=use_sp,
            t5_cpu=bool(config.get("t5_cpu", False)),
            init_on_cpu=bool(config.get("init_on_cpu", True)),
            convert_model_dtype=bool(config.get("convert_model_dtype", False)),
        )
        noise_model = engine.noise_model

        reference_text_encoder = None
        reference_audio_encoder = None
        reference_vae = None
        if not bool(config.get("use_native_components", True)):
            reference_text_encoder = engine.text_encoder
            reference_audio_encoder = engine.audio_encoder
            reference_vae = engine.vae
        else:
            if hasattr(engine, "text_encoder"):
                engine.text_encoder = None
            if hasattr(engine, "audio_encoder"):
                engine.audio_encoder = None
            if hasattr(engine, "vae"):
                engine.vae = None

        return cls(
            noise_model=noise_model,
            component_model_path=component_model_path,
            config=config,
            config_obj=config_obj,
            device=local_device,
            reference_text_encoder=reference_text_encoder,
            reference_audio_encoder=reference_audio_encoder,
            reference_vae=reference_vae,
        )

    def get_default_negative_prompt(self) -> str:
        return self.sample_neg_prompt

    def encode_prompts_reference(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        offload_model: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_text_encoder is None:
            raise RuntimeError("Reference text encoder is not initialized")
        if not self.t5_cpu:
            self.reference_text_encoder.model.to(self.device)
            context = self.reference_text_encoder([prompt], self.device)
            context_null = self.reference_text_encoder([negative_prompt], self.device)
            if offload_model:
                self.reference_text_encoder.model.cpu()
        else:
            cpu_device = torch.device("cpu")
            context = self.reference_text_encoder([prompt], cpu_device)
            context_null = self.reference_text_encoder([negative_prompt], cpu_device)
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]
        return context[0:1], context_null[0:1]

    def encode_audio_reference(
        self,
        *,
        audio_path: str,
        infer_frames: int,
    ) -> tuple[torch.Tensor, int]:
        if self.reference_audio_encoder is None:
            raise RuntimeError("Reference audio encoder is not initialized")
        embeddings = self.reference_audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True
        )
        bucket, num_repeat = self.reference_audio_encoder.get_audio_embed_bucket_fps(
            embeddings,
            fps=self.fps,
            batch_frames=infer_frames,
            m=self.audio_sample_m,
        )
        bucket = bucket.to(self.device, self.param_dtype)
        bucket = bucket.unsqueeze(0)
        if bucket.ndim == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.ndim == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        return bucket, num_repeat

    def read_last_n_frames(
        self,
        video_path: str,
        n_frames: int,
        *,
        target_fps: int = 16,
        reverse: bool = False,
    ):
        from decord import VideoReader

        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)
        interval = max(1, round(original_fps / target_fps))
        required_span = (n_frames - 1) * interval
        start_frame = max(0, total_frames - required_span - 1) if not reverse else 0
        sampled_indices = []
        for i in range(n_frames):
            idx = start_frame + i * interval
            if idx >= total_frames:
                break
            sampled_indices.append(idx)
        return vr.get_batch(sampled_indices).asnumpy()

    def load_pose_cond_reference(
        self,
        *,
        pose_video: str | None,
        num_repeat: int,
        infer_frames: int,
        size: tuple[int, int],
    ) -> list[torch.Tensor]:
        if self.reference_vae is None:
            raise RuntimeError("Reference VAE is not initialized")

        height, width = size
        if pose_video is not None:
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True,
            )
            resize = transforms.Resize(min(height, width))
            crop = transforms.CenterCrop((height, width))
            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop(resize(cond_tensor)).permute(1, 0, 2, 3).unsqueeze(0)
            padding_frames = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat(
                [
                    cond_tensor,
                    -torch.ones([1, 3, padding_frames, height, width]),
                ],
                dim=2,
            )
            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, height, width])]

        conditions = []
        for cond in cond_tensors:
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_lat = torch.stack(
                self.reference_vae.encode(
                    cond.to(dtype=self.param_dtype, device=self.device)
                )
            )[:, :, 1:].cpu()
            conditions.append(cond_lat)
        return conditions

    def prepare_reference_s2v_inputs(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        audio_path: str,
        pose_video_path: str | None,
        num_frames: int,
    ) -> dict[str, Any]:
        if self.reference_vae is None:
            raise RuntimeError("Reference VAE is not initialized")
        infer_frames = self._normalize_infer_frames(num_frames)
        height, width = self.get_generation_size(image_path=image_path)
        resize = transforms.Resize(min(height, width))
        crop = transforms.CenterCrop((height, width))
        to_tensor = transforms.ToTensor()

        model_pic = crop(resize(Image.open(image_path).convert("RGB")))
        ref_pixel_values = to_tensor(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.reference_vae.dtype, device=self.reference_vae.device
        )
        ref_latents = torch.stack(self.reference_vae.encode(ref_pixel_values))

        motion_frames = self.motion_frames
        motion_pixels = torch.zeros(
            [1, 3, motion_frames, height, width],
            dtype=self.param_dtype,
            device=self.device,
        )
        motion_latents = torch.stack(self.reference_vae.encode(motion_pixels))
        lat_motion_frames = (motion_frames + 3) // 4

        audio_input, max_num_repeat = self.encode_audio_reference(
            audio_path=audio_path,
            infer_frames=infer_frames,
        )
        if max_num_repeat < 1:
            raise ValueError(f"Audio path produced no valid clips: {audio_path}")

        cond_states = self.load_pose_cond_reference(
            pose_video=pose_video_path,
            num_repeat=1,
            infer_frames=infer_frames,
            size=(height, width),
        )[0].to(dtype=self.param_dtype, device=self.device)
        if pose_video_path is None:
            cond_states = cond_states * 0

        prompt_embeds, negative_prompt_embeds = self.encode_prompts_reference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            offload_model=bool(self.config.get("offload_model", True)),
        )
        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
        return {
            "height": int(height),
            "width": int(width),
            "infer_frames": int(infer_frames),
            "latent_shape": (1, 16, lat_target_frames, height // 8, width // 8),
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "ref_latents": ref_latents,
            "motion_latents": motion_latents,
            "cond_states": cond_states,
            "audio_input": audio_input[..., :infer_frames],
            "motion_frames": [motion_frames, lat_motion_frames],
            "drop_motion_frames": bool(self.drop_first_motion),
        }

    def decode_reference_output(
        self,
        *,
        latents: torch.Tensor,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        infer_frames: int,
        drop_motion_frames: bool,
    ) -> torch.Tensor:
        if self.reference_vae is None:
            raise RuntimeError("Reference VAE is not initialized")
        if drop_motion_frames:
            decode_latents = torch.cat([ref_latents, latents], dim=2)
        else:
            decode_latents = torch.cat([motion_latents, latents], dim=2)
        image = torch.stack(self.reference_vae.decode(decode_latents))
        image = image[:, :, -infer_frames:]
        if drop_motion_frames:
            image = image[:, :, 3:]
        return image

    def _normalize_infer_frames(self, num_frames: int) -> int:
        infer_frames = max(int(num_frames) - 1, 4)
        if infer_frames % 4 != 0:
            infer_frames = max((infer_frames // 4) * 4, 4)
        return infer_frames

    def get_size_less_than_area(
        self,
        height: int,
        width: int,
        *,
        target_area: int = 1024 * 704,
        divisor: int = 64,
    ) -> tuple[int, int]:
        if height * width <= target_area:
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            max_upper_area = target_area
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area
            min_scale = (-b + (b**2 - 2 * a * c) ** 0.5) / (2 * a)
            max_scale = (max_upper_area / (height * width)) ** 0.5

        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            padded_height = new_height + pad_height
            padded_width = new_width + pad_width
            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width

        aspect_ratio = width / height
        target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
        target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)
        return target_height, target_width

    def get_generation_size(self, *, image_path: str) -> tuple[int, int]:
        ref_image = np.array(Image.open(image_path).convert("RGB"))
        height, width = ref_image.shape[:2]
        return self.get_size_less_than_area(
            int(height),
            int(width),
            target_area=int(self.config.get("max_area", 720 * 1280)),
        )

    def prepare_standard_s2v_latents(
        self,
        *,
        latent_shape: tuple[int, ...],
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        return randn_tensor(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.param_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        ref_latents: torch.Tensor | None = None,
        motion_latents: torch.Tensor | None = None,
        cond_states: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        motion_frames: list[int] | tuple[int, int] | None = None,
        drop_motion_frames: bool = False,
        add_last_motion: bool | int | None = None,
        seq_len: int | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance
        if timestep is None and "t" in kwargs:
            timestep = kwargs.pop("t")
        if timestep is None:
            raise ValueError("Wan S2V forward requires timestep")
        if encoder_hidden_states is None:
            raise ValueError("Wan S2V forward requires encoder_hidden_states")
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) == 0:
                raise ValueError("encoder_hidden_states list cannot be empty")
            context = encoder_hidden_states
        elif isinstance(encoder_hidden_states, torch.Tensor):
            if encoder_hidden_states.ndim == 3:
                context = [
                    encoder_hidden_states[i]
                    for i in range(encoder_hidden_states.shape[0])
                ]
            else:
                context = [encoder_hidden_states]
        else:
            raise TypeError(
                "Wan S2V encoder_hidden_states must be a tensor or list of tensors"
            )
        if seq_len is None:
            seq_len = int(
                hidden_states.shape[2]
                * hidden_states.shape[3]
                * hidden_states.shape[4]
                // 4
            )
        if ref_latents is None or motion_latents is None or cond_states is None:
            raise ValueError(
                "Wan S2V forward requires ref_latents, motion_latents, and cond_states"
            )
        if motion_frames is None:
            motion_frames = [
                self.motion_frames,
                (self.motion_frames + 3) // 4,
            ]
        if add_last_motion is None:
            add_last_motion = 2
        output = self.noise_model(
            hidden_states,
            t=timestep,
            context=context,
            seq_len=seq_len,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            cond_states=cond_states,
            audio_input=audio_input,
            motion_frames=motion_frames,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            **kwargs,
        )
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError(
                    f"Wan S2V noise model returned unexpected output length: {len(output)}"
                )
            return output[0]
        return output


EntryClass = WanS2VTransformer3DModel
