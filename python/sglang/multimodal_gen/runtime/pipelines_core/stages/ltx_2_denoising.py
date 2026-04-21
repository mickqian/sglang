import copy
from dataclasses import dataclass, field

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)

LTX23_TWO_STAGE_STAGE1_GUIDER_DEFAULTS: dict[str, object] = {
    "video_cfg_scale": 3.0,
    "video_stg_scale": 1.0,
    "video_rescale_scale": 0.7,
    "video_modality_scale": 3.0,
    "video_skip_step": 0,
    "video_stg_blocks": [28],
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 1.0,
    "audio_rescale_scale": 0.7,
    "audio_modality_scale": 3.0,
    "audio_skip_step": 0,
    "audio_stg_blocks": [28],
}

LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS: dict[str, object] = {
    "video_cfg_scale": 3.0,
    "video_stg_scale": 0.0,
    "video_rescale_scale": 0.45,
    "video_modality_scale": 3.0,
    "video_skip_step": 0,
    "video_stg_blocks": [],
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 0.0,
    "audio_rescale_scale": 1.0,
    "audio_modality_scale": 3.0,
    "audio_skip_step": 0,
    "audio_stg_blocks": [],
}


@dataclass(slots=True)
class LTX2DenoisingContext(DenoisingContext):
    """Loop-scoped denoising state for joint LTX-2 video and audio generation."""

    audio_latents: torch.Tensor | None = None
    audio_scheduler: object | None = None
    is_ltx23_variant: bool = False
    use_ltx23_legacy_one_stage: bool = False
    replicate_audio_for_sp: bool = False
    stage: str = "one_stage"
    latent_num_frames_for_model: int = 0
    latent_height: int = 0
    latent_width: int = 0
    denoise_mask: torch.Tensor | None = None
    clean_latent: torch.Tensor | None = None
    last_denoised_video: torch.Tensor | None = None
    last_denoised_audio: torch.Tensor | None = None
    trajectory_audio_latents: list[torch.Tensor] = field(default_factory=list)


class LTX2DenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        *,
        sampler_name: str = "euler",
        **kwargs,
    ):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.sampler_name = sampler_name

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
        return int(
            (batch.num_frames - 1)
            // int(pc.vae_config.arch_config.temporal_compression_ratio)
            + 1
        )

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

    def _get_ltx2_stage1_guider_params(
        self, batch: Req, server_args: ServerArgs, stage: str
    ) -> dict[str, object] | None:
        if stage != "stage1":
            return None
        request_params = batch.extra.get("ltx2_stage1_guider_params")
        if not is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return request_params
        default_params = copy.deepcopy(
            LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS
            if server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"
            else LTX23_TWO_STAGE_STAGE1_GUIDER_DEFAULTS
        )
        if request_params is None:
            return default_params
        for key, value in request_params.items():
            if value is None:
                continue
            default_params[key] = list(value) if isinstance(value, list) else value
        return default_params

    @staticmethod
    def _randn_like_with_batch_generators(
        reference_tensor: torch.Tensor, batch: Req
    ) -> torch.Tensor:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list):
            bsz = int(reference_tensor.shape[0])
            valid_generators = [g for g in generator if isinstance(g, torch.Generator)]
            if len(valid_generators) == 1:
                generator = valid_generators[0]
            elif len(valid_generators) >= bsz:
                generator = valid_generators[:bsz]
            else:
                generator = None
        elif not isinstance(generator, torch.Generator):
            generator = None

        return randn_tensor(
            reference_tensor.shape,
            generator=generator,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

    @staticmethod
    def _ltx2_should_skip_step(step_index: int, skip_step: int) -> bool:
        if skip_step == 0:
            return False
        return step_index % (skip_step + 1) != 0

    @staticmethod
    def _ltx2_apply_rescale(
        cond: torch.Tensor, pred: torch.Tensor, rescale_scale: float
    ) -> torch.Tensor:
        if rescale_scale == 0.0:
            return pred
        factor = cond.std() / pred.std()
        factor = rescale_scale * factor + (1.0 - rescale_scale)
        return pred * factor

    @staticmethod
    def _ltx2_apply_clean_latent_mask(
        latents: torch.Tensor,
        ctx: LTX2DenoisingContext,
    ) -> torch.Tensor:
        if ctx.denoise_mask is None or ctx.clean_latent is None:
            return latents
        return (
            latents.float() * ctx.denoise_mask
            + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
        ).to(dtype=latents.dtype)

    @staticmethod
    def _ltx2_phi_1(neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 1.0 + 0.5 * neg_h + (neg_h * neg_h) / 6.0
        return torch.where(small, series, torch.expm1(neg_h) / neg_h)

    @classmethod
    def _ltx2_phi_2(cls, neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 0.5 + neg_h / 6.0 + (neg_h * neg_h) / 24.0
        exact = (torch.expm1(neg_h) - neg_h) / (neg_h * neg_h)
        return torch.where(small, series, exact)

    @classmethod
    def _ltx2_get_res2s_coefficients(
        cls, h: torch.Tensor, c2: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a21 = c2 * cls._ltx2_phi_1(-h * c2)
        b2 = cls._ltx2_phi_2(-h) / c2
        b1 = cls._ltx2_phi_1(-h) - b2
        return a21, b1, b2

    @staticmethod
    def _ltx2_get_sde_coeff(
        sigma_next: torch.Tensor,
        *,
        sigma_up: torch.Tensor | None = None,
        sigma_down: torch.Tensor | None = None,
        sigma_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigma_down is not None:
            alpha_ratio = (1.0 - sigma_next) / (1.0 - sigma_down)
            sigma_up = torch.sqrt(
                torch.clamp(
                    sigma_next.square() - sigma_down.square() * alpha_ratio.square(),
                    min=0.0,
                )
            )
        elif sigma_up is not None:
            sigma_up = torch.minimum(sigma_up, sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else torch.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = torch.sqrt(
                torch.clamp(sigma_next.square() - sigma_up.square(), min=0.0)
            )
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = torch.zeros_like(sigma_next)
        return (
            torch.nan_to_num(alpha_ratio),
            torch.nan_to_num(sigma_down),
            torch.nan_to_num(sigma_up),
        )

    @classmethod
    def _ltx2_res2s_sde_step(
        cls,
        *,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        noise: torch.Tensor,
        eta: float = 0.5,
    ) -> torch.Tensor:
        alpha_ratio, sigma_down, sigma_up = cls._ltx2_get_sde_coeff(
            sigma_next,
            sigma_up=sigma_next * eta,
        )
        if bool((sigma_up == 0).any()) or bool((sigma_next == 0).any()):
            return denoised_sample.to(dtype=sample.dtype)
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next
        x_noised = (
            alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        )
        return x_noised.to(dtype=sample.dtype)

    @staticmethod
    def _prepare_ltx2_ti2v_clean_state(
        latents: torch.Tensor,
        image_latent: torch.Tensor,
        num_img_tokens: int,
        zero_clean_latent: bool,
        clean_latent_background: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = latents.clone()
        conditioned = image_latent[:, :num_img_tokens, :].to(
            device=latents.device, dtype=latents.dtype
        )
        latents[:, :num_img_tokens, :] = conditioned
        denoise_mask = torch.ones(
            (latents.shape[0], latents.shape[1], 1),
            device=latents.device,
            dtype=torch.float32,
        )
        denoise_mask[:, :num_img_tokens, :] = 0.0
        if clean_latent_background is not None:
            clean_latent = (
                clean_latent_background.detach()
                .clone()
                .to(device=latents.device, dtype=latents.dtype)
            )
        elif zero_clean_latent:
            clean_latent = torch.zeros_like(latents)
        else:
            clean_latent = latents.detach().clone()
        clean_latent[:, :num_img_tokens, :] = conditioned
        return latents, denoise_mask, clean_latent

    @staticmethod
    def _ltx2_velocity_to_x0(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            while sigma.ndim < sample.ndim:
                sigma = sigma.unsqueeze(-1)
            return (sample.float() - sigma * velocity.float()).to(sample.dtype)
        return (sample.float() - float(sigma) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _repeat_batch_dim(tensor: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Repeat along batch dim while preserving any tokenwise timestep layout."""
        if tensor.shape[0] == int(target_batch_size):
            return tensor
        if tensor.shape[0] <= 0 or int(target_batch_size) % int(tensor.shape[0]) != 0:
            raise ValueError(
                f"Cannot repeat tensor with batch={tensor.shape[0]} to target_batch_size={target_batch_size}"
            )
        repeat_factor = int(target_batch_size) // int(tensor.shape[0])
        return tensor.repeat(repeat_factor, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def _build_ltx2_sp_padding_mask(
        batch: Req,
        *,
        seq_len: int,
        batch_size: int,
        key: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        valid = getattr(batch, key, None)
        if valid is None:
            return None
        valid = int(valid)
        if valid <= 0 or valid >= int(seq_len):
            return None
        mask = torch.ones(
            (batch_size, int(seq_len)), device=device, dtype=torch.float32
        )
        mask[:, valid:] = 0.0
        return mask

    @staticmethod
    def _get_ltx_prompt_attention_mask(
        batch: Req,
        *,
        is_ltx23_variant: bool,
        negative: bool = False,
    ) -> torch.Tensor | None:
        if is_ltx23_variant:
            return None
        return (
            batch.negative_attention_mask if negative else batch.prompt_attention_mask
        )

    @classmethod
    def _should_use_ltx23_legacy_one_stage(
        cls,
        server_args: ServerArgs,
        pipeline_name: str | None,
    ) -> bool:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            return False
        if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return False
        return not is_ltx2_two_stage_pipeline_name(pipeline_name)

    @classmethod
    def _should_shard_ltx23_legacy_one_stage_audio_latents(
        cls,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        return bool(
            get_sp_world_size() > 1
            and is_ltx23_native_variant(
                server_args.pipeline_config.vae_config.arch_config
            )
            and cls._should_use_ltx23_legacy_one_stage(server_args, None)
            and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                batch.audio_latents
            )
        )

    @staticmethod
    def _should_use_ltx23_two_stage_cfg_pair_batch(
        batch: Req,
        ctx: LTX2DenoisingContext,
        server_args: ServerArgs,
    ) -> bool:
        """Use only cond/neg pair batching for native LTX-2.3 two-stage runs.

        Keeping perturbed/modality passes in the same large batch changes stage-1 guider
        numerics enough to drift from the sequential reference. Keep the CFG pair batched,
        but leave the extra guider branches as separate forwards across both single-GPU
        and TP runs so stage-1 semantics stay aligned.
        """
        if not is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            return False
        if not ctx.is_ltx23_variant or ctx.use_ltx23_legacy_one_stage:
            return False
        if int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0:
            return False
        if get_sp_world_size() != 1:
            return False
        return True

    @classmethod
    def _ltx2_calculate_guided_x0(
        cls,
        *,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        modality_scale: float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (cfg_scale - 1.0) * (cond - uncond_text)
            + stg_scale * (cond - uncond_perturbed)
            + (modality_scale - 1.0) * (cond - uncond_modality)
        )
        return cls._ltx2_apply_rescale(cond, pred, rescale_scale)

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        """LTX-2 TI2V applies image_latent in token space *after* SP sharding,
        so the base implementation must not shard it."""
        saved = batch.image_latent
        batch.image_latent = None
        super()._preprocess_sp_latents(batch, server_args)
        batch.image_latent = saved

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

    @staticmethod
    def _should_replicate_ltx23_audio_for_sp(
        batch: Req,
        server_args: ServerArgs,
        *,
        is_ltx23_variant: bool,
    ) -> bool:
        return False

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        """Extend the base context with LTX-2 audio, SP, and TI2V state."""
        self._disable_cache_dit_for_request = batch.image_path is not None
        base_ctx = super()._prepare_denoising_loop(batch, server_args)
        ctx = LTX2DenoisingContext(**base_ctx.to_kwargs())
        ctx.is_ltx23_variant = is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )
        phase = batch.extra.get("ltx2_phase")
        pipeline = self.pipeline() if self.pipeline else None
        pipeline_name = pipeline.pipeline_name if pipeline is not None else None
        ctx.use_ltx23_legacy_one_stage = self._should_use_ltx23_legacy_one_stage(
            server_args, pipeline_name
        )
        ctx.stage = (
            phase
            if phase is not None
            else ("stage1" if ctx.use_ltx23_legacy_one_stage else "one_stage")
        )
        ctx.audio_latents = batch.audio_latents
        # Video and audio keep separate scheduler state throughout the denoising loop.
        ctx.audio_scheduler = copy.deepcopy(self.scheduler)

        do_ti2v = self._should_apply_ltx2_ti2v(batch)

        if ctx.use_ltx23_legacy_one_stage:
            batch.ltx23_audio_replicated_for_sp = False
            batch.did_sp_shard_audio_latents = False
        else:
            ctx.replicate_audio_for_sp = self._should_replicate_ltx23_audio_for_sp(
                batch,
                server_args,
                is_ltx23_variant=ctx.is_ltx23_variant,
            )
            batch.ltx23_audio_replicated_for_sp = bool(ctx.replicate_audio_for_sp)
            if (
                ctx.is_ltx23_variant
                and get_sp_world_size() > 1
                and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                    batch.audio_latents
                )
                and not ctx.replicate_audio_for_sp
            ):
                (
                    batch.audio_latents,
                    batch.did_sp_shard_audio_latents,
                ) = server_args.pipeline_config.shard_audio_latents_for_sp(
                    batch, batch.audio_latents
                )
                ctx.audio_latents = batch.audio_latents
            else:
                batch.did_sp_shard_audio_latents = False

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        ctx.latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=ctx.latents
        )
        ctx.latent_height = (
            batch.height
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        ctx.latent_width = (
            batch.width
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        if do_ti2v:
            if not (isinstance(ctx.latents, torch.Tensor) and ctx.latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            clean_latent_background = getattr(
                batch, "ltx2_ti2v_clean_latent_background", None
            )
            if not (
                isinstance(clean_latent_background, torch.Tensor)
                and clean_latent_background.shape == ctx.latents.shape
            ):
                clean_latent_background = None
            # Keep conditioned tokens clean and reuse the mask during every step update.
            ctx.latents, ctx.denoise_mask, ctx.clean_latent = (
                self._prepare_ltx2_ti2v_clean_state(
                    latents=ctx.latents,
                    image_latent=batch.image_latent,
                    num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                    zero_clean_latent=ctx.is_ltx23_variant,
                    clean_latent_background=clean_latent_background,
                )
            )
        return ctx

    def _before_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Reset the mirrored audio scheduler before the shared loop begins."""
        if ctx.stage in ("stage1", "stage2"):
            pipeline = self.pipeline() if self.pipeline else None
            switch_lora_phase = (
                getattr(pipeline, "switch_lora_phase", None)
                if pipeline is not None
                else None
            )
            if callable(switch_lora_phase):
                switch_lora_phase(ctx.stage)
            ensure_phase_ready = (
                getattr(pipeline, "ensure_ltx2_phase_ready", None)
                if pipeline is not None
                else None
            )
            if callable(ensure_phase_ready):
                ensure_phase_ready(ctx.stage)
        super()._before_denoising_loop(ctx, batch, server_args)
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")
        ctx.audio_scheduler.set_begin_index(0)

    def _prepare_step_attn_metadata(
        self,
        ctx: LTX2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ):
        """Preserve the legacy LTX-2 attention-metadata contract."""
        # Legacy LTX-2 paths used the plain attention-metadata builder call here.
        del ctx, t_int, timesteps_cpu
        return self._build_attn_metadata(step_index, batch, server_args)

    def _run_denoising_step(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Run one joint video/audio denoising step with LTX-2-specific guidance."""
        if ctx.audio_latents is None:
            raise ValueError("LTX-2 requires audio latents for denoising.")
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")

        # 1. Read the scheduler sigma pair and derive the Euler delta.
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is None or not isinstance(sigmas, torch.Tensor):
            raise ValueError("Expected scheduler.sigmas to be a tensor for LTX-2.")
        sigma = sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_next = sigmas[step.step_index + 1].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        dt = sigma_next - sigma

        # 2. Materialize the current video/audio latent inputs in the compute dtype.
        latent_model_input = ctx.latents.to(ctx.target_dtype)
        audio_latent_model_input = ctx.audio_latents.to(ctx.target_dtype)
        stage1_guider_params = self._get_ltx2_stage1_guider_params(
            batch, server_args, ctx.stage
        )

        if audio_latent_model_input.ndim == 3:
            audio_num_frames_latent = int(audio_latent_model_input.shape[1])
        elif audio_latent_model_input.ndim == 4:
            audio_num_frames_latent = int(audio_latent_model_input.shape[2])
        else:
            raise ValueError(
                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
            )

        # 3. Prepare any LTX-specific RoPE coordinates and timestep layouts.
        video_coords = None
        audio_coords = None
        if not ctx.use_ltx23_legacy_one_stage:
            video_coords = server_args.pipeline_config.prepare_video_rope_coords_for_sp(
                step.current_model,
                batch,
                latent_model_input,
                num_frames=ctx.latent_num_frames_for_model,
                height=ctx.latent_height,
                width=ctx.latent_width,
            )
            audio_coords = server_args.pipeline_config.prepare_audio_rope_coords_for_sp(
                step.current_model,
                batch,
                audio_latent_model_input,
                num_frames=audio_num_frames_latent,
            )
        batch_size = int(latent_model_input.shape[0])
        timestep = step.t_device.expand(batch_size)
        if ctx.denoise_mask is not None:
            timestep_video = timestep.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
        elif ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
            timestep_video = timestep.view(batch_size, 1).expand(
                batch_size, int(latent_model_input.shape[1])
            )
        else:
            timestep_video = timestep

        if (
            ctx.is_ltx23_variant
            and not ctx.use_ltx23_legacy_one_stage
            and audio_latent_model_input.ndim == 3
        ):
            timestep_audio = timestep.view(batch_size, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1])
            )
        else:
            timestep_audio = timestep

        prompt_timestep_video = None
        prompt_timestep_audio = None
        if ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
            timestep_scale_multiplier = float(
                getattr(step.current_model, "timestep_scale_multiplier", 1000)
            )
            prompt_timestep_video = (
                sigma.to(device=latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)
            prompt_timestep_audio = (
                sigma.to(device=audio_latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)

        # 4. Build attention masks that account for SP padding and replicated audio.
        if ctx.use_ltx23_legacy_one_stage:
            video_self_attention_mask = None
            audio_self_attention_mask = None
            a2v_cross_attention_mask = None
            v2a_cross_attention_mask = None
        else:
            video_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=int(latent_model_input.shape[1]),
                batch_size=batch_size,
                key="sp_video_valid_token_count",
                device=latent_model_input.device,
            )
            audio_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=audio_num_frames_latent,
                batch_size=batch_size,
                key="sp_audio_valid_token_count",
                device=audio_latent_model_input.device,
            )
            a2v_cross_attention_mask = audio_self_attention_mask
            v2a_cross_attention_mask = video_self_attention_mask

        def build_model_kwargs(
            *,
            encoder_hidden_states: torch.Tensor,
            audio_encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None,
            skip_video_self_attn_blocks: tuple[int, ...] | None = None,
            skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
            disable_a2v_cross_attn: bool = False,
            disable_v2a_cross_attn: bool = False,
        ) -> dict[str, object]:
            use_native_ltx23_context_mask = not (
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            )
            kwargs: dict[str, object] = {
                "hidden_states": latent_model_input,
                "audio_hidden_states": audio_latent_model_input,
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "timestep": timestep_video,
                "audio_timestep": timestep_audio,
                # Native LTX-2.3 connector outputs already replace padded tokens
                # with learnable registers and zero video-only masked positions.
                # The official pipeline therefore leaves text context unmasked
                # inside the DiT cross-attention path for native (non-legacy)
                # LTX-2.3 execution.
                "encoder_attention_mask": (
                    encoder_attention_mask
                    if use_native_ltx23_context_mask
                    else None
                ),
                "audio_encoder_attention_mask": (
                    encoder_attention_mask
                    if use_native_ltx23_context_mask
                    else None
                ),
                "num_frames": ctx.latent_num_frames_for_model,
                "height": ctx.latent_height,
                "width": ctx.latent_width,
                "fps": batch.fps,
                "audio_num_frames": audio_num_frames_latent,
                "video_coords": video_coords,
                "audio_coords": audio_coords,
                "return_latents": False,
                "return_dict": False,
            }
            if not ctx.use_ltx23_legacy_one_stage:
                kwargs.update(
                    {
                        "prompt_timestep": prompt_timestep_video,
                        "audio_prompt_timestep": prompt_timestep_audio,
                        "video_self_attention_mask": video_self_attention_mask,
                        "audio_self_attention_mask": audio_self_attention_mask,
                        "a2v_cross_attention_mask": a2v_cross_attention_mask,
                        "v2a_cross_attention_mask": v2a_cross_attention_mask,
                        "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                        "legacy_ltx23_one_stage_semantics": False,
                    }
                )
            kwargs["ltx2_phase"] = ctx.stage
            if skip_video_self_attn_blocks is not None:
                kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
            if skip_audio_self_attn_blocks is not None:
                kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
            if disable_a2v_cross_attn:
                kwargs["disable_a2v_cross_attn"] = True
            if disable_v2a_cross_attn:
                kwargs["disable_v2a_cross_attn"] = True
            return kwargs

        # 5. Run the branch-specific LTX forward path and apply CFG/guider logic.
        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
        )
        has_negative_prompt_embeds = (
            isinstance(batch.negative_prompt_embeds, list)
            and len(batch.negative_prompt_embeds) > 0
        ) or torch.is_tensor(batch.negative_prompt_embeds)
        has_negative_audio_prompt_embeds = (
            isinstance(batch.negative_audio_prompt_embeds, list)
            and len(batch.negative_audio_prompt_embeds) > 0
        ) or torch.is_tensor(batch.negative_audio_prompt_embeds)
        use_official_cfg_path = (
            stage1_guider_params is None
            or not batch.do_classifier_free_guidance
            or not has_negative_prompt_embeds
            or not has_negative_audio_prompt_embeds
        )
        if use_official_cfg_path:
            encoder_hidden_states = batch.prompt_embeds[0]
            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
            encoder_attention_mask = prompt_attention_mask
            if batch.do_classifier_free_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
                audio_latent_model_input = torch.cat(
                    [audio_latent_model_input] * 2, dim=0
                )
                encoder_hidden_states = torch.cat(
                    [batch.negative_prompt_embeds[0], encoder_hidden_states], dim=0
                )
                audio_encoder_hidden_states = torch.cat(
                    [
                        batch.negative_audio_prompt_embeds[0],
                        audio_encoder_hidden_states,
                    ],
                    dim=0,
                )
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.cat(
                        [
                            self._get_ltx_prompt_attention_mask(
                                batch,
                                is_ltx23_variant=(
                                    ctx.is_ltx23_variant
                                    and not ctx.use_ltx23_legacy_one_stage
                                ),
                                negative=True,
                            ),
                            encoder_attention_mask,
                        ],
                        dim=0,
                    )
                cfg_batch_size = int(latent_model_input.shape[0])
                timestep_video = self._repeat_batch_dim(timestep_video, cfg_batch_size)
                timestep_audio = self._repeat_batch_dim(timestep_audio, cfg_batch_size)
                if prompt_timestep_video is not None:
                    prompt_timestep_video = self._repeat_batch_dim(
                        prompt_timestep_video, cfg_batch_size
                    )
                if prompt_timestep_audio is not None:
                    prompt_timestep_audio = self._repeat_batch_dim(
                        prompt_timestep_audio, cfg_batch_size
                    )
                if video_self_attention_mask is not None:
                    video_self_attention_mask = self._repeat_batch_dim(
                        video_self_attention_mask, cfg_batch_size
                    )
                if audio_self_attention_mask is not None:
                    audio_self_attention_mask = self._repeat_batch_dim(
                        audio_self_attention_mask, cfg_batch_size
                    )
                if a2v_cross_attention_mask is not None:
                    a2v_cross_attention_mask = self._repeat_batch_dim(
                        a2v_cross_attention_mask, cfg_batch_size
                    )
                if v2a_cross_attention_mask is not None:
                    v2a_cross_attention_mask = self._repeat_batch_dim(
                        v2a_cross_attention_mask, cfg_batch_size
                    )

            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                model_video, model_audio = step.current_model(
                    **build_model_kwargs(
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                )

            model_video = model_video.float()
            model_audio = model_audio.float()
            if batch.do_classifier_free_guidance:
                model_video_uncond, model_video_text = model_video.chunk(2)
                model_audio_uncond, model_audio_text = model_audio.chunk(2)
                model_video = model_video_uncond + (
                    batch.guidance_scale * (model_video_text - model_video_uncond)
                )
                model_audio = model_audio_uncond + (
                    batch.guidance_scale * (model_audio_text - model_audio_uncond)
                )
            else:
                model_video_uncond = model_video
                model_audio_uncond = model_audio
                model_video_text = model_video
                model_audio_text = model_audio

            sigma_val = float(sigma.item())
            video_sigma_for_x0: float | torch.Tensor = sigma_val
            if ctx.denoise_mask is not None:
                video_sigma_for_x0 = sigma.to(
                    device=ctx.latents.device, dtype=torch.float32
                ) * ctx.denoise_mask.squeeze(-1)
            cond_video_x0 = self._ltx2_velocity_to_x0(
                ctx.latents, model_video_text, video_sigma_for_x0
            )
            cond_audio_x0 = self._ltx2_velocity_to_x0(
                ctx.audio_latents, model_audio_text, sigma_val
            )
            neg_video_x0 = self._ltx2_velocity_to_x0(
                ctx.latents, model_video_uncond, video_sigma_for_x0
            )
            neg_audio_x0 = self._ltx2_velocity_to_x0(
                ctx.audio_latents, model_audio_uncond, sigma_val
            )
            guided_video_x0 = self._ltx2_velocity_to_x0(
                ctx.latents, model_video, video_sigma_for_x0
            )
            guided_audio_x0 = self._ltx2_velocity_to_x0(
                ctx.audio_latents, model_audio, sigma_val
            )

            next_video_latents = self.scheduler.step(
                model_video, step.t_device, ctx.latents, return_dict=False
            )[0]
            next_audio_latents = ctx.audio_scheduler.step(
                model_audio, step.t_device, ctx.audio_latents, return_dict=False
            )[0]
            if ctx.denoise_mask is not None and ctx.clean_latent is not None:
                next_video_latents = (
                    next_video_latents.float() * ctx.denoise_mask
                    + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
                ).to(dtype=next_video_latents.dtype)
            ctx.latents = self.post_forward_for_ti2v_task(
                batch, server_args, ctx.reserved_frames_mask, next_video_latents, ctx.z
            )
            ctx.audio_latents = next_audio_latents
            return

        encoder_hidden_states = batch.prompt_embeds[0]
        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
        encoder_attention_mask = prompt_attention_mask
        negative_encoder_hidden_states = batch.negative_prompt_embeds[0]
        negative_audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
        negative_encoder_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
            negative=True,
        )

        video_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["video_skip_step"])
        )
        audio_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["audio_skip_step"])
        )
        need_perturbed = (
            float(stage1_guider_params["video_stg_scale"]) != 0.0
            or float(stage1_guider_params["audio_stg_scale"]) != 0.0
        )
        need_modality = (
            float(stage1_guider_params["video_modality_scale"]) != 1.0
            or float(stage1_guider_params["audio_modality_scale"]) != 1.0
        )

        timestep_scale_multiplier = float(
            getattr(step.current_model, "timestep_scale_multiplier", 1000)
        )

        def build_stage1_step_inputs(
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
            sigma_value: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ]:
            latent_input = video_latents.to(ctx.target_dtype)
            audio_input = audio_latents.to(ctx.target_dtype)
            local_batch_size = int(latent_input.shape[0])
            if ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
                timestep_scalar = (
                    sigma_value.to(device=latent_input.device, dtype=torch.float32)
                    * timestep_scale_multiplier
                ).expand(local_batch_size)
            else:
                timestep_scalar = step.t_device.to(
                    device=latent_input.device, dtype=torch.float32
                ).expand(local_batch_size)

            if ctx.denoise_mask is not None:
                timestep_video_value = (
                    timestep_scalar.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
                )
            elif ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
                timestep_video_value = timestep_scalar.view(local_batch_size, 1).expand(
                    local_batch_size, int(latent_input.shape[1])
                )
            else:
                timestep_video_value = timestep_scalar

            if (
                ctx.is_ltx23_variant
                and not ctx.use_ltx23_legacy_one_stage
                and audio_input.ndim == 3
            ):
                timestep_audio_value = timestep_scalar.view(
                    local_batch_size, 1
                ).expand(local_batch_size, int(audio_input.shape[1]))
            else:
                timestep_audio_value = timestep_scalar

            prompt_timestep_video_value = None
            prompt_timestep_audio_value = None
            if ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
                prompt_timestep_video_value = (
                    sigma_value.to(device=latent_input.device, dtype=torch.float32)
                    * timestep_scale_multiplier
                ).expand(local_batch_size)
                prompt_timestep_audio_value = (
                    sigma_value.to(device=audio_input.device, dtype=torch.float32)
                    * timestep_scale_multiplier
                ).expand(local_batch_size)

            return (
                latent_input,
                audio_input,
                timestep_video_value,
                timestep_audio_value,
                prompt_timestep_video_value,
                prompt_timestep_audio_value,
            )

        def evaluate_stage1_guided_x0(
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
            sigma_value: torch.Tensor,
            *,
            update_skip_cache: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            (
                latent_input,
                audio_input,
                timestep_video_value,
                timestep_audio_value,
                prompt_timestep_video_value,
                prompt_timestep_audio_value,
            ) = build_stage1_step_inputs(video_latents, audio_latents, sigma_value)

            def build_local_model_kwargs(
                *,
                encoder_hidden_states_value: torch.Tensor,
                audio_encoder_hidden_states_value: torch.Tensor,
                encoder_attention_mask_value: torch.Tensor | None,
                skip_video_self_attn_blocks: tuple[int, ...] | None = None,
                skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
                disable_a2v_cross_attn: bool = False,
                disable_v2a_cross_attn: bool = False,
            ) -> dict[str, object]:
                use_native_ltx23_context_mask = not (
                    ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
                )
                kwargs: dict[str, object] = {
                    "hidden_states": latent_input,
                    "audio_hidden_states": audio_input,
                    "encoder_hidden_states": encoder_hidden_states_value,
                    "audio_encoder_hidden_states": audio_encoder_hidden_states_value,
                    "timestep": timestep_video_value,
                    "audio_timestep": timestep_audio_value,
                    "encoder_attention_mask": (
                        encoder_attention_mask_value
                        if use_native_ltx23_context_mask
                        else None
                    ),
                    "audio_encoder_attention_mask": (
                        encoder_attention_mask_value
                        if use_native_ltx23_context_mask
                        else None
                    ),
                    "num_frames": ctx.latent_num_frames_for_model,
                    "height": ctx.latent_height,
                    "width": ctx.latent_width,
                    "fps": batch.fps,
                    "audio_num_frames": audio_num_frames_latent,
                    "video_coords": video_coords,
                    "audio_coords": audio_coords,
                    "return_latents": False,
                    "return_dict": False,
                    "ltx2_phase": ctx.stage,
                }
                if not ctx.use_ltx23_legacy_one_stage:
                    kwargs.update(
                        {
                            "prompt_timestep": prompt_timestep_video_value,
                            "audio_prompt_timestep": prompt_timestep_audio_value,
                            "video_self_attention_mask": video_self_attention_mask,
                            "audio_self_attention_mask": audio_self_attention_mask,
                            "a2v_cross_attention_mask": a2v_cross_attention_mask,
                            "v2a_cross_attention_mask": v2a_cross_attention_mask,
                            "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                            "legacy_ltx23_one_stage_semantics": False,
                        }
                    )
                if skip_video_self_attn_blocks is not None:
                    kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
                if skip_audio_self_attn_blocks is not None:
                    kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
                if disable_a2v_cross_attn:
                    kwargs["disable_a2v_cross_attn"] = True
                if disable_v2a_cross_attn:
                    kwargs["disable_v2a_cross_attn"] = True
                return kwargs

            def cat_or_none(items: list[torch.Tensor | None]) -> torch.Tensor | None:
                if items[0] is None:
                    return None
                return torch.cat(items, dim=0)

            if ctx.use_ltx23_legacy_one_stage:
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    v_pos, a_v_pos = step.current_model(
                        **build_local_model_kwargs(
                            encoder_hidden_states_value=encoder_hidden_states,
                            audio_encoder_hidden_states_value=audio_encoder_hidden_states,
                            encoder_attention_mask_value=encoder_attention_mask,
                        )
                    )
                    v_neg, a_v_neg = step.current_model(
                        **build_local_model_kwargs(
                            encoder_hidden_states_value=negative_encoder_hidden_states,
                            audio_encoder_hidden_states_value=negative_audio_encoder_hidden_states,
                            encoder_attention_mask_value=negative_encoder_attention_mask,
                        )
                    )

                v_pos = v_pos.float()
                a_v_pos = a_v_pos.float()
                v_neg = v_neg.float()
                a_v_neg = a_v_neg.float()

                v_ptb = None
                a_v_ptb = None
                if need_perturbed:
                    with set_forward_context(
                        current_timestep=step.step_index, attn_metadata=step.attn_metadata
                    ):
                        v_ptb, a_v_ptb = step.current_model(
                            **build_local_model_kwargs(
                                encoder_hidden_states_value=encoder_hidden_states,
                                audio_encoder_hidden_states_value=audio_encoder_hidden_states,
                                encoder_attention_mask_value=encoder_attention_mask,
                                skip_video_self_attn_blocks=tuple(
                                    stage1_guider_params["video_stg_blocks"]
                                ),
                                skip_audio_self_attn_blocks=tuple(
                                    stage1_guider_params["audio_stg_blocks"]
                                ),
                            )
                        )
                    v_ptb = v_ptb.float()
                    a_v_ptb = a_v_ptb.float()

                v_mod = None
                a_v_mod = None
                if need_modality:
                    with set_forward_context(
                        current_timestep=step.step_index, attn_metadata=step.attn_metadata
                    ):
                        v_mod, a_v_mod = step.current_model(
                            **build_local_model_kwargs(
                                encoder_hidden_states_value=encoder_hidden_states,
                                audio_encoder_hidden_states_value=audio_encoder_hidden_states,
                                encoder_attention_mask_value=encoder_attention_mask,
                                disable_a2v_cross_attn=True,
                                disable_v2a_cross_attn=True,
                            )
                        )
                    v_mod = v_mod.float()
                    a_v_mod = a_v_mod.float()
            else:
                use_split_two_stage_ti2v_guider = (
                    is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name)
                    and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
                )
                use_cfg_pair_batch = self._should_use_ltx23_two_stage_cfg_pair_batch(
                    batch=batch,
                    ctx=ctx,
                    server_args=server_args,
                )
                local_batch_size = int(latent_input.shape[0])

                if use_cfg_pair_batch:
                    cfg_pair_batch_size = local_batch_size * 2
                    cfg_pair_perturbation_configs = tuple(
                        (
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": False,
                                "skip_v2a_cross_attn": False,
                            }
                            for _ in range(cfg_pair_batch_size)
                        )
                    )
                    with set_forward_context(
                        current_timestep=step.step_index, attn_metadata=step.attn_metadata
                    ):
                        cfg_pair_video, cfg_pair_audio = step.current_model(
                            hidden_states=self._repeat_batch_dim(
                                latent_input, cfg_pair_batch_size
                            ),
                            audio_hidden_states=self._repeat_batch_dim(
                                audio_input, cfg_pair_batch_size
                            ),
                            encoder_hidden_states=torch.cat(
                                [
                                    encoder_hidden_states,
                                    negative_encoder_hidden_states,
                                ],
                                dim=0,
                            ),
                            audio_encoder_hidden_states=torch.cat(
                                [
                                    audio_encoder_hidden_states,
                                    negative_audio_encoder_hidden_states,
                                ],
                                dim=0,
                            ),
                            timestep=self._repeat_batch_dim(
                                timestep_video_value, cfg_pair_batch_size
                            ),
                            audio_timestep=self._repeat_batch_dim(
                                timestep_audio_value, cfg_pair_batch_size
                            ),
                            prompt_timestep=(
                                None
                                if prompt_timestep_video_value is None
                                else self._repeat_batch_dim(
                                    prompt_timestep_video_value, cfg_pair_batch_size
                                )
                            ),
                            audio_prompt_timestep=(
                                None
                                if prompt_timestep_audio_value is None
                                else self._repeat_batch_dim(
                                    prompt_timestep_audio_value, cfg_pair_batch_size
                                )
                            ),
                            encoder_attention_mask=cat_or_none(
                                [
                                    encoder_attention_mask,
                                    negative_encoder_attention_mask,
                                ]
                            ),
                            audio_encoder_attention_mask=cat_or_none(
                                [
                                    encoder_attention_mask,
                                    negative_encoder_attention_mask,
                                ]
                            ),
                            num_frames=ctx.latent_num_frames_for_model,
                            height=ctx.latent_height,
                            width=ctx.latent_width,
                            fps=batch.fps,
                            audio_num_frames=audio_num_frames_latent,
                            video_coords=(
                                None
                                if video_coords is None
                                else self._repeat_batch_dim(
                                    video_coords, cfg_pair_batch_size
                                )
                            ),
                            audio_coords=(
                                None
                                if audio_coords is None
                                else self._repeat_batch_dim(
                                    audio_coords, cfg_pair_batch_size
                                )
                            ),
                            video_self_attention_mask=(
                                None
                                if video_self_attention_mask is None
                                else self._repeat_batch_dim(
                                    video_self_attention_mask, cfg_pair_batch_size
                                )
                            ),
                            audio_self_attention_mask=(
                                None
                                if audio_self_attention_mask is None
                                else self._repeat_batch_dim(
                                    audio_self_attention_mask, cfg_pair_batch_size
                                )
                            ),
                            a2v_cross_attention_mask=(
                                None
                                if a2v_cross_attention_mask is None
                                else self._repeat_batch_dim(
                                    a2v_cross_attention_mask, cfg_pair_batch_size
                                )
                            ),
                            v2a_cross_attention_mask=(
                                None
                                if v2a_cross_attention_mask is None
                                else self._repeat_batch_dim(
                                    v2a_cross_attention_mask, cfg_pair_batch_size
                                )
                            ),
                            audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                            perturbation_configs=cfg_pair_perturbation_configs,
                            ltx2_phase=ctx.stage,
                            return_latents=False,
                            return_dict=False,
                        )
                    cfg_pair_video = cfg_pair_video.float()
                    cfg_pair_audio = cfg_pair_audio.float()
                    v_pos, v_neg = cfg_pair_video.chunk(2, dim=0)
                    a_v_pos, a_v_neg = cfg_pair_audio.chunk(2, dim=0)

                    v_ptb = None
                    a_v_ptb = None
                    if need_perturbed:
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            v_ptb, a_v_ptb = step.current_model(
                                **build_local_model_kwargs(
                                    encoder_hidden_states_value=encoder_hidden_states,
                                    audio_encoder_hidden_states_value=audio_encoder_hidden_states,
                                    encoder_attention_mask_value=encoder_attention_mask,
                                    skip_video_self_attn_blocks=tuple(
                                        stage1_guider_params["video_stg_blocks"]
                                    ),
                                    skip_audio_self_attn_blocks=tuple(
                                        stage1_guider_params["audio_stg_blocks"]
                                    ),
                                )
                            )
                        v_ptb = v_ptb.float()
                        a_v_ptb = a_v_ptb.float()

                    v_mod = None
                    a_v_mod = None
                    if need_modality:
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            v_mod, a_v_mod = step.current_model(
                                **build_local_model_kwargs(
                                    encoder_hidden_states_value=encoder_hidden_states,
                                    audio_encoder_hidden_states_value=audio_encoder_hidden_states,
                                    encoder_attention_mask_value=encoder_attention_mask,
                                    disable_a2v_cross_attn=True,
                                    disable_v2a_cross_attn=True,
                                )
                            )
                        v_mod = v_mod.float()
                        a_v_mod = a_v_mod.float()
                else:
                    pass_specs: list[
                        tuple[
                            str,
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor | None,
                            dict[str, object],
                        ]
                    ] = [
                        (
                            "cond",
                            encoder_hidden_states,
                            audio_encoder_hidden_states,
                            encoder_attention_mask,
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": False,
                                "skip_v2a_cross_attn": False,
                            },
                        ),
                        (
                            "neg",
                            negative_encoder_hidden_states,
                            negative_audio_encoder_hidden_states,
                            negative_encoder_attention_mask,
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": False,
                                "skip_v2a_cross_attn": False,
                            },
                        ),
                    ]
                    if need_perturbed:
                        pass_specs.append(
                            (
                                "perturbed",
                                encoder_hidden_states,
                                audio_encoder_hidden_states,
                                encoder_attention_mask,
                                {
                                    "skip_video_self_attn_blocks": tuple(
                                        stage1_guider_params["video_stg_blocks"]
                                    ),
                                    "skip_audio_self_attn_blocks": tuple(
                                        stage1_guider_params["audio_stg_blocks"]
                                    ),
                                    "skip_a2v_cross_attn": False,
                                    "skip_v2a_cross_attn": False,
                                },
                            )
                        )
                    if need_modality:
                        pass_specs.append(
                            (
                                "modality",
                                encoder_hidden_states,
                                audio_encoder_hidden_states,
                                encoder_attention_mask,
                                {
                                    "skip_video_self_attn_blocks": (),
                                    "skip_audio_self_attn_blocks": (),
                                    "skip_a2v_cross_attn": True,
                                    "skip_v2a_cross_attn": True,
                                },
                            )
                        )

                    num_passes = len(pass_specs)
                    expanded_batch_size = local_batch_size * num_passes
                    perturbation_configs = tuple(
                        perturbation_config
                        for _, _, _, _, perturbation_config in pass_specs
                        for _ in range(local_batch_size)
                    )
                    batched_hidden_states = self._repeat_batch_dim(
                        latent_input, expanded_batch_size
                    )
                    batched_audio_hidden_states = self._repeat_batch_dim(
                        audio_input, expanded_batch_size
                    )
                    batched_encoder_hidden_states = torch.cat(
                        [item[1] for item in pass_specs], dim=0
                    )
                    batched_audio_encoder_hidden_states = torch.cat(
                        [item[2] for item in pass_specs], dim=0
                    )
                    batched_timestep_video = self._repeat_batch_dim(
                        timestep_video_value, expanded_batch_size
                    )
                    batched_timestep_audio = self._repeat_batch_dim(
                        timestep_audio_value, expanded_batch_size
                    )
                    batched_prompt_timestep_video = (
                        None
                        if prompt_timestep_video_value is None
                        else self._repeat_batch_dim(
                            prompt_timestep_video_value, expanded_batch_size
                        )
                    )
                    batched_prompt_timestep_audio = (
                        None
                        if prompt_timestep_audio_value is None
                        else self._repeat_batch_dim(
                            prompt_timestep_audio_value, expanded_batch_size
                        )
                    )
                    batched_encoder_attention_mask = cat_or_none(
                        [item[3] for item in pass_specs]
                    )
                    batched_audio_encoder_attention_mask = cat_or_none(
                        [item[3] for item in pass_specs]
                    )
                    batched_video_coords = (
                        None
                        if video_coords is None
                        else self._repeat_batch_dim(video_coords, expanded_batch_size)
                    )
                    batched_audio_coords = (
                        None
                        if audio_coords is None
                        else self._repeat_batch_dim(audio_coords, expanded_batch_size)
                    )
                    batched_video_self_attention_mask = (
                        None
                        if video_self_attention_mask is None
                        else self._repeat_batch_dim(
                            video_self_attention_mask, expanded_batch_size
                        )
                    )
                    batched_audio_self_attention_mask = (
                        None
                        if audio_self_attention_mask is None
                        else self._repeat_batch_dim(
                            audio_self_attention_mask, expanded_batch_size
                        )
                    )
                    batched_a2v_cross_attention_mask = (
                        None
                        if a2v_cross_attention_mask is None
                        else self._repeat_batch_dim(
                            a2v_cross_attention_mask, expanded_batch_size
                        )
                    )
                    batched_v2a_cross_attention_mask = (
                        None
                        if v2a_cross_attention_mask is None
                        else self._repeat_batch_dim(
                            v2a_cross_attention_mask, expanded_batch_size
                        )
                    )
                    if use_split_two_stage_ti2v_guider:
                        split_sizes = [1] * expanded_batch_size

                        def split_or_none(
                            tensor: torch.Tensor | None,
                        ) -> list[torch.Tensor | None]:
                            if tensor is None:
                                return [None] * len(split_sizes)
                            return list(tensor.split(split_sizes, dim=0))

                        batched_video_chunks = []
                        batched_audio_chunks = []
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            for (
                                hidden_states_chunk,
                                audio_hidden_states_chunk,
                                encoder_hidden_states_chunk,
                                audio_encoder_hidden_states_chunk,
                                timestep_video_chunk,
                                timestep_audio_chunk,
                                prompt_timestep_video_chunk,
                                prompt_timestep_audio_chunk,
                                encoder_attention_mask_chunk,
                                audio_encoder_attention_mask_chunk,
                                video_coords_chunk,
                                audio_coords_chunk,
                                video_self_attention_mask_chunk,
                                audio_self_attention_mask_chunk,
                                a2v_cross_attention_mask_chunk,
                                v2a_cross_attention_mask_chunk,
                                perturbation_config_chunk,
                            ) in zip(
                                batched_hidden_states.split(split_sizes, dim=0),
                                batched_audio_hidden_states.split(split_sizes, dim=0),
                                batched_encoder_hidden_states.split(split_sizes, dim=0),
                                batched_audio_encoder_hidden_states.split(
                                    split_sizes, dim=0
                                ),
                                batched_timestep_video.split(split_sizes, dim=0),
                                batched_timestep_audio.split(split_sizes, dim=0),
                                split_or_none(batched_prompt_timestep_video),
                                split_or_none(batched_prompt_timestep_audio),
                                split_or_none(batched_encoder_attention_mask),
                                split_or_none(batched_audio_encoder_attention_mask),
                                split_or_none(batched_video_coords),
                                split_or_none(batched_audio_coords),
                                split_or_none(batched_video_self_attention_mask),
                                split_or_none(batched_audio_self_attention_mask),
                                split_or_none(batched_a2v_cross_attention_mask),
                                split_or_none(batched_v2a_cross_attention_mask),
                                ((cfg,) for cfg in perturbation_configs),
                                strict=True,
                            ):
                                video_chunk, audio_chunk = step.current_model(
                                    hidden_states=hidden_states_chunk,
                                    audio_hidden_states=audio_hidden_states_chunk,
                                    encoder_hidden_states=encoder_hidden_states_chunk,
                                    audio_encoder_hidden_states=audio_encoder_hidden_states_chunk,
                                    timestep=timestep_video_chunk,
                                    audio_timestep=timestep_audio_chunk,
                                    prompt_timestep=prompt_timestep_video_chunk,
                                    audio_prompt_timestep=prompt_timestep_audio_chunk,
                                    encoder_attention_mask=encoder_attention_mask_chunk,
                                    audio_encoder_attention_mask=audio_encoder_attention_mask_chunk,
                                    num_frames=ctx.latent_num_frames_for_model,
                                    height=ctx.latent_height,
                                    width=ctx.latent_width,
                                    fps=batch.fps,
                                    audio_num_frames=audio_num_frames_latent,
                                    video_coords=video_coords_chunk,
                                    audio_coords=audio_coords_chunk,
                                    video_self_attention_mask=video_self_attention_mask_chunk,
                                    audio_self_attention_mask=audio_self_attention_mask_chunk,
                                    a2v_cross_attention_mask=a2v_cross_attention_mask_chunk,
                                    v2a_cross_attention_mask=v2a_cross_attention_mask_chunk,
                                    audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                                    perturbation_configs=perturbation_config_chunk,
                                    ltx2_phase=ctx.stage,
                                    return_latents=False,
                                    return_dict=False,
                                )
                                batched_video_chunks.append(video_chunk)
                                batched_audio_chunks.append(audio_chunk)

                        batched_video = torch.cat(batched_video_chunks, dim=0)
                        batched_audio = torch.cat(batched_audio_chunks, dim=0)
                    else:
                        with set_forward_context(
                            current_timestep=step.step_index,
                            attn_metadata=step.attn_metadata,
                        ):
                            batched_video, batched_audio = step.current_model(
                                hidden_states=batched_hidden_states,
                                audio_hidden_states=batched_audio_hidden_states,
                                encoder_hidden_states=batched_encoder_hidden_states,
                                audio_encoder_hidden_states=batched_audio_encoder_hidden_states,
                                timestep=batched_timestep_video,
                                audio_timestep=batched_timestep_audio,
                                prompt_timestep=batched_prompt_timestep_video,
                                audio_prompt_timestep=batched_prompt_timestep_audio,
                                encoder_attention_mask=batched_encoder_attention_mask,
                                audio_encoder_attention_mask=batched_audio_encoder_attention_mask,
                                num_frames=ctx.latent_num_frames_for_model,
                                height=ctx.latent_height,
                                width=ctx.latent_width,
                                fps=batch.fps,
                                audio_num_frames=audio_num_frames_latent,
                                video_coords=batched_video_coords,
                                audio_coords=batched_audio_coords,
                                video_self_attention_mask=batched_video_self_attention_mask,
                                audio_self_attention_mask=batched_audio_self_attention_mask,
                                a2v_cross_attention_mask=batched_a2v_cross_attention_mask,
                                v2a_cross_attention_mask=batched_v2a_cross_attention_mask,
                                audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                                perturbation_configs=perturbation_configs,
                                ltx2_phase=ctx.stage,
                                return_latents=False,
                                return_dict=False,
                            )

                    batched_video = batched_video.float()
                    batched_audio = batched_audio.float()
                    pass_outputs = {
                        pass_name: (
                            video_chunk,
                            audio_chunk,
                        )
                        for (pass_name, _, _, _, _), video_chunk, audio_chunk in zip(
                            pass_specs,
                            batched_video.chunk(num_passes, dim=0),
                            batched_audio.chunk(num_passes, dim=0),
                            strict=True,
                        )
                    }
                    v_pos, a_v_pos = pass_outputs["cond"]
                    v_neg, a_v_neg = pass_outputs["neg"]
                    v_ptb, a_v_ptb = pass_outputs.get("perturbed", (None, None))
                    v_mod, a_v_mod = pass_outputs.get("modality", (None, None))

            sigma_val_local = float(sigma_value.item())
            video_sigma_for_x0_local: float | torch.Tensor = sigma_val_local
            if ctx.denoise_mask is not None:
                video_sigma_for_x0_local = sigma_value.to(
                    device=video_latents.device, dtype=torch.float32
                ) * ctx.denoise_mask.squeeze(-1)

            denoised_video_local = self._ltx2_velocity_to_x0(
                video_latents, v_pos, video_sigma_for_x0_local
            )
            denoised_audio_local = self._ltx2_velocity_to_x0(
                audio_latents, a_v_pos, sigma_val_local
            )
            denoised_video_neg = self._ltx2_velocity_to_x0(
                video_latents, v_neg, video_sigma_for_x0_local
            )
            denoised_audio_neg = self._ltx2_velocity_to_x0(
                audio_latents, a_v_neg, sigma_val_local
            )
            denoised_video_perturbed = (
                None
                if v_ptb is None
                else self._ltx2_velocity_to_x0(
                    video_latents, v_ptb, video_sigma_for_x0_local
                )
            )
            denoised_audio_perturbed = (
                None
                if a_v_ptb is None
                else self._ltx2_velocity_to_x0(audio_latents, a_v_ptb, sigma_val_local)
            )
            denoised_video_modality = (
                None
                if v_mod is None
                else self._ltx2_velocity_to_x0(
                    video_latents, v_mod, video_sigma_for_x0_local
                )
            )
            denoised_audio_modality = (
                None
                if a_v_mod is None
                else self._ltx2_velocity_to_x0(audio_latents, a_v_mod, sigma_val_local)
            )

            if not video_skip:
                denoised_video_local = self._ltx2_calculate_guided_x0(
                    cond=denoised_video_local,
                    uncond_text=denoised_video_neg,
                    uncond_perturbed=(
                        denoised_video_perturbed
                        if denoised_video_perturbed is not None
                        else 0.0
                    ),
                    uncond_modality=(
                        denoised_video_modality
                        if denoised_video_modality is not None
                        else 0.0
                    ),
                    cfg_scale=float(stage1_guider_params["video_cfg_scale"]),
                    stg_scale=float(stage1_guider_params["video_stg_scale"]),
                    rescale_scale=float(stage1_guider_params["video_rescale_scale"]),
                    modality_scale=float(stage1_guider_params["video_modality_scale"]),
                )
                if update_skip_cache:
                    ctx.last_denoised_video = denoised_video_local
            elif ctx.last_denoised_video is not None:
                denoised_video_local = ctx.last_denoised_video

            if not audio_skip:
                denoised_audio_local = self._ltx2_calculate_guided_x0(
                    cond=denoised_audio_local,
                    uncond_text=denoised_audio_neg,
                    uncond_perturbed=(
                        denoised_audio_perturbed
                        if denoised_audio_perturbed is not None
                        else 0.0
                    ),
                    uncond_modality=(
                        denoised_audio_modality
                        if denoised_audio_modality is not None
                        else 0.0
                    ),
                    cfg_scale=float(stage1_guider_params["audio_cfg_scale"]),
                    stg_scale=float(stage1_guider_params["audio_stg_scale"]),
                    rescale_scale=float(stage1_guider_params["audio_rescale_scale"]),
                    modality_scale=float(stage1_guider_params["audio_modality_scale"]),
                )
                if update_skip_cache:
                    ctx.last_denoised_audio = denoised_audio_local
            elif ctx.last_denoised_audio is not None:
                denoised_audio_local = ctx.last_denoised_audio

            return (
                self._ltx2_apply_clean_latent_mask(denoised_video_local, ctx),
                denoised_audio_local,
            )

        denoised_video, denoised_audio = evaluate_stage1_guided_x0(
            ctx.latents,
            ctx.audio_latents,
            sigma,
            update_skip_cache=True,
        )

        if self.sampler_name == "res2s":
            sigma_val = float(sigma.item())
            sigma_next_val = float(sigma_next.item())
            if sigma_val == 0.0 or sigma_next_val == 0.0:
                next_video_latents = denoised_video.to(dtype=ctx.latents.dtype)
                next_audio_latents = denoised_audio.to(dtype=ctx.audio_latents.dtype)
            else:
                h = -torch.log(torch.clamp(sigma_next / sigma, min=1e-12))
                a21, b1, b2 = self._ltx2_get_res2s_coefficients(h)
                anchor_video = ctx.latents.float()
                anchor_audio = ctx.audio_latents.float()
                eps1_video = denoised_video.float() - anchor_video
                eps1_audio = denoised_audio.float() - anchor_audio
                midpoint_video = anchor_video + h * a21 * eps1_video
                midpoint_audio = anchor_audio + h * a21 * eps1_audio
                sub_sigma = torch.sqrt(torch.clamp(sigma * sigma_next, min=0.0))
                midpoint_video = self._ltx2_res2s_sde_step(
                    sample=midpoint_video,
                    denoised_sample=denoised_video.float(),
                    sigma=sigma,
                    sigma_next=sub_sigma,
                    noise=self._randn_like_with_batch_generators(ctx.latents, batch).float(),
                )
                midpoint_audio = self._ltx2_res2s_sde_step(
                    sample=midpoint_audio,
                    denoised_sample=denoised_audio.float(),
                    sigma=sigma,
                    sigma_next=sub_sigma,
                    noise=self._randn_like_with_batch_generators(
                        ctx.audio_latents, batch
                    ).float(),
                )
                midpoint_video = self._ltx2_apply_clean_latent_mask(
                    midpoint_video.to(dtype=ctx.latents.dtype), ctx
                )
                midpoint_audio = midpoint_audio.to(dtype=ctx.audio_latents.dtype)
                midpoint_denoised_video, midpoint_denoised_audio = (
                    evaluate_stage1_guided_x0(
                        midpoint_video,
                        midpoint_audio,
                        sub_sigma,
                        update_skip_cache=False,
                    )
                )
                eps2_video = midpoint_denoised_video.float() - anchor_video
                eps2_audio = midpoint_denoised_audio.float() - anchor_audio
                next_video_latents = anchor_video + h * (b1 * eps1_video + b2 * eps2_video)
                next_audio_latents = anchor_audio + h * (b1 * eps1_audio + b2 * eps2_audio)
                next_video_latents = self._ltx2_res2s_sde_step(
                    sample=next_video_latents,
                    denoised_sample=denoised_video.float(),
                    sigma=sigma,
                    sigma_next=sigma_next,
                    noise=self._randn_like_with_batch_generators(ctx.latents, batch).float(),
                )
                next_audio_latents = self._ltx2_res2s_sde_step(
                    sample=next_audio_latents,
                    denoised_sample=denoised_audio.float(),
                    sigma=sigma,
                    sigma_next=sigma_next,
                    noise=self._randn_like_with_batch_generators(
                        ctx.audio_latents, batch
                    ).float(),
                )
                next_video_latents = self._ltx2_apply_clean_latent_mask(
                    next_video_latents.to(dtype=ctx.latents.dtype), ctx
                )
                next_audio_latents = next_audio_latents.to(dtype=ctx.audio_latents.dtype)
        else:
            sigma_val = float(sigma.item())
            if sigma_val == 0.0:
                v_video = torch.zeros_like(denoised_video)
                v_audio = torch.zeros_like(denoised_audio)
            else:
                v_video = ((ctx.latents.float() - denoised_video.float()) / sigma_val).to(
                    ctx.latents.dtype
                )
                v_audio = (
                    (ctx.audio_latents.float() - denoised_audio.float()) / sigma_val
                ).to(ctx.audio_latents.dtype)

            next_video_latents = (ctx.latents.float() + v_video.float() * dt).to(
                dtype=ctx.latents.dtype
            )
            next_audio_latents = (ctx.audio_latents.float() + v_audio.float() * dt).to(
                dtype=ctx.audio_latents.dtype
            )
            next_video_latents = self._ltx2_apply_clean_latent_mask(
                next_video_latents, ctx
            )

        ctx.latents = next_video_latents
        ctx.audio_latents = next_audio_latents
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )

    def _record_trajectory(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Record audio trajectory alongside the base video trajectory."""
        super()._record_trajectory(ctx, step, batch, server_args)
        if batch.return_trajectory_latents and ctx.audio_latents is not None:
            ctx.trajectory_audio_latents.append(ctx.audio_latents)

    def _finalize_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Expose audio latents before delegating to AV-aware postprocessing."""
        batch.audio_latents = ctx.audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=ctx.latents,
            trajectory_latents=ctx.trajectory_latents,
            trajectory_timesteps=ctx.trajectory_timesteps,
            trajectory_audio_latents=ctx.trajectory_audio_latents,
            server_args=server_args,
            is_warmup=ctx.is_warmup,
        )

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
        trajectory_audio_latents: list | None = None,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Trim SP token padding before delegating to the base finalizer."""
        if trajectory_audio_latents:
            batch.trajectory_audio_latents = torch.stack(
                trajectory_audio_latents, dim=1
            ).cpu()
        latents = self._truncate_sp_padded_token_latents(batch, latents)
        super()._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )

    def _get_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list prompt embeddings for LTX-2 prompts."""
        del batch
        return lambda x: V.is_tensor(x) or V.list_not_empty(x)

    def _get_negative_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list negative prompt embeddings for LTX-2 CFG."""
        return (
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x)
        )
