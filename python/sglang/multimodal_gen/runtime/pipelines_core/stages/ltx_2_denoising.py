import copy
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
    LTX23VideoConditionEncoder,
)
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


@dataclass(slots=True)
class LTX2DenoisingContext(DenoisingContext):
    """Loop-scoped denoising state for joint LTX-2 video and audio generation."""

    audio_latents: torch.Tensor | None = None
    audio_scheduler: object | None = None
    is_ltx23_variant: bool = False
    use_ltx23_native_one_stage_semantics: bool = False
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

    def __init__(self, transformer, scheduler, vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self._condition_image_encoder = None
        self._condition_image_encoder_dir = None

    @staticmethod
    def _ltx_perf_context(stage_name: str, batch: Req):
        enabled = (
            batch.perf_dump_path is not None
            or os.environ.get("SGLANG_LTX_PROFILE_HOTSPOTS", "0") == "1"
        )
        if not enabled:
            return nullcontext()
        return StageProfiler(
            stage_name,
            logger=logger,
            metrics=batch.metrics,
            perf_dump_path_provided=True,
        )

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
        return batch.extra.get("ltx2_stage1_guider_params")

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
    def _repeat_batch_dim_or_none(
        cls, tensor: torch.Tensor | None, target_batch_size: int
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        return cls._repeat_batch_dim(tensor, target_batch_size)

    def _build_ltx2_model_kwargs(
        self,
        *,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        ctx: LTX2DenoisingContext,
        batch: Req,
        audio_num_frames_latent: int,
        video_coords: torch.Tensor | None,
        audio_coords: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        video_self_attention_mask: torch.Tensor | None,
        audio_self_attention_mask: torch.Tensor | None,
        a2v_cross_attention_mask: torch.Tensor | None,
        v2a_cross_attention_mask: torch.Tensor | None,
        skip_video_self_attn_blocks: tuple[int, ...] | None = None,
        skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
        disable_a2v_cross_attn: bool = False,
        disable_v2a_cross_attn: bool = False,
        perturbation_configs: tuple[dict[str, object], ...] | None = None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "hidden_states": latent_model_input,
            "audio_hidden_states": audio_latent_model_input,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep_video,
            "audio_timestep": timestep_audio,
            "encoder_attention_mask": encoder_attention_mask,
            "audio_encoder_attention_mask": encoder_attention_mask,
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
        if not ctx.use_ltx23_native_one_stage_semantics:
            kwargs.update(
                {
                    "prompt_timestep": prompt_timestep_video,
                    "audio_prompt_timestep": prompt_timestep_audio,
                    "video_self_attention_mask": video_self_attention_mask,
                    "audio_self_attention_mask": audio_self_attention_mask,
                    "a2v_cross_attention_mask": a2v_cross_attention_mask,
                    "v2a_cross_attention_mask": v2a_cross_attention_mask,
                    "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                }
            )
            # NOTE: sequential alignment flags (ltx2_align_all_*) were removed here.
            # Probe testing confirmed B=2 batched forward is bit-exact vs sequential,
            # so per-item attention processing is unnecessary overhead for CFG B=2.
        if skip_video_self_attn_blocks is not None:
            kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
        if skip_audio_self_attn_blocks is not None:
            kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
        if disable_a2v_cross_attn:
            kwargs["disable_a2v_cross_attn"] = True
        if disable_v2a_cross_attn:
            kwargs["disable_v2a_cross_attn"] = True
        if perturbation_configs is not None:
            kwargs["perturbation_configs"] = perturbation_configs
        return kwargs

    @staticmethod
    def _ltx2_probe_tensor_report(
        reference: torch.Tensor, candidate: torch.Tensor
    ) -> dict[str, object]:
        reference = reference.detach().float().cpu()
        candidate = candidate.detach().float().cpu()
        diff = candidate - reference
        mse = torch.mean(diff.square()).item()
        cosine = torch.nn.functional.cosine_similarity(
            reference.reshape(1, -1), candidate.reshape(1, -1), dim=1
        ).item()
        return {
            "shape": list(reference.shape),
            "bit_exact": bool(torch.equal(reference, candidate)),
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "rmse": float(mse**0.5),
            "cosine": float(cosine),
        }

    @classmethod
    def _ltx2_probe_cfg_report(
        cls,
        reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, object]:
        return {
            "video_uncond": cls._ltx2_probe_tensor_report(
                reference[0], candidate[0]
            ),
            "video_cond": cls._ltx2_probe_tensor_report(reference[1], candidate[1]),
            "audio_uncond": cls._ltx2_probe_tensor_report(
                reference[2], candidate[2]
            ),
            "audio_cond": cls._ltx2_probe_tensor_report(reference[3], candidate[3]),
        }

    @staticmethod
    def _ltx2_probe_split_candidate_cond_pair(
        reference_uncond: torch.Tensor,
        reference_cond: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate.shape[0] == 2:
            return candidate[0:1], candidate[1:2]
        if (
            candidate.shape[0] == 1
            and reference_uncond.shape[0] == 1
            and reference_cond.shape[0] == 1
            and candidate.ndim >= 2
            and candidate.shape[2:] == reference_uncond.shape[2:]
            and candidate.shape[2:] == reference_cond.shape[2:]
            and candidate.shape[1]
            == reference_uncond.shape[1] + reference_cond.shape[1]
        ):
            split = reference_uncond.shape[1]
            return candidate[:, :split], candidate[:, split:]
        raise ValueError(
            "Unable to split LTX2 probe candidate into uncond/cond pair: "
            f"reference_uncond_shape={tuple(reference_uncond.shape)}, "
            f"reference_cond_shape={tuple(reference_cond.shape)}, "
            f"candidate_shape={tuple(candidate.shape)}"
        )

    @classmethod
    def _ltx2_probe_cfg_report_from_batched_tensors(
        cls,
        *,
        reference_video_uncond: torch.Tensor,
        reference_video_cond: torch.Tensor,
        reference_audio_uncond: torch.Tensor,
        reference_audio_cond: torch.Tensor,
        candidate_video: torch.Tensor,
        candidate_audio: torch.Tensor,
    ) -> dict[str, object]:
        candidate_video_uncond, candidate_video_cond = (
            cls._ltx2_probe_split_candidate_cond_pair(
                reference_video_uncond, reference_video_cond, candidate_video
            )
        )
        candidate_audio_uncond, candidate_audio_cond = (
            cls._ltx2_probe_split_candidate_cond_pair(
                reference_audio_uncond, reference_audio_cond, candidate_audio
            )
        )
        return cls._ltx2_probe_cfg_report(
            (
                reference_video_uncond,
                reference_video_cond,
                reference_audio_uncond,
                reference_audio_cond,
            ),
            (
                candidate_video_uncond.float(),
                candidate_video_cond.float(),
                candidate_audio_uncond.float(),
                candidate_audio_cond.float(),
            ),
        )

    @classmethod
    def _ltx2_probe_cond_pair_report(
        cls,
        reference: tuple[torch.Tensor, torch.Tensor],
        candidate: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, object]:
        return {
            "uncond": cls._ltx2_probe_tensor_report(reference[0], candidate[0]),
            "cond": cls._ltx2_probe_tensor_report(reference[1], candidate[1]),
        }

    @classmethod
    def _ltx2_probe_cond_pair_report_from_batched_tensor(
        cls,
        *,
        reference_uncond: torch.Tensor,
        reference_cond: torch.Tensor,
        candidate: torch.Tensor,
    ) -> dict[str, object]:
        candidate_uncond, candidate_cond = cls._ltx2_probe_split_candidate_cond_pair(
            reference_uncond, reference_cond, candidate
        )
        return cls._ltx2_probe_cond_pair_report(
            (reference_uncond, reference_cond),
            (candidate_uncond.float(), candidate_cond.float()),
        )

    @staticmethod
    def _ltx2_probe_slice_model_kwargs(
        model_kwargs: dict[str, object], start: int, end: int
    ) -> dict[str, object]:
        sliced: dict[str, object] = {}
        for key, value in model_kwargs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] >= end
            ):
                sliced[key] = value[start:end]
            elif isinstance(value, tuple) and value and len(value) >= end:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    @classmethod
    def _ltx2_probe_clone_value(cls, value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, tuple):
            return tuple(cls._ltx2_probe_clone_value(item) for item in value)
        if isinstance(value, list):
            return [cls._ltx2_probe_clone_value(item) for item in value]
        return copy.deepcopy(value)

    @classmethod
    def _ltx2_probe_clone_model_kwargs(
        cls, model_kwargs: dict[str, object]
    ) -> dict[str, object]:
        return {
            key: cls._ltx2_probe_clone_value(value)
            for key, value in model_kwargs.items()
        }

    @staticmethod
    def _ltx2_probe_merge_model_kwargs(
        *items: dict[str, object],
    ) -> dict[str, object]:
        if len(items) < 2:
            raise ValueError("Need at least 2 kwargs dicts to merge")
        merged: dict[str, object] = {}
        all_keys: set[str] = set()
        for d in items:
            all_keys |= d.keys()
        for key in all_keys:
            values = [d.get(key) for d in items]
            if all(
                isinstance(v, torch.Tensor) and v.ndim > 0 for v in values
            ) and len({v.shape[1:] for v in values}) == 1:
                merged[key] = torch.cat(values, dim=0)
            elif all(isinstance(v, tuple) for v in values) and len({len(v) for v in values}) == 1:
                merged[key] = tuple(
                    sum(sub_items[1:], sub_items[0])
                    if all(isinstance(si, tuple) for si in sub_items)
                    else sub_items[0]
                    for sub_items in zip(*values)
                )
            else:
                merged[key] = values[0]
        return merged

    def _ltx2_probe_output_path(self) -> str | None:
        return os.getenv("SGLANG_LTX2_PROBE_OUTPUT")

    @staticmethod
    def _ltx2_probe_block_index() -> int:
        value = os.getenv("SGLANG_LTX2_PROBE_BLOCK_INDEX")
        if value is None:
            return 0
        return int(value)

    def _record_ltx2_probe_model_call(
        self, *, step: DenoisingStepState, model_kwargs: dict[str, object]
    ) -> None:
        if self._ltx2_probe_output_path() is None:
            return
        calls = getattr(self, "_ltx2_probe_model_forward_calls", [])
        if len(calls) >= 16:
            return
        hidden_states = model_kwargs.get("hidden_states")
        encoder_hidden_states = model_kwargs.get("encoder_hidden_states")
        calls.append(
            {
                "stage": getattr(self, "_ltx2_probe_current_stage", None),
                "step_index": int(step.step_index),
                "hidden_states_shape": (
                    list(hidden_states.shape)
                    if isinstance(hidden_states, torch.Tensor)
                    else None
                ),
                "encoder_hidden_states_shape": (
                    list(encoder_hidden_states.shape)
                    if isinstance(encoder_hidden_states, torch.Tensor)
                    else None
                ),
                "has_perturbation_configs": model_kwargs.get("perturbation_configs")
                is not None,
            }
        )
        self._ltx2_probe_model_forward_calls = calls

    def _write_ltx2_probe_state(self, payload: dict[str, object]) -> None:
        output_path = self._ltx2_probe_output_path()
        if output_path is None or getattr(self, "_ltx2_probe_written", False):
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": None,
            "aux": None,
            "official_cfg": None,
            "probe_block_index": self._ltx2_probe_block_index(),
            "model_forward_calls": getattr(
                self, "_ltx2_probe_model_forward_calls", []
            ),
        }
        state.update(payload)
        Path(output_path).write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._ltx2_probe_written = True

    def _maybe_probe_ltx2_official_cfg_forward(
        self,
        *,
        step: DenoisingStepState,
        model_kwargs: dict[str, object],
        out_video: torch.Tensor,
        out_audio: torch.Tensor,
    ) -> None:
        if (
            self._ltx2_probe_output_path() is None
            or getattr(self, "_ltx2_probe_written", False)
            or getattr(self, "_ltx2_probe_current_stage", None) != "stage1"
        ):
            return
        hidden_states = model_kwargs.get("hidden_states")
        encoder_hidden_states = model_kwargs.get("encoder_hidden_states")
        if not (
            isinstance(hidden_states, torch.Tensor)
            and isinstance(encoder_hidden_states, torch.Tensor)
            and model_kwargs.get("perturbation_configs") is None
        ):
            return
        if hidden_states.shape[0] == 2 and encoder_hidden_states.shape[0] == 2:
            is_sp_pair_probe = False
            batched_model_kwargs = model_kwargs
            batched_out_video = out_video
            batched_out_audio = out_audio
            uncond_kwargs = self._ltx2_probe_slice_model_kwargs(model_kwargs, 0, 1)
            cond_kwargs = self._ltx2_probe_slice_model_kwargs(model_kwargs, 1, 2)
            ref_video_uncond = None
            ref_audio_uncond = None
            ref_video_cond = None
            ref_audio_cond = None
        elif hidden_states.shape[0] == 1 and encoder_hidden_states.shape[0] == 1:
            is_sp_pair_probe = True
            sequential_pair = getattr(self, "_ltx2_probe_sp_official_cfg_pair", [])
            sequential_pair.append(
                (
                    self._ltx2_probe_clone_model_kwargs(model_kwargs),
                    out_video.detach().clone(),
                    out_audio.detach().clone(),
                )
            )
            self._ltx2_probe_sp_official_cfg_pair = sequential_pair
            if len(sequential_pair) < 2:
                return
            uncond_kwargs, ref_video_uncond, ref_audio_uncond = sequential_pair[0]
            cond_kwargs, ref_video_cond, ref_audio_cond = sequential_pair[1]
            batched_model_kwargs = self._ltx2_probe_merge_model_kwargs(
                uncond_kwargs, cond_kwargs
            )
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                batched_out_video, batched_out_audio = step.current_model(
                    **batched_model_kwargs
                )
            batched_out_video = batched_out_video.float()
            batched_out_audio = batched_out_audio.float()
            self._ltx2_probe_sp_official_cfg_pair = []
        else:
            return

        probe_block_index = self._ltx2_probe_block_index()
        if not (0 <= probe_block_index < len(step.current_model.transformer_blocks)):
            raise ValueError(
                f"Invalid SGLANG_LTX2_PROBE_BLOCK_INDEX={probe_block_index}, "
                f"num_blocks={len(step.current_model.transformer_blocks)}."
            )
        probe_block = step.current_model.transformer_blocks[probe_block_index]
        batched_actual_block_trace = getattr(probe_block, "_ltx2_probe_stage_trace", None)
        batched_actual_audio_ff_trace = getattr(
            probe_block.audio_ff, "_ltx2_probe_ff_trace", None
        )
        if ref_video_uncond is None or ref_audio_uncond is None:
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                ref_video_uncond, ref_audio_uncond = step.current_model(**uncond_kwargs)
        ref_actual_block_trace_uncond = getattr(probe_block, "_ltx2_probe_stage_trace", None)
        ref_audio_ff_trace_uncond = getattr(
            probe_block.audio_ff, "_ltx2_probe_ff_trace", None
        )
        if ref_video_cond is None or ref_audio_cond is None:
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                ref_video_cond, ref_audio_cond = step.current_model(**cond_kwargs)
        ref_actual_block_trace_cond = getattr(probe_block, "_ltx2_probe_stage_trace", None)
        ref_audio_ff_trace_cond = getattr(
            probe_block.audio_ff, "_ltx2_probe_ff_trace", None
        )
        with set_forward_context(
            current_timestep=step.step_index, attn_metadata=step.attn_metadata
        ):
            ref2_video_uncond, ref2_audio_uncond = step.current_model(**uncond_kwargs)
        with set_forward_context(
            current_timestep=step.step_index, attn_metadata=step.attn_metadata
        ):
            ref2_video_cond, ref2_audio_cond = step.current_model(**cond_kwargs)

        model = step.current_model
        pretrace: dict[str, object] = {}
        batched_patchify_video, _ = model.patchify_proj(
            batched_model_kwargs["hidden_states"]
        )
        batched_patchify_audio, _ = model.audio_patchify_proj(
            batched_model_kwargs["audio_hidden_states"]
        )
        ref_patchify_video_uncond, _ = model.patchify_proj(uncond_kwargs["hidden_states"])
        ref_patchify_video_cond, _ = model.patchify_proj(cond_kwargs["hidden_states"])
        ref_patchify_audio_uncond, _ = model.audio_patchify_proj(
            uncond_kwargs["audio_hidden_states"]
        )
        ref_patchify_audio_cond, _ = model.audio_patchify_proj(
            cond_kwargs["audio_hidden_states"]
        )
        pretrace["patchify"] = self._ltx2_probe_cfg_report_from_batched_tensors(
            reference_video_uncond=ref_patchify_video_uncond.float(),
            reference_video_cond=ref_patchify_video_cond.float(),
            reference_audio_uncond=ref_patchify_audio_uncond.float(),
            reference_audio_cond=ref_patchify_audio_cond.float(),
            candidate_video=batched_patchify_video.float(),
            candidate_audio=batched_patchify_audio.float(),
        )

        if model.caption_projection is not None:
            batched_caption_video = model.caption_projection(
                batched_model_kwargs["encoder_hidden_states"]
            )
            batched_caption_audio = model.audio_caption_projection(
                batched_model_kwargs["audio_encoder_hidden_states"]
            )
            ref_caption_video_uncond = model.caption_projection(
                uncond_kwargs["encoder_hidden_states"]
            )
            ref_caption_video_cond = model.caption_projection(
                cond_kwargs["encoder_hidden_states"]
            )
            ref_caption_audio_uncond = model.audio_caption_projection(
                uncond_kwargs["audio_encoder_hidden_states"]
            )
            ref_caption_audio_cond = model.audio_caption_projection(
                cond_kwargs["audio_encoder_hidden_states"]
            )
            pretrace["caption_projection"] = (
                self._ltx2_probe_cfg_report_from_batched_tensors(
                    reference_video_uncond=ref_caption_video_uncond.float(),
                    reference_video_cond=ref_caption_video_cond.float(),
                    reference_audio_uncond=ref_caption_audio_uncond.float(),
                    reference_audio_cond=ref_caption_audio_cond.float(),
                    candidate_video=batched_caption_video.float(),
                    candidate_audio=batched_caption_audio.float(),
                )
            )

        batched_temb, _ = model.adaln_single(
            batched_model_kwargs["timestep"].flatten(),
            hidden_dtype=batched_patchify_video.dtype,
        )
        batched_temb = batched_temb.view(2, -1, batched_temb.size(-1))
        batched_temb_audio, _ = model.audio_adaln_single(
            batched_model_kwargs["audio_timestep"].flatten(),
            hidden_dtype=batched_patchify_audio.dtype,
        )
        batched_temb_audio = batched_temb_audio.view(
            2, -1, batched_temb_audio.size(-1)
        )
        ref_temb_uncond, _ = model.adaln_single(
            uncond_kwargs["timestep"].flatten(), hidden_dtype=batched_patchify_video.dtype
        )
        ref_temb_uncond = ref_temb_uncond.view(
            1, -1, ref_temb_uncond.size(-1)
        )
        ref_temb_cond, _ = model.adaln_single(
            cond_kwargs["timestep"].flatten(), hidden_dtype=batched_patchify_video.dtype
        )
        ref_temb_cond = ref_temb_cond.view(1, -1, ref_temb_cond.size(-1))
        ref_temb_audio_uncond, _ = model.audio_adaln_single(
            uncond_kwargs["audio_timestep"].flatten(),
            hidden_dtype=batched_patchify_audio.dtype,
        )
        ref_temb_audio_uncond = ref_temb_audio_uncond.view(
            1, -1, ref_temb_audio_uncond.size(-1)
        )
        ref_temb_audio_cond, _ = model.audio_adaln_single(
            cond_kwargs["audio_timestep"].flatten(),
            hidden_dtype=batched_patchify_audio.dtype,
        )
        ref_temb_audio_cond = ref_temb_audio_cond.view(
            1, -1, ref_temb_audio_cond.size(-1)
        )
        pretrace["adaln_single"] = self._ltx2_probe_cfg_report_from_batched_tensors(
            reference_video_uncond=ref_temb_uncond.float(),
            reference_video_cond=ref_temb_cond.float(),
            reference_audio_uncond=ref_temb_audio_uncond.float(),
            reference_audio_cond=ref_temb_audio_cond.float(),
            candidate_video=batched_temb.float(),
            candidate_audio=batched_temb_audio.float(),
        )

        batched_caption_video = batched_model_kwargs["encoder_hidden_states"]
        batched_caption_audio = batched_model_kwargs["audio_encoder_hidden_states"]
        ref_caption_video_uncond = uncond_kwargs["encoder_hidden_states"]
        ref_caption_video_cond = cond_kwargs["encoder_hidden_states"]
        ref_caption_audio_uncond = uncond_kwargs["audio_encoder_hidden_states"]
        ref_caption_audio_cond = cond_kwargs["audio_encoder_hidden_states"]

        batched_temb_prompt = None
        batched_temb_audio_prompt = None
        ref_temb_prompt_uncond = None
        ref_temb_prompt_cond = None
        ref_temb_audio_prompt_uncond = None
        ref_temb_audio_prompt_cond = None
        if model.prompt_adaln_single is not None:
            prompt_timestep = batched_model_kwargs.get("prompt_timestep")
            if prompt_timestep is None:
                prompt_timestep = model._collapse_prompt_timestep(
                    batched_model_kwargs["timestep"]
                )
            batched_temb_prompt, _ = model.prompt_adaln_single(
                prompt_timestep.flatten(), hidden_dtype=batched_patchify_video.dtype
            )
            batched_temb_prompt = batched_temb_prompt.view(
                2, -1, batched_temb_prompt.size(-1)
            )

            prompt_timestep_uncond = uncond_kwargs.get("prompt_timestep")
            if prompt_timestep_uncond is None:
                prompt_timestep_uncond = model._collapse_prompt_timestep(
                    uncond_kwargs["timestep"]
                )
            ref_temb_prompt_uncond, _ = model.prompt_adaln_single(
                prompt_timestep_uncond.flatten(),
                hidden_dtype=batched_patchify_video.dtype,
            )
            ref_temb_prompt_uncond = ref_temb_prompt_uncond.view(
                1, -1, ref_temb_prompt_uncond.size(-1)
            )

            prompt_timestep_cond = cond_kwargs.get("prompt_timestep")
            if prompt_timestep_cond is None:
                prompt_timestep_cond = model._collapse_prompt_timestep(
                    cond_kwargs["timestep"]
                )
            ref_temb_prompt_cond, _ = model.prompt_adaln_single(
                prompt_timestep_cond.flatten(), hidden_dtype=batched_patchify_video.dtype
            )
            ref_temb_prompt_cond = ref_temb_prompt_cond.view(
                1, -1, ref_temb_prompt_cond.size(-1)
            )

            audio_prompt_timestep = batched_model_kwargs.get("audio_prompt_timestep")
            if audio_prompt_timestep is None:
                audio_prompt_timestep = model._collapse_prompt_timestep(
                    batched_model_kwargs["audio_timestep"]
                )
            batched_temb_audio_prompt, _ = model.audio_prompt_adaln_single(
                audio_prompt_timestep.flatten(),
                hidden_dtype=batched_patchify_audio.dtype,
            )
            batched_temb_audio_prompt = batched_temb_audio_prompt.view(
                2, -1, batched_temb_audio_prompt.size(-1)
            )

            audio_prompt_timestep_uncond = uncond_kwargs.get("audio_prompt_timestep")
            if audio_prompt_timestep_uncond is None:
                audio_prompt_timestep_uncond = model._collapse_prompt_timestep(
                    uncond_kwargs["audio_timestep"]
                )
            ref_temb_audio_prompt_uncond, _ = model.audio_prompt_adaln_single(
                audio_prompt_timestep_uncond.flatten(),
                hidden_dtype=batched_patchify_audio.dtype,
            )
            ref_temb_audio_prompt_uncond = ref_temb_audio_prompt_uncond.view(
                1, -1, ref_temb_audio_prompt_uncond.size(-1)
            )

            audio_prompt_timestep_cond = cond_kwargs.get("audio_prompt_timestep")
            if audio_prompt_timestep_cond is None:
                audio_prompt_timestep_cond = model._collapse_prompt_timestep(
                    cond_kwargs["audio_timestep"]
                )
            ref_temb_audio_prompt_cond, _ = model.audio_prompt_adaln_single(
                audio_prompt_timestep_cond.flatten(),
                hidden_dtype=batched_patchify_audio.dtype,
            )
            ref_temb_audio_prompt_cond = ref_temb_audio_prompt_cond.view(
                1, -1, ref_temb_audio_prompt_cond.size(-1)
            )

            pretrace["prompt_adaln"] = self._ltx2_probe_cfg_report_from_batched_tensors(
                reference_video_uncond=ref_temb_prompt_uncond.float(),
                reference_video_cond=ref_temb_prompt_cond.float(),
                reference_audio_uncond=ref_temb_audio_prompt_uncond.float(),
                reference_audio_cond=ref_temb_audio_prompt_cond.float(),
                candidate_video=batched_temb_prompt.float(),
                candidate_audio=batched_temb_audio_prompt.float(),
            )

        av_ca_video_timestep, av_ca_audio_timestep = model._get_av_ca_timesteps(
            batched_model_kwargs["timestep"],
            batched_model_kwargs["audio_timestep"],
            batched_model_kwargs.get("prompt_timestep"),
            batched_model_kwargs.get("audio_prompt_timestep"),
        )
        batched_temb_ca_scale_shift, _ = model.av_ca_video_scale_shift_adaln_single(
            av_ca_video_timestep.flatten(), hidden_dtype=batched_patchify_video.dtype
        )
        batched_temb_ca_scale_shift = batched_temb_ca_scale_shift.view(
            2, -1, batched_temb_ca_scale_shift.size(-1)
        )
        av_ca_gate_factor = model._get_av_ca_gate_timestep_factor()
        batched_temb_ca_gate, _ = model.av_ca_a2v_gate_adaln_single(
            av_ca_video_timestep.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_video.dtype,
        )
        batched_temb_ca_gate = batched_temb_ca_gate.view(
            2, -1, batched_temb_ca_gate.size(-1)
        )

        batched_temb_ca_audio_scale_shift, _ = (
            model.av_ca_audio_scale_shift_adaln_single(
                av_ca_audio_timestep.flatten(), hidden_dtype=batched_patchify_audio.dtype
            )
        )
        batched_temb_ca_audio_scale_shift = batched_temb_ca_audio_scale_shift.view(
            2, -1, batched_temb_ca_audio_scale_shift.size(-1)
        )
        batched_temb_ca_audio_gate, _ = model.av_ca_v2a_gate_adaln_single(
            av_ca_audio_timestep.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_audio.dtype,
        )
        batched_temb_ca_audio_gate = batched_temb_ca_audio_gate.view(
            2, -1, batched_temb_ca_audio_gate.size(-1)
        )

        ref_av_ca_video_timestep_uncond, ref_av_ca_audio_timestep_uncond = (
            model._get_av_ca_timesteps(
                uncond_kwargs["timestep"],
                uncond_kwargs["audio_timestep"],
                uncond_kwargs.get("prompt_timestep"),
                uncond_kwargs.get("audio_prompt_timestep"),
            )
        )
        ref_temb_ca_scale_shift_uncond, _ = model.av_ca_video_scale_shift_adaln_single(
            ref_av_ca_video_timestep_uncond.flatten(),
            hidden_dtype=batched_patchify_video.dtype,
        )
        ref_temb_ca_scale_shift_uncond = ref_temb_ca_scale_shift_uncond.view(
            1, -1, ref_temb_ca_scale_shift_uncond.size(-1)
        )
        ref_temb_ca_gate_uncond, _ = model.av_ca_a2v_gate_adaln_single(
            ref_av_ca_video_timestep_uncond.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_video.dtype,
        )
        ref_temb_ca_gate_uncond = ref_temb_ca_gate_uncond.view(
            1, -1, ref_temb_ca_gate_uncond.size(-1)
        )
        ref_temb_ca_audio_scale_shift_uncond, _ = (
            model.av_ca_audio_scale_shift_adaln_single(
                ref_av_ca_audio_timestep_uncond.flatten(),
                hidden_dtype=batched_patchify_audio.dtype,
            )
        )
        ref_temb_ca_audio_scale_shift_uncond = (
            ref_temb_ca_audio_scale_shift_uncond.view(
                1, -1, ref_temb_ca_audio_scale_shift_uncond.size(-1)
            )
        )
        ref_temb_ca_audio_gate_uncond, _ = model.av_ca_v2a_gate_adaln_single(
            ref_av_ca_audio_timestep_uncond.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_audio.dtype,
        )
        ref_temb_ca_audio_gate_uncond = ref_temb_ca_audio_gate_uncond.view(
            1, -1, ref_temb_ca_audio_gate_uncond.size(-1)
        )

        ref_av_ca_video_timestep_cond, ref_av_ca_audio_timestep_cond = (
            model._get_av_ca_timesteps(
                cond_kwargs["timestep"],
                cond_kwargs["audio_timestep"],
                cond_kwargs.get("prompt_timestep"),
                cond_kwargs.get("audio_prompt_timestep"),
            )
        )
        ref_temb_ca_scale_shift_cond, _ = model.av_ca_video_scale_shift_adaln_single(
            ref_av_ca_video_timestep_cond.flatten(),
            hidden_dtype=batched_patchify_video.dtype,
        )
        ref_temb_ca_scale_shift_cond = ref_temb_ca_scale_shift_cond.view(
            1, -1, ref_temb_ca_scale_shift_cond.size(-1)
        )
        ref_temb_ca_gate_cond, _ = model.av_ca_a2v_gate_adaln_single(
            ref_av_ca_video_timestep_cond.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_video.dtype,
        )
        ref_temb_ca_gate_cond = ref_temb_ca_gate_cond.view(
            1, -1, ref_temb_ca_gate_cond.size(-1)
        )
        ref_temb_ca_audio_scale_shift_cond, _ = (
            model.av_ca_audio_scale_shift_adaln_single(
                ref_av_ca_audio_timestep_cond.flatten(),
                hidden_dtype=batched_patchify_audio.dtype,
            )
        )
        ref_temb_ca_audio_scale_shift_cond = ref_temb_ca_audio_scale_shift_cond.view(
            1, -1, ref_temb_ca_audio_scale_shift_cond.size(-1)
        )
        ref_temb_ca_audio_gate_cond, _ = model.av_ca_v2a_gate_adaln_single(
            ref_av_ca_audio_timestep_cond.flatten() * av_ca_gate_factor,
            hidden_dtype=batched_patchify_audio.dtype,
        )
        ref_temb_ca_audio_gate_cond = ref_temb_ca_audio_gate_cond.view(
            1, -1, ref_temb_ca_audio_gate_cond.size(-1)
        )

        pretrace["av_ca_scale_shift"] = self._ltx2_probe_cfg_report_from_batched_tensors(
            reference_video_uncond=ref_temb_ca_scale_shift_uncond.float(),
            reference_video_cond=ref_temb_ca_scale_shift_cond.float(),
            reference_audio_uncond=ref_temb_ca_audio_scale_shift_uncond.float(),
            reference_audio_cond=ref_temb_ca_audio_scale_shift_cond.float(),
            candidate_video=batched_temb_ca_scale_shift.float(),
            candidate_audio=batched_temb_ca_audio_scale_shift.float(),
        )
        pretrace["av_ca_gate"] = self._ltx2_probe_cfg_report_from_batched_tensors(
            reference_video_uncond=ref_temb_ca_gate_uncond.float(),
            reference_video_cond=ref_temb_ca_gate_cond.float(),
            reference_audio_uncond=ref_temb_ca_audio_gate_uncond.float(),
            reference_audio_cond=ref_temb_ca_audio_gate_cond.float(),
            candidate_video=batched_temb_ca_gate.float(),
            candidate_audio=batched_temb_ca_audio_gate.float(),
        )
        if (
            not is_sp_pair_probe
            and
            batched_actual_block_trace is not None
            and ref_actual_block_trace_uncond is not None
            and ref_actual_block_trace_cond is not None
        ):
            actual_block_key = f"actual_block{probe_block_index}"
            for trace_key, trace_name in (
                ("before_block", f"{actual_block_key}_before_block"),
                ("after_self_attn", f"{actual_block_key}_after_self_attn"),
                (
                    "after_prompt_cross_attn",
                    f"{actual_block_key}_after_prompt_cross_attn",
                ),
                ("after_av_cross_attn", f"{actual_block_key}_after_av_cross_attn"),
                ("after_ff", f"{actual_block_key}_after_ff"),
            ):
                batched_video_stage, batched_audio_stage = batched_actual_block_trace[
                    trace_key
                ]
                ref_video_stage_uncond, ref_audio_stage_uncond = (
                    ref_actual_block_trace_uncond[trace_key]
                )
                ref_video_stage_cond, ref_audio_stage_cond = (
                    ref_actual_block_trace_cond[trace_key]
                )
                pretrace[trace_name] = self._ltx2_probe_cfg_report_from_batched_tensors(
                    reference_video_uncond=ref_video_stage_uncond.float(),
                    reference_video_cond=ref_video_stage_cond.float(),
                    reference_audio_uncond=ref_audio_stage_uncond.float(),
                    reference_audio_cond=ref_audio_stage_cond.float(),
                    candidate_video=batched_video_stage.float(),
                    candidate_audio=batched_audio_stage.float(),
                )
            if (
                batched_actual_audio_ff_trace is not None
                and ref_audio_ff_trace_uncond is not None
                and ref_audio_ff_trace_cond is not None
            ):
                for ff_key in ("proj_in", "act", "proj_out"):
                    pretrace[f"{actual_block_key}_audio_ff_{ff_key}"] = (
                        self._ltx2_probe_cond_pair_report_from_batched_tensor(
                            reference_uncond=ref_audio_ff_trace_uncond[ff_key].float(),
                            reference_cond=ref_audio_ff_trace_cond[ff_key].float(),
                            candidate=batched_actual_audio_ff_trace[ff_key].float(),
                        )
                    )

        if (
            not is_sp_pair_probe
            and
            "num_frames" in batched_model_kwargs
            and "height" in batched_model_kwargs
            and "width" in batched_model_kwargs
            and "audio_num_frames" in batched_model_kwargs
            and len(model.transformer_blocks) > 0
        ):
            probe_block_index = self._ltx2_probe_block_index()
            if not (0 <= probe_block_index < len(model.transformer_blocks)):
                raise ValueError(
                    f"Invalid SGLANG_LTX2_PROBE_BLOCK_INDEX={probe_block_index}, "
                    f"num_blocks={len(model.transformer_blocks)}."
                )
            fps = float(batched_model_kwargs.get("fps", 24.0))
            batched_video_coords = batched_model_kwargs.get("video_coords")
            if batched_video_coords is None:
                batched_video_coords = model.rope.prepare_video_coords(
                    batch_size=2,
                    num_frames=int(batched_model_kwargs["num_frames"]),
                    height=int(batched_model_kwargs["height"]),
                    width=int(batched_model_kwargs["width"]),
                    device=batched_model_kwargs["hidden_states"].device,
                    fps=fps,
                    start_frame=0,
                )
            batched_audio_coords = batched_model_kwargs.get("audio_coords")
            if batched_audio_coords is None:
                batched_audio_coords = model.audio_rope.prepare_audio_coords(
                    batch_size=2,
                    num_frames=int(batched_model_kwargs["audio_num_frames"]),
                    device=batched_model_kwargs["audio_hidden_states"].device,
                )

            batched_video_coords = model._maybe_quantize_video_rope_coords(
                batched_video_coords,
                batched_model_kwargs["hidden_states"].device,
                batched_model_kwargs["hidden_states"].dtype,
            )
            batched_audio_coords = batched_audio_coords.to(
                device=batched_model_kwargs["audio_hidden_states"].device
            )

            batched_video_rotary_emb = model.rope(
                batched_video_coords,
                device=batched_model_kwargs["hidden_states"].device,
                out_dtype=batched_model_kwargs["hidden_states"].dtype,
            )
            batched_audio_rotary_emb = model.audio_rope(
                batched_audio_coords,
                device=batched_model_kwargs["audio_hidden_states"].device,
                out_dtype=batched_model_kwargs["audio_hidden_states"].dtype,
            )
            batched_ca_video_rotary_emb = model.cross_attn_rope(
                batched_video_coords[:, 0:1, :],
                device=batched_model_kwargs["hidden_states"].device,
                out_dtype=batched_model_kwargs["hidden_states"].dtype,
            )
            batched_ca_audio_rotary_emb = model.cross_attn_audio_rope(
                batched_audio_coords[:, 0:1, :],
                device=batched_model_kwargs["audio_hidden_states"].device,
                out_dtype=batched_model_kwargs["audio_hidden_states"].dtype,
            )

            ref_video_coords_uncond = uncond_kwargs.get("video_coords")
            if ref_video_coords_uncond is None:
                ref_video_coords_uncond = model.rope.prepare_video_coords(
                    batch_size=1,
                    num_frames=int(uncond_kwargs["num_frames"]),
                    height=int(uncond_kwargs["height"]),
                    width=int(uncond_kwargs["width"]),
                    device=uncond_kwargs["hidden_states"].device,
                    fps=float(uncond_kwargs.get("fps", 24.0)),
                    start_frame=0,
                )
            ref_audio_coords_uncond = uncond_kwargs.get("audio_coords")
            if ref_audio_coords_uncond is None:
                ref_audio_coords_uncond = model.audio_rope.prepare_audio_coords(
                    batch_size=1,
                    num_frames=int(uncond_kwargs["audio_num_frames"]),
                    device=uncond_kwargs["audio_hidden_states"].device,
                )
            ref_video_coords_uncond = model._maybe_quantize_video_rope_coords(
                ref_video_coords_uncond,
                uncond_kwargs["hidden_states"].device,
                uncond_kwargs["hidden_states"].dtype,
            )
            ref_audio_coords_uncond = ref_audio_coords_uncond.to(
                device=uncond_kwargs["audio_hidden_states"].device
            )
            ref_video_rotary_emb_uncond = model.rope(
                ref_video_coords_uncond,
                device=uncond_kwargs["hidden_states"].device,
                out_dtype=uncond_kwargs["hidden_states"].dtype,
            )
            ref_audio_rotary_emb_uncond = model.audio_rope(
                ref_audio_coords_uncond,
                device=uncond_kwargs["audio_hidden_states"].device,
                out_dtype=uncond_kwargs["audio_hidden_states"].dtype,
            )
            ref_ca_video_rotary_emb_uncond = model.cross_attn_rope(
                ref_video_coords_uncond[:, 0:1, :],
                device=uncond_kwargs["hidden_states"].device,
                out_dtype=uncond_kwargs["hidden_states"].dtype,
            )
            ref_ca_audio_rotary_emb_uncond = model.cross_attn_audio_rope(
                ref_audio_coords_uncond[:, 0:1, :],
                device=uncond_kwargs["audio_hidden_states"].device,
                out_dtype=uncond_kwargs["audio_hidden_states"].dtype,
            )

            ref_video_coords_cond = cond_kwargs.get("video_coords")
            if ref_video_coords_cond is None:
                ref_video_coords_cond = model.rope.prepare_video_coords(
                    batch_size=1,
                    num_frames=int(cond_kwargs["num_frames"]),
                    height=int(cond_kwargs["height"]),
                    width=int(cond_kwargs["width"]),
                    device=cond_kwargs["hidden_states"].device,
                    fps=float(cond_kwargs.get("fps", 24.0)),
                    start_frame=0,
                )
            ref_audio_coords_cond = cond_kwargs.get("audio_coords")
            if ref_audio_coords_cond is None:
                ref_audio_coords_cond = model.audio_rope.prepare_audio_coords(
                    batch_size=1,
                    num_frames=int(cond_kwargs["audio_num_frames"]),
                    device=cond_kwargs["audio_hidden_states"].device,
                )
            ref_video_coords_cond = model._maybe_quantize_video_rope_coords(
                ref_video_coords_cond,
                cond_kwargs["hidden_states"].device,
                cond_kwargs["hidden_states"].dtype,
            )
            ref_audio_coords_cond = ref_audio_coords_cond.to(
                device=cond_kwargs["audio_hidden_states"].device
            )
            ref_video_rotary_emb_cond = model.rope(
                ref_video_coords_cond,
                device=cond_kwargs["hidden_states"].device,
                out_dtype=cond_kwargs["hidden_states"].dtype,
            )
            ref_audio_rotary_emb_cond = model.audio_rope(
                ref_audio_coords_cond,
                device=cond_kwargs["audio_hidden_states"].device,
                out_dtype=cond_kwargs["audio_hidden_states"].dtype,
            )
            ref_ca_video_rotary_emb_cond = model.cross_attn_rope(
                ref_video_coords_cond[:, 0:1, :],
                device=cond_kwargs["hidden_states"].device,
                out_dtype=cond_kwargs["hidden_states"].dtype,
            )
            ref_ca_audio_rotary_emb_cond = model.cross_attn_audio_rope(
                ref_audio_coords_cond[:, 0:1, :],
                device=cond_kwargs["audio_hidden_states"].device,
                out_dtype=cond_kwargs["audio_hidden_states"].dtype,
            )

            probe_block = model.transformer_blocks[probe_block_index]
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                batched_block_video, batched_block_audio = probe_block(
                    batched_patchify_video,
                    batched_patchify_audio,
                    batched_caption_video,
                    batched_caption_audio,
                    temb=batched_temb,
                    temb_audio=batched_temb_audio,
                    temb_prompt=batched_temb_prompt,
                    temb_audio_prompt=batched_temb_audio_prompt,
                    temb_ca_scale_shift=batched_temb_ca_scale_shift,
                    temb_ca_audio_scale_shift=batched_temb_ca_audio_scale_shift,
                    temb_ca_gate=batched_temb_ca_gate,
                    temb_ca_audio_gate=batched_temb_ca_audio_gate,
                    video_rotary_emb=batched_video_rotary_emb,
                    audio_rotary_emb=batched_audio_rotary_emb,
                    ca_video_rotary_emb=batched_ca_video_rotary_emb,
                    ca_audio_rotary_emb=batched_ca_audio_rotary_emb,
                    encoder_attention_mask=batched_model_kwargs.get("encoder_attention_mask"),
                    audio_encoder_attention_mask=batched_model_kwargs.get(
                        "audio_encoder_attention_mask"
                    ),
                    video_self_attention_mask=batched_model_kwargs.get(
                        "video_self_attention_mask"
                    ),
                    audio_self_attention_mask=batched_model_kwargs.get(
                        "audio_self_attention_mask"
                    ),
                    a2v_cross_attention_mask=batched_model_kwargs.get(
                        "a2v_cross_attention_mask"
                    ),
                    v2a_cross_attention_mask=batched_model_kwargs.get(
                        "v2a_cross_attention_mask"
                    ),
                    audio_replicated_for_sp=bool(
                        batched_model_kwargs.get("audio_replicated_for_sp", False)
                    ),
                )
            batched_block_trace = getattr(probe_block, "_ltx2_probe_stage_trace", None)
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                ref_block_video_uncond, ref_block_audio_uncond = probe_block(
                    ref_patchify_video_uncond,
                    ref_patchify_audio_uncond,
                    ref_caption_video_uncond,
                    ref_caption_audio_uncond,
                    temb=ref_temb_uncond,
                    temb_audio=ref_temb_audio_uncond,
                    temb_prompt=ref_temb_prompt_uncond,
                    temb_audio_prompt=ref_temb_audio_prompt_uncond,
                    temb_ca_scale_shift=ref_temb_ca_scale_shift_uncond,
                    temb_ca_audio_scale_shift=ref_temb_ca_audio_scale_shift_uncond,
                    temb_ca_gate=ref_temb_ca_gate_uncond,
                    temb_ca_audio_gate=ref_temb_ca_audio_gate_uncond,
                    video_rotary_emb=ref_video_rotary_emb_uncond,
                    audio_rotary_emb=ref_audio_rotary_emb_uncond,
                    ca_video_rotary_emb=ref_ca_video_rotary_emb_uncond,
                    ca_audio_rotary_emb=ref_ca_audio_rotary_emb_uncond,
                    encoder_attention_mask=uncond_kwargs.get(
                        "encoder_attention_mask"
                    ),
                    audio_encoder_attention_mask=uncond_kwargs.get(
                        "audio_encoder_attention_mask"
                    ),
                    video_self_attention_mask=uncond_kwargs.get(
                        "video_self_attention_mask"
                    ),
                    audio_self_attention_mask=uncond_kwargs.get(
                        "audio_self_attention_mask"
                    ),
                    a2v_cross_attention_mask=uncond_kwargs.get(
                        "a2v_cross_attention_mask"
                    ),
                    v2a_cross_attention_mask=uncond_kwargs.get(
                        "v2a_cross_attention_mask"
                    ),
                    audio_replicated_for_sp=bool(
                        uncond_kwargs.get("audio_replicated_for_sp", False)
                    ),
                )
            ref_block_trace_uncond = getattr(
                probe_block, "_ltx2_probe_stage_trace", None
            )
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                ref_block_video_cond, ref_block_audio_cond = probe_block(
                    ref_patchify_video_cond,
                    ref_patchify_audio_cond,
                    ref_caption_video_cond,
                    ref_caption_audio_cond,
                    temb=ref_temb_cond,
                    temb_audio=ref_temb_audio_cond,
                    temb_prompt=ref_temb_prompt_cond,
                    temb_audio_prompt=ref_temb_audio_prompt_cond,
                    temb_ca_scale_shift=ref_temb_ca_scale_shift_cond,
                    temb_ca_audio_scale_shift=ref_temb_ca_audio_scale_shift_cond,
                    temb_ca_gate=ref_temb_ca_gate_cond,
                    temb_ca_audio_gate=ref_temb_ca_audio_gate_cond,
                    video_rotary_emb=ref_video_rotary_emb_cond,
                    audio_rotary_emb=ref_audio_rotary_emb_cond,
                    ca_video_rotary_emb=ref_ca_video_rotary_emb_cond,
                    ca_audio_rotary_emb=ref_ca_audio_rotary_emb_cond,
                    encoder_attention_mask=cond_kwargs.get("encoder_attention_mask"),
                    audio_encoder_attention_mask=cond_kwargs.get(
                        "audio_encoder_attention_mask"
                    ),
                    video_self_attention_mask=cond_kwargs.get(
                        "video_self_attention_mask"
                    ),
                    audio_self_attention_mask=cond_kwargs.get(
                        "audio_self_attention_mask"
                    ),
                    a2v_cross_attention_mask=cond_kwargs.get(
                        "a2v_cross_attention_mask"
                    ),
                    v2a_cross_attention_mask=cond_kwargs.get(
                        "v2a_cross_attention_mask"
                    ),
                    audio_replicated_for_sp=bool(
                        cond_kwargs.get("audio_replicated_for_sp", False)
                    ),
                )
            ref_block_trace_cond = getattr(probe_block, "_ltx2_probe_stage_trace", None)
            block_key = f"block{probe_block_index}"
            pretrace[block_key] = self._ltx2_probe_cfg_report_from_batched_tensors(
                reference_video_uncond=ref_block_video_uncond.float(),
                reference_video_cond=ref_block_video_cond.float(),
                reference_audio_uncond=ref_block_audio_uncond.float(),
                reference_audio_cond=ref_block_audio_cond.float(),
                candidate_video=batched_block_video.float(),
                candidate_audio=batched_block_audio.float(),
            )
            if (
                batched_block_trace is not None
                and ref_block_trace_uncond is not None
                and ref_block_trace_cond is not None
            ):
                for trace_key, trace_name in (
                    ("before_block", f"{block_key}_before_block"),
                    ("after_self_attn", f"{block_key}_after_self_attn"),
                    ("after_prompt_cross_attn", f"{block_key}_after_prompt_cross_attn"),
                    ("after_av_cross_attn", f"{block_key}_after_av_cross_attn"),
                    ("after_ff", f"{block_key}_after_ff"),
                ):
                    batched_video_stage, batched_audio_stage = batched_block_trace[
                        trace_key
                    ]
                    ref_video_stage_uncond, ref_audio_stage_uncond = (
                        ref_block_trace_uncond[trace_key]
                    )
                    ref_video_stage_cond, ref_audio_stage_cond = (
                        ref_block_trace_cond[trace_key]
                    )
                    pretrace[trace_name] = (
                        self._ltx2_probe_cfg_report_from_batched_tensors(
                            reference_video_uncond=ref_video_stage_uncond.float(),
                            reference_video_cond=ref_video_stage_cond.float(),
                            reference_audio_uncond=ref_audio_stage_uncond.float(),
                            reference_audio_cond=ref_audio_stage_cond.float(),
                            candidate_video=batched_video_stage.float(),
                            candidate_audio=batched_audio_stage.float(),
                        )
                    )

        # B=3 test: batch [uncond, cond, uncond_dup] and compare each item
        # against the B=1 reference to measure batch-size-dependent drift.
        b3_test: dict[str, object] | None = None
        if not is_sp_pair_probe:
            uncond_kwargs_dup = self._ltx2_probe_clone_model_kwargs(uncond_kwargs)
            b3_kwargs = self._ltx2_probe_merge_model_kwargs(
                uncond_kwargs, cond_kwargs, uncond_kwargs_dup
            )
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                b3_video, b3_audio = step.current_model(**b3_kwargs)
            b3_video = b3_video.float()
            b3_audio = b3_audio.float()
            b3_test = {
                "item0_vs_ref_uncond": {
                    "video": self._ltx2_probe_tensor_report(
                        ref_video_uncond.float(), b3_video[0:1]
                    ),
                    "audio": self._ltx2_probe_tensor_report(
                        ref_audio_uncond.float(), b3_audio[0:1]
                    ),
                },
                "item1_vs_ref_cond": {
                    "video": self._ltx2_probe_tensor_report(
                        ref_video_cond.float(), b3_video[1:2]
                    ),
                    "audio": self._ltx2_probe_tensor_report(
                        ref_audio_cond.float(), b3_audio[1:2]
                    ),
                },
                "item2_vs_ref_uncond_dup": {
                    "video": self._ltx2_probe_tensor_report(
                        ref_video_uncond.float(), b3_video[2:3]
                    ),
                    "audio": self._ltx2_probe_tensor_report(
                        ref_audio_uncond.float(), b3_audio[2:3]
                    ),
                },
                "item0_vs_item2_self_consistency": {
                    "video": self._ltx2_probe_tensor_report(
                        b3_video[0:1], b3_video[2:3]
                    ),
                    "audio": self._ltx2_probe_tensor_report(
                        b3_audio[0:1], b3_audio[2:3]
                    ),
                },
            }

        self._write_ltx2_probe_state(
            {
                "official_cfg": self._ltx2_probe_cfg_report_from_batched_tensors(
                    reference_video_uncond=ref_video_uncond.float(),
                    reference_video_cond=ref_video_cond.float(),
                    reference_audio_uncond=ref_audio_uncond.float(),
                    reference_audio_cond=ref_audio_cond.float(),
                    candidate_video=batched_out_video.float(),
                    candidate_audio=batched_out_audio.float(),
                ),
                "official_cfg_reference_repeat": self._ltx2_probe_cfg_report(
                    (
                        ref_video_uncond.float(),
                        ref_video_cond.float(),
                        ref_audio_uncond.float(),
                        ref_audio_cond.float(),
                    ),
                    (
                        ref2_video_uncond.float(),
                        ref2_video_cond.float(),
                        ref2_audio_uncond.float(),
                        ref2_audio_cond.float(),
                    ),
                ),
                "official_cfg_pretrace": pretrace,
                "b3_test": b3_test,
            }
        )

    def _maybe_finalize_ltx2_probe_state(
        self, *, ctx: LTX2DenoisingContext, step: DenoisingStepState
    ) -> None:
        if (
            self._ltx2_probe_output_path() is None
            or getattr(self, "_ltx2_probe_written", False)
            or ctx.stage != "stage1"
            or int(step.step_index) != 0
        ):
            return
        self._write_ltx2_probe_state({})

    def _run_ltx2_model_forward(
        self,
        *,
        step: DenoisingStepState,
        model_kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with set_forward_context(
            current_timestep=step.step_index, attn_metadata=step.attn_metadata
        ):
            out_video, out_audio = step.current_model(**model_kwargs)
        out_video = out_video.float()
        out_audio = out_audio.float()
        self._record_ltx2_probe_model_call(step=step, model_kwargs=model_kwargs)
        self._maybe_probe_ltx2_official_cfg_forward(
            step=step,
            model_kwargs=model_kwargs,
            out_video=out_video,
            out_audio=out_audio,
        )
        return out_video, out_audio

    def _run_ltx2_stage1_aux_forward(
        self,
        *,
        step: DenoisingStepState,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        ctx: LTX2DenoisingContext,
        batch: Req,
        audio_num_frames_latent: int,
        video_coords: torch.Tensor | None,
        audio_coords: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        video_self_attention_mask: torch.Tensor | None,
        audio_self_attention_mask: torch.Tensor | None,
        a2v_cross_attention_mask: torch.Tensor | None,
        v2a_cross_attention_mask: torch.Tensor | None,
        skip_video_self_attn_blocks: tuple[int, ...] | None = None,
        skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
        disable_a2v_cross_attn: bool = False,
        disable_v2a_cross_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_kwargs = self._build_ltx2_model_kwargs(
            latent_model_input=latent_model_input,
            audio_latent_model_input=audio_latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            encoder_attention_mask=encoder_attention_mask,
            ctx=ctx,
            batch=batch,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=video_coords,
            audio_coords=audio_coords,
            prompt_timestep_video=prompt_timestep_video,
            prompt_timestep_audio=prompt_timestep_audio,
            video_self_attention_mask=video_self_attention_mask,
            audio_self_attention_mask=audio_self_attention_mask,
            a2v_cross_attention_mask=a2v_cross_attention_mask,
            v2a_cross_attention_mask=v2a_cross_attention_mask,
            skip_video_self_attn_blocks=skip_video_self_attn_blocks,
            skip_audio_self_attn_blocks=skip_audio_self_attn_blocks,
            disable_a2v_cross_attn=disable_a2v_cross_attn,
            disable_v2a_cross_attn=disable_v2a_cross_attn,
        )
        return self._run_ltx2_model_forward(step=step, model_kwargs=model_kwargs)

    def _run_ltx2_stage1_batched_aux_forward(
        self,
        *,
        step: DenoisingStepState,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        ctx: LTX2DenoisingContext,
        batch: Req,
        batch_size: int,
        audio_num_frames_latent: int,
        video_coords: torch.Tensor | None,
        audio_coords: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        video_self_attention_mask: torch.Tensor | None,
        audio_self_attention_mask: torch.Tensor | None,
        a2v_cross_attention_mask: torch.Tensor | None,
        v2a_cross_attention_mask: torch.Tensor | None,
        pass_specs: list[tuple[str, dict[str, object]]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        num_passes = len(pass_specs)
        expanded_batch_size = batch_size * num_passes
        perturbation_configs = tuple(
            perturbation_config
            for _, perturbation_config in pass_specs
            for _ in range(batch_size)
        )
        batched_encoder_attention_mask = None
        if encoder_attention_mask is not None:
            batched_encoder_attention_mask = torch.cat(
                [encoder_attention_mask] * num_passes, dim=0
            )
        model_kwargs = self._build_ltx2_model_kwargs(
            latent_model_input=self._repeat_batch_dim(
                latent_model_input, expanded_batch_size
            ),
            audio_latent_model_input=self._repeat_batch_dim(
                audio_latent_model_input, expanded_batch_size
            ),
            encoder_hidden_states=torch.cat([encoder_hidden_states] * num_passes, dim=0),
            audio_encoder_hidden_states=torch.cat(
                [audio_encoder_hidden_states] * num_passes, dim=0
            ),
            timestep_video=self._repeat_batch_dim(timestep_video, expanded_batch_size),
            timestep_audio=self._repeat_batch_dim(timestep_audio, expanded_batch_size),
            encoder_attention_mask=batched_encoder_attention_mask,
            ctx=ctx,
            batch=batch,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=self._repeat_batch_dim_or_none(
                video_coords, expanded_batch_size
            ),
            audio_coords=self._repeat_batch_dim_or_none(
                audio_coords, expanded_batch_size
            ),
            prompt_timestep_video=self._repeat_batch_dim_or_none(
                prompt_timestep_video, expanded_batch_size
            ),
            prompt_timestep_audio=self._repeat_batch_dim_or_none(
                prompt_timestep_audio, expanded_batch_size
            ),
            video_self_attention_mask=self._repeat_batch_dim_or_none(
                video_self_attention_mask, expanded_batch_size
            ),
            audio_self_attention_mask=self._repeat_batch_dim_or_none(
                audio_self_attention_mask, expanded_batch_size
            ),
            a2v_cross_attention_mask=self._repeat_batch_dim_or_none(
                a2v_cross_attention_mask, expanded_batch_size
            ),
            v2a_cross_attention_mask=self._repeat_batch_dim_or_none(
                v2a_cross_attention_mask, expanded_batch_size
            ),
            perturbation_configs=perturbation_configs,
        )
        batched_video, batched_audio = self._run_ltx2_model_forward(
            step=step, model_kwargs=model_kwargs
        )
        return {
            pass_name: (video_chunk, audio_chunk)
            for (pass_name, _), video_chunk, audio_chunk in zip(
                pass_specs,
                batched_video.chunk(num_passes, dim=0),
                batched_audio.chunk(num_passes, dim=0),
                strict=True,
            )
        }

    def _run_ltx2_cfg_batched_forward(
        self,
        *,
        step: DenoisingStepState,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        negative_encoder_hidden_states: torch.Tensor,
        negative_audio_encoder_hidden_states: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        negative_encoder_attention_mask: torch.Tensor | None,
        ctx: LTX2DenoisingContext,
        batch: Req,
        batch_size: int,
        audio_num_frames_latent: int,
        video_coords: torch.Tensor | None,
        audio_coords: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        video_self_attention_mask: torch.Tensor | None,
        audio_self_attention_mask: torch.Tensor | None,
        a2v_cross_attention_mask: torch.Tensor | None,
        v2a_cross_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg_batch_size = batch_size * 2
        batched_encoder_attention_mask = None
        if encoder_attention_mask is not None:
            batched_encoder_attention_mask = torch.cat(
                [negative_encoder_attention_mask, encoder_attention_mask], dim=0
            )
        model_kwargs = self._build_ltx2_model_kwargs(
            latent_model_input=self._repeat_batch_dim(
                latent_model_input, cfg_batch_size
            ),
            audio_latent_model_input=self._repeat_batch_dim(
                audio_latent_model_input, cfg_batch_size
            ),
            encoder_hidden_states=torch.cat(
                [negative_encoder_hidden_states, encoder_hidden_states], dim=0
            ),
            audio_encoder_hidden_states=torch.cat(
                [
                    negative_audio_encoder_hidden_states,
                    audio_encoder_hidden_states,
                ],
                dim=0,
            ),
            timestep_video=self._repeat_batch_dim(timestep_video, cfg_batch_size),
            timestep_audio=self._repeat_batch_dim(timestep_audio, cfg_batch_size),
            encoder_attention_mask=batched_encoder_attention_mask,
            ctx=ctx,
            batch=batch,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=video_coords,
            audio_coords=audio_coords,
            prompt_timestep_video=self._repeat_batch_dim_or_none(
                prompt_timestep_video, cfg_batch_size
            ),
            prompt_timestep_audio=self._repeat_batch_dim_or_none(
                prompt_timestep_audio, cfg_batch_size
            ),
            video_self_attention_mask=self._repeat_batch_dim_or_none(
                video_self_attention_mask, cfg_batch_size
            ),
            audio_self_attention_mask=self._repeat_batch_dim_or_none(
                audio_self_attention_mask, cfg_batch_size
            ),
            a2v_cross_attention_mask=self._repeat_batch_dim_or_none(
                a2v_cross_attention_mask, cfg_batch_size
            ),
            v2a_cross_attention_mask=self._repeat_batch_dim_or_none(
                v2a_cross_attention_mask, cfg_batch_size
            ),
        )
        batched_cfg_video, batched_cfg_audio = self._run_ltx2_model_forward(
            step=step, model_kwargs=model_kwargs
        )
        if batched_cfg_video.shape[0] != cfg_batch_size:
            raise ValueError(
                "Batched CFG video output batch size mismatch: "
                f"{batched_cfg_video.shape[0]} != {cfg_batch_size}."
            )
        if batched_cfg_audio.shape[0] != cfg_batch_size:
            raise ValueError(
                "Batched CFG audio output batch size mismatch: "
                f"{batched_cfg_audio.shape[0]} != {cfg_batch_size}."
            )
        v_neg, v_pos = batched_cfg_video.chunk(2, dim=0)
        a_v_neg, a_v_pos = batched_cfg_audio.chunk(2, dim=0)
        return v_neg, v_pos, a_v_neg, a_v_pos

    def _run_ltx2_cfg_sequential_forward(
        self,
        *,
        step: DenoisingStepState,
        latent_model_input: torch.Tensor,
        audio_latent_model_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        negative_encoder_hidden_states: torch.Tensor,
        negative_audio_encoder_hidden_states: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        negative_encoder_attention_mask: torch.Tensor | None,
        ctx: LTX2DenoisingContext,
        batch: Req,
        audio_num_frames_latent: int,
        video_coords: torch.Tensor | None,
        audio_coords: torch.Tensor | None,
        prompt_timestep_video: torch.Tensor | None,
        prompt_timestep_audio: torch.Tensor | None,
        video_self_attention_mask: torch.Tensor | None,
        audio_self_attention_mask: torch.Tensor | None,
        a2v_cross_attention_mask: torch.Tensor | None,
        v2a_cross_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_pos, a_v_pos = self._run_ltx2_stage1_aux_forward(
            step=step,
            latent_model_input=latent_model_input,
            audio_latent_model_input=audio_latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            encoder_attention_mask=encoder_attention_mask,
            ctx=ctx,
            batch=batch,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=video_coords,
            audio_coords=audio_coords,
            prompt_timestep_video=prompt_timestep_video,
            prompt_timestep_audio=prompt_timestep_audio,
            video_self_attention_mask=video_self_attention_mask,
            audio_self_attention_mask=audio_self_attention_mask,
            a2v_cross_attention_mask=a2v_cross_attention_mask,
            v2a_cross_attention_mask=v2a_cross_attention_mask,
        )
        v_neg, a_v_neg = self._run_ltx2_stage1_aux_forward(
            step=step,
            latent_model_input=latent_model_input,
            audio_latent_model_input=audio_latent_model_input,
            encoder_hidden_states=negative_encoder_hidden_states,
            audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            encoder_attention_mask=negative_encoder_attention_mask,
            ctx=ctx,
            batch=batch,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=video_coords,
            audio_coords=audio_coords,
            prompt_timestep_video=prompt_timestep_video,
            prompt_timestep_audio=prompt_timestep_audio,
            video_self_attention_mask=video_self_attention_mask,
            audio_self_attention_mask=audio_self_attention_mask,
            a2v_cross_attention_mask=a2v_cross_attention_mask,
            v2a_cross_attention_mask=v2a_cross_attention_mask,
        )
        return v_neg, v_pos, a_v_neg, a_v_pos

    @staticmethod
    def _should_use_ltx2_single_gpu_batched_forward(
        *,
        server_args: ServerArgs,
        ctx: LTX2DenoisingContext,
        batch: Req,
    ) -> bool:
        if os.getenv("SGLANG_LTX2_FORCE_SEQUENTIAL_FORWARD", "0") == "1":
            return False
        return bool(
            server_args.num_gpus == 1
            and get_sp_world_size() == 1
            and not getattr(batch, "did_sp_shard_latents", False)
            and not getattr(batch, "did_sp_shard_audio_latents", False)
            and not ctx.replicate_audio_for_sp
        )

    @classmethod
    def _should_use_ltx23_native_one_stage_semantics(
        cls,
        server_args: ServerArgs,
        pipeline_name: str | None,
    ) -> bool:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            return False
        if server_args.pipeline_class_name == "LTX2TwoStagePipeline":
            return False
        return pipeline_name != "LTX2TwoStagePipeline"

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
            img_array = LTX2DenoisingStage._apply_video_codec_compression(
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

    @staticmethod
    def _should_replicate_ltx23_audio_for_sp(
        batch: Req,
        server_args: ServerArgs,
        *,
        is_ltx23_variant: bool,
    ) -> bool:
        return bool(
            is_ltx23_variant
            and get_sp_world_size() > 1
            and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                batch.audio_latents
            )
        )

    def _get_condition_image_encoder(
        self,
        server_args: ServerArgs,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LTX23VideoConditionEncoder | None:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir:
            return None

        vae_model_path = server_args.model_paths["vae"]
        encoder_dir = os.path.join(vae_model_path, encoder_subdir)
        config_path = os.path.join(encoder_dir, "config.json")
        weights_path = os.path.join(encoder_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(
                f"LTX-2 condition encoder files not found under {encoder_dir}"
            )

        cached_dir = self._condition_image_encoder_dir
        encoder = self._condition_image_encoder
        if encoder is None or cached_dir != encoder_dir:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            encoder = LTX23VideoConditionEncoder(config)
            encoder.load_state_dict(safetensors_load_file(weights_path), strict=True)
            self._condition_image_encoder = encoder
            self._condition_image_encoder_dir = encoder_dir

        encoder = encoder.to(device=device, dtype=dtype)
        return encoder

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
        img_array = np.array(img).astype(np.uint8)[..., :3]
        img_array = self._apply_video_codec_compression(img_array, crf=33)
        conditioned_img = PIL.Image.fromarray(img_array)
        batch.condition_image = self._resize_center_crop(
            conditioned_img, width=int(batch.width), height=int(batch.height)
        )

        latents_device = (
            batch.latents.device
            if isinstance(batch.latents, torch.Tensor)
            else torch.device("cpu")
        )
        encode_dtype = batch.latents.dtype
        original_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            original_dtype != torch.float32
        ) and not server_args.disable_autocast
        condition_image_encoder = self._get_condition_image_encoder(
            server_args, device=latents_device, dtype=encode_dtype
        )
        if condition_image_encoder is None:
            self.vae = self.vae.to(device=latents_device, dtype=encode_dtype)

        video_condition = self._resize_center_crop_tensor(
            conditioned_img,
            width=int(batch.width),
            height=int(batch.height),
            device=latents_device,
            dtype=encode_dtype,
            apply_codec_compression=False,
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=original_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if (
                    condition_image_encoder is None
                    and server_args.pipeline_config.vae_tiling
                ):
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(encode_dtype)

            if condition_image_encoder is not None:
                latent = condition_image_encoder(video_condition)
            else:
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

        if condition_image_encoder is None:
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
        else:
            latent = latent.to(dtype=encode_dtype)

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

        if condition_image_encoder is None:
            self.vae.to(original_dtype)
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
            if condition_image_encoder is not None:
                self._condition_image_encoder = condition_image_encoder.to("cpu")

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        """Extend the base context with LTX-2 audio, SP, and TI2V state."""
        with self._ltx_perf_context("LTX2PrepareDenoisingLoop", batch):
            self._disable_cache_dit_for_request = batch.image_path is not None
            base_ctx = super()._prepare_denoising_loop(batch, server_args)
            ctx = LTX2DenoisingContext(**base_ctx.to_kwargs())
            ctx.is_ltx23_variant = is_ltx23_native_variant(
                server_args.pipeline_config.vae_config.arch_config
            )
            phase = batch.extra.get("ltx2_phase")
            pipeline = self.pipeline() if self.pipeline else None
            pipeline_name = pipeline.pipeline_name if pipeline is not None else None
            ctx.use_ltx23_native_one_stage_semantics = (
                self._should_use_ltx23_native_one_stage_semantics(
                    server_args, pipeline_name
                )
            )
            ctx.stage = (
                phase
                if phase is not None
                else (
                    "stage1"
                    if ctx.use_ltx23_native_one_stage_semantics
                    else "one_stage"
                )
            )
            ctx.audio_latents = batch.audio_latents
            # Video and audio keep separate scheduler state throughout the denoising loop.
            ctx.audio_scheduler = copy.deepcopy(self.scheduler)

            # Prepare image latents and embeddings for LTX-2 TI2V generation.
            self._prepare_ltx2_image_latent(batch, server_args)
            do_ti2v = self._should_apply_ltx2_ti2v(batch)

            if ctx.use_ltx23_native_one_stage_semantics:
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
            ctx.latent_num_frames_for_model = (
                self._get_video_latent_num_frames_for_model(
                    batch=batch, server_args=server_args, latents=ctx.latents
                )
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
                if not (
                    isinstance(ctx.latents, torch.Tensor) and ctx.latents.ndim == 3
                ):
                    raise ValueError(
                        "LTX-2 TI2V expects packed token latents [B, S, D]."
                    )
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
        with self._ltx_perf_context("LTX2BeforeDenoisingLoop", batch):
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
        """Preserve the base LTX-2 attention-metadata contract."""
        # LTX-2 uses the plain attention-metadata builder call here.
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
        self._ltx2_probe_current_stage = ctx.stage
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
        if not ctx.use_ltx23_native_one_stage_semantics:
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
        elif ctx.is_ltx23_variant and not ctx.use_ltx23_native_one_stage_semantics:
            timestep_video = timestep.view(batch_size, 1).expand(
                batch_size, int(latent_model_input.shape[1])
            )
        else:
            timestep_video = timestep

        if (
            ctx.is_ltx23_variant
            and not ctx.use_ltx23_native_one_stage_semantics
            and audio_latent_model_input.ndim == 3
        ):
            timestep_audio = timestep.view(batch_size, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1])
            )
        else:
            timestep_audio = timestep

        prompt_timestep_video = None
        prompt_timestep_audio = None
        if ctx.is_ltx23_variant and not ctx.use_ltx23_native_one_stage_semantics:
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
        if ctx.use_ltx23_native_one_stage_semantics:
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

        # 5. Run the branch-specific LTX forward path and apply CFG/guider logic.
        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant
                and not ctx.use_ltx23_native_one_stage_semantics
            ),
        )
        use_single_gpu_batched_forward = (
            self._should_use_ltx2_single_gpu_batched_forward(
                server_args=server_args, ctx=ctx, batch=batch
            )
        )
        use_official_cfg_path = stage1_guider_params is None
        if use_official_cfg_path:
            encoder_hidden_states = batch.prompt_embeds[0]
            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
            encoder_attention_mask = prompt_attention_mask
            if batch.do_classifier_free_guidance and use_single_gpu_batched_forward:
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
                                    and not ctx.use_ltx23_native_one_stage_semantics
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
                prompt_timestep_video = self._repeat_batch_dim_or_none(
                    prompt_timestep_video, cfg_batch_size
                )
                prompt_timestep_audio = self._repeat_batch_dim_or_none(
                    prompt_timestep_audio, cfg_batch_size
                )
                video_self_attention_mask = self._repeat_batch_dim_or_none(
                    video_self_attention_mask, cfg_batch_size
                )
                audio_self_attention_mask = self._repeat_batch_dim_or_none(
                    audio_self_attention_mask, cfg_batch_size
                )
                a2v_cross_attention_mask = self._repeat_batch_dim_or_none(
                    a2v_cross_attention_mask, cfg_batch_size
                )
                v2a_cross_attention_mask = self._repeat_batch_dim_or_none(
                    v2a_cross_attention_mask, cfg_batch_size
                )

            first_step_forward_context = (
                self._ltx_perf_context("LTX2FirstStepForward", batch)
                if step.step_index == 0
                else nullcontext()
            )
            with first_step_forward_context:
                if batch.do_classifier_free_guidance:
                    if use_single_gpu_batched_forward:
                        model_video, model_audio = self._run_ltx2_model_forward(
                            step=step,
                            model_kwargs=self._build_ltx2_model_kwargs(
                                latent_model_input=latent_model_input,
                                audio_latent_model_input=audio_latent_model_input,
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                timestep_video=timestep_video,
                                timestep_audio=timestep_audio,
                                encoder_attention_mask=encoder_attention_mask,
                                ctx=ctx,
                                batch=batch,
                                audio_num_frames_latent=audio_num_frames_latent,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                prompt_timestep_video=prompt_timestep_video,
                                prompt_timestep_audio=prompt_timestep_audio,
                                video_self_attention_mask=video_self_attention_mask,
                                audio_self_attention_mask=audio_self_attention_mask,
                                a2v_cross_attention_mask=a2v_cross_attention_mask,
                                v2a_cross_attention_mask=v2a_cross_attention_mask,
                            ),
                        )
                        model_video_uncond, model_video_text = model_video.chunk(2)
                        model_audio_uncond, model_audio_text = model_audio.chunk(2)
                    else:
                        (
                            model_video_uncond,
                            model_video_text,
                            model_audio_uncond,
                            model_audio_text,
                        ) = self._run_ltx2_cfg_sequential_forward(
                            step=step,
                            latent_model_input=ctx.latents.to(ctx.target_dtype),
                            audio_latent_model_input=ctx.audio_latents.to(
                                ctx.target_dtype
                            ),
                            encoder_hidden_states=batch.prompt_embeds[0],
                            audio_encoder_hidden_states=batch.audio_prompt_embeds[0],
                            negative_encoder_hidden_states=batch.negative_prompt_embeds[
                                0
                            ],
                            negative_audio_encoder_hidden_states=batch.negative_audio_prompt_embeds[
                                0
                            ],
                            timestep_video=timestep_video,
                            timestep_audio=timestep_audio,
                            encoder_attention_mask=prompt_attention_mask,
                            negative_encoder_attention_mask=self._get_ltx_prompt_attention_mask(
                                batch,
                                is_ltx23_variant=(
                                    ctx.is_ltx23_variant
                                    and not ctx.use_ltx23_native_one_stage_semantics
                                ),
                                negative=True,
                            ),
                            ctx=ctx,
                            batch=batch,
                            audio_num_frames_latent=audio_num_frames_latent,
                            video_coords=video_coords,
                            audio_coords=audio_coords,
                            prompt_timestep_video=prompt_timestep_video,
                            prompt_timestep_audio=prompt_timestep_audio,
                            video_self_attention_mask=video_self_attention_mask,
                            audio_self_attention_mask=audio_self_attention_mask,
                            a2v_cross_attention_mask=a2v_cross_attention_mask,
                            v2a_cross_attention_mask=v2a_cross_attention_mask,
                        )
                    model_video = model_video_uncond + (
                        batch.guidance_scale * (model_video_text - model_video_uncond)
                    )
                    model_audio = model_audio_uncond + (
                        batch.guidance_scale * (model_audio_text - model_audio_uncond)
                    )
                else:
                    model_video, model_audio = self._run_ltx2_model_forward(
                        step=step,
                        model_kwargs=self._build_ltx2_model_kwargs(
                            latent_model_input=latent_model_input,
                            audio_latent_model_input=audio_latent_model_input,
                            encoder_hidden_states=encoder_hidden_states,
                            audio_encoder_hidden_states=audio_encoder_hidden_states,
                            timestep_video=timestep_video,
                            timestep_audio=timestep_audio,
                            encoder_attention_mask=encoder_attention_mask,
                            ctx=ctx,
                            batch=batch,
                            audio_num_frames_latent=audio_num_frames_latent,
                            video_coords=video_coords,
                            audio_coords=audio_coords,
                            prompt_timestep_video=prompt_timestep_video,
                            prompt_timestep_audio=prompt_timestep_audio,
                            video_self_attention_mask=video_self_attention_mask,
                            audio_self_attention_mask=audio_self_attention_mask,
                            a2v_cross_attention_mask=a2v_cross_attention_mask,
                            v2a_cross_attention_mask=v2a_cross_attention_mask,
                        ),
                    )
                ctx.latents = self.scheduler.step(
                    model_video, step.t_device, ctx.latents, return_dict=False
                )[0]
                ctx.audio_latents = ctx.audio_scheduler.step(
                    model_audio, step.t_device, ctx.audio_latents, return_dict=False
                )[0]
                if ctx.denoise_mask is not None and ctx.clean_latent is not None:
                    ctx.latents = (
                        ctx.latents.float() * ctx.denoise_mask
                        + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
                    ).to(dtype=ctx.latents.dtype)
                ctx.latents = self.post_forward_for_ti2v_task(
                    batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
                )
                self._maybe_finalize_ltx2_probe_state(ctx=ctx, step=step)
            return

        encoder_hidden_states = batch.prompt_embeds[0]
        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
        encoder_attention_mask = prompt_attention_mask
        negative_encoder_hidden_states = batch.negative_prompt_embeds[0]
        negative_audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
        negative_encoder_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant
                and not ctx.use_ltx23_native_one_stage_semantics
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

        if use_single_gpu_batched_forward:
            v_neg, v_pos, a_v_neg, a_v_pos = self._run_ltx2_cfg_batched_forward(
                step=step,
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                negative_encoder_hidden_states=negative_encoder_hidden_states,
                negative_audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                encoder_attention_mask=encoder_attention_mask,
                negative_encoder_attention_mask=negative_encoder_attention_mask,
                ctx=ctx,
                batch=batch,
                batch_size=batch_size,
                audio_num_frames_latent=audio_num_frames_latent,
                video_coords=video_coords,
                audio_coords=audio_coords,
                prompt_timestep_video=prompt_timestep_video,
                prompt_timestep_audio=prompt_timestep_audio,
                video_self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_cross_attention_mask,
                v2a_cross_attention_mask=v2a_cross_attention_mask,
            )
        else:
            v_pos, a_v_pos = self._run_ltx2_stage1_aux_forward(
                step=step,
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                encoder_attention_mask=encoder_attention_mask,
                ctx=ctx,
                batch=batch,
                audio_num_frames_latent=audio_num_frames_latent,
                video_coords=video_coords,
                audio_coords=audio_coords,
                prompt_timestep_video=prompt_timestep_video,
                prompt_timestep_audio=prompt_timestep_audio,
                video_self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_cross_attention_mask,
                v2a_cross_attention_mask=v2a_cross_attention_mask,
            )
            v_neg, a_v_neg = self._run_ltx2_stage1_aux_forward(
                step=step,
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                encoder_hidden_states=negative_encoder_hidden_states,
                audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                encoder_attention_mask=negative_encoder_attention_mask,
                ctx=ctx,
                batch=batch,
                audio_num_frames_latent=audio_num_frames_latent,
                video_coords=video_coords,
                audio_coords=audio_coords,
                prompt_timestep_video=prompt_timestep_video,
                prompt_timestep_audio=prompt_timestep_audio,
                video_self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_cross_attention_mask,
                v2a_cross_attention_mask=v2a_cross_attention_mask,
            )

        v_ptb = None
        a_v_ptb = None
        v_mod = None
        a_v_mod = None
        if need_perturbed:
            if not use_single_gpu_batched_forward:
                v_ptb, a_v_ptb = self._run_ltx2_stage1_aux_forward(
                    step=step,
                    latent_model_input=latent_model_input,
                    audio_latent_model_input=audio_latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    timestep_video=timestep_video,
                    timestep_audio=timestep_audio,
                    encoder_attention_mask=encoder_attention_mask,
                    ctx=ctx,
                    batch=batch,
                    audio_num_frames_latent=audio_num_frames_latent,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    prompt_timestep_video=prompt_timestep_video,
                    prompt_timestep_audio=prompt_timestep_audio,
                    video_self_attention_mask=video_self_attention_mask,
                    audio_self_attention_mask=audio_self_attention_mask,
                    a2v_cross_attention_mask=a2v_cross_attention_mask,
                    v2a_cross_attention_mask=v2a_cross_attention_mask,
                    skip_video_self_attn_blocks=tuple(
                        stage1_guider_params["video_stg_blocks"]
                    ),
                    skip_audio_self_attn_blocks=tuple(
                        stage1_guider_params["audio_stg_blocks"]
                    ),
                )
                if need_modality:
                    v_mod, a_v_mod = self._run_ltx2_stage1_aux_forward(
                        step=step,
                        latent_model_input=latent_model_input,
                        audio_latent_model_input=audio_latent_model_input,
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        timestep_video=timestep_video,
                        timestep_audio=timestep_audio,
                        encoder_attention_mask=encoder_attention_mask,
                        ctx=ctx,
                        batch=batch,
                        audio_num_frames_latent=audio_num_frames_latent,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        prompt_timestep_video=prompt_timestep_video,
                        prompt_timestep_audio=prompt_timestep_audio,
                        video_self_attention_mask=video_self_attention_mask,
                        audio_self_attention_mask=audio_self_attention_mask,
                        a2v_cross_attention_mask=a2v_cross_attention_mask,
                        v2a_cross_attention_mask=v2a_cross_attention_mask,
                        disable_a2v_cross_attn=True,
                        disable_v2a_cross_attn=True,
                    )
            else:
                pass_specs = [
                    (
                        "perturbed",
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
                ]
                if need_modality:
                    pass_specs.append(
                        (
                            "modality",
                            {
                                "skip_video_self_attn_blocks": (),
                                "skip_audio_self_attn_blocks": (),
                                "skip_a2v_cross_attn": True,
                                "skip_v2a_cross_attn": True,
                            },
                        )
                    )
                aux_outputs = self._run_ltx2_stage1_batched_aux_forward(
                    step=step,
                    latent_model_input=latent_model_input,
                    audio_latent_model_input=audio_latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    timestep_video=timestep_video,
                    timestep_audio=timestep_audio,
                    encoder_attention_mask=encoder_attention_mask,
                    ctx=ctx,
                    batch=batch,
                    batch_size=batch_size,
                    audio_num_frames_latent=audio_num_frames_latent,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    prompt_timestep_video=prompt_timestep_video,
                    prompt_timestep_audio=prompt_timestep_audio,
                    video_self_attention_mask=video_self_attention_mask,
                    audio_self_attention_mask=audio_self_attention_mask,
                    a2v_cross_attention_mask=a2v_cross_attention_mask,
                    v2a_cross_attention_mask=v2a_cross_attention_mask,
                    pass_specs=pass_specs,
                )
                v_ptb, a_v_ptb = aux_outputs["perturbed"]
                v_mod, a_v_mod = aux_outputs.get("modality", (None, None))
        elif need_modality:
            v_mod, a_v_mod = self._run_ltx2_stage1_aux_forward(
                step=step,
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                encoder_attention_mask=encoder_attention_mask,
                ctx=ctx,
                batch=batch,
                audio_num_frames_latent=audio_num_frames_latent,
                video_coords=video_coords,
                audio_coords=audio_coords,
                prompt_timestep_video=prompt_timestep_video,
                prompt_timestep_audio=prompt_timestep_audio,
                video_self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_cross_attention_mask,
                v2a_cross_attention_mask=v2a_cross_attention_mask,
                disable_a2v_cross_attn=True,
                disable_v2a_cross_attn=True,
            )

        sigma_val = float(sigma.item())
        video_sigma_for_x0: float | torch.Tensor = sigma_val
        if ctx.denoise_mask is not None:
            video_sigma_for_x0 = sigma.to(
                device=ctx.latents.device, dtype=torch.float32
            ) * ctx.denoise_mask.squeeze(-1)

        denoised_video = self._ltx2_velocity_to_x0(
            ctx.latents, v_pos, video_sigma_for_x0
        )
        denoised_audio = self._ltx2_velocity_to_x0(
            ctx.audio_latents, a_v_pos, sigma_val
        )
        denoised_video_neg = self._ltx2_velocity_to_x0(
            ctx.latents, v_neg, video_sigma_for_x0
        )
        denoised_audio_neg = self._ltx2_velocity_to_x0(
            ctx.audio_latents, a_v_neg, sigma_val
        )
        denoised_video_perturbed = (
            None
            if v_ptb is None
            else self._ltx2_velocity_to_x0(ctx.latents, v_ptb, video_sigma_for_x0)
        )
        denoised_audio_perturbed = (
            None
            if a_v_ptb is None
            else self._ltx2_velocity_to_x0(ctx.audio_latents, a_v_ptb, sigma_val)
        )
        denoised_video_modality = (
            None
            if v_mod is None
            else self._ltx2_velocity_to_x0(ctx.latents, v_mod, video_sigma_for_x0)
        )
        denoised_audio_modality = (
            None
            if a_v_mod is None
            else self._ltx2_velocity_to_x0(ctx.audio_latents, a_v_mod, sigma_val)
        )

        if not video_skip:
            denoised_video = self._ltx2_calculate_guided_x0(
                cond=denoised_video,
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
            ctx.last_denoised_video = denoised_video
        elif ctx.last_denoised_video is not None:
            denoised_video = ctx.last_denoised_video

        if not audio_skip:
            denoised_audio = self._ltx2_calculate_guided_x0(
                cond=denoised_audio,
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
            ctx.last_denoised_audio = denoised_audio
        elif ctx.last_denoised_audio is not None:
            denoised_audio = ctx.last_denoised_audio

        if ctx.denoise_mask is not None and ctx.clean_latent is not None:
            denoised_video = (
                denoised_video * ctx.denoise_mask
                + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
            ).to(denoised_video.dtype)

        # 6. Convert x0 predictions back to velocity and update both latent streams.
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

        ctx.latents = (ctx.latents.float() + v_video.float() * dt).to(
            dtype=ctx.latents.dtype
        )
        ctx.audio_latents = (ctx.audio_latents.float() + v_audio.float() * dt).to(
            dtype=ctx.audio_latents.dtype
        )
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )
        self._maybe_finalize_ltx2_probe_state(ctx=ctx, step=step)

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
