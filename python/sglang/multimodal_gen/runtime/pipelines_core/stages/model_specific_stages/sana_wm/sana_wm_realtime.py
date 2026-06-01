# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    enable_ltx2_streaming_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

from .refiner import DiffusersLTX2Refiner
from .scheduler import SelfForcingFlowEulerSampler, SelfForcingSamplerConfig
from .utils import (
    action_string_to_c2w,
    estimate_intrinsics_with_pi3x,
    load_camera,
    load_intrinsics,
    pil_to_model_tensor,
    prepare_camera_conditions,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)

logger = init_logger(__name__)

SANA_WM_HEIGHT = 704
SANA_WM_WIDTH = 1280
DEFAULT_REFINER_BLOCK_SIZE = 3
DEFAULT_REFINER_KV_MAX_FRAMES = 11


@contextmanager
def _deterministic_vae_encode_context():
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic


def _as_int_tuple(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(v) for v in values)


def _normalize_camera_actions(payload: Any) -> list[list[str]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("camera_actions must be list[list[str]]")
    out: list[list[str]] = []
    for item in payload:
        if not isinstance(item, list):
            raise ValueError("camera_actions must be list[list[str]]")
        out.append([str(key).lower() for key in item])
    return out


def _actions_to_action_string(actions: list[list[str]]) -> str:
    if not actions:
        return "none-1"

    segments: list[str] = []
    current = tuple(sorted(set(actions[0])))
    count = 0
    for frame_actions in actions:
        normalized = tuple(sorted(set(frame_actions)))
        if normalized == current:
            count += 1
            continue
        key = "".join(current) if current else "none"
        segments.append(f"{key}-{count}")
        current = normalized
        count = 1
    key = "".join(current) if current else "none"
    segments.append(f"{key}-{count}")
    return ",".join(segments)


def _normalize_intrinsics_array(arr: Any, num_frames: int) -> np.ndarray:
    intrinsics = np.asarray(arr, dtype=np.float32)
    if intrinsics.shape == (4,):
        return np.broadcast_to(intrinsics, (num_frames, 4)).copy()
    if intrinsics.shape == (3, 3):
        vec = np.array(
            [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ],
            dtype=np.float32,
        )
        return np.broadcast_to(vec, (num_frames, 4)).copy()
    if intrinsics.ndim == 2 and intrinsics.shape[1] == 4:
        if intrinsics.shape[0] < num_frames:
            pad = np.broadcast_to(intrinsics[-1:], (num_frames - intrinsics.shape[0], 4))
            intrinsics = np.concatenate([intrinsics, pad], axis=0)
        return intrinsics[:num_frames].copy()
    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        if intrinsics.shape[0] < num_frames:
            pad = np.broadcast_to(
                intrinsics[-1:], (num_frames - intrinsics.shape[0], 3, 3)
            )
            intrinsics = np.concatenate([intrinsics, pad], axis=0)
        intrinsics = intrinsics[:num_frames]
        return np.stack(
            [
                intrinsics[:, 0, 0],
                intrinsics[:, 1, 1],
                intrinsics[:, 0, 2],
                intrinsics[:, 1, 2],
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(
        "intrinsics must have shape (4,), (3,3), (F,4), or (F,3,3), "
        f"got {intrinsics.shape}"
    )


def _motion_param(batch: Req, name: str, default: float) -> float:
    value = (batch.condition_inputs or {}).get(name)
    if value is None:
        value = batch.extra.get(f"sana_wm_{name}", default)
    return float(value)


def _vae_scaling_factor(vae: torch.nn.Module) -> float | torch.Tensor:
    config = getattr(vae, "config", None)
    arch = getattr(config, "arch_config", None)
    value = getattr(arch, "scaling_factor", None)
    if value is None:
        value = getattr(config, "scaling_factor", None)
    is_zero = False
    if isinstance(value, torch.Tensor):
        is_zero = bool(value.numel() == 1 and value.item() == 0)
    elif value is not None:
        is_zero = value == 0
    if value is None or is_zero:
        value = getattr(vae, "scaling_factor", 1.0)
    return value


def _vae_stats(
    vae: torch.nn.Module,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(tensor.shape[1])
    mean = getattr(vae, "latents_mean", None)
    std = getattr(vae, "latents_std", None)
    if mean is None:
        mean = torch.zeros(channels, device=tensor.device, dtype=tensor.dtype)
    if std is None:
        std = torch.ones(channels, device=tensor.device, dtype=tensor.dtype)
    mean = mean.view(1, -1, 1, 1, 1).to(tensor.device, tensor.dtype)
    std = std.view(1, -1, 1, 1, 1).to(tensor.device, tensor.dtype)
    return mean, std


class SanaWMStreamingState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.prompt: str = ""
        self.image: Image.Image | None = None
        self.intrinsics_image: Image.Image | None = None
        self.src_size: tuple[int, int] | None = None
        self.resized_size: tuple[int, int] | None = None
        self.crop_offset: tuple[int, int] | None = None
        self.intrinsics_raw: np.ndarray | None = None
        self.camera_actions: list[list[str]] = []
        self.max_camera_actions = 0
        self.static_c2w: np.ndarray | None = None
        self.latents: torch.Tensor | None = None
        self.latents_full: torch.Tensor | None = None
        self.refined_full: torch.Tensor | None = None
        self.rollover_first_latent: torch.Tensor | None = None
        self.sampler: SelfForcingFlowEulerSampler | None = None
        self.stage1_iter = None
        self.stage1_chunks = 0
        self.stage1_idx = 0
        self.produced_until = 0
        self.tick = 0
        self.use_refiner = False
        self.refiner: DiffusersLTX2Refiner | None = None
        self.refiner_runner = None
        self.sink_size = 1
        self.refiner_block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES
        self.n_blocks = 0
        self.next_ref_idx = 0
        self.next_dec_idx = 0
        self.decoder_first_chunk = True
        self.reset_decoder_cache = None
        self.latent_t = 0
        self.translation_speed = 0.04
        self.rotation_speed_deg = 1.2

    def dispose(self):
        reset_decoder_cache = self.reset_decoder_cache
        self.reset_decoder_cache = None
        if callable(reset_decoder_cache):
            reset_decoder_cache()
        self.initialized = False
        self.prompt = ""
        self.image = None
        self.intrinsics_image = None
        self.src_size = None
        self.resized_size = None
        self.crop_offset = None
        self.intrinsics_raw = None
        self.camera_actions = []
        self.max_camera_actions = 0
        self.static_c2w = None
        self.sampler = None
        self.stage1_iter = None
        self.stage1_chunks = 0
        self.stage1_idx = 0
        self.produced_until = 0
        self.tick = 0
        self.use_refiner = False
        self.refiner = None
        self.refiner_runner = None
        self.sink_size = 1
        self.refiner_block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES
        self.n_blocks = 0
        self.next_ref_idx = 0
        self.next_dec_idx = 0
        self.decoder_first_chunk = True
        self.latent_t = 0
        self.translation_speed = 0.04
        self.rotation_speed_deg = 1.2
        self.latents = None
        self.latents_full = None
        self.refined_full = None
        self.rollover_first_latent = None


class SanaWMRealtimeStage(PipelineStage):
    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        model_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.model_path = model_path
        self.refiner: DiffusersLTX2Refiner | None = None
        self.first_frame_latent_cache = None

    @property
    def role_affinity(self):
        return RoleType.MONOLITHIC

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "transformer",
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
                memory_intensive=True,
                keep_ready_after_warmup=True,
            ),
            ComponentUse(
                stage_name,
                "vae",
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision],
                keep_ready_after_warmup=True,
            ),
        ]

    def _empty_output(self, batch: Req) -> OutputBatch:
        output = torch.empty(
            (1, 3, 0, int(batch.height or SANA_WM_HEIGHT), int(batch.width or SANA_WM_WIDTH)),
            dtype=torch.float32,
            device=get_local_torch_device(),
        )
        return OutputBatch(output=output, metrics=batch.metrics)

    def _prepare_image(
        self, batch: Req
    ) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
        if isinstance(batch.condition_image, Image.Image):
            original = batch.condition_image.convert("RGB")
        elif batch.image_path is not None and isinstance(batch.image_path, str):
            original = Image.open(batch.image_path).convert("RGB")
        else:
            raise ValueError("SANA-WM realtime requires a first-frame image")
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(
            original, SANA_WM_HEIGHT, SANA_WM_WIDTH
        )
        return cropped, original, src_size, resized_size, crop_offset

    def _prepare_static_camera(
        self,
        batch: Req,
        *,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
    ) -> np.ndarray | None:
        condition_inputs = batch.condition_inputs or {}
        camera_path = condition_inputs.get("camera_path")
        if camera_path is not None:
            c2w = load_camera(Path(str(camera_path)))
        elif condition_inputs.get("camera") is not None:
            c2w = np.asarray(condition_inputs["camera"], dtype=np.float32)
        elif condition_inputs.get("action") is not None:
            c2w = action_string_to_c2w(
                str(condition_inputs["action"]),
                translation_speed=translation_speed,
                rotation_speed_deg=rotation_speed_deg,
            )
        else:
            return None

        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError(f"camera trajectory must have shape (F,4,4), got {c2w.shape}")
        if c2w.shape[0] < num_frames:
            pad = np.broadcast_to(c2w[-1:], (num_frames - c2w.shape[0], 4, 4))
            c2w = np.concatenate([c2w, pad], axis=0)
        return c2w[:num_frames].astype(np.float32)

    def _prepare_intrinsics(
        self,
        batch: Req,
        state: SanaWMStreamingState,
        *,
        num_frames: int,
        device: torch.device,
    ) -> np.ndarray:
        condition_inputs = batch.condition_inputs or {}
        if condition_inputs.get("intrinsics_path") is not None:
            return load_intrinsics(Path(str(condition_inputs["intrinsics_path"])), num_frames)
        if condition_inputs.get("intrinsics") is not None:
            return _normalize_intrinsics_array(condition_inputs["intrinsics"], num_frames)
        if state.intrinsics_raw is not None:
            return state.intrinsics_raw
        if state.intrinsics_image is None:
            raise ValueError("SANA-WM image is not initialized")
        estimated = estimate_intrinsics_with_pi3x(state.intrinsics_image, device)
        return np.broadcast_to(estimated, (num_frames, 4)).copy()

    def _append_realtime_camera_actions(
        self, batch: Req, state: SanaWMStreamingState
    ) -> None:
        actions = _normalize_camera_actions(
            (batch.condition_inputs or {}).get("camera_actions")
        )
        if actions:
            state.camera_actions.extend(actions)
            if (
                state.max_camera_actions > 0
                and len(state.camera_actions) > state.max_camera_actions
            ):
                del state.camera_actions[state.max_camera_actions :]

    def _camera_from_state(
        self,
        state: SanaWMStreamingState,
        *,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
    ) -> np.ndarray:
        if state.static_c2w is not None:
            return state.static_c2w[:num_frames]

        num_actions = max(0, num_frames - 1)
        actions = list(state.camera_actions[:num_actions])
        if len(actions) < num_actions:
            actions.extend([[] for _ in range(num_actions - len(actions))])
        return action_string_to_c2w(
            _actions_to_action_string(actions),
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
        )[:num_frames]

    def _update_camera_tensors(
        self,
        batch: Req,
        state: SanaWMStreamingState,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if state.sampler is None:
            return
        if state.src_size is None or state.resized_size is None or state.crop_offset is None:
            raise ValueError("SANA-WM crop metadata is not initialized")

        num_frames = int(batch.num_frames)
        condition_inputs = batch.condition_inputs or {}
        if "translation_speed" in condition_inputs:
            state.translation_speed = float(condition_inputs["translation_speed"])
        if "rotation_speed_deg" in condition_inputs:
            state.rotation_speed_deg = float(condition_inputs["rotation_speed_deg"])
        c2w = self._camera_from_state(
            state,
            num_frames=num_frames,
            translation_speed=state.translation_speed,
            rotation_speed_deg=state.rotation_speed_deg,
        )
        intrinsics_raw = self._prepare_intrinsics(
            batch,
            state,
            num_frames=num_frames,
            device=device,
        )
        state.intrinsics_raw = intrinsics_raw
        intrinsics = transform_intrinsics_for_crop(
            intrinsics_raw,
            state.src_size,
            state.resized_size,
            state.crop_offset,
        )
        camera = prepare_camera_conditions(c2w, intrinsics)
        raymap = camera["raymap"].unsqueeze(0).to(device=device, dtype=dtype)
        chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(
            device=device, dtype=dtype
        )
        state.sampler.extra_model_kwargs["camera_conditions"] = raymap
        state.sampler.extra_model_kwargs["chunk_plucker"] = chunk_plucker

    @torch.inference_mode()
    def _encode_first_frame(
        self,
        image: Image.Image,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> torch.Tensor:
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        image_tensor = pil_to_model_tensor(image, device=device, dtype=vae_dtype)
        with _deterministic_vae_encode_context():
            posterior = self.vae.encode(image_tensor.to(device=device, dtype=vae_dtype)).latent_dist
            z = posterior.mode()
        mean, std = _vae_stats(self.vae, z)
        scaling_factor = _vae_scaling_factor(self.vae)
        z = (z - mean) * scaling_factor / std
        return z.to(device=device, dtype=latent_dtype)

    def _first_frame_cache_key(
        self,
        batch: Req,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> tuple | None:
        if batch.image_path is None or not isinstance(batch.image_path, str):
            return None
        stat = os.stat(batch.image_path)
        return (
            batch.image_path,
            stat.st_mtime_ns,
            stat.st_size,
            device.type,
            device.index,
            vae_dtype,
            latent_dtype,
        )

    @torch.inference_mode()
    def _get_first_frame_latent(
        self,
        batch: Req,
        image: Image.Image,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = self._first_frame_cache_key(
            batch,
            device=device,
            vae_dtype=vae_dtype,
            latent_dtype=latent_dtype,
        )
        if (
            cache_key is not None
            and self.first_frame_latent_cache is not None
            and self.first_frame_latent_cache[0] == cache_key
        ):
            return self.first_frame_latent_cache[1]

        first_latent = self._encode_first_frame(
            image,
            device=device,
            vae_dtype=vae_dtype,
            latent_dtype=latent_dtype,
        )
        if cache_key is not None:
            self.first_frame_latent_cache = (cache_key, first_latent.detach())
        return first_latent

    def _ensure_streaming_decoder(self, state: SanaWMStreamingState) -> None:
        if not hasattr(self.vae, "decode_per_frame_with_cache") or not hasattr(
            self.vae, "clear_decoder_cache"
        ):
            enable_ltx2_streaming_cache(self.vae)
        if hasattr(self.vae, "use_framewise_encoding"):
            self.vae.use_framewise_encoding = True
            self.vae.use_framewise_decoding = True
            self.vae.tile_sample_stride_num_frames = 64
            self.vae.tile_sample_min_num_frames = 96
        state.reset_decoder_cache = getattr(self.vae, "clear_decoder_cache", None)

    @torch.inference_mode()
    def _decode_chunk(
        self,
        latents: torch.Tensor,
        state: SanaWMStreamingState,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        self._ensure_streaming_decoder(state)
        mean, std = _vae_stats(self.vae, latents)
        scaling_factor = _vae_scaling_factor(self.vae)
        z = latents * std / scaling_factor + mean
        chunks = list(
            self.vae.decode_per_frame_with_cache(
                z.to(vae_dtype),
                temb=None,
                causal=True,
                reset_cache=state.decoder_first_chunk,
            )
        )
        state.decoder_first_chunk = False
        if not chunks:
            return torch.empty(
                (latents.shape[0], 3, 0, SANA_WM_HEIGHT, SANA_WM_WIDTH),
                dtype=torch.float32,
                device=latents.device,
            )
        decoded = torch.cat(chunks, dim=2)
        return (decoded / 2 + 0.5).clamp(0, 1)

    def _refiner_paths(self) -> tuple[str, str] | None:
        refiner_root = os.path.join(self.model_path, "refiner_diffusers")
        gemma_root = os.path.join(self.model_path, "gemma3_12b")
        if os.path.isdir(refiner_root) and os.path.isdir(gemma_root):
            return refiner_root, gemma_root
        return None

    def _get_refiner(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DiffusersLTX2Refiner:
        refiner_paths = self._refiner_paths()
        if refiner_paths is None:
            raise RuntimeError(
                "SANA-WM realtime requires refiner_diffusers and gemma3_12b"
            )
        if (
            self.refiner is not None
            and self.refiner.device == device
            and self.refiner.dtype == dtype
        ):
            return self.refiner

        refiner_root, gemma_root = refiner_paths
        self.refiner = DiffusersLTX2Refiner(
            refiner_root,
            gemma_root,
            dtype=dtype,
            device=device,
        )
        return self.refiner

    def _initialize_state(
        self,
        batch: Req,
        state: SanaWMStreamingState,
        *,
        transformer: torch.nn.Module,
        device: torch.device,
        weight_dtype: torch.dtype,
        vae_dtype: torch.dtype,
        first_latent: torch.Tensor | None = None,
    ) -> None:
        image, intrinsics_image, src_size, resized_size, crop_offset = self._prepare_image(batch)
        state.image = image
        state.intrinsics_image = intrinsics_image
        state.src_size = src_size
        state.resized_size = resized_size
        state.crop_offset = crop_offset
        state.prompt = str(batch.prompt)

        num_frames = snap_num_frames(int(batch.num_frames), stride=8)
        batch.num_frames = num_frames
        state.max_camera_actions = max(0, num_frames - 1)
        self._append_realtime_camera_actions(batch, state)
        if first_latent is None:
            first_latent = self._get_first_frame_latent(
                batch,
                image,
                device=device,
                vae_dtype=vae_dtype,
                latent_dtype=weight_dtype,
            )
        else:
            first_latent = first_latent.to(device=device, dtype=weight_dtype)
        latent_t = (num_frames - 1) // 8 + 1
        generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(int(batch.seed))
        latents = torch.randn(
            1,
            first_latent.shape[1],
            latent_t,
            first_latent.shape[-2],
            first_latent.shape[-1],
            device=device,
            dtype=weight_dtype,
            generator=generator,
        )
        latents[:, :, :1] = first_latent

        cond = batch.prompt_embeds[0].to(device=device, dtype=weight_dtype)
        neg = (
            batch.negative_prompt_embeds[0].to(device=device, dtype=weight_dtype)
            if batch.do_classifier_free_guidance
            and batch.negative_prompt_embeds
            else torch.zeros_like(cond)
        )
        cond_mask = (
            batch.prompt_attention_mask[0]
            if isinstance(batch.prompt_attention_mask, list)
            else batch.prompt_attention_mask
        )
        if cond_mask is None:
            cond_mask = torch.ones(cond.shape[0], cond.shape[2], device=device)
        cond_mask = cond_mask.to(device=device)
        if batch.do_classifier_free_guidance and batch.negative_attention_mask:
            neg_mask = batch.negative_attention_mask[0].to(device=device)
            mask = torch.cat([neg_mask, cond_mask], dim=0)
        else:
            mask = cond_mask

        state.translation_speed = _motion_param(batch, "translation_speed", 0.04)
        state.rotation_speed_deg = _motion_param(batch, "rotation_speed_deg", 1.2)
        state.static_c2w = self._prepare_static_camera(
            batch,
            num_frames=num_frames,
            translation_speed=state.translation_speed,
            rotation_speed_deg=state.rotation_speed_deg,
        )

        sampler = SelfForcingFlowEulerSampler(
            transformer,
            cond,
            neg,
            SelfForcingSamplerConfig(
                steps=int(batch.num_inference_steps),
                cfg_scale=float(batch.guidance_scale),
                flow_shift=float(batch.extra.get("flow_shift", 8.0)),
                denoising_step_list=_as_int_tuple(
                    batch.extra.get("sana_wm_denoising_step_list"),
                    (1000, 960, 889, 727, 0),
                ),
                num_frame_per_block=int(
                    batch.extra.get("sana_wm_num_frame_per_block", 3)
                ),
                num_cached_blocks=int(batch.extra.get("sana_wm_num_cached_blocks", 2)),
                sink_token=int(batch.extra.get("sana_wm_sink_size", 1)) > 0,
            ),
            {
                "data_info": {"condition_frame_info": {0: 0.0}},
                "mask": mask,
            },
        )
        state.sampler = sampler
        self._update_camera_tensors(batch, state, device=device, dtype=weight_dtype)

        state.latents = latents
        state.latent_t = latent_t
        state.stage1_chunks = len(sampler.create_autoregressive_segments(latent_t)) - 1
        state.stage1_iter = sampler.sample_chunks(latents, generator=generator)
        state.sink_size = int(batch.extra.get("sana_wm_sink_size", 1))
        state.refiner_block_size = DEFAULT_REFINER_BLOCK_SIZE
        state.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES
        active_frames = latent_t - state.sink_size
        if active_frames <= 0:
            raise ValueError(
                f"SANA-WM active latent frames must be positive, got {active_frames}"
            )
        state.n_blocks = math.ceil(active_frames / state.refiner_block_size)
        state.latents_full = torch.empty_like(latents)
        state.latents_full[:, :, : state.sink_size] = latents[:, :, : state.sink_size]
        state.refined_full = torch.empty_like(latents)
        state.refined_full[:, :, : state.sink_size] = latents[:, :, : state.sink_size]
        state.rollover_first_latent = first_latent.detach().clone()

        refiner = self._get_refiner(
            dtype=weight_dtype,
            device=device,
        )
        state.refiner_runner = refiner.build_chunk_runner(
            state.prompt,
            fps=float(batch.fps),
            source_sink_frames=state.sink_size,
            block_size=state.refiner_block_size,
            kv_max_frames=state.refiner_kv_max_frames,
            seed=int(batch.extra.get("sana_wm_refiner_seed", batch.seed)),
            spatial_shape=(int(latents.shape[3]), int(latents.shape[4])),
        )
        state.use_refiner = True
        logger.info(
            "SANA-WM realtime uses Stage-1 plus LTX-2 refiner streaming."
        )

        self._ensure_streaming_decoder(state)
        state.initialized = True
        logger.info(
            "SANA-WM realtime initialized: latent_t=%s stage1_chunks=%s "
            "refiner=%s decode_blocks=%s",
            latent_t,
            state.stage1_chunks,
            state.use_refiner,
            state.n_blocks,
        )

    def _advance_stage1(
        self,
        state: SanaWMStreamingState,
    ) -> tuple[torch.Tensor, int, int] | None:
        if state.stage1_iter is None or state.stage1_idx >= state.stage1_chunks:
            return None
        _, latent_view, start_f, end_f = next(state.stage1_iter)
        if state.latents_full is not None:
            state.latents_full[:, :, start_f:end_f].copy_(latent_view)
        state.produced_until = max(state.produced_until, int(end_f))
        state.stage1_idx += 1
        return latent_view, int(start_f), int(end_f)

    def _store_rollover_first_latent(
        self,
        state: SanaWMStreamingState,
        latents: torch.Tensor,
        frame_idx: int,
    ) -> None:
        state.rollover_first_latent = (
            latents[:, :, frame_idx : frame_idx + 1].detach().clone()
        )

    def _run_stage1_only_tick(
        self,
        state: SanaWMStreamingState,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        advanced = self._advance_stage1(state)
        state.tick += 1
        if advanced is None:
            return None
        _, start_f, end_f = advanced
        if state.latents is None:
            return None
        z_slice = state.latents[:, :, :end_f] if start_f == 0 else state.latents[:, :, start_f:end_f]
        frames = self._decode_chunk(z_slice, state, vae_dtype=vae_dtype)
        self._store_rollover_first_latent(state, state.latents, end_f - 1)
        if start_f == 0 and state.sink_size > 0:
            frames = frames[:, :, 1:]
        return frames

    def _run_refiner_tick(
        self,
        state: SanaWMStreamingState,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if state.latents_full is None or state.refined_full is None:
            return None

        self._advance_stage1(state)

        if state.next_ref_idx < state.n_blocks:
            block_start = state.sink_size + state.next_ref_idx * state.refiner_block_size
            block_end = min(block_start + state.refiner_block_size, state.latent_t)
            if block_end <= state.produced_until:
                if state.refiner_runner is None:
                    raise RuntimeError("SANA-WM refiner runner is not initialized")
                refined = state.refiner_runner.refine_block(
                    block_idx=state.next_ref_idx,
                    clean_block=state.latents_full[:, :, block_start:block_end],
                    block_start=block_start,
                    block_end=block_end,
                    sink_seed_frames=(
                        state.latents_full[:, :, : state.sink_size]
                        if state.next_ref_idx == 0
                        else None
                    ),
                )
                state.refined_full[:, :, block_start:block_end].copy_(refined)
                state.next_ref_idx += 1
            elif state.stage1_idx >= state.stage1_chunks:
                raise RuntimeError(
                    f"SANA-WM Stage-1 ended at latent frame {state.produced_until}, "
                    f"but refiner block {state.next_ref_idx} needs {block_end}"
                )

        frames = None
        if state.next_dec_idx < state.next_ref_idx:
            block_start = state.sink_size + state.next_dec_idx * state.refiner_block_size
            block_end = min(block_start + state.refiner_block_size, state.latent_t)
            z_slice = (
                state.refined_full[:, :, :block_end]
                if state.next_dec_idx == 0
                else state.refined_full[:, :, block_start:block_end]
            )
            frames = self._decode_chunk(z_slice, state, vae_dtype=vae_dtype)
            self._store_rollover_first_latent(
                state,
                state.refined_full,
                block_end - 1,
            )
            if state.next_dec_idx == 0 and state.sink_size > 0:
                frames = frames[:, :, 1:]
            state.next_dec_idx += 1

        state.tick += 1
        return frames

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if batch.session is None:
            raise ValueError("SANA-WM realtime pipeline requires a realtime session")

        state = batch.session.get_or_create_state(SanaWMStreamingState)
        assert isinstance(state, SanaWMStreamingState)

        device = get_local_torch_device()
        weight_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        transformer = self.transformer.to(device=device, dtype=weight_dtype).eval()
        self.vae = self.vae.to(device=device, dtype=vae_dtype).eval()

        if batch.block_idx == 0 or not state.initialized:
            state.dispose()
            self._initialize_state(
                batch,
                state,
                transformer=transformer,
                device=device,
                weight_dtype=weight_dtype,
                vae_dtype=vae_dtype,
            )
        else:
            self._append_realtime_camera_actions(batch, state)
            self._update_camera_tensors(
                batch,
                state,
                device=device,
                dtype=weight_dtype,
            )

        if state.use_refiner:
            frames = self._run_refiner_tick(state, vae_dtype=vae_dtype)
        else:
            frames = self._run_stage1_only_tick(state, vae_dtype=vae_dtype)

        if frames is None and state.rollover_first_latent is not None:
            first_latent = state.rollover_first_latent
            logger.info(
                "SANA-WM realtime horizon rollover: block_idx=%s, latent_t=%s",
                batch.block_idx,
                state.latent_t,
            )
            state.dispose()
            self._initialize_state(
                batch,
                state,
                transformer=transformer,
                device=device,
                weight_dtype=weight_dtype,
                vae_dtype=vae_dtype,
                first_latent=first_latent,
            )
            if state.use_refiner:
                frames = self._run_refiner_tick(state, vae_dtype=vae_dtype)
            else:
                frames = self._run_stage1_only_tick(state, vae_dtype=vae_dtype)

        if frames is None:
            return self._empty_output(batch)
        return OutputBatch(output=frames.to(torch.float32), metrics=batch.metrics)
