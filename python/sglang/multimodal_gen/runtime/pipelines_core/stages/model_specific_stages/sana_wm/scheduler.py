#!/usr/bin/env python3
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Minimal SANA-WM sampler.

Only the public SANA-WM default is implemented: LTX flow-Euler with hard
conditioning on the first latent frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm


def _stream_debug_dump(name: str, tensor: torch.Tensor) -> None:
    root = os.getenv("SANAWM_STREAM_DEBUG_DIR")
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(root, f"{name}.pt"))


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 60
    cfg_scale: float = 5.0
    flow_shift: float = 9.8


@dataclass(frozen=True)
class SelfForcingSamplerConfig:
    steps: int = 4
    cfg_scale: float = 1.0
    flow_shift: float = 8.0
    denoising_step_list: tuple[int, ...] = (1000, 960, 889, 727, 0)
    num_frame_per_block: int = 3
    num_cached_blocks: int = 2
    sink_token: bool = True


class LTXFlowEulerSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        config: SamplerConfig,
        model_kwargs: dict[str, object],
    ) -> None:
        self.model = model
        self.condition = condition
        self.uncondition = uncondition
        self.config = config
        self.model_kwargs = model_kwargs
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=config.flow_shift)

    def sample(self, latents: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        timesteps, _ = retrieve_timesteps(self.scheduler, self.config.steps, self.condition.device, None)
        use_cfg = self.config.cfg_scale > 1.0
        prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0) if use_cfg else self.condition

        condition_frame_info = dict(self.model_kwargs.get("data_info", {}).get("condition_frame_info", {}))
        condition_mask = torch.zeros_like(latents)
        image_noise_scale = 0.0
        for frame_idx, frame_weight in condition_frame_info.items():
            condition_mask[:, :, int(frame_idx)] = 1
            image_noise_scale = max(image_noise_scale, float(frame_weight))

        init_latents = latents.clone()
        iterator = enumerate(timesteps)
        iterator = tqdm(list(iterator), disable=os.getenv("DPM_TQDM", "False") == "True")
        for _, timestep_scalar in iterator:
            if image_noise_scale > 0:
                latents = self._noise_conditioned_frames(
                    timestep_scalar / 1000.0,
                    init_latents,
                    latents,
                    image_noise_scale,
                    condition_mask,
                    generator,
                )

            condition_mask_input = torch.cat([condition_mask] * 2) if use_cfg else condition_mask
            latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
            timestep = timestep_scalar.expand(condition_mask_input.shape).float()
            timestep = torch.min(timestep, (1 - condition_mask_input) * 1000.0)

            noise_pred = self.model(
                latent_model_input,
                timestep[:, :1, :, 0, 0],
                prompt_embeds,
                **self.model_kwargs,
            )
            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]
            if use_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.config.cfg_scale * (noise_text - noise_uncond)
                timestep = timestep.chunk(2)[0]

            latents = self._step(latents, noise_pred, timestep_scalar, timestep, condition_mask)
        return latents

    def _step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep_scalar: torch.Tensor,
        timestep: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = latents.dtype
        b, c, f, h, w = latents.shape
        denoised = self.scheduler.step(
            -noise_pred.reshape(b, c, -1).transpose(1, 2),
            timestep_scalar,
            latents.reshape(b, c, -1).transpose(1, 2),
            per_token_timesteps=timestep.reshape(b, c, -1)[:, 0],
            return_dict=False,
        )[0]
        denoised = denoised.transpose(1, 2).reshape(latents.shape)
        denoise_mask = timestep_scalar / 1000 - 1e-6 < (1.0 - condition_mask)
        latents = torch.where(denoise_mask, denoised, latents)
        return latents.to(dtype) if latents.dtype != dtype else latents

    @staticmethod
    def _noise_conditioned_frames(
        timestep: torch.Tensor,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        condition_mask: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised = init_latents + noise_scale * noise * (timestep**2)
        return torch.where(condition_mask > 1.0 - 1e-6, noised, latents)


def _inject_sliced_extras(extra: dict[str, object], kwargs: dict[str, object], num_chunk_frames: int, end_f: int) -> None:
    begin_f = end_f - num_chunk_frames
    for key, value in extra.items():
        if key in kwargs:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim == 5 and value.shape[2] > num_chunk_frames:
                kwargs[key] = value[:, :, begin_f:end_f]
            elif value.ndim >= 3 and value.shape[1] > num_chunk_frames:
                kwargs[key] = value[:, begin_f:end_f]
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value


def _pop_extra_model_kwargs(model_kwargs: dict[str, object]) -> dict[str, object]:
    extra: dict[str, object] = {}
    for key in list(model_kwargs):
        if key not in ("mask", "data_info"):
            extra[key] = model_kwargs.pop(key)
    return extra


class SelfForcingFlowEulerSampler:
    _num_cache_slots = 10
    _slot_k = 0
    _slot_v = 1
    _slot_cam_k = 2
    _slot_cam_v = 3
    _slot_shortconv = 4
    _slot_type_flag = 6
    _slot_tconv = 9

    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        config: SelfForcingSamplerConfig,
        model_kwargs: dict[str, object],
    ) -> None:
        self.model = model
        self.condition = condition
        self.uncondition = uncondition
        self.config = config
        self.model_kwargs = dict(model_kwargs)
        self.extra_model_kwargs = _pop_extra_model_kwargs(self.model_kwargs)
        self.mask = self.model_kwargs.pop("mask", None)
        self.num_model_blocks = len(getattr(model, "blocks"))
        self._chunk_indices: list[int] = []

    def create_autoregressive_segments(self, total_frames: int) -> list[int]:
        base = int(self.config.num_frame_per_block)
        remained = total_frames % base
        num_chunks = total_frames // base
        chunk_indices = [0]
        for idx in range(num_chunks):
            cur = chunk_indices[-1] + base
            if idx == 0:
                cur += remained
            chunk_indices.append(cur)
        return chunk_indices

    def sample(self, latents: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        for _ in self.sample_chunks(latents, generator=generator):
            pass
        return latents

    def sample_chunks(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> Iterator[tuple[int, torch.Tensor, int, int]]:
        del generator
        device = self.condition.device
        schedule = list(self.config.denoising_step_list)
        if len(schedule) < 2 or schedule[-1] != 0:
            raise ValueError(f"denoising_step_list must end with 0, got {schedule}")
        explicit_sigmas = [float(t) / 1000.0 for t in schedule[:-1]]
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        use_cfg = self.config.cfg_scale > 1.0
        batch_size, channels, total_frames, height, width = latents.shape
        if total_frames <= self.config.num_frame_per_block:
            raise ValueError("streaming self-forcing requires more than one latent chunk")

        chunk_indices = self.create_autoregressive_segments(total_frames)
        self._chunk_indices = chunk_indices
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        data_info = dict(self.model_kwargs.get("data_info", {}))
        condition_frame_info = dict(data_info.pop("condition_frame_info", {}))
        init_latents = latents.clone()

        for chunk_idx in range(num_chunks):
            chunk_kv_cache, _, sink_num, _ = self._accumulate_kv_cache(kv_cache, chunk_idx)
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            chunk_frames = end_f - start_f
            prompt_embeds = self.condition[chunk_idx : chunk_idx + 1] if self.condition.shape[0] == num_chunks else self.condition
            if use_cfg:
                prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)
            mask = self.mask
            if isinstance(mask, torch.Tensor) and mask.shape[0] == num_chunks:
                mask = mask[chunk_idx : chunk_idx + 1]

            frame_index = torch.arange(start_f, end_f, device=device, dtype=torch.long) if sink_num > 0 else None
            local_data_info = dict(data_info)

            condition_mask_chunk = torch.zeros(
                batch_size,
                channels,
                chunk_frames,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
            cond_local_indices: list[int] = []
            for frame_idx in condition_frame_info:
                frame_idx = int(frame_idx)
                if start_f <= frame_idx < end_f:
                    loc = frame_idx - start_f
                    cond_local_indices.append(loc)
                    condition_mask_chunk[:, :, loc] = 1.0

            scheduler.set_timesteps(sigmas=explicit_sigmas, device=device)
            iterator = tqdm(list(scheduler.timesteps), disable=os.getenv("DPM_TQDM", "False") == "True", desc=f"stage1 chunk {chunk_idx}")
            for step_idx, timestep_scalar in enumerate(iterator):
                latent_model_input = torch.cat([latents[:, :, start_f:end_f]] * 2) if use_cfg else latents[:, :, start_f:end_f]
                timestep_tensor = (1.0 - condition_mask_chunk) * timestep_scalar.to(device=device, dtype=torch.float32).view(1, 1, 1, 1, 1)
                timestep_model = torch.cat([timestep_tensor, timestep_tensor], dim=0) if use_cfg else timestep_tensor
                if chunk_idx == 0 and step_idx == 0:
                    _stream_debug_dump("stage1_input_000_000", latent_model_input)
                    _stream_debug_dump("stage1_timestep_000_000", timestep_model)
                    _stream_debug_dump("stage1_condition_mask_000", condition_mask_chunk)
                call_kwargs = {"mask": mask, "data_info": local_data_info}
                _inject_sliced_extras(self.extra_model_kwargs, call_kwargs, chunk_frames, end_f)
                noise_pred, _ = self.model(
                    latent_model_input,
                    timestep_model[:, :1, :, 0, 0],
                    prompt_embeds,
                    start_f=start_f,
                    end_f=end_f,
                    frame_index=frame_index,
                    save_kv_cache=False,
                    kv_cache=chunk_kv_cache,
                    **call_kwargs,
                )
                if isinstance(noise_pred, Transformer2DModelOutput):
                    noise_pred = noise_pred[0]
                if use_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.config.cfg_scale * (noise_text - noise_uncond)
                if chunk_idx == 0 and step_idx == 0:
                    _stream_debug_dump("stage1_noise_pred_000_000", noise_pred)

                chunk_latents = latents[:, :, start_f:end_f]
                denoised = scheduler.step(
                    -noise_pred.reshape(batch_size, channels, -1).transpose(1, 2),
                    timestep_scalar,
                    chunk_latents.reshape(batch_size, channels, -1).transpose(1, 2),
                    per_token_timesteps=timestep_tensor.reshape(batch_size, channels, -1)[:, 0],
                    return_dict=False,
                )[0]
                latents[:, :, start_f:end_f] = denoised.transpose(1, 2).reshape(chunk_latents.shape).to(latents.dtype)
                for loc in cond_local_indices:
                    latents[:, :, start_f + loc] = init_latents[:, :, start_f + loc]

            latent_model_input = torch.cat([latents[:, :, start_f:end_f]] * 2) if use_cfg else latents[:, :, start_f:end_f]
            timestep_zero = torch.zeros(latent_model_input.shape[0], 1, chunk_frames, device=device, dtype=torch.float32)
            call_kwargs = {"mask": mask, "data_info": local_data_info}
            _inject_sliced_extras(self.extra_model_kwargs, call_kwargs, chunk_frames, end_f)
            _, updated_kv_cache = self.model(
                latent_model_input,
                timestep_zero,
                prompt_embeds,
                start_f=start_f,
                end_f=end_f,
                frame_index=frame_index,
                save_kv_cache=True,
                kv_cache=chunk_kv_cache,
                **call_kwargs,
            )
            kv_cache[chunk_idx] = updated_kv_cache
            yield chunk_idx, latents[:, :, start_f:end_f], start_f, end_f

    def _initialize_kv_cache(self, num_chunks: int) -> list[list[list[torch.Tensor | None]]]:
        return [[[None] * self._num_cache_slots for _ in range(self.num_model_blocks)] for _ in range(num_chunks)]

    def _accumulate_kv_cache(self, kv_cache: list, chunk_idx: int) -> tuple[list, int, int, int]:
        if chunk_idx == 0:
            return kv_cache[0], 0, 0, 0
        cur_kv_cache = kv_cache[chunk_idx]
        start_chunk_idx = max(chunk_idx - self.config.num_cached_blocks, 0) if self.config.num_cached_blocks > 0 else 0
        valid_cached_chunks = list(range(start_chunk_idx, chunk_idx))
        sink_num = 0
        if self.config.sink_token and self.config.num_cached_blocks > 0:
            sink_start = max(chunk_idx - self.config.num_cached_blocks + 1, 0)
            if sink_start > 0:
                valid_cached_chunks = [0] + list(range(sink_start, chunk_idx))
                sink_num = self._chunk_indices[1] - self._chunk_indices[0]
        num_cached_frames = sum(self._chunk_indices[idx + 1] - self._chunk_indices[idx] for idx in valid_cached_chunks)

        for block_id in range(self.num_model_blocks):
            prev_last = kv_cache[chunk_idx - 1][block_id]
            type_flag = prev_last[self._slot_type_flag] if prev_last[self._slot_type_flag] is not None else None
            if type_flag is not None and type_flag.item() > 0.5:
                cur_kv_cache[block_id] = [
                    prev_last[0],
                    prev_last[1],
                    prev_last[2],
                    prev_last[3],
                    prev_last[self._slot_shortconv],
                    None,
                    prev_last[self._slot_type_flag],
                    None,
                    None,
                    prev_last[self._slot_tconv],
                ]
                continue

            acc: list[torch.Tensor | None] = [None] * self._num_cache_slots
            for idx in valid_cached_chunks:
                prev = kv_cache[idx][block_id]
                if prev[0] is None:
                    continue
                for slot in (self._slot_k, self._slot_v, self._slot_cam_k, self._slot_cam_v):
                    if prev[slot] is None:
                        continue
                    acc[slot] = prev[slot].clone() if acc[slot] is None else torch.cat([acc[slot], prev[slot]], dim=2)
            cur_kv_cache[block_id] = [
                acc[0],
                acc[1],
                acc[2],
                acc[3],
                prev_last[self._slot_shortconv],
                None,
                prev_last[self._slot_type_flag],
                None,
                None,
                prev_last[self._slot_tconv],
            ]
        self._evict_stale_kv_cache(kv_cache, chunk_idx, valid_cached_chunks)
        return cur_kv_cache, len(valid_cached_chunks), sink_num, num_cached_frames

    def _evict_stale_kv_cache(self, kv_cache: list, chunk_idx: int, valid_cached_chunks: list[int]) -> None:
        if self.config.num_cached_blocks <= 0:
            return
        keep = set(valid_cached_chunks)
        keep.add(chunk_idx)
        for stale_idx in range(chunk_idx):
            if stale_idx in keep:
                continue
            kv_cache[stale_idx] = [[None] * self._num_cache_slots for _ in range(self.num_model_blocks)]
