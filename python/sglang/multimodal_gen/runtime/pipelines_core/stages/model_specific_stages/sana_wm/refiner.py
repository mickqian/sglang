#!/usr/bin/env python3
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""SANA-WM LTX-2 refiner adapter.

The refiner is the SANA-WM Stage-2 path from PR #379, reduced to the pieces
needed for inference: load diffusers LTX-2 transformer/connectors, encode the
prompt with Gemma3, and apply the sink/current streaming self-attention mask.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch import nn

STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class DiffusersLTX2Refiner(nn.Module):
    def __init__(
        self,
        refiner_root: str | Path,
        gemma_root: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.refiner_root = Path(refiner_root)
        self.gemma_root = Path(gemma_root)
        self.dtype = dtype
        self.device = torch.device(device)
        self.text_max_sequence_length = int(text_max_sequence_length)
        self.transformer, self.connectors = self._load_diffusers_components()

    def _load_diffusers_components(self) -> tuple[nn.Module, nn.Module]:
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            self.refiner_root,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).eval()
        connectors = LTX2TextConnectors.from_pretrained(
            self.refiner_root,
            subfolder="connectors",
            torch_dtype=self.dtype,
        ).eval()
        return transformer, connectors

    @torch.inference_mode()
    def refine_latents(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
    ) -> torch.Tensor:
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}")

        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt)

        self.transformer.to(self.device)
        z = sana_latent.to(device=self.device, dtype=self.dtype)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas[0])
        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        eps = torch.randn(current.shape, generator=generator, device=self.device, dtype=self.dtype)
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        iterator = range(len(sigmas) - 1)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="refiner", unit="step")

        for step_index in iterator:
            sigma = sigmas[step_index]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
            )
            noisy_tokens = pack_latents(
                noisy,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = noisy_tokens.float() + velocity * (sigmas[step_index + 1] - sigma).float()
            noisy = unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(self.gemma_root)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval()
        text_encoder.to(self.device)
        text_backbone = getattr(text_encoder, "model", text_encoder)
        outputs = text_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        prompt_embeds = pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=self.device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=self.dtype)

        del text_encoder, text_backbone, outputs, hidden_states
        _empty_cuda_cache()

        self.connectors.to(self.device)
        connector_embeds, _, connector_mask = self.connectors(prompt_embeds, attention_mask)
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        _empty_cuda_cache()
        return connector_embeds.to(device=self.device, dtype=self.dtype), connector_mask.to(device=self.device)

    def _predict_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch, _, num_frames, height, width = full_latent.shape
        latent_tokens = pack_latents(
            full_latent,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        n_context_tokens = pack_latents(
            sink,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        ).shape[1]

        raw_timestep = torch.zeros(batch, latent_tokens.shape[1], 1, dtype=torch.float32, device=self.device)
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(self.transformer.config.timestep_scale_multiplier)
        velocity = self._forward_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(self.dtype)

    @torch.inference_mode()
    def build_chunk_runner(
        self,
        prompt: str,
        *,
        fps: float,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        spatial_shape: tuple[int, int],
        sigmas: tuple[float, ...] = STAGE_2_DISTILLED_SIGMA_VALUES,
    ) -> "RefinerChunkRunner":
        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt)
        self.transformer.to(self.device)
        return RefinerChunkRunner(
            self,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=float(fps),
            sigmas=torch.tensor(sigmas, dtype=torch.float32, device=self.device),
            source_sink_frames=int(source_sink_frames),
            block_size=int(block_size),
            kv_max_frames=int(kv_max_frames),
            seed=int(seed),
            spatial_shape=spatial_shape,
        )

    def _predict_x0_active_block(
        self,
        *,
        active: torch.Tensor,
        active_positions: list[int],
        sigma_cur: float,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        kv_prefix_per_layer: list[dict[str, object]] | None,
    ) -> torch.Tensor:
        latent_tokens = pack_latents(
            active,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch, seq_len, _ = latent_tokens.shape
        timestep = torch.full(
            (batch, seq_len),
            float(sigma_cur) * float(self.transformer.config.timestep_scale_multiplier),
            dtype=torch.float32,
            device=self.device,
        )
        video_rotary_emb = build_rotary_emb_for_absolute_positions(
            transformer=self.transformer,
            batch_size=batch,
            frame_positions=active_positions,
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            device=self.device,
            fps=float(fps),
        )
        set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        try:
            velocity = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
            )
        finally:
            clear_kv_prefix_on_blocks(self.transformer)
        raw_sigma = torch.full((batch, seq_len, 1), float(sigma_cur), dtype=torch.float32, device=self.device)
        denoised = latent_tokens.float() - velocity.float() * raw_sigma
        return unpack_latents(
            denoised.to(self.dtype),
            num_frames=int(active.shape[2]),
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )

    def _capture_block_kv(
        self,
        *,
        clean_block: torch.Tensor,
        frame_positions: list[int],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        capture_mode: str,
        kv_prefix_per_layer: list[dict[str, object]] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        latent_tokens = pack_latents(
            clean_block,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch, seq_len, _ = latent_tokens.shape
        timestep = torch.zeros(batch, seq_len, dtype=torch.float32, device=self.device)
        video_rotary_emb = build_rotary_emb_for_absolute_positions(
            transformer=self.transformer,
            batch_size=batch,
            frame_positions=frame_positions,
            height=int(clean_block.shape[3]),
            width=int(clean_block.shape[4]),
            device=self.device,
            fps=float(fps),
        )
        set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        set_capture_flag_on_blocks(self.transformer, capture_mode, enable=True)
        try:
            _ = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
            )
        finally:
            set_capture_flag_on_blocks(self.transformer, capture_mode, enable=False)
            clear_kv_prefix_on_blocks(self.transformer)
        return collect_captured_kv_from_blocks(self.transformer, capture_mode)

    def _forward_video_only_with_rope(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        n_context_tokens: int,
    ) -> torch.Tensor:
        transformer = self.transformer
        batch = hidden_states.size(0)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(),
            batch_size=batch,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch, -1, embedded_timestep.size(-1))
        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch, -1, hidden_states.size(-1))
        for block in transformer.transformer_blocks:
            hidden_states = forward_video_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )
        scale_shift = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)

    def _forward_video_only(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        n_context_tokens: int,
    ) -> torch.Tensor:
        transformer = self.transformer
        batch = hidden_states.size(0)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        video_coords = transformer.rope.prepare_video_coords(
            batch, num_frames, height, width, hidden_states.device, fps=fps
        )
        video_rotary_emb = transformer.rope(video_coords, device=hidden_states.device)
        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(),
            batch_size=batch,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch, -1, embedded_timestep.size(-1))
        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch, -1, hidden_states.size(-1))

        for block in transformer.transformer_blocks:
            hidden_states = forward_video_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )

        scale_shift = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)


def forward_video_block(
    *,
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    encoder_attention_mask: torch.Tensor | None,
    n_context_tokens: int,
) -> torch.Tensor:
    batch = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    attn_hidden_states = streaming_self_attention(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
        n_context_tokens=n_context_tokens,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa
    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states
    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    return hidden_states + block.ff(norm_hidden_states) * gate_mlp


def streaming_self_attention(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    sequence_length = hidden_states.shape[1]
    has_streaming_hooks = (
        getattr(attn, "_kv_cache_capture", False)
        or getattr(attn, "_tf_capture_kv", False)
        or getattr(attn, "_tf_kv_prefix", None) is not None
    )
    if n_context_tokens >= sequence_length and not has_streaming_hooks:
        return attn(hidden_states=hidden_states, encoder_hidden_states=None, query_rotary_emb=query_rotary_emb)

    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

    gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    if getattr(attn, "_kv_cache_capture", False):
        attn._cached_kv_pre = (key.detach().clone(), value.detach().clone())

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")
    if getattr(attn, "_tf_capture_kv", False):
        attn._cached_kv_post = (key.detach().clone(), value.detach().clone())

    tf_prefix = getattr(attn, "_tf_kv_prefix", None)
    if isinstance(tf_prefix, dict) and tf_prefix.get("mode") == "rf_shifted_sink":
        prefix_k_parts: list[torch.Tensor] = []
        prefix_v_parts: list[torch.Tensor] = []
        sink_k_pre = tf_prefix.get("sink_k_pre")
        sink_v = tf_prefix.get("sink_v")
        if sink_k_pre is not None and sink_v is not None and sink_k_pre.shape[1] > 0:
            sink_pe = tf_prefix.get("sink_pe")
            if sink_pe is None:
                raise RuntimeError("rf_shifted_sink prefix requires sink_pe")
            if attn.rope_type == "interleaved":
                sink_k = apply_interleaved_rotary_emb(sink_k_pre.to(key.dtype), sink_pe)
            else:
                sink_k = apply_split_rotary_emb(sink_k_pre.to(key.dtype), sink_pe)
            prefix_k_parts.append(sink_k)
            prefix_v_parts.append(sink_v.to(value.dtype))
        history_k = tf_prefix.get("history_k")
        history_v = tf_prefix.get("history_v")
        if history_k is not None and history_v is not None and history_k.shape[1] > 0:
            prefix_k_parts.append(history_k.to(key.dtype))
            prefix_v_parts.append(history_v.to(value.dtype))
        if prefix_k_parts:
            key = torch.cat([*prefix_k_parts, key], dim=1)
            value = torch.cat([*prefix_v_parts, value], dim=1)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)
    if n_context_tokens <= 0 or n_context_tokens >= query.shape[1]:
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=backend,
            parallel_config=parallel_config,
        )
    else:
        context = dispatch_attention_fn(
            query[:, :n_context_tokens],
            key[:, :n_context_tokens],
            value[:, :n_context_tokens],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=backend,
            parallel_config=parallel_config,
        )
        current = dispatch_attention_fn(
            query[:, n_context_tokens:],
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=backend,
            parallel_config=parallel_config,
        )
        hidden_states = torch.cat([context, current], dim=1)
    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


class RefinerChunkRunner:
    def __init__(
        self,
        refiner: DiffusersLTX2Refiner,
        *,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        sigmas: torch.Tensor,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        spatial_shape: tuple[int, int],
    ) -> None:
        self.refiner = refiner
        self.prompt_embeds = prompt_embeds
        self.prompt_attention_mask = prompt_attention_mask
        self.fps = float(fps)
        self.sigmas = sigmas
        self.source_sink_frames = int(source_sink_frames)
        self.block_size = int(block_size)
        self.kv_max_frames = int(kv_max_frames)
        self.max_history_frames = self.kv_max_frames - self.source_sink_frames
        self.generator = torch.Generator(device=refiner.device).manual_seed(int(seed))
        self.device = refiner.device
        self.dtype = refiner.dtype
        self.height, self.width = int(spatial_shape[0]), int(spatial_shape[1])
        transformer = refiner.transformer
        self.tokens_per_frame = (
            int(self.height // transformer.config.patch_size)
            * int(self.width // transformer.config.patch_size)
            * int(transformer.config.patch_size_t)
        )
        self.sink_kv_pre: list[tuple[torch.Tensor | None, torch.Tensor | None]] | None = None
        self.history_kv_post: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(transformer.transformer_blocks)
        self.history_frames = 0

    @torch.inference_mode()
    def refine_block(
        self,
        *,
        block_idx: int,
        clean_block: torch.Tensor,
        block_start: int,
        block_end: int,
        sink_seed_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_idx
        refiner = self.refiner
        if block_start < self.source_sink_frames:
            raise ValueError("refiner block overlaps source sink")
        if self.sink_kv_pre is None:
            if self.source_sink_frames == 0:
                self.sink_kv_pre = [(None, None) for _ in self.history_kv_post]
            elif sink_seed_frames is None:
                raise ValueError("first refine_block call requires sink_seed_frames")
            else:
                self.sink_kv_pre = refiner._capture_block_kv(
                    clean_block=sink_seed_frames.contiguous(),
                    frame_positions=list(range(self.source_sink_frames)),
                    prompt_embeds=self.prompt_embeds,
                    prompt_attention_mask=self.prompt_attention_mask,
                    fps=self.fps,
                    capture_mode="pre_rope",
                    kv_prefix_per_layer=None,
                )

        batch = int(clean_block.shape[0])
        sink_rope_offset = block_start - self.history_frames - self.source_sink_frames
        sink_pe = None
        if self.source_sink_frames > 0:
            sink_pe = build_rotary_emb_for_absolute_positions(
                transformer=refiner.transformer,
                batch_size=batch,
                frame_positions=list(range(sink_rope_offset, sink_rope_offset + self.source_sink_frames)),
                height=self.height,
                width=self.width,
                device=self.device,
                fps=self.fps,
            )
        kv_prefix_per_layer: list[dict[str, object]] = []
        for layer_idx, sink_kv in enumerate(self.sink_kv_pre):
            history = self.history_kv_post[layer_idx]
            kv_prefix_per_layer.append(
                {
                    "mode": "rf_shifted_sink",
                    "sink_k_pre": sink_kv[0],
                    "sink_v": sink_kv[1],
                    "sink_pe": sink_pe,
                    "history_k": history[0] if history is not None else None,
                    "history_v": history[1] if history is not None else None,
                }
            )

        sigma0 = float(self.sigmas[0].item())
        eps = torch.randn(clean_block.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        x_t = ((1.0 - sigma0) * clean_block.float() + sigma0 * eps.float()).to(self.dtype)
        active_positions = list(range(int(block_start), int(block_end)))
        for level in range(int(self.sigmas.numel()) - 1):
            sigma_cur = float(self.sigmas[level].item())
            sigma_next = float(self.sigmas[level + 1].item())
            pred_x0 = refiner._predict_x0_active_block(
                active=x_t,
                active_positions=active_positions,
                sigma_cur=sigma_cur,
                prompt_embeds=self.prompt_embeds,
                prompt_attention_mask=self.prompt_attention_mask,
                fps=self.fps,
                kv_prefix_per_layer=kv_prefix_per_layer,
            )
            if sigma_cur <= 1.0e-6:
                x_t = pred_x0.to(self.dtype)
            else:
                ratio = sigma_next / sigma_cur
                x_t = (ratio * x_t.float() + (1.0 - ratio) * pred_x0.float()).to(self.dtype)

        block_kv_post = refiner._capture_block_kv(
            clean_block=x_t,
            frame_positions=active_positions,
            prompt_embeds=self.prompt_embeds,
            prompt_attention_mask=self.prompt_attention_mask,
            fps=self.fps,
            capture_mode="post_rope",
            kv_prefix_per_layer=kv_prefix_per_layer,
        )
        for layer_idx, new_kv in enumerate(block_kv_post):
            old = self.history_kv_post[layer_idx]
            self.history_kv_post[layer_idx] = new_kv if old is None else (
                torch.cat([old[0], new_kv[0]], dim=1),
                torch.cat([old[1], new_kv[1]], dim=1),
            )
        self.history_frames += int(block_end - block_start)
        if self.max_history_frames > 0 and self.history_frames > self.max_history_frames:
            keep_tokens = self.max_history_frames * self.tokens_per_frame
            for layer_idx, old in enumerate(self.history_kv_post):
                if old is not None:
                    self.history_kv_post[layer_idx] = (old[0][:, -keep_tokens:], old[1][:, -keep_tokens:])
            self.history_frames = self.max_history_frames
        return x_t


def build_rotary_emb_for_absolute_positions(
    *,
    transformer: nn.Module,
    batch_size: int,
    frame_positions: list[int],
    height: int,
    width: int,
    device: torch.device,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = transformer.rope
    patch_size_t = int(rope.patch_size_t)
    patch_size = int(rope.patch_size)
    f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
    if patch_size_t > 1:
        f_positions = f_positions[::patch_size_t]
    grid_h = torch.arange(0, height, patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(0, width, patch_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)
    patch_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
    patch_ends = grid + patch_delta.view(3, 1, 1, 1)
    latent_coords = torch.stack([grid, patch_ends], dim=-1).flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale = torch.tensor(rope.scale_factors, device=device)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale.view(*broadcast_shape)
    pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return rope(pixel_coords, device=device)


def set_kv_prefix_on_blocks(transformer: nn.Module, kv_prefix_per_layer: list[dict[str, object]] | None) -> None:
    if kv_prefix_per_layer is None:
        clear_kv_prefix_on_blocks(transformer)
        return
    for block, prefix in zip(transformer.transformer_blocks, kv_prefix_per_layer):
        block.attn1._tf_kv_prefix = prefix


def clear_kv_prefix_on_blocks(transformer: nn.Module) -> None:
    for block in transformer.transformer_blocks:
        block.attn1._tf_kv_prefix = None


def set_capture_flag_on_blocks(transformer: nn.Module, mode: str, *, enable: bool) -> None:
    if mode == "pre_rope":
        attr, clear_attr = "_kv_cache_capture", "_cached_kv_pre"
    elif mode == "post_rope":
        attr, clear_attr = "_tf_capture_kv", "_cached_kv_post"
    else:
        raise ValueError(f"unsupported capture mode: {mode}")
    for block in transformer.transformer_blocks:
        setattr(block.attn1, attr, bool(enable))
        if enable and hasattr(block.attn1, clear_attr):
            setattr(block.attn1, clear_attr, None)


def collect_captured_kv_from_blocks(transformer: nn.Module, mode: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    attr = "_cached_kv_pre" if mode == "pre_rope" else "_cached_kv_post"
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in transformer.transformer_blocks:
        cached = getattr(block.attn1, attr, None)
        if cached is None:
            raise RuntimeError(f"missing captured KV on {attr}")
        out.append(cached)
        setattr(block.attn1, attr, None)
    return out


def pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch, seq_len, hidden_dim, _ = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype
    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        mask = token_indices >= (seq_len - sequence_lengths[:, None])
    else:
        raise ValueError(f"padding_side must be left or right, got {padding_side}")

    mask = mask[:, :, None, None]
    masked = text_hidden_states.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * hidden_dim).view(batch, 1, 1, 1)
    masked_mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized = normalized * scale_factor
    normalized = normalized.flatten(2)
    flat_mask = mask.squeeze(-1).expand(-1, -1, normalized.shape[-1])
    return normalized.masked_fill(~flat_mask, 0.0).to(dtype=original_dtype)


def pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch, _, frames, height, width = latents.shape
    latents = latents.reshape(
        batch,
        -1,
        frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch = latents.size(0)
    latents = latents.reshape(batch, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
