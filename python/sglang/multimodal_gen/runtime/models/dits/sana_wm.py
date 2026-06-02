#!/usr/bin/env python3
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Clean SANA-WM 1600M DiT definition.

The architecture is intentionally fixed to the public SANA-WM 720p checkpoint:
20 blocks, bidirectional UCPE camera control, GLUMBConvTemp FFN, WAN RoPE, and
chunk Plucker post-attention injection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    CaptionEmbedder,
    DropPath,
    GLUMBConvTemp,
    MultiHeadCrossAttention,
    PatchEmbedMS3D,
    RMSNorm,
    SanaWMGDNAttention,
    SanaWMSoftmaxAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    WanRotaryPosEmbed,
    process_camera_raymats_ucpe,
    t2i_modulate,
)


@dataclass(frozen=True)
class SanaWMSequenceShardInfo:
    full_token_shape: tuple[int, int, int]
    local_token_shape: tuple[int, int, int]
    axis: int

    @property
    def local_h(self) -> int:
        return self.local_token_shape[1]

    @property
    def local_w(self) -> int:
        return self.local_token_shape[2]


def _slice_tokens_for_sp(
    tensor: torch.Tensor,
    token_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, SanaWMSequenceShardInfo | None]:
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return tensor, None
    b, _, c = tensor.shape
    frames, h, w = token_shape
    if h % sp_world_size == 0:
        axis = 1
        local_h = h // sp_world_size
        local_w = w
    elif w % sp_world_size == 0:
        axis = 2
        local_h = h
        local_w = w // sp_world_size
    else:
        raise ValueError(
            f"SANA-WM sequence shard requires token height or width divisible by SP degree: "
            f"token_shape={token_shape}, sp_degree={sp_world_size}"
        )
    sp_rank = get_sp_parallel_rank()
    tensor = tensor.reshape(b, frames, h, w, c)
    if axis == 1:
        h0 = sp_rank * local_h
        local = tensor[:, :, h0 : h0 + local_h]
    else:
        w0 = sp_rank * local_w
        local = tensor[:, :, :, w0 : w0 + local_w]
    local = local.reshape(b, frames * local_h * local_w, c)
    return local, SanaWMSequenceShardInfo(
        full_token_shape=token_shape,
        local_token_shape=(frames, local_h, local_w),
        axis=axis,
    )


def _slice_existing_tokens_for_sp(
    tensor: torch.Tensor,
    info: SanaWMSequenceShardInfo,
) -> torch.Tensor:
    b, _, c = tensor.shape
    frames, h, w = info.full_token_shape
    tensor = tensor.reshape(b, frames, h, w, c)
    sp_rank = get_sp_parallel_rank()
    if info.axis == 1:
        h0 = sp_rank * info.local_h
        tensor = tensor[:, :, h0 : h0 + info.local_h]
    else:
        w0 = sp_rank * info.local_w
        tensor = tensor[:, :, :, w0 : w0 + info.local_w]
    return tensor.reshape(b, frames * info.local_h * info.local_w, c)


def _gather_tokens_for_sp(
    tensor: torch.Tensor,
    info: SanaWMSequenceShardInfo,
) -> torch.Tensor:
    b, _, c = tensor.shape
    frames, h, w = info.full_token_shape
    tensor = tensor.reshape(b, frames, info.local_h, info.local_w, c).contiguous()
    tensor = sequence_model_parallel_all_gather(tensor, dim=2 if info.axis == 1 else 3)
    return tensor.reshape(b, frames * h * w, c)


def _slice_rotary_for_sp(
    rotary_emb: torch.Tensor | None,
    info: SanaWMSequenceShardInfo | None,
) -> torch.Tensor | None:
    if rotary_emb is None or info is None:
        return rotary_emb
    frames, h, w = info.full_token_shape
    d = rotary_emb.shape[-1]
    rotary = rotary_emb.reshape(1, 1, frames, h, w, d)
    sp_rank = get_sp_parallel_rank()
    if info.axis == 1:
        h0 = sp_rank * info.local_h
        rotary = rotary[:, :, :, h0 : h0 + info.local_h]
    else:
        w0 = sp_rank * info.local_w
        rotary = rotary[:, :, :, :, w0 : w0 + info.local_w]
    return rotary.reshape(1, 1, frames * info.local_h * info.local_w, d)


def _slice_raymats_for_sp(
    raymats: torch.Tensor,
    info: SanaWMSequenceShardInfo | None,
) -> torch.Tensor:
    if info is None:
        return raymats
    sp_rank = get_sp_parallel_rank()
    if info.axis == 1:
        h0 = sp_rank * info.local_h
        return raymats[:, :, h0 : h0 + info.local_h]
    w0 = sp_rank * info.local_w
    return raymats[:, :, :, w0 : w0 + info.local_w]


def _gather_latents_for_sp(
    latents: torch.Tensor,
    info: SanaWMSequenceShardInfo | None,
) -> torch.Tensor:
    if info is None:
        return latents
    return sequence_model_parallel_all_gather(
        latents.contiguous(),
        dim=3 if info.axis == 1 else 4,
    )


def _stream_debug_dump_once(owner: object, name: str, tensor: torch.Tensor) -> None:
    debug_dir = os.environ.get("SANAWM_STREAM_DEBUG_DIR")
    if not debug_dir or getattr(owner, f"_stream_debug_dumped_{name}", False):
        return
    os.makedirs(debug_dir, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(debug_dir, f"{name}.pt"))
    setattr(owner, f"_stream_debug_dumped_{name}", True)


def _stream_debug_selected(block_idx: object) -> bool:
    if block_idx is None:
        return False
    idx = int(block_idx)
    spec = os.environ.get("SANAWM_STREAM_DEBUG_BLOCKS")
    if not spec:
        return idx == 0
    if spec.strip().lower() in {"*", "all"}:
        return True
    return any(part.strip() and int(part.strip()) == idx for part in spec.split(","))


class SanaWMBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        use_softmax_attention: bool,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        attn_cls = SanaWMSoftmaxAttention if use_softmax_attention else SanaWMGDNAttention
        self.attn = attn_cls(
            hidden_size,
            hidden_size,
            heads=num_heads,
            cam_dim=hidden_size,
            cam_heads=num_heads,
            patch_size=(1, 1, 1),
            qk_norm=True,
            conv_kernel_size=4,
            k_conv_only=True,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConvTemp(
            in_features=hidden_size,
            hidden_features=hidden_size * 3,
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
            t_kernel_size=3,
        )
        self.drop_path = DropPath(drop_path)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.plucker_proj.weight)
        nn.init.zeros_(self.plucker_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        token_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor,
        camera_conditions: torch.Tensor,
        plucker_emb: torch.Tensor,
        camera_raymats: torch.Tensor | None = None,
        full_rotary_emb: torch.Tensor | None = None,
        full_camera_raymats: torch.Tensor | None = None,
        sequence_shard_info: SanaWMSequenceShardInfo | None = None,
        kv_cache: list[torch.Tensor | None] | None = None,
        save_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        b, n, c = x.shape
        frames = t.shape[2]
        debug_idx = getattr(self, "_stream_debug_block_idx", None)
        debug_detail = _stream_debug_selected(debug_idx)
        debug_prefix = f"block{int(debug_idx):02d}" if debug_idx is not None else "blockxx"
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_x_in", x)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t.reshape(b, frames, 6, -1)
        ).chunk(6, dim=-2)

        x_msa = t2i_modulate(self.norm1(x).reshape(b, frames, -1, c), shift_msa, scale_msa).reshape(b, n, c)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_x_msa_in", x_msa)
        if sequence_shard_info is not None and isinstance(self.attn, SanaWMSoftmaxAttention):
            attn_input = _gather_tokens_for_sp(x_msa, sequence_shard_info)
            attn_out = self.attn(
                attn_input,
                HW=sequence_shard_info.full_token_shape,
                rotary_emb=full_rotary_emb,
                camera_conditions=camera_conditions,
                camera_raymats=full_camera_raymats,
                kv_cache=kv_cache,
                save_kv_cache=save_kv_cache,
            )
        else:
            attn_out = self.attn(
                x_msa,
                HW=token_shape,
                rotary_emb=rotary_emb,
                camera_conditions=camera_conditions,
                camera_raymats=camera_raymats,
                sequence_parallel=sequence_shard_info is not None,
                kv_cache=kv_cache,
                save_kv_cache=save_kv_cache,
            )
        if kv_cache is not None:
            attn, kv_cache = attn_out
        else:
            attn = attn_out
        if sequence_shard_info is not None and isinstance(self.attn, SanaWMSoftmaxAttention):
            attn = _slice_existing_tokens_for_sp(attn, sequence_shard_info)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_attn_raw", attn)
        attn = (gate_msa * attn.reshape(b, frames, -1, c)).reshape(b, n, c)
        x = x + self.drop_path(attn)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_after_attn", x)
        plucker_delta = self.plucker_proj(plucker_emb)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_plucker_emb", plucker_emb)
            _stream_debug_dump_once(self, f"{debug_prefix}_plucker_delta", plucker_delta)
        x = x + plucker_delta
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_after_plucker", x)
        x = x + self.cross_attn(x, y, mask=mask)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_after_cross", x)

        x_mlp = t2i_modulate(self.norm2(x).reshape(b, frames, -1, c), shift_mlp, scale_mlp).reshape(b, n, c)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_x_mlp_in", x_mlp)
        if sequence_shard_info is not None:
            mlp_input = _gather_tokens_for_sp(x_mlp, sequence_shard_info)
            mlp_out = self.mlp(
                mlp_input,
                HW=sequence_shard_info.full_token_shape,
                kv_cache=kv_cache,
                save_kv_cache=save_kv_cache,
            )
        else:
            mlp_out = self.mlp(x_mlp, HW=token_shape, kv_cache=kv_cache, save_kv_cache=save_kv_cache)
        if kv_cache is not None:
            mlp, kv_cache = mlp_out
        else:
            mlp = mlp_out
        if sequence_shard_info is not None:
            mlp = _slice_existing_tokens_for_sp(mlp, sequence_shard_info)
        if debug_detail:
            _stream_debug_dump_once(self, f"{debug_prefix}_mlp_raw", mlp)
        mlp = mlp.reshape(b, frames, -1, c)
        x = x + self.drop_path((gate_mlp * mlp).reshape(b, n, c))
        if debug_idx is not None:
            _stream_debug_dump_once(self, f"block{int(debug_idx):02d}_out", x)
        return (x, kv_cache) if kv_cache is not None else x


class SanaWM1600M(nn.Module):
    param_names_mapping: dict[str, str] = {r"^transformer\.(.*)$": r"\1"}
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []

    latent_channels = 128
    hidden_size = 2240
    depth = 20
    num_heads = 20
    model_max_length = 300
    caption_channels = 2304
    patch_size = (1, 1, 1)

    def __init__(
        self,
        config: object | None = None,
        hf_config: dict[str, object] | None = None,
        quant_config: object | None = None,
    ) -> None:
        super().__init__()
        del config, hf_config, quant_config
        hidden = self.hidden_size
        self.pred_sigma = False
        self.in_channels = self.latent_channels
        self.out_channels = self.latent_channels
        self.timestep_norm_scale_factor = 1.0
        self.y_norm = True
        self.x_embedder = PatchEmbedMS3D(self.patch_size, self.in_channels, hidden, kernel_size=self.patch_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=self.caption_channels,
            hidden_size=hidden,
            uncond_prob=0.0,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            token_num=self.model_max_length,
        )
        self.attention_y_norm = RMSNorm(hidden, scale_factor=0.01, eps=1e-5)
        self.raymap_embedder = PatchEmbedMS3D(self.patch_size, 3, hidden, kernel_size=self.patch_size, bias=True)
        self.plucker_embedder = PatchEmbedMS3D(self.patch_size, 48, hidden, kernel_size=self.patch_size, bias=True)
        self.rope = WanRotaryPosEmbed(attention_head_dim=112, patch_size=self.patch_size, max_seq_len=1024)
        self.blocks = nn.ModuleList(
            [
                SanaWMBlock(hidden, self.num_heads, use_softmax_attention=((i + 1) % 4 == 0))
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden, self.patch_size, self.out_channels)
        self.initialize_weights()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _sequence_shard_enabled(self) -> bool:
        forward_batch = get_forward_context().forward_batch
        return (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and get_sp_world_size() > 1
        )

    def post_load_weights(self) -> None:
        self.rope.materialize_freqs()

    def initialize_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view(self.x_embedder.proj.weight.shape[0], -1))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        nn.init.zeros_(self.raymap_embedder.proj.weight)
        nn.init.zeros_(self.raymap_embedder.proj.bias)
        nn.init.zeros_(self.plucker_embedder.proj.weight)
        nn.init.zeros_(self.plucker_embedder.proj.bias)
        for block in self.blocks:
            if hasattr(block.attn, "_init_gates"):
                block.attn._init_gates()
            nn.init.zeros_(block.attn.out_proj_cam.weight)
            nn.init.zeros_(block.attn.out_proj_cam.bias)
            nn.init.zeros_(block.plucker_proj.weight)
            nn.init.zeros_(block.plucker_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: torch.Tensor,
        camera_conditions: torch.Tensor,
        chunk_plucker: torch.Tensor,
        start_f: int | None = None,
        end_f: int | None = None,
        frame_index: torch.Tensor | None = None,
        kv_cache: list[list[torch.Tensor | None]] | None = None,
        save_kv_cache: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[torch.Tensor | None]]]:
        if kv_cache is not None or start_f is not None or end_f is not None or frame_index is not None:
            return self.forward_long(
                x,
                timestep,
                y,
                mask=mask,
                camera_conditions=camera_conditions,
                chunk_plucker=chunk_plucker,
                start_f=start_f,
                end_f=end_f,
                frame_index=frame_index,
                kv_cache=kv_cache,
                save_kv_cache=save_kv_cache,
            )

        x = x.to(self.dtype)
        y = y.to(self.dtype)
        camera_conditions = camera_conditions.to(self.dtype)
        timestep = timestep.float()
        frames, height, width = x.shape[-3:]
        token_shape = (frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2])
        camera_raymats = process_camera_raymats_ucpe(camera_conditions, token_shape, self.patch_size)

        x = self.x_embedder(x)
        plucker_emb = self.plucker_embedder(chunk_plucker.to(self.dtype))
        rotary_emb = self.rope(token_shape, x.device)
        sequence_shard_info = None
        full_rotary_emb = rotary_emb
        full_camera_raymats = camera_raymats
        if self._sequence_shard_enabled():
            x, sequence_shard_info = _slice_tokens_for_sp(x, token_shape)
            plucker_emb = _slice_existing_tokens_for_sp(plucker_emb, sequence_shard_info)
            rotary_emb = _slice_rotary_for_sp(rotary_emb, sequence_shard_info)
            camera_raymats = _slice_raymats_for_sp(camera_raymats, sequence_shard_info)
            token_shape = sequence_shard_info.local_token_shape

        t = self.t_embedder(timestep.flatten())
        t0 = self.t_block(t).unflatten(dim=0, sizes=timestep.shape)
        t = t.unflatten(dim=0, sizes=timestep.shape)

        y = self.y_embedder(y, self.training, mask=mask)
        y = self.attention_y_norm(y)
        mask = mask.to(torch.int16)
        if mask.shape[0] != y.shape[0]:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
        mask = mask.squeeze(1).squeeze(1) if mask.ndim > 2 else mask

        for block in self.blocks:
            x = block(
                x,
                y,
                t0,
                mask,
                token_shape,
                rotary_emb,
                camera_conditions,
                plucker_emb,
                camera_raymats=camera_raymats,
                full_rotary_emb=full_rotary_emb,
                full_camera_raymats=full_camera_raymats,
                sequence_shard_info=sequence_shard_info,
            )

        x = self.final_layer(x, t)
        return _gather_latents_for_sp(
            self.unpatchify(x, token_shape),
            sequence_shard_info,
        )

    def forward_long(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: torch.Tensor,
        camera_conditions: torch.Tensor,
        chunk_plucker: torch.Tensor,
        start_f: int | None = None,
        end_f: int | None = None,
        frame_index: torch.Tensor | None = None,
        kv_cache: list[list[torch.Tensor | None]] | None = None,
        save_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor | None]]]:
        if kv_cache is None:
            kv_cache = [[None] * 10 for _ in range(len(self.blocks))]
        x = x.to(self.dtype)
        y = y.to(self.dtype)
        frames, height, width = x.shape[-3:]
        if timestep.ndim == 1:
            timestep = timestep[:, None, None].expand(-1, 1, frames)
        elif timestep.ndim == 2:
            timestep = timestep[:, None, :]
        timestep = timestep.float()
        start = 0 if start_f is None else int(start_f)
        end = start + frames if end_f is None else int(end_f)
        token_shape = (frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2])

        if camera_conditions.shape[1] != frames:
            camera_conditions = camera_conditions[:, start:end]
        if chunk_plucker.shape[2] != frames:
            chunk_plucker = chunk_plucker[:, :, start:end]
        if camera_conditions.shape[0] != x.shape[0]:
            camera_conditions = camera_conditions.repeat(x.shape[0] // camera_conditions.shape[0], 1, 1)
        if chunk_plucker.shape[0] != x.shape[0]:
            chunk_plucker = chunk_plucker.repeat(x.shape[0] // chunk_plucker.shape[0], 1, 1, 1, 1)
        camera_conditions = camera_conditions.to(self.dtype)
        camera_raymats = process_camera_raymats_ucpe(camera_conditions, token_shape, self.patch_size)

        x = self.x_embedder(x)
        plucker_emb = self.plucker_embedder(chunk_plucker.to(self.dtype))
        rotary_fhw: tuple[int | tuple[int, int], int, int]
        rotary_fhw = (token_shape[0], token_shape[1], token_shape[2]) if frame_index is not None else ((start, end), token_shape[1], token_shape[2])
        rotary_emb = self.rope(rotary_fhw, x.device, frame_index=frame_index)
        sequence_shard_info = None
        full_rotary_emb = rotary_emb
        full_camera_raymats = camera_raymats
        if self._sequence_shard_enabled():
            x, sequence_shard_info = _slice_tokens_for_sp(x, token_shape)
            plucker_emb = _slice_existing_tokens_for_sp(plucker_emb, sequence_shard_info)
            rotary_emb = _slice_rotary_for_sp(rotary_emb, sequence_shard_info)
            camera_raymats = _slice_raymats_for_sp(camera_raymats, sequence_shard_info)
            token_shape = sequence_shard_info.local_token_shape

        t = self.t_embedder(timestep.flatten())
        t0 = self.t_block(t).unflatten(dim=0, sizes=timestep.shape)
        t = t.unflatten(dim=0, sizes=timestep.shape)

        y = self.y_embedder(y, self.training, mask=mask)
        y = self.attention_y_norm(y)
        mask = mask.to(torch.int16)
        if mask.shape[0] != y.shape[0]:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
        mask = mask.squeeze(1).squeeze(1) if mask.ndim > 2 else mask

        new_cache: list[list[torch.Tensor | None]] = []
        for block_idx, block in enumerate(self.blocks):
            block._stream_debug_block_idx = block_idx
            block.attn._stream_debug_block_idx = block_idx
            block.mlp._stream_debug_block_idx = block_idx
            x, block_cache = block(
                x,
                y,
                t0,
                mask,
                token_shape,
                rotary_emb,
                camera_conditions,
                plucker_emb,
                camera_raymats=camera_raymats,
                full_rotary_emb=full_rotary_emb,
                full_camera_raymats=full_camera_raymats,
                sequence_shard_info=sequence_shard_info,
                kv_cache=kv_cache[block_idx],
                save_kv_cache=save_kv_cache,
            )
            new_cache.append(block_cache)

        x = self.final_layer(x, t)
        return _gather_latents_for_sp(
            self.unpatchify(x, token_shape),
            sequence_shard_info,
        ), new_cache

    def unpatchify(self, x: torch.Tensor, token_shape: tuple[int, int, int]) -> torch.Tensor:
        frames, height, width = token_shape
        pf, ph, pw = self.patch_size
        c = self.out_channels
        x = x.reshape(x.shape[0], frames, height, width, pf, ph, pw, c)
        x = torch.einsum("bfhwopqc->bcfohpwq", x)
        return x.reshape(x.shape[0], c, frames * pf, height * ph, width * pw)


def create_model() -> SanaWM1600M:
    return SanaWM1600M()


EntryClass = SanaWM1600M
