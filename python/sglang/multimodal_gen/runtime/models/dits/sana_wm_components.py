#!/usr/bin/env python3
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""SANA-WM 1600M building blocks.

This file contains reusable layers, UCPE camera transforms, and attention
blocks for the fixed inference model.
"""

from __future__ import annotations

import math
import os
import types
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_grid(height: int, width: int, *, batch: int | None = None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=dtype, device=device),
        torch.linspace(0, width - 1, width, dtype=dtype, device=device),
        indexing="ij",
    )
    grid = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    return grid if batch is None else grid.unsqueeze(0).expand(batch, -1, -1, -1)


def _flat_param(x: torch.Tensor | float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype).reshape(-1)
    return torch.tensor([x], dtype=dtype, device=device)


def compute_fx_from_fov_xi(x_fov: torch.Tensor | float, xi: torch.Tensor | float, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x_fov = _flat_param(x_fov, dtype=dtype, device=device)
    xi = _flat_param(xi, dtype=dtype, device=device)
    b = max(x_fov.numel(), xi.numel())
    x_fov = x_fov.expand(b)
    xi = xi.expand(b)
    theta = torch.deg2rad(0.5 * x_fov)
    return (width * 0.5) * (torch.cos(theta) + xi) / torch.sin(theta).clamp_min(torch.finfo(dtype).eps)


def compute_fov_from_fx_xi(fx: torch.Tensor | float, xi: torch.Tensor | float, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    fx = _flat_param(fx, dtype=dtype, device=device)
    xi = _flat_param(xi, dtype=dtype, device=device)
    b = max(fx.numel(), xi.numel())
    fx = fx.expand(b)
    xi = xi.expand(b)
    a = 2.0 * fx / width
    theta = torch.asin((xi / torch.sqrt(a * a + 1.0)).clamp(-1.0, 1.0)) + torch.atan(1.0 / a)
    return torch.rad2deg(2.0 * theta)


def ucm_unproject_grid(
    *,
    height: int,
    width: int,
    fx: torch.Tensor | float,
    fy: torch.Tensor | float,
    cx: torch.Tensor | float,
    cy: torch.Tensor | float,
    xi: torch.Tensor | float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    fx, fy, cx, cy, xi = [_flat_param(v, dtype=dtype, device=device) for v in (fx, fy, cx, cy, xi)]
    b = max(v.numel() for v in (fx, fy, cx, cy, xi))
    fx, fy, cx, cy, xi = [v.expand(b)[:, None, None] for v in (fx, fy, cx, cy, xi)]
    grid = create_grid(height, width, batch=b, dtype=dtype, device=device)
    x = (grid[..., 0] - cx) / fx
    y = (grid[..., 1] - cy) / fy
    r2 = x * x + y * y
    gamma = (xi + torch.sqrt(1 + (1 - xi * xi) * r2)) / (1 + r2)
    return torch.stack([gamma * x, gamma * y, gamma - xi], dim=-1)


def ucm_unproject_grid_fov(
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    fx = compute_fx_from_fov_xi(x_fov, xi, width, dtype=dtype, device=device)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, dtype=dtype, device=device)
    return ucm_unproject_grid(height=height, width=width, fx=fx, fy=fy, cx=cx, cy=cy, xi=xi, dtype=dtype, device=device)


def project_ucm_points(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    xi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def reshape(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = p.to(device=target.device, dtype=target.dtype)
        if p.ndim == 1 and target.ndim == 4:
            return p.view(target.shape[0], target.shape[1], 1, 1)
        while p.ndim < target.ndim:
            p = p.unsqueeze(-1)
        return p

    fx, fy, cx, cy, xi = [reshape(p, X) for p in (fx, fy, cx, cy, xi)]
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi * r
    return fx * (X / alpha) + cx, fy * (Y / alpha) + cy


def project_ucm_points_fov(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fx = compute_fx_from_fov_xi(x_fov, xi, width, dtype=X.dtype, device=X.device)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, dtype=X.dtype, device=X.device)
    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


def world_to_ray_mats(d_cam: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if d_cam.ndim == 4:
        d_cam = d_cam[:, None].expand(-1, c2w.shape[1], -1, -1, -1)
    b, t, h, w, _ = d_cam.shape
    r_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", r_cam, d_cam)
    cam_y = r_cam[..., :, 1][:, :, None, None].expand(-1, -1, h, w, -1)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    r_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    r_w2l = r_l2w.transpose(-1, -2)
    t_w2l = -torch.einsum("bthwij,btj->bthwi", r_w2l, t_cam)
    mats = torch.zeros(b, t, h, w, 4, 4, device=d_cam.device, dtype=d_cam.dtype)
    mats[..., :3, :3] = r_w2l
    mats[..., :3, 3] = t_w2l
    mats[..., 3, 3] = 1.0
    return mats


def compute_absmap(
    c2w: torch.Tensor,
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
) -> torch.Tensor:
    b, t = c2w.shape[:2]
    dtype, device = c2w.dtype, c2w.device
    d_cam = ucm_unproject_grid_fov(x_fov, y_fov, xi, height, width, cx, cy, dtype=torch.float32, device=device)
    d_cam = d_cam.view(b, t, height, width, 3)
    d_world = torch.einsum("btij,bthwj->bthwi", c2w[..., :3, :3].float(), d_cam)
    d_world = F.normalize(d_world, dim=-1, eps=1e-8)
    xw, yw, zw = d_world.unbind(-1)
    lat_map = torch.atan2(-yw, torch.sqrt(xw * xw + zw * zw)).unsqueeze(-1)
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = F.normalize(torch.cross(d_world, up_world.expand_as(d_world), dim=-1), dim=-1, eps=1e-8)
    delta = torch.tensor(0.1, device=device, dtype=torch.float32)
    v_rot = d_world * torch.cos(delta) + torch.cross(k, d_world, dim=-1) * torch.sin(delta)
    dirs_cam = torch.einsum("btij,bthwj->bthwi", c2w[..., :3, :3].float().transpose(-1, -2), v_rot)
    du, dv = project_ucm_points_fov(
        dirs_cam[..., 0],
        dirs_cam[..., 1],
        dirs_cam[..., 2],
        x_fov.float(),
        y_fov.float(),
        xi.float(),
        height,
        width,
        cx.float(),
        cy.float(),
    )
    grid = create_grid(height, width, batch=b, dtype=torch.float32, device=device)
    up_map = torch.stack((du - grid[:, None, ..., 0], dv - grid[:, None, ..., 1]), dim=-1)
    up_map = F.normalize(up_map, dim=-1, eps=1e-8)
    return torch.cat([up_map, lat_map], dim=-1).to(dtype)


def process_camera_conditions_ucpe(
    camera_conditions: torch.Tensor,
    token_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    b, frames = camera_conditions.shape[:2]
    c2w = camera_conditions[..., :16].view(b, frames, 4, 4)
    fx, fy, cx, cy = [camera_conditions[..., i] for i in range(16, 20)]
    _, h, w = token_shape
    image_h, image_w = h * patch_size[1], w * patch_size[2]
    xi = torch.zeros((b, frames), device=camera_conditions.device, dtype=camera_conditions.dtype)
    x_fov = compute_fov_from_fx_xi(fx, xi, image_w, dtype=camera_conditions.dtype, device=camera_conditions.device).view(b, frames)
    y_fov = compute_fov_from_fx_xi(fy, xi, image_h, dtype=camera_conditions.dtype, device=camera_conditions.device).view(b, frames)
    d_cam = ucm_unproject_grid_fov(
        x_fov,
        y_fov,
        xi,
        h,
        w,
        cx / patch_size[2],
        cy / patch_size[1],
        dtype=camera_conditions.dtype,
        device=camera_conditions.device,
    ).view(b, frames, h, w, 3)
    raymats = world_to_ray_mats(d_cam, c2w)
    absmap = compute_absmap(c2w, x_fov, y_fov, xi, image_h, image_w, cx, cy)
    return raymats, absmap


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    r_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", r_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def apply_ray_matrix(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    b, heads, n, dim = feats.shape
    d = matrix.shape[-1]
    x = feats.reshape(b, heads, n, -1, d)
    return torch.einsum("bnij,bhnkj->bhnki", matrix, x).reshape(b, heads, n, dim)


def apply_complex_rope(x: torch.Tensor, freqs: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    z = torch.view_as_complex(x.to(torch.float64).contiguous().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    return torch.view_as_real(z * freqs).flatten(-2, -1).type_as(x)


def _block_apply(x: torch.Tensor, pairs: tuple[tuple[Callable[[torch.Tensor], torch.Tensor], int], ...]) -> torch.Tensor:
    chunks = torch.split(x, [size for _, size in pairs], dim=-1)
    return torch.cat([fn(chunk) for chunk, (fn, _) in zip(chunks, pairs)], dim=-1)


def _slice_cam_rope(rotary_emb: torch.Tensor | None, head_dim: int) -> torch.Tensor | None:
    if rotary_emb is None:
        return None
    orig_t = head_dim // 2 - 2 * (head_dim // 6)
    orig_h = head_dim // 6
    cam_dim = head_dim // 2
    new_t = cam_dim // 2 - 2 * (cam_dim // 6)
    new_h = cam_dim // 6
    new_w = cam_dim // 6
    return torch.cat(
        [
            rotary_emb[..., :new_t],
            rotary_emb[..., orig_t : orig_t + new_h],
            rotary_emb[..., orig_t + orig_h : orig_t + orig_h + new_w],
        ],
        dim=-1,
    )


def prepare_prope_fns(
    head_dim: int,
    camera_conditions: torch.Tensor,
    token_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    rotary_emb: torch.Tensor | None,
    raymats: torch.Tensor | None = None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    b = camera_conditions.shape[0]
    if raymats is None:
        raymats, _ = process_camera_conditions_ucpe(camera_conditions, token_shape, patch_size)
    p = raymats.reshape(b, -1, 4, 4)
    p_t = p.transpose(-1, -2)
    p_inv = invert_se3(p)
    rope = _slice_cam_rope(rotary_emb, head_dim)
    rope_fn = (lambda x: x) if rope is None else partial(apply_complex_rope, freqs=rope, inverse=False)
    rope_inv = (lambda x: x) if rope is None else partial(apply_complex_rope, freqs=rope, inverse=True)
    q_pairs = ((partial(apply_ray_matrix, matrix=p_t), head_dim // 2), (rope_fn, head_dim // 2))
    kv_pairs = ((partial(apply_ray_matrix, matrix=p_inv), head_dim // 2), (rope_fn, head_dim // 2))
    out_pairs = ((partial(apply_ray_matrix, matrix=p), head_dim // 2), (rope_inv, head_dim // 2))
    return partial(_block_apply, pairs=q_pairs), partial(_block_apply, pairs=kv_pairs), partial(_block_apply, pairs=out_pairs)


OUTPUT_GATE_INIT_BIAS = 1.278464542761074

_NUM_STREAM_CACHE_SLOTS = 10
_SLOT_K = 0
_SLOT_V = 1
_SLOT_CAM_K = 2
_SLOT_CAM_V = 3
_SLOT_SHORTCONV = 4
_SLOT_TYPE_FLAG = 6
_SLOT_FFN_TCONV = 9
_CACHE_TYPE_CONCAT = 0.0
_CACHE_TYPE_STATE = 1.0


def _stream_debug_dump_once(owner: object, name: str, tensor: torch.Tensor) -> None:
    debug_dir = os.environ.get("SANAWM_STREAM_DEBUG_DIR")
    if not debug_dir or getattr(owner, f"_stream_debug_dumped_{name}", False):
        return
    os.makedirs(debug_dir, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(debug_dir, f"{name}.pt"))
    setattr(owner, f"_stream_debug_dumped_{name}", True)


def _stream_debug_prefix(owner: object, stem: str) -> str | None:
    idx = getattr(owner, "_stream_debug_block_idx", None)
    if idx is None:
        return None
    idx = int(idx)
    spec = os.environ.get("SANAWM_STREAM_DEBUG_BLOCKS")
    if not spec:
        selected = idx == 0
    elif spec.strip().lower() in {"*", "all"}:
        selected = True
    else:
        selected = any(part.strip() and int(part.strip()) == idx for part in spec.split(","))
    return f"{stem}{idx:02d}" if selected else None


def _clone_cache_value(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.detach().clone() if value is not None else None


def _slice_rope_to_current_chunk(rotary_emb: torch.Tensor | None, current_tokens: int) -> torch.Tensor | None:
    if rotary_emb is None or rotary_emb.shape[-2] == current_tokens:
        return rotary_emb
    return rotary_emb[..., -current_tokens:, :]


class ShortConvolution(nn.Module):
    """Small depthwise causal convolution used by SANA-WM attention."""

    def __init__(self, hidden_size: int, kernel_size: int, activation: None = None) -> None:
        super().__init__()
        del activation
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight.to(dtype=x.dtype),
            padding=self.weight.shape[-1] - 1,
            groups=x.shape[-1],
        )
        y = y[..., : x.shape[1]].transpose(1, 2)
        return y, None


def flip_and_shift(x: torch.Tensor, dim: int, shift_val: float) -> torch.Tensor:
    x = torch.flip(x, dims=[dim])
    x = x.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    pad = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=dim)


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    z = torch.view_as_complex(x.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
    out = torch.view_as_real(z * freqs).flatten(3, 4).permute(0, 1, 3, 2)
    return out.type_as(x)


def sdpa_with_head_padding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pad head_dim so PyTorch SDPA can use flash kernels for dim=112."""
    dtype = q.dtype
    if q.dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    dim = q.shape[-1]
    needs_pad = dim not in (32, 64, 128, 256) and dim < 256
    if needs_pad:
        pad_to = 128 if dim <= 128 else 256
        pad = pad_to - dim
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))

    out = F.scaled_dot_product_attention(q, k, v)
    if needs_pad:
        out = out[..., :dim]
    return out.to(dtype) if out.dtype != dtype else out


def _recurrent_gdn_components(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    kv_state: torch.Tensor | None = None,
    z_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, heads, dim, n = q.shape
    frames = beta.shape[2]
    spatial = n // frames

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, heads, dim, frames, spatial).permute(0, 1, 3, 2, 4)

    q, k, v = to_time(q), to_time(k), to_time(v)
    q_rot, k_rot = to_time(q_rot), to_time(k_rot)
    beta = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(b, heads, frames, 1, 1)
    decay = decay.view(b, heads, frames, 1, 1)
    if kv_state is None:
        kv_state = torch.zeros(b, heads, dim, dim, device=q.device, dtype=q.dtype)
    else:
        kv_state = kv_state.to(device=q.device, dtype=q.dtype)
    if z_state is None:
        z_state = torch.zeros(b, heads, dim, 1, device=q.device, dtype=q.dtype)
    else:
        z_state = z_state.to(device=q.device, dtype=q.dtype)
    nums, dens = [], []
    for frame in range(frames):
        qt, kt, vt = q[:, :, frame], k[:, :, frame], v[:, :, frame]
        qrt, krt = q_rot[:, :, frame], k_rot[:, :, frame]
        bt, gt = beta[:, :, frame], decay[:, :, frame]
        kv_state = kv_state * gt
        z_state = z_state * gt
        delta_v = (vt - torch.matmul(kv_state, krt)) * bt
        kv_state = kv_state + torch.matmul(delta_v, krt.transpose(-1, -2))
        delta_z = (1.0 - torch.matmul(z_state.transpose(-1, -2), kt)) * bt
        z_state = z_state + torch.matmul(kt, delta_z.transpose(-1, -2))
        nums.append(torch.matmul(kv_state, qrt))
        dens.append(torch.matmul(z_state.transpose(-1, -2), qt))

    num = torch.stack(nums, dim=2).permute(0, 1, 3, 2, 4).reshape(b, heads, dim, n)
    den = torch.stack(dens, dim=2).permute(0, 1, 3, 2, 4).reshape(b, heads, 1, n)
    return num, den, kv_state, z_state


def recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    eps: float,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    num, den, _, _ = _recurrent_gdn_components(q, k, v, q_rot, k_rot, beta, decay)
    if return_components:
        return num, den
    return num / (den + eps)


def recurrent_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    b, heads, dim, n = q.shape
    frames = beta.shape[2]
    spatial = n // frames

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, heads, dim, frames, spatial).permute(0, 1, 3, 2, 4)

    q, k, v = to_time(q), to_time(k), to_time(v)
    beta = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(b, heads, frames, 1, 1)
    decay = decay.view(b, heads, frames, 1, 1)
    state = torch.zeros(b, heads, dim, dim, device=q.device, dtype=q.dtype)
    outs = []
    for frame in range(frames):
        qt, kt, vt = q[:, :, frame], k[:, :, frame], v[:, :, frame]
        state = state * decay[:, :, frame]
        delta_v = (vt - torch.matmul(state, kt)) * beta[:, :, frame]
        state = state + torch.matmul(delta_v, kt.transpose(-1, -2))
        outs.append(torch.matmul(state, qt))
    return torch.stack(outs, dim=2).permute(0, 1, 3, 2, 4).reshape(b, heads, dim, n)


def recurrent_gdn_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    eps: float,
    kv_state: torch.Tensor | None,
    z_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, heads, dim, n = q.shape
    frames = beta.shape[2]
    spatial = n // frames
    q_flat, k_flat, v_flat = q, k, v
    q_rot_flat, k_rot_flat = q_rot, k_rot
    beta_flat, decay_flat = beta, decay

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, heads, dim, frames, spatial).permute(0, 1, 3, 2, 4)

    def from_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(b, heads, dim, n)

    num_f, den_f, kv_state, z_state = _recurrent_gdn_components(
        q,
        k,
        v,
        q_rot,
        k_rot,
        beta,
        decay,
        kv_state=kv_state,
        z_state=z_state,
    )

    num_b, den_b, _, _ = _recurrent_gdn_components(
        from_time(torch.flip(to_time(q_flat), dims=[2])),
        from_time(flip_and_shift(to_time(k_flat), dim=2, shift_val=0.0)),
        from_time(flip_and_shift(to_time(v_flat), dim=2, shift_val=0.0)),
        from_time(torch.flip(to_time(q_rot_flat), dims=[2])),
        from_time(flip_and_shift(to_time(k_rot_flat), dim=2, shift_val=0.0)),
        flip_and_shift(beta_flat, dim=2, shift_val=0.0),
        flip_and_shift(decay_flat, dim=2, shift_val=1.0),
    )
    num_b = torch.flip(num_b.view(b, heads, dim, frames, spatial), dims=[3]).reshape_as(num_f)
    den_b = torch.flip(den_b.view(b, heads, 1, frames, spatial), dims=[3]).reshape_as(den_f)
    out = (num_f + num_b) / (den_f + den_b + eps)
    return out, kv_state, z_state, decay[:, :, -1]


def recurrent_delta_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, heads, dim, n = q.shape
    frames = beta.shape[2]
    spatial = n // frames
    q_flat, k_flat, v_flat = q, k, v
    beta_flat, decay_flat = beta, decay

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, heads, dim, frames, spatial).permute(0, 1, 3, 2, 4)

    def from_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(b, heads, dim, n)

    q, k, v = to_time(q), to_time(k), to_time(v)
    beta = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(b, heads, frames, 1, 1)
    decay = decay.view(b, heads, frames, 1, 1)
    if state is None:
        state = torch.zeros(b, heads, dim, dim, device=q.device, dtype=q.dtype)
    else:
        state = state.to(device=q.device, dtype=q.dtype)
    outs = []
    for frame in range(frames):
        qt, kt, vt = q[:, :, frame], k[:, :, frame], v[:, :, frame]
        state = state * decay[:, :, frame]
        delta_v = (vt - torch.matmul(state, kt)) * beta[:, :, frame]
        state = state + torch.matmul(delta_v, kt.transpose(-1, -2))
        outs.append(torch.matmul(state, qt))
    out_f = torch.stack(outs, dim=2).permute(0, 1, 3, 2, 4).reshape(b, heads, dim, n)
    out_b = recurrent_delta(
        from_time(torch.flip(to_time(q_flat), dims=[2])),
        from_time(flip_and_shift(to_time(k_flat), dim=2, shift_val=0.0)),
        from_time(flip_and_shift(to_time(v_flat), dim=2, shift_val=0.0)),
        flip_and_shift(beta_flat, dim=2, shift_val=0.0),
        flip_and_shift(decay_flat, dim=2, shift_val=1.0),
    )
    out_b = torch.flip(out_b.view(b, heads, dim, frames, spatial), dims=[3]).reshape_as(out_f)
    out = out_f + out_b
    return out, state


class SanaWMGDNAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        cam_dim: int,
        cam_heads: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-15,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.k_conv_only = k_conv_only
        self.conv_kernel_size = conv_kernel_size
        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
        self.proj = nn.Linear(out_dim, out_dim)
        self.q_norm = RMSNorm(in_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(in_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True)
        if conv_kernel_size > 0:
            self.conv_k = ShortConvolution(out_dim, conv_kernel_size)
            self.conv_q = None if k_conv_only else ShortConvolution(out_dim, conv_kernel_size)
            self.conv_v = None if k_conv_only else ShortConvolution(out_dim, conv_kernel_size)
        else:
            self.conv_q = self.conv_k = self.conv_v = None
        self._init_gates()
        if cam_dim != in_dim or cam_heads != self.heads:
            raise ValueError("SANA-WM UCPE uses shared dimensions between main and camera branches")
        self.patch_size = patch_size
        self.cam_dim = cam_dim
        self.cam_heads = cam_heads
        self.cam_head_dim = cam_dim // cam_heads
        self.q_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.k_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.v_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.out_proj_cam = nn.Linear(cam_dim, out_dim, bias=True)
        self.q_norm_cam = deepcopy(self.q_norm)
        self.k_norm_cam = deepcopy(self.k_norm)
        nn.init.zeros_(self.out_proj_cam.weight)
        nn.init.zeros_(self.out_proj_cam.bias)
        if conv_kernel_size > 0:
            self.conv_k_cam = ShortConvolution(cam_dim, conv_kernel_size)
            self.conv_q_cam = None if k_conv_only else ShortConvolution(cam_dim, conv_kernel_size)
            self.conv_v_cam = None if k_conv_only else ShortConvolution(cam_dim, conv_kernel_size)
            for conv in (self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
                if conv is not None:
                    conv.weight.data.zero_()
                    conv.weight.data[:, 0, -1] = 1.0
        else:
            self.conv_q_cam = self.conv_k_cam = self.conv_v_cam = None

    def _init_gates(self) -> None:
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.constant_(self.dt_bias, -5.0)
        nn.init.constant_(self.A_log, 0.0)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.constant_(self.output_gate.bias, OUTPUT_GATE_INIT_BIAS)
        for conv in (self.conv_q, self.conv_k, self.conv_v):
            if conv is not None:
                conv.weight.data.zero_()
                conv.weight.data[:, 0, -1] = 1.0

    def _reshape_to_temporal(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, int, int, int]:
        b, _, c = x.shape
        frames, h, w = HW
        spatial = h * w
        return x.reshape(b, frames, spatial, c).permute(0, 2, 1, 3).reshape(b * spatial, frames, c), b, spatial, frames

    @staticmethod
    def _reshape_from_temporal(x: torch.Tensor, b: int, spatial: int, frames: int) -> torch.Tensor:
        c = x.shape[-1]
        return x.reshape(b, spatial, frames, c).permute(0, 2, 1, 3).reshape(b, frames * spatial, c)

    def _causal_conv(self, x: torch.Tensor, conv: ShortConvolution) -> torch.Tensor:
        y, _ = conv(x)
        return y.to(x.dtype)

    def _temporal_conv(self, x: torch.Tensor, conv: ShortConvolution, HW: tuple[int, int, int]) -> torch.Tensor:
        x, b, spatial, frames = self._reshape_to_temporal(x, HW)
        y_fwd = self._causal_conv(x, conv)
        y_bwd = self._causal_conv(x.flip(1), conv).flip(1)
        center = x * conv.weight[:, 0, -1].to(x.dtype).view(1, 1, -1)
        return self._reshape_from_temporal(y_fwd + y_bwd - center, b, spatial, frames)

    def _temporal_conv_cached(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
    ) -> torch.Tensor:
        x_temporal, b, spatial, frames = self._reshape_to_temporal(x, HW)
        pad = conv.weight.shape[-1] - 1
        if kv_cache[_SLOT_SHORTCONV] is not None:
            prefix = kv_cache[_SLOT_SHORTCONV].to(device=x_temporal.device, dtype=x_temporal.dtype)
            conv_in = torch.cat([prefix, x_temporal], dim=1)
            y_fwd = self._causal_conv(conv_in, conv)[:, -frames:]
        else:
            y_fwd = self._causal_conv(x_temporal, conv)
        y_bwd = self._causal_conv(x_temporal.flip(1), conv).flip(1)
        center = x_temporal * conv.weight[:, 0, -1].to(x_temporal.dtype).view(1, 1, -1)
        y = y_fwd + y_bwd - center
        if save_kv_cache and pad > 0:
            kv_cache[_SLOT_SHORTCONV] = x_temporal[:, -pad:].detach().clone()
        return self._reshape_from_temporal(y, b, spatial, frames)

    def _compute_frame_gates(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, c = x.shape
        frames, h, w = HW
        spatial = h * w
        beta = self.beta_proj(x).sigmoid().reshape(b, frames, spatial, self.heads).permute(0, 3, 1, 2)
        pooled = x.reshape(b, frames, spatial, c).mean(dim=2)
        decay = (-self.A_log.float().exp().view(1, 1, -1) * F.softplus(self.gate_proj(pooled).float() + self.dt_bias.float().view(1, 1, -1))).exp()
        return beta, decay.transpose(1, 2)

    def _apply_output_gate(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return out * F.silu(self.output_gate(x).float())

    def _cam_linear_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        return F.linear(x, qkv_w, qkv_b).chunk(3, dim=-1)

    def _qkv(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._temporal_conv(q.reshape(b, n, c), self.conv_q, HW).reshape(b, n, self.heads, self.dim)
            k = self._temporal_conv(k.reshape(b, n, c), self.conv_k, HW).reshape(b, n, self.heads, self.dim)
            if self.conv_v is not None:
                v = self._temporal_conv(v.reshape(b, n, c), self.conv_v, HW).reshape(b, n, self.heads, self.dim)
        q = self.q_norm(q.reshape(b, n, c)).reshape(b, n, self.heads, self.dim)
        k = self.k_norm(k.reshape(b, n, c)).reshape(b, n, self.heads, self.dim)
        return F.relu(q), F.relu(k), v

    def _qkv_cached(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._temporal_conv(q.reshape(b, n, c), self.conv_q, HW).reshape(b, n, self.heads, self.dim)
            k = self._temporal_conv_cached(k.reshape(b, n, c), self.conv_k, HW, kv_cache, save_kv_cache).reshape(b, n, self.heads, self.dim)
            if self.conv_v is not None:
                v = self._temporal_conv(v.reshape(b, n, c), self.conv_v, HW).reshape(b, n, self.heads, self.dim)
        q = self.q_norm(q.reshape(b, n, c)).reshape(b, n, self.heads, self.dim)
        k = self.k_norm(k.reshape(b, n, c)).reshape(b, n, self.heads, self.dim)
        return F.relu(q), F.relu(k), v

    def _bidirectional_main(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        apply_output_gate: bool = True,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_: object,
    ) -> torch.Tensor:
        b, n, c = x.shape
        frames, h, w = HW
        q, k, v = self._qkv(x, HW)
        k = k * ((self.dim**-0.5) * ((h * w) ** -0.5))
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        q_rot = apply_rotary(q, rotary_emb) if rotary_emb is not None else q
        k_rot = apply_rotary(k, rotary_emb) if rotary_emb is not None else k
        beta, decay = precomputed_gates or self._compute_frame_gates(x, HW)
        qf, kf, vf = q.float(), k.float(), v.float()
        qrf, krf = q_rot.float(), k_rot.float()
        beta, decay = beta.float(), decay.float()
        num_f, den_f = recurrent_gdn(qf, kf, vf, qrf, krf, beta, decay, eps=self.eps, return_components=True)

        def to_time(tensor: torch.Tensor, dim_actual: int) -> torch.Tensor:
            return tensor.view(b, self.heads, dim_actual, frames, h * w).permute(0, 1, 3, 2, 4)

        def from_time(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(b, self.heads, tensor.shape[3], n)

        num_b, den_b = recurrent_gdn(
            from_time(torch.flip(to_time(qf, self.dim), dims=[2])),
            from_time(flip_and_shift(to_time(kf, self.dim), dim=2, shift_val=0.0)),
            from_time(flip_and_shift(to_time(vf, self.dim), dim=2, shift_val=0.0)),
            from_time(torch.flip(to_time(qrf, self.dim), dims=[2])),
            from_time(flip_and_shift(to_time(krf, self.dim), dim=2, shift_val=0.0)),
            flip_and_shift(beta, dim=2, shift_val=0.0),
            flip_and_shift(decay, dim=2, shift_val=1.0),
            eps=self.eps,
            return_components=True,
        )
        num_b = torch.flip(num_b.view(b, self.heads, self.dim, frames, h * w), dims=[3]).reshape_as(num_f)
        den_b = torch.flip(den_b.view(b, self.heads, 1, frames, h * w), dims=[3]).reshape_as(den_f)
        out = ((num_f + num_b) / (den_f + den_b + self.eps)).to(x.dtype).permute(0, 3, 1, 2).reshape(b, n, c)
        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out

    def _cam_qkv(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        q, k, v = self._cam_linear_qkv(x)
        if self.conv_q_cam is not None:
            q = self._temporal_conv(q, self.conv_q_cam, HW)
        if self.conv_k_cam is not None:
            k = self._temporal_conv(k, self.conv_k_cam, HW)
        if self.conv_v_cam is not None:
            v = self._temporal_conv(v, self.conv_v_cam, HW)
        q = self.q_norm_cam(q).reshape(b, n, self.cam_heads, self.cam_head_dim)
        k = self.k_norm_cam(k).reshape(b, n, self.cam_heads, self.cam_head_dim)
        v = v.reshape(b, n, self.cam_heads, self.cam_head_dim)
        return F.relu(q).permute(0, 2, 3, 1), F.relu(k).permute(0, 2, 3, 1), v.permute(0, 2, 3, 1)

    def _cam_qkv_raw(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        q, k, v = self._cam_linear_qkv(x)
        if self.conv_q_cam is not None:
            q = self._temporal_conv(q, self.conv_q_cam, HW)
        if self.conv_k_cam is not None:
            k = self._temporal_conv(k, self.conv_k_cam, HW)
        if self.conv_v_cam is not None:
            v = self._temporal_conv(v, self.conv_v_cam, HW)
        return (
            q.reshape(b, n, self.cam_heads, self.cam_head_dim).contiguous(),
            k.reshape(b, n, self.cam_heads, self.cam_head_dim).contiguous(),
            v.reshape(b, n, self.cam_heads, self.cam_head_dim).contiguous(),
        )

    def _cam_prep_fused_order(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        HW: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        b, n, heads, dim = q_raw.shape
        _, h, w = HW
        half = dim // 2
        groups = half // 4
        dtype = q_raw.dtype
        raymats, _ = process_camera_conditions_ucpe(camera_conditions, HW, self.patch_size)
        p = raymats.reshape(b, n, 4, 4)
        p_t = p.transpose(-1, -2).float()
        p_inv = invert_se3(p).float()
        q_inv = torch.rsqrt(q_raw.float().square().sum(dim=(-1, -2)) / (heads * dim) + self.q_norm_cam.eps)
        k_inv = torch.rsqrt(k_raw.float().square().sum(dim=(-1, -2)) / (heads * dim) + self.k_norm_cam.eps)
        q_w = self.q_norm_cam.weight.float().reshape(heads, dim)
        k_w = self.k_norm_cam.weight.float().reshape(heads, dim)
        k_scale = (dim**-0.5) * ((h * w) ** -0.5)
        q = (q_raw.float() * q_inv[:, :, None, None] * q_w[None, None]).clamp_min(0.0)
        k = (k_raw.float() * k_inv[:, :, None, None] * k_w[None, None]).clamp_min(0.0) * k_scale
        v = v_raw.float()
        q_half = q[..., :half].reshape(b, n, heads, groups, 4)
        k_half = k[..., :half].reshape(b, n, heads, groups, 4)
        v_half = v[..., :half].reshape(b, n, heads, groups, 4)

        def apply_4x4(mat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
            rows = []
            for i in range(4):
                row = tensor[..., 0] * mat[:, :, None, None, i, 0]
                row = row + tensor[..., 1] * mat[:, :, None, None, i, 1]
                row = row + tensor[..., 2] * mat[:, :, None, None, i, 2]
                row = row + tensor[..., 3] * mat[:, :, None, None, i, 3]
                rows.append(row)
            return torch.stack(rows, dim=-1)

        q_half_out = apply_4x4(p_t, q_half).reshape(b, n, heads, half)
        k_half_out = apply_4x4(p_inv, k_half).reshape(b, n, heads, half)
        v_half_out = apply_4x4(p_inv, v_half).reshape(b, n, heads, half)
        rotary_cur = _slice_rope_to_current_chunk(rotary_emb, n)
        rope = _slice_cam_rope(rotary_cur, dim)
        if rope is None:
            cos = torch.ones(n, half, device=q_raw.device, dtype=torch.float32)
            sin = torch.zeros(n, half, device=q_raw.device, dtype=torch.float32)
        else:
            freqs = rope.squeeze(0).squeeze(0)
            cos = freqs.real.float().repeat_interleave(2, dim=-1)
            sin_half = freqs.imag.float()
            sin = torch.stack((-sin_half, sin_half), dim=-1).reshape(n, half)
        pair = torch.arange(half, device=q_raw.device) ^ 1

        def rope_second(tensor: torch.Tensor) -> torch.Tensor:
            second = tensor[..., half:]
            return second * cos[None, :, None, :] + second[..., pair] * sin[None, :, None, :]

        q_post = torch.cat((q_half_out, rope_second(q)), dim=-1)
        k_post = torch.cat((k_half_out, rope_second(k)), dim=-1)
        v_post = torch.cat((v_half_out, rope_second(v)), dim=-1)
        q_out = q_post.permute(0, 2, 3, 1).contiguous().to(dtype)
        k_out = k_post.permute(0, 2, 3, 1).contiguous().to(dtype)
        v_out = v_post.permute(0, 2, 3, 1).contiguous().to(dtype)
        k_pre_sq = k.square().sum(dim=-1).permute(0, 2, 1)
        k_post_sq = k_post.square().sum(dim=-1).permute(0, 2, 1)
        inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)
        _, _, apply_out = prepare_prope_fns(self.cam_head_dim, camera_conditions, HW, self.patch_size, rotary_cur, raymats=raymats)
        return q_out, k_out, v_out, inflation_sq, apply_out

    def _cached_main(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
        gates: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        b, n, c = x.shape
        _, h, w = HW
        q, k, v = self._qkv_cached(x, HW, kv_cache, save_kv_cache)
        debug_prefix = _stream_debug_prefix(self, "attn")
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_main_q", q.permute(0, 2, 3, 1))
            _stream_debug_dump_once(self, f"{debug_prefix}_main_k_unscaled", k.permute(0, 2, 3, 1))
            _stream_debug_dump_once(self, f"{debug_prefix}_main_v", v.permute(0, 2, 3, 1))
        k = k * ((self.dim**-0.5) * ((h * w) ** -0.5))
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_main_k", k)
        rotary_emb = _slice_rope_to_current_chunk(rotary_emb, n)
        q_rot = apply_rotary(q, rotary_emb) if rotary_emb is not None else q
        k_rot = apply_rotary(k, rotary_emb) if rotary_emb is not None else k
        beta, decay = gates
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_main_q_rot", q_rot)
            _stream_debug_dump_once(self, f"{debug_prefix}_main_k_rot", k_rot)
            _stream_debug_dump_once(self, f"{debug_prefix}_main_beta", beta)
            _stream_debug_dump_once(self, f"{debug_prefix}_main_decay", decay)
        out, kv_state, z_state, _ = recurrent_gdn_cached(
            q.float(),
            k.float(),
            v.float(),
            q_rot.float(),
            k_rot.float(),
            beta.float(),
            decay.float(),
            eps=self.eps,
            kv_state=kv_cache[_SLOT_K],
            z_state=kv_cache[_SLOT_V],
        )
        if save_kv_cache:
            kv_cache[_SLOT_K] = kv_state.detach().clone()
            kv_cache[_SLOT_V] = z_state.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor([_CACHE_TYPE_STATE], device=x.device)
        return out.to(x.dtype).permute(0, 3, 1, 2).reshape(b, n, c), kv_cache

    def _camera_branch(self, x: torch.Tensor, HW: tuple[int, int, int], camera_conditions: torch.Tensor, rotary_emb: torch.Tensor | None, precomputed_gates: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        b, n, _ = x.shape
        _, h, w = HW
        q, k, v = self._cam_qkv(x, HW)
        k = k * ((self.cam_head_dim**-0.5) * ((h * w) ** -0.5))
        pre_k_norm = torch.linalg.vector_norm(k, dim=2, keepdim=True).clamp_min(1e-6)
        apply_q, apply_kv, apply_out = prepare_prope_fns(self.cam_head_dim, camera_conditions, HW, self.patch_size, rotary_emb)
        q_t = apply_q(q.transpose(-1, -2)).transpose(-1, -2).contiguous()
        kv_t = apply_kv(torch.cat([k, v], dim=1).transpose(-1, -2)).transpose(-1, -2).contiguous()
        k_t, v_t = torch.chunk(kv_t, 2, dim=1)
        post_k_norm = torch.linalg.vector_norm(k_t, dim=2, keepdim=True).clamp_min(1e-6)
        inflation_sq = (post_k_norm / pre_k_norm) ** 2
        beta, decay = precomputed_gates
        frame_inflation_sq = inflation_sq.view(b, self.cam_heads, HW[0], h * w).mean(dim=-1)
        beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)
        out_f = recurrent_delta(q_t.float(), k_t.float(), v_t.float(), beta.float(), decay.float())

        def to_time(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(b, self.cam_heads, self.cam_head_dim, HW[0], h * w).permute(0, 1, 3, 2, 4)

        def from_time(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(b, self.cam_heads, self.cam_head_dim, n)

        out_b = recurrent_delta(
            from_time(torch.flip(to_time(q_t.float()), dims=[2])),
            from_time(flip_and_shift(to_time(k_t.float()), dim=2, shift_val=0.0)),
            from_time(flip_and_shift(to_time(v_t.float()), dim=2, shift_val=0.0)),
            flip_and_shift(beta.float(), dim=2, shift_val=0.0),
            flip_and_shift(decay.float(), dim=2, shift_val=1.0),
        )
        out_b = torch.flip(out_b.view(b, self.cam_heads, self.cam_head_dim, HW[0], h * w), dims=[3]).reshape_as(out_f)
        out = (out_f + out_b).to(x.dtype)
        out = apply_out(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        return out.reshape(b, self.cam_dim, n).permute(0, 2, 1)

    def _cached_camera_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
        gates: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, n, _ = x.shape
        _, h, w = HW
        q_raw, k_raw, v_raw = self._cam_qkv_raw(x, HW)
        q_t, k_t, v_t, inflation_sq, apply_out = self._cam_prep_fused_order(q_raw, k_raw, v_raw, camera_conditions, rotary_emb, HW)
        debug_prefix = _stream_debug_prefix(self, "attn")
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_q_trans", q_t)
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_k_trans", k_t)
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_v_trans", v_t)
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_inflation_sq", inflation_sq)
        beta, decay = gates
        frame_inflation_sq = inflation_sq.view(b, self.cam_heads, HW[0], h * w).mean(dim=-1)
        beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_beta_adj", beta)
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_decay", decay)
        out, cam_state = recurrent_delta_cached(
            q_t.float(),
            k_t.float(),
            v_t.float(),
            beta.float(),
            decay.float(),
            state=kv_cache[_SLOT_CAM_K],
        )
        if save_kv_cache:
            kv_cache[_SLOT_CAM_K] = cam_state.detach().clone()
        out = out.to(x.dtype)
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_scan_out", out)
        out = apply_out(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_cam_apply_out", out)
        return out.reshape(b, self.cam_dim, n).permute(0, 2, 1)

    def forward(self, x: torch.Tensor, HW: tuple[int, int, int], rotary_emb: torch.Tensor | None, camera_conditions: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if camera_conditions is None:
            raise ValueError("SANA-WM inference requires camera_conditions")
        kv_cache = kwargs.get("kv_cache")
        save_kv_cache = bool(kwargs.get("save_kv_cache", False))
        gates = self._compute_frame_gates(x, HW)
        if kv_cache is not None:
            main, kv_cache = self._cached_main(x, HW, rotary_emb, kv_cache, save_kv_cache, gates)
            cam_raw = self._cached_camera_branch(x, HW, camera_conditions, rotary_emb, kv_cache, save_kv_cache, gates)
            cam = self.out_proj_cam(cam_raw)
            debug_prefix = _stream_debug_prefix(self, "attn")
            if debug_prefix is not None:
                _stream_debug_dump_once(self, f"{debug_prefix}_main_raw", main)
                _stream_debug_dump_once(self, f"{debug_prefix}_cam_raw", cam_raw)
                _stream_debug_dump_once(self, f"{debug_prefix}_cam_proj", cam)
            out = self._apply_output_gate(main + cam, x)
            proj = self.proj(out.to(self.proj.weight.dtype))
            if debug_prefix is not None:
                _stream_debug_dump_once(self, f"{debug_prefix}_after_gate", out)
                _stream_debug_dump_once(self, f"{debug_prefix}_proj", proj)
            return proj, kv_cache
        main = self._bidirectional_main(x, HW=HW, rotary_emb=rotary_emb, apply_output_gate=False, precomputed_gates=gates)
        cam = self.out_proj_cam(self._camera_branch(x, HW, camera_conditions, rotary_emb, gates))
        out = self._apply_output_gate(main + cam, x)
        return self.proj(out.to(self.proj.weight.dtype))


class SanaWMSoftmaxAttention(SanaWMGDNAttention):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["conv_kernel_size"] = 0
        super().__init__(*args, **kwargs)

    def _softmax_main(self, x: torch.Tensor, HW: tuple[int, int, int], rotary_emb: torch.Tensor | None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q.reshape(b, n, c)).reshape(b, n, self.heads, self.dim).permute(0, 2, 1, 3)
        k = self.k_norm(k.reshape(b, n, c)).reshape(b, n, self.heads, self.dim).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if rotary_emb is not None:
            q = apply_rotary(q.transpose(-1, -2), rotary_emb).transpose(-1, -2)
            k = apply_rotary(k.transpose(-1, -2), rotary_emb).transpose(-1, -2)
        out = sdpa_with_head_padding(q, k, v)
        return out.transpose(1, 2).reshape(b, n, c)

    def _softmax_main_cached(
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q.reshape(b, n, c)).reshape(b, n, self.heads, self.dim).permute(0, 2, 3, 1)
        k = self.k_norm(k.reshape(b, n, c)).reshape(b, n, self.heads, self.dim).permute(0, 2, 3, 1)
        rotary_emb = _slice_rope_to_current_chunk(rotary_emb, n)
        if rotary_emb is not None:
            q = apply_rotary(q, rotary_emb)
            k = apply_rotary(k, rotary_emb)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.permute(0, 2, 1, 3)
        cached_k = kv_cache[_SLOT_K]
        cached_v = kv_cache[_SLOT_V]
        if save_kv_cache:
            kv_cache[_SLOT_K] = k.detach().clone()
            kv_cache[_SLOT_V] = v.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor([_CACHE_TYPE_CONCAT], device=x.device)
        if cached_k is not None:
            k = torch.cat([cached_k.to(k.dtype), k], dim=2)
            v = torch.cat([cached_v.to(v.dtype), v], dim=2)
        out = sdpa_with_head_padding(q, k, v)
        return out.transpose(1, 2).reshape(b, n, c), kv_cache

    def _camera_branch(self, x: torch.Tensor, HW: tuple[int, int, int], camera_conditions: torch.Tensor, rotary_emb: torch.Tensor | None, precomputed_gates: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        del precomputed_gates
        b, n, _ = x.shape
        q, k, v = self._cam_qkv_softmax(x, HW)
        apply_q, apply_kv, apply_out = prepare_prope_fns(self.cam_head_dim, camera_conditions, HW, self.patch_size, rotary_emb)
        q_t = apply_q(q.transpose(-1, -2)).transpose(-1, -2).contiguous()
        kv_t = apply_kv(torch.cat([k, v], dim=1).transpose(-1, -2)).transpose(-1, -2).contiguous()
        k_t, v_t = torch.chunk(kv_t, 2, dim=1)
        out = sdpa_with_head_padding(q_t.transpose(-1, -2), k_t.transpose(-1, -2), v_t.transpose(-1, -2))
        out = apply_out(out).transpose(-1, -2).contiguous()
        return out.reshape(b, self.cam_dim, n).permute(0, 2, 1)

    def _cam_qkv_softmax(self, x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        q, k, v = self._cam_linear_qkv(x)
        if self.conv_q_cam is not None:
            q = self._temporal_conv(q, self.conv_q_cam, HW)
        if self.conv_k_cam is not None:
            k = self._temporal_conv(k, self.conv_k_cam, HW)
        if self.conv_v_cam is not None:
            v = self._temporal_conv(v, self.conv_v_cam, HW)
        q = self.q_norm_cam(q).reshape(b, n, self.cam_heads, self.cam_head_dim)
        k = self.k_norm_cam(k).reshape(b, n, self.cam_heads, self.cam_head_dim)
        v = v.reshape(b, n, self.cam_heads, self.cam_head_dim)
        return q.permute(0, 2, 3, 1), k.permute(0, 2, 3, 1), v.permute(0, 2, 3, 1)

    def _camera_branch_softmax_cached(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list[torch.Tensor | None],
        save_kv_cache: bool,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._cam_qkv_softmax(x, HW)
        rotary_emb = _slice_rope_to_current_chunk(rotary_emb, n)
        apply_q, apply_kv, apply_out = prepare_prope_fns(self.cam_head_dim, camera_conditions, HW, self.patch_size, rotary_emb)
        q_t = apply_q(q.transpose(-1, -2)).transpose(-1, -2).contiguous()
        kv_t = apply_kv(torch.cat([k, v], dim=1).transpose(-1, -2)).transpose(-1, -2).contiguous()
        k_t, v_t = torch.chunk(kv_t, 2, dim=1)
        q_sdpa = q_t.transpose(-1, -2)
        k_sdpa = k_t.transpose(-1, -2)
        v_sdpa = v_t.transpose(-1, -2)
        cached_cam_k = kv_cache[_SLOT_CAM_K]
        cached_cam_v = kv_cache[_SLOT_CAM_V]
        if save_kv_cache:
            kv_cache[_SLOT_CAM_K] = k_sdpa.detach().clone()
            kv_cache[_SLOT_CAM_V] = v_sdpa.detach().clone()
        if cached_cam_k is not None:
            k_sdpa = torch.cat([cached_cam_k.to(k_sdpa.dtype), k_sdpa], dim=2)
            v_sdpa = torch.cat([cached_cam_v.to(v_sdpa.dtype), v_sdpa], dim=2)
        out = sdpa_with_head_padding(q_sdpa, k_sdpa, v_sdpa)
        out = apply_out(out).transpose(-1, -2).contiguous()
        return out.reshape(b, self.cam_dim, n).permute(0, 2, 1)

    def forward(self, x: torch.Tensor, HW: tuple[int, int, int], rotary_emb: torch.Tensor | None, camera_conditions: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if camera_conditions is None:
            raise ValueError("SANA-WM inference requires camera_conditions")
        kv_cache = kwargs.get("kv_cache")
        save_kv_cache = bool(kwargs.get("save_kv_cache", False))
        if kv_cache is not None:
            main, kv_cache = self._softmax_main_cached(x, rotary_emb, kv_cache, save_kv_cache)
            cam_raw = self._camera_branch_softmax_cached(x, HW, camera_conditions, rotary_emb, kv_cache, save_kv_cache)
            cam = self.out_proj_cam(cam_raw)
            out = self._apply_output_gate(main + cam, x)
            proj = self.proj(out.to(self.proj.weight.dtype))
            debug_prefix = _stream_debug_prefix(self, "attn")
            if debug_prefix is not None:
                _stream_debug_dump_once(self, f"{debug_prefix}_main_raw", main)
                _stream_debug_dump_once(self, f"{debug_prefix}_cam_raw", cam_raw)
                _stream_debug_dump_once(self, f"{debug_prefix}_cam_proj", cam)
                _stream_debug_dump_once(self, f"{debug_prefix}_after_gate", out)
                _stream_debug_dump_once(self, f"{debug_prefix}_proj", proj)
            return proj, kv_cache
        main = self._softmax_main(x, HW, rotary_emb)
        cam = self.out_proj_cam(self._camera_branch(x, HW, camera_conditions, rotary_emb, self._compute_frame_gates(x, HW)))
        out = self._apply_output_gate(main + cam, x)
        return self.proj(out.to(self.proj.weight.dtype))


def to_tuple(value: int | tuple[int, ...], length: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return (value,) * length


def same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    return kernel_size // 2


def build_act(name: str | None, *, inplace: bool = True) -> nn.Module | None:
    if name is None or name.lower() == "none":
        return None
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    raise ValueError(f"unsupported activation: {name}")


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale_factor: float = 1.0, eps: float = 1e-6, norm_dim: int = -1) -> None:
        super().__init__()
        self.eps = eps
        self.norm_dim = norm_dim
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_shape = [1] * x.ndim
        weight_shape[self.norm_dim] = -1
        weight = self.weight.view(*weight_shape)
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(self.norm_dim, keepdim=True) + self.eps)
        return (weight * y).type_as(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[[], nn.Module],
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
        use_bias: bool = False,
        norm: str | None = None,
        act: str | None = "relu",
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = same_padding(kernel_size)
            if isinstance(padding, int):
                padding *= dilation
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        if norm is not None:
            raise ValueError("mini_sanawm only supports norm=None in GLUMBConv")
        self.norm = None
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_bias: bool | tuple[bool, bool, bool] = False,
        norm: tuple[str | None, str | None, str | None] = (None, None, None),
        act: tuple[str | None, str | None, str | None] = ("silu", "silu", None),
    ) -> None:
        super().__init__()
        out_feature = out_feature or in_features
        use_bias = to_tuple(use_bias, 3) if not isinstance(use_bias, tuple) else use_bias
        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(in_features, hidden_features * 2, 1, use_bias=use_bias[0], norm=norm[0], act=act[0])
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
        )
        self.point_conv = ConvLayer(hidden_features, out_feature, 1, use_bias=use_bias[2], norm=norm[2], act=act[2])

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        a, g = torch.chunk(x, 2, dim=1)
        return self.point_conv(a * self.glu_act(g))

    def forward(self, x: torch.Tensor, HW: tuple[int, ...] | None = None) -> torch.Tensor:
        b, n, c = x.shape
        if HW is None:
            h = w = int(n**0.5)
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        elif len(HW) == 2:
            h, w = HW
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:
            t, h, w = HW
            x = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        x = self._apply_spatial(x)
        if HW is not None and len(HW) == 3:
            return x.reshape(b * t, c, h * w).permute(0, 2, 1).reshape(b, n, c)
        return x.reshape(b, c, n).permute(0, 2, 1)


class GLUMBConvTemp(GLUMBConv):
    def __init__(self, *args: object, t_kernel_size: int = 3, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        out_feature = kwargs.get("out_feature") or kwargs.get("in_features") or args[0]
        self.t_conv = nn.Conv2d(
            out_feature,
            out_feature,
            kernel_size=(t_kernel_size, 1),
            stride=1,
            padding=(t_kernel_size // 2, 0),
            bias=False,
        )
        nn.init.zeros_(self.t_conv.weight)

    def forward(self, x: torch.Tensor, HW: tuple[int, int, int] | None = None, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if HW is None or len(HW) != 3:
            raise ValueError("GLUMBConvTemp requires HW=(T,H,W)")
        kv_cache = kwargs.get("kv_cache")
        save_kv_cache = bool(kwargs.get("save_kv_cache", False))
        b, n, c = x.shape
        t, h, w = HW
        x = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        x = self._apply_spatial(x)
        debug_prefix = _stream_debug_prefix(self, "mlp")
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_spatial", x)
        x_t = x.view(b, t, c, h * w).permute(0, 2, 1, 3)
        if kv_cache is None:
            tconv_out = self.t_conv(x_t)
            if debug_prefix is not None:
                _stream_debug_dump_once(self, f"{debug_prefix}_tconv_in", x_t)
                _stream_debug_dump_once(self, f"{debug_prefix}_tconv_out", tconv_out)
            x_t = x_t + tconv_out
            if debug_prefix is not None:
                _stream_debug_dump_once(self, f"{debug_prefix}_out", x_t)
            return x_t.permute(0, 2, 3, 1).reshape(b, n, c)

        pad = self.t_conv.kernel_size[0] // 2
        conv_in = x_t
        padded_size = 0
        cache_in = kv_cache[_SLOT_FFN_TCONV]
        if kv_cache[_SLOT_FFN_TCONV] is not None:
            prefix = cache_in.to(device=x_t.device, dtype=x_t.dtype)
            conv_in = torch.cat([prefix[:, :, -pad:], x_t], dim=2)
            padded_size = conv_in.shape[2] - x_t.shape[2]
        if save_kv_cache and pad > 0:
            kv_cache[_SLOT_FFN_TCONV] = x_t[:, :, -pad:].detach().clone()
        tconv_full = self.t_conv(conv_in)
        tconv_out = tconv_full[:, :, padded_size:]
        if debug_prefix is not None:
            cache_dump = cache_in if cache_in is not None else torch.empty(0, device=x_t.device, dtype=x_t.dtype)
            _stream_debug_dump_once(self, f"{debug_prefix}_cache_in", cache_dump)
            _stream_debug_dump_once(self, f"{debug_prefix}_tconv_in", conv_in)
            _stream_debug_dump_once(self, f"{debug_prefix}_tconv_full", tconv_full)
            _stream_debug_dump_once(self, f"{debug_prefix}_tconv_out", tconv_out)
        x_t = x_t + tconv_out
        if debug_prefix is not None:
            _stream_debug_dump_once(self, f"{debug_prefix}_out", x_t)
        return x_t.permute(0, 2, 3, 1).reshape(b, n, c), kv_cache


class PatchEmbedMS3D(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_chans: int = 3,
        embed_dim: int = 768,
        kernel_size: tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        flatten: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_tuple(patch_size, 3)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.flatten = flatten
        if not padding and kernel_size[-1] % 2 > 0:
            padding = same_padding(kernel_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float,
        act_layer: Callable[[], nn.Module] | nn.Module = nn.GELU(approximate="tanh"),
        token_num: int = 300,
    ) -> None:
        super().__init__()
        self.y_proj = Mlp(in_channels, hidden_size, hidden_size, act_layer=act_layer, drop=0.0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        self.uncond_prob = uncond_prob

    def forward(self, caption: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        if (train and self.uncond_prob > 0) or force_drop_ids is not None:
            if force_drop_ids is None:
                drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            else:
                drop_ids = force_drop_ids == 1
            y_embedding = self.y_embedding[: caption.shape[-2]]
            caption = torch.where(drop_ids[:, None, None, None], y_embedding, caption)
        return self.y_proj(caption)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, qk_norm: bool = False, **_: object) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(0.0)
        self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q_norm(self.q_linear(x)).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(cond).view(b, -1, 2, c)
        k, v = kv.unbind(2)
        k = self.k_norm(k).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None and mask.ndim == 2:
            attn_mask = (1 - mask.to(q.dtype))[:, None, None] * -10000.0
            attn_mask = attn_mask.repeat(1, self.num_heads, 1, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(y))


class T2IFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int | tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        patch_size = to_tuple(patch_size, 2) if isinstance(patch_size, int) else patch_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 2:
            b, n, c = x.shape
            frames = t.shape[2]
            shift, scale = (self.scale_shift_table[None, None, :, :] + t.transpose(1, 2)).chunk(2, dim=-2)
            x = t2i_modulate(self.norm_final(x).reshape(b, frames, -1, c), shift, scale).reshape(b, n, c)
        else:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
            x = t2i_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_1d_rotary_pos_embed(dim: int, pos: int | np.ndarray, theta: float = 10000.0, freqs_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if isinstance(pos, int):
        pos = torch.arange(pos)
    elif isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: dim // 2] / dim))
    freqs = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        fhw_dim: Optional[tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        if fhw_dim is None:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim
        else:
            t_dim, h_dim, w_dim = fhw_dim
        self.freqs = torch.cat(
            [get_1d_rotary_pos_embed(dim, max_seq_len, theta) for dim in (t_dim, h_dim, w_dim)],
            dim=1,
        )

    def forward(
        self,
        fhw: tuple[int | tuple[int, int], int, int],
        device: torch.device,
        frame_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        f_spec, h, w = fhw
        if frame_index is not None:
            f_pos = frame_index.to(device=device, dtype=torch.long)
        elif isinstance(f_spec, tuple):
            f_pos = torch.arange(f_spec[0], f_spec[1], device=device, dtype=torch.long)
        else:
            f_pos = torch.arange(int(f_spec), device=device, dtype=torch.long)
        f = int(f_pos.numel())
        freqs = self.freqs.to(device).split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )
        freqs_f = freqs[0][f_pos].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        return torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, f * h * w, -1)


class LTX2DecoderCacheManager:
    def __init__(self) -> None:
        self.feat_map: list[object | None] = []
        self.cache_mode: bool | None = None

    def clear(self) -> None:
        self.feat_map = []
        self.cache_mode = None

    def validate_mode(self, causal: bool) -> None:
        if self.cache_mode is None:
            self.cache_mode = bool(causal)
        elif self.cache_mode != bool(causal):
            raise ValueError("cannot mix causal and non-causal streaming decoder cache modes")


def _conv_output_size(in_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return max((in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1, 0)


def _ltx2_causal_conv_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    causal: bool = True,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    time_kernel_size = self.kernel_size[0]
    cache_len = max(time_kernel_size - 1, 0)
    if hidden_states.shape[2] == 0:
        if causal and feat_cache is not None and feat_idx is not None and cache_len > 0:
            while len(feat_cache) <= feat_idx[0]:
                feat_cache.append(None)
            feat_idx[0] += 1
        batch = hidden_states.shape[0]
        out_h = _conv_output_size(
            hidden_states.shape[3], self.conv.kernel_size[1], self.conv.stride[1], self.conv.padding[1], self.conv.dilation[1]
        )
        out_w = _conv_output_size(
            hidden_states.shape[4], self.conv.kernel_size[2], self.conv.stride[2], self.conv.padding[2], self.conv.dilation[2]
        )
        return hidden_states.new_empty(batch, self.conv.out_channels, 0, out_h, out_w)

    if causal:
        if feat_cache is not None and feat_idx is not None and cache_len > 0:
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)
            prefix = feat_cache[idx]
            current_cache = hidden_states[:, :, -cache_len:, :, :].clone()
            if isinstance(prefix, torch.Tensor) and current_cache.shape[2] < cache_len:
                needed = cache_len - current_cache.shape[2]
                carry = prefix.to(hidden_states.device)[:, :, -needed:, :, :]
                if carry.shape[2] < needed:
                    fill = carry[:, :, :1, :, :].repeat((1, 1, needed - carry.shape[2], 1, 1))
                    carry = torch.cat([fill, carry], dim=2)
                current_cache = torch.cat([carry, current_cache], dim=2)
            if prefix is not None:
                prefix = prefix.to(hidden_states.device)
                if prefix.shape[2] > cache_len:
                    prefix = prefix[:, :, -cache_len:, :, :]
                if prefix.shape[2] < cache_len:
                    fill = prefix[:, :, :1, :, :].repeat((1, 1, cache_len - prefix.shape[2], 1, 1))
                    prefix = torch.cat([fill, prefix], dim=2)
            else:
                prefix = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))
            hidden_states = torch.cat([prefix, hidden_states], dim=2)
            feat_cache[idx] = current_cache
            feat_idx[0] += 1
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states], dim=2)
    else:
        pad = (time_kernel_size - 1) // 2
        pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad, 1, 1))
        pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad, 1, 1))
        hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)
    return self.conv(hidden_states)


def _ltx2_resnet_forward(
    self: nn.Module,
    inputs: torch.Tensor,
    temb: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    causal: bool = True,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    hidden_states = self.norm1(inputs)
    if self.scale_shift_table is not None:
        temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
        shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
        hidden_states = hidden_states * (1 + scale_1) + shift_1
    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.conv1(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.per_channel_scale1 is not None:
        spatial_shape = hidden_states.shape[-2:]
        spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
        hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]
    hidden_states = self.norm2(hidden_states)
    if self.scale_shift_table is not None:
        hidden_states = hidden_states * (1 + scale_2) + shift_2
    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.per_channel_scale2 is not None:
        spatial_shape = hidden_states.shape[-2:]
        spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
        hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]
    residual = inputs
    if self.norm3 is not None:
        residual = self.norm3(residual.movedim(1, -1)).movedim(-1, 1)
    if self.conv_shortcut is not None:
        residual = self.conv_shortcut(residual)
    return hidden_states + residual


def _ltx2_mid_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    causal: bool = True,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if self.time_embedder is not None:
        temb = self.time_embedder(
            timestep=temb.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=hidden_states.size(0),
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
    for resnet in self.resnets:
        hidden_states = resnet(hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    return hidden_states


def _ltx2_upsampler_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    causal: bool = True,
    feat_cache: list[object | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    batch_size, _, num_frames, height, width = hidden_states.shape
    trim_t = max(self.stride[0] - 1, 0)
    if trim_t > 0 and feat_cache is not None and feat_idx is not None:
        idx = feat_idx[0]
        while len(feat_cache) <= idx:
            feat_cache.append(None)
        state = feat_cache[idx]
        if not isinstance(state, dict):
            state = {"trim_applied": False}
        trim_start = trim_t if not state["trim_applied"] else 0
        state["trim_applied"] = True
        feat_cache[idx] = state
        feat_idx[0] += 1
    else:
        trim_start = trim_t

    if self.residual:
        residual = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
        residual = residual.repeat(1, repeats, 1, 1, 1)
        residual = residual[:, :, trim_start:]
    hidden_states = self.conv(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    hidden_states = hidden_states.reshape(
        batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
    )
    hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    hidden_states = hidden_states[:, :, trim_start:]
    return hidden_states + residual if self.residual else hidden_states


def _ltx2_up_block_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    causal: bool = True,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if self.conv_in is not None:
        hidden_states = self.conv_in(hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.time_embedder is not None:
        temb = self.time_embedder(
            timestep=temb.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=hidden_states.size(0),
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    for resnet in self.resnets:
        hidden_states = resnet(hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    return hidden_states


def _ltx2_decoder_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor | None = None,
    causal: bool | None = None,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    causal = causal or self.is_causal
    if feat_cache is not None and feat_idx is None:
        feat_idx = [0]
    hidden_states = self.conv_in(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.timestep_scale_multiplier is not None:
        temb = temb * self.timestep_scale_multiplier
    hidden_states = self.mid_block(hidden_states, temb, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    for up_block in self.up_blocks:
        hidden_states = up_block(hidden_states, temb, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    hidden_states = self.norm_out(hidden_states)
    if self.time_embedder is not None:
        temb = self.time_embedder(
            timestep=temb.flatten(),
            resolution=None,
            aspect_ratio=None,
            batch_size=hidden_states.size(0),
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
        temb = temb + self.scale_shift_table[None, ..., None, None, None]
        shift, scale = temb.unbind(dim=1)
        hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.conv_act(hidden_states)
    hidden_states = self.conv_out(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
    p = self.patch_size
    p_t = self.patch_size_t
    batch_size, _, num_frames, height, width = hidden_states.shape
    hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
    return hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _vae_clear_decoder_cache(self: nn.Module) -> None:
    self._decoder_cache.clear()


def _vae_decode_per_frame_with_cache(
    self: nn.Module,
    z: torch.Tensor,
    temb: torch.Tensor | None = None,
    causal: bool | None = None,
    reset_cache: bool = False,
):
    if self.use_slicing and z.shape[0] > 1:
        raise ValueError("decode_per_frame_with_cache does not support batch slicing; pass batch size 1 or disable slicing.")
    if z.shape[2] <= 0:
        raise ValueError("Input latent video must contain at least 1 frame.")
    causal = self.decoder.is_causal if causal is None else causal
    if reset_cache:
        self._decoder_cache.clear()
    self._decoder_cache.validate_mode(causal)
    for frame in range(z.shape[2]):
        z_chunk = z[:, :, frame : frame + 1]
        yield self.decoder(
            z_chunk,
            temb,
            causal=causal,
            feat_cache=self._decoder_cache.feat_map,
            feat_idx=[0],
        )


def enable_ltx2_streaming_cache(vae: nn.Module) -> nn.Module:
    """Install causal streaming decode support on diffusers' LTX-2 VAE instance."""
    vae._decoder_cache = LTX2DecoderCacheManager()
    vae._minimal_streaming_cache_enabled = True
    vae.clear_decoder_cache = types.MethodType(_vae_clear_decoder_cache, vae)
    vae.decode_per_frame_with_cache = types.MethodType(_vae_decode_per_frame_with_cache, vae)

    for module in vae.decoder.modules():
        name = module.__class__.__name__
        if name == "LTX2VideoCausalConv3d":
            module.forward = types.MethodType(_ltx2_causal_conv_forward, module)
        elif name == "LTX2VideoResnetBlock3d":
            module.forward = types.MethodType(_ltx2_resnet_forward, module)
        elif name == "LTX2VideoMidBlock3d":
            module.forward = types.MethodType(_ltx2_mid_forward, module)
        elif name == "LTX2VideoUpsampler3d":
            module.forward = types.MethodType(_ltx2_upsampler_forward, module)
        elif name == "LTX2VideoUpBlock3d":
            module.forward = types.MethodType(_ltx2_up_block_forward, module)
        elif name == "LTX2VideoDecoder3d":
            module.forward = types.MethodType(_ltx2_decoder_forward, module)
    return vae
