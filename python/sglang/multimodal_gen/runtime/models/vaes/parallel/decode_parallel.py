import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)


def tensor_pad(x: torch.Tensor, len_to_pad: int, dim: int = -2):
    x = torch.cat(
        [
            x,
            torch.zeros(
                *x.shape[:dim],
                len_to_pad,
                *x.shape[dim + 1 :],
                dtype=x.dtype,
                device=x.device,
            ),
        ],
        dim=dim,
    )
    return x


def tensor_chunk(x: torch.Tensor, dim: int = -2, world_size: int = 1, rank: int = 0):
    if x is None:
        return None
    if world_size <= 1:
        return x
    len_to_padding = (int(math.ceil(x.shape[dim] / world_size)) * world_size) - x.shape[
        dim
    ]
    if len_to_padding != 0:
        x = tensor_pad(x, len_to_padding, dim=dim)
    return torch.chunk(x, world_size, dim=dim)[rank]


def split_for_parallel_encode(
    x: torch.Tensor, downsample_count: int, world_size: int, rank: int
):
    orig_height = x.shape[-2]
    expected_height = orig_height // (2**downsample_count)
    factor = world_size * (2**downsample_count)
    pad_h = (factor - orig_height % factor) % factor
    if pad_h:
        x = F.pad(x, (0, 0, 0, pad_h, 0, 0))
    expected_local_height = (orig_height + pad_h) // (2**downsample_count) // world_size
    x = tensor_chunk(x, dim=-2, world_size=world_size, rank=rank)
    return x, expected_height, expected_local_height


def ensure_local_height(x: torch.Tensor, expected_local_height: int | None):
    if expected_local_height is None:
        return x
    if x.shape[-2] < expected_local_height:
        pad = expected_local_height - x.shape[-2]
        return F.pad(x, (0, 0, 0, pad, 0, 0))
    if x.shape[-2] > expected_local_height:
        return x[..., :expected_local_height, :].contiguous()
    return x


def split_for_parallel_decode(
    x: torch.Tensor, upsample_count: int, world_size: int, rank: int
):
    expected_height = x.shape[-2] * (2**upsample_count)
    x = tensor_chunk(x, dim=-2, world_size=world_size, rank=rank)
    return x, expected_height


def maybe_contiguous_for_decode_gather(x: torch.Tensor) -> torch.Tensor:
    if (
        x.dim() == 5
        and hasattr(torch, "channels_last_3d")
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and not x.is_contiguous()
    ):
        return x.contiguous()
    return x


def _halo_memory_format(reference: torch.Tensor) -> torch.memory_format:
    if reference.dim() > 1 and reference.stride(1) == 1:
        if reference.dim() == 5 and hasattr(torch, "channels_last_3d"):
            return torch.channels_last_3d
        if reference.dim() == 4:
            return torch.channels_last
    return torch.contiguous_format


def gather_and_trim_height(x: torch.Tensor, expected_height: int | None):
    if expected_height is None:
        return x
    x = get_decode_parallel_group_coordinator().all_gather(
        maybe_contiguous_for_decode_gather(x), dim=-2
    )
    if x.shape[-2] != expected_height:
        x = x[..., :expected_height, :].contiguous()
    return x


def _ensure_recv_buf(
    recv_buf: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor:
    memory_format = _halo_memory_format(reference)
    if (
        recv_buf is None
        or recv_buf.shape != reference.shape
        or recv_buf.dtype != reference.dtype
        or recv_buf.device != reference.device
        or not recv_buf.is_contiguous(memory_format=memory_format)
    ):
        return torch.empty(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            memory_format=memory_format,
        )
    return recv_buf


def _fill_boundary_halo(
    recv_buf: torch.Tensor,
    x: torch.Tensor,
    height_halo_size: int,
    *,
    top: bool,
    mode: str,
) -> None:
    if mode == "zeros":
        recv_buf.zero_()
        return
    if mode == "replicate":
        row = x[..., :1, :] if top else x[..., -1:, :]
        recv_buf.copy_(row.expand_as(recv_buf))
        return
    if mode == "reflect":
        if top:
            recv_buf.copy_(x[..., 1 : height_halo_size + 1, :].flip(-2))
        else:
            recv_buf.copy_(x[..., -height_halo_size - 1 : -1, :].flip(-2))
        return
    raise ValueError(f"Unsupported halo boundary mode: {mode}")


def halo_exchange(
    x: torch.Tensor,
    height_halo_size: int = 1,
    recv_top_buf: torch.Tensor | None = None,
    recv_bottom_buf: torch.Tensor | None = None,
    boundary_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if height_halo_size == 0:
        return x, recv_top_buf, recv_bottom_buf

    decode_group = get_decode_parallel_group_coordinator()
    rank = get_decode_parallel_rank()
    world_size = get_decode_parallel_world_size()
    group = decode_group.device_group
    group_ranks = decode_group.ranks

    top_row_ref = x[..., :height_halo_size, :]
    bottom_row_ref = x[..., -height_halo_size:, :]

    recv_top_buf = _ensure_recv_buf(recv_top_buf, top_row_ref)
    recv_bottom_buf = _ensure_recv_buf(recv_bottom_buf, bottom_row_ref)

    p2p_ops = []

    if rank > 0:
        prev_rank = group_ranks[rank - 1]
        top_row = top_row_ref.contiguous(memory_format=_halo_memory_format(top_row_ref))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_top_buf, prev_rank, group))
        p2p_ops.append(dist.P2POp(dist.isend, top_row, prev_rank, group))
    if rank < world_size - 1:
        next_rank = group_ranks[rank + 1]
        bottom_row = bottom_row_ref.contiguous(
            memory_format=_halo_memory_format(bottom_row_ref)
        )
        p2p_ops.append(dist.P2POp(dist.isend, bottom_row, next_rank, group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_bottom_buf, next_rank, group))

    if rank == 0:
        _fill_boundary_halo(
            recv_top_buf, x, height_halo_size, top=True, mode=boundary_mode
        )
    if rank == world_size - 1:
        _fill_boundary_halo(
            recv_bottom_buf, x, height_halo_size, top=False, mode=boundary_mode
        )

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    return (
        torch.concat([recv_top_buf, x, recv_bottom_buf], dim=-2),
        recv_top_buf,
        recv_bottom_buf,
    )
