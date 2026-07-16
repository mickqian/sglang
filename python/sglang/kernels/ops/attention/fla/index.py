# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Sequence

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    seq_lens_cpu: Optional[Sequence[int]] = None,
) -> torch.LongTensor:
    # CPU lengths avoid synchronizing the CUDA cu_seqlens via .tolist().
    create_on_device = seq_lens_cpu is not None
    if seq_lens_cpu is None:
        seq_lens_cpu = prepare_lens(cu_seqlens).tolist()
    indices = torch.cat(
        [
            torch.arange(
                triton.cdiv(seq_len, chunk_size),
                device=cu_seqlens.device if create_on_device else None,
                dtype=cu_seqlens.dtype if create_on_device else None,
            )
            for seq_len in seq_lens_cpu
        ]
    )
    chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    return chunk_indices if create_on_device else chunk_indices.to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
