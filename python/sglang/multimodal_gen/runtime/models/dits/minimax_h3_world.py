# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import flex_attention

_ScoreMod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


@dataclass(frozen=True)
class MiniMaxH3WorldControlAttention:
    used: int
    score_mod: _ScoreMod


_WORLD_CONTROL_MASKS: dict[
    tuple[str, int],
    tuple[torch.Tensor, torch.Tensor, MiniMaxH3WorldControlAttention],
] = {}


@functools.cache
def _compiled_flex_attention():
    return torch.compile(
        flex_attention,
        dynamic=False,
        fullgraph=True,
        backend="inductor",
    )


def build_minimax_h3_world_control_attention(
    *,
    action_text_rows: torch.Tensor,
    video_start: int,
    frame_rows: int,
    used: int,
    device: torch.device,
) -> MiniMaxH3WorldControlAttention:
    if action_text_rows.ndim != 2 or action_text_rows.shape[1] != 2:
        raise ValueError("action_text_rows must have shape [video_latents, 2]")
    if video_start <= 0 or frame_rows <= 0 or used <= video_start:
        raise ValueError("invalid H3-World packed-sequence boundaries")

    key = (str(device), used)
    cached = _WORLD_CONTROL_MASKS.get(key)
    if cached is None:
        action_ids = torch.full((used,), -1, dtype=torch.int32, device=device)
        frame_ids = torch.full((used,), -1, dtype=torch.int32, device=device)

        def score_mod(score, _batch, _head, query, key_value):
            query_action = action_ids[query]
            key_action = action_ids[key_value]
            query_frame = frame_ids[query]
            key_frame = frame_ids[key_value]
            same_action = (
                (query_action >= 0) & (key_action >= 0) & (query_action == key_action)
            )
            frame_reads_action = (
                (query_frame >= 0) & (key_action >= 0) & (query_frame == key_action)
            )
            action_reads_other_frame = (
                (query_action >= 0) & (key_frame >= 0) & (query_action != key_frame)
            )
            action_leaks = (key_action >= 0) & ~same_action & ~frame_reads_action
            return torch.where(
                action_leaks | action_reads_other_frame,
                float("-inf"),
                score,
            )

        attention = MiniMaxH3WorldControlAttention(used=used, score_mod=score_mod)
        cached = (action_ids, frame_ids, attention)
        _WORLD_CONTROL_MASKS[key] = cached

    action_ids, frame_ids, attention = cached
    action_ids.fill_(-1)
    frame_ids.fill_(-1)
    rows = action_text_rows.to(device="cpu", dtype=torch.long).tolist()
    for index, (start, stop) in enumerate(rows):
        if not 0 <= start < stop <= video_start:
            raise ValueError(
                f"action text span {index}={(start, stop)!r} crosses the text segment"
            )
        action_ids[start:stop] = index

    video_stop = video_start + len(rows) * frame_rows
    if video_stop > used:
        raise ValueError("H3-World target video rows exceed the packed live sequence")
    target_rows = torch.arange(video_start, video_stop, device=device)
    frame_ids[video_start:video_stop] = ((target_rows - video_start) // frame_rows).to(
        torch.int32
    )
    return attention


def minimax_h3_world_control_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    control: MiniMaxH3WorldControlAttention,
    softmax_scale: float,
) -> torch.Tensor:
    used = control.used
    output = torch.zeros_like(query)
    live = _compiled_flex_attention()(
        query[:used].permute(1, 0, 2)[None],
        key[:used].permute(1, 0, 2)[None],
        value[:used].permute(1, 0, 2)[None],
        score_mod=control.score_mod,
        scale=softmax_scale,
    )
    output[:used] = live[0].permute(1, 0, 2)
    return output


__all__ = [
    "MiniMaxH3WorldControlAttention",
    "build_minimax_h3_world_control_attention",
    "minimax_h3_world_control_attention",
]
