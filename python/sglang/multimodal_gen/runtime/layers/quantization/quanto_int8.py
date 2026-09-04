# SPDX-License-Identifier: Apache-2.0
"""Checkpoint normalization for serialized Optimum Quanto qint8 weights."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch


def normalize_quanto_int8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Translate flattened Quanto tensors to native linear parameter names."""

    for name, tensor in weights:
        if name.endswith((".input_scale", ".output_scale")):
            if tensor.numel() != 1 or tensor.item() != 1:
                raise ValueError(f"Quanto weight-only scale {name!r} must equal 1")
            continue
        if name.endswith(".weight._data"):
            name = name.removesuffix("._data")
        elif name.endswith(".weight._scale"):
            name = name.removesuffix("._scale") + "_scale"
        yield name, tensor


__all__ = ["normalize_quanto_int8_weights"]
