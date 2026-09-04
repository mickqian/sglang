# SPDX-License-Identifier: Apache-2.0
"""Runtime ABI shared by serialized rowwise INT8 weight-only formats."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


class Int8WeightOnlyLinearMethod(LinearMethodBase):
    """Keep INT8 weights packed and dequantize only the active matrix."""

    def __init__(self, scale_dtype: torch.dtype) -> None:
        self.scale_dtype = scale_dtype

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                1,
                dtype=self.scale_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"output_dim": 0})
        set_weight_attrs(weight_scale, extra_weight_attrs)
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.to(dtype=x.dtype)
        weight.mul_(layer.weight_scale.to(dtype=x.dtype))
        return F.linear(x, weight, bias)


__all__ = ["Int8WeightOnlyLinearMethod"]
