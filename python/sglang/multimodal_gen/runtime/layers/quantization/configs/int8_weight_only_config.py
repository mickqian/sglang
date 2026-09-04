# SPDX-License-Identifier: Apache-2.0
"""Shared config for serialized per-output-row INT8 weight-only linears."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase as DiffusionLinearBase,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    UnquantizedLinearMethod as DiffusionUnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.int8_weight_only import (
    Int8WeightOnlyLinearMethod,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod as SrtUnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config as SrtW8A8Int8Config
from sglang.srt.layers.quantization.w8a8_int8 import (
    W8A8Int8LinearMethod,
)


class Int8WeightOnlyConfig(QuantizationConfig):
    """Select exactly the checkpoint linears that share the rowwise INT8 ABI."""

    supports_srt_linear_layers = True
    normalizes_checkpoint_weights = True

    def __init__(self, layer_scale_dtypes: dict[str, torch.dtype]) -> None:
        super().__init__()
        if not layer_scale_dtypes:
            raise ValueError("INT8 weight-only checkpoints need at least one linear")
        self.layer_scale_dtypes = layer_scale_dtypes
        self.layer_prefixes = set(layer_scale_dtypes)
        self.selected: set[str] = set()

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Int8WeightOnlyConfig:
        raise ValueError(
            f"{cls.__name__} must be constructed from checkpoint tensor metadata"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, DiffusionLinearBase):
            unquantized_method = DiffusionUnquantizedLinearMethod
        elif isinstance(layer, SrtLinearBase):
            unquantized_method = SrtUnquantizedLinearMethod
        else:
            return None
        scale_dtype = self.layer_scale_dtypes.get(prefix)
        if scale_dtype is None:
            return unquantized_method()
        self.selected.add(prefix)
        return Int8WeightOnlyLinearMethod(scale_dtype)

    def retain_checkpoint_layers(self, include_weight: Callable[[str], bool]) -> None:
        self.layer_scale_dtypes = {
            prefix: dtype
            for prefix, dtype in self.layer_scale_dtypes.items()
            if include_weight(f"{prefix}.weight")
        }
        self.layer_prefixes = set(self.layer_scale_dtypes)

    def supports_cpu_weight_loading(self) -> bool:
        return True


class W8A8Int8Config(QuantizationConfig):
    """Select serialized W8A8 INT8 linears and reuse SRT's fused kernel."""

    normalizes_checkpoint_weights = True

    def __init__(self, layer_prefixes: set[str]) -> None:
        super().__init__()
        if not layer_prefixes:
            raise ValueError("W8A8 INT8 checkpoints need at least one linear")
        self.layer_prefixes = layer_prefixes
        self.selected: set[str] = set()
        self._srt_config = SrtW8A8Int8Config({"is_dynamic": True})

    @classmethod
    def get_name(cls) -> str:
        return "w8a8_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return SrtW8A8Int8Config.get_supported_act_dtypes()

    @classmethod
    def get_min_capability(cls) -> int:
        return SrtW8A8Int8Config.get_min_capability()

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W8A8Int8Config:
        raise ValueError(
            f"{cls.__name__} must be constructed from checkpoint tensor metadata"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, DiffusionLinearBase):
            return None
        if prefix not in self.layer_prefixes:
            return DiffusionUnquantizedLinearMethod()
        self.selected.add(prefix)
        return W8A8Int8LinearMethod(self._srt_config)

    def remap_checkpoint_prefixes(self, param_names_mapping: dict) -> None:
        map_name = get_param_names_mapping(param_names_mapping)
        self.layer_prefixes = {
            map_name(f"{prefix}.weight")[0].removesuffix(".weight")
            for prefix in self.layer_prefixes
        }

    def normalize_checkpoint_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            if name.endswith(".weight_scale_inv"):
                name = name.removesuffix("_inv")
            yield name, weight

    def supports_cpu_weight_loading(self) -> bool:
        return True


__all__ = ["Int8WeightOnlyConfig", "W8A8Int8Config"]
