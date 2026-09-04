# SPDX-License-Identifier: Apache-2.0
"""Admission for serialized TorchAO per-row INT8 weight-only checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.configs.int8_weight_only_config import (
    Int8WeightOnlyConfig,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

_QDATA_SUFFIX = "._weight_qdata"
_SCALE_SUFFIX = "._weight_scale"
_ZERO_POINT_SUFFIX = "._weight_zero_point"

ParameterNameMapper = Callable[[str], tuple[str, int | None, int | None]]


def _validate_int8_weight_only_declaration(config: dict[str, Any]) -> None:
    quant_type = config.get("quant_type")
    if not isinstance(quant_type, dict) or set(quant_type) != {"default"}:
        raise ValueError("TorchAO quant_type must contain exactly one 'default' rule")
    rule = quant_type["default"]
    if not isinstance(rule, dict):
        raise ValueError("TorchAO quant_type.default must be an object")
    if rule.get("_type") != "Int8WeightOnlyConfig" or rule.get("_version") != 2:
        raise ValueError(
            "Only TorchAO Int8WeightOnlyConfig serialization version 2 is supported"
        )
    data = rule.get("_data")
    if not isinstance(data, dict) or data.get("group_size") is not None:
        raise ValueError("TorchAO INT8 weight-only checkpoints must use group_size=null")
    granularity = data.get("granularity")
    if not isinstance(granularity, dict):
        raise ValueError("TorchAO INT8 weight-only granularity must be an object")
    if (
        granularity.get("_type") != "PerRow"
        or granularity.get("_version") != 1
        or granularity.get("_data") != {"dim": -1}
    ):
        raise ValueError("Only TorchAO PerRow(dim=-1) INT8 weights are supported")


class TorchAOInt8Config(Int8WeightOnlyConfig):
    """Run TorchAO's serialized symmetric per-row INT8 weights natively."""

    @classmethod
    def get_name(cls) -> str:
        return "torchao_int8"

    def normalize_checkpoint_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return normalize_torchao_int8_weights(weights)


def inspect_torchao_int8_checkpoint(
    model_config: dict[str, Any],
    safetensors_list: list[str],
    *,
    param_name_mapper: ParameterNameMapper | None = None,
) -> TorchAOInt8Config | None:
    """Validate the exact TorchAO ABI supported by the native runtime."""

    quant_spec = resolve_checkpoint_quant_spec(model_config)
    if quant_spec is None or quant_spec.declared_method != "torchao":
        return None
    _validate_int8_weight_only_declaration(quant_spec.config)
    if not safetensors_list:
        raise ValueError("TorchAO INT8 checkpoint does not contain safetensors weights")

    tensor_meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name in tensor_meta:
                    raise ValueError(f"Duplicate TorchAO checkpoint tensor {name!r}")
                tensor = checkpoint.get_slice(name)
                tensor_meta[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))

    data_prefixes = {
        name.removesuffix(_QDATA_SUFFIX)
        for name in tensor_meta
        if name.endswith(_QDATA_SUFFIX)
    }
    if not data_prefixes:
        raise ValueError("TorchAO checkpoint does not contain serialized INT8 weights")

    mapped_scale_dtypes: dict[str, torch.dtype] = {}
    mapped_sources: dict[str, set[int]] = {}
    mapped_counts: dict[str, int] = {}
    for prefix in sorted(data_prefixes):
        names = {
            "data": f"{prefix}{_QDATA_SUFFIX}",
            "scale": f"{prefix}{_SCALE_SUFFIX}",
            "zero_point": f"{prefix}{_ZERO_POINT_SUFFIX}",
        }
        missing = set(names.values()) - tensor_meta.keys()
        if missing:
            raise ValueError(
                f"TorchAO layer {prefix!r} is missing tensors: {sorted(missing)}"
            )
        if f"{prefix}.weight" in tensor_meta:
            raise ValueError(
                f"TorchAO layer {prefix!r} contains both packed and dense weights"
            )

        data_dtype, data_shape = tensor_meta[names["data"]]
        scale_dtype, scale_shape = tensor_meta[names["scale"]]
        zero_dtype, zero_shape = tensor_meta[names["zero_point"]]
        if data_dtype != "I8" or len(data_shape) != 2:
            raise ValueError(
                f"TorchAO layer {prefix!r} needs a 2D I8 weight, got "
                f"{data_dtype} {data_shape}"
            )
        expected_aux_shape = (data_shape[0], 1)
        if scale_dtype != "F32" or scale_shape != expected_aux_shape:
            raise ValueError(
                f"TorchAO layer {prefix!r} needs an F32 row scale shaped "
                f"{expected_aux_shape}, got {scale_dtype} {scale_shape}"
            )
        if zero_dtype != "I8" or zero_shape != expected_aux_shape:
            raise ValueError(
                f"TorchAO layer {prefix!r} needs an I8 zero point shaped "
                f"{expected_aux_shape}, got {zero_dtype} {zero_shape}"
            )

        source_weight = f"{prefix}.weight"
        mapped_weight, merge_index, merge_count = (
            param_name_mapper(source_weight)
            if param_name_mapper is not None
            else (source_weight, None, None)
        )
        if not mapped_weight.endswith(".weight"):
            raise ValueError(
                f"TorchAO parameter mapping produced invalid weight {mapped_weight!r}"
            )
        mapped_prefix = mapped_weight.removesuffix(".weight")
        if merge_index is None:
            if mapped_prefix in mapped_sources:
                raise ValueError(
                    f"TorchAO layers collide after mapping at {mapped_prefix!r}"
                )
            mapped_sources[mapped_prefix] = set()
        else:
            if merge_count is None or merge_index < 0 or merge_index >= merge_count:
                raise ValueError(
                    f"Invalid stacked mapping for TorchAO layer {prefix!r}"
                )
            previous_count = mapped_counts.setdefault(mapped_prefix, merge_count)
            if previous_count != merge_count:
                raise ValueError(
                    f"Conflicting stacked mapping for TorchAO layer {prefix!r}"
                )
            indices = mapped_sources.setdefault(mapped_prefix, set())
            if merge_index in indices:
                raise ValueError(
                    f"Duplicate stacked mapping for TorchAO layer {prefix!r}"
                )
            indices.add(merge_index)
        mapped_scale_dtypes[mapped_prefix] = torch.float32

    for mapped_prefix, merge_count in mapped_counts.items():
        indices = mapped_sources[mapped_prefix]
        if indices != set(range(merge_count)):
            raise ValueError(
                f"Incomplete stacked TorchAO layer {mapped_prefix!r}: "
                f"found indices {sorted(indices)}, expected {list(range(merge_count))}"
            )

    return TorchAOInt8Config(mapped_scale_dtypes)


def normalize_torchao_int8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Translate TorchAO tensor-subclass state into the native INT8 ABI."""

    for name, tensor in weights:
        if name.endswith(_ZERO_POINT_SUFFIX):
            if torch.count_nonzero(tensor).item() != 0:
                raise ValueError(
                    f"TorchAO asymmetric zero point is not supported for {name!r}"
                )
            continue
        if name.endswith(_QDATA_SUFFIX):
            name = name.removesuffix(_QDATA_SUFFIX) + ".weight"
        elif name.endswith(_SCALE_SUFFIX):
            name = name.removesuffix(_SCALE_SUFFIX) + ".weight_scale"
        yield name, tensor


__all__ = [
    "TorchAOInt8Config",
    "inspect_torchao_int8_checkpoint",
    "normalize_torchao_int8_weights",
]
