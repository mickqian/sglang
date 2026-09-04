# SPDX-License-Identifier: Apache-2.0
"""Config and checkpoint admission for Optimum Quanto qint8 weights."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable

import torch
from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.configs.int8_weight_only_config import (
    Int8WeightOnlyConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.quanto_int8 import (
    normalize_quanto_int8_weights,
)

_FLOAT_DTYPES = {"BF16", "F16", "F32"}


class QuantoInt8Config(Int8WeightOnlyConfig):
    """Dispatch linears declared qint8 in an Optimum Quanto quantization map."""

    @classmethod
    def get_name(cls) -> str:
        return "quanto_int8"

    def normalize_checkpoint_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return normalize_quanto_int8_weights(weights)


def inspect_quanto_int8_checkpoint(
    file_path: str,
    param_name_mapper: Callable[[str], str] | None = None,
) -> QuantoInt8Config | None:
    """Validate a self-describing Quanto qint8 safetensors checkpoint."""

    with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("quantization_format") != "quanto":
            return None

        encoded_map = metadata.get("quantization_map_base64")
        if encoded_map is None:
            raise ValueError("Quanto checkpoint is missing quantization_map_base64")
        try:
            quantization_map = json.loads(
                base64.b64decode(encoded_map, validate=True).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid Quanto quantization_map_base64") from error
        if not isinstance(quantization_map, dict) or not quantization_map:
            raise ValueError("Quanto quantization map must be a non-empty object")
        if not all(
            isinstance(prefix, str) and isinstance(spec, dict)
            for prefix, spec in quantization_map.items()
        ):
            raise ValueError("Quanto quantization map entries must be named objects")

        checkpoint_keys = set(checkpoint.keys())
        data_suffix = ".weight._data"
        data_prefixes = {
            name.removesuffix(data_suffix)
            for name in checkpoint_keys
            if name.endswith(data_suffix)
        }
        map_prefixes = set(quantization_map)
        if data_prefixes != map_prefixes:
            missing_map = data_prefixes - map_prefixes
            missing_data = map_prefixes - data_prefixes
            raise ValueError(
                "Quanto tensor/map prefixes do not match: "
                f"missing metadata={sorted(missing_map)[:5]}, "
                f"missing tensors={sorted(missing_data)[:5]}"
            )

        mapped_scale_dtypes: dict[str, torch.dtype] = {}
        for prefix, quantization in quantization_map.items():
            if quantization.get("weights") != "qint8":
                raise ValueError(
                    f"Unsupported Quanto weight type for {prefix!r}: "
                    f"{quantization.get('weights')!r}"
                )
            if quantization.get("activations") != "none":
                raise ValueError(
                    f"Quanto activation quantization is not supported for {prefix!r}"
                )

            names = {
                "data": f"{prefix}.weight._data",
                "scale": f"{prefix}.weight._scale",
                "input": f"{prefix}.input_scale",
                "output": f"{prefix}.output_scale",
            }
            missing = set(names.values()) - checkpoint_keys
            if missing:
                raise ValueError(
                    f"Quanto layer {prefix!r} is missing tensors: {sorted(missing)}"
                )
            if f"{prefix}.weight" in checkpoint_keys:
                raise ValueError(
                    f"Quanto layer {prefix!r} contains both packed and dense weights"
                )

            data_slice = checkpoint.get_slice(names["data"])
            scale_slice = checkpoint.get_slice(names["scale"])
            data_shape = tuple(data_slice.get_shape())
            scale_shape = tuple(scale_slice.get_shape())
            if data_slice.get_dtype() != "I8" or len(data_shape) != 2:
                raise ValueError(
                    f"Quanto layer {prefix!r} needs a 2D I8 weight, got "
                    f"{data_slice.get_dtype()} {data_shape}"
                )
            if scale_slice.get_dtype() not in _FLOAT_DTYPES or scale_shape != (
                data_shape[0],
                1,
            ):
                raise ValueError(
                    f"Quanto layer {prefix!r} has incompatible scale "
                    f"{scale_slice.get_dtype()} {scale_shape}"
                )
            for scale_name in (names["input"], names["output"]):
                scale = checkpoint.get_slice(scale_name)
                if (
                    scale.get_dtype() not in _FLOAT_DTYPES
                    or tuple(scale.get_shape()) != ()
                ):
                    raise ValueError(
                        f"Quanto auxiliary scale {scale_name!r} must be a float scalar"
                    )

            mapped_prefix = (
                param_name_mapper(prefix) if param_name_mapper is not None else prefix
            )
            if mapped_prefix in mapped_scale_dtypes:
                raise ValueError(
                    f"Quanto layers collide after parameter mapping at {mapped_prefix!r}"
                )
            mapped_scale_dtypes[mapped_prefix] = {
                "BF16": torch.bfloat16,
                "F16": torch.float16,
                "F32": torch.float32,
            }[scale_slice.get_dtype()]

    return QuantoInt8Config(mapped_scale_dtypes)


__all__ = ["QuantoInt8Config", "inspect_quanto_int8_checkpoint"]
