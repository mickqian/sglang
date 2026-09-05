# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection for MiniMax-H3 transformer overrides."""

import re
from dataclasses import dataclass
from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
)
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
    comfy_quant_key_filter,
    inspect_comfy_quant_markers,
    resolve_comfy_checkpoint_quantization,
)

_SEPARATE_QKV_WEIGHT_RE = re.compile(
    r"^((?:token_refiner\.)?blocks\.\d+\.attn)\.([qkv])_proj\.weight$"
)
_GATE_COMPRESS_WEIGHT_RE = re.compile(
    r"^(?:transformer_blocks|blocks)\.\d+\.attn\.to_gate_compress\.weight$"
)


@dataclass(frozen=True)
class MiniMaxH3CheckpointLayout:
    adaln_curve_shape: tuple[int, int] | None
    adaln_curve_basis_shape: tuple[int, int] | None
    has_gate_compress: bool
    layer_markers: dict[str, dict[str, Any]]
    uses_diffusers_layout: bool
    uses_separate_qkv: bool


def _map_separate_qkv_prefix(prefix: str) -> str:
    match = re.fullmatch(r"((?:token_refiner\.)?blocks\.\d+\.attn)\.[qkv]_proj", prefix)
    return f"{match.group(1)}.qkv_proj" if match is not None else prefix


def inspect_minimax_h3_safetensors(
    safetensors_list: list[str],
) -> MiniMaxH3CheckpointLayout:
    """Read H3 architecture metadata and Comfy per-layer format markers."""
    adaln_curve_shape = None
    adaln_curve_basis_shape = None
    adaln_curve_mean_shape = None
    has_gate_compress = False
    uses_diffusers_layout = False
    separate_qkv_parts: dict[str, set[str]] = {}

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = checkpoint.keys()
            if "adaln_t_table" in keys:
                shape = tuple(checkpoint.get_slice("adaln_t_table").get_shape())
                if len(shape) != 2 or shape[0] < 2:
                    raise ValueError(
                        "MiniMax-H3 adaln_t_table must have shape [N, D] with "
                        f"N >= 2, got {shape} in {path}"
                    )
                if adaln_curve_shape is not None and adaln_curve_shape != shape:
                    raise ValueError(
                        "MiniMax-H3 checkpoint shards disagree on adaln_t_table "
                        f"shape: {adaln_curve_shape} vs {shape}"
                    )
                adaln_curve_shape = shape
            if "adaln_curve_basis" in keys:
                tensor_slice = checkpoint.get_slice("adaln_curve_basis")
                shape = tuple(tensor_slice.get_shape())
                if len(shape) != 2 or min(shape) <= 0:
                    raise ValueError(
                        "MiniMax-H3 adaln_curve_basis must have shape [D, R], "
                        f"got {shape} in {path}"
                    )
                if (
                    adaln_curve_basis_shape is not None
                    and adaln_curve_basis_shape != shape
                ):
                    raise ValueError(
                        "MiniMax-H3 checkpoint shards disagree on "
                        f"adaln_curve_basis shape: {adaln_curve_basis_shape} vs {shape}"
                    )
                if tensor_slice.get_dtype() != "F32":
                    raise ValueError("MiniMax-H3 adaln_curve_basis must be FP32")
                adaln_curve_basis_shape = shape
            if "adaln_curve_mean" in keys:
                tensor_slice = checkpoint.get_slice("adaln_curve_mean")
                shape = tuple(tensor_slice.get_shape())
                if tensor_slice.get_dtype() != "F32":
                    raise ValueError("MiniMax-H3 adaln_curve_mean must be FP32")
                if (
                    adaln_curve_mean_shape is not None
                    and adaln_curve_mean_shape != shape
                ):
                    raise ValueError(
                        "MiniMax-H3 checkpoint shards disagree on "
                        f"adaln_curve_mean shape: {adaln_curve_mean_shape} vs {shape}"
                    )
                adaln_curve_mean_shape = shape
            for key in keys:
                has_gate_compress |= _GATE_COMPRESS_WEIGHT_RE.fullmatch(key) is not None
                uses_diffusers_layout |= key.startswith(
                    ("transformer_blocks.", "token_refiner.refiner_blocks.")
                )
                match = _SEPARATE_QKV_WEIGHT_RE.fullmatch(key)
                if match is not None:
                    separate_qkv_parts.setdefault(match.group(1), set()).add(
                        match.group(2)
                    )

    if adaln_curve_shape is not None and adaln_curve_basis_shape is not None:
        raise ValueError(
            "MiniMax-H3 checkpoint cannot contain both adaln_t_table and "
            "adaln_curve_basis"
        )
    if adaln_curve_basis_shape is not None:
        expected_mean_shape = (adaln_curve_basis_shape[0],)
        if adaln_curve_mean_shape != expected_mean_shape:
            raise ValueError(
                "MiniMax-H3 adaln_curve_mean must match the basis input width: "
                f"expected {expected_mean_shape}, got {adaln_curve_mean_shape}"
            )
    elif adaln_curve_mean_shape is not None:
        raise ValueError(
            "MiniMax-H3 checkpoint contains adaln_curve_mean without adaln_curve_basis"
        )
    incomplete_qkv = {
        prefix: parts
        for prefix, parts in separate_qkv_parts.items()
        if parts != {"q", "k", "v"}
    }
    if incomplete_qkv:
        prefix, parts = next(iter(incomplete_qkv.items()))
        raise ValueError(
            f"MiniMax-H3 separate QKV layer {prefix!r} has {sorted(parts)}, "
            "expected q, k, and v"
        )

    uses_separate_qkv = bool(separate_qkv_parts)
    layer_markers = inspect_comfy_quant_markers(
        safetensors_list,
        param_name_mapper=_map_separate_qkv_prefix if uses_separate_qkv else None,
    )

    return MiniMaxH3CheckpointLayout(
        adaln_curve_shape=adaln_curve_shape,
        adaln_curve_basis_shape=adaln_curve_basis_shape,
        has_gate_compress=has_gate_compress,
        layer_markers=layer_markers,
        uses_diffusers_layout=uses_diffusers_layout,
        uses_separate_qkv=uses_separate_qkv,
    )


def resolve_minimax_h3_checkpoint_quantization(
    layer_markers: dict[str, dict[str, Any]],
    safetensors_list: list[str] | None = None,
    param_names_mapping: dict | None = None,
    reverse_param_names_mapping: dict | None = None,
) -> QuantizationConfig | None:
    formats = {str(marker.get("format")) for marker in layer_markers.values()}
    if "nvfp4" in formats:
        unsupported = formats - {"nvfp4", "int8_tensorwise", "float8_e4m3fn"}
        if unsupported:
            raise NotImplementedError(
                "Unsupported Comfy NVFP4 companion format(s): "
                + ", ".join(sorted(unsupported))
            )
        if safetensors_list is None:
            raise ValueError("MiniMax-H3 NVFP4 metadata requires checkpoint files")
        config = build_nvfp4_config_from_safetensors_list(
            safetensors_list,
            param_names_mapping,
            reverse_param_names_mapping,
        )
        if not isinstance(config, ModelOptFp4Config):
            raise ValueError("Could not resolve MiniMax-H3 NVFP4 checkpoint layout")
        config.set_comfy_layer_markers(layer_markers)
        config.checkpoint_uses_comfy_quantization = True
        config.checkpoint_uses_native_qkv_layout = True
        config.checkpoint_weight_scale_layout = "swizzled"
        config.swap_weight_nibbles = True
        return config
    return resolve_comfy_checkpoint_quantization(layer_markers)


def validate_minimax_h3_checkpoint_variant(
    checkpoint_paths: list[str], selected_variant: str
) -> None:
    names = " ".join(path.lower() for path in checkpoint_paths)
    checkpoint_variants = {
        variant for variant in ("fl2va", "ref2va") if variant in names
    }
    if (
        len(checkpoint_variants) == 1
        and selected_variant.lower() not in checkpoint_variants
    ):
        (checkpoint_variant,) = checkpoint_variants
        raise ValueError(
            f"MiniMax-H3 checkpoint variant {checkpoint_variant!r} does not match "
            f"--model-variant {selected_variant!r}"
        )


__all__ = [
    "MiniMaxH3CheckpointLayout",
    "comfy_quant_key_filter",
    "inspect_minimax_h3_safetensors",
    "resolve_minimax_h3_checkpoint_quantization",
    "validate_minimax_h3_checkpoint_variant",
]
