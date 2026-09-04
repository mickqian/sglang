# SPDX-License-Identifier: Apache-2.0
"""Diffusion-specific GGUF tensor layout and iteration."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.utils.hf_transformers import check_gguf_file

if TYPE_CHECKING:
    import gguf
    from gguf import GGMLQuantizationType as WeightType

_GGML_F32, _GGML_F16, _GGML_BF16, _GGML_NVFP4 = 0, 1, 30, 40
_UNQUANTIZED_TYPES = {_GGML_F32, _GGML_F16, _GGML_BF16}
# SRT has no batched MMQ kernel for I-matrix types and may dequantize them.
_SUPER_BLOCK_DEQUANT_TYPES = {16, 17, 18, 19, 20, 21, 22, 23, 29}
_GGML_SUPER_BLOCK = 256
_GGML_NVFP4_BLOCK, _GGML_NVFP4_TYPE_SIZE = 64, 36


@dataclass(frozen=True)
class GGUFTensorMeta:
    """Logical and packed layouts read before constructing a DiT."""

    ggml_type: int
    logical_shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    stored_dtype: torch.dtype
    param_name: str
    dequantize_on_load: bool = False

    @property
    def weight_type(self) -> WeightType:
        from gguf import GGMLQuantizationType as WeightType

        return WeightType(self.ggml_type)

    @property
    def is_quantized(self) -> bool:
        return self.ggml_type not in _UNQUANTIZED_TYPES

    @property
    def is_packed(self) -> bool:
        return self.is_quantized and not self.dequantize_on_load

    @property
    def is_native_nvfp4(self) -> bool:
        return self.ggml_type == _GGML_NVFP4 and self.is_packed


def _gguf_module() -> Any:
    try:
        import gguf
    except ImportError as exc:
        raise ImportError(
            "Reading a GGUF checkpoint requires the `gguf` package"
        ) from exc
    return gguf


def _open_reader(gguf_file: str) -> gguf.GGUFReader:
    gguf = _gguf_module()
    try:
        reader = gguf.GGUFReader(gguf_file)
    except Exception as exc:
        size = os.path.getsize(gguf_file) if os.path.isfile(gguf_file) else 0
        raise ValueError(
            f"Failed to read GGUF {gguf_file} ({size} bytes). An incomplete or "
            f"corrupt download is the usual cause. Underlying error: {exc}"
        ) from exc
    if reader.byte_order == "S":
        raise ValueError(
            f"GGUF file {gguf_file} uses the opposite byte order from this host"
        )
    return reader


def read_gguf_tensor_meta(gguf_file: str) -> dict[str, GGUFTensorMeta]:
    """Read the exact packed shape required by diffusion parameters."""
    gguf = _gguf_module()
    WeightType = gguf.GGMLQuantizationType
    reader = _open_reader(gguf_file)
    metadata: dict[str, GGUFTensorMeta] = {}
    for tensor in reader.tensors:
        weight_type = WeightType(tensor.tensor_type)
        shape_field = reader.fields.get(f"comfy.gguf.orig_shape.{tensor.name}")
        logical_shape = (
            tuple(int(dim) for dim in shape_field.contents())
            if shape_field is not None
            else tuple(int(dim) for dim in reversed(tensor.shape))
        )
        if math.prod(logical_shape) != tensor.n_elements:
            raise ValueError(
                f"GGUF tensor {tensor.name} declares original shape "
                f"{logical_shape}, which contains {math.prod(logical_shape)} "
                f"elements instead of {tensor.n_elements}"
            )
        is_quantized = int(weight_type) not in _UNQUANTIZED_TYPES
        dequantize_on_load = False
        if is_quantized:
            block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
            if len(logical_shape) != 2 or not tensor.name.endswith(".weight"):
                dequantize_on_load = True
                stored_shape = logical_shape
            elif logical_shape[-1] % block_size:
                if shape_field is None:
                    raise ValueError(
                        f"GGUF tensor {tensor.name} has inner dimension "
                        f"{logical_shape[-1]}, "
                        f"which is not a multiple of block size {block_size}"
                    )
                dequantize_on_load = True
                stored_shape = logical_shape
            else:
                stored_shape = (
                    *logical_shape[:-1],
                    logical_shape[-1] // block_size * type_size,
                )
            if (
                int(weight_type) in _SUPER_BLOCK_DEQUANT_TYPES
                and math.prod(logical_shape) % _GGML_SUPER_BLOCK
            ):
                raise ValueError(
                    f"GGUF tensor {tensor.name} is not aligned to "
                    f"{_GGML_SUPER_BLOCK}-element super blocks"
                )
            stored_dtype = torch.bfloat16 if dequantize_on_load else torch.uint8
        else:
            stored_shape = logical_shape
            stored_dtype = {
                _GGML_F32: torch.float32,
                _GGML_F16: torch.float16,
                _GGML_BF16: torch.bfloat16,
            }[int(weight_type)]

        keeps_weight_name = int(weight_type) == _GGML_NVFP4
        param_name = tensor.name
        if is_quantized and not dequantize_on_load and not keeps_weight_name:
            param_name = f"{tensor.name.removesuffix('.weight')}.qweight"
        metadata[tensor.name] = GGUFTensorMeta(
            ggml_type=int(weight_type),
            logical_shape=logical_shape,
            stored_shape=stored_shape,
            stored_dtype=stored_dtype,
            param_name=param_name,
            dequantize_on_load=dequantize_on_load,
        )
    return metadata


def remap_gguf_tensor_meta(
    tensor_meta: dict[str, GGUFTensorMeta],
    name_mapper: Callable[[str], str],
    dequantize_prefixes: tuple[str, ...] = (),
) -> dict[str, GGUFTensorMeta]:
    """Map checkpoint tensor names while retaining raw lookup aliases."""
    remapped: dict[str, GGUFTensorMeta] = {}
    for checkpoint_name, metadata in tensor_meta.items():
        if metadata.is_quantized and checkpoint_name.startswith(dequantize_prefixes):
            metadata = replace(
                metadata,
                stored_shape=metadata.logical_shape,
                stored_dtype=torch.bfloat16,
                param_name=checkpoint_name,
                dequantize_on_load=True,
            )
        parameter_name = name_mapper(checkpoint_name)
        mapped_param_name = parameter_name
        if metadata.is_packed and not metadata.is_native_nvfp4:
            mapped_param_name = f"{parameter_name.removesuffix('.weight')}.qweight"
        mapped_metadata = replace(metadata, param_name=mapped_param_name)
        for alias in (checkpoint_name, parameter_name):
            previous = remapped.get(alias)
            if previous is not None and previous != mapped_metadata:
                raise ValueError(
                    f"GGUF tensors collide after parameter mapping at {alias!r}"
                )
            remapped[alias] = mapped_metadata
    return remapped


def _tensor_to_torch(tensor, metadata: GGUFTensorMeta) -> torch.Tensor:
    if metadata.dequantize_on_load:
        gguf = _gguf_module()
        value = gguf.dequantize(tensor.data, metadata.weight_type)
        return torch.from_numpy(value.reshape(metadata.logical_shape)).to(
            metadata.stored_dtype
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        value = torch.from_numpy(tensor.data)
    if metadata.ggml_type == _GGML_BF16:
        return value.view(torch.bfloat16).reshape(metadata.stored_shape).clone()
    value = value.reshape(metadata.stored_shape)
    return value.clone() if not metadata.is_packed else value


def _unpack_native_nvfp4(
    value: torch.Tensor, metadata: GGUFTensorMeta
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split GGML's interleaved NVFP4 blocks into standard weight and scales."""
    if metadata.logical_shape[-1] % _GGML_NVFP4_BLOCK:
        raise ValueError(
            f"GGUF NVFP4 tensor {metadata.param_name} is not block aligned"
        )
    blocks = value.reshape(
        *metadata.logical_shape[:-1],
        metadata.logical_shape[-1] // _GGML_NVFP4_BLOCK,
        _GGML_NVFP4_TYPE_SIZE,
    )
    scales = (
        blocks[..., :4]
        .reshape(*metadata.logical_shape[:-1], metadata.logical_shape[-1] // 16)
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    block_values = blocks[..., 4:].reshape(
        *metadata.logical_shape[:-1], -1, 4, 8
    )
    unpacked = torch.cat((block_values & 0x0F, block_values >> 4), dim=-1)
    packed = (unpacked[..., 0::2] | (unpacked[..., 1::2] << 4)).reshape(
        *metadata.logical_shape[:-1], metadata.logical_shape[-1] // 2
    )
    return packed.contiguous(), scales


def gguf_weights_iterator(
    gguf_file: str,
    tensor_meta: dict[str, GGUFTensorMeta],
    key_filter: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield checkpoint tensors under their diffusion parameter names."""
    reader = _open_reader(gguf_file)
    for tensor in reader.tensors:
        if key_filter is not None and not key_filter(tensor.name):
            continue
        metadata = tensor_meta[tensor.name]
        value = _tensor_to_torch(tensor, metadata)
        if metadata.is_native_nvfp4:
            weight, weight_scale = _unpack_native_nvfp4(value, metadata)
            yield metadata.param_name, weight
            yield (
                f"{metadata.param_name.removesuffix('.weight')}.weight_scale",
                weight_scale,
            )
        else:
            yield metadata.param_name, value


def names_gguf_checkpoint(reference: str) -> bool:
    """Recognize an explicit local or Hub GGUF reference without downloading."""
    if not reference:
        return False
    if check_gguf_file(reference):
        return True
    if os.path.exists(reference):
        return False
    if os.path.isabs(reference) or reference.startswith((".", "~")):
        return reference.endswith(".gguf")
    if ":" in reference:
        repo_id, _, quant_type = reference.rpartition(":")
        return repo_id.count("/") == 1 and bool(quant_type)
    return reference.endswith(".gguf") and len(reference.strip("/").split("/")) >= 3


__all__ = [
    "GGUFTensorMeta",
    "gguf_weights_iterator",
    "names_gguf_checkpoint",
    "read_gguf_tensor_meta",
    "remap_gguf_tensor_meta",
]
