# SPDX-License-Identifier: Apache-2.0
import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
from kernels import get_kernel
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

logger = init_logger(__name__)

try:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda as legacy_svdq_gemm
    from nunchaku.ops.gemv import awq_gemv_w4a16_cuda as legacy_awq_gemv
    from nunchaku.ops.quantize import (
        svdq_quantize_w4a4_act_fuse_lora_cuda as legacy_svdq_quantize,
    )
except ImportError:
    legacy_svdq_gemm = None
    legacy_awq_gemv = None
    legacy_svdq_quantize = None


_NUNCHAKU_LITE_KERNEL_REPO = "rootonchair/nunchaku-lite-kernels"
_NUNCHAKU_LITE_KERNEL_VERSION = 2
_nunchaku_lite_ops: Any | None = None


def is_legacy_nunchaku_available() -> bool:
    return all(
        op is not None
        for op in (legacy_svdq_gemm, legacy_awq_gemv, legacy_svdq_quantize)
    )


def initialize_nunchaku_lite_runtime(
    *, trust_remote_code: bool, precision: str, needs_awq: bool
) -> None:
    global _nunchaku_lite_ops
    if not trust_remote_code:
        raise ValueError(
            "Nunchaku Lite loads CUDA kernels from "
            f"{_NUNCHAKU_LITE_KERNEL_REPO!r}; pass --trust-remote-code to "
            "authorize this optional runtime"
        )
    if not torch.cuda.is_available():
        raise ValueError("Nunchaku Lite requires a CUDA-capable NVIDIA GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] == 9:
        raise ValueError("Nunchaku Lite does not support Hopper GPUs")
    if precision == "nvfp4" and capability < (10, 0):
        raise ValueError("Nunchaku Lite NVFP4 requires a Blackwell or newer NVIDIA GPU")
    if precision == "int4" and capability < (7, 5):
        raise ValueError("Nunchaku Lite INT4 requires a Turing or newer NVIDIA GPU")
    if _nunchaku_lite_ops is not None:
        if needs_awq:
            _nunchaku_lite_ops.awq_gemm_w4a16_g64_int32
            _nunchaku_lite_ops.gemv_awq
        return

    kernel = get_kernel(
        _NUNCHAKU_LITE_KERNEL_REPO,
        version=_NUNCHAKU_LITE_KERNEL_VERSION,
        trust_remote_code=True,
    )
    # Resolve required entry points during startup, before model creation.
    kernel.ops.quantize_w4a4_act_fuse_lora
    kernel.ops.gemm_w4a4
    if needs_awq:
        kernel.ops.awq_gemm_w4a16_g64_int32
        kernel.ops.gemv_awq
    _nunchaku_lite_ops = kernel.ops


def _nunchaku_lite_quantize(
    x: torch.Tensor,
    *,
    lora_down: torch.Tensor,
    smooth: torch.Tensor,
    fp4: bool,
    pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _nunchaku_lite_ops is not None
    rows, channels = x.shape
    padded_rows = math.ceil(rows / pad_size) * pad_size
    quantized_x = torch.empty(
        padded_rows, channels // 2, dtype=torch.uint8, device=x.device
    )
    scale_group_size = 16 if fp4 else 64
    scale_dtype = torch.float8_e4m3fn if fp4 else x.dtype
    ascales = torch.empty(
        channels // scale_group_size,
        padded_rows,
        dtype=scale_dtype,
        device=x.device,
    )
    lora_act = torch.empty(
        padded_rows,
        lora_down.shape[1],
        dtype=torch.float32,
        device=x.device,
    )
    _nunchaku_lite_ops.quantize_w4a4_act_fuse_lora(
        x,
        quantized_x,
        ascales,
        lora_down,
        lora_act,
        smooth,
        False,
        fp4,
    )
    return quantized_x, ascales, lora_act


def _nunchaku_lite_gemm(
    *,
    act: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None,
    fp4: bool,
    alpha: torch.Tensor | None,
    wcscales: torch.Tensor | None,
    act_unsigned: bool,
) -> None:
    assert _nunchaku_lite_ops is not None
    lora_scales = [1.0] * math.ceil(lora_up.shape[1] / 16)
    _nunchaku_lite_ops.gemm_w4a4(
        act,
        weight,
        out,
        None,
        ascales,
        wscales,
        None,
        None,
        lora_act,
        lora_up,
        None,
        None,
        None,
        None,
        None,
        bias,
        None,
        None,
        None,
        act_unsigned,
        lora_scales,
        False,
        fp4,
        alpha,
        wcscales,
        None,
        None,
        None,
        0,
    )


def _nunchaku_lite_awq(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    assert _nunchaku_lite_ops is not None
    in_features = x.shape[1]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if (
        x.shape[0] >= 16
        and group_size == 64
        and in_features % 64 == 0
        and out_features % 128 == 0
    ):
        return _nunchaku_lite_ops.awq_gemm_w4a16_g64_int32(x, weight, wscales, wzeros)

    outputs = []
    for start in range(0, x.shape[0], 8):
        chunk = x[start : start + 8]
        outputs.append(
            _nunchaku_lite_ops.gemv_awq(
                chunk,
                weight,
                wscales,
                wzeros,
                chunk.shape[0],
                out_features,
                in_features,
                group_size,
            )
        )
    return torch.cat(outputs, dim=0)


class NunchakuSVDQLinearMethod(LinearMethodBase):
    def __init__(
        self,
        precision: str = "int4",
        rank: int = 32,
        act_unsigned: bool = False,
        compact_checkpoint: bool = False,
    ):
        self.precision = precision
        self.rank = rank
        self.act_unsigned = act_unsigned
        self.compact_checkpoint = compact_checkpoint

        if precision == "nvfp4":
            self.group_size = 16
        else:
            self.group_size = 64

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        num_groups = input_size_per_partition // self.group_size
        if self.precision == "nvfp4":
            scale_dtype = torch.float8_e4m3fn
        else:
            scale_dtype = params_dtype
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=scale_dtype),
            requires_grad=False,
        )

        smooth_factor = Parameter(
            torch.empty(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        smooth_factor_orig = (
            None
            if self.compact_checkpoint
            else Parameter(
                torch.empty(input_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
        )

        proj_down = Parameter(
            torch.empty(input_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )
        proj_up = Parameter(
            torch.empty(output_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )

        if self.precision == "nvfp4":
            wcscales = Parameter(
                torch.empty(
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            wtscale = Parameter(
                torch.empty(1, dtype=params_dtype),
                requires_grad=False,
            )
        else:
            wcscales = None
            wtscale = None

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("smooth_factor_orig", smooth_factor_orig)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("wcscales", wcscales)
        layer.register_parameter("wtscale", wtscale)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.precision = self.precision
        layer.rank = self.rank
        layer.group_size = self.group_size
        layer.act_unsigned = self.act_unsigned

        set_weight_attrs(wscales, {"input_dim": 0, "output_dim": 1})
        set_weight_attrs(smooth_factor, {"input_dim": 0})
        set_weight_attrs(proj_down, {"input_dim": 0})
        set_weight_attrs(proj_up, {"output_dim": 0})
        if smooth_factor_orig is not None:
            set_weight_attrs(smooth_factor_orig, {"input_dim": 0})
        if wcscales is not None:
            set_weight_attrs(wcscales, {"output_dim": 0})

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(smooth_factor, {"weight_loader": weight_loader})
            set_weight_attrs(proj_down, {"weight_loader": weight_loader})
            set_weight_attrs(proj_up, {"weight_loader": weight_loader})
            if smooth_factor_orig is not None:
                set_weight_attrs(
                    smooth_factor_orig,
                    {"weight_loader": weight_loader},
                )
            if wcscales is not None:
                set_weight_attrs(wcscales, {"weight_loader": weight_loader})
            if wtscale is not None:
                set_weight_attrs(wtscale, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.smooth_factor = Parameter(layer.smooth_factor.data, requires_grad=False)
        if layer.smooth_factor_orig is not None:
            layer.smooth_factor_orig = Parameter(
                layer.smooth_factor_orig.data, requires_grad=False
            )
        layer.proj_down = Parameter(layer.proj_down.data, requires_grad=False)
        layer.proj_up = Parameter(layer.proj_up.data, requires_grad=False)
        if layer.wcscales is not None:
            layer.wcscales = Parameter(layer.wcscales.data, requires_grad=False)
        if layer.wtscale is not None:
            layer.wtscale = Parameter(layer.wtscale.data, requires_grad=False)

        alpha: float | None = None
        wtscale = layer.wtscale
        if wtscale is not None:
            if isinstance(wtscale, Parameter):
                wtscale = wtscale.data
            if isinstance(wtscale, torch.Tensor):
                alpha = float(wtscale.detach().cpu().item())
            else:
                alpha = float(wtscale)
        layer._nunchaku_alpha = alpha

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        if self.compact_checkpoint:
            quantized_x, ascales, lora_act_out = _nunchaku_lite_quantize(
                x_2d,
                lora_down=layer.proj_down,
                smooth=layer.smooth_factor,
                fp4=layer.precision == "nvfp4",
                pad_size=256,
            )
        else:
            assert legacy_svdq_quantize is not None
            quantized_x, ascales, lora_act_out = legacy_svdq_quantize(
                x_2d,
                lora_down=layer.proj_down,
                smooth=layer.smooth_factor,
                fp4=layer.precision == "nvfp4",
                pad_size=256,
            )
        out_2d = torch.empty(
            x_2d.shape[0],
            layer.output_size_per_partition,
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        if self.compact_checkpoint:
            _nunchaku_lite_gemm(
                act=quantized_x,
                weight=layer.qweight,
                out=out_2d,
                ascales=ascales,
                wscales=layer.wscales,
                lora_act=lora_act_out,
                lora_up=layer.proj_up,
                bias=bias,
                fp4=layer.precision == "nvfp4",
                alpha=layer.wtscale,
                wcscales=layer.wcscales,
                act_unsigned=layer.act_unsigned,
            )
        else:
            assert legacy_svdq_gemm is not None
            legacy_svdq_gemm(
                act=quantized_x,
                wgt=layer.qweight,
                out=out_2d,
                ascales=ascales,
                wscales=layer.wscales,
                lora_act_in=lora_act_out,
                lora_up=layer.proj_up,
                bias=bias,
                fp4=layer.precision == "nvfp4",
                alpha=layer._nunchaku_alpha,
                wcscales=layer.wcscales,
                act_unsigned=layer.act_unsigned,
            )
        out = out_2d.reshape(*orig_shape[:-1], layer.output_size_per_partition)
        return out


class NunchakuAWQLinearMethod(LinearMethodBase):
    def __init__(self, group_size: int = 64, compact_checkpoint: bool = False):
        self.group_size = group_size
        self.pack_factor = 8
        self.compact_checkpoint = compact_checkpoint

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition // 4,
                input_size_per_partition // 2,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        num_groups = input_size_per_partition // self.group_size
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        wzeros = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("wzeros", wzeros)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = self.group_size
        layer.pack_factor = self.pack_factor

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(wzeros, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.wzeros = Parameter(layer.wzeros.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])

        in_features = layer.input_size_per_partition
        out_features = layer.output_size_per_partition
        if self.compact_checkpoint:
            out_2d = _nunchaku_lite_awq(
                x=x_2d.contiguous(),
                weight=layer.qweight,
                wscales=layer.wscales,
                wzeros=layer.wzeros,
                out_features=out_features,
                group_size=layer.group_size,
            )
        else:
            assert legacy_awq_gemv is not None
            out_2d = legacy_awq_gemv(
                in_feats=x_2d,
                kernel=layer.qweight,
                scaling_factors=layer.wscales,
                zeros=layer.wzeros,
                m=x_2d.shape[0],
                n=out_features,
                k=in_features,
                group_size=layer.group_size,
            )
        if bias is not None:
            view_shape = [1] * (out_2d.ndim - 1) + [-1]
            out_2d.add_(bias.view(view_shape))

        out = out_2d.reshape(*orig_shape[:-1], out_features)
        return out
