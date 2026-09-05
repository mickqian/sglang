# SPDX-License-Identifier: Apache-2.0
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.nunchaku_linear import (
    NunchakuAWQLinearMethod,
    NunchakuSVDQLinearMethod,
    is_legacy_nunchaku_available,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .base_config import QuantizationConfig, QuantizeMethodBase

logger = init_logger(__name__)


def is_nunchaku_available() -> bool:
    return is_legacy_nunchaku_available()


@dataclass
class NunchakuConfig(QuantizationConfig):
    """
    Configuration for Nunchaku (SVDQuant) W4A4-style quantization.

    Attributes:
        precision: Quantization precision type. Options:
            - "int4": Standard INT4 quantization
            - "nvfp4": FP4 quantization
        rank: SVD low-rank dimension for absorbing outliers
        group_size: Quantization group size (automatically set based on precision)
        act_unsigned: Use unsigned activation quantization
        transformer_weights_path: Path to pre-quantized transformer weights (.safetensors)
        model_cls: DiT model class that provides quantization rules via get_nunchaku_quant_rules()
    """

    precision: str = "int4"
    rank: int = 32
    group_size: Optional[int] = None
    act_unsigned: bool = False
    transformer_weights_path: Optional[str] = None
    model_cls: Optional[type] = None
    compact_targets: dict[str, str] = field(default_factory=dict)
    compact_checkpoint: bool = False
    awq_group_size: int = 64

    @classmethod
    def get_name(cls) -> str:
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NunchakuConfig":
        svdq = config.get("svdq_w4a4")
        awq = config.get("awq_w4a16")
        if svdq is None and awq is None:
            return cls(
                precision=config.get("precision", "int4"),
                rank=int(config.get("rank", 32)),
                group_size=config.get("group_size"),
                act_unsigned=bool(config.get("act_unsigned", False)),
                transformer_weights_path=config.get("transformer_weights_path"),
            )

        compact_targets: dict[str, str] = {}
        for method, method_config in (
            ("svdq_w4a4", svdq),
            ("awq_w4a16", awq),
        ):
            if method_config is None:
                continue
            targets = method_config.get("targets")
            if (
                not isinstance(targets, list)
                or not targets
                or not all(isinstance(target, str) and target for target in targets)
            ):
                raise ValueError(
                    f"Nunchaku Lite {method}.targets must be a non-empty string list"
                )
            for target in targets:
                if target in compact_targets:
                    raise ValueError(f"Duplicate Nunchaku Lite target {target!r}")
                compact_targets[target] = method

        svdq_config = svdq or {}
        awq_config = awq or {}
        return cls(
            precision=svdq_config.get("precision", "int4"),
            rank=int(svdq_config.get("rank", 32)),
            group_size=svdq_config.get("group_size"),
            act_unsigned=bool(config.get("act_unsigned", False)),
            transformer_weights_path=config.get("transformer_weights_path"),
            compact_targets=compact_targets,
            compact_checkpoint=True,
            awq_group_size=int(awq_config.get("group_size", 64)),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None

        if self.compact_checkpoint:
            method = self.compact_targets.get(prefix)
            if method == "svdq_w4a4":
                return NunchakuSVDQLinearMethod(
                    precision=self.precision,
                    rank=self.rank,
                    act_unsigned=self.act_unsigned,
                    compact_checkpoint=True,
                )
            if method == "awq_w4a16":
                return NunchakuAWQLinearMethod(
                    group_size=self.awq_group_size,
                    compact_checkpoint=True,
                )
            return UnquantizedLinearMethod()

        # get quantization rules from model class
        quant_rules = self._get_quant_rules()

        # priority: skip > awq_w4a16 > svdq_w4a4 > default
        skip_patterns = quant_rules.get("skip", [])
        for pattern in skip_patterns:
            if pattern in prefix.lower():
                return None

        awq_patterns = quant_rules.get("awq_w4a16", [])
        for pattern in awq_patterns:
            if pattern in prefix:
                return NunchakuAWQLinearMethod(group_size=64)

        svdq_patterns = quant_rules.get("svdq_w4a4", [])
        for pattern in svdq_patterns:
            if pattern in prefix:
                return NunchakuSVDQLinearMethod(
                    precision=self.precision,
                    rank=self.rank,
                    act_unsigned=self.act_unsigned,
                )

        # default: apply svdq_w4a4 to all remaining linear layers
        return NunchakuSVDQLinearMethod(
            precision=self.precision,
            rank=self.rank,
            act_unsigned=self.act_unsigned,
        )

    def remap_checkpoint_prefixes(self, param_names_mapping: dict) -> None:
        if not self.compact_checkpoint:
            return
        mapping = get_param_names_mapping(param_names_mapping)
        remapped: dict[str, str] = {}
        sources: dict[str, str] = {}
        for source, method in self.compact_targets.items():
            target, merge_index, _ = mapping(f"{source}.weight")
            target = target.removesuffix(".weight")
            if merge_index is not None or target in remapped:
                previous = sources.get(target)
                raise ValueError(
                    "Nunchaku Lite targets cannot be fused because each target "
                    "owns independent smoothing and low-rank factors: "
                    f"{previous!r} and {source!r} map to {target!r}"
                )
            remapped[target] = method
            sources[target] = source
        self.compact_targets = remapped

    def has_packed_weight(self, prefix: str) -> bool:
        return self.compact_checkpoint and prefix in self.compact_targets

    def _get_quant_rules(self) -> dict[str, list[str]]:
        if self.model_cls is None:
            return {}
        return self.model_cls.get_nunchaku_quant_rules()

    def __post_init__(self):
        if self.group_size is None:
            if self.precision == "nvfp4":
                self.group_size = 16
            elif self.precision == "int4":
                self.group_size = 64
            else:
                raise ValueError(
                    f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
                )

        if self.precision not in ["int4", "nvfp4"]:
            raise ValueError(
                f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
            )

        if self.rank <= 0:
            raise ValueError(f"Rank must be positive, got {self.rank}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NunchakuConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "precision": self.precision,
            "rank": self.rank,
            "group_size": self.group_size,
            "act_unsigned": self.act_unsigned,
            "transformer_weights_path": self.transformer_weights_path,
        }

    @classmethod
    def from_pretrained(cls, model_path: str) -> Optional["NunchakuConfig"]:
        for filename in cls.get_config_filenames():
            config_path = os.path.join(model_path, filename)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                if config_dict.get("quant_method") in {
                    cls.get_name(),
                    "nunchaku_lite",
                }:
                    return cls.from_config(config_dict)
        return None


def _patch_native_svdq_linear(
    module: nn.Module, tensor: Any, svdq_linear_cls: type
) -> bool:
    if (
        isinstance(module, svdq_linear_cls)
        and getattr(module, "wtscale", None) is not None
    ):
        module.wtscale = tensor
        return True
    return False


def _patch_sglang_svdq_linear(
    module: nn.Module, tensor: Any, svdq_method_cls: type
) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if not isinstance(quant_method, svdq_method_cls):
        return False

    existing = getattr(module, "wtscale", None)
    if isinstance(existing, nn.Parameter):
        with torch.no_grad():
            existing.data.copy_(tensor.to(existing.data.dtype))
    else:
        module.wtscale = tensor

    # Keep alpha in sync (kernel reads `layer._nunchaku_alpha`)
    try:
        module._nunchaku_alpha = float(tensor.detach().cpu().item())
    except Exception:
        module._nunchaku_alpha = None
    return True


def _patch_sglang_svdq_wcscales(
    module: nn.Module, tensor: Any, svdq_method_cls: type
) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if not isinstance(quant_method, svdq_method_cls):
        return False

    existing = getattr(module, "wcscales", None)
    if isinstance(existing, nn.Parameter):
        with torch.no_grad():
            existing.data.copy_(tensor.to(existing.data.dtype))
    else:
        module.wcscales = tensor
    return True


def _patch_nunchaku_scales(
    model: nn.Module,
    safetensors_list: list[str],
) -> None:
    """Patch transformer module with Nunchaku scale tensors from safetensors weights.

    For NVFP4 checkpoints, correctness depends on `wtscale` and attention
    `wcscales`. The FSDP loader may skip some of these metadata tensors.
    """

    if not safetensors_list:
        return

    if len(safetensors_list) != 1:
        logger.warning(
            "Nunchaku scale patch expects a single safetensors file, "
            "but got %d files. Skipping.",
            len(safetensors_list),
        )
        return

    from nunchaku.models.linear import SVDQW4A4Linear  # type: ignore[import]

    state_dict = safetensors_load_file(safetensors_list[0])
    if state_dict is None:
        return

    num_wtscale = 0
    num_wcscales = 0

    for name, module in model.named_modules():
        wt = state_dict.get(f"{name}.wtscale")
        if wt is not None:
            if _patch_native_svdq_linear(module, wt, SVDQW4A4Linear):
                num_wtscale += 1
            elif _patch_sglang_svdq_linear(module, wt, NunchakuSVDQLinearMethod):
                num_wtscale += 1

        wc = state_dict.get(f"{name}.wcscales")
        if wc is not None:
            # Some modules may have wcscales as a direct attribute/Parameter.
            existing = getattr(module, "wcscales", None)
            if isinstance(existing, nn.Parameter):
                with torch.no_grad():
                    existing.data.copy_(wc.to(existing.data.dtype))
                num_wcscales += 1
            elif existing is not None:
                setattr(module, "wcscales", wc)
                num_wcscales += 1
            elif _patch_sglang_svdq_wcscales(module, wc, NunchakuSVDQLinearMethod):
                num_wcscales += 1

    if num_wtscale > 0:
        logger.info("Patched wtscale for %d layers", num_wtscale)
    if num_wcscales > 0:
        logger.info("Patched wcscales for %d layers", num_wcscales)
