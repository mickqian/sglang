# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.configs.models.dits import WanS2VConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel


class WanS2VTransformer3DModel(WanTransformer3DModel):
    """Native Wan S2V transformer entry point.

    This currently reuses the Wan video transformer execution skeleton so the
    S2V pipeline can migrate toward the standard SGLang denoising path without
    changing loader/model-index semantics a second time.

    The S2V-specific audio injection, pose conditioning, and framepack logic
    will be layered into this class incrementally in follow-up changes.
    """

    _aliases = ["WanS2VTransformer3DModel"]
    _fsdp_shard_conditions = WanS2VConfig()._fsdp_shard_conditions
    _compile_conditions = WanS2VConfig()._compile_conditions
    _supported_attention_backends = WanS2VConfig()._supported_attention_backends
    param_names_mapping = WanS2VConfig().param_names_mapping
    reverse_param_names_mapping = WanS2VConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanS2VConfig().lora_param_names_mapping

    def __init__(
        self,
        config: WanS2VConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)


EntryClass = WanS2VTransformer3DModel
