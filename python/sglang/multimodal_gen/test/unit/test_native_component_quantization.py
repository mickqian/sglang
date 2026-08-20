from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
    _admit_native_library_quantization,
)


def test_transformers_auto_quantizer_is_the_capability_source_of_truth():
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
        }
    }
    with patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader."
        "AutoHfQuantizer.from_config"
    ) as admit:
        assert _admit_native_library_quantization(config, "transformers", "connector")

    admit.assert_called_once_with(config["quantization_config"])


def test_native_library_quantization_fails_closed_for_unsupported_metadata():
    nested = {"text_config": {"quantization_config": {"quant_method": "fp8"}}}
    with pytest.raises(ComponentCheckpointUnsupportedError, match="text_config"):
        _admit_native_library_quantization(nested, "transformers", "connector")

    unsupported = {"quantization_config": {"quant_method": "future-format"}}
    with patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader."
        "DiffusersAutoQuantizer.from_config",
        side_effect=ValueError("unknown quantizer"),
    ), pytest.raises(ComponentCheckpointUnsupportedError, match="future-format"):
        _admit_native_library_quantization(unsupported, "diffusers", "connector")


def test_customized_loader_detects_declared_quantization():
    config = {"quantization_config": {"quant_method": "bitsandbytes_4bit"}}
    with patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader."
        "get_diffusers_component_config",
        return_value=config,
    ):
        assert ComponentLoader._checkpoint_declares_quantization(
            "component-path", None, "connector", "diffusers"
        )
