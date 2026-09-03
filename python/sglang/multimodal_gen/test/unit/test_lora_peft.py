"""PEFT LoRA normalization and fail-closed capability tests."""

import math

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    PARAMETER_DELTA_SUFFIX,
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.peft_adapter import (
    get_peft_lora_alpha,
    load_peft_config,
)


def test_peft_wrapper_slot_and_rslora_scaling_preserve_delta():
    lora_a = torch.randn(4, 8)
    lora_b = torch.randn(16, 4)
    normalized = normalize_lora_state_dict(
        {
            "base_model.model.transformer.proj.lora_A.default.weight": lora_a,
            "base_model.model.transformer.proj.lora_B.default.weight": lora_b,
        },
        adapter_config={
            "use_rslora": True,
            "lora_alpha": 8,
            "alpha_pattern": {"transformer.proj": 8},
        },
    )

    normalized_b = normalized["transformer.proj.lora_B.weight"]
    ordinary_delta = (8 / 4) * normalized_b @ lora_a
    expected_delta = (8 / math.sqrt(4)) * lora_b @ lora_a
    torch.testing.assert_close(ordinary_delta, expected_delta)
    assert normalized["transformer.proj.alpha"].item() == 8


@pytest.mark.parametrize("shape", [(7, 3), (3, 7)])
def test_additive_linear_patch_is_preserved_as_lora(shape):
    delta = torch.randn(shape)
    bias = torch.randn(shape[0])
    normalized = normalize_lora_state_dict(
        {"transformer.proj.diff": delta, "transformer.proj.diff_b": bias}
    )

    lora_a = normalized["transformer.proj.lora_A.weight"]
    lora_b = normalized["transformer.proj.lora_B.weight"]
    torch.testing.assert_close(lora_b @ lora_a, delta)
    assert normalized["transformer.proj.alpha"].item() == min(shape)
    assert normalized["transformer.proj.lora_output_offset"] is bias
    assert not any(name.endswith((".diff", ".diff_b")) for name in normalized)


def test_additive_parameter_and_standalone_lora_bias_are_preserved():
    delta = torch.randn(7)
    bias = torch.randn(5)
    normalized = normalize_lora_state_dict(
        {
            "transformer.norm.diff": delta,
            "transformer.proj.lora_A.weight": torch.randn(3, 7),
            "transformer.proj.lora_B.weight": torch.randn(5, 3),
            "transformer.proj.diff_b": bias,
        }
    )

    assert normalized[f"transformer.norm{PARAMETER_DELTA_SUFFIX}"] is delta
    assert normalized["transformer.proj.lora_output_offset"] is bias


@pytest.mark.parametrize(
    ("state_dict", "message"),
    [
        ({"proj.diff_b": torch.zeros(4)}, "without a matching"),
        ({"proj.diff": torch.zeros(2, 3, 4)}, "2D linear weight"),
        (
            {
                "proj.diff": torch.zeros(4, 3),
                "proj.lora_A.weight": torch.zeros(3, 3),
            },
            "conflicts",
        ),
        ({"proj.set_weight": torch.zeros(4, 3)}, "replacement .set_weight"),
    ],
)
def test_non_composable_additive_patches_fail_closed(state_dict, message):
    with pytest.raises(ValueError, match=message):
        normalize_lora_state_dict(state_dict)


@pytest.mark.parametrize(
    ("state_dict", "adapter_config"),
    [
        ({}, {"use_dora": True}),
        ({}, {"modules_to_save": ["head"]}),
        ({}, {"target_parameters": ["experts.weight"]}),
        ({}, {"fan_in_fan_out": True}),
        ({"encoder.lora_embedding_A.weight": torch.ones(2, 2)}, {}),
    ],
)
def test_unsupported_peft_runtime_semantics_fail_closed(state_dict, adapter_config):
    with pytest.raises(ValueError, match="not supported|unsupported"):
        normalize_lora_state_dict(state_dict, adapter_config=adapter_config)


def test_invalid_peft_lora_alpha_fails_closed():
    with pytest.raises(ValueError, match="positive integer"):
        get_peft_lora_alpha({"lora_alpha": 8.5})


def test_safetensors_alpha_metadata_supplies_peft_config(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    save_file(
        {"proj.lora_A.default.weight": torch.ones(4, 8)},
        weight_path,
        metadata={"alpha": "128"},
    )

    assert load_peft_config(str(weight_path))["lora_alpha"] == 128


def test_mixed_rank_native_safetensors_does_not_apply_global_alpha(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "gate.lora_A.weight": torch.ones(4, 8),
            "proj.lora_A.weight": torch.ones(8, 8),
        },
        weight_path,
        metadata={"lora_alpha": "8"},
    )

    assert "lora_alpha" not in load_peft_config(str(weight_path))
