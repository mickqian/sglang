import math
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora_format_adapter import (
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
    LoRAPipeline,
    _normalize_peft_scaling,
    _peft_lora_alpha,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora

_RANK_PATCH = "sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline.dist.get_rank"


class _TestLoRAPipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        return None


def _make_layer() -> BaseLayerWithLoRA:
    return BaseLayerWithLoRA(torch.nn.Linear(2, 2, bias=False))


def _make_pipeline(layer: BaseLayerWithLoRA) -> _TestLoRAPipeline:
    pipeline = object.__new__(_TestLoRAPipeline)
    pipeline.modules = {"transformer": torch.nn.Module()}
    pipeline.server_args = SimpleNamespace(lora_merge_mode="dynamic")
    pipeline.lora_initialized = True
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {"adapter": "/adapter"}
    pipeline.loaded_adapter_alphas = {"adapter": None}
    pipeline.cur_adapter_name = {}
    pipeline.cur_adapter_path = {}
    pipeline.cur_adapter_strength = {}
    pipeline.cur_adapter_config = {}
    pipeline.lora_layers = {"linear": layer}
    pipeline.lora_layers_transformer_2 = {}
    pipeline.lora_layers_critic = {}
    pipeline.is_lora_merged = {}

    pipeline.lora_adapters["adapter"]["linear.lora_A"] = torch.ones(1, 2)
    pipeline.lora_adapters["adapter"]["linear.lora_B"] = torch.ones(2, 1)
    return pipeline


def test_dynamic_lora_reactivates_cached_layers_without_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    first_lora_a = layer.lora_A
    first_lora_b = layer.lora_B
    assert context_calls == 0
    assert not layer.disable_lora

    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    pipeline.deactivate_lora_weights("transformer")
    assert layer.disable_lora

    def fail_apply(*args, **kwargs):
        raise AssertionError("cached dynamic LoRA should not rebuild weights")

    context_calls = 0
    pipeline._temporarily_disable_offload = counted_context
    pipeline._apply_lora_to_layers = fail_apply

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    assert context_calls == 0
    assert not layer.disable_lora
    assert layer.lora_A is first_lora_a
    assert layer.lora_B is first_lora_b


def test_merged_lora_still_uses_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=1.0,
            merge_mode="merge",
        )

    assert context_calls == 1
    assert layer.merged
    assert pipeline.is_lora_merged["transformer"]


def test_lora_alpha_override_updates_cached_adapter_scale():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=1.0,
            merge_mode="dynamic",
            lora_alpha=8,
        )

    assert pipeline.loaded_adapter_alphas["adapter"] == 8
    assert layer.lora_rank == 1
    assert layer.lora_alpha == 8


def test_pinned_lora_weight_selects_one_file_from_multi_adapter_source(tmp_path):
    weight_name = "adapter-v4.safetensors"
    weight_path = tmp_path / weight_name
    weight_path.touch()

    (tmp_path / "another-adapter.safetensors").touch()
    actual = maybe_download_lora(str(tmp_path), weight_name=weight_name)

    assert actual == str(weight_path)


def test_peft_named_adapter_slot_is_normalized():
    state_dict = {
        "base_model.model.transformer.blocks.0.proj.lora_A.default.weight": (
            torch.ones(4, 8)
        ),
        "base_model.model.transformer.blocks.0.proj.lora_B.default.weight": (
            torch.ones(8, 4)
        ),
    }

    normalized = normalize_lora_state_dict(state_dict)

    assert set(normalized) == {
        "transformer.blocks.0.proj.lora_A.weight",
        "transformer.blocks.0.proj.lora_B.weight",
    }


def test_multiple_peft_adapter_slots_fail_closed():
    state_dict = {
        "proj.lora_A.first.weight": torch.ones(4, 8),
        "proj.lora_A.second.weight": torch.ones(4, 8),
    }

    with pytest.raises(ValueError, match="multiple PEFT adapter slots"):
        normalize_lora_state_dict(state_dict)


def test_normalize_peft_rslora_scaling_preserves_effective_delta():
    lora_a = torch.randn(4, 8)
    lora_b = torch.randn(16, 4)
    normalized = _normalize_peft_scaling(
        {
            "transformer.block.proj.lora_A.weight": lora_a,
            "transformer.block.proj.lora_B.weight": lora_b,
        },
        {"use_rslora": True, "lora_alpha": 8},
    )

    ordinary_delta = (
        (8 / 4) * normalized["transformer.block.proj.lora_B.weight"] @ lora_a
    )
    expected_delta = (8 / math.sqrt(4)) * lora_b @ lora_a
    torch.testing.assert_close(ordinary_delta, expected_delta)


@pytest.mark.parametrize(
    ("state_dict", "adapter_config", "message"),
    [
        (
            {"transformer.block.lora_magnitude_vector.weight": torch.ones(8)},
            {},
            "DoRA adapters",
        ),
        ({}, {"use_dora": True}, "DoRA adapters"),
        ({}, {"alpha_pattern": []}, "alpha_pattern must be an object"),
    ],
)
def test_unsupported_peft_scaling_fails_closed(state_dict, adapter_config, message):
    with pytest.raises(ValueError, match=message):
        _normalize_peft_scaling(state_dict, adapter_config)


@pytest.mark.parametrize(
    "adapter_config",
    [
        {"modules_to_save": ["head"]},
        {"target_parameters": ["experts.weight"]},
        {"bias": "lora_only"},
        {"fan_in_fan_out": True},
    ],
)
def test_peft_auxiliary_runtime_features_fail_closed(adapter_config):
    with pytest.raises(ValueError, match="auxiliary/runtime features"):
        _normalize_peft_scaling({}, adapter_config)


@pytest.mark.parametrize("alpha", [True, 0, -1, 8.5, "8"])
def test_invalid_peft_lora_alpha_fails_closed(alpha):
    with pytest.raises(ValueError, match="positive integer"):
        _peft_lora_alpha({"lora_alpha": alpha})


def test_peft_alpha_pattern_becomes_per_layer_alpha_tensor():
    state_dict = {
        "transformer.blocks.0.proj.lora_A.weight": torch.ones(4, 8),
        "transformer.blocks.0.proj.lora_B.weight": torch.ones(8, 4),
        "transformer.blocks.1.proj.lora_A.weight": torch.ones(8, 8),
        "transformer.blocks.1.proj.lora_B.weight": torch.ones(8, 8),
    }

    normalized = _normalize_peft_scaling(
        state_dict,
        {"alpha_pattern": {"blocks.0.proj": 8, "blocks.1.proj": 16}},
    )

    assert normalized["transformer.blocks.0.proj.alpha"].item() == 8
    assert normalized["transformer.blocks.1.proj.alpha"].item() == 16


def test_peft_alpha_pattern_rejects_missing_alpha_value():
    state_dict = {
        "transformer.blocks.0.proj.lora_A.weight": torch.ones(4, 8),
        "transformer.blocks.0.proj.lora_B.weight": torch.ones(8, 4),
    }

    with pytest.raises(ValueError, match="must be a positive integer"):
        _normalize_peft_scaling(state_dict, {"alpha_pattern": {"blocks.0.proj": None}})
