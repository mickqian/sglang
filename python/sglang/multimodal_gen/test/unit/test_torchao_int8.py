import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.torchao_int8_config import (
    TorchAOInt8Config,
    inspect_torchao_int8_checkpoint,
    normalize_torchao_int8_weights,
)
from sglang.multimodal_gen.runtime.layers.quantization.int8_weight_only import (
    Int8WeightOnlyLinearMethod,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _diffusers_h3_checkpoint,
)


def _torchao_config(config_type="Int8WeightOnlyConfig"):
    return {
        "quantization_config": {
            "quant_method": "torchao",
            "quant_type": {
                "default": {
                    "_type": config_type,
                    "_version": 2,
                    "_data": {
                        "granularity": {
                            "_type": "PerRow",
                            "_version": 1,
                            "_data": {"dim": -1},
                        },
                        "group_size": None,
                        "set_inductor_config": True,
                    },
                }
            },
        }
    }


def _save_h3_qkv(path, *, zero_point=0):
    tensors = {}
    for index, projection in enumerate(("to_q", "to_k", "to_v"), start=1):
        prefix = f"transformer_blocks.0.attn.{projection}"
        tensors[f"{prefix}._weight_qdata"] = torch.full(
            (2, 3), index, dtype=torch.int8
        )
        tensors[f"{prefix}._weight_scale"] = torch.full(
            (2, 1), index / 10, dtype=torch.float32
        )
        tensors[f"{prefix}._weight_zero_point"] = torch.full(
            (2, 1), zero_point, dtype=torch.int8
        )
    save_file(tensors, path)


def test_torchao_int8_maps_split_h3_qkv_to_one_native_linear(tmp_path):
    checkpoint = tmp_path / "transformer.safetensors"
    _save_h3_qkv(checkpoint)
    mapping = get_param_names_mapping(MiniMaxH3DiTArchConfig().param_names_mapping)

    config = inspect_torchao_int8_checkpoint(
        _torchao_config(), [str(checkpoint)], param_name_mapper=mapping
    )

    assert isinstance(config, TorchAOInt8Config)
    assert config.layer_scale_dtypes == {
        "blocks.0.attn.qkv_proj": torch.float32
    }
    mapped = _diffusers_h3_checkpoint(load_file(checkpoint).items())
    normalized = dict(normalize_torchao_int8_weights(mapped))

    assert list(normalized) == [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.qkv_proj.weight_scale",
    ]
    assert normalized["blocks.0.attn.qkv_proj.weight"].shape == (6, 3)
    assert normalized["blocks.0.attn.qkv_proj.weight_scale"].dtype == torch.float32


def test_torchao_int8_rejects_unsupported_schema_and_asymmetric_weights(tmp_path):
    checkpoint = tmp_path / "transformer.safetensors"
    _save_h3_qkv(checkpoint)
    with pytest.raises(ValueError, match="Only TorchAO Int8WeightOnlyConfig"):
        inspect_torchao_int8_checkpoint(
            _torchao_config("Int4WeightOnlyConfig"), [str(checkpoint)]
        )

    _save_h3_qkv(checkpoint, zero_point=1)
    with pytest.raises(ValueError, match="asymmetric zero point"):
        list(
            normalize_torchao_int8_weights(
                _diffusers_h3_checkpoint(load_file(checkpoint).items())
            )
        )


def test_int8_weight_only_linear_uses_per_row_scales():
    layer = torch.nn.Module()
    method = Int8WeightOnlyLinearMethod(torch.float32)
    method.create_weights(layer, 3, [2], 3, 2, torch.bfloat16)
    layer.weight.data.copy_(torch.tensor([[1, -2, 3], [4, 5, -6]]))
    layer.weight_scale.data.copy_(torch.tensor([[0.25], [0.5]]))
    inputs = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.bfloat16)

    actual = method.apply(layer, inputs)
    expected = torch.nn.functional.linear(
        inputs,
        layer.weight.to(inputs.dtype) * layer.weight_scale.to(inputs.dtype),
    )

    torch.testing.assert_close(actual, expected)
