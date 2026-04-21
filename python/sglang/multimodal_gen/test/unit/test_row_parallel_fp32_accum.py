import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.linear import apply_fp32_row_parallel_linear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)


class _DummyRowParallelLayer:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


class _DummyRowParallelBaseLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 2))
        self.bias = torch.nn.Parameter(torch.randn(4))
        self.input_is_parallel = True
        self.accumulate_in_fp32 = True
        self.skip_bias_add = False
        self.forward_calls = 0

    def forward(self, x: torch.Tensor):
        self.forward_calls += 1
        return x + 1, None


class _DummyColumnParallelBaseLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 4))
        self.bias = torch.nn.Parameter(torch.randn(2))
        self.skip_bias_add = False
        self.gather_output = False
        self.accumulate_in_fp32 = True
        self.forward_calls = 0

    def forward(self, x: torch.Tensor):
        self.forward_calls += 1
        return x - 1, None


def _mse(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).pow(2).mean().item()


def test_fp32_row_parallel_partial_accumulation_tracks_full_linear_more_closely():
    torch.manual_seed(0)

    batch, seq_len, input_size, output_size = 1, 32, 128, 96
    tp_size = 2
    shard_size = input_size // tp_size

    x = torch.randn(batch, seq_len, input_size, dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)

    reference = F.linear(x, weight)

    bf16_partial_outputs = []
    fp32_partial_outputs = []
    for tp_rank in range(tp_size):
        start = tp_rank * shard_size
        end = start + shard_size
        x_shard = x[..., start:end]
        weight_shard = weight[:, start:end]

        bf16_partial_outputs.append(F.linear(x_shard, weight_shard))

        layer = _DummyRowParallelLayer(weight_shard)
        fp32_partial_outputs.append(apply_fp32_row_parallel_linear(layer, x_shard))

    bf16_row_parallel = sum(bf16_partial_outputs)
    fp32_row_parallel = sum(fp32_partial_outputs).to(dtype=x.dtype)

    assert _mse(fp32_row_parallel, reference) < _mse(bf16_row_parallel, reference)


def test_row_parallel_lora_wrapper_delegates_to_base_layer_when_lora_inactive():
    base_layer = _DummyRowParallelBaseLayer()
    wrapper = RowParallelLinearWithLoRA(base_layer)
    x = torch.randn(3, 2)

    wrapper.disable_lora = True
    out_disabled, bias_disabled = wrapper(x)
    assert base_layer.forward_calls == 1
    assert torch.equal(out_disabled, x + 1)
    assert bias_disabled is None

    wrapper.disable_lora = False
    wrapper.merged = True
    out_merged, bias_merged = wrapper(x)
    assert base_layer.forward_calls == 2
    assert torch.equal(out_merged, x + 1)
    assert bias_merged is None


def test_column_parallel_lora_wrapper_delegates_to_base_layer_when_lora_inactive():
    base_layer = _DummyColumnParallelBaseLayer()
    wrapper = ColumnParallelLinearWithLoRA(base_layer)
    x = torch.randn(3, 2)

    wrapper.disable_lora = True
    out_disabled, bias_disabled = wrapper(x)
    assert base_layer.forward_calls == 1
    assert torch.equal(out_disabled, x - 1)
    assert bias_disabled is None

    wrapper.disable_lora = False
    wrapper.merged = True
    out_merged, bias_merged = wrapper(x)
    assert base_layer.forward_calls == 2
    assert torch.equal(out_merged, x - 1)
    assert bias_merged is None
