"""NVFP4 rejects pre-Blackwell GPUs at load instead of inside the CUDA kernel."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    _linear_nvfp4_scale_weight_loader,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability

CAPABILITY_PATH = (
    "sglang.multimodal_gen.runtime.layers.quantization."
    "modelopt_quant.current_platform.get_device_capability"
)


def _config() -> ModelOptFp4Config:
    return ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, columns = scales.shape
    return (
        scales.reshape(1, rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(rows, columns)
    )


class TestModelOptFp4CapabilityGate(unittest.TestCase):
    def test_rejects_pre_blackwell_with_an_actionable_message(self):
        for major, minor, name in [(8, 6, "3090"), (8, 9, "4090"), (9, 0, "H100")]:
            with self.subTest(gpu=name):
                with patch(
                    CAPABILITY_PATH, return_value=DeviceCapability(major, minor)
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        ModelOptFp4LinearMethod(_config())
                message = str(caught.exception)
                self.assertIn(f"{major}.{minor}", message)
                self.assertIn("Blackwell", message)

    def test_allows_blackwell(self):
        # 10.0 is B200/B300, 12.0 is the consumer Blackwell line; both carry the
        # FlashInfer FP4 kernels, so neither may be rejected
        for major, minor, name in [(10, 0, "B200"), (12, 0, "RTX 5090")]:
            with self.subTest(gpu=name):
                with patch(
                    CAPABILITY_PATH, return_value=DeviceCapability(major, minor)
                ):
                    ModelOptFp4LinearMethod(_config())

    def test_undetectable_capability_is_left_alone(self):
        # the gate exists to turn a kernel crash into a message, so a GPU whose
        # capability cannot be read keeps the previous behavior rather than
        # gaining a new way to fail
        with patch(CAPABILITY_PATH, return_value=None):
            ModelOptFp4LinearMethod(_config())

    def test_swizzled_scales_are_normalized_before_tp_sharding(self):
        linear = torch.arange(256 * 8).reshape(256, 8)
        checkpoint = _swizzle_scales(linear)

        for dimension, start, expected in (
            (0, 128, linear[128:, :]),
            (1, 4, linear[:, 4:]),
        ):
            with self.subTest(dimension=dimension):
                target = torch.empty_like(expected)

                def load_tp_shard(param, loaded_weight):
                    shard = loaded_weight.narrow(
                        dimension, start, param.shape[dimension]
                    )
                    param.copy_(shard)

                _linear_nvfp4_scale_weight_loader(load_tp_shard)(target, checkpoint)
                torch.testing.assert_close(target, expected)


if __name__ == "__main__":
    unittest.main()
