import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.configs.models.encoders.clip import CLIPVisionConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.image_encoder_loader import (
    ImageEncoderLoader,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config as SRTFp8Config


class TestImageEncoderQuantizationAdmission(unittest.TestCase):
    def setUp(self):
        self.loader = ImageEncoderLoader()
        load_native_patcher = mock.patch.object(
            self.loader, "load_native", return_value=object()
        )
        self.load_native = load_native_patcher.start()
        self.addCleanup(load_native_patcher.stop)
        load_model_patcher = mock.patch.object(
            self.loader, "load_model", return_value=object()
        )
        self.load_model = load_model_patcher.start()
        self.addCleanup(load_model_patcher.stop)
        self.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                image_encoder_config=CLIPVisionConfig(),
                image_encoder_precision="bf16",
                native_only_components=(),
            ),
            encoder_parallel="replicate",
            resolve_component_attention_backend=lambda _name: (None, None),
        )

    def _component_config(self, architecture, *, quantized):
        config = {"architectures": [architecture]}
        if quantized:
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        return config

    def _config_patch(self, config):
        return mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "image_encoder_loader.get_diffusers_component_config",
            return_value=config,
        )

    def _load(self):
        return self.loader.load(
            "/model/image_encoder", self.server_args, "image_encoder", "transformers"
        )

    def test_serialized_fp8_clip_checkpoint_uses_srt_adapter(self):
        config = self._component_config("CLIPVisionModelWithProjection", quantized=True)
        with self._config_patch(config):
            self.loader.load_customized("/model/image_encoder", self.server_args)

        self.assertIsInstance(
            self.server_args.pipeline_config.image_encoder_config.quant_config,
            SRTFp8Config,
        )

    def test_bitsandbytes_clip_checkpoint_uses_transformers_loader(self):
        config = self._component_config(
            "CLIPVisionModelWithProjection", quantized=False
        )
        config["quantization_config"] = {
            "load_in_4bit": True,
            "quant_method": "bitsandbytes",
        }

        with self._config_patch(config):
            self._load()

        self.load_native.assert_called_once_with(
            "/model/image_encoder",
            self.server_args,
            "transformers",
            "image_encoder",
        )

    def test_unknown_quantized_architecture_does_not_fall_back(self):
        config = self._component_config("UnknownVisionModel", quantized=True)
        with self._config_patch(config), self.assertRaises(
            ComponentCheckpointUnsupportedError
        ):
            self._load()
        self.load_native.assert_not_called()

    def test_admitted_quantized_native_failure_does_not_fall_back(self):
        config = self._component_config("CLIPVisionModelWithProjection", quantized=True)
        self.load_model.side_effect = RuntimeError("synthetic post-load failure")
        with self._config_patch(config), self.assertRaisesRegex(
            RuntimeError, "native fallback is disabled"
        ):
            self._load()
        self.load_native.assert_not_called()

    def test_unknown_unquantized_architecture_keeps_native_fallback(self):
        config = self._component_config("UnknownVisionModel", quantized=False)
        with self._config_patch(config):
            self._load()
        self.load_native.assert_called_once()
