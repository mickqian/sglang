import gc
import pathlib
import unittest
import weakref
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_2_TI2V_5B_Config,
    Wan2_2_I2V_A14B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import vae_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _backfill_ltx2_audio_vae_latent_stats,
    _match_checkpoint_dtypes,
    _require_native_loader_for_quantized_vae,
    _should_use_channels_last_3d,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    checkpoint_bytes,
    keep_checkpoint_mapped,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    host_memory_budget,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae


class _FakeServerArgs:
    def __init__(self, pipeline_config, num_gpus=1):
        self.pipeline_config = pipeline_config
        self.num_gpus = num_gpus
        self.model_paths = {}
        self.component_weights_paths = {}
        self.revision = "test-revision"
        self.trust_remote_code = True
        self.layerwise_components = set()
        self.component_quantizations = {}
        self.direct_components = set()
        self.validated_direct_components = []

    def resolve_component_attention_backend(self, _component_name):
        return None, None

    def should_start_component_on_cpu(self, _component_name):
        return False

    def should_configure_layerwise_offload_for_lazy_component(self, component_name):
        return component_name in self.layerwise_components

    def should_direct_gpu_weight_load_component(
        self, component_name, *, legacy_fallback
    ):
        return component_name in self.direct_components

    def validate_direct_gpu_weight_loading_component(self, component_name):
        self.validated_direct_components.append(component_name)


class _RecordingVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state = config.state
        self.state.construct_device = torch.empty(0).device.type
        self.weight = nn.Parameter(torch.empty(1))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.state.load_args = (strict, assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class MiniMaxH3VideoVAE(_RecordingVAE):
    pass


class MiniMaxH3AudioVAE(_RecordingVAE):
    pass


MiniMaxH3VideoVAE.__module__ = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3"
MiniMaxH3AudioVAE.__module__ = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3"


class TestDeploymentBytesRoot(unittest.TestCase):
    """A hub repo id is not a directory; the component path always is."""

    def test_the_component_parent_carries_the_variant_weight(self):
        with TemporaryDirectory() as root:
            variant = pathlib.Path(root) / "FL2VA"
            (variant / "video_vae").mkdir(parents=True)
            (variant / "transformer").mkdir()
            (variant / "video_vae" / "w.safetensors").write_bytes(b"x" * 128)
            (variant / "transformer" / "w.safetensors").write_bytes(b"x" * 512)
            self.assertEqual(
                checkpoint_bytes(str(variant)),
                640,
                "the parent of a component dir sums every sibling's shards",
            )
            self.assertEqual(
                checkpoint_bytes("MiniMaxAI/MiniMax-H3"),
                0,
                "a repo id globs nothing -- which is why the gate must never "
                "be fed one",
            )


class TestKeepCheckpointMapped(unittest.TestCase):
    """The mapping is for hosts that cannot afford the whole deployment."""

    def test_a_small_deployment_on_a_roomy_host_copies(self):
        with unittest.mock.patch.object(
            host_memory_budget, "host_memory_available_bytes", lambda: 64 * 1024**3
        ):
            self.assertFalse(
                keep_checkpoint_mapped(weight_bytes=3 * 1024**3, component="vae (VAE)"),
                "copies are the faster choice when the host has room: their "
                "pages are resident where a mapping's first use pays a fault",
            )

    def test_a_deployment_larger_than_the_host_stays_mapped(self):
        with unittest.mock.patch.object(
            host_memory_budget, "host_memory_available_bytes", lambda: 19 * 1024**3
        ):
            self.assertTrue(
                keep_checkpoint_mapped(
                    weight_bytes=117 * 1024**3, component="vae (VAE)"
                )
            )


class TestMatchCheckpointDtypes(unittest.TestCase):
    """Assignment replaces a parameter, so only matching dtypes may stay mapped."""

    def test_a_matching_tensor_is_left_alone(self):
        loaded = {"w": torch.zeros(4, dtype=torch.float32)}
        before = loaded["w"]
        _match_checkpoint_dtypes(loaded, {"w": torch.zeros(4, dtype=torch.float32)})
        self.assertIs(loaded["w"], before)

    def test_a_mismatched_tensor_is_converted(self):
        loaded = {"w": torch.zeros(4, dtype=torch.float32)}
        _match_checkpoint_dtypes(loaded, {"w": torch.zeros(4, dtype=torch.bfloat16)})
        self.assertEqual(loaded["w"].dtype, torch.bfloat16)

    def test_a_tensor_the_module_does_not_want_is_left_alone(self):
        loaded = {"extra": torch.zeros(4, dtype=torch.float32)}
        before = loaded["extra"]
        _match_checkpoint_dtypes(loaded, {})
        self.assertIs(loaded["extra"], before)


class TestVAELoader(unittest.TestCase):
    @staticmethod
    def _pipeline_config(state, native_components):
        config = SimpleNamespace(state=state, update_model_arch=lambda _config: None)
        return SimpleNamespace(
            native_only_components=native_components,
            vae_config=config,
            vae_precision="fp32",
            audio_vae_config=config,
            audio_vae_precision="fp32",
        )

    def _run_customized(
        self,
        state,
        *,
        component_name="video_vae",
        vae_cls=MiniMaxH3VideoVAE,
        direct=True,
        header_quantized=False,
        source_dtype=torch.float32,
    ):
        state.load_calls = []
        server_args = _FakeServerArgs(
            self._pipeline_config(state, ("video_vae", "audio_vae"))
        )
        if direct:
            server_args.direct_components.add(component_name)
        loader = vae_loader.VAELoader()

        def load_file(path, **kwargs):
            state.load_calls.append((path, kwargs))
            tensor = torch.ones(1, dtype=source_dtype)
            state.source_ref = weakref.ref(tensor)
            state.source_data_ptr = tensor.data_ptr()
            return {"weight": tensor}

        def hold_decoder(_model, _args, _name, component_path):
            gc.collect()
            state.hold_path = component_path
            state.released_at_hold = state.source_ref() is None

        def optimize_vae(model):
            gc.collect()
            state.released_at_optimize = state.source_ref() is None
            return model

        with TemporaryDirectory() as root:
            checkpoint = pathlib.Path(root) / "weights.safetensors"
            checkpoint.touch()
            component_config = {
                "_class_name": vae_cls.__name__,
                "auto_map": {"AutoModel": "remote_module.RemoteVAE"},
            }
            with (
                patch.multiple(
                    vae_loader,
                    get_diffusers_component_config=Mock(return_value=component_config),
                    safetensors_declares_quantization=Mock(
                        return_value=header_quantized
                    ),
                    safetensors_load_file=load_file,
                    _hold_decoder_weights_in_decode_dtype=hold_decoder,
                ),
                patch.object(
                    vae_loader.ModelRegistry,
                    "resolve_model_cls",
                    return_value=(vae_cls, vae_cls.__name__),
                ),
                patch.object(
                    loader,
                    "target_device",
                    return_value=torch.device("cuda:2" if direct else "cpu"),
                ),
                patch.object(
                    vae_loader.current_platform,
                    "optimize_vae",
                    new=optimize_vae,
                ),
            ):
                result = loader.load_customized(
                    str(checkpoint), server_args, component_name
                )
        return result, server_args

    def test_native_h3_vaes_load_directly_to_cuda_and_assign(self):
        cases = (
            ("video_vae", MiniMaxH3VideoVAE),
            ("audio_vae", MiniMaxH3AudioVAE),
        )
        for component_name, vae_cls in cases:
            with self.subTest(component_name=component_name):
                state = SimpleNamespace()
                loaded, server_args = self._run_customized(
                    state, component_name=component_name, vae_cls=vae_cls
                )
                self.assertEqual(
                    server_args.validated_direct_components, [component_name]
                )
                self.assertEqual(state.construct_device, "cpu")
                self.assertEqual(state.load_calls[0][1], {"device": "cuda:2"})
                self.assertEqual(state.load_args, (True, True))
                self.assertEqual(loaded.weight.data_ptr(), state.source_data_ptr)
                self.assertEqual(state.hold_path, "")
                self.assertTrue(state.released_at_hold)
                self.assertTrue(state.released_at_optimize)

    def test_direct_vae_rejects_quantized_header_and_raw_dtype(self):
        for declared, source_dtype, message in (
            (True, torch.float32, "unquantized safetensors"),
            (False, torch.int8, "torch.int8"),
        ):
            with self.subTest(declared=declared, source_dtype=source_dtype):
                state = SimpleNamespace()
                with self.assertRaisesRegex(
                    ComponentCheckpointUnsupportedError, message
                ):
                    self._run_customized(
                        state,
                        header_quantized=declared,
                        source_dtype=source_dtype,
                    )

    def test_direct_vae_rejects_nonnative_and_unrecognized_native_classes(self):
        cases = (
            (
                (),
                {"_class_name": "RemoteVAE", "auto_map": {"AutoModel": "x.Remote"}},
                (_RecordingVAE, "RemoteVAE"),
                "auto_map",
            ),
            (
                ("video_vae",),
                {"_class_name": "OtherVAE"},
                (_RecordingVAE, "OtherVAE"),
                "MiniMaxH3VideoVAE",
            ),
        )
        for native_components, config, resolved, message in cases:
            with self.subTest(config=config):
                state = SimpleNamespace()
                server_args = _FakeServerArgs(
                    self._pipeline_config(state, native_components)
                )
                server_args.direct_components.add("video_vae")
                loader = vae_loader.VAELoader()

                with (
                    patch.object(
                        vae_loader,
                        "get_diffusers_component_config",
                        return_value=config,
                    ),
                    patch.object(
                        vae_loader.ModelRegistry,
                        "resolve_model_cls",
                        return_value=resolved,
                    ),
                    patch.object(
                        loader, "target_device", return_value=torch.device("cpu")
                    ),
                ):
                    with self.assertRaisesRegex(
                        ComponentCheckpointUnsupportedError, message
                    ):
                        loader.load_customized(
                            "/unused/video_vae",
                            server_args,
                            "video_vae",
                        )
                self.assertTrue(
                    loader.should_raise_customized_load_error(server_args, "video_vae")
                )

    def test_disabled_direct_vae_preserves_cpu_checkpoint_load(self):
        state = SimpleNamespace()
        self._run_customized(state, direct=False)
        self.assertEqual(state.load_calls[0][1], {})
        self.assertEqual(state.load_args, (True, False))

    def test_weights_override_keeps_base_component_config(self):
        loader = vae_loader.VAELoader()
        server_args = _FakeServerArgs(QwenImagePipelineConfig())
        server_args.component_weights_paths = {
            "audio_vae": "owner/repo/audio_vae.safetensors"
        }

        with (
            patch.object(vae_loader, "resolve_weight", return_value="resolved"),
            patch.object(
                vae_loader,
                "materialize_weight",
                return_value="/cache/audio.safetensors",
            ),
        ):
            self.assertEqual(
                loader.resolve_model_weights_path(
                    "/base/audio_vae", server_args, "audio_vae"
                ),
                "/cache/audio.safetensors",
            )

    def test_mps_layerwise_load_uses_residency_api(self):
        loader = vae_loader.VAELoader()
        server_args = _FakeServerArgs(QwenImagePipelineConfig())
        server_args.layerwise_components.add("vae")

        with patch.object(vae_loader.current_platform, "is_mps", return_value=True):
            self.assertEqual(
                loader.customized_load_kwargs_for_component(server_args, "vae"),
                {"cpu_offload_flag": True},
            )
            self.assertEqual(
                loader.customized_load_kwargs_for_component(server_args, "audio_vae"),
                {},
            )

    def test_quantized_vae_admission_leaves_plain_configs_unchanged(self):
        _require_native_loader_for_quantized_vae(
            {"_class_name": "AutoencoderKL"}, "vae"
        )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "compression_config"
        ):
            _require_native_loader_for_quantized_vae(
                {
                    "_class_name": "AutoencoderKL",
                    "compression_config": {"quant_method": "compressed-tensors"},
                },
                "vae",
            )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            r"text_config\.quantization_config",
        ):
            _require_native_loader_for_quantized_vae(
                {
                    "_class_name": "AutoencoderKL",
                    "text_config": {
                        "quantization_config": {
                            "quant_method": "bitsandbytes",
                            "load_in_4bit": True,
                        }
                    },
                },
                "vae",
            )

    def test_quantized_vae_routes_to_diffusers_native_loader(self):
        loader = vae_loader.VAELoader()
        server_args = _FakeServerArgs(QwenImagePipelineConfig())
        native_vae = nn.Linear(1, 1)

        with (
            TemporaryDirectory() as component_path,
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value={
                    "_class_name": "AutoencoderKL",
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_4bit": True,
                    },
                },
            ),
            patch(
                "diffusers.AutoModel.from_pretrained",
                return_value=native_vae,
            ) as native_load,
            patch.object(loader, "target_device", return_value=torch.device("cpu")),
            patch.object(native_vae, "to", wraps=native_vae.to) as module_to,
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                side_effect=[10.0, 9.0],
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.get_memory_usage_of_component",
                return_value=1.0,
            ),
        ):
            loaded, consumed = loader.load(
                component_path, server_args, "vae", "diffusers"
            )

        self.assertIs(loaded, native_vae)
        self.assertFalse(loaded.training)
        self.assertEqual(consumed, 1.0)
        self.assertEqual(server_args.model_paths["vae"], component_path)
        native_load.assert_called_once_with(
            component_path,
            revision="test-revision",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        module_to.assert_called_once_with(torch.device("cpu"))

    def test_native_only_quantized_vae_fails_closed(self):
        pipeline_config = QwenImagePipelineConfig()
        pipeline_config.native_only_components = ("vae",)
        server_args = _FakeServerArgs(pipeline_config)
        loader = vae_loader.VAELoader()

        with (
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value={
                    "_class_name": "AutoencoderKL",
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_4bit": True,
                    },
                },
            ),
            patch("diffusers.AutoModel.from_pretrained") as native_load,
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=10.0,
            ),
        ):
            with self.assertRaisesRegex(
                ComponentCheckpointUnsupportedError, "native-only SGLang"
            ):
                loader.load("/quantized/vae", server_args, "vae", "diffusers")

        native_load.assert_not_called()

    def test_backfill_ltx2_audio_vae_latent_stats_maps_official_keys(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([3.0, 4.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_does_not_override_existing(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
            "latents_mean": torch.tensor([5.0, 6.0]),
            "latents_std": torch.tensor([7.0, 8.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([7.0, 8.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_skips_non_audio_vae(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0]),
            "per_channel_statistics.std-of-means": torch.tensor([2.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "vae")

        self.assertNotIn("latents_mean", loaded)
        self.assertNotIn("latents_std", loaded)

    def test_channels_last_3d_defaults_true_for_qwen_image_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertTrue(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_fast_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(FastWan2_2_TI2V_5B_Config(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(Wan2_2_I2V_A14B_Config(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_can_be_disabled_by_env(self):
        with (
            patch.dict(
                "os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "false"}
            ),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_can_be_enabled_by_env(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "true"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_auto_uses_model_policy(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "auto"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            wan_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            ltx_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)

            self.assertTrue(_should_use_channels_last_3d(wan_args, "video_vae"))
            self.assertFalse(_should_use_channels_last_3d(ltx_args, "video_vae"))

    def test_channels_last_3d_skips_non_video_vae_components(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "audio_vae"))

    def test_channels_last_3d_skips_unsupported_platforms(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=False),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_skips_non_cuda_platforms(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=False),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertIs(out, x)

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_uses_channels_last_3d_on_cuda(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=True),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))


if __name__ == "__main__":
    unittest.main()
