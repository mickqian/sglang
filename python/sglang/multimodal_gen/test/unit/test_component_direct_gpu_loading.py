# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.bridge_loader import (
    BridgeLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class TestComponentDirectGpuLoading(unittest.TestCase):
    MODEL_INDEX = {
        "_class_name": "TestPipeline",
        "_diffusers_version": "0",
        "transformer": ["diffusers", "Transformer"],
        "transformer_2": ["diffusers", "Transformer"],
        "dual_tower_bridge": ["diffusers", "Bridge"],
        "vae": ["diffusers", "AutoencoderKL"],
    }

    @staticmethod
    def _args(component_values, *, legacy=True):
        validated = []

        def selected(component_name, *, legacy_fallback):
            if component_name in component_values:
                return component_values[component_name]
            return legacy and legacy_fallback

        return (
            SimpleNamespace(
                component_direct_gpu_weight_loading=component_values,
                direct_gpu_weight_loading=legacy,
                should_direct_gpu_weight_load_component=selected,
                validate_direct_gpu_weight_loading_component=validated.append,
            ),
            validated,
        )

    def test_exact_selection_overrides_transformer_legacy_and_enables_bridge(self):
        legacy_args, legacy_validated = self._args({})
        legacy_selected = (
            ComposedPipelineBase._validate_component_direct_gpu_weight_loading(
                self.MODEL_INDEX, legacy_args
            )
        )
        self.assertEqual(legacy_selected, {"transformer", "transformer_2"})
        self.assertEqual(legacy_validated, [])

        args, validated = self._args({"transformer": False, "dual_tower_bridge": True})

        selected = ComposedPipelineBase._validate_component_direct_gpu_weight_loading(
            self.MODEL_INDEX, args
        )

        self.assertEqual(selected, {"transformer_2", "dual_tower_bridge"})
        self.assertEqual(validated, [])

    def test_unknown_and_unsupported_selections_fail_closed(self):
        cases = (
            ({"missing": False}, "Unknown component"),
            ({"vae": True}, "not supported"),
        )
        for component_values, message in cases:
            with self.subTest(component_values=component_values):
                args, _ = self._args(component_values, legacy=False)
                with self.assertRaisesRegex(ValueError, message):
                    ComposedPipelineBase._validate_component_direct_gpu_weight_loading(
                        self.MODEL_INDEX, args
                    )

    def test_diffusers_backend_rejects_component_selection(self):
        args, _ = self._args({"transformer": True}, legacy=False)
        with self.assertRaisesRegex(ValueError, "Diffusers pipeline backend"):
            DiffusersPipeline("/unused", args)

        disabled_args, _ = self._args({"transformer": False}, legacy=False)
        with (
            patch.object(
                DiffusersPipeline, "_load_diffusers_pipeline", return_value=object()
            ),
            patch.object(DiffusersPipeline, "_detect_pipeline_type"),
        ):
            DiffusersPipeline("/unused", disabled_args, executor=object())

    def test_loader_type_does_not_replace_logical_component_name(self):
        loaded_names = []
        loader = SimpleNamespace(
            allow_global_attention_backend_fallback=False,
            load=lambda _path, _args, name, _library: (loaded_names.append(name), 0),
        )
        with patch.object(
            ComponentLoader, "for_component_type", return_value=loader
        ) as loader_for_component:
            PipelineComponentLoader.load_component(
                component_name="transformer_2",
                component_type="transformer",
                component_model_path="/unused",
                transformers_or_diffusers="diffusers",
                server_args=object(),
            )

        loader_for_component.assert_called_once_with("transformer", "diffusers", None)
        self.assertEqual(loaded_names, ["transformer_2"])

    def test_existing_optional_loader_arguments_remain_positional(self):
        loader = SimpleNamespace(
            allow_global_attention_backend_fallback=False,
            load=lambda *_args: (object(), 0),
        )
        with patch.object(
            ComponentLoader, "for_component_type", return_value=loader
        ) as loader_for_component:
            PipelineComponentLoader.load_component(
                "transformer", "/unused", "diffusers", object(), "Architecture"
            )

        loader_for_component.assert_called_once_with(
            "transformer", "diffusers", "Architecture"
        )

    def test_bridge_selection_enables_direct_load_plan(self):
        args, validated = self._args({"dual_tower_bridge": True}, legacy=False)
        args.model_paths = {}
        args.pipeline_config = SimpleNamespace(
            bridge_config=SimpleNamespace(update_model_arch=lambda _config: None)
        )
        args.should_use_fsdp_for_component = lambda _name: False
        args.should_start_component_on_cpu = lambda _name: False
        args.residency_mode = lambda _name: "resident"
        args.hsdp_replicate_dim = args.hsdp_shard_dim = 1
        args.pin_cpu_memory = False
        fake_model = SimpleNamespace(parameters=lambda: ())

        with (
            patch.object(
                BridgeLoader,
                "load_component_config",
                return_value={"_class_name": "Bridge"},
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader.ModelRegistry.resolve_model_cls",
                return_value=(object, None),
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader._list_safetensors_files",
                return_value=["weights.safetensors"],
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader.resolve_precision",
                return_value=torch.float16,
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader.get_local_torch_device",
                return_value=torch.device("cuda:0"),
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader.maybe_load_fsdp_model",
                return_value=fake_model,
            ) as load_model,
        ):
            BridgeLoader().load_customized("/model/bridge", args, "dual_tower_bridge")

        plan = load_model.call_args.kwargs["weight_load_plan"]
        self.assertEqual(validated, ["dual_tower_bridge"])
        self.assertTrue(plan.load_full_state_dict_on_device)
        self.assertEqual(plan.checkpoint_load_device, torch.device("cuda:0"))

    def test_bridge_direct_loading_rejects_prequantized_checkpoint(self):
        args, _ = self._args({"dual_tower_bridge": True}, legacy=False)

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.get_diffusers_component_config",
                return_value={
                    "_class_name": "Bridge",
                    "quantization_config": {"quant_method": "bitsandbytes"},
                },
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "bridge_loader.maybe_load_fsdp_model"
            ) as load_model,
            self.assertRaises(ComponentCheckpointUnsupportedError),
        ):
            BridgeLoader().load_customized("/model/bridge", args, "dual_tower_bridge")

        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
