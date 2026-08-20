import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.loader.artifact_manifest import (
    load_artifact_manifest_defaults,
)
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs


def _write_manifest(tmp_path, value):
    manifest = tmp_path / "sglang_artifacts.json"
    manifest.write_text(json.dumps(value))
    return manifest


def test_manifest_resolves_default_component_and_lora_sources(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "adapter.safetensors").write_bytes(b"fixture")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "base_model": "owner/base",
            "model_variant": "distilled",
            "entries": [
                {
                    "id": "encoder",
                    "path": "text_encoder",
                    "role": "component",
                    "component": "text_encoder",
                    "default": True,
                },
                {
                    "id": "turbo",
                    "path": "adapter.safetensors",
                    "role": "lora",
                    "default": True,
                    "adapter": {"alpha": 16, "scale": 0.25},
                },
            ],
            "request_defaults": {"num_inference_steps": 6},
        },
    )

    defaults = load_artifact_manifest_defaults(str(manifest))

    assert defaults.model_path == "owner/base"
    assert defaults.component_paths == {"text_encoder": str(tmp_path / "text_encoder")}
    assert defaults.lora_path == str(tmp_path / "adapter.safetensors")
    assert defaults.lora_alpha == 16
    assert defaults.lora_scale == 0.25
    assert defaults.request_defaults == {"num_inference_steps": 6}


def test_cli_values_override_manifest_defaults():
    manifest_defaults = SimpleNamespace(
        model_path="owner/base",
        model_variant="distilled",
        component_paths={"text_encoder": "owner/default-encoder"},
        component_weights_paths={"transformer": "owner/default-dit"},
        lora_path="owner/default-lora",
        lora_alpha=16,
        lora_scale=0.25,
        request_defaults={"num_inference_steps": 6},
    )
    args = argparse.Namespace(
        artifact_manifest="owner/package",
        model_path="owner/explicit-base",
        lora_scale=0.5,
        _sglang_explicit_arg_names={"artifact_manifest", "model_path", "lora_scale"},
    )
    with patch(
        "sglang.multimodal_gen.runtime.server_args.server_args."
        "load_artifact_manifest_defaults",
        return_value=manifest_defaults,
    ), patch.object(ServerArgs, "from_dict", side_effect=lambda value: value):
        values = ServerArgs.from_cli_args(
            args,
            ["--component-paths.text_encoder", "owner/explicit-encoder"],
        )

    assert values["model_path"] == "owner/explicit-base"
    assert values["lora_scale"] == 0.5
    assert values["component_paths"] == {"text_encoder": "owner/explicit-encoder"}
    assert values["component_weights_paths"] == {"transformer": "owner/default-dit"}


def test_manifest_request_defaults_are_lower_priority_than_user_values():
    server_args = SimpleNamespace(
        artifact_request_defaults={"num_inference_steps": 6, "guidance_scale": 1.0},
        pipeline_class_name=None,
        backend=None,
        model_id=None,
        pipeline_config=SimpleNamespace(),
    )
    base = SamplingParams(num_inference_steps=28, guidance_scale=4.0)
    with patch.object(
        SamplingParams, "from_pretrained", return_value=base
    ), patch.object(SamplingParams, "_adjust"), patch.object(
        SamplingParams, "_validate_with_pipeline_config"
    ):
        params = SamplingParams.from_user_sampling_params_args(
            "owner/base",
            server_args,
            guidance_scale=7.5,
        )

    assert params.num_inference_steps == 6
    assert params.guidance_scale == 7.5
