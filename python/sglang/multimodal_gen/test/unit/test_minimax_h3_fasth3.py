# SPDX-License-Identifier: Apache-2.0
"""FastH3 (4-step VSA-distilled MiniMax-H3) registration and admission contracts."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import FastH3SamplingParams
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    FastH3Pipeline,
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_registry_resolves_fasth3_configs() -> None:
    info = get_model_info(FASTH3_MODEL_ID)
    assert info.sampling_param_cls is FastH3SamplingParams
    assert info.pipeline_config_cls is FastH3PipelineConfig
    assert get_non_diffusers_pipeline_name(FASTH3_MODEL_ID) == "FastH3Pipeline"
    materialized = (
        "/cache/materialized_models/"
        "FastVideo__FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-0123abcd"
    )
    assert get_model_info(materialized).sampling_param_cls is FastH3SamplingParams


def test_modular_h3_component_names_and_release_defaults() -> None:
    modular_index = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "vae": ["diffusers", "AutoencoderKLMiniMaxH3", {}],
    }
    base_pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    base_pipeline._extra_config_module_map = {}
    base_pipeline._configure_component_names(modular_index)
    assert base_pipeline._extra_config_module_map == {"video_vae": "vae"}
    base_metadata = base_pipeline._release_metadata_from_model_index(modular_index)
    assert base_metadata.tasks == ("t2va", "fl2va")

    fast_pipeline = FastH3Pipeline.__new__(FastH3Pipeline)
    fast_pipeline._extra_config_module_map = {}
    fast_metadata = fast_pipeline._release_metadata_from_model_index(modular_index)
    assert fast_metadata.tasks == ("t2va",)

    video_vae_cls, _ = ModelRegistry.resolve_model_cls("AutoencoderKLMiniMaxH3")
    audio_vae_cls, _ = ModelRegistry.resolve_model_cls(
        "AutoencoderKLMiniMaxH3Audio"
    )
    assert video_vae_cls.__name__ == "MiniMaxH3VideoVAE"
    assert audio_vae_cls.__name__ == "MiniMaxH3AudioVAE"


def test_fasth3_sampling_defaults_and_task_rejection() -> None:
    params = FastH3SamplingParams(prompt="p")
    assert params.num_inference_steps == 5
    assert params.guidance_scale == 1.0

    with pytest.raises(ValueError, match="exactly five sigma grid points"):
        FastH3SamplingParams(prompt="p", num_inference_steps=50)

    with pytest.raises(ValueError, match="distilled for t2va only"):
        FastH3SamplingParams(
            prompt="p",
            task="fl2va",
            conditions=[{"type": "image", "uri": "x.png", "role": "first_frame"}],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )


def test_fasth3_pipeline_config_gates_and_rejections() -> None:
    config = FastH3PipelineConfig()
    assert config.dit_config.arch_config.has_gate_compress
    assert config.dmd_denoising_steps == [999, 749, 500, 250]
    assert minimax_h3_time_shift_sigmas(
        num_steps=5,
        shift_scale=1.0,
        denoising_steps=config.dmd_denoising_steps,
    ) == pytest.approx([0.999, 0.749, 0.5, 0.25, 0.0])
    server_args = SimpleNamespace(pipeline_config=config)
    assert (
        MiniMaxH3TimestepPreparationStage._denoising_steps_for_request(
            SimpleNamespace(is_warmup=True), server_args
        )
        is None
    )
    assert MiniMaxH3TimestepPreparationStage._denoising_steps_for_request(
        SimpleNamespace(is_warmup=False), server_args
    ) == [999, 749, 500, 250]
    assert not MiniMaxH3PipelineConfig().dit_config.arch_config.has_gate_compress
    mapping = config.dit_config.arch_config.param_names_mapping
    source = "transformer_blocks.7.attn.to_gate_compress.weight"
    targets = [
        re.sub(pattern, target if isinstance(target, str) else target[0], source)
        for pattern, target in mapping.items()
        if re.match(pattern, source)
    ]
    assert targets == ["blocks.7.attn.to_gate_compress.weight"]

    with pytest.raises(ValueError, match="--model-variant does not apply"):
        config.validate_server_args(SimpleNamespace(model_variant="ref2va"))
    with pytest.raises(ValueError, match="no.*audited high-quality deployment"):
        config.validate_quality_deployment(server_args=None)


def test_fasth3_lora_adapter_accepts_normalized_tensors() -> None:
    model = SimpleNamespace(
        arch=SimpleNamespace(adaln_affine_input_dim=None),
        _adaln_precomputed=False,
    )
    plain = {
        "blocks.0.attn.qkv_proj.lora_A": torch.zeros(3, 64, 8),
        "blocks.0.attn.qkv_proj.lora_B": torch.zeros(3, 8, 64),
    }
    assert MiniMaxH3DiTModel.prepare_lora_adapter(model, dict(plain)) == plain


def test_fasth3_gates_stay_bf16_under_runtime_quantization() -> None:
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=FastH3PipelineConfig().dit_config,
            hf_config={},
            quant_config=Fp8Config(),
        )

    attn = model.blocks[0].attn
    assert not isinstance(attn.qkv_proj.quant_method, UnquantizedLinearMethod)
    assert isinstance(attn.to_gate_compress.quant_method, UnquantizedLinearMethod)
    assert attn.to_gate_compress.weight.dtype == torch.bfloat16
    assert attn.to_gate_compress.weight.missing_param_init == "error"
    assert model.token_refiner.blocks[0].attn.to_gate_compress is None
