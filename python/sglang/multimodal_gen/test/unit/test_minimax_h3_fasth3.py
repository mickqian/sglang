# SPDX-License-Identifier: Apache-2.0
"""FastH3 (4-step VSA-distilled MiniMax-H3) registration and admission contracts."""

from __future__ import annotations

import re
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import FastH3SamplingParams
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
from sglang.multimodal_gen.runtime.loader.gguf_weights import GGUFTensorMeta
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3FinalLayer,
    _pdd_partition_widths,
    _pdd_plan_from_sigmas,
    _pdd_retime_sigmas,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    FastH3Pipeline,
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
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


def test_registry_routes_modular_fasth3_class_to_native_pipeline() -> None:
    model_id = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    try:
        with patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            return_value={"_class_name": "MiniMaxH3ModularPipeline"},
        ):
            info = get_model_info(model_id)
    finally:
        get_model_info.cache_clear()
        _get_config_info.cache_clear()

    assert info is not None
    assert info.pipeline_cls is FastH3Pipeline
    assert info.pipeline_config_cls is FastH3PipelineConfig
    assert info.sampling_param_cls is FastH3SamplingParams


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
    audio_vae_cls, _ = ModelRegistry.resolve_model_cls("AutoencoderKLMiniMaxH3Audio")
    assert video_vae_cls.__name__ == "MiniMaxH3VideoVAE"
    assert audio_vae_cls.__name__ == "MiniMaxH3AudioVAE"


def test_root_modular_h3_ref2va_routing() -> None:
    modular_index = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel", {}],
    }
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.model_path = "org/root-ref2va-bundle"
    pipeline._extra_config_module_map = {}
    pipeline.server_args = SimpleNamespace(
        model_variant="ref2va",
        model_subfolder=None,
        revision=None,
    )

    with patch.object(
        ComposedPipelineBase, "_load_config", return_value=modular_index
    ):
        assert pipeline._load_config() is modular_index

    assert pipeline.server_args.model_subfolder is None
    assert pipeline.default_model_subfolder == "Ref2VA"
    assert pipeline._extra_config_module_map == {"transformer": "transformer_ref"}
    assert pipeline.release_metadata.partition == "ref2va"
    assert pipeline.release_metadata.tasks == ("ref2va",)


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


def test_h3_native_lora_declares_schedule_and_reorders_grouped_qkv() -> None:
    model = SimpleNamespace(
        arch=SimpleNamespace(num_attention_heads=2, attention_head_dim=2)
    )
    grouped = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    adapter = {
        "transformer.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(1, 4),
        "transformer.blocks.0.attn.qkv_proj.lora_B.weight": grouped,
    }
    ordinary, state = MiniMaxH3DiTModel.split_lora_runtime_state(
        model,
        adapter,
        {
            "key_format": "minimax-h3-native",
            "qkv_layout": "grouped",
            "base_schedule": "1.0,0.7,0.4,0.15,0.0",
            "tasks": "t2va",
        },
    )

    assert state == {
        "base_schedule": (1.0, 0.7, 0.4, 0.15, 0.0),
        "tasks": frozenset({"t2va"}),
    }
    torch.testing.assert_close(
        ordinary["transformer.blocks.0.attn.qkv_proj.lora_B.weight"],
        torch.tensor(
            [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11], dtype=torch.float32
        ).reshape(12, 1),
    )
    unannotated, unannotated_state = MiniMaxH3DiTModel.split_lora_runtime_state(
        model, adapter, {}
    )
    assert not unannotated_state
    torch.testing.assert_close(
        unannotated["transformer.blocks.0.attn.qkv_proj.lora_B.weight"],
        ordinary["transformer.blocks.0.attn.qkv_proj.lora_B.weight"],
    )

    with pytest.raises(ValueError, match="requires qkv_layout"):
        MiniMaxH3DiTModel.split_lora_runtime_state(
            model,
            adapter,
            {
                "key_format": "minimax-h3-native",
                "base_schedule": "1.0,0.0",
                "tasks": "t2va",
            },
        )


def test_h3_native_lora_schedule_drives_timestep_preparation() -> None:
    transformer = SimpleNamespace(
        get_lora_denoise_schedule=lambda: (
            (1.0, 0.7, 0.4, 0.15, 0.0),
            frozenset({"t2va"}),
        )
    )
    stage = MiniMaxH3TimestepPreparationStage(transformer=transformer)
    plan = SimpleNamespace(
        task="t2va",
        flow_shift=None,
        audio_flow_shift=None,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
    )
    batch = SimpleNamespace(
        extra={"explicit_fields": []},
        num_inference_steps=50,
        sampling_params=None,
        is_warmup=False,
    )

    stage._generate_sigmas_from_plan(batch, plan, None)
    assert len(batch.extra["minimax_h3_sigmas"]["video"]) == 5
    assert batch.extra["minimax_h3_sigmas"]["video"][1] == pytest.approx(
        12.0 * 0.7 / (1.0 + 11.0 * 0.7)
    )

    batch.extra = {"explicit_fields": ["num_inference_steps"]}
    batch.num_inference_steps = 5
    with pytest.raises(ValueError, match="requires num_inference_steps=4"):
        stage._generate_sigmas_from_plan(batch, plan, None)


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


def test_h3_pdd_head_bank_is_split_from_plain_lora_and_tp_sharded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_tp_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_tp_rank", lambda: 1
    )
    video = SimpleNamespace(weight=torch.empty(2, 3))
    audio = SimpleNamespace(weight=torch.empty(1, 3))
    model = SimpleNamespace(
        arch=SimpleNamespace(hidden_size=3),
        device=torch.device("cpu"),
        final_layer=SimpleNamespace(video_out=video, audio_out=audio),
    )
    adapter = {
        "proj_out.weight": torch.arange(4 * 4 * 3).view(4, 4, 3).float(),
        "proj_out.bias": torch.arange(4 * 4).view(4, 4).float(),
        "audio_proj_out.weight": torch.arange(4 * 2 * 3).view(4, 2, 3).float(),
        "audio_proj_out.bias": torch.arange(4 * 2).view(4, 2).float(),
        "transformer_blocks.0.attn.to_q.lora_A": torch.ones(1, 3),
    }

    ordinary, state = MiniMaxH3DiTModel.split_lora_runtime_state(
        model,
        adapter,
        {"pdd_num_steps": "4", "pdd_block_size": "2"},
    )

    assert list(ordinary) == ["transformer_blocks.0.attn.to_q.lora_A"]
    torch.testing.assert_close(state["video_weight"], adapter["proj_out.weight"][:, 2:])
    torch.testing.assert_close(
        state["audio_bias"], adapter["audio_proj_out.bias"][:, 1:]
    )
    assert (state["num_steps"], state["block_size"]) == (4, 2)

    applied = []
    model.final_layer.set_pdd_heads = applied.append
    MiniMaxH3DiTModel.set_lora_runtime_state(model, [state], [1.0])
    MiniMaxH3DiTModel.set_lora_runtime_state(model, [], [])
    assert applied == [state, None]
    with pytest.raises(ValueError, match="absolute weights"):
        MiniMaxH3DiTModel.set_lora_runtime_state(model, [state], [0.5])


def test_h3_pdd_plan_matches_released_shifted_fine_grid() -> None:
    sigmas = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=12.0)
    plans = _pdd_plan_from_sigmas(sigmas, num_steps=32, block_size=4)
    base = torch.linspace(1.0, 0.0, 33, dtype=torch.float64)
    fine = 12.0 * base / (1.0 + 11.0 * base)
    expected = fine[:4] - fine[1:5]
    expected = (expected / expected.sum()).to(torch.float32)

    assert len(plans) == 8
    assert [start for start, _ in plans] == list(range(0, 32, 4))
    torch.testing.assert_close(plans[0][1], expected)
    assert all(
        float(coefficients.sum()) == pytest.approx(1.0) for _, coefficients in plans
    )
    hidden = torch.arange(12, dtype=torch.float32).view(4, 3)
    weight = torch.arange(32 * 2 * 3, dtype=torch.float32).view(32, 2, 3)
    bias = torch.arange(32 * 2, dtype=torch.float32).view(32, 2)
    full_plan = torch.zeros(32)
    start, coefficients = plans[2]
    full_plan[start : start + coefficients.numel()] = coefficients
    expected_output = torch.nn.functional.linear(
        hidden,
        torch.einsum("n,noi->oi", full_plan, weight),
        torch.einsum("n,no->o", full_plan, bias),
    )
    torch.testing.assert_close(
        MiniMaxH3FinalLayer._pdd_linear(hidden, plans[2], weight, bias),
        expected_output,
    )

    with pytest.raises(ValueError, match="uniformly spaced shifted schedule"):
        _pdd_plan_from_sigmas(
            [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0],
            num_steps=32,
            block_size=4,
        )


def test_h3_pdd_retimes_nonuniform_trained_envelope() -> None:
    sigmas = minimax_h3_time_shift_sigmas(num_steps=6, shift_scale=12.0)
    retimed = _pdd_retime_sigmas(
        sigmas, num_steps=32, block_size=4, expected_shift=12.0
    )
    plans = _pdd_plan_from_sigmas(retimed, num_steps=32, block_size=4)

    assert _pdd_partition_widths(32, 5, 4) == [8, 8, 8, 4, 4]
    assert [start for start, _ in plans] == [0, 8, 16, 24, 28]
    assert [coefficients.numel() for _, coefficients in plans] == [8, 8, 8, 4, 4]
    assert retimed != sigmas

    with pytest.raises(ValueError, match="requires flow shift 3"):
        _pdd_retime_sigmas(sigmas, num_steps=32, block_size=4, expected_shift=3.0)

    with pytest.raises(ValueError, match="cannot tile"):
        _pdd_retime_sigmas(
            minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=12.0),
            num_steps=32,
            block_size=4,
        )


def test_h3_pdd_comfy_sidecar_is_normalized_without_ignored_tensors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_tp_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_tp_rank", lambda: 1
    )
    video = SimpleNamespace(weight=torch.arange(2 * 3).view(2, 3).float() + 6)
    audio = SimpleNamespace(weight=torch.empty(1, 3))
    model = SimpleNamespace(
        arch=SimpleNamespace(
            hidden_size=3,
            time_embed_dim=3,
            num_attention_heads=1,
            attention_head_dim=1,
        ),
        device=torch.device("cpu"),
        final_layer=SimpleNamespace(video_out=video, audio_out=audio),
    )
    model._normalize_pdd_sidecar = MethodType(
        MiniMaxH3DiTModel._normalize_pdd_sidecar, model
    )
    adapter = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(1, 3),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones(3, 1),
        "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(1.0),
        "h3_pdd.adaln.blocks.0.lora_A": torch.ones(1, 3),
        "h3_pdd.adaln.blocks.0.lora_B": torch.ones(6, 1),
        "h3_pdd.adaln.blocks.0.alpha": torch.tensor(1.0),
        "h3_pdd.bank.video.weight": torch.ones(4, 4, 3),
        "h3_pdd.bank.video.bias": torch.ones(4, 4),
        "h3_pdd.bank.audio.weight": torch.ones(4, 2, 3),
        "h3_pdd.bank.audio.bias": torch.ones(4, 2),
        "h3_pdd.base_video_out": torch.arange(4 * 3).view(4, 3).float(),
        "h3_pdd.silu_temb_grid": torch.ones(2, 3),
        "h3_pdd.backbone_probe": torch.ones(1, 3, dtype=torch.int8),
        "h3_pdd.backbone_probe_scale": torch.ones(1, 1),
    }
    metadata = {
        "pdd_num_steps": "4",
        "pdd_block_size": "2",
        "pdd_shift_video": "12.0",
        "pdd_shift_audio": "3.0",
        "pdd_grid_rows": "2",
        "adaln_modules": "1",
        "backbone_modules": "1",
        "h3_pdd_backbone": "full",
    }

    ordinary, state = MiniMaxH3DiTModel.split_lora_runtime_state(
        model, adapter, metadata
    )

    assert not any(name.startswith("h3_pdd.") for name in ordinary)
    assert "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight" in ordinary
    torch.testing.assert_close(
        state["video_weight"], adapter["h3_pdd.bank.video.weight"][:, 2:]
    )
    assert (state["num_steps"], state["block_size"]) == (4, 2)
    assert (state["shift_video"], state["shift_audio"]) == (12.0, 3.0)

    invalid = dict(adapter)
    invalid["h3_pdd.base_video_out"] = torch.zeros_like(
        adapter["h3_pdd.base_video_out"]
    )
    with pytest.raises(ValueError, match="fingerprint is degenerate"):
        MiniMaxH3DiTModel.split_lora_runtime_state(model, invalid, metadata)


def test_h3_pdd_synthetic_warmup_keeps_runtime_head_shape() -> None:
    model = SimpleNamespace(
        final_layer=SimpleNamespace(pdd_num_steps=32, pdd_block_size=4)
    )

    plans = MiniMaxH3DiTModel.prepare_pdd_plans(
        model,
        [1.0, 0.0],
        [1.0, 0.0],
        device=torch.device("cpu"),
        synthetic_warmup=True,
    )

    assert plans is not None and len(plans) == 1
    assert plans[0][0][0] == 0
    torch.testing.assert_close(plans[0][0][1], torch.ones(1))


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


def test_fasth3_gate_uses_checkpoint_declared_packed_layout() -> None:
    _ensure_single_process_parallel_runtime()
    prefix = "blocks.0.attn.to_gate_compress"
    hidden_size = FastH3PipelineConfig().dit_config.arch_config.hidden_size
    metadata = GGUFTensorMeta(
        ggml_type=12,
        logical_shape=(hidden_size, hidden_size),
        stored_shape=(hidden_size, hidden_size // 256 * 144),
        stored_dtype=torch.uint8,
        param_name=f"{prefix}.qweight",
    )
    quant_config = GGUFConfig("gate.gguf", {f"{prefix}.weight": metadata})

    assert quant_config.has_packed_weight(prefix)
    assert not quant_config.has_packed_weight("blocks.0.attn.qkv_proj")

    class GateCheckpointConfig(Fp8Config):
        def has_packed_weight(self, candidate: str) -> bool:
            return candidate == prefix

    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=FastH3PipelineConfig().dit_config,
            hf_config={},
            quant_config=GateCheckpointConfig(),
        )

    gate = model.blocks[0].attn.to_gate_compress
    assert gate is not None
    assert not isinstance(gate.quant_method, UnquantizedLinearMethod)
    assert model.token_refiner.blocks[0].attn.to_gate_compress is None
