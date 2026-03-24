from sglang.multimodal_gen.configs.models.dits import WanS2VConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_S2V_14B_Config
from sglang.multimodal_gen.configs.sample.wan import Wan2_2_S2V_14B_SamplingParam
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
import torch


def test_wan_s2v_sampling_defaults():
    sampling = Wan2_2_S2V_14B_SamplingParam(
        prompt="a singer on stage",
        image_path="image.png",
        audio_path="audio.wav",
    )
    assert sampling.width == 1024
    assert sampling.height == 704
    assert sampling.num_frames == 81
    assert sampling.fps == 16
    assert sampling.guidance_scale == 4.5


def test_wan_s2v_requires_image_and_audio():
    pipeline_config = Wan2_2_S2V_14B_Config()

    sampling = Wan2_2_S2V_14B_SamplingParam(
        prompt="a singer on stage",
        image_path="image.png",
        audio_path="audio.wav",
    )
    sampling._validate_with_pipeline_config(pipeline_config)

    missing_audio = Wan2_2_S2V_14B_SamplingParam(
        prompt="a singer on stage",
        image_path="image.png",
    )
    try:
        missing_audio._validate_with_pipeline_config(pipeline_config)
    except ValueError as exc:
        assert "audio_path" in str(exc)
    else:
        raise AssertionError("Expected audio_path validation failure")


def test_wan_s2v_uses_native_dit_config():
    pipeline_config = Wan2_2_S2V_14B_Config()

    assert isinstance(pipeline_config.dit_config, WanS2VConfig)
    assert pipeline_config.dit_config.arch_config.model_type == "s2v"
    assert pipeline_config.dit_config.arch_config.audio_dim == 5120


def test_wan_s2v_prepare_cfg_condition_kwargs():
    pipeline_config = Wan2_2_S2V_14B_Config()
    batch = Req()
    batch.extra["wan_s2v"] = {
        "ref_latents": torch.randn(1, 16, 1, 2, 2),
        "motion_latents": torch.randn(1, 16, 2, 2, 2),
        "cond_states": torch.randn(1, 3, 2, 2, 2),
        "audio_input": torch.randn(1, 4, 2, 8),
        "motion_frames": [17, 5],
        "drop_motion_frames": True,
    }

    pos = pipeline_config.prepare_pos_cond_kwargs(
        batch=batch,
        device=torch.device("cpu"),
        rotary_emb=None,
        dtype=torch.float32,
    )
    neg = pipeline_config.prepare_neg_cond_kwargs(
        batch=batch,
        device=torch.device("cpu"),
        rotary_emb=None,
        dtype=torch.float32,
    )

    assert pos["motion_frames"] == [17, 5]
    assert pos["drop_motion_frames"] is True
    assert torch.equal(pos["ref_latents"], batch.extra["wan_s2v"]["ref_latents"])
    assert torch.equal(
        pos["audio_input"],
        batch.extra["wan_s2v"]["audio_input"],
    )
    assert torch.count_nonzero(neg["audio_input"]) == 0
