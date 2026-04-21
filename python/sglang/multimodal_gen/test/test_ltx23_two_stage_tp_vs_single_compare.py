from types import SimpleNamespace

import torch

from sglang.multimodal_gen.tools.compare_ltx23_two_stage_tp_vs_single import (
    _extract_stage_trajectory_data,
    summarize_stage_trajectory_data,
)


def test_extract_stage_trajectory_data_uses_explicit_stage_payload():
    stage_payload = {
        "stage1": {
            "timesteps": torch.tensor([1.0, 0.5]),
            "video_latents": torch.ones(1, 2, 3),
        }
    }
    result = SimpleNamespace(
        stage_trajectory_data=stage_payload,
        trajectory_timesteps=None,
        trajectory_latents=None,
        trajectory_audio_latents=None,
    )

    extracted = _extract_stage_trajectory_data(result)

    assert extracted is stage_payload


def test_extract_stage_trajectory_data_falls_back_to_stage2():
    result = SimpleNamespace(
        stage_trajectory_data=None,
        trajectory_timesteps=torch.tensor([1.0, 0.5]),
        trajectory_latents=torch.ones(1, 2, 3),
        trajectory_audio_latents=torch.zeros(1, 2, 4),
    )

    extracted = _extract_stage_trajectory_data(result)

    assert set(extracted) == {"stage2"}
    assert set(extracted["stage2"]) == {"timesteps", "video_latents", "audio_latents"}


def test_summarize_stage_trajectory_data_reports_video_and_audio_metrics():
    reference = {
        "stage1": {
            "timesteps": torch.tensor([1.0, 0.5]),
            "video_latents": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "audio_latents": torch.tensor([[[0.5, 1.0], [1.5, 2.0]]]),
        },
        "stage2": {
            "timesteps": torch.tensor([0.9, 0.1]),
            "video_latents": torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        },
    }
    candidate = {
        "stage1": {
            "timesteps": torch.tensor([1.0, 0.5]),
            "video_latents": torch.tensor([[[1.0, 2.5], [3.0, 3.5]]]),
            "audio_latents": torch.tensor([[[0.5, 1.0], [1.0, 2.0]]]),
        },
        "stage2": {
            "timesteps": torch.tensor([0.9, 0.1]),
            "video_latents": torch.tensor([[[5.0, 6.5], [7.5, 8.0]]]),
        },
    }

    summary = summarize_stage_trajectory_data(reference, candidate, step_index=-1)

    assert set(summary) == {"stage1", "stage2"}
    assert "video_trajectory_metrics" in summary["stage1"]
    assert "audio_trajectory_metrics" in summary["stage1"]
    assert "video_trajectory_metrics" in summary["stage2"]
    assert summary["stage1"]["video_trajectory_metrics"]["selected_step_index"] == 1
    assert summary["stage1"]["audio_trajectory_metrics"]["selected_step_index"] == 1
    assert summary["stage2"]["video_trajectory_metrics"]["num_steps"] == 2


def test_summarize_stage_trajectory_data_reports_extra_tensor_metrics():
    reference = {
        "stage2": {
            "timesteps": torch.tensor([0.9, 0.1]),
            "video_latents": torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
            "video_latents_pre_renoise": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "video_latents_post_renoise": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        }
    }
    candidate = {
        "stage2": {
            "timesteps": torch.tensor([0.9, 0.1]),
            "video_latents": torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
            "video_latents_pre_renoise": torch.tensor([[1.0, 2.5], [3.0, 4.0]]),
            "video_latents_post_renoise": torch.tensor([[2.5, 3.0], [4.0, 5.0]]),
        }
    }

    summary = summarize_stage_trajectory_data(reference, candidate, step_index=-1)

    assert "extra_tensor_metrics" in summary["stage2"]
    assert set(summary["stage2"]["extra_tensor_metrics"]) == {
        "video_latents_post_renoise",
        "video_latents_pre_renoise",
    }
    assert (
        summary["stage2"]["extra_tensor_metrics"]["video_latents_pre_renoise"]["rmse"]
        > 0.0
    )
