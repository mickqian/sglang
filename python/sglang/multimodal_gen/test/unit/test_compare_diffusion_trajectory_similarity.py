import numpy as np

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    summarize_output_frame_metrics,
    summarize_psnr_delta_against_official,
)


def _solid_frame(value: int, *, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_summarize_output_frame_metrics_contains_keyframes():
    reference = [_solid_frame(10), _solid_frame(20), _solid_frame(30)]
    candidate = [_solid_frame(10), _solid_frame(20), _solid_frame(30)]

    metrics = summarize_output_frame_metrics(reference, candidate)

    assert metrics["num_frames"] == 3
    assert metrics["last_frame_index"] == 2
    assert len(metrics["keyframe_metrics"]) == 3
    assert metrics["min_keyframe_psnr_db"] == float("inf")


def test_summarize_psnr_delta_against_official_zero_when_equal():
    reference = [_solid_frame(10), _solid_frame(20), _solid_frame(30)]
    candidate = [_solid_frame(10), _solid_frame(20), _solid_frame(30)]
    official = [_solid_frame(9), _solid_frame(21), _solid_frame(29)]

    summary = summarize_psnr_delta_against_official(reference, candidate, official)

    assert summary["max_keyframe_delta_psnr_db"] == 0.0
    assert summary["keyframe_delta_psnr_db"]["frame0"] == 0.0
    assert summary["keyframe_delta_psnr_db"]["mid_frame"] == 0.0
    assert summary["keyframe_delta_psnr_db"]["last_frame"] == 0.0
