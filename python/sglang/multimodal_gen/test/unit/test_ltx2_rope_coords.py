import torch

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2VideoTransformer3DModel


def test_quantize_video_rope_coords_to_hidden_dtype_enabled():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.quantize_video_rope_coords_to_hidden_dtype = True

    video_coords = torch.randn(1, 3, 4, 2, dtype=torch.float32)
    quantized = model._maybe_quantize_video_rope_coords(video_coords, torch.bfloat16)

    assert quantized.dtype == torch.bfloat16


def test_quantize_video_rope_coords_to_hidden_dtype_disabled():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.quantize_video_rope_coords_to_hidden_dtype = False

    video_coords = torch.randn(1, 3, 4, 2, dtype=torch.float32)
    unchanged = model._maybe_quantize_video_rope_coords(video_coords, torch.bfloat16)

    assert unchanged.dtype == torch.float32
