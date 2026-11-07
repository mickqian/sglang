# SPDX-License-Identifier: Apache-2.0
"""Image VAE tests.

Tiled decode equals untiled decode, and posterior sampling without a generator
draws from the global RNG.
"""

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3ImageVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKLLegacy,
    DiagonalGaussianDistribution,
)

TINY_VIT = {
    "dim_head": 16,
    "ffn_activation_fn": "silu",
    "ffn_use_gated": True,
    "heads": 2,
    "norm_affine": True,
    "norm_type": "rms_norm",
    "num_layers": 2,
    "qk_norm_affine": False,
    "qk_norm_type": "rms_norm",
    "rope_dim_ratio": 0.5,
    "rope_theta": 100.0,
}


def tiny_legacy(**overrides):
    kwargs = dict(
        ch=32,
        ch_mult=[1] * 6,
        embed_dim=24,
        z_channels=24,
        use_3d_conv=True,
        space_down=[2, 2, 2, 2, 1, 1],
        time_down=[1, 2, 2, 1, 1, 1],
        padding_mode="reflect",
        use_t_isolated_gn=True,
        use_vit_decoder=True,
        vit_decoder_2d=True,
        vit_decoder_kwargs=TINY_VIT,
        encoder_tiling=False,
        decoder_tiling=False,
        token_drop=0,
    )
    kwargs.update(overrides)
    return AutoencoderKLLegacy(**kwargs).eval()


@pytest.fixture
def tiny_image(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_USE_FUSED_NORM", "false")
    config = MiniMaxH3VideoVAEConfig(use_tiling=False)
    config.update_model_arch(
        {
            "vae_token_drop": 0,
            "vae_encoder_tiling": 0,
            "vit_decoder_2d": True,
            "vit_decoder_kwargs": TINY_VIT,
        }
    )
    with torch.device("meta"):
        model = MiniMaxH3ImageVAE(config)
    tiny = tiny_legacy()
    for name in ("encoder", "decoder", "quant_conv", "post_quant_conv"):
        setattr(model, name, getattr(tiny, name))
    return model.eval()


def test_local_tiling_decode_matches_untiled(tiny_image):
    tiny_image.setup_forward(
        encoder_tiling=True,
        decoder_tiling=True,
        parallel_tiling=False,
        tile_size=32,
        tile_overlap_min=16,
        token_drop=0,
    )
    pixels = torch.rand(1, 3, 48, 48)
    moments = tiny_image.tiled_encode(tiny_image.transform(pixels).unsqueeze(2))
    latents = DiagonalGaussianDistribution(moments).mean
    assert latents.shape == (1, 24, 1, 3, 3)
    torch.testing.assert_close(
        tiny_image.decode_base(latents, frame_num=1, process_image=True),
        tiny_image.tiled_decode(latents)[:, :, 0],
        rtol=0,
        atol=0,
    )
    assert tiny_image.decode_image_raw(latents).shape == (1, 3, 48, 48)


def test_posterior_sample_without_generator_uses_global_cpu_rng():
    posterior = DiagonalGaussianDistribution(torch.randn(1, 48, 1, 2, 2))
    with torch.random.fork_rng():
        torch.manual_seed(812)
        noise = torch.randn(posterior.mean.shape)
        expected = noise.mul_(posterior.std).add_(posterior.mean)
        torch.manual_seed(812)
        actual = posterior.sample()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
