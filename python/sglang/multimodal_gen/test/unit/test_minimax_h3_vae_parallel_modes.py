# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
    CausalAttention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    RotaryEmbeddingND,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


def test_diffusers_video_vae_weight_names_map_to_native_layout():
    config = MiniMaxH3VideoVAEConfig()
    state, _ = hf_to_custom_state_dict(
        {
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight": torch.ones(1),
            "decoder.mask_token": torch.zeros(1, 1, 1),
            "decoder.proj_in.weight": torch.ones(1),
            "decoder.transformer_blocks.0.attn.to_q.weight": torch.zeros(1, 1),
            "decoder.transformer_blocks.0.attn.to_k.weight": torch.ones(1, 1),
            "decoder.transformer_blocks.0.attn.to_v.weight": torch.full((1, 1), 2),
        },
        get_param_names_mapping(config.arch_config.param_names_mapping),
    )

    assert set(state) == {
        "encoder.down.1.block.0.nin_shortcut.weight",
        "decoder.x_embedder.weight",
        "decoder.transformer_blocks.0.attn.to_qkv.weight",
    }
    torch.testing.assert_close(
        state["decoder.transformer_blocks.0.attn.to_qkv.weight"],
        torch.tensor([[0.0], [1.0], [2.0]]),
    )


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()


def test_vit_attention_uses_local_usp_backend_dispatch():
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(f"{module}.USPAttention", autospec=True) as usp_attention,
    ):
        Attention(heads=2, dim_head=64)

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape


def test_vit_rope_rebuilds_nonpersistent_buffer_after_meta_construction():
    with torch.device("meta"):
        rope = RotaryEmbeddingND(dim=96, rotary_base=100, n_dim=3)

    rope.to("cpu")

    assert not rope.inv_freq.is_meta
    torch.testing.assert_close(
        rope.inv_freq,
        1 / 100 ** torch.arange(0, 1, 1 / 16, dtype=torch.float32),
    )


def test_vit_decoder_rebuilds_inference_mask_buffer_after_meta_construction():
    with torch.device("meta"):
        decoder = ViT3DDecoder(
            in_channels=2,
            out_channels=2,
            num_layers=0,
            heads=1,
            dim_head=6,
            num_register_tokens=0,
        )

    decoder.to("cpu")

    assert not decoder.mask_token.is_meta
    torch.testing.assert_close(decoder.mask_token, torch.zeros(1, 1, 6))


def test_audio_vae_attention_defaults_to_local_sdpa_and_allows_fa():
    class RecordingFA(nn.Module):
        backend = AttentionBackendEnum.FA
        dtype = torch.bfloat16

        def forward(self, query, key, value):
            self.input_dtype = query.dtype
            return query

    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae"
    recording_fa = RecordingFA()
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(
            f"{module}.USPAttention", autospec=True, return_value=recording_fa
        ) as usp_attention,
    ):
        attention = CausalAttention(in_dim=64, out_dim=32, num_heads=2)
        output = attention(torch.randn(1, 4, 64))

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["causal"] is True
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }
    assert recording_fa.input_dtype == torch.bfloat16
    assert output.dtype == torch.float32
