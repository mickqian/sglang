# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import (
    DacAudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)


class MiniMaxH3VideoVAE(AutoencoderKLLegacy, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # EncoderFCN3D indexes its down containers instead of calling them, so they
    # cannot host layerwise hooks. Keep the small encoder resident.
    layer_names = ["decoder.transformer_blocks"]

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        arch = config.arch_config
        parallel_decode_mode = config.resolved_parallel_decode_mode()
        use_tiled_decode = config.use_tiling and parallel_decode_mode == "tiled"
        super().__init__(
            in_channels=3,
            out_ch=3,
            ch=128,
            embed_dim=24,
            z_channels=24,
            use_3d_conv=True,
            zq_ch_encoder=None,
            zq_ch_decoder=None,
            num_res_blocks=2,
            num_res_blocks_decoder=None,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            space_up=[1, 2, 2, 2, 2, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            time_up=None,
            padding_mode="reflect",
            padding_mode_t=None,
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_2d=arch.vit_decoder_2d,
            vit_decoder_kwargs=arch.vit_decoder_kwargs,
            shift_factor=0.0,
            scaling_factor=1.0,
            pixel_norm_type="imagenet",
            clip_length=arch.vae_clip_length,
            token_drop=arch.vae_token_drop,
            encoder_tiling=bool(arch.vae_encoder_tiling),
            decoder_tiling=use_tiled_decode,
            parallel_tiling=use_tiled_decode
            and config.use_parallel_decode
            and config.use_parallel_tiling
            and bool(arch.vae_parallel_tiling),
            stack_tiling=config.stack_tiling,
            tile_size=int(arch.vae_tile_size),
            tile_overlap_min=int(arch.vae_tile_overlap_min),
            encoder_parallel=False,
            decoder_parallel=False,
            chunk_dim=int(arch.vae_chunk_dim),
        )
        self.sglang_config = config
        self.use_parallel_decode = config.use_parallel_decode
        self.parallel_decode_mode = parallel_decode_mode

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        return self.decoder.prepare_autocast_linear_weights(dtype)


class MiniMaxH3ImageVAE(MiniMaxH3VideoVAE):
    """H3 CNN encoder with the 2D ViT image decoder (``vit_decoder_2d``)."""

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__(config)
        self.decode_fast_path_gate = VaeFastPathGate()
        register_vae_fast_path_gate(self, self.decode_fast_path_gate)

    def _run_tile_tasks(
        self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None
    ):
        # Reuse native overlapping-tile assembly and batched ViT attention.
        # Bound activation memory as the image canvas grows. Encoding and exact
        # requests retain their original scheduling and numerical path.
        if (
            self.decode_fast_path_gate.enabled
            and not self.training
            and tile_indices
            and tiles[0].is_cuda
            and forward_fn == self.decode
        ):
            chunk_size = max(1, 128 // tiles[0].shape[0])
            outputs = []
            for start in range(0, len(tile_indices), chunk_size):
                outputs.extend(
                    super()._run_tile_tasks(
                        tiles,
                        tile_indices[start : start + chunk_size],
                        forward_fn,
                        stack_tiling=True,
                        cls_agg=cls_agg,
                    )
                )
            return outputs
        return super()._run_tile_tasks(
            tiles, tile_indices, forward_fn, stack_tiling, cls_agg
        )

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        # Cast every decoder-side parameter and floating buffer to ``dtype``,
        # as diffusers from_pretrained(torch_dtype=...) does.
        stale = sum(
            tensor.is_floating_point() and tensor.dtype != dtype
            for module in (self.post_quant_conv, self.decoder)
            for tensor in (*module.parameters(), *module.buffers())
        )
        if stale:
            self.post_quant_conv.to(dtype)
            self.decoder.to(dtype)
        return stale

    @torch.no_grad()
    def decode_image_raw(self, raw_latents: torch.Tensor) -> torch.Tensor:
        """Raw BC1HW latents to BCHW pixels: BF16 decoder under BF16 autocast."""
        self.prepare_decoder_autocast_weights(torch.bfloat16)
        with torch.autocast(device_type=raw_latents.device.type, dtype=torch.bfloat16):
            decoded = self.decode_base(raw_latents, process_image=True, frame_num=1)
        return self.processor.revert_tensor(decoded).squeeze(2)


class MiniMaxH3AudioVAE(DacAudioVAE, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # BigVGAN stores each executable upsampler inside a one-element ModuleList.
    # The outer ``decoder.ups`` containers are indexed but never called, so hooks
    # must target the inner lists whose ConvTranspose1d modules run forward.
    layer_names = [
        "encoder.block",
        *(f"decoder.ups.{index}" for index in range(7)),
        "decoder.resblocks",
    ]

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__(
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 5, 5],
            latent_dim=2048,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            sample_rate=32000,
            vae_latent_channels=32,
            attn_proj=True,
            decoder_type="bigvgan",
        )
        self.config = config


EntryClass = [MiniMaxH3VideoVAE, MiniMaxH3ImageVAE, MiniMaxH3AudioVAE]

__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3ImageVAE", "MiniMaxH3VideoVAE"]
