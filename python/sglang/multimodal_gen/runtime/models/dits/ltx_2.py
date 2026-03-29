# Copied and adapted from LTX-2 and WanVideo implementations.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig, LTX2Config
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_rank,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import Timesteps
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_LTX2_DEBUG_FORWARD_COUNT = 0
_LTX2_DEBUG_STATE: dict[str, Any] = {
    "enabled": False,
    "dir": None,
    "attn_name": None,
    "ff_name": None,
}


def _ltx2_should_force_cudnn_sdpa(q: torch.Tensor, sdpa_mask: torch.Tensor | None) -> bool:
    disable_env = os.environ.get("SGLANG_DISABLE_CUDNN_SDPA", "").strip().lower()
    if disable_env in ("1", "true", "yes", "on"):
        return False

    version = torch.__version__.split("+", 1)[0]
    parts = version.split(".")
    try:
        torch_major_minor = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False

    return (
        torch_major_minor >= (2, 9)
        and q.is_cuda
        and sdpa_mask is None
        and q.dtype in (torch.float16, torch.bfloat16)
    )


def _ltx2_debug_start_forward() -> bool:
    global _LTX2_DEBUG_FORWARD_COUNT
    debug_dir = os.environ.get("LTX2_DEBUG_DIR")
    target_forward = int(os.environ.get("LTX2_DEBUG_FORWARD_INDEX", "0"))
    enabled = bool(debug_dir) and _LTX2_DEBUG_FORWARD_COUNT == target_forward
    _LTX2_DEBUG_STATE["enabled"] = enabled
    _LTX2_DEBUG_STATE["dir"] = debug_dir
    _LTX2_DEBUG_FORWARD_COUNT += 1
    return enabled


def _ltx2_debug_end_forward() -> None:
    _LTX2_DEBUG_STATE["enabled"] = False
    _LTX2_DEBUG_STATE["dir"] = None
    _LTX2_DEBUG_STATE["attn_name"] = None
    _LTX2_DEBUG_STATE["ff_name"] = None
    _LTX2_DEBUG_STATE["time_embed_name"] = None


def _ltx2_debug_save(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None or not _LTX2_DEBUG_STATE["enabled"]:
        return
    debug_dir = _LTX2_DEBUG_STATE["dir"]
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    torch.save(
        tensor.detach().to(device="cpu"),
        os.path.join(debug_dir, f"{name}.pt"),
    )


def apply_interleaved_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def apply_split_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


# ==============================================================================
# Layers and Embeddings
# ==============================================================================


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )
        self.rope_type = rope_type

        self.base_num_frames = int(base_num_frames)
        self.num_attention_heads = int(num_attention_heads)

        self.base_height = int(base_height)
        self.base_width = int(base_width)

        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.audio_latents_per_second = (
            float(self.sampling_rate) / float(self.hop_length) / float(scale_factors[0])
        )

        self.scale_factors = tuple(int(x) for x in scale_factors)
        self.theta = float(theta)
        self.causal_offset = int(causal_offset)

        self.modality = modality
        self.coords_dtype = torch.bfloat16 if modality == "video" else torch.float32
        if self.modality not in ["video", "audio"]:
            raise ValueError(
                f"Modality {modality} is not supported. Supported modalities are `video` and `audio`."
            )
        self.double_precision = bool(double_precision)

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )
        grid_h = torch.arange(
            start=0,
            end=height,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid_w = torch.arange(
            start=0,
            end=width,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )

        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = grid_f * audio_scale_factor
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)
        audio_coords = audio_coords.unsqueeze(1)
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self, coords: torch.Tensor, device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device
        num_pos_dims = coords.shape[1]
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)

        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        else:
            max_positions = (self.base_num_frames,)

        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)

        num_rope_elems = num_pos_dims * 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = torch.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        else:
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
                sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

            b = cos_freq.shape[0]
            t = cos_freq.shape[1]
            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
            cos_freqs = torch.swapaxes(cos_freq, 1, 2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)

        return cos_freqs, sin_freqs


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return hidden_states.to(dtype=input_dtype)


def _ltx2_forward_linear_exact(
    layer: nn.Module,
    x: torch.Tensor,
    force_compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    base_layer = getattr(layer, "base_layer", layer)
    weight = base_layer.weight
    bias = getattr(base_layer, "bias", None)
    compute_dtype = force_compute_dtype or weight.dtype
    if x.dtype != compute_dtype:
        x = x.to(dtype=compute_dtype)
    if weight.dtype != compute_dtype:
        weight = weight.to(dtype=compute_dtype)
    if bias is not None and bias.dtype != compute_dtype:
        bias = bias.to(dtype=compute_dtype)
    out = F.linear(x, weight, bias)

    if (
        hasattr(layer, "compute_lora_delta")
        and not getattr(layer, "merged", True)
        and not getattr(layer, "disable_lora", True)
    ):
        lora_a = layer.lora_A
        lora_b = layer.lora_B
        if hasattr(lora_a, "to_local"):
            lora_a = lora_a.to_local()
        if hasattr(lora_b, "to_local"):
            lora_b = lora_b.to_local()
        out = out + layer.compute_lora_delta(x, lora_a, lora_b).to(dtype=out.dtype)

    return out


class LTX2TextProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = ColumnParallelLinear(
            in_features, hidden_size, bias=True, gather_output=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = ColumnParallelLinear(
            hidden_size, out_features, bias=True, gather_output=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        if get_tp_world_size() == 1:
            hidden_states = _ltx2_forward_linear_exact(self.linear_1, caption)
        else:
            hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        if get_tp_world_size() == 1:
            hidden_states = _ltx2_forward_linear_exact(self.linear_2, hidden_states)
        else:
            hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LTX2TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, in_channels: int = 256) -> None:
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            in_channels, embedding_dim, bias=True, gather_output=True
        )
        self.linear_2 = ColumnParallelLinear(
            embedding_dim, embedding_dim, bias=True, gather_output=True
        )

    def forward(
        self, t_emb: torch.Tensor, use_exact_linear: bool = False
    ) -> torch.Tensor:
        debug_time_embed_name = _LTX2_DEBUG_STATE.get("time_embed_name")
        if _LTX2_DEBUG_STATE["enabled"] and debug_time_embed_name:
            for idx, layer in [("1", self.linear_1), ("2", self.linear_2)]:
                layer_base = getattr(layer, "base_layer", layer)
                _ltx2_debug_save(
                    f"sglang_timestep_linear{idx}_weight_{debug_time_embed_name}",
                    layer_base.weight,
                )
                _ltx2_debug_save(
                    f"sglang_timestep_linear{idx}_bias_{debug_time_embed_name}",
                    getattr(layer_base, "bias", None),
                )
                if getattr(layer, "lora_A", None) is not None:
                    layer_lora_a = layer.lora_A
                    layer_lora_b = layer.lora_B
                    if hasattr(layer_lora_a, "to_local"):
                        layer_lora_a = layer_lora_a.to_local()
                    if hasattr(layer_lora_b, "to_local"):
                        layer_lora_b = layer_lora_b.to_local()
                    _ltx2_debug_save(
                        f"sglang_timestep_linear{idx}_lora_A_{debug_time_embed_name}",
                        layer_lora_a,
                    )
                    _ltx2_debug_save(
                        f"sglang_timestep_linear{idx}_lora_B_{debug_time_embed_name}",
                        layer_lora_b,
                    )
        if use_exact_linear and get_tp_world_size() == 1:
            x = _ltx2_forward_linear_exact(
                self.linear_1,
                t_emb,
                force_compute_dtype=torch.float32,
            )
        else:
            x, _ = self.linear_1(t_emb)
        if _LTX2_DEBUG_STATE["enabled"] and debug_time_embed_name:
            _ltx2_debug_save(f"sglang_timestep_linear1_{debug_time_embed_name}", x)
        x = F.silu(x)
        if _LTX2_DEBUG_STATE["enabled"] and debug_time_embed_name:
            _ltx2_debug_save(f"sglang_timestep_silu_{debug_time_embed_name}", x)
        if use_exact_linear and get_tp_world_size() == 1:
            x = _ltx2_forward_linear_exact(
                self.linear_2,
                x,
                force_compute_dtype=torch.float32,
            )
        else:
            x, _ = self.linear_2(x)
        if _LTX2_DEBUG_STATE["enabled"] and debug_time_embed_name:
            _ltx2_debug_save(f"sglang_timestep_linear2_{debug_time_embed_name}", x)
        return x


class LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = LTX2TimestepEmbedder(embedding_dim, in_channels=256)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype | None = None,
        use_exact_linear: bool = False,
    ) -> torch.Tensor:
        t = timestep.reshape(-1).to(dtype=torch.float32)
        t_emb = self.time_proj(t).to(dtype=torch.float32)
        debug_time_embed_name = _LTX2_DEBUG_STATE.get("time_embed_name")
        if _LTX2_DEBUG_STATE["enabled"] and debug_time_embed_name:
            _ltx2_debug_save(f"sglang_timestep_proj_{debug_time_embed_name}", t_emb)
        if hidden_dtype is not None:
            t_emb = t_emb.to(dtype=hidden_dtype)
        return self.timestep_embedder(t_emb, use_exact_linear=use_exact_linear)


class LTX2AdaLayerNormSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        embedding_coefficient: int = 6,
        force_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.force_fp32 = force_fp32
        self.emb = LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
            gather_output=True,
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = torch.float32 if self.force_fp32 else hidden_dtype
        embedded_timestep = self.emb(
            timestep,
            hidden_dtype=compute_dtype,
            use_exact_linear=self.force_fp32,
        ).to(dtype=torch.float32 if self.force_fp32 else self.linear.weight.dtype)
        silu_embedded = self.silu(embedded_timestep)
        if self.force_fp32 and get_tp_world_size() == 1:
            out = _ltx2_forward_linear_exact(
                self.linear,
                silu_embedded,
                force_compute_dtype=torch.float32,
            )
        else:
            out, _ = self.linear(silu_embedded)
        if self.force_fp32 and hidden_dtype is not None:
            out = out.to(dtype=hidden_dtype)
            embedded_timestep = embedded_timestep.to(dtype=hidden_dtype)
        return out, embedded_timestep


class LTX2TPRMSNormAcrossHeads(nn.Module):
    def __init__(
        self, full_hidden_size: int, local_hidden_size: int, eps: float
    ) -> None:
        super().__init__()
        self.full_hidden_size = full_hidden_size
        self.local_hidden_size = local_hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_hidden_size))

        tp_rank = get_tp_rank()

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            shard = loaded_weight.narrow(
                0, tp_rank * local_hidden_size, local_hidden_size
            )
            param.data.copy_(shard.to(dtype=param.dtype, device=param.device))

        setattr(self.weight, "weight_loader", _weight_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep track of the original dtype. We do the statistics in fp32 for
        # numerical stability, but cast the output back to the input dtype to
        orig_dtype = x.dtype
        if get_tp_world_size() == 1:
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
        else:
            local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
            global_sumsq = tensor_model_parallel_all_reduce(local_sumsq)
            var = global_sumsq / float(self.full_hidden_size)

        inv_rms_fp32 = torch.rsqrt(var + self.eps)
        y = (x.float() * inv_rms_fp32).to(dtype=orig_dtype)
        return y * self.weight.to(dtype=orig_dtype)


class LTX2Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        qk_norm: bool = True,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.query_dim = int(query_dim)
        self.context_dim = int(query_dim if context_dim is None else context_dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.norm_eps = float(norm_eps)
        self.qk_norm = bool(qk_norm)

        tp_size = get_tp_world_size()
        if tp_size <= 0:
            raise ValueError(f"Invalid {tp_size=}. Expected tp_size >= 1.")
        if self.heads % tp_size != 0:
            raise ValueError(
                f"LTX2Attention requires heads divisible by tp_size, got "
                f"{self.heads=} {tp_size=}."
            )
        if self.inner_dim % tp_size != 0:
            # This should follow from heads % tp_size, but keep explicit for clarity.
            raise ValueError(
                f"LTX2Attention requires inner_dim divisible by tp_size, got "
                f"{self.inner_dim=} {tp_size=}."
            )
        self.local_heads = self.heads // tp_size

        self.to_q = ColumnParallelLinear(
            self.query_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_k = ColumnParallelLinear(
            self.context_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_v = ColumnParallelLinear(
            self.context_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )

        self.q_norm: nn.Module | None = None
        self.k_norm: nn.Module | None = None
        if self.qk_norm:
            if tp_size == 1:
                self.q_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
                self.k_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
            else:
                self.q_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )
                self.k_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )

        self.to_out = nn.Sequential(
            RowParallelLinear(
                self.inner_dim,
                self.query_dim,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
            ),
            nn.Identity(),
        )

        self.attn = USPAttention(
            num_heads=self.local_heads,
            head_size=self.dim_head,
            num_kv_heads=self.local_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        debug_attn_name = _LTX2_DEBUG_STATE.get("attn_name")
        q, _ = self.to_q(x)
        context_ = x if context is None else context
        k, _ = self.to_k(context_)
        v, _ = self.to_v(context_)
        if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
            _ltx2_debug_save(f"sglang_{debug_attn_name}_q_weight", self.to_q.weight)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_k_weight", self.to_k.weight)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_v_weight", self.to_v.weight)
            if self.to_q.bias is not None:
                _ltx2_debug_save(f"sglang_{debug_attn_name}_q_bias", self.to_q.bias)
            if self.to_k.bias is not None:
                _ltx2_debug_save(f"sglang_{debug_attn_name}_k_bias", self.to_k.bias)
            if self.to_v.bias is not None:
                _ltx2_debug_save(f"sglang_{debug_attn_name}_v_bias", self.to_v.bias)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_q_linear", q)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_k_linear", k)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_v_linear", v)

        if self.qk_norm:
            assert self.q_norm is not None and self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)
            if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
                _ltx2_debug_save(f"sglang_{debug_attn_name}_q_norm", q)
                _ltx2_debug_save(f"sglang_{debug_attn_name}_k_norm", k)

        if pe is not None:
            cos, sin = pe
            k_cos, k_sin = pe if k_pe is None else k_pe
            tp_size = get_tp_world_size()
            if tp_size > 1:
                tp_rank = get_tp_rank()
                cos, sin = self._slice_rope_for_tp(
                    cos, sin, tp_rank=tp_rank, tp_size=tp_size
                )
                k_cos, k_sin = self._slice_rope_for_tp(
                    k_cos, k_sin, tp_rank=tp_rank, tp_size=tp_size
                )
            if cos.dim() == 3:
                q = apply_interleaved_rotary_emb(q, (cos, sin))
                k = apply_interleaved_rotary_emb(k, (k_cos, k_sin))
            else:
                q = apply_split_rotary_emb(q, (cos, sin))
                k = apply_split_rotary_emb(k, (k_cos, k_sin))
            if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
                _ltx2_debug_save(f"sglang_{debug_attn_name}_q_rope", q)
                _ltx2_debug_save(f"sglang_{debug_attn_name}_k_rope", k)

        q = q.view(*q.shape[:-1], self.local_heads, self.dim_head)
        k = k.view(*k.shape[:-1], self.local_heads, self.dim_head)
        v = v.view(*v.shape[:-1], self.local_heads, self.dim_head)
        if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
            _ltx2_debug_save(f"sglang_{debug_attn_name}_q_heads", q)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_k_heads", k)
            _ltx2_debug_save(f"sglang_{debug_attn_name}_v_heads", v)

        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        sdpa_mask = None
        if mask is not None:
            if torch.is_floating_point(mask):
                m = mask
                if m.dim() == 2:
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    m = m[:, None, :, :]
                sdpa_mask = m.to(dtype=q_.dtype, device=q_.device)
            else:
                m = mask.to(dtype=q_.dtype, device=q_.device)
                if m.dim() == 2:
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    m = m[:, None, :, :]
                sdpa_mask = (m - 1.0) * torch.finfo(q_.dtype).max

        if sdpa_mask is None and get_sp_world_size() == 1 and _ltx2_should_force_cudnn_sdpa(
            q_, None
        ):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False
                ).transpose(1, 2)
        elif sdpa_mask is None:
            # Keep USPAttention for the no-mask path when sequence parallelism is
            # active or when CUDNN SDPA is not appropriate on this runtime.
            out = self.attn(q, k, v)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            ).transpose(1, 2)
        if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
            _ltx2_debug_save(f"sglang_{debug_attn_name}_sdpa_out", out)

        out = out.flatten(2)
        out, _ = self.to_out[0](out)
        if _LTX2_DEBUG_STATE["enabled"] and debug_attn_name:
            _ltx2_debug_save(f"sglang_{debug_attn_name}_out_proj", out)
        return out

    def _slice_rope_for_tp(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice RoPE tensors to the local TP shard.

        - split-rope: cos/sin are shaped [B, H, T, R] (head-major), slice by heads.
        - interleaved-rope: cos/sin are shaped [B, T, D], where D matches the projected
          feature dimension and is sharded by TP.
        """
        if cos.ndim == 4:
            # [B, H, T, R]
            start = tp_rank * self.local_heads
            end = start + self.local_heads
            return cos[:, start:end, :, :], sin[:, start:end, :, :]
        elif cos.ndim == 3:
            # [B, T, D]
            d = cos.shape[-1]
            if d % tp_size != 0:
                raise ValueError(
                    f"RoPE dim must be divisible by tp_size, got {d=} {tp_size=}."
                )
            local_d = d // tp_size
            start = tp_rank * local_d
            end = start + local_d
            return cos[:, :, start:end], sin[:, :, start:end]
        raise ValueError(f"Unexpected RoPE tensor rank: {cos.ndim}. Expected 3 or 4.")


class LTX2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        if dim_out is None:
            dim_out = dim
        inner_dim = int(dim * mult)

        self.proj_in = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=True, quant_config=quant_config
        )
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = ColumnParallelLinear(
            inner_dim, dim_out, bias=True, gather_output=True, quant_config=quant_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_ff_name = _LTX2_DEBUG_STATE.get("ff_name")
        x, _ = self.proj_in(x)
        if _LTX2_DEBUG_STATE["enabled"] and debug_ff_name:
            _ltx2_debug_save(f"sglang_{debug_ff_name}_proj_in", x)
        x = self.act(x)
        if _LTX2_DEBUG_STATE["enabled"] and debug_ff_name:
            _ltx2_debug_save(f"sglang_{debug_ff_name}_act", x)
        x, _ = self.proj_out(x)
        if _LTX2_DEBUG_STATE["enabled"] and debug_ff_name:
            _ltx2_debug_save(f"sglang_{debug_ff_name}_proj_out", x)
        return x


class LTX2TransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        video_cross_attn_adaln: bool = False,
        audio_cross_attn_adaln: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        self.video_cross_attn_adaln = bool(video_cross_attn_adaln)
        self.audio_cross_attn_adaln = bool(audio_cross_attn_adaln)
        self.cross_attn_adaln = (
            self.video_cross_attn_adaln or self.audio_cross_attn_adaln
        )

        # 1. Self-Attention (video and audio)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
            quant_config=quant_config,
        )
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn1",
            quant_config=quant_config,
        )

        # 2. Prompt Cross-Attention
        self.attn2 = LTX2Attention(
            query_dim=dim,
            context_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn2",
            quant_config=quant_config,
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            context_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn2",
            quant_config=quant_config,
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            context_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_to_video_attn",
            quant_config=quant_config,
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            context_dim=dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.video_to_audio_attn",
            quant_config=quant_config,
        )

        # 4. Feedforward layers
        self.ff = LTX2FeedForward(dim, dim_out=dim, quant_config=quant_config)
        self.audio_ff = LTX2FeedForward(
            audio_dim, dim_out=audio_dim, quant_config=quant_config
        )

        # 5. Modulation Parameters
        video_mod_param_num = 9 if self.video_cross_attn_adaln else 6
        audio_mod_param_num = 9 if self.audio_cross_attn_adaln else 6
        self.scale_shift_table = nn.Parameter(
            (torch.randn(video_mod_param_num, dim) / dim**0.5).to(torch.float32)
        )
        self.audio_scale_shift_table = nn.Parameter(
            (torch.randn(audio_mod_param_num, audio_dim) / audio_dim**0.5).to(
                torch.float32
            )
        )
        if self.cross_attn_adaln:
            self.prompt_scale_shift_table = nn.Parameter(
                torch.randn(2, dim, dtype=torch.float32)
            )
            self.audio_prompt_scale_shift_table = nn.Parameter(
                torch.randn(2, audio_dim, dtype=torch.float32)
            )
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, dim, dtype=torch.float32)
        )
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, audio_dim, dtype=torch.float32)
        )

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = int(scale_shift_table.shape[0])
        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[
                :, :, indices, :
            ]
        ).unbind(dim=2)
        return [t.squeeze(2) for t in ada_values]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        temb_prompt: Optional[torch.Tensor] = None,
        temb_prompt_audio: Optional[torch.Tensor] = None,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        a2v_cross_attention_mask: Optional[torch.Tensor] = None,
        v2a_cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden_states.size(0)
        debug_block0 = bool(_LTX2_DEBUG_STATE["enabled"]) and self.idx == 0

        # 1. Video and Audio Self-Attention
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, batch_size, temb, slice(0, 3)
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_self_attn_video_shift", vshift_msa)
            _ltx2_debug_save("sglang_block0_self_attn_video_scale", vscale_msa)
            _ltx2_debug_save("sglang_block0_self_attn_video_gate", vgate_msa)
        norm_hidden_states = (
            rms_norm(hidden_states, self.norm_eps) * (1 + vscale_msa) + vshift_msa
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_self_attn_video_input", norm_hidden_states)
            _LTX2_DEBUG_STATE["attn_name"] = "block0_self_attn_video"
        attn_hidden_states = self.attn1(norm_hidden_states, pe=video_rotary_emb)
        if debug_block0:
            _LTX2_DEBUG_STATE["attn_name"] = None
            _ltx2_debug_save("sglang_block0_self_attn_video_output", attn_hidden_states)
        hidden_states = hidden_states + attn_hidden_states * vgate_msa
        if debug_block0:
            _ltx2_debug_save("sglang_block0_self_attn_video_residual", hidden_states)

        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(0, 3)
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_self_attn_audio_shift", ashift_msa)
            _ltx2_debug_save("sglang_block0_self_attn_audio_scale", ascale_msa)
            _ltx2_debug_save("sglang_block0_self_attn_audio_gate", agate_msa)
        norm_audio_hidden_states = (
            rms_norm(audio_hidden_states, self.norm_eps) * (1 + ascale_msa) + ashift_msa
        )
        if debug_block0:
            _ltx2_debug_save(
                "sglang_block0_self_attn_audio_input", norm_audio_hidden_states
            )
            _LTX2_DEBUG_STATE["attn_name"] = "block0_self_attn_audio"
        attn_audio_hidden_states = self.audio_attn1(
            norm_audio_hidden_states, pe=audio_rotary_emb
        )
        if debug_block0:
            _LTX2_DEBUG_STATE["attn_name"] = None
            _ltx2_debug_save(
                "sglang_block0_self_attn_audio_output", attn_audio_hidden_states
            )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * agate_msa
        if debug_block0:
            _ltx2_debug_save("sglang_block0_self_attn_audio_residual", audio_hidden_states)

        # 2. Prompt Cross-Attention
        if self.cross_attn_adaln:
            if temb_prompt is None or temb_prompt_audio is None:
                raise ValueError(
                    "Prompt modulation is enabled, but temb_prompt / temb_prompt_audio is missing."
                )
            shift_text_kv, scale_text_kv = self.get_ada_values(
                self.prompt_scale_shift_table, batch_size, temb_prompt, slice(0, 2)
            )
            audio_shift_text_kv, audio_scale_text_kv = self.get_ada_values(
                self.audio_prompt_scale_shift_table,
                batch_size,
                temb_prompt_audio,
                slice(0, 2),
            )
        if self.video_cross_attn_adaln:
            shift_text_q, scale_text_q, gate_text_q = self.get_ada_values(
                self.scale_shift_table, batch_size, temb, slice(6, 9)
            )
        if self.audio_cross_attn_adaln:
            audio_shift_text_q, audio_scale_text_q, audio_gate_text_q = (
                self.get_ada_values(
                    self.audio_scale_shift_table, batch_size, temb_audio, slice(6, 9)
                )
            )

        norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
        if self.video_cross_attn_adaln:
            norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q
        video_encoder_hidden_states = encoder_hidden_states
        if self.cross_attn_adaln:
            video_encoder_hidden_states = video_encoder_hidden_states * (
                1 + scale_text_kv
            ) + shift_text_kv
        if debug_block0:
            _ltx2_debug_save("sglang_block0_cross_attn_video_input", norm_hidden_states)
            _LTX2_DEBUG_STATE["attn_name"] = "block0_cross_attn_video"
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            context=video_encoder_hidden_states,
            mask=encoder_attention_mask,
        )
        if self.video_cross_attn_adaln:
            attn_hidden_states = attn_hidden_states * gate_text_q
        if debug_block0:
            _LTX2_DEBUG_STATE["attn_name"] = None
            _ltx2_debug_save("sglang_block0_cross_attn_video_output", attn_hidden_states)
        hidden_states = hidden_states + attn_hidden_states
        if debug_block0:
            _ltx2_debug_save("sglang_block0_cross_attn_video_residual", hidden_states)

        norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)
        if self.audio_cross_attn_adaln:
            norm_audio_hidden_states = norm_audio_hidden_states * (
                1 + audio_scale_text_q
            ) + audio_shift_text_q
        mod_audio_encoder_hidden_states = audio_encoder_hidden_states
        if self.cross_attn_adaln:
            mod_audio_encoder_hidden_states = mod_audio_encoder_hidden_states * (
                1 + audio_scale_text_kv
            ) + audio_shift_text_kv
        if debug_block0:
            _ltx2_debug_save(
                "sglang_block0_cross_attn_audio_input", norm_audio_hidden_states
            )
            _LTX2_DEBUG_STATE["attn_name"] = "block0_cross_attn_audio"
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            context=mod_audio_encoder_hidden_states,
            mask=audio_encoder_attention_mask,
        )
        if self.audio_cross_attn_adaln:
            attn_audio_hidden_states = attn_audio_hidden_states * audio_gate_text_q
        if debug_block0:
            _LTX2_DEBUG_STATE["attn_name"] = None
            _ltx2_debug_save(
                "sglang_block0_cross_attn_audio_output", attn_audio_hidden_states
            )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states
        if debug_block0:
            _ltx2_debug_save("sglang_block0_cross_attn_audio_residual", audio_hidden_states)

        # 3. Audio-to-Video and Video-to-Audio Cross-Attention
        norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
        norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)

        # Compute combined ada params
        video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[
            :4, :
        ]
        video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]

        video_ca_scale_shift_table = (
            video_per_layer_ca_scale_shift[None, None, :, :].to(
                dtype=temb_ca_scale_shift.dtype, device=temb_ca_scale_shift.device
            )
            + temb_ca_scale_shift.reshape(
                batch_size, temb_ca_scale_shift.shape[1], 4, -1
            )
        ).unbind(dim=2)
        video_ca_gate = (
            video_per_layer_ca_gate[None, None, :, :].to(
                dtype=temb_ca_gate.dtype, device=temb_ca_gate.device
            )
            + temb_ca_gate.reshape(batch_size, temb_ca_gate.shape[1], 1, -1)
        ).unbind(dim=2)

        (
            video_a2v_ca_scale,
            video_a2v_ca_shift,
            video_v2a_ca_scale,
            video_v2a_ca_shift,
        ) = [t.squeeze(2) for t in video_ca_scale_shift_table]
        a2v_gate = video_ca_gate[0].squeeze(2)
        if debug_block0:
            _ltx2_debug_save("sglang_block0_a2v_video_scale", video_a2v_ca_scale)
            _ltx2_debug_save("sglang_block0_a2v_video_shift", video_a2v_ca_shift)
            _ltx2_debug_save("sglang_block0_v2a_video_scale", video_v2a_ca_scale)
            _ltx2_debug_save("sglang_block0_v2a_video_shift", video_v2a_ca_shift)
            _ltx2_debug_save("sglang_block0_a2v_gate", a2v_gate)

        audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[
            :4, :
        ]
        audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]

        audio_ca_scale_shift_table = (
            audio_per_layer_ca_scale_shift[None, None, :, :].to(
                dtype=temb_ca_audio_scale_shift.dtype,
                device=temb_ca_audio_scale_shift.device,
            )
            + temb_ca_audio_scale_shift.reshape(
                batch_size, temb_ca_audio_scale_shift.shape[1], 4, -1
            )
        ).unbind(dim=2)
        audio_ca_gate = (
            audio_per_layer_ca_gate[None, None, :, :].to(
                dtype=temb_ca_audio_gate.dtype, device=temb_ca_audio_gate.device
            )
            + temb_ca_audio_gate.reshape(batch_size, temb_ca_audio_gate.shape[1], 1, -1)
        ).unbind(dim=2)

        (
            audio_a2v_ca_scale,
            audio_a2v_ca_shift,
            audio_v2a_ca_scale,
            audio_v2a_ca_shift,
        ) = [t.squeeze(2) for t in audio_ca_scale_shift_table]
        v2a_gate = audio_ca_gate[0].squeeze(2)
        if debug_block0:
            _ltx2_debug_save("sglang_block0_a2v_audio_scale", audio_a2v_ca_scale)
            _ltx2_debug_save("sglang_block0_a2v_audio_shift", audio_a2v_ca_shift)
            _ltx2_debug_save("sglang_block0_v2a_audio_scale", audio_v2a_ca_scale)
            _ltx2_debug_save("sglang_block0_v2a_audio_shift", audio_v2a_ca_shift)
            _ltx2_debug_save("sglang_block0_v2a_gate", v2a_gate)

        # A2V
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_a2v_ca_scale) + video_a2v_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_a2v_ca_scale) + audio_a2v_ca_shift
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_a2v_video_input", mod_norm_hidden_states)
            _ltx2_debug_save("sglang_block0_a2v_audio_input", mod_norm_audio_hidden_states)
            _LTX2_DEBUG_STATE["attn_name"] = "block0_a2v"

        a2v_attn_hidden_states = self.audio_to_video_attn(
            mod_norm_hidden_states,
            context=mod_norm_audio_hidden_states,
            pe=ca_video_rotary_emb,
            k_pe=ca_audio_rotary_emb,
            mask=a2v_cross_attention_mask,
        )
        if debug_block0:
            _LTX2_DEBUG_STATE["attn_name"] = None
            _ltx2_debug_save("sglang_block0_a2v_output", a2v_attn_hidden_states)
        hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states
        if debug_block0:
            _ltx2_debug_save("sglang_block0_a2v_video_residual", hidden_states)

        # V2A
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_v2a_ca_scale) + video_v2a_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_v2a_ca_scale) + audio_v2a_ca_shift
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_v2a_video_input", mod_norm_hidden_states)
            _ltx2_debug_save("sglang_block0_v2a_audio_input", mod_norm_audio_hidden_states)

        v2a_attn_hidden_states = self.video_to_audio_attn(
            mod_norm_audio_hidden_states,
            context=mod_norm_hidden_states,
            pe=ca_audio_rotary_emb,
            k_pe=ca_video_rotary_emb,
            mask=v2a_cross_attention_mask,
        ) if not debug_block0 else self._debug_v2a_attn(
            mod_norm_audio_hidden_states=mod_norm_audio_hidden_states,
            mod_norm_hidden_states=mod_norm_hidden_states,
            ca_audio_rotary_emb=ca_audio_rotary_emb,
            ca_video_rotary_emb=ca_video_rotary_emb,
            v2a_cross_attention_mask=v2a_cross_attention_mask,
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_v2a_output", v2a_attn_hidden_states)
        audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states
        if debug_block0:
            _ltx2_debug_save("sglang_block0_v2a_audio_residual", audio_hidden_states)

        # 4. Feedforward
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, batch_size, temb, slice(3, 6)
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_ff_video_shift", vshift_mlp)
            _ltx2_debug_save("sglang_block0_ff_video_scale", vscale_mlp)
            _ltx2_debug_save("sglang_block0_ff_video_gate", vgate_mlp)
            _ltx2_debug_save("sglang_block0_ff_video_prenorm_hidden", hidden_states)
        norm_hidden_states = (
            rms_norm(hidden_states, self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_ff_video_input", norm_hidden_states)
            _LTX2_DEBUG_STATE["ff_name"] = "block0_ff_video"
        ff_output = self.ff(norm_hidden_states)
        if debug_block0:
            _LTX2_DEBUG_STATE["ff_name"] = None
            _ltx2_debug_save("sglang_block0_ff_video_output", ff_output)
        hidden_states = hidden_states + ff_output * vgate_mlp

        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(3, 6)
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_ff_audio_shift", ashift_mlp)
            _ltx2_debug_save("sglang_block0_ff_audio_scale", ascale_mlp)
            _ltx2_debug_save("sglang_block0_ff_audio_gate", agate_mlp)
            _ltx2_debug_save("sglang_block0_ff_audio_prenorm_hidden", audio_hidden_states)
        norm_audio_hidden_states = (
            rms_norm(audio_hidden_states, self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
        )
        if debug_block0:
            _ltx2_debug_save("sglang_block0_ff_audio_input", norm_audio_hidden_states)
            _LTX2_DEBUG_STATE["ff_name"] = "block0_ff_audio"
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        if debug_block0:
            _LTX2_DEBUG_STATE["ff_name"] = None
            _ltx2_debug_save("sglang_block0_ff_audio_output", audio_ff_output)
        audio_hidden_states = audio_hidden_states + audio_ff_output * agate_mlp

        if debug_block0:
            _ltx2_debug_save("sglang_block0_video_output", hidden_states)
            _ltx2_debug_save("sglang_block0_audio_output", audio_hidden_states)

        return hidden_states, audio_hidden_states

    def _debug_v2a_attn(
        self,
        *,
        mod_norm_audio_hidden_states: torch.Tensor,
        mod_norm_hidden_states: torch.Tensor,
        ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        v2a_cross_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _LTX2_DEBUG_STATE["attn_name"] = "block0_v2a"
        try:
            return self.video_to_audio_attn(
                mod_norm_audio_hidden_states,
                context=mod_norm_hidden_states,
                pe=ca_audio_rotary_emb,
                k_pe=ca_video_rotary_emb,
                mask=v2a_cross_attention_mask,
            )
        finally:
            _LTX2_DEBUG_STATE["attn_name"] = None


class LTX2VideoTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = LTX2ArchConfig()._fsdp_shard_conditions
    _compile_conditions = LTX2ArchConfig()._compile_conditions
    _supported_attention_backends = LTX2ArchConfig()._supported_attention_backends
    param_names_mapping = LTX2ArchConfig().param_names_mapping
    reverse_param_names_mapping = LTX2ArchConfig().reverse_param_names_mapping
    lora_param_names_mapping = LTX2ArchConfig().lora_param_names_mapping

    def _validate_tp_config(self, *, arch: LTX2ArchConfig, tp_size: int) -> None:
        """Validate TP-related dimension constraints (fail-fast)."""
        if tp_size < 1:
            raise ValueError(f"Invalid tp_size={tp_size}. Expected tp_size >= 1.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "video hidden_size must be divisible by num_attention_heads, got "
                f"{self.hidden_size=} {self.num_attention_heads=}."
            )
        if self.audio_hidden_size % self.audio_num_attention_heads != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by audio_num_attention_heads, got "
                f"{self.audio_hidden_size=} {self.audio_num_attention_heads=}."
            )

        if tp_size == 1:
            return

        if self.num_attention_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by tp_size, got "
                f"{self.num_attention_heads=} {tp_size=}."
            )
        if self.audio_num_attention_heads % tp_size != 0:
            raise ValueError(
                "audio_num_attention_heads must be divisible by tp_size, got "
                f"{self.audio_num_attention_heads=} {tp_size=}."
            )
        if self.hidden_size % tp_size != 0:
            raise ValueError(
                "hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.hidden_size=} {tp_size=}."
            )
        if self.audio_hidden_size % tp_size != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.audio_hidden_size=} {tp_size=}."
            )
        if int(arch.out_channels) % tp_size != 0:
            raise ValueError(
                "out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.out_channels=} {tp_size=}."
            )
        if int(arch.audio_out_channels) % tp_size != 0:
            raise ValueError(
                "audio_out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.audio_out_channels=} {tp_size=}."
            )

    def __init__(
        self,
        config: LTX2Config,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.audio_hidden_size = arch.audio_hidden_size
        self.audio_num_attention_heads = arch.audio_num_attention_heads
        self.norm_eps = arch.norm_eps
        self.cross_attn_mod = bool(
            hf_config.get("cross_attn_mod", arch.cross_attn_mod)
        )
        self.audio_cross_attn_mod = bool(
            hf_config.get("audio_cross_attn_mod", arch.audio_cross_attn_mod)
        )
        self.prompt_modulation = self.cross_attn_mod or self.audio_cross_attn_mod

        tp_size = get_tp_world_size()
        self._validate_tp_config(arch=arch, tp_size=tp_size)

        # 1. Patchification input projections
        # Matches LTX2Config().param_names_mapping
        self.patchify_proj = ColumnParallelLinear(
            arch.in_channels,
            self.hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )
        self.audio_patchify_proj = ColumnParallelLinear(
            arch.audio_in_channels,
            self.audio_hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        # 2. Prompt embeddings
        self.caption_projection = LTX2TextProjection(
            in_features=arch.caption_channels, hidden_size=self.hidden_size
        )
        self.audio_caption_projection = LTX2TextProjection(
            in_features=arch.caption_channels, hidden_size=self.audio_hidden_size
        )

        # 3. Timestep Modulation Params and Embedding
        self.adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size,
            embedding_coefficient=9 if self.cross_attn_mod else 6,
        )
        self.audio_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size,
            embedding_coefficient=9 if self.audio_cross_attn_mod else 6,
        )

        # Global Cross Attention Modulation Parameters
        self.av_ca_video_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=4, force_fp32=True
        )
        self.av_ca_a2v_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=1, force_fp32=True
        )
        self.av_ca_audio_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=4, force_fp32=True
        )
        self.av_ca_v2a_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=1, force_fp32=True
        )
        if self.prompt_modulation:
            self.prompt_adaln_single = LTX2AdaLayerNormSingle(
                self.hidden_size, embedding_coefficient=2
            )
            self.audio_prompt_adaln_single = LTX2AdaLayerNormSingle(
                self.audio_hidden_size, embedding_coefficient=2
            )

        # Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.hidden_size) / self.hidden_size**0.5
        )
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(2, self.audio_hidden_size) / self.audio_hidden_size**0.5
        )

        hf_patch_size = int(hf_config.get("patch_size", 1))
        hf_patch_size_t = int(hf_config.get("patch_size_t", 1))
        self.patch_size = (hf_patch_size_t, hf_patch_size, hf_patch_size)

        hf_audio_patch_size = int(hf_config.get("audio_patch_size", 1))
        hf_audio_patch_size_t = int(hf_config.get("audio_patch_size_t", 1))

        rope_type = (
            arch.rope_type.value
            if hasattr(arch.rope_type, "value")
            else str(arch.rope_type)
        )
        rope_double_precision = bool(
            hf_config.get("rope_double_precision", arch.double_precision_rope)
        )
        causal_offset = int(hf_config.get("causal_offset", 1))

        pos_embed_max_pos = int(arch.positional_embedding_max_pos[0])
        base_height = int(arch.positional_embedding_max_pos[1])
        base_width = int(arch.positional_embedding_max_pos[2])

        audio_pos_embed_max_pos = int(arch.audio_positional_embedding_max_pos[0])

        self.video_scale_factors = (8, 32, 32)
        self.audio_scale_factors = (4,)

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.hidden_size,
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=self.video_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.audio_hidden_size,
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            scale_factors=self.audio_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        self.cross_pe_max_pos = cross_attn_pos_embed_max_pos

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock(
                    idx=idx,
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.hidden_size // self.num_attention_heads,
                    cross_attention_dim=arch.cross_attention_dim,
                    audio_dim=self.audio_hidden_size,
                    audio_num_attention_heads=self.audio_num_attention_heads,
                    audio_attention_head_dim=self.audio_hidden_size
                    // self.audio_num_attention_heads,
                    audio_cross_attention_dim=arch.audio_cross_attention_dim,
                    norm_eps=self.norm_eps,
                    video_cross_attn_adaln=self.cross_attn_mod,
                    audio_cross_attn_adaln=self.audio_cross_attn_mod,
                    qk_norm=True,  # Always True in LTX2
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=config.prefix,
                    quant_config=quant_config,
                )
                for idx in range(arch.num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.proj_out = ColumnParallelLinear(
            self.hidden_size,
            arch.out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        self.audio_norm_out = nn.LayerNorm(
            self.audio_hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.audio_proj_out = ColumnParallelLinear(
            self.audio_hidden_size,
            arch.audio_out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        self.out_channels_raw = arch.out_channels // (
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        )
        self.audio_out_channels = arch.audio_out_channels
        self.timestep_scale_multiplier = arch.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = arch.av_ca_timestep_scale_multiplier

        self.layer_names = ["transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        debug_forward = _ltx2_debug_start_forward()
        batch_size = hidden_states.size(0)
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        try:
            if (
                encoder_attention_mask is not None
                and encoder_attention_mask.ndim == 2
            ):
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            if (
                audio_encoder_attention_mask is not None
                and audio_encoder_attention_mask.ndim == 2
            ):
                audio_encoder_attention_mask = (
                    1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
                ) * -10000.0
                audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "num_frames/height/width must be provided for RoPE coordinate generation."
                )
            if audio_num_frames is None:
                raise ValueError(
                    "audio_num_frames must be provided for RoPE coordinate generation."
                )

            if video_coords is None:
                # Wan-style SP-RoPE: when SP is enabled, each rank runs on its local
                # time shard but RoPE positions must be offset to global time.
                #
                # We assume equal time sharding across SP ranks.
                if model_parallel_is_initialized():
                    sp_world_size = get_sp_world_size()
                    sp_rank = get_sp_parallel_rank()
                else:
                    sp_world_size = 1
                    sp_rank = 0

                video_shift = int(sp_rank) * int(num_frames) if sp_world_size > 1 else 0
                video_coords = self.rope.prepare_video_coords(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    device=hidden_states.device,
                    fps=fps,
                    start_frame=video_shift,
                )
            if audio_coords is None:
                audio_coords = self.audio_rope.prepare_audio_coords(
                    batch_size=batch_size,
                    num_frames=audio_num_frames,
                    device=audio_hidden_states.device,
                )

            video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
            audio_rotary_emb = self.audio_rope(
                audio_coords, device=audio_hidden_states.device
            )
            ca_video_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device=hidden_states.device
            )
            ca_audio_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device=audio_hidden_states.device
            )

            # 2. Patchify input projections
            patchify_input = hidden_states
            audio_patchify_input = audio_hidden_states
            patchify_layer = getattr(self.patchify_proj, "base_layer", self.patchify_proj)
            patchify_bias = (
                patchify_layer.bias
                if not patchify_layer.skip_bias_add
                else None
            )
            with torch.autocast(device_type=patchify_input.device.type, enabled=False):
                patchify_output_parallel = F.linear(
                    patchify_input.float(),
                    patchify_layer.weight.float(),
                    patchify_bias.float() if patchify_bias is not None else None,
                )
            if patchify_layer.gather_output:
                hidden_states = tensor_model_parallel_all_gather(
                    patchify_output_parallel, tp_group=patchify_layer.tp_group
                )
            else:
                hidden_states = patchify_output_parallel
            hidden_states = hidden_states.to(dtype=patchify_input.dtype)
            audio_hidden_states, _ = self.audio_patchify_proj(audio_hidden_states)
            if debug_forward:
                _ltx2_debug_save("sglang_stage1_step0_input_video", patchify_input)
                _ltx2_debug_save("sglang_stage1_step0_input_audio", audio_patchify_input)
                audio_patchify_layer = getattr(
                    self.audio_patchify_proj, "base_layer", self.audio_patchify_proj
                )
                audio_patchify_bias = (
                    audio_patchify_layer.bias
                    if getattr(audio_patchify_layer, "bias", None) is not None
                    else None
                )
                with torch.autocast(
                    device_type=patchify_input.device.type, enabled=False
                ):
                    manual_patchify_video = F.linear(
                        patchify_input.float(),
                        patchify_layer.weight.float(),
                        patchify_bias.float() if patchify_bias is not None else None,
                    )
                manual_patchify_video = manual_patchify_video.to(
                    dtype=patchify_input.dtype
                )
                manual_patchify_audio = F.linear(
                    audio_patchify_input,
                    audio_patchify_layer.weight,
                    audio_patchify_bias,
                )
                _ltx2_debug_save(
                    "sglang_patchify_proj_weight", patchify_layer.weight
                )
                _ltx2_debug_save("sglang_patchify_proj_bias", patchify_layer.bias)
                _ltx2_debug_save(
                    "sglang_audio_patchify_proj_weight", audio_patchify_layer.weight
                )
                _ltx2_debug_save(
                    "sglang_audio_patchify_proj_bias", audio_patchify_layer.bias
                )
                _ltx2_debug_save("sglang_patchify_video", hidden_states)
                _ltx2_debug_save("sglang_patchify_audio", audio_hidden_states)
                _ltx2_debug_save("sglang_patchify_video_manual", manual_patchify_video)
                _ltx2_debug_save("sglang_patchify_audio_manual", manual_patchify_audio)
                debug_dir = _LTX2_DEBUG_STATE["dir"]
                if debug_dir:
                    patchify_type = type(self.patchify_proj).__name__
                    audio_patchify_type = type(self.audio_patchify_proj).__name__
                    patchify_base = getattr(self.patchify_proj, "base_layer", None)
                    audio_patchify_base = getattr(
                        self.audio_patchify_proj, "base_layer", None
                    )
                    with open(
                        os.path.join(debug_dir, "sglang_patchify_proj_debug.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"patchify_type={patchify_type}\n")
                        f.write(f"audio_patchify_type={audio_patchify_type}\n")
                        f.write(
                            "patchify_has_base_layer="
                            f"{patchify_base is not None}\n"
                        )
                        f.write(
                            "audio_patchify_has_base_layer="
                            f"{audio_patchify_base is not None}\n"
                        )
                        if patchify_base is not None:
                            f.write(
                                "patchify_base_type="
                                f"{type(patchify_base).__name__}\n"
                            )
                            f.write(
                                "patchify_merged="
                                f"{getattr(self.patchify_proj, 'merged', 'NA')}\n"
                            )
                            f.write(
                                "patchify_disable_lora="
                                f"{getattr(self.patchify_proj, 'disable_lora', 'NA')}\n"
                            )
                        if audio_patchify_base is not None:
                            f.write(
                                "audio_patchify_base_type="
                                f"{type(audio_patchify_base).__name__}\n"
                            )
                            f.write(
                                "audio_patchify_merged="
                                f"{getattr(self.audio_patchify_proj, 'merged', 'NA')}\n"
                            )
                            f.write(
                                "audio_patchify_disable_lora="
                                f"{getattr(self.audio_patchify_proj, 'disable_lora', 'NA')}\n"
                            )

            # 3. Prepare timestep embeddings
            # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "video"
                _ltx2_debug_save("sglang_timestep_video", timestep.flatten())
            temb, embedded_timestep = self.adaln_single(
                timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(
                batch_size, -1, embedded_timestep.size(-1)
            )

            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "audio"
                _ltx2_debug_save("sglang_timestep_audio", audio_timestep.flatten())
            temb_audio, audio_embedded_timestep = self.audio_adaln_single(
                audio_timestep.flatten(),
                hidden_dtype=audio_hidden_states.dtype,
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_timestep = audio_embedded_timestep.view(
                batch_size, -1, audio_embedded_timestep.size(-1)
            )
            if self.prompt_modulation:
                temb_prompt, _ = self.prompt_adaln_single(
                    timestep.flatten(), hidden_dtype=hidden_states.dtype
                )
                temb_prompt_audio, _ = self.audio_prompt_adaln_single(
                    audio_timestep.flatten(), hidden_dtype=audio_hidden_states.dtype
                )
                temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
                temb_prompt_audio = temb_prompt_audio.view(
                    batch_size, -1, temb_prompt_audio.size(-1)
                )
            else:
                temb_prompt = temb_prompt_audio = None
            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = None
                _ltx2_debug_save("sglang_temb_video", temb)
                _ltx2_debug_save("sglang_temb_audio", temb_audio)
                _ltx2_debug_save("sglang_embedded_timestep_video", embedded_timestep)
                _ltx2_debug_save(
                    "sglang_embedded_timestep_audio", audio_embedded_timestep
                )
                ca_video_linear = getattr(
                    self.av_ca_video_scale_shift_adaln_single.linear,
                    "base_layer",
                    self.av_ca_video_scale_shift_adaln_single.linear,
                )
                ca_audio_linear = getattr(
                    self.av_ca_audio_scale_shift_adaln_single.linear,
                    "base_layer",
                    self.av_ca_audio_scale_shift_adaln_single.linear,
                )
                a2v_gate_linear = getattr(
                    self.av_ca_a2v_gate_adaln_single.linear,
                    "base_layer",
                    self.av_ca_a2v_gate_adaln_single.linear,
                )
                v2a_gate_linear = getattr(
                    self.av_ca_v2a_gate_adaln_single.linear,
                    "base_layer",
                    self.av_ca_v2a_gate_adaln_single.linear,
                )
                _ltx2_debug_save(
                    "sglang_av_ca_video_scale_shift_linear_weight",
                    ca_video_linear.weight,
                )
                _ltx2_debug_save(
                    "sglang_av_ca_video_scale_shift_linear_bias",
                    getattr(ca_video_linear, "bias", None),
                )
                if getattr(self.av_ca_video_scale_shift_adaln_single.linear, "lora_A", None) is not None:
                    lora_a = self.av_ca_video_scale_shift_adaln_single.linear.lora_A
                    lora_b = self.av_ca_video_scale_shift_adaln_single.linear.lora_B
                    if hasattr(lora_a, "to_local"):
                        lora_a = lora_a.to_local()
                    if hasattr(lora_b, "to_local"):
                        lora_b = lora_b.to_local()
                    _ltx2_debug_save("sglang_av_ca_video_scale_shift_linear_lora_A", lora_a)
                    _ltx2_debug_save("sglang_av_ca_video_scale_shift_linear_lora_B", lora_b)
                ca_emb = self.av_ca_video_scale_shift_adaln_single.emb.timestep_embedder
                for idx, layer in [("1", ca_emb.linear_1), ("2", ca_emb.linear_2)]:
                    layer_base = getattr(layer, "base_layer", layer)
                    _ltx2_debug_save(
                        f"sglang_av_ca_video_scale_shift_embed_linear_{idx}_weight",
                        layer_base.weight,
                    )
                    _ltx2_debug_save(
                        f"sglang_av_ca_video_scale_shift_embed_linear_{idx}_bias",
                        getattr(layer_base, "bias", None),
                    )
                    if getattr(layer, "lora_A", None) is not None:
                        layer_lora_a = layer.lora_A
                        layer_lora_b = layer.lora_B
                        if hasattr(layer_lora_a, "to_local"):
                            layer_lora_a = layer_lora_a.to_local()
                        if hasattr(layer_lora_b, "to_local"):
                            layer_lora_b = layer_lora_b.to_local()
                        _ltx2_debug_save(
                            f"sglang_av_ca_video_scale_shift_embed_linear_{idx}_lora_A",
                            layer_lora_a,
                        )
                        _ltx2_debug_save(
                            f"sglang_av_ca_video_scale_shift_embed_linear_{idx}_lora_B",
                            layer_lora_b,
                        )
                _ltx2_debug_save(
                    "sglang_av_ca_audio_scale_shift_linear_weight",
                    ca_audio_linear.weight,
                )
                _ltx2_debug_save(
                    "sglang_av_ca_audio_scale_shift_linear_bias",
                    getattr(ca_audio_linear, "bias", None),
                )
                _ltx2_debug_save(
                    "sglang_av_ca_a2v_gate_linear_weight",
                    a2v_gate_linear.weight,
                )
                _ltx2_debug_save(
                    "sglang_av_ca_a2v_gate_linear_bias",
                    getattr(a2v_gate_linear, "bias", None),
                )
                _ltx2_debug_save(
                    "sglang_av_ca_v2a_gate_linear_weight",
                    v2a_gate_linear.weight,
                )
                _ltx2_debug_save(
                    "sglang_av_ca_v2a_gate_linear_bias",
                    getattr(v2a_gate_linear, "bias", None),
                )

            # 3.2. Prepare global modality cross attention modulation parameters
            hidden_dtype = hidden_states.dtype
            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "ca_scale_shift_video"
            temb_ca_scale_shift, embedded_temb_ca_scale_shift = (
                self.av_ca_video_scale_shift_adaln_single(
                timestep.flatten(), hidden_dtype=hidden_dtype
                )
            )
            temb_ca_scale_shift = temb_ca_scale_shift.view(
                batch_size, -1, temb_ca_scale_shift.shape[-1]
            )
            embedded_temb_ca_scale_shift = embedded_temb_ca_scale_shift.view(
                batch_size, -1, embedded_temb_ca_scale_shift.shape[-1]
            )

            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "ca_gate_video"
            temb_ca_gate, embedded_temb_ca_gate = self.av_ca_a2v_gate_adaln_single(
                timestep.flatten() * self.av_ca_timestep_scale_multiplier,
                hidden_dtype=hidden_dtype,
            )
            temb_ca_gate = temb_ca_gate.view(batch_size, -1, temb_ca_gate.shape[-1])
            embedded_temb_ca_gate = embedded_temb_ca_gate.view(
                batch_size, -1, embedded_temb_ca_gate.shape[-1]
            )

            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "ca_scale_shift_audio"
            temb_ca_audio_scale_shift, embedded_temb_ca_audio_scale_shift = (
                self.av_ca_audio_scale_shift_adaln_single(
                    audio_timestep.flatten(), hidden_dtype=audio_hidden_states.dtype
                )
            )
            temb_ca_audio_scale_shift = temb_ca_audio_scale_shift.view(
                batch_size, -1, temb_ca_audio_scale_shift.shape[-1]
            )
            embedded_temb_ca_audio_scale_shift = (
                embedded_temb_ca_audio_scale_shift.view(
                    batch_size, -1, embedded_temb_ca_audio_scale_shift.shape[-1]
                )
            )

            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = "ca_gate_audio"
            temb_ca_audio_gate, embedded_temb_ca_audio_gate = (
                self.av_ca_v2a_gate_adaln_single(
                audio_timestep.flatten() * self.av_ca_timestep_scale_multiplier,
                hidden_dtype=audio_hidden_states.dtype,
                )
            )
            temb_ca_audio_gate = temb_ca_audio_gate.view(
                batch_size, -1, temb_ca_audio_gate.shape[-1]
            )
            embedded_temb_ca_audio_gate = embedded_temb_ca_audio_gate.view(
                batch_size, -1, embedded_temb_ca_audio_gate.shape[-1]
            )
            if debug_forward:
                _LTX2_DEBUG_STATE["time_embed_name"] = None
                _ltx2_debug_save("sglang_temb_ca_scale_shift_video", temb_ca_scale_shift)
                _ltx2_debug_save("sglang_temb_ca_gate_video", temb_ca_gate)
                _ltx2_debug_save(
                    "sglang_temb_ca_scale_shift_audio", temb_ca_audio_scale_shift
                )
                _ltx2_debug_save("sglang_temb_ca_gate_audio", temb_ca_audio_gate)
                _ltx2_debug_save(
                    "sglang_embedded_timestep_ca_scale_shift_video",
                    embedded_temb_ca_scale_shift,
                )
                _ltx2_debug_save(
                    "sglang_embedded_timestep_ca_gate_video",
                    embedded_temb_ca_gate,
                )
                _ltx2_debug_save(
                    "sglang_embedded_timestep_ca_scale_shift_audio",
                    embedded_temb_ca_audio_scale_shift,
                )
                _ltx2_debug_save(
                    "sglang_embedded_timestep_ca_gate_audio",
                    embedded_temb_ca_audio_gate,
                )

            # 4. Prepare prompt embeddings
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            audio_encoder_hidden_states = self.audio_caption_projection(
                audio_encoder_hidden_states
            )
            if debug_forward:
                _ltx2_debug_save(
                    "sglang_caption_linear1_weight",
                    self.caption_projection.linear_1.weight,
                )
                _ltx2_debug_save(
                    "sglang_caption_linear1_bias",
                    self.caption_projection.linear_1.bias,
                )
                _ltx2_debug_save(
                    "sglang_caption_linear2_weight",
                    self.caption_projection.linear_2.weight,
                )
                _ltx2_debug_save(
                    "sglang_caption_linear2_bias",
                    self.caption_projection.linear_2.bias,
                )
                _ltx2_debug_save(
                    "sglang_audio_caption_linear1_weight",
                    self.audio_caption_projection.linear_1.weight,
                )
                _ltx2_debug_save(
                    "sglang_audio_caption_linear1_bias",
                    self.audio_caption_projection.linear_1.bias,
                )
                _ltx2_debug_save(
                    "sglang_audio_caption_linear2_weight",
                    self.audio_caption_projection.linear_2.weight,
                )
                _ltx2_debug_save(
                    "sglang_audio_caption_linear2_bias",
                    self.audio_caption_projection.linear_2.bias,
                )
                _ltx2_debug_save("sglang_caption_projection_video", encoder_hidden_states)
                _ltx2_debug_save(
                    "sglang_caption_projection_audio", audio_encoder_hidden_states
                )
                _ltx2_debug_save(
                    "sglang_encoder_attention_mask", encoder_attention_mask
                )
                _ltx2_debug_save(
                    "sglang_audio_encoder_attention_mask",
                    audio_encoder_attention_mask,
                )

            # 5. Run blocks
            for block in self.transformer_blocks:
                hidden_states, audio_hidden_states = block(
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    # Keep the first 4 args positional to stay compatible with cache-dit's
                    # LTX2 adapter, which treats `audio_hidden_states` as `encoder_hidden_states`
                    # under ForwardPattern.Pattern_0.
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=temb_ca_scale_shift,
                    temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
                    temb_ca_gate=temb_ca_gate,
                    temb_ca_audio_gate=temb_ca_audio_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=ca_video_rotary_emb,
                    ca_audio_rotary_emb=ca_audio_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                )

            # 6. Output layers
            # Video
            scale_shift_values = self.scale_shift_table[None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) + embedded_timestep[:, :, None].to(dtype=hidden_states.dtype)
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states, _ = self.proj_out(hidden_states)

            # Audio
            audio_scale_shift_values = self.audio_scale_shift_table[None, None].to(
                device=audio_hidden_states.device, dtype=audio_hidden_states.dtype
            ) + audio_embedded_timestep[:, :, None].to(dtype=audio_hidden_states.dtype)
            audio_shift, audio_scale = (
                audio_scale_shift_values[:, :, 0],
                audio_scale_shift_values[:, :, 1],
            )
            with torch.autocast(
                device_type=audio_hidden_states.device.type, enabled=False
            ):
                audio_hidden_states = self.audio_norm_out(audio_hidden_states)
            audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
            audio_hidden_states, _ = self.audio_proj_out(audio_hidden_states)

            # Unpatchify if requested (default True for pipeline compatibility)
            return_latents = kwargs.get("return_latents", True)

            if return_latents:
                # Unpatchify Video
                # [B, N, C_out_raw*patch_vol] -> [B, C_out_raw, T, H, W]
                # Requires num_frames, height, width to be known
                if num_frames is not None and height is not None and width is not None:
                    p_t, p_h, p_w = self.patch_size
                    post_t, post_h, post_w = (
                        num_frames // p_t,
                        height // p_h,
                        width // p_w,
                    )
                    b = batch_size
                    hidden_states = hidden_states.reshape(
                        b,
                        post_t,
                        post_h,
                        post_w,
                        self.out_channels_raw,
                        p_t,
                        p_h,
                        p_w,
                    )
                    hidden_states = hidden_states.permute(
                        0, 4, 1, 5, 2, 6, 3, 7
                    ).reshape(b, self.out_channels_raw, num_frames, height, width)

                # Unpatchify Audio
                # [B, N, C_out] -> [B, C_out, T] (or 4D/5D)
                if audio_num_frames is not None:
                    audio_hidden_states = audio_hidden_states.permute(
                        0, 2, 1
                    )  # [B, C, T]

            return hidden_states, audio_hidden_states
        finally:
            _ltx2_debug_end_forward()


# Backward-compatible alias (older internal name).
LTXModel = LTX2VideoTransformer3DModel
EntryClass = LTX2VideoTransformer3DModel
