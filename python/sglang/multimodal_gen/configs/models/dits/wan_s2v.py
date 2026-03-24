# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig


@dataclass
class WanS2VArchConfig(WanVideoArchConfig):
    # Official Wan2.2 S2V transformer defaults.
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    hidden_size: int = 2048
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    ffn_dim: int = 8192
    num_layers: int = 32
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6

    # S2V-specific conditioning controls from the official implementation.
    cond_dim: int = 16
    audio_dim: int = 5120
    num_audio_token: int = 4
    enable_adain: bool = False
    adain_mode: str = "attn_norm"
    audio_inject_layers: list[int] = field(
        default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 27]
    )
    zero_init: bool = False
    zero_timestep: bool = False
    enable_motioner: bool = True
    add_last_motion: bool = True
    enable_tsm: bool = False
    trainable_token_pos_emb: bool = False
    motion_token_num: int = 1024
    enable_framepack: bool = False
    framepack_drop_mode: str = "drop"
    model_type: str = "s2v"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanS2VConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanS2VArchConfig)
    prefix: str = "WanS2V"
