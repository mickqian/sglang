from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class LTX2ConnectorArchConfig(AdapterArchConfig):
    audio_connector_attention_head_dim: int = 128
    audio_connector_num_attention_heads: int = 30
    audio_connector_num_layers: int = 2
    audio_connector_num_learnable_registers: int = 128
    audio_gated_attn: bool = False
    audio_hidden_dim: int = 2048
    caption_channels: int = 3840
    causal_temporal_positioning: bool = False
    connector_rope_base_seq_len: int = 4096
    per_modality_projections: bool = False
    proj_bias: bool = False
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    text_proj_in_factor: int = 49
    video_connector_attention_head_dim: int = 128
    video_connector_num_attention_heads: int = 30
    video_connector_num_layers: int = 2
    video_connector_num_learnable_registers: int = 128
    video_gated_attn: bool = False
    video_hidden_dim: int = 4096


@dataclass
class LTX2ConnectorConfig(AdapterConfig):

    arch_config: AdapterArchConfig = field(default_factory=LTX2ConnectorArchConfig)

    prefix: str = "LTX2"
