# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_sana_wm_block(name: str, module) -> bool:
    del module
    return name.startswith("blocks.") and name.split(".")[1].isdigit()


@dataclass
class SanaWMArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_sana_wm_block])

    patch_size: tuple[int, int, int] = (1, 1, 1)
    in_channels: int = 128
    out_channels: int = 128
    hidden_size: int = 2240
    num_attention_heads: int = 20
    attention_head_dim: int = 112
    num_layers: int = 20
    caption_channels: int = 2304
    model_max_length: int = 300
    num_frames_per_block: int = 3
    num_cached_blocks: int = 2
    sink_size: int = 1
    target_height: int = 704
    target_width: int = 1280

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^transformer\.(.*)$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = self.out_channels


@dataclass
class SanaWMConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaWMArchConfig)
    prefix: str = "SanaWM"
