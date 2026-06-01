# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


CHI_PROMPT = "\n".join(
    [
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
)


def sana_wm_stage1_postprocess_text(
    outputs: BaseEncoderOutput,
    text_inputs: dict,
    *,
    return_attention_mask: bool = False,
):
    del return_attention_mask
    if outputs.last_hidden_state is None:
        raise ValueError("SANA-WM Gemma2 encoder must return last_hidden_state")
    embeds = outputs.last_hidden_state
    mask = text_inputs["attention_mask"]
    select = [0] + list(range(-299, 0))
    return embeds[:, None, select], mask[:, select]


def sana_wm_stage1_preprocess_text(prompt: str) -> str:
    return CHI_PROMPT + prompt


@dataclass
class SanaWMRealtimeConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V
    skip_input_image_preprocess: bool = True
    should_use_guidance: bool = False
    generator_device: str = "cuda"
    dit_config: DiTConfig = field(default_factory=SanaWMConfig)
    vae_config: VAEConfig = field(default_factory=LTXVideoVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = True
    vae_sp: bool = False
    in_channels: int = 128
    out_channels: int = 128
    flow_shift: float = 8.0

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                "padding": "max_length",
                "return_attention_mask": True,
            },
        ]
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (sana_wm_stage1_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_wm_stage1_postprocess_text,)
    )

    @property
    def vae_scale_factor(self):
        return self.vae_config.arch_config.spatial_compression_ratio

    @property
    def vae_temporal_compression(self):
        return self.vae_config.arch_config.temporal_compression_ratio

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        latent_frames = (int(num_frames) - 1) // int(self.vae_temporal_compression) + 1
        return (
            batch_size,
            self.in_channels,
            latent_frames,
            int(batch.height) // int(self.vae_scale_factor),
            int(batch.width) // int(self.vae_scale_factor),
        )

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def shard_latents_for_sp(self, batch, latents):
        return latents, False

    def gather_latents_for_sp(self, latents):
        return latents

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_disable_default_layerwise_offload_min_available_memory_gb=70,
            auto_disable_component_offload_min_available_memory_gb=70,
        )

    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
        tok_kwargs = dict(tok_kwargs)
        tok_kwargs["max_length"] = (
            len(tokenizer.encode(CHI_PROMPT))
            + int(self.dit_config.arch_config.model_max_length)
            - 2
        )
        return tokenizer(prompt, **tok_kwargs)
