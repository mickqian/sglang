# SPDX-License-Identifier: Apache-2.0
"""
Wan2.2 Speech-to-Video pipeline.

This pipeline keeps the Wan S2V front half native to SGLang while loading the
official Wan S2V transformer and scheduler as explicit diffusers-style
components through the overlay materialization path.
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file as save_safetensors
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    VAELoader,
)
from sglang.multimodal_gen.runtime.models.encoders.wan_s2v_audio import (
    WanS2VAudioEncoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VBeforeDenoisingStage,
    WanS2VDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _prepare_wan_s2v_shift(batch, server_args: ServerArgs):
    del batch
    return "shift", float(server_args.pipeline_config.flow_shift or 1.0)


class _WanS2VOfficialTextEncoder:
    def __init__(self, helper) -> None:
        self.helper = helper

    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        return self.helper([prompt], device)[0]

    def parameters(self):
        return self.helper.model.parameters()


class _WanS2VOfficialAudioEncoder:
    def __init__(self, helper, device: torch.device) -> None:
        self.helper = helper
        self.device = device

    def encode_audio(
        self,
        audio_path: str,
        *,
        infer_frames: int,
        fps: int,
        dtype: torch.dtype,
        m: int = 0,
    ) -> tuple[torch.Tensor, int]:
        embeddings = self.helper.extract_audio_feat(
            audio_path,
            return_all_layers=True,
            dtype=torch.float32,
        )
        bucket, num_repeat = self.helper.get_audio_embed_bucket_fps(
            embeddings,
            fps=fps,
            batch_frames=infer_frames,
            m=m,
        )
        bucket = bucket.to(self.device, dtype).unsqueeze(0)
        if bucket.ndim == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.ndim == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        return bucket, num_repeat


class _WanS2VOfficialVAE:
    def __init__(self, helper) -> None:
        self.helper = helper
        self.device = helper.device
        self.dtype = helper.dtype

    def to(self, *args, **kwargs):
        return self

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            self.helper.encode(video.to(dtype=self.dtype, device=self.device))
        )

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            self.helper.decode(latents.to(dtype=self.dtype, device=self.device))
        )


def _convert_official_wan_vae_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}

    residual_rewrites = {
        ".residual.0.": ".norm1.",
        ".residual.2.": ".conv1.",
        ".residual.3.": ".norm2.",
        ".residual.6.": ".conv2.",
    }

    def _map_decoder_upsample(idx: int) -> str:
        if idx <= 3:
            block = 0
            inner = idx
        elif idx <= 7:
            block = 1
            inner = idx - 4
        elif idx <= 11:
            block = 2
            inner = idx - 8
        else:
            block = 3
            inner = idx - 12
        if inner == 3:
            return f"decoder.up_blocks.{block}.upsamplers.0"
        return f"decoder.up_blocks.{block}.resnets.{inner}"

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.conv1."):
            new_key = key.replace("encoder.conv1.", "encoder.conv_in.", 1)
        elif key.startswith("encoder.head.0."):
            new_key = key.replace("encoder.head.0.", "encoder.norm_out.", 1)
        elif key.startswith("encoder.head.2."):
            new_key = key.replace("encoder.head.2.", "encoder.conv_out.", 1)
        elif key.startswith("encoder.middle.0."):
            new_key = key.replace(
                "encoder.middle.0.", "encoder.mid_block.resnets.0.", 1
            )
        elif key.startswith("encoder.middle.1."):
            new_key = key.replace(
                "encoder.middle.1.", "encoder.mid_block.attentions.0.", 1
            )
        elif key.startswith("encoder.middle.2."):
            new_key = key.replace(
                "encoder.middle.2.", "encoder.mid_block.resnets.1.", 1
            )
        elif key.startswith("encoder.downsamples."):
            parts = key.split(".")
            idx = int(parts[2])
            suffix = ".".join(parts[3:])
            new_key = f"encoder.down_blocks.{idx}.{suffix}"
            new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
        elif key == "conv1.weight" or key == "conv1.bias":
            new_key = key.replace("conv1", "quant_conv", 1)
        elif key == "conv2.weight" or key == "conv2.bias":
            new_key = key.replace("conv2", "post_quant_conv", 1)
        elif key.startswith("decoder.conv1."):
            new_key = key.replace("decoder.conv1.", "decoder.conv_in.", 1)
        elif key.startswith("decoder.head.0."):
            new_key = key.replace("decoder.head.0.", "decoder.norm_out.", 1)
        elif key.startswith("decoder.head.2."):
            new_key = key.replace("decoder.head.2.", "decoder.conv_out.", 1)
        elif key.startswith("decoder.middle.0."):
            new_key = key.replace(
                "decoder.middle.0.", "decoder.mid_block.resnets.0.", 1
            )
        elif key.startswith("decoder.middle.1."):
            new_key = key.replace(
                "decoder.middle.1.", "decoder.mid_block.attentions.0.", 1
            )
        elif key.startswith("decoder.middle.2."):
            new_key = key.replace(
                "decoder.middle.2.", "decoder.mid_block.resnets.1.", 1
            )
        elif key.startswith("decoder.upsamples."):
            parts = key.split(".")
            idx = int(parts[2])
            suffix = ".".join(parts[3:])
            new_key = f"{_map_decoder_upsample(idx)}.{suffix}"
            new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
        for old, new in residual_rewrites.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        converted[new_key] = value
    return converted


class WanSpeechToVideoPipeline(ComposedPipelineBase):
    pipeline_name = "WanSpeechToVideoPipeline"
    is_video_pipeline = True
    _required_config_modules = [
        "transformer",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_encoder",
    ]

    def _get_checkpoint_root(self, transformer) -> str:
        checkpoint_root = transformer.config.get("wan_checkpoint_root", "checkpoints")
        return os.path.join(transformer.component_model_path, checkpoint_root)

    def _build_native_text_encoder(self, server_args: ServerArgs, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        helper_dir = os.path.join(
            transformer.component_model_path, "_native_text_encoder"
        )
        os.makedirs(helper_dir, exist_ok=True)
        source_pth = os.path.join(checkpoint_root, "models_t5_umt5-xxl-enc-bf16.pth")
        target_pth = os.path.join(helper_dir, "model.pt")
        if not os.path.exists(target_pth):
            if os.path.lexists(target_pth):
                os.unlink(target_pth)
            os.symlink(source_pth, target_pth)
        loader = TextEncoderLoader()
        text_encoder = loader.load_model(
            helper_dir,
            server_args.pipeline_config.text_encoder_configs[0],
            server_args,
            server_args.pipeline_config.text_encoder_precisions[0],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(checkpoint_root, "google", "umt5-xxl"),
            padding_side="right",
        )
        return text_encoder, tokenizer

    def _build_official_text_encoder(self, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        code_root = os.path.join(
            transformer.component_model_path,
            transformer.config.get("wan_code_root", "official_code"),
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        module_configs = __import__("wan.configs", fromlist=["WAN_CONFIGS"])
        module_t5 = __import__("wan.modules.t5", fromlist=["T5EncoderModel"])
        task_name = str(transformer.config.get("wan_task_name", "s2v-14B"))
        config_obj = module_configs.WAN_CONFIGS[task_name]
        helper = module_t5.T5EncoderModel(
            text_len=config_obj.text_len,
            dtype=config_obj.t5_dtype,
            device=transformer.device,
            checkpoint_path=os.path.join(checkpoint_root, config_obj.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_root, config_obj.t5_tokenizer),
            shard_fn=None,
        )
        return _WanS2VOfficialTextEncoder(helper), None

    def _build_official_vae(self, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        code_root = os.path.join(
            transformer.component_model_path,
            transformer.config.get("wan_code_root", "official_code"),
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        module_configs = __import__("wan.configs", fromlist=["WAN_CONFIGS"])
        module_vae = __import__("wan.modules.vae2_1", fromlist=["Wan2_1_VAE"])
        task_name = str(transformer.config.get("wan_task_name", "s2v-14B"))
        config_obj = module_configs.WAN_CONFIGS[task_name]
        helper = module_vae.Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_root, config_obj.vae_checkpoint),
            device=transformer.device,
        )
        return _WanS2VOfficialVAE(helper)

    def _build_official_audio_encoder(self, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        code_root = os.path.join(
            transformer.component_model_path,
            transformer.config.get("wan_code_root", "official_code"),
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        module_audio = __import__(
            "wan.modules.s2v.audio_encoder",
            fromlist=["AudioEncoder"],
        )
        helper = module_audio.AudioEncoder(
            device=transformer.device,
            model_id=os.path.join(checkpoint_root, "wav2vec2-large-xlsr-53-english"),
        )
        return _WanS2VOfficialAudioEncoder(helper, transformer.device)

    def _build_native_vae(self, server_args: ServerArgs, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        helper_dir = os.path.join(transformer.component_model_path, "_native_vae")
        os.makedirs(helper_dir, exist_ok=True)
        config_path = os.path.join(helper_dir, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump({"_class_name": "AutoencoderKLWan"}, fout)

        safetensors_path = os.path.join(
            helper_dir, "diffusion_pytorch_model.safetensors"
        )
        state_dict = torch.load(
            os.path.join(checkpoint_root, "Wan2.1_VAE.pth"),
            map_location="cpu",
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        save_safetensors(
            _convert_official_wan_vae_state_dict(state_dict),
            safetensors_path,
        )

        vae, _ = VAELoader().load(
            helper_dir,
            server_args,
            component_name="vae",
            transformers_or_diffusers="diffusers",
        )
        return vae

    def _build_native_audio_encoder(self, server_args: ServerArgs, transformer):
        checkpoint_root = self._get_checkpoint_root(transformer)
        return WanS2VAudioEncoder(
            config=server_args.pipeline_config.audio_encoder_config,
            component_model_path=os.path.join(
                checkpoint_root, "wav2vec2-large-xlsr-53-english"
            ),
            torch_dtype=torch.float32,
            target_device=(
                "cpu"
                if server_args.audio_encoder_cpu_offload
                else self.get_module("transformer").device
            ),
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        scheduler = self.modules.get("scheduler")
        if scheduler is not None and server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        transformer = self.get_module("transformer")
        self.add_stage(
            WanS2VBeforeDenoisingStage(
                transformer=transformer,
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                audio_encoder=self.get_module("audio_encoder"),
            )
        )
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_prepare_wan_s2v_shift]
        )
        self.add_stage(
            DenoisingStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            )
        )
        self.add_stage(
            WanS2VDecodingStage(
                vae=self.get_module("vae"),
                transformer=transformer,
            )
        )


EntryClass = WanSpeechToVideoPipeline
