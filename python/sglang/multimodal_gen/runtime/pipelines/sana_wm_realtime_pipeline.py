# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMRealtimeStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

DEFAULT_SANA_WM_TEXT_ENCODER = "Efficient-Large-Model/gemma-2-2b-it"


class SanaWMRealtimePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "SanaWMRealtimePipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
    ]

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        if (
            module_name in {"text_encoder", "tokenizer"}
            and module_name not in server_args.component_paths
        ):
            return maybe_download_model(DEFAULT_SANA_WM_TEXT_ENCODER)
        return super()._resolve_component_path(
            server_args,
            module_name,
            load_module_name,
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(
            SanaWMRealtimeStage(
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                model_path=self.model_path,
            )
        )


EntryClass = SanaWMRealtimePipeline
