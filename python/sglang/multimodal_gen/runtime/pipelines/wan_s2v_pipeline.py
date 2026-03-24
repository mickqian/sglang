# SPDX-License-Identifier: Apache-2.0
"""
Wan2.2 Speech-to-Video pipeline.

Initial implementation uses an official-engine wrapper so that the official
Wan S2V codepath can be driven through SGLang's model registry, overlay
resolution, request validation, and output handling.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VBeforeDenoisingStage,
    WanS2VDecodingStage,
    WanS2VExecutionStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _prepare_wan_s2v_shift(batch, server_args: ServerArgs):
    del batch
    return "shift", float(server_args.pipeline_config.flow_shift or 1.0)


class WanSpeechToVideoPipeline(ComposedPipelineBase):
    pipeline_name = "WanSpeechToVideoPipeline"
    is_video_pipeline = True
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, server_args: ServerArgs):
        if server_args.num_gpus != 1:
            raise NotImplementedError(
                "WanSpeechToVideoPipeline currently supports num_gpus=1 only"
            )
        transformer = self.get_module("transformer")
        scheduler = None
        if hasattr(transformer, "create_standard_scheduler"):
            scheduler = transformer.create_standard_scheduler()
        if scheduler is None:
            scheduler = FlowUniPCMultistepScheduler(
                shift=1.0,
                use_dynamic_shifting=False,
            )
        self.modules["scheduler"] = scheduler

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        transformer = self.get_module("transformer")
        if getattr(transformer, "supports_standard_denoising", False):
            self.add_stage(WanS2VBeforeDenoisingStage(transformer=transformer))
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
            self.add_stage(WanS2VDecodingStage(transformer=transformer))
            return

        self.add_stage(WanS2VExecutionStage(engine=transformer))


EntryClass = WanSpeechToVideoPipeline
