# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.lingbot_world_camera import (
    prepare_lingbot_world_condition,
)


class LingBotWorldConditioningStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        c2ws_plucker_emb = prepare_lingbot_world_condition(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
            device=self.device,
            dtype=torch.bfloat16,
        )
        if c2ws_plucker_emb is not None:
            batch.c2ws_plucker_emb = c2ws_plucker_emb
        return batch

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        result = VerificationResult()
        result.add_check(
            "c2ws_plucker_emb", batch.c2ws_plucker_emb, V.none_or_tensor_with_dims(5)
        )
        return result
