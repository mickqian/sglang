"""
Pipeline executor for disaggregated execution.
Note: The actual scheduling and communication logic has been moved to
sglang.multimodal_gen.runtime.managers.scheduler_pp.SchedulerPPMixin.
This executor now only handles stage execution based on the assigned phase.
"""

from enum import Enum, auto
from typing import List

from sglang.multimodal_gen.runtime.distributed.dist_utils import get_disagg_communicator
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    PPPhase,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    TimestepPreparationStage, LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class StageDisaggregationRole(Enum):
    # the role of a stage in a pipeline
    NONE_DENOISE = auto()
    DENOISE = auto()
    COMMON = auto()


def get_stage_disagg_role(stage: PipelineStage):
    if isinstance(stage, DenoisingStage) or isinstance(stage, TimestepPreparationStage) or isinstance(stage,
                                                                                                      LatentPreparationStage):
        return StageDisaggregationRole.DENOISE
    else:
        return StageDisaggregationRole.NONE_DENOISE


class DisaggregatedExecutor(PipelineExecutor):
    """
    Executor that handles the disaggregated pipeline flow.
    It purely executes the stages corresponding to the current PPPhase set by the Scheduler.
    Communication is handled by the Scheduler.
    """

    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)
        self.comm = get_disagg_communicator()
        # Note: We rely on the Scheduler to handle communication and synchronization.

    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Executes stages based on the current PPPhase.
        """
        # Determine split points
        denoise_start_idx = -1
        denoise_end_idx = -1

        for i, stage in enumerate(stages):
            if get_stage_disagg_role(stage) == StageDisaggregationRole.DENOISE:
                if denoise_start_idx == -1:
                    denoise_start_idx = i
                denoise_end_idx = i

        if denoise_start_idx == -1:
            # Fallback for models without explicit denoise stage structure
            return self._run_local(stages, batch)

        pre_denoise_stages = stages[:denoise_start_idx]
        denoise_stages = stages[denoise_start_idx: denoise_end_idx + 1]
        post_denoise_stages = stages[denoise_end_idx + 1:]

        # Determine which stages to run based on Phase
        # The Scheduler sets `batch.pp_phase`.
        current_phase = batch.pp_phase

        stages_to_run = []

        if current_phase == PPPhase.PRE_DENOISING:
            if self.comm.is_non_dit_rank():
                stages_to_run = pre_denoise_stages
            else:
                logger.warning(f"Received PRE_DENOISING phase on DiT rank. Skipping.")

        elif current_phase == PPPhase.DENOISING:
            if self.comm.is_dit_rank():
                stages_to_run = denoise_stages
            else:
                logger.warning(f"Received DENOISING phase on Non-DiT rank. Skipping.")

        elif current_phase == PPPhase.POST_DENOISING:
            if self.comm.is_non_dit_rank():
                stages_to_run = post_denoise_stages
            else:
                logger.warning(f"Received POST_DENOISING phase on DiT rank. Skipping.")

        else:
            raise RuntimeError(
                f"Unexpected Req with empty PPPhase {self.comm.is_dit_rank()=}"
            )

        # Execute selected stages
        for stage in stages_to_run:
            with Timer(stage.__class__.__name__):
                batch = stage(batch, server_args)

        # Construct OutputBatch
        # Note: In PRE/DENOISING phases, we return an empty/partial OutputBatch.
        # Only in POST_DENOISING (or full run) do we return the final result.

        if current_phase == PPPhase.POST_DENOISING or current_phase is None:
            return OutputBatch(
                output=batch.output,
                trajectory_timesteps=batch.trajectory_timesteps,
                trajectory_latents=batch.trajectory_latents,
                trajectory_decoded=getattr(batch, "trajectory_decoded", None),
                timings=batch.timings,
                error=None,
            )
        else:
            # For intermediate phases, return a placeholder or the batch state wrapper
            # The Scheduler ignores the output content mostly, but checks for errors.
            return OutputBatch(error=None)

    def _run_local(self, stages, batch):
        """Fallback: run all stages locally."""
        for stage in stages:
            batch = stage(batch, self.server_args)
        return batch
