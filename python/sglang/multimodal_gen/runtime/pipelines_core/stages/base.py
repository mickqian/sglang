# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class StageParallelismType(Enum):
    # execute on all gpus
    REPLICATED = auto()
    # executed on main rank only
    MAIN_RANK_ONLY = auto()
    # this stage requires a cfg-parallel
    CFG_PARALLEL = auto()


class StageVerificationError(Exception):
    """Exception raised when stage verification fails."""

    pass


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.

    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

    def __init__(self):
        self.server_args = get_global_server_args()

    def log_info(self, msg, *args):
        """Logs an informational message with the stage name as a prefix."""
        if self.server_args.comfyui_mode:
            return
        logger.info(f"[{self.__class__.__name__}] {msg}", *args)

    def log_warning(self, msg, *args):
        """Logs a warning message with the stage name as a prefix."""
        logger.warning(f"[{self.__class__.__name__}] {msg}", *args)

    def log_error(self, msg, *args):
        """Logs an error message with the stage name as a prefix."""
        logger.error(f"[{self.__class__.__name__}] {msg}", *args)

    def log_debug(self, msg, *args):
        """Logs a debug message with the stage name as a prefix."""
        logger.debug(f"[{self.__class__.__name__}] {msg}", *args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        Verify the input for the stage.

        Example:
            from sglang.multimodal_gen.runtime.pipelines.stages.validators import V, VerificationResult

            def verify_input(self, batch, server_args):
                result = VerificationResult()
                result.add_check("height", batch.height, V.positive_int_divisible(8))
                result.add_check("width", batch.width, V.positive_int_divisible(8))
                result.add_check("image_latent", batch.image_latent, V.is_tensor)
                return result

        """
        # Default implementation - no verification
        return VerificationResult()

    def maybe_free_model_hooks(self):
        pass

    def load_model(self):
        """
        Load the model for the stage.
        """
        pass

    def offload_model(self):
        """
        Offload the model for the stage.
        """
        pass

    # Default role affinity: ENCODER. Override in subclasses for DENOISING/DECODER.
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    # execute on all ranks by default
    @property
    def parallelism_type(self) -> StageParallelismType:
        # if get_global_server_args().enable_cfg_parallel:
        #     return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        Verify the output for the stage.



        Returns:
            A VerificationResult containing the verification status.
        """
        # Default implementation - no verification
        return VerificationResult()

    def _run_verification(
        self,
        verification_result: VerificationResult,
        stage_name: str,
        verification_type: str,
    ) -> None:
        """
        Run verification and raise errors if any checks fail.

        Args:
            verification_result: Results from verify_input or verify_output
            stage_name: Name of the current stage
            verification_type: "input" or "output"
        """
        if not verification_result.is_valid():
            failed_fields = verification_result.get_failed_fields()
            if failed_fields:
                # Get detailed failure information
                detailed_summary = verification_result.get_failure_summary()

                failed_fields_str = ", ".join(failed_fields)
                error_msg = (
                    f"{verification_type.capitalize()} verification failed for {stage_name}: "
                    f"Failed fields: {failed_fields_str}\n"
                    f"Details: {detailed_summary}"
                )
                raise StageVerificationError(error_msg)

    @property
    def device(self) -> torch.device:
        """Get the device for this stage."""
        return torch.device(
            current_platform.device_type,
        )

    def set_logging(self, enable: bool):
        """
        Enable or disable logging for this stage.

        Args:
            enable: Whether to enable logging.
        """
        self._enable_logging = enable

    def __call__(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the stage's processing on the batch with optional verification and logging.
        Should not be overridden by subclasses.



        Returns:
            The updated batch information after this stage's processing.
        """
        stage_name = self.__class__.__name__
        # Check if verification is enabled (simple approach for prototype)

        # Pre-execution input verification
        try:
            input_result = self.verify_input(batch, server_args)
            self._run_verification(input_result, stage_name, "input")
        except Exception as e:
            logger.error("Input verification failed for %s: %s", stage_name, str(e))
            raise

        # Execute the actual stage logic with unified profiling
        with StageProfiler(
            stage_name,
            logger=logger,
            metrics=batch.metrics,
            log_stage_start_end=not batch.is_warmup
            and not (self.server_args and self.server_args.comfyui_mode),
            perf_dump_path_provided=batch.perf_dump_path is not None,
        ):
            result = self.forward(batch, server_args)

        # Post-execution output verification
        try:
            output_result = self.verify_output(result, server_args)
            self._run_verification(output_result, stage_name, "output")
        except Exception as e:
            logger.error("Output verification failed for %s: %s", stage_name, str(e))
            raise

        return result

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Any]:
        """Run this stage for a group of independent requests.

        A grouped request is still a list of normal ``Req`` objects. The group
        boundary only gives a stage the opportunity to reduce duplicate work.
        The default implementation preserves the single-request contract by
        calling ``self(batch, server_args)`` for every request, so stages that do
        not override this method keep exactly the same behavior as before.

        Stage overrides decide their own reuse granularity. A simple stage may
        group by a single full-stage key, compute once, then copy or split the
        stage-local outputs back to every request. A mixed stage may instead
        reuse only one subprocess, such as positive prompt encoding, while
        still running another subprocess per request. Overrides must preserve
        input order and return one result per input request.

        ``get_dedup_key`` and ``_group_requests_by_dedup_key`` are convenience
        helpers for the full-stage case. They are not required for stages that
        need finer internal grouping.

        This hook is deliberately not a global cache: deduplication is local to
        the current stage and current group. A dedup key must only contain
        fields that can change this stage's outputs. Request metadata such as
        request id, output path, or seed should be excluded unless this stage
        actually reads it.
        """
        return [self(batch, server_args) for batch in batches]

    def get_dedup_key(self, batch: Req, server_args: ServerArgs) -> Any:
        """Return the stage-local equivalence key for grouped execution.

        The key describes the inputs that determine this stage's complete
        output. Stages that do not implement full-stage grouped dedup can ignore
        this method; the default key is unique per request, which means "never
        merge by key".

        When overriding, include every field that can affect this stage and
        exclude fields that only matter to other stages. For tensor or nested
        values, use ``_freeze_for_dedup_key`` so the key is hashable.
        """
        return id(batch)

    @staticmethod
    def _freeze_for_dedup_key(value: Any) -> Any:
        """Convert common nested values into a hashable dedup-key fragment.

        Small tensors include their values so scheduler/timestep overrides can
        distinguish user-provided tensors. Larger tensors include shape, dtype,
        and device only; they should not normally be part of a dedup key unless
        the stage has a stronger equivalence guarantee.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() <= 256:
                return (
                    "tensor",
                    tuple(value.shape),
                    str(value.dtype),
                    tuple(value.detach().cpu().reshape(-1).tolist()),
                )
            return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, PipelineStage._freeze_for_dedup_key(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(PipelineStage._freeze_for_dedup_key(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(PipelineStage._freeze_for_dedup_key(item) for item in value)
            )
        return value

    @staticmethod
    def _group_requests_by_dedup_key(
        batches: list[Req],
        key_fn,
    ) -> list[tuple[Any, list[tuple[int, Req]]]]:
        """Group requests by a stage-local dedup key while preserving order.

        The return value is ``[(key, [(original_index, req), ...]), ...]``.
        Group order follows the first appearance of each key, and requests
        inside a group keep their original relative order. Callers can fill a
        result list by ``original_index`` to preserve input/output ordering.
        """
        groups: dict[Any, list[tuple[int, Req]]] = {}
        for index, batch in enumerate(batches):
            key = key_fn(batch)
            groups.setdefault(key, []).append((index, batch))
        return list(groups.items())

    @abstractmethod
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Forward pass of the stage's processing.

        This method should be implemented by subclasses to provide the forward
        processing logic for the stage.



        Returns:
            The updated batch information after this stage's processing.
        """
        raise NotImplementedError

    def backward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        raise NotImplementedError
