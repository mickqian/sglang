# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime input validation.

The generic input validation stage has to support all diffusion pipelines.  The
LingBot realtime path repeatedly sends chunks for the same session, so the
conditioning image and resolved dimensions can be reused after the first chunk.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotWorldInputValidationState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.condition_image = None
        self.original_condition_image_size = None
        self.height = None
        self.width = None

    def dispose(self):
        super().dispose()
        self.image_path = None
        self.condition_image = None
        self.original_condition_image_size = None
        self.height = None
        self.width = None


class LingBotWorldInputValidationStage(InputValidationStage):
    """Reuse LingBot realtime conditioning image validation across chunks."""

    def _cache_batch(self, batch: Req, state: LingBotWorldInputValidationState) -> None:
        state.image_path = batch.image_path
        state.condition_image = batch.condition_image
        state.original_condition_image_size = batch.original_condition_image_size
        state.height = batch.height
        state.width = batch.width

    def _can_reuse_cached_image(
        self, batch: Req, state: LingBotWorldInputValidationState
    ) -> bool:
        if batch.block_idx == 0 or state.condition_image is None:
            return False
        return (
            batch.image_path in (None, state.image_path)
            and batch.height in (None, state.height)
            and batch.width in (None, state.width)
        )

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)

        state = batch.session.get_or_create_state(LingBotWorldInputValidationState)
        if self._can_reuse_cached_image(batch, state):
            original_image_path = batch.image_path
            batch.image_path = None
            batch.condition_image = state.condition_image
            batch.original_condition_image_size = state.original_condition_image_size
            if batch.height is None:
                batch.height = state.height
            if batch.width is None:
                batch.width = state.width
            try:
                return super().forward(batch, server_args)
            finally:
                batch.image_path = original_image_path

        batch = super().forward(batch, server_args)
        self._cache_batch(batch, state)
        return batch
