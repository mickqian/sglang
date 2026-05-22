# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from collections import deque
from typing import TYPE_CHECKING

from fastapi import WebSocket

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    register_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    REALTIME_SESSION_ID_EXTRA_KEY,
    RETURN_ENCODED_FRAMES_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )


class LingBotWorldRealtimeState:
    def __init__(self):
        self.prompt_queue = deque(maxlen=1)
        self.control_queue = deque(maxlen=512)
        self.last_control_actions: list[str] = []
        self.has_control_state = False

    def clear(self) -> None:
        self.prompt_queue.clear()
        self.control_queue.clear()
        self.last_control_actions = []
        self.has_control_state = False

    def append_prompt(self, prompt: str) -> None:
        self.prompt_queue.append(prompt)

    def _append_control_frame(self, actions: list[str]) -> None:
        normalized = list(actions)
        self.control_queue.append(normalized)
        self.last_control_actions = normalized
        self.has_control_state = True

    def append_control_chunk(self, control_chunk: list[list[str]]) -> None:
        if len(control_chunk) == 0:
            self.last_control_actions = []
            self.has_control_state = True
            return
        for actions in control_chunk:
            self._append_control_frame(actions)

    def sample_prompt(self) -> str:
        return self.prompt_queue.popleft()

    def sample_control_chunk(self, chunk_size: int) -> list[list[str]] | None:
        if chunk_size <= 0:
            return None

        chunk: list[list[str]] = []
        while len(chunk) < chunk_size and len(self.control_queue) > 0:
            chunk.append(list(self.control_queue.popleft()))

        if len(chunk) == 0 and not self.has_control_state:
            # Keep emitting an explicit no-op control chunk before any user
            # control arrives.
            return [[] for _ in range(chunk_size)]

        pad_actions = list(self.last_control_actions)
        while len(chunk) < chunk_size:
            chunk.append(list(pad_actions))
        return chunk


class LingBotWorldRealtimeAdapter:
    name = "lingbot_world"

    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    def create_state(self) -> LingBotWorldRealtimeState:
        return LingBotWorldRealtimeState()

    def _state(self, session: GenerateSession) -> LingBotWorldRealtimeState:
        state = session.adapter_state
        if not isinstance(state, LingBotWorldRealtimeState):
            raise TypeError("LingBot realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        if request.first_frame is None:
            return

        server_args = get_global_server_args()
        if server_args.input_save_path is not None:
            uploads_dir = server_args.input_save_path
            os.makedirs(uploads_dir, exist_ok=True)
        else:
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
            uploads_dir = session.input_temp_dir

        target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
        image_path = await save_image_to_path(request.first_frame, target_path)
        request.first_frame = image_path

    def ingest_action(
        self,
        session: GenerateSession,
        action: RealtimeAction,
    ) -> str:
        state = self._state(session)
        if action.type == "control":
            control_chunk = action.control_chunk
            if control_chunk is None:
                raise ValueError("control action requires control_chunk")
            state.append_control_chunk(control_chunk)
            return f"type=control, chunk_len={len(control_chunk)}"
        if action.type == "prompt":
            if not action.prompt:
                raise ValueError("prompt action requires prompt")
            state.append_prompt(action.prompt)
            return f"type=prompt, prompt_len={len(action.prompt)}"
        raise ValueError(f"unsupported action type: {action.type}")

    def build_sampling_params(self, session: GenerateSession):
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        if session.generate_chunk_cnt == 0:
            prompt = request.prompt
        elif len(state.prompt_queue) > 0:
            prompt = state.sample_prompt()
            request.prompt = prompt
        else:
            prompt = request.prompt

        return build_sampling_params(
            session.request_id,
            prompt=prompt,
            size=request.size,
            num_frames=request.num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=session.request_id,
            save_output=False,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            guidance_scale_2=request.guidance_scale_2,
            negative_prompt=request.negative_prompt,
            enable_teacache=request.enable_teacache,
            enable_frame_interpolation=request.enable_frame_interpolation,
            frame_interpolation_exp=request.frame_interpolation_exp,
            frame_interpolation_scale=request.frame_interpolation_scale,
            frame_interpolation_model_path=request.frame_interpolation_model_path,
            enable_upscaling=request.enable_upscaling,
            upscaling_model_path=request.upscaling_model_path,
            upscaling_scale=request.upscaling_scale,
            diffusers_kwargs=request.diffusers_kwargs,
            profile=request.profile,
            num_profiled_timesteps=request.num_profiled_timesteps,
            profile_all_stages=request.profile_all_stages,
            perf_dump_path=request.perf_dump_path,
            output_path=request.output_path,
            output_compression=request.output_compression,
            output_quality=request.output_quality,
        )

    def prepare_request(self, session: GenerateSession, batch: Req) -> Req:
        state = self._state(session)
        batch.session = session.realtime_session
        batch.extra[REALTIME_SESSION_ID_EXTRA_KEY] = session.id
        batch.extra[RETURN_ENCODED_FRAMES_EXTRA_KEY] = True
        batch.block_idx = session.generate_chunk_cnt
        chunk_size = batch.extra.get("chunk_size", 1)
        control_chunk = state.sample_control_chunk(chunk_size)
        if control_chunk is not None:
            batch.extra["actions"] = control_chunk
        return batch

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        return await self.output_adapter.send(ws, session, result, batch)

    def on_chunk_complete(self, session: GenerateSession, result: OutputBatch) -> None:
        del result
        session.generate_chunk_completed()

    def dispose(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, LingBotWorldRealtimeState):
            state.clear()


register_realtime_model_adapter(
    LingBotWorldCausalDMDConfig,
    LingBotWorldRealtimeAdapter,
)
