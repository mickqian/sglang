# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import tempfile
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeChunkInputs,
    RealtimeModelAdapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.utils import (
    parse_action_string,
    snap_num_frames,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SanaWMRealtimeAdapterState:
    def __init__(self):
        self.events = ConditionEventQueue(max_events={"camera_actions": 1024})
        self.base_condition_inputs: dict[str, Any] = {}
        self.latest_event_id: int | None = None

    def clear(self) -> None:
        self.events.clear()
        self.base_condition_inputs.clear()
        self.latest_event_id = None

    def receive_camera_actions(
        self,
        camera_actions: list[list[str]],
        *,
        replace_pending: bool = False,
        event_id: int | None = None,
    ) -> None:
        signals = [
            ControlSignal(kind="camera_actions", payload=list(actions))
            for actions in camera_actions
        ]
        event = ConditionEvent(kind="camera_actions", payload=signals)
        if replace_pending:
            self.events.replace(event)
        else:
            self.events.push(event)
        self.latest_event_id = event_id

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        action_list = self.events.sample_chunk(
            "camera_actions",
            ConditionSamplingParams(chunk_size=chunk_size, default_item=[]),
        )
        if action_list is None:
            return None
        return [list(actions) for actions in action_list]


class SanaWMRealtimeAdapter(RealtimeModelAdapter):
    name = "sana_wm_realtime"

    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    def create_state(self) -> SanaWMRealtimeAdapterState:
        return SanaWMRealtimeAdapterState()

    def _state(self, session: GenerateSession) -> SanaWMRealtimeAdapterState:
        state = session.adapter_state
        if not isinstance(state, SanaWMRealtimeAdapterState):
            raise TypeError("SANA-WM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError("camera_actions event payload must be list[list[str]]")
        normalized = []
        for frame_actions in payload:
            if not isinstance(frame_actions, list):
                raise ValueError("camera_actions event payload must be list[list[str]]")
            normalized.append([str(action).lower() for action in frame_actions])
        return normalized

    @staticmethod
    def _default_max_chunks(num_frames: int) -> int:
        latent_t = (snap_num_frames(int(num_frames), stride=8) - 1) // 8 + 1
        n_blocks = math.ceil((latent_t - 1) / 3)
        return n_blocks + 2

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        if request.first_frame is None:
            raise ValueError("SANA-WM realtime requires first_frame")

        request.size = "1280x704"
        request.num_frames = int(request.num_frames or 961)
        request.fps = int(request.fps or 16)
        request.num_inference_steps = int(request.num_inference_steps or 4)
        request.guidance_scale = float(request.guidance_scale or 1.0)
        if request.negative_prompt is None:
            request.negative_prompt = ""
        if request.generator_device is None:
            request.generator_device = "cuda"
        if request.max_chunks is None:
            request.max_chunks = self._default_max_chunks(request.num_frames)

        state = self._state(session)
        condition_inputs = dict(request.condition_inputs or {})
        camera_actions = condition_inputs.pop("camera_actions", None)
        if camera_actions is not None:
            state.receive_camera_actions(self._validate_camera_actions(camera_actions))
        state.base_condition_inputs = condition_inputs

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

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        del session

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind == "camera_actions":
            camera_actions = self._validate_camera_actions(event.payload)
            state.receive_camera_actions(
                camera_actions,
                replace_pending=True,
                event_id=event.event_id,
            )
            return f"kind=camera_actions, frames={len(camera_actions)}"
        if event.kind == "action":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("action event payload must be a non-empty string")
            camera_actions = parse_action_string(event.payload)
            state.receive_camera_actions(
                camera_actions,
                replace_pending=True,
                event_id=event.event_id,
            )
            return f"kind=action, frames={len(camera_actions)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def _sample_chunk_inputs(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        condition_inputs = dict(state.base_condition_inputs) if chunk.index == 0 else {}
        camera_actions = state.sample_camera_actions(chunk_size)
        if camera_actions is not None:
            condition_inputs["camera_actions"] = camera_actions
        return RealtimeChunkInputs(
            prompt=request.prompt,
            condition_inputs=condition_inputs,
        )

    def _build_sampling_params(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        return build_sampling_params(
            chunk.request_id,
            prompt=chunk_inputs.prompt,
            size=request.size,
            num_frames=request.num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=chunk.request_id,
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
            condition_inputs=chunk_inputs.condition_inputs,
            realtime_chunk_size=chunk_size,
        )

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        arch_config = server_args.pipeline_config.dit_config.arch_config
        chunk_size = int(getattr(arch_config, "num_frames_per_block", 3))
        chunk_inputs = self._sample_chunk_inputs(session, chunk, chunk_size)
        sampling_params = self._build_sampling_params(
            session,
            chunk,
            chunk_inputs,
            chunk_size,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        batch.session = session.realtime_session
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = chunk.index
        batch.realtime_event_id = self._state(session).latest_event_id
        if session.request is not None:
            batch.realtime_output_format = session.request.realtime_output_format
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
        if isinstance(state, SanaWMRealtimeAdapterState):
            state.clear()
        self.output_adapter.reset()
