# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class RealtimeModelAdapter(Protocol):
    name: str

    def create_state(self) -> Any: ...

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None: ...

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str: ...

    def build_sampling_params(self, session: GenerateSession): ...

    def prepare_request(self, session: GenerateSession, batch: Req) -> Req: ...

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats: ...

    def on_chunk_complete(
        self, session: GenerateSession, result: OutputBatch
    ) -> None: ...

    def dispose(self, session: GenerateSession) -> None: ...


_REALTIME_ADAPTER_REGISTRY: dict[type, type[RealtimeModelAdapter]] = {}


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[RealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> RealtimeModelAdapter:
    pipeline_config = server_args.pipeline_config
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(pipeline_config).__name__}; no realtime adapter is registered."
    )
