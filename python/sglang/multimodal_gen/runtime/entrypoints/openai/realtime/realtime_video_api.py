# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import os
import shutil
import tempfile

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb
from PIL import Image
from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeVideoMode,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    prepare_request,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()


async def _send_encoded_frame_batches(
    ws: WebSocket,
    encoded_frame_batches: list[list[bytes]],
    *,
    content_type: str,
    chunk_index_start: int,
    request_id: str,
) -> None:
    chunk_index = chunk_index_start
    for encoded_frames in encoded_frame_batches:
        await ws.send_bytes(
            packb(
                {
                    "type": "frame_batch",
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "content_type": content_type,
                    "num_frames": len(encoded_frames),
                    "frames": encoded_frames,
                    "total_size": sum(len(frame) for frame in encoded_frames),
                }
            )
        )
        chunk_index += 1


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            # For stream-driven v2v (no first_frame),
            # wait until enough frames are buffered for this block.
            if (
                session.request is not None
                and session.is_v2v_enabled()
                and session.request.first_frame is None
            ):
                while not session.has_pending_video_frames():
                    await asyncio.sleep(0.01)

            session.new_request()

            # send to scheduler and generate video chunk
            server_args = get_global_server_args()
            sampling_params = session.build_sampling_params()
            batch = prepare_request(
                server_args=server_args,
                sampling_params=sampling_params,
            )
            batch.session = session.realtime_session
            batch.extra["realtime_session_id"] = session.id
            batch.block_idx = session.generate_chunk_cnt
            chunk_size = batch.extra.get("chunk_size", 1)
            control_chunk = session.sample_control_chunk(chunk_size)
            if control_chunk is not None:
                logger.debug(
                    "consume realtime control, session_id=%s, block_idx=%s, chunk_size=%s, num_control_frames=%s",
                    session.id,
                    batch.block_idx,
                    chunk_size,
                    len(control_chunk),
                )
                batch.extra["actions"] = control_chunk
            batch.input_video = (
                session.sample_video_frames() if session.is_v2v_enabled() else None
            )
            _, result = await process_generation_batch(async_scheduler_client, batch)

            if result.encoded_frame_batches is not None:
                await _send_encoded_frame_batches(
                    ws,
                    result.encoded_frame_batches,
                    content_type=result.encoded_frame_content_type,
                    chunk_index_start=batch.block_idx,
                    request_id=session.request_id,
                )

            # finish
            session.generate_chunk_completed()

            logger.info(
                f"generate video chunk, "
                f"request_id: {session.request_id},"
                f"chunk_cnt: {session.generate_chunk_cnt},"
            )

        except asyncio.CancelledError:
            logger.info(f"generation completed, session_id: {session.id}")
            break
        except Exception as e:
            err_msg = str(e).splitlines()[0]
            logger.error(f"error during generate loop: {err_msg}")
            try:
                await write_error_msg(f"error during generate loop: {err_msg}", ws)
            except Exception as e:
                logger.error(f"error during sending complete msg: {e}")
                pass
            break


async def _listen_actions(ws: WebSocket, session: GenerateSession):
    async for data in ws.iter_bytes():
        data = unpackb(data)
        try:
            realtime_action = RealtimeAction.model_validate(data)
            if realtime_action.type == "video":
                if not session.is_v2v_enabled():
                    logger.warning(
                        "ignore video action in non-v2v mode, session_id=%s",
                        session.id,
                    )
                    await write_error_msg(
                        "video action requires mode=v2v (or first_frame in auto mode)",
                        ws,
                    )
                    continue

                encoded_frames = list(realtime_action.video_frames or [])
                if realtime_action.video_frame is not None:
                    encoded_frames.append(realtime_action.video_frame)
                if len(encoded_frames) == 0:
                    raise ValueError(
                        "video action requires video_frame or video_frames"
                    )

                frames = []
                for frame_bytes in encoded_frames:
                    image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    frames.append(image)
                session.append_video_frames(frames)
                total_bytes = sum(len(frame) for frame in encoded_frames)
                action_log = (
                    f"type=video, num_frames={len(encoded_frames)}, "
                    f"total_bytes={total_bytes}"
                )
            elif realtime_action.type == "control":
                control_chunk = realtime_action.control_chunk
                if control_chunk is not None:
                    session.append_control_chunk(control_chunk)
                    action_log = (
                        f"type=control, mode=chunk, chunk_len={len(control_chunk)}"
                    )
                else:
                    raise ValueError("control action requires control_chunk")
            else:
                if not realtime_action.action_content:
                    raise ValueError("prompt action requires action_content")
                session.append_action(realtime_action)
                action_log = (
                    f"type=prompt, content_len={len(realtime_action.action_content)}"
                )
            logger.debug(
                f"receive realtime action, session_id: {session.id}, {action_log}"
            )
        except (ValidationError, ValueError) as e:
            action_type = data.get("type") if isinstance(data, dict) else None
            logger.warning(f"invalid action, type={action_type}, error={e}")
            await write_error_msg("invalid action", ws)
            continue


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = unpackb(await ws.receive_bytes())
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            mode_raw = data.get("mode")
            if mode_raw is None:
                mode = None
            else:
                if isinstance(mode_raw, bytes):
                    try:
                        mode_raw = mode_raw.decode("utf-8")
                    except Exception as e:
                        raise ValueError("mode must be a utf-8 string") from e
                if not isinstance(mode_raw, str):
                    raise ValueError("mode must be one of: t2v, v2v")
                try:
                    mode = RealtimeVideoMode(mode_raw)
                except Exception as e:
                    raise ValueError("mode must be one of: t2v, v2v") from e
            if mode == RealtimeVideoMode.T2V and data.get("first_frame") is not None:
                raise ValueError("first_frame is not allowed when mode=t2v")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            # TODO(puf147): convert RGB for krea
            # params.start_frame = Image.open(params.start_frame).convert("RGB")
            if realtime_req.first_frame is not None:
                server_args = get_global_server_args()
                if server_args.input_save_path is not None:
                    uploads_dir = server_args.input_save_path
                    os.makedirs(uploads_dir, exist_ok=True)
                else:
                    if session.input_temp_dir is None:
                        session.input_temp_dir = tempfile.mkdtemp(
                            prefix="sglang_input_"
                        )
                    uploads_dir = session.input_temp_dir

                target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
                image_path = await save_image_to_path(
                    realtime_req.first_frame, target_path
                )
                realtime_req.first_frame = image_path

            # Keep session state update atomic with validated request.
            session.set_mode(mode)
            session.set_request(realtime_req)
            break
        except Exception as e:
            logger.warning(
                f"invalid generate request, session_id: {session.id}, error={e}"
            )
            await write_error_msg("invalid generate request", ws)
            continue


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    if _ACTIVE_SESSION_IDS:
        logger.warning(
            "reject realtime session because another session is active: %s",
            sorted(_ACTIVE_SESSION_IDS),
        )
        try:
            await write_error_msg(
                "another realtime session is already active", websocket
            )
        finally:
            await websocket.close(code=1008)
        return

    session = GenerateSession()
    _ACTIVE_SESSION_IDS.add(session.id)
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(websocket, session)

        # generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # listen for user actions
        listen_task = asyncio.create_task(_listen_actions(websocket, session))

        wait_tasks = [generate_task, listen_task]
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

    except WebSocketDisconnect:
        logger.info(f"client disconnected, session_id: {session.id}")
    finally:
        logger.info(f"terminating session, session_id: {session.id}")
        _ACTIVE_SESSION_IDS.discard(session.id)
        if generate_task and not generate_task.done():
            generate_task.cancel()
        if listen_task and not listen_task.done():
            listen_task.cancel()
        try:
            await async_scheduler_client.forward(
                ReleaseRealtimeSessionReq(session_id=session.id)
            )
        except Exception as e:
            logger.warning(
                "failed to release realtime session on scheduler, session_id=%s, error=%s",
                session.id,
                e,
            )
        if session.input_temp_dir is not None:
            shutil.rmtree(session.input_temp_dir, ignore_errors=True)
        if session:
            session.dispose()


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "error", "content": error_msg}))
