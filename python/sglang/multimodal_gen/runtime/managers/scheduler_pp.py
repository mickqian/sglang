# SPDX-License-Identifier: Apache-2.0
import pickle
import time
from collections import deque
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import zmq
from torch.distributed import Work

from sglang.multimodal_gen.runtime.distributed.disagg_communicators.base_communicator import (
    DisaggCommunicator,
)
from sglang.multimodal_gen.runtime.distributed.disagg_communicators.pytorch_communicator import (
    AsyncWorkRegistry,
)
from sglang.multimodal_gen.runtime.distributed.dist_utils import get_disagg_communicator
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    PPPhase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


class SchedulerPPMixin:
    """

    Process order:
       * 1. Non-DiT Master(Usually the last rank): Receives from ZMQ, process the request with pre-denoising stages, then broadcast to DiT group.
       * 2. DiT (rank 0+): Receives from Non-DiT Master, forward requests to Non-DiT group after processing denoising stage
       * 3. Non-DiT: Receives from DiT Master, final process the post-denoising stages (Decoding), then return the result to client

    Communication Architecture:
       * Non-DiT -> DiT: Uses NCCL backend (can be blocking)
         - Signal: CUDA tensor via p2p_group
         - Data: CUDA tensors via broadcast_group
         - High performance for GPU-to-GPU transfer

       * DiT -> Non-DiT: Uses Gloo backend (truly non-blocking)
         - Signal: CPU tensor via signal_group
         - Data: CPU tensors via signal_group, then moved to CUDA
         - Ensures Non-DiT Master can poll without blocking event loop
    """

    def get_next_batch_to_run_pp(self: "Scheduler", comm: DisaggCommunicator):
        """Pick a request based on priority and execute."""
        req_to_run = None
        req_identity = None

        if comm.is_non_dit_rank():
            # Priority: Decode (Post-Denoising) > Encode (Pre-Denoising)
            if self.post_denoising_queue:
                req_to_run = self.post_denoising_queue.popleft()
            elif self.waiting_queue:
                # encoding reqs
                item = self.waiting_queue.popleft()
                if isinstance(item, tuple):
                    req_identity, req_to_run = item
                else:
                    req_to_run = item
        else:
            # DiT only handles Denoising
            if self.denoising_queue:
                req_to_run = self.denoising_queue.popleft()

        return req_to_run, req_identity

    def process_result(
        self: "Scheduler",
        req: Req,
        output_batch: OutputBatch | None,
        identity,
        comm: DisaggCommunicator,
    ):
        """
        Either:
            1. send to another dit-group if the pipeline hasn't finished, or
            2. return the result to client via ZMQ

        """
        current_phase = req.pp_phase

        if current_phase == PPPhase.PRE_DENOISING:
            assert comm.is_non_dit_rank()
            # finished encoding, non-dit -> dit
            self._async_send_batch_to_dit(req, comm)
        elif current_phase == PPPhase.DENOISING:
            assert comm.is_dit_rank()
            # finished denoising, dit -> non-dit
            self._async_send_batch_to_non_dit(req, comm)
        elif current_phase == PPPhase.POST_DENOISING:
            assert comm.is_non_dit_rank()
            # finished decoding, return to client
            final_ident = identity or getattr(req, "client_identity", None)
            self.return_result(output_batch, final_ident)

    def run_batch(
        self: "Scheduler", comm: DisaggCommunicator, req_to_run: Req, req_identity
    ):
        try:
            # Execute
            output = self.worker.execute_forward([req_to_run])

            # Handle result
            self.process_result(req_to_run, output, req_identity, comm)
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            self._handle_error(OutputBatch(error=str(e)), req_identity)

    def filter_reqs(self: "Scheduler", new_reqs: list[Req], comm: DisaggCommunicator):
        """Select from the received reqs that will actually be processed on this rank, then put them in waiting_queue"""
        if new_reqs:
            if comm.is_non_dit_rank():
                for item in new_reqs:
                    identity = None
                    req = item
                    if isinstance(item, tuple):
                        identity, req = item

                    if req.pp_phase is None:
                        req.pp_phase = PPPhase.PRE_DENOISING
                        req.client_identity = identity
                        self.waiting_queue.append((identity, req))

                    elif req.pp_phase == PPPhase.DENOISING:
                        req.pp_phase = PPPhase.POST_DENOISING
                        self.post_denoising_queue.append(req)
                    else:
                        assert False
            else:
                for item in new_reqs:
                    req = item
                    assert req.pp_phase == PPPhase.PRE_DENOISING
                    req.pp_phase = PPPhase.DENOISING
                    self.denoising_queue.append(req)

        for i, req in self.waiting_queue:
            assert req.pp_phase is not None

    def recv_reqs_pp(self: "Scheduler") -> List[tuple[bytes, Any]]:
        """Receive requests from ZMQ (External). Internal rank-to-rank is handled via registry."""
        if self.receiver is not None:
            try:
                identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)
                recv_reqs = pickle.loads(payload)
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]
                return [(identity, req) for req in recv_reqs]
            except zmq.error.Again:
                pass
        return []

    def event_loop_pp(self: "Scheduler") -> None:
        """
        The main event loop that listens for ZMQ requests and handles pipeline parallelism.
        """
        comm = get_disagg_communicator()
        assert (
            comm is not None
        ), "DisaggregatedExecutor requires an initialized DisaggCommunicator"

        # Queues
        # waiting_queue: inherited from Scheduler, used for Client -> Non-DiT (Phase 1)
        self.post_denoising_queue: deque[Req] = deque()  # DiT -> Non-DiT (Phase 3)
        self.denoising_queue: deque[Req] = deque()  # Non-DiT -> DiT (Phase 2)

        # Async Communication State
        self.async_registry = AsyncWorkRegistry()
        self.pending_sends: deque[Tuple[List[Work], Any, List[torch.Tensor]]] = deque()
        self.max_pending_transfers = 2

        logger.info(
            f"Scheduler PP loop started. Role: {'Non-DiT' if comm.is_non_dit_rank() else 'DiT'}, Rank: {dist.get_rank()}"
        )

        # Initial Setup: Start persistent async listeners
        self._setup_async_receivers(comm)

        # 关键修复：添加全局屏障，确保所有 ranks 都完成了 irecv 注册后再继续
        # 这样可以防止 Non-DiT Master 在 DiT ranks 注册 irecv 之前就发送信号
        logger.info(
            f"[Rank {dist.get_rank()}] Waiting at barrier after setup_async_receivers..."
        )
        dist.barrier()
        logger.info(f"[Rank {dist.get_rank()}] Barrier passed, entering main loop...")

        last_log_time = time.time()
        last_poll_log_time = time.time()
        while self._running:
            # 1. Poll completed async tasks (signals, data transfers)
            # 定期打印 poll 状态以便调试
            if time.time() - last_poll_log_time > 10.0:
                logger.info(
                    f"[Rank {dist.get_rank()}] Polling: async_registry has {len(self.async_registry)} pending works"
                )
                # 详细检查每个 pending work 的状态
                for i, aw in enumerate(self.async_registry.pending_works):
                    work_statuses = [w.is_completed() for w in aw.works]
                    logger.info(
                        f"[Rank {dist.get_rank()}] PendingWork[{i}]: {len(aw.works)} works, "
                        f"statuses={work_statuses}, tensor_ptrs={[t.data_ptr() for t in aw.tensors]}, "
                        f"tensor_values={[t.item() if t.numel() == 1 else t.shape for t in aw.tensors]}"
                    )
                last_poll_log_time = time.time()

            completed = self.async_registry.poll()
            if completed:
                logger.info(
                    f"[Rank {dist.get_rank()}] Poll completed {len(completed)} async works"
                )

            self._cleanup_pending_sends(comm)

            # 2. Receive external requests (ZMQ)
            try:
                new_reqs = self.recv_reqs_pp()
                if new_reqs:
                    logger.info(
                        f"[Rank {dist.get_rank()}] Received {len(new_reqs)} new requests from ZMQ"
                    )
            except Exception as e:
                logger.error(
                    f"Error receiving requests from client: {e}", exc_info=True
                )
                new_reqs = []

            # 3. Filter and queue requests
            self.filter_reqs(new_reqs, comm)

            # 4. Schedule & Execute
            req, identity = self.get_next_batch_to_run_pp(comm)
            if req:
                logger.info(
                    f"[Rank {dist.get_rank()}] RUNNING batch: req_id={getattr(req, 'req_id', 'N/A')}, phase={req.pp_phase}"
                )
                self.run_batch(comm, req, identity)

            # 5. Idle Sleep and Periodic Status Log
            is_idle = (
                not self.waiting_queue
                and not self.post_denoising_queue
                and not self.denoising_queue
                and len(self.async_registry) == 0
                and not self.pending_sends
            )

            if is_idle:
                time.sleep(0.001)
                if time.time() - last_log_time > 5.0:
                    logger.info(
                        f"[Rank {dist.get_rank()}] IDLE: waiting={len(self.waiting_queue)}, post={len(self.post_denoising_queue)}, denois={len(self.denoising_queue)}, async={len(self.async_registry)}"
                    )
                    last_log_time = time.time()

        logger.info("Scheduler PP loop terminated.")
        if self.receiver is not None:
            self.receiver.close()
        self.context.term()

    def _setup_async_receivers(self, comm: DisaggCommunicator):
        """Initialize persistent listeners for inter-group communication."""
        if comm.is_non_dit_rank():
            if self.receiver is not None:
                # Non-DiT Master: Listen for results from DiT Master
                self._start_listening_for_dit_results(comm)
                logger.info("Non-DiT Master: Started async result listener")
        else:
            # All DiT Ranks: Listen for signals from Non-DiT Master
            self._start_listening_for_non_dit_signals(comm)
            logger.info(f"DiT Rank {dist.get_rank()}: Started async signal listener")

    def _start_listening_for_dit_results(self, comm: DisaggCommunicator):
        """Register a non-blocking receiver for results coming back from DiT."""
        signal_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        work = comm.irecv_signal_from_dit(signal_tensor)

        def on_signal_received(aw):
            print(f"on_signal_received on non-dit-ranks")
            # 1. Receive the actual object (via Gloo)
            obj = comm.recv_object_from_dit(known_size_tensor=None)
            print(f"reqs received on non-dit-ranks")
            if obj:
                self.filter_reqs([obj], comm)
            # 2. Restart listener
            self._start_listening_for_dit_results(comm)

        if work:
            # 添加 expected_signal_value=1 作为 Gloo backend workaround
            self.async_registry.add(
                work,
                signal_tensor,
                callback=on_signal_received,
                expected_signal_value=1,
            )

    def _start_listening_for_non_dit_signals(self, comm: DisaggCommunicator):
        """Register a non-blocking listener for dispatch signals from Non-DiT Master."""
        # Use CPU tensor for Gloo dispatch signal
        signal_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        logger.info(
            f"[Rank {dist.get_rank()}] _start_listening_for_non_dit_signals: "
            f"Creating signal_tensor at {signal_tensor.data_ptr()}, initial value: {signal_tensor.item()}"
        )
        work = comm.irecv_size_from_non_dit(signal_tensor)

        def on_dispatch_signal(aw):
            logger.info(
                f"[Rank {dist.get_rank()}] on_dispatch_signal TRIGGERED! "
                f"signal_tensor value: {signal_tensor.item()}"
            )
            print(f"on_dispatch_signal start on dit-ranks")
            # Signal received: all DiT ranks join the data broadcast (NCCL)
            reqs = comm.broadcast_object_from_non_dit(None)
            print(f"reqs received on dit-ranks: {reqs}")
            if reqs:
                reqs = [reqs] if not isinstance(reqs, list) else reqs
                for req in reqs:
                    print(f"{req.latents=}")
                    # print(f"{req.raw_latent_shape=}")
                self.filter_reqs(reqs, comm)
            # Restart listener
            self._start_listening_for_non_dit_signals(comm)

        if work:
            logger.info(
                f"[Rank {dist.get_rank()}] Registering irecv work in async_registry, "
                f"expecting signal value to change to 1"
            )
            # 添加 expected_signal_value=1 作为 Gloo backend workaround
            self.async_registry.add(
                work,
                signal_tensor,
                callback=on_dispatch_signal,
                expected_signal_value=1,
            )
        else:
            logger.warning(
                f"[Rank {dist.get_rank()}] irecv_size_from_non_dit returned None!"
            )

    def _cleanup_pending_sends(self, comm: DisaggCommunicator):
        """Check and clean up completed async sends."""
        while self.pending_sends:
            works, _, _ = self.pending_sends[0]

            # Check if all works are completed
            all_completed = all(work is None or work.is_completed() for work in works)

            if all_completed:
                # All works completed, remove from queue
                self.pending_sends.popleft()
            elif len(self.pending_sends) >= self.max_pending_transfers:
                # Queue full, wait for oldest to complete
                comm.wait_all_works(works)
                self.pending_sends.popleft()
            else:
                # Not completed yet and queue not full, keep it
                break

    def _handle_error(self: "Scheduler", output_batch: OutputBatch, identity):
        final_ident = identity or getattr(output_batch, "client_identity", None)
        if final_ident:
            self.return_result(output_batch, final_ident)

    # --- Communication Helpers ---

    def _async_send_batch_to_dit(self, batch: Req, comm: DisaggCommunicator):
        """Non-DiT Master sends processed request to ALL DiT ranks."""
        logger.info(
            f"[Rank {dist.get_rank()}] _async_send_batch_to_dit: Dispatching Req to DiT, phase={batch.pp_phase}"
        )
        print(
            f"[Rank {dist.get_rank()}] Dispatching Req to DiT: phase={batch.pp_phase}"
        )

        # 1. Send signal to ALL DiT ranks via Gloo (MUST use CPU)
        signal_tensor = torch.tensor([1], dtype=torch.long, device="cpu")
        logger.info(
            f"[Rank {dist.get_rank()}] Created signal_tensor at {signal_tensor.data_ptr()}, value: {signal_tensor.item()}"
        )
        works = comm.isend_signal_to_dit(signal_tensor)

        if works:
            logger.info(
                f"[Rank {dist.get_rank()}] isend_signal_to_dit returned {len(works)} work handles"
            )
            # Registry keeps signal_tensor alive
            self.async_registry.add(works, signal_tensor)

            # Wait for dispatch signals to be sent before NCCL broadcast.
            logger.info(
                f"[Rank {dist.get_rank()}] Waiting for Gloo signal sends to complete..."
            )
            for i, w in enumerate(works):
                if w is not None:
                    w.wait()
                    logger.info(
                        f"[Rank {dist.get_rank()}] Gloo signal send {i} completed"
                    )
            logger.info(f"[Rank {dist.get_rank()}] All Gloo signal sends completed")
        else:
            logger.warning(
                f"[Rank {dist.get_rank()}] isend_signal_to_dit returned empty works list!"
            )

        # 2. Broadcast the object (NCCL)
        logger.info(
            f"[Rank {dist.get_rank()}] Starting NCCL broadcast_object_from_non_dit..."
        )
        print(f"Blocking send reqs to dit on non-dit ranks")
        comm.broadcast_object_from_non_dit(batch)
        logger.info(
            f"[Rank {dist.get_rank()}] NCCL broadcast_object_from_non_dit completed"
        )
        print(f"Sent reqs to dit on non-dit ranks")

    def _async_send_batch_to_non_dit(
        self, batch: Req, comm: DisaggCommunicator
    ) -> List[Optional[Work]]:
        """DiT Master -> Non-DiT Master (P2P via Gloo, non-blocking)"""
        print(
            f"[Rank {dist.get_rank()}] Sending Req back to Non-DiT: phase={batch.pp_phase}"
        )

        # 1. Send signal to Non-DiT Master (Gloo, non-blocking)
        signal_tensor = torch.tensor([1], dtype=torch.long, device="cpu")
        work_signal, persistent_signal_tensor = comm.isend_signal_to_non_dit(
            signal_tensor
        )

        # 2. Send the actual object (Gloo, non-blocking)
        works_obj, keep_alive_tensors = comm.isend_object_to_non_dit(batch)

        all_works = []
        if work_signal:
            all_works.append(work_signal)
        all_works.extend(works_obj)

        all_keep_alive = keep_alive_tensors
        if persistent_signal_tensor is not None:
            all_keep_alive.append(persistent_signal_tensor)

        if all_works:
            # Registry manages lifetime and cleanup
            self.async_registry.add(all_works, all_keep_alive, metadata={"req": batch})
            self.pending_sends.append((all_works, batch, all_keep_alive))

        return all_works
