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
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.

    Process order:
       * 1. Non-DiT Master(Usually the last rank): Receives from ZMQ, process the request with pre-denoising stages, then broadcast to DiT group.
       * 2. DiT: Receives from Non-DiT Master, forward requests to Non-DiT group after processing denoising stage
       * 3. Non-DiT: Receives from DiT Master, final process the post-denoising stages (Decoding), then return the result to client
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
            if output_batch.error:
                self._handle_error(req, output_batch.error, final_ident)
            elif final_ident:
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
            self._handle_error(req_to_run, str(e), req_identity)

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
        """
        Receive requests from ZMQ or another group with DisaggCommunicator
        """
        assert self.server_args.enable_disagg
        comm = get_disagg_communicator()

        if self.receiver is not None:
            # Non-DiT Master: receive from ZMQ only
            try:
                identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)
                recv_reqs = pickle.loads(payload)
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]
                print(f"[Rank {dist.get_rank()}] Received {len(recv_reqs)} reqs from ZMQ")
                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
            except zmq.error.Again:
                recv_reqs = []
            return recv_reqs
        else:
            # DiT ranks: receive from Non-DiT group
            recv_reqs = self._async_recv_batch_from_non_dit(comm)
            if recv_reqs:
                print(f"[Rank {dist.get_rank()}] Received {len(recv_reqs)} reqs from Non-DiT")
            return recv_reqs

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
        self.pending_sends: deque[Tuple[List[Work], Any]] = deque()
        self.max_pending_transfers = 2

        logger.info(
            f"Scheduler PP loop started. Role: {'Non-DiT' if comm.is_non_dit_rank() else 'DiT'}"
        )

        # Initial Setup: Start listening for incoming pipeline data
        # self._post_recv_size(comm)

        while self._running:
            # 1. Clean up completed sends
            self._cleanup_pending_sends(comm)

            try:
                new_reqs = self.recv_reqs_pp()
            except Exception as e:
                logger.error(
                    f"Error receiving requests from client: {e}", exc_info=True
                )
                new_reqs = []

            self.filter_reqs(new_reqs, comm)

            # 4. Schedule & Execute
            req, identity = self.get_next_batch_to_run_pp(comm)

            if req:
                self.run_batch(comm, req, identity)

            # 5. Idle Sleep
            if (
                not self.waiting_queue
                and not self.post_denoising_queue
                and not self.denoising_queue
                and not self.pending_sends
            ):
                time.sleep(0.001)

        logger.info("Scheduler PP loop terminated.")
        if self.receiver is not None:
            self.receiver.close()
        self.context.term()

    def _cleanup_pending_sends(self, comm: DisaggCommunicator):
        """Check and clean up completed async sends."""
        while self.pending_sends:
            # Simple flow control: if queue full, wait for oldest
            if len(self.pending_sends) >= self.max_pending_transfers:
                works, _ = self.pending_sends[0]
                comm.wait_all_works(works)
                self.pending_sends.popleft()
            else:
                # Poll oldest (best effort)
                works, _ = self.pending_sends[0]
                # If all completed, pop. Note: is_completed might not be reliable on all backends.
                # If not reliable, we just skip until max_pending_transfers forces wait.
                # Assuming we rely on max limit for backpressure.
                break

    def _post_recv_size(self, comm):
        """Removed: no longer used after simplification."""
        pass


    def _handle_error(self: "Scheduler", req, msg, identity):
        final_ident = identity or getattr(req, "client_identity", None)
        if final_ident:
            self.return_result({"status": "error", "message": msg}, final_ident)


    # --- Communication Helpers ---

    def _async_send_batch_to_dit(self, batch: Req, comm: DisaggCommunicator):
        """Non-DiT Master 发送处理完 pre-denoising 的请求给 DiT group"""
        print(f"[Rank {dist.get_rank()}] Sending Req to DiT: phase={batch.pp_phase}")

        # 1. 先发送信号给 DiT Master
        signal_tensor = torch.tensor([1], dtype=torch.long, device="cuda")
        work = comm.isend_signal_to_dit(signal_tensor)
        if work:
            work.wait()

        # 2. 然后广播对象给所有 DiT 节点
        comm.broadcast_object_from_non_dit(batch)

    def _async_send_batch_to_non_dit(self, batch: Req, comm) -> List[Optional[Work]]:
        """DiT Master -> Non-DiT Master (P2P)"""
        print(f"[Rank {dist.get_rank()}] Sending Req to Non-DiT: phase={batch.pp_phase}")
        return comm.isend_object_to_non_dit(batch)

    def _async_recv_batch_from_dit(self, comm: DisaggCommunicator, size_tensor: torch.Tensor):
        """Non-DiT Master receives from DiT Master (P2P)"""
        obj = comm.recv_object_from_dit(size_tensor)
        if obj:
            print(f"[Rank {dist.get_rank()}] Received Req from DiT: phase={getattr(obj, 'pp_phase', 'N/A')}")
        return obj

    def _async_recv_batch_from_non_dit(self, comm: DisaggCommunicator):
        """DiT 节点尝试从 Non-DiT 接收已处理完 pre-denoising 的请求"""
        # 1. DiT Master 尝试接收信号（非阻塞）
        if not hasattr(self, '_recv_signal_work') or self._recv_signal_work is None:
            self._recv_signal_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            self._recv_signal_work = comm.irecv_size_from_non_dit(self._recv_signal_tensor)
            return []

        # 2. 检查信号是否到达
        if self._recv_signal_work is not None and not self._recv_signal_work.is_completed():
            return []

        # 信号到达！清除状态，为下次准备
        signal_value = self._recv_signal_tensor.item()
        self._recv_signal_work = None

        if signal_value == 0:
            # 可能是虚假唤醒，继续监听
            return []

        print(f"[Rank {dist.get_rank()}] Signal received from Non-DiT, joining broadcast...")

        # 3. 所有 DiT 节点进入 broadcast 接收数据
        reqs = comm.broadcast_object_from_non_dit(None)

        if reqs:
            sample = reqs[0] if isinstance(reqs, list) else reqs
            print(f"[Rank {dist.get_rank()}] Received from Non-DiT: phase={getattr(sample, 'pp_phase', 'N/A')}")

        if not isinstance(reqs, list):
            reqs = [reqs] if reqs else []

        return reqs
