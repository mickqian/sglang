# SPDX-License-Identifier: Apache-2.0
import io
import pickle
import time
from collections import deque
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import zmq
from torch.distributed import Work

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
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

    # TODO: improve this by let rank 0 to be non-dit master
    Process order:
       * 1. DiT Master (Rank 0): Receives from ZMQ, forwards to Non-DiT Master, and returns an empty list (delegating Phase 1 processing to Non-DiT ranks).
       * 2. Non-DiT: Receives from DiT Master, forward requests to DiT group after processing pre-denoising stages
       * 3. DiT: Receives from Non-DiT Master, forward requests to Non-DiT group after processing denoising stage
       * 4. Non-DiT: Receives from DiT Master, final process the post-denoising stages (Decoding), then return the result
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
        # Check if this is a non-dit rank in disagg mode
        assert self.server_args.enable_disagg

        comm = get_disagg_communicator()

        if comm is not None:
            is_non_dit_rank = comm.is_non_dit_rank()
        else:
            is_non_dit_rank = False

        if self.receiver is not None:
            # receives reqs on non-dit master
            try:
                identity, _, payload = self.receiver.recv_multipart()
                recv_reqs = pickle.loads(payload)
            except zmq.error.Again:
                # no request received
                recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
        else:
            # dit-ranks, try to receive from DisaggCommunicator
            recv_reqs = self._async_receive_batch_from_non_dit(comm)

        # Non-dit ranks receive requests via world broadcast and return immediately
        # They don't participate in dit's sp/cfg/tp parallel groups
        if is_non_dit_rank:
            return recv_reqs

        # # Dit ranks continue with their internal parallel group broadcasts
        # # TODO: fix this condition
        # if self.server_args.sp_degree != 1:
        #     recv_reqs = broadcast_pyobj(
        #         recv_reqs,
        #         self.worker.sp_group.rank,
        #         self.worker.sp_cpu_group,
        #         src=self.worker.sp_group.ranks[0],
        #     )
        #
        # if self.server_args.enable_cfg_parallel:
        #     recv_reqs = broadcast_pyobj(
        #         recv_reqs,
        #         self.worker.cfg_group.rank,
        #         self.worker.cfg_cpu_group,
        #         src=self.worker.cfg_group.ranks[0],
        #     )
        #
        # if self.server_args.tp_size > 1:
        #     recv_reqs = broadcast_pyobj(
        #         recv_reqs,
        #         self.worker.tp_group.rank,
        #         self.worker.tp_cpu_group,
        #         src=self.worker.tp_group.ranks[0],
        #     )

        assert recv_reqs is not None

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

        # Persistent Recv State
        self.recv_size_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        if dist.get_backend() == "nccl":
            self.recv_size_tensor = self.recv_size_tensor.cuda()
        self.recv_size_work: Optional[Work] = None

        logger.info(
            f"Scheduler PP loop started. Role: {'Non-DiT' if comm.is_non_dit_rank() else 'DiT'}"
        )

        # Initial Setup: Start listening for incoming pipeline data
        self._post_recv_size(comm)

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
        """Post an asynchronous receive for the size header."""
        if self.recv_size_work is not None:
            return

        try:
            # We assume a single peer per stage direction for now or master-to-master
            # Pass SHAPE, not Tensor instance
            req_list = [(torch.Size([1]), torch.long)]
            if comm.is_non_dit_rank():
                # Listen to DiT
                tensors, works = comm.batch_irecv_from_dit(req_list)
            else:
                # Listen to Non-DiT
                tensors, works = comm.batch_irecv_from_non_dit(req_list)

            if works:
                self.recv_size_work = works[0]
                self.recv_size_tensor = tensors[
                    0
                ]  # Store the tensor that will be filled
        except Exception as e:
            logger.error(f"Failed to post recv size: {e}")

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

    def _handle_error(self: "Scheduler", req, msg, identity):
        final_ident = identity or getattr(req, "client_identity", None)
        if final_ident:
            self.return_result({"status": "error", "message": msg}, final_ident)

    # --- Communication Helpers ---

    def _async_send_batch_to_dit(self, batch: Req, comm: DisaggCommunicator):
        comm.broadcast_object_from_non_dit(batch)

    def _async_receive_batch_from_non_dit(self, comm: DisaggCommunicator):
        req = comm.broadcast_object_from_non_dit(None)
        return req

    def _async_send_batch_to_non_dit(self, batch: Req, comm) -> List[Optional[Work]]:
        tensors, t_infos, l_infos = self._extract_tensors(batch)
        metadata = self._extract_metadata(batch, t_infos, l_infos)
        all_tensors = self._pack_tensors(metadata, t_infos, l_infos, tensors)
        return comm.batch_isend_to_non_dit(all_tensors)


    def _extract_tensors(self, batch):
        tensors, infos, list_infos = {}, {}, {}
        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors[k] = v
                    infos[k] = (v.shape, v.dtype)
                elif isinstance(v, list) and v and len(v) > 0:
                    ind, act, shp = [], [], []
                    for i, t in enumerate(v):
                        if isinstance(t, torch.Tensor):
                            ind.append(i)
                            act.append(t)
                            shp.append((t.shape, t.dtype))
                    if act:
                        tensors[k] = act
                        list_infos[k] = (len(v), ind, shp)
        return tensors, infos, list_infos

    def _extract_metadata(self, batch, t_infos, l_infos):
        return {
            k: v
            for k, v in batch.__dict__.items()
            if k not in t_infos and k not in l_infos and not k.startswith("_")
        }

    def _pack_tensors(self, metadata, t_infos, l_infos, tensors):
        buf = io.BytesIO()
        pickle.dump((metadata, t_infos, l_infos), buf)
        meta_bytes = torch.tensor(
            bytearray(buf.getvalue()), dtype=torch.uint8, device="cpu"
        )
        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )
        all_tensors = [size_tensor, meta_bytes]

        for name in t_infos:
            t = tensors[name]
            if not t.is_cuda:
                t = t.cuda()
            all_tensors.append(t)
        for name in l_infos:
            for t in tensors[name]:
                if not t.is_cuda:
                    t = t.cuda()
                all_tensors.append(t)
        return all_tensors
