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
    get_world_group,
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
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


class SchedulerPPMixin:
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
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
            output_batch = self.worker.execute_forward([req_to_run])

            if output_batch.error:
                self._handle_error(req_to_run, output_batch.error, req_identity)
            else:
                self._handle_success(req_to_run, output_batch, req_identity, comm)
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            self._handle_error(req_to_run, str(e), req_identity)

    def filter_reqs(self: "Scheduler", new_reqs: list[Req], comm: DisaggCommunicator):
        """Select from the received reqs that will actually be processed on this rank, then put them in waiting_queue"""
        # Only Non-DiT ranks process new client requests (Phase 1)
        if new_reqs:
            if comm.is_non_dit_rank():
                for item in new_reqs:
                    identity = None
                    req = item
                    if isinstance(item, tuple):
                        identity, req = item

                    req.pp_phase = PPPhase.PRE_DENOISING
                    req.client_identity = identity
                    self.waiting_queue.append((identity, req))
            else:
                # ignore the fresh new reqs on dit-ranks
                pass

        for i, req in self.waiting_queue:
            assert req.pp_phase is not None

    def recv_reqs_pp(self: "Scheduler") -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        # Check if this is a non-dit rank in disagg mode
        is_non_dit_rank = False
        assert self.server_args.enable_disagg
        comm = get_disagg_communicator()
        if comm is not None:
            is_non_dit_rank = comm.is_non_dit_rank()

        if self.receiver is not None:
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
            recv_reqs = None

        # In disagg mode, first broadcast to ALL ranks (dit + non-dit) using world group
        # This ensures non-dit ranks receive the requests
        world_group = get_world_group()
        recv_reqs = broadcast_pyobj(
            recv_reqs,
            world_group.rank_in_group,
            world_group.cpu_group,
            src=0,  # rank 0 is the source (dit master with ZMQ receiver)
        )

        # Non-dit ranks receive requests via world broadcast and return immediately
        # They don't participate in dit's sp/cfg/tp parallel groups
        if is_non_dit_rank:
            return recv_reqs

        # Dit ranks continue with their internal parallel group broadcasts
        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

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

            # 2. Ingest: From Client (ZMQ)
            # IMPORTANT: recv_reqs() performs a collective broadcast internally.
            # All ranks MUST participate to avoid deadlock.
            try:
                new_reqs = self.recv_reqs_pp()
            except Exception as e:
                logger.error(
                    f"Error receiving requests from client: {e}", exc_info=True
                )
                new_reqs = []

            self.filter_reqs(new_reqs, comm)

            # 3. Ingest: From Pipeline (Network)
            self._process_pipeline_recv(comm)

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

    def _process_pipeline_recv(self, comm: DisaggCommunicator):
        """transfer req between dit-ranks and non-dit-ranks"""
        if self.recv_size_work is None:
            self._post_recv_size(comm)
            return

        if self.recv_size_work.is_completed():
            # Size received, proceed to receive the rest synchronously (fast following)
            try:
                # Sync logic for reliability once header is detected
                size = self.recv_size_tensor.item()
                self.recv_size_work = None  # Reset

                # 1. Recv Metadata
                # Pass SHAPE, not Tensor instance
                meta_shape = [(torch.Size([size]), torch.uint8)]
                if comm.is_non_dit_rank():
                    tensors, w = comm.batch_irecv_from_dit(meta_shape)
                else:
                    tensors, w = comm.batch_irecv_from_non_dit(meta_shape)

                comm.wait_all_works(w)
                meta_tensor = tensors[0]

                meta_bytes = meta_tensor.cpu().numpy().tobytes()
                metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_bytes)

                # 2. Prep Batch
                batch = Req(None, None, None)
                batch._recv_metadata = metadata
                batch._recv_tensor_infos = tensor_infos
                batch._recv_list_tensor_infos = list_tensor_infos

                for k, v in metadata.items():
                    setattr(batch, k, v)

                # 3. Recv Data Tensors
                all_shapes = []
                tensor_names = []

                for name, (shape, dtype) in tensor_infos.items():
                    all_shapes.append((shape, dtype))
                    tensor_names.append(("single", name))

                for name, list_info in list_tensor_infos.items():
                    length, indices, shapes = list_info
                    for idx, (s, d) in enumerate(shapes):
                        all_shapes.append((s, d))
                        tensor_names.append(("list", name, idx, length, indices))

                if all_shapes:
                    if comm.is_non_dit_rank():
                        tensors, w = comm.batch_irecv_from_dit(all_shapes)
                    else:
                        tensors, w = comm.batch_irecv_from_non_dit(all_shapes)
                    comm.wait_all_works(w)

                    # Reconstruct
                    idx_t = 0
                    list_buffers = {}
                    for info in tensor_names:
                        t = tensors[idx_t]
                        if info[0] == "single":
                            setattr(batch, info[1], t)
                        else:
                            name, idx, length, indices = (
                                info[1],
                                info[2],
                                info[3],
                                info[4],
                            )
                            if name not in list_buffers:
                                list_buffers[name] = {
                                    "length": length,
                                    "indices": indices,
                                    "tensors": [],
                                }
                            list_buffers[name]["tensors"].append(t)
                        idx_t += 1

                    for name, data in list_buffers.items():
                        res = [None] * data["length"]
                        for i, t in zip(data["indices"], data["tensors"]):
                            res[i] = t
                        setattr(batch, name, res)

                # 4. Enqueue
                if comm.is_non_dit_rank():
                    batch.pp_phase = PPPhase.POST_DENOISING
                    self.post_denoising_queue.append(batch)
                else:
                    batch.pp_phase = PPPhase.DENOISING
                    # Broadcast to other DiT workers
                    self._broadcast_batch_in_dit_group(batch, comm)
                    self.denoising_queue.append(batch)

            except Exception as e:
                logger.error(f"Error processing pipeline recv: {e}", exc_info=True)
            finally:
                # Always repost listener
                self._post_recv_size(comm)

    def _handle_success(
        self: "Scheduler",
        req: Req,
        output_batch: OutputBatch,
        identity,
        comm: DisaggCommunicator,
    ):
        """Handle execution success and transition/send."""
        current_phase = getattr(req, "pp_phase", None)

        if comm.is_non_dit_rank():
            if current_phase == PPPhase.PRE_DENOISING:
                # Finished Encoding -> Send to DiT
                works = self._async_send_batch_to_dit(req, comm)
                self.pending_sends.append((works, req))  # Keep req alive

            elif current_phase == PPPhase.POST_DENOISING:
                # Finished Decoding -> Return to Client
                # If we have an identity (from original ZMQ), send reply
                # Note: identity might be stored in req if it round-tripped
                final_ident = identity or getattr(req, "client_identity", None)
                if final_ident:
                    self.return_result(output_batch, final_ident)

        elif comm.is_dit_rank():
            if current_phase == PPPhase.DENOISING:
                # Finished Denoising -> Send back to Non-DiT
                # Only Master sends
                if dist.get_rank() == comm.dit_master_rank:
                    works = self._async_send_batch_to_non_dit(req, comm)
                    self.pending_sends.append((works, req))

    def _handle_error(self: "Scheduler", req, msg, identity):
        final_ident = identity or getattr(req, "client_identity", None)
        if final_ident:
            self.return_result({"status": "error", "message": msg}, final_ident)

    # --- Communication Helpers ---

    def _async_send_batch_to_dit(self, batch: Req, comm) -> List[Optional[Work]]:
        tensors, t_infos, l_infos = self._extract_tensors(batch)
        metadata = self._extract_metadata(batch, t_infos, l_infos)
        all_tensors = self._pack_tensors(metadata, t_infos, l_infos, tensors)
        return comm.batch_isend_to_dit(all_tensors)

    def _async_send_batch_to_non_dit(self, batch: Req, comm) -> List[Optional[Work]]:
        tensors, t_infos, l_infos = self._extract_tensors(batch)
        metadata = self._extract_metadata(batch, t_infos, l_infos)
        all_tensors = self._pack_tensors(metadata, t_infos, l_infos, tensors)
        return comm.batch_isend_to_non_dit(all_tensors)

    def _broadcast_batch_in_dit_group(self, batch, comm: DisaggCommunicator):
        from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj

        # Only master has the metadata initially
        if dist.get_rank() == comm.dit_master_rank:
            pkg = (
                batch._recv_metadata,
                batch._recv_tensor_infos,
                batch._recv_list_tensor_infos,
            )
        else:
            pkg = None

        group = comm.get_my_group()
        if not group or dist.get_world_size(group=group) <= 1:
            return

        # 1. Broadcast Meta
        pkg = broadcast_pyobj(
            pkg,
            dist.get_rank(),
            group,
            comm.dit_master_rank,
            force_cpu_device=(dist.get_backend(group) != "nccl"),
        )

        if dist.get_rank() != comm.dit_master_rank:
            meta, t_infos, l_infos = pkg
            for k, v in meta.items():
                setattr(batch, k, v)

            dev = get_local_torch_device()
            for name, (shape, dtype) in t_infos.items():
                setattr(batch, name, torch.empty(shape, dtype=dtype, device=dev))
            for name, (length, indices, shapes) in l_infos.items():
                lst = [None] * length
                for idx, (shape, dtype) in zip(indices, shapes):
                    lst[idx] = torch.empty(shape, dtype=dtype, device=dev)
                setattr(batch, name, lst)

        # 2. Broadcast Tensors
        # We walk the batch attributes to find what to broadcast
        for k, v in batch.__dict__.items():
            if k.startswith("_"):
                continue

            if isinstance(v, torch.Tensor):
                if not v.is_cuda:
                    v = v.cuda()
                comm.broadcast_in_group(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, torch.Tensor):
                        if not item.is_cuda:
                            item = item.cuda()
                        comm.broadcast_in_group(item)

    # --- Serialization Helpers ---

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
