"""
Life cycle of a request in the decode server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.base import BaseKVManager, KVPoll
from sglang.srt.disaggregation.base.conn import EmbeddingArgs
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    EncoderMetadataBuffers,
    KVClassType,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import require_mlp_sync

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


class EncodeBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        # mm_embedding_pool: PagedMultiModalEmbeddingPool,
        # draft_mm_embedding_pool: Optional[PagedMultiModalCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: EncoderMetadataBuffers,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        encode_dp_size: int,
        scheduler: Scheduler,
        transfer_backend: TransferBackend,
    ):
        # self.mm_embedding_pool = mm_embedding_pool
        # self.draft_mm_embedding_pool = draft_mm_embedding_pool
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.encode_dp_size = encode_dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> BaseKVManager:
        # TODO
        embedding_class = EmbeddingArgs()
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        # kv_data_ptrs, kv_data_lens, kv_item_lens = (
        #     self.mm_embedding_pool.get_pointers_from_locs()
        # )
        #
        #
        # kv_args.kv_data_ptrs = kv_data_ptrs
        # kv_args.kv_data_lens = kv_data_lens
        # kv_args.kv_item_lens = kv_item_lens

        # kv_args.page_size = self.mm_embedding_pool.page_size
        #
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        # kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        # kv_args.gpu_id = self.scheduler.gpu_id
        #
        # kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        # kv_manager = kv_manager_class(
        #     kv_args,
        #     DisaggregationMode.PREFILL,
        #     self.scheduler.server_args,
        # )
        # return kv_manager

    def add(self, req: Req, num_kv_heads: int) -> None:
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)

        # TODO: embedding sender class
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            # self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so EncodeAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if not return_failed_reqs:
                return []
            else:
                return [], []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue], self.gloo_group
        )
        for i, (req, poll) in enumerate(zip(self.queue, polls)):

            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue
                # Either waiting for input or failed
                assert poll == KVPoll.WaitingForInput or poll == KVPoll.Failed

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Encode bootstrap failed for request rank={req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                continue

            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.mm_embedding_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationEncodeMixin:
    """
    Mixin for Scheduler to handle disaggregation encode
    """

    @torch.no_grad()
    def event_loop_normal_disagg_encode(self: Scheduler) -> None:
        """A normal scheduler loop for encode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_encode_bootstrap_queue.pop_bootstrapped()
            )
            batch = self.get_new_batch_encode()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_encode(batch, result)

            # self.process_disagg_encode_inflight_queue()

            if batch is None and len(self.disagg_encode_inflight_queue) == 0:
                # self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_encode(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_encode_bootstrap_queue.pop_bootstrapped()
            )
            # self.process_prefill_chunk()
            batch = self.get_new_batch_encode()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.set_next_batch_sampling_info_done(tmp_batch)

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result_disagg_encode(tmp_batch, tmp_result)

            if len(self.disagg_encode_inflight_queue) > 0:
                self.process_disagg_encode_inflight_queue()

            if batch is None and len(self.disagg_encode_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_encode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            _next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        # if self.enable_overlap:
        #     raise NotImplementedError()
        # wait
        # logits_output, next_token_ids, _ = self.tp_worker.resolve_last_batch_result(
        #     launch_done
        # )
        # else:
        # _next_token_ids = result.next_token_ids.tolist()
        # if batch.return_logprob:
        #     if logits_output.next_token_logprobs is not None:
        #         logits_output.next_token_logprobs = (
        #             logits_output.next_token_logprobs.tolist()
        #         )
        #     if logits_output.input_token_logprobs is not None:
        #         logits_output.input_token_logprobs = tuple(
        #             logits_output.input_token_logprobs.tolist()
        #         )

        hidden_state_offset = 0
        # for i, (req, next_token_id) in enumerate(
        #     zip(batch.reqs, _next_token_ids, strict=True)
        # ):
        #     req: Req
        #     assert req.is_chunked <= 0
        #     # There is no output_ids for prefill
        #     req.output_ids.append(next_token_id)
        #     self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
        #     self.disagg_encode_inflight_queue.append(req)
        #     if logits_output.hidden_states is not None:
        #         last_hidden_index = (
        #             hidden_state_offset + extend_input_len_per_req[i] - 1
        #         )
        #         req.hidden_states_tensor = (
        #             logits_output.hidden_states[last_hidden_index].cpu().clone()
        #         )
        #         hidden_state_offset += extend_input_len_per_req[i]
        #     else:
        #         req.hidden_states_tensor = None
        #     if req.return_logprob:
        #         assert extend_logprob_start_len_per_req is not None
        #         assert extend_input_len_per_req is not None
        #         extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        #         extend_input_len = extend_input_len_per_req[i]
        #         num_input_logprobs = extend_input_len - extend_logprob_start_len
        #         self.add_logprob_return_values(
        #             i,
        #             req,
        #             logprob_pt,
        #             _next_token_ids,
        #             num_input_logprobs,
        #             logits_output,
        #         )
        #         logprob_pt += num_input_logprobs
        # FIXME: manually set finish reason to let req response
        for i, req in enumerate(batch.reqs):
            req.finished_reason = FINISH_LENGTH(length=0)

        self.send_embedding_chunk(result)
        # We need to remove the sync in the following function for overlap schedule.
        self.set_next_batch_sampling_info_done(batch)
        # print(f"{batch.reqs=}")
        self.stream_output(batch.reqs, batch.return_logprob)

    def process_disagg_encode_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_encode_inflight_queue) == 0:
            return []
        # print(f"process_disagg_encode_inflight_queue")

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_encode_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_encode_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                assert poll == KVPoll.Success or poll == KVPoll.Failed

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"Encode transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
            else:
                assert False, f"Unexpected polling state {poll=}"

        # Stream requests which have finished transfer
        print(f"streaming requests out")
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

        self.disagg_encode_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_encode_inflight_queue],
            self.tp_worker.get_tp_group().cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_encode_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def send_embedding_chunk(
        self: Scheduler,
        result: GenerationBatchResult,
        req: Req = None,
    ) -> None:
        """
        Send a embedding to the prefill server
        """
        s = socket.socket()
        ip = "127.0.0.1"
        port = 60003
        print(f"connecting...")
        s.connect((ip, port))
        print(f"connected, sending...")

        embeddings: torch.Tensor = result.logits_output

        shape = embeddings.shape
        s.sendall(struct.pack("2I", *shape))  # 2个无符号int
        print(f"{embeddings.shape=}")

        data = embeddings.to(torch.float32).cpu().numpy().tobytes()
        s.sendall(data)
        print(f"sent")
        s.close()
        # print(f"{req=}")
        # req.disagg_kv_sender.send_embedding(embeddings, [0])
