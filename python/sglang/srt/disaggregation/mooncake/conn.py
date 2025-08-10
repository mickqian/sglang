from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import socket
import struct
import threading
import time
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
    TransferMMTokenizedData,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_auto,
    is_valid_ipv6_address,
    maybe_wrap_ipv6_address,
)

logger = logging.getLogger(__name__)

# Global store for MM metadata so different managers in the same process can access it
_MM_METADATA_STORE: Dict[int, TransferMMTokenizedData] = {}
_MM_METADATA_LOCK = threading.Lock()


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


@dataclasses.dataclass
class TransferEmbeddingChunk:
    room: int
    mm_indices: npt.NDArray[np.int32]


# decode
@dataclasses.dataclass
class TransferInfoCommon:
    room: Optional[int]
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: Optional[npt.NDArray[np.int32]]
    dst_aux_index: Optional[int]

    def to_bytes_list(self):
        return [
            (str(self.room) if self.room else "None").encode("ascii"),
            self.endpoint.encode("ascii"),
            str(self.dst_port).encode("ascii"),
            self.mooncake_session_id.encode("ascii"),
            self.dst_kv_indices if self.dst_kv_indices is not None else b"",
            self.dst_aux_index if self.dst_aux_index is not None else b"",
        ]

    @classmethod
    def from_zmq(cls, msg: List[bytes], dst_kv_indices, dst_aux_index):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
        )


# decode
@dataclasses.dataclass
class TransferInfo:
    common_info: TransferInfoCommon
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            is_dummy = False
        info_common = TransferInfoCommon.from_zmq(msg, dst_kv_indices, dst_aux_index)
        return cls(
            info_common,
            required_dst_info_num=int(msg[6].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    common_info: TransferInfoCommon
    # room: str
    # endpoint: str
    # dst_port: int
    # mooncake_session_id: str
    # dst_kv_ptrs: list[int]
    # dst_aux_ptrs: list[int]
    #
    #
    dst_tp_rank: int
    dst_tp_size: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # print(f"{msg[5]=}")
        common_info = TransferInfoCommon(
            room=msg[0].decode("ascii"),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_index=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
        )
        return cls(
            common_info,
            # room=str(msg[0].decode("ascii")),
            # endpoint=msg[1].decode("ascii"),
            # dst_port=int(msg[2].decode("ascii")),
            # mooncake_session_id=msg[3].decode("ascii"),
            # dst_kv_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            # dst_aux_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_tp_rank=int(msg[6].decode("ascii")),
            dst_tp_size=int(msg[7].decode("ascii")),
            dst_kv_item_len=int(msg[8].decode("ascii")),
        )


class MooncakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        # TODO: for mm manager, prefill is the receiver
        is_mm_embedding_manager: bool = False,
    ):
        self.kv_args = args
        # TODO: move to parent ?
        self.server_args = server_args
        self.local_ip = get_local_ip_auto()
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.init_engine()
        # for p/d multi node infer
        self.bootstrap_port = server_args.get_bootstrap_sending_port()
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        if is_valid_ipv6_address(self.local_ip):
            self.server_socket.setsockopt(zmq.IPV6, 1)
        self.is_mm_embedding_manager = is_mm_embedding_manager
        self.register_buffer_to_engine()
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and not is_mm_embedding_manager
        ) or self.disaggregation_mode == DisaggregationMode.ENCODE:
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.start_prefill_thread()
            else:
                # ENCODE delays registering until after sending MM metadata
                self.start_encode_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]

            print(f"213 {is_mm_embedding_manager=}")

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                for queue, executor in zip(self.transfer_queues, self.executors):
                    threading.Thread(
                        target=self.transfer_worker, args=(queue, executor), daemon=True
                    ).start()
            else:
                for queue, executor in zip(self.transfer_queues, self.executors):
                    threading.Thread(
                        target=self.transfer_embedding_worker,
                        args=(queue, executor),
                        daemon=True,
                    ).start()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300
            )
        elif (
            self.disaggregation_mode == DisaggregationMode.DECODE
            or self.disaggregation_mode == DisaggregationMode.TEXT
            or (
                self.disaggregation_mode == DisaggregationMode.PREFILL
                and is_mm_embedding_manager
            )
        ):
            print(f"237 {is_mm_embedding_manager=}")

            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()
            self.required_prefill_response_num_table: Dict[int, int] = {}
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            print(f"248 {self.disaggregation_mode=}")
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.start_decode_thread()
            elif self.disaggregation_mode == DisaggregationMode.TEXT:
                self.start_text_thread()
            else:
                self.start_prefill_thread()

            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.prefill_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def init_engine(self):
        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_args.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)

                status = self.engine.transfer_sync(
                    mooncake_session_id, src_addr, dst_addr, length
                )
                if status != 0:
                    return status
            return 0

        futures = [
            executor.submit(
                process_layer,
                src_ptr,
                dst_ptr,
                item_len,
            )
            for (src_ptr, dst_ptr, item_len) in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        dst_tp_rank: int,
        dst_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        local_tp_size = self.tp_size // self.dp_size
        local_tp_rank_in_group = self.kv_args.engine_rank % local_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_tp_size
        num_kv_heads = self.kv_args.kv_head_num
        num_layers = len(self.kv_args.kv_data_ptrs)
        page_size = self.kv_args.page_size

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * local_tp_size // dst_tp_size
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine slicing parameters based on TP configuration
        if local_tp_size > dst_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = dst_tp_rank_in_group * dst_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        layers_params = []
        for layer_id in range(num_layers):
            # Calculate precise byte offset and length for the sub-slice within the token
            src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
            dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
            heads_bytes_per_token_to_send = (
                num_heads_to_send * bytes_per_head_slice_to_send
            )

            # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
            # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
            if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
                logger.error(
                    f"[{mooncake_session_id}] Layer {layer_id}: "
                    f"slice size ({heads_bytes_per_token_to_send}) exceeds "
                    f"target token slot size ({dst_kv_item_len // page_size})"
                )
                return -1
            layers_params.append(
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    src_kv_item_len,
                    dst_kv_item_len,
                    src_head_slice_offset,
                    dst_head_slice_offset,
                    heads_bytes_per_token_to_send,
                )
            )

        def process_layer_tp_aware(layer_params):
            (
                src_ptr,
                dst_ptr,
                src_item_len,
                dst_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            ) = layer_params
            src_addr_list = []
            dst_addr_list = []
            length_list = []

            # Calculate strides for a single token slot
            bytes_per_token_on_prefill = src_item_len // page_size
            bytes_per_token_on_decode = dst_item_len // page_size

            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addr_list.append(src_slice_addr)
                    dst_addr_list.append(dst_slice_addr)
                    length_list.append(heads_bytes_per_token_to_send)

            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = [
            executor.submit(
                process_layer_tp_aware,
                layer_params,
            )
            for layer_params in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        src_addr_list = []
        dst_addr_list = []
        length_list = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens
        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addr_list.append(src_addr)
            dst_addr_list.append(dst_addr)
            length_list.append(length)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, src_addr_list, dst_addr_list, length_list
        )

    def send_embedding(
        self,
        session_id: str,
        mm_indices: npt.NDArray[np.int32],
        dst_mm_ptrs: List[int],
        dst_mm_indices: npt.NDArray[np.int32],
    ):
        encode_mm_blocks, dst_mm_blocks = group_concurrent_contiguous(
            mm_indices, dst_mm_indices
        )
        item_len = self.kv_args.kv_item_lens[0]

        for encode_index, dst_index in zip(encode_mm_blocks, dst_mm_blocks):
            src_addr = self.kv_args.kv_data_ptrs[0] + int(encode_index[0]) * item_len
            dst_addr = dst_mm_ptrs[0] + int(dst_index[0]) * item_len
            length = item_len * len(encode_index)
            logger.debug(
                f"{len(dst_mm_ptrs)=} {len(dst_mm_indices)=} {len(mm_indices)=}, {item_len=}"
            )
            status = self.engine.transfer_sync(session_id, src_addr, dst_addr, length)
            if status != 0:
                logger.error(
                    f"Embedding transfer failed: session_id={session_id}, status={status}"
                )
                break

        return status

    def send_mm_metadata(
        self,
        bootstrap_addr: str,
        bootstrap_room: int,
        data: TransferMMTokenizedData,
    ):
        """
        Send dynamic multimodal metadata (non-registered memory) from encoder to prefill.

        This runs before transfer_embedding and bootstrap. It broadcasts to all TP ranks
        within the target DP group on the prefill side via the prefill bootstrap server.
        """
        logger.debug("send_mm_metadata")
        try:
            # 1) fetch prefill parallel info
            # url = f"http://{bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            # response = requests.get(url, timeout=5)
            # if response.status_code != 200:
            #     logger.error(
            #         f"Failed to fetch prefill parallel info for MM metadata: {response.status_code}, {response.text}"
            #     )
            #     return

            # info = response.json()
            # prefill_tp_size = self.prefill_tp_size_table[bootstrap_addr]
            # prefill_dp_size = self.prefill_dp_size_table[bootstrap_addr]
            # if prefill_tp_size is None or prefill_dp_size is None or prefill_dp_size <= 0:
            #     logger.error("Invalid prefill parallel info for MM metadata")
            #     return

            reqs_to_be_processed = (
                self.transfer_infos[bootstrap_room].values()
                if bootstrap_room in self.transfer_infos
                else []
            )

            # print(f"{reqs_to_be_processed=}")
            # print(f"{self.transfer_infos=}")
            # print(f"{bootstrap_room=}")

            # 2) compute target dp group and per-dp tp size
            # target_dp_group = bootstrap_room % prefill_dp_size
            # tp_size_per_dp = prefill_tp_size // prefill_dp_size

            # 3) broadcast to all tp ranks in the target dp group
            bytes_list = data.get_bytes()
            for req in reqs_to_be_processed:
                # print(f"{req=}")
                self._connect(
                    format_tcp_address(
                        req.common_info.endpoint, req.common_info.dst_port
                    ),
                    is_ipv6=is_valid_ipv6_address(req.common_info.endpoint),
                ).send_multipart(
                    [
                        b"MM_META",
                        str(bootstrap_room).encode("ascii"),
                    ]
                    + bytes_list
                )
                # self._connect(req.endpoint, is_ipv6=is_ipv6).send_multipart(
                #     [
                #         b"MM_META",
                #         str(bootstrap_room).encode("ascii"),
                #         input_ids_bytes,
                #         mm_lens_bytes,
                #         mm_hashes_bytes,
                #     ]
                # )

            # for tp_rank_in_group in range(tp_size_per_dp):
            #     try:
            #         url = (
            #             f"http://{bootstrap_addr}/route?engine_rank={tp_rank_in_group}&target_dp_group={target_dp_group}"
            #         )
            #         r = requests.get(url, timeout=5)
            #         if r.status_code != 200:
            #             logger.error(
            #                 f"Failed to fetch prefill bootstrap info for DP {target_dp_group} TP {tp_rank_in_group}: {r.status_code}, {r.text}"
            #             )
            #             continue
            #         bootstrap_info = r.json()
            #         ip_address = bootstrap_info["rank_ip"]
            #         port = bootstrap_info["rank_port"]
            #         endpoint = format_tcp_address(ip_address, port)
            #         is_ipv6 = is_valid_ipv6_address(ip_address)
            #         # reuse cached socket
            #         self._connect(endpoint, is_ipv6=is_ipv6).send_multipart(
            #             [
            #                 b"MM_META",
            #                 str(bootstrap_room).encode("ascii"),
            #                 input_ids_bytes,
            #                 mm_lens_bytes,
            #                 mm_hashes_bytes,
            #             ]
            #         )
            #     except Exception as e:
            #         logger.error(f"Failed to send MM metadata to {target_dp_group=}, {tp_rank_in_group=}: {e}")
        except Exception as e:
            logger.error(f"send_mm_metadata failed: {e}")

    def sync_status_to_dst_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, src_rank: int
    ):
        logger.debug(f"sync_status_to_dst_endpoint with {status=}")
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(src_rank).encode("ascii"),
            ]
        )

    def transfer_embedding_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                embedding_chunk: TransferEmbeddingChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[embedding_chunk.room].values()
                    if embedding_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.kv_args.engine_rank
                for req in reqs_to_be_processed:
                    with self.session_lock:
                        if req.common_info.mooncake_session_id in self.failed_sessions:
                            self.record_failure(
                                embedding_chunk.room,
                                f"Decode instance could be dead, remote mooncake session {req.common_info.mooncake_session_id} is not alive",
                            )
                            self.update_status(embedding_chunk.room, KVPoll.Failed)
                            self.sync_status_to_dst_endpoint(
                                req.common_info.endpoint,
                                req.common_info.dst_port,
                                req.common_info.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                    dst_mm_indices = req.common_info.dst_kv_indices
                    print(f"{dst_mm_indices=}")
                    target_rank_registration_info: KVArgsRegisterInfo = (
                        self.decode_kv_args_table[req.common_info.mooncake_session_id]
                    )
                    logger.debug(f"{dst_mm_indices=} {embedding_chunk.mm_indices=}")

                    ret = self.send_embedding(
                        session_id=req.common_info.mooncake_session_id,
                        mm_indices=embedding_chunk.mm_indices,
                        dst_mm_ptrs=target_rank_registration_info.common_info.dst_kv_indices,
                        dst_mm_indices=dst_mm_indices,
                    )
                    if ret != 0:
                        with self.session_lock:
                            self.session_failures[
                                req.common_info.mooncake_session_id
                            ] += 1
                            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                            if (
                                self.session_failures[
                                    req.common_info.mooncake_session_id
                                ]
                                >= 1
                            ):
                                self.failed_sessions.add(
                                    req.common_info.mooncake_session_id
                                )
                                logger.error(
                                    f"Session {req.common_info.mooncake_session_id} failed."
                                )
                        self.record_failure(
                            embedding_chunk.room,
                            f"Failed to send kv chunk of {embedding_chunk.room} to {req.common_info.endpoint}:{req.common_info.dst_port}",
                        )
                        self.update_status(embedding_chunk.room, KVPoll.Failed)
                        self.sync_status_to_dst_endpoint(
                            req.common_info.endpoint,
                            req.common_info.dst_port,
                            req.common_info.room,
                            KVPoll.Failed,
                            local_rank,
                        )
                        break
                    polls.append(ret == 0)
                    dst_ranks_infos.append(
                        (
                            req.common_info.endpoint,
                            req.common_info.dst_port,
                            req.common_info.room,
                        )
                    )

                    # Only sync status when all the dst ranks have received the kvcache
                    if len(polls) == req.required_dst_info_num:
                        status = KVPoll.Success if all(polls) else KVPoll.Failed
                        self.update_status(req.common_info.room, status)
                        for endpoint, dst_port, room in dst_ranks_infos:
                            self.sync_status_to_dst_endpoint(
                                endpoint, dst_port, room, status, local_rank
                            )

            except Exception as e:
                # NOTE(yizhang2077): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.debug("transfer_worker kv_chunk")
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                logger.debug(f"transfer_worker {len(reqs_to_be_processed)=}")

                polls = []
                dst_ranks_infos = []
                local_rank = self.kv_args.engine_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if (
                                req.common_info.mooncake_session_id
                                in self.failed_sessions
                            ):
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.common_info.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_dst_endpoint(
                                    req.common_info.endpoint,
                                    req.common_info.dst_port,
                                    req.common_info.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.common_info.dst_kv_indices[
                            kv_chunk.index_slice
                        ]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[
                                req.common_info.mooncake_session_id
                            ]
                        )
                        local_tp_size = self.tp_size // self.dp_size
                        if self.is_mla_backend or (
                            local_tp_size == target_rank_registration_info.dst_tp_size
                        ):
                            ret = self.send_kvcache(
                                req.common_info.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.common_info.dst_kv_indices,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            ret = self.send_kvcache_slice(
                                req.common_info.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.common_info.dst_kv_indices,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        # print(f"{ret=}")
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[
                                    req.common_info.mooncake_session_id
                                ] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if (
                                    self.session_failures[
                                        req.common_info.mooncake_session_id
                                    ]
                                    >= 1
                                ):
                                    self.failed_sessions.add(
                                        req.common_info.mooncake_session_id
                                    )
                                    logger.error(
                                        f"Session {req.common_info.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.common_info.endpoint}:{req.common_info.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_dst_endpoint(
                                req.common_info.endpoint,
                                req.common_info.dst_port,
                                req.common_info.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break
                        logger.debug(f"{kv_chunk.is_last=}")
                        if kv_chunk.is_last:
                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
                                req.common_info.mooncake_session_id,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.common_info.dst_aux_index,
                                req.common_info.dst_aux_index,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (
                                    req.common_info.endpoint,
                                    req.common_info.dst_port,
                                    req.common_info.room,
                                )
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.common_info.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_dst_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if (
                            kv_chunk.is_last
                            and req.common_info.room in self.request_status
                        ):
                            self.update_status(req.common_info.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def _bind_server_socket(self):
        print(f"_bind_server_socket {self.local_ip=} {self.rank_port=}")
        self.server_socket.bind(format_tcp_address(self.local_ip, self.rank_port))

    def try_handle_mm_metadata(self, waiting_req_bytes) -> bool:
        logger.debug("check_if_mm_metadata")
        # Handle encoder->prefill MM metadata
        if (
            self.server_args.encoder_disaggregated
            and waiting_req_bytes
            and waiting_req_bytes[0] == b"MM_META"
        ):
            try:
                room = int(waiting_req_bytes[1].decode("ascii"))
                logger.debug(f"receiver received mm_metadata of room {room}")
                input_ids = np.frombuffer(waiting_req_bytes[2], dtype=np.int32)
                mm_embedding_lens = np.frombuffer(waiting_req_bytes[3], dtype=np.int32)
                mm_offsets = np.frombuffer(waiting_req_bytes[4], dtype=np.int32)

                mm_pad_values = np.frombuffer(waiting_req_bytes[5], dtype=np.int32)
                if len(waiting_req_bytes) > 6:
                    mrope_positions = np.frombuffer(
                        waiting_req_bytes[6], dtype=np.int32
                    )
                    mrope_positions_delta = np.frombuffer(
                        waiting_req_bytes[7], dtype=np.int32
                    )
                else:
                    mrope_positions = None
                    mrope_positions_delta = None
                with _MM_METADATA_LOCK:
                    _MM_METADATA_STORE[room] = TransferMMTokenizedData(
                        input_ids=input_ids,
                        mm_embedding_lens=mm_embedding_lens,
                        mm_offsets=mm_offsets,
                        mm_pad_values=mm_pad_values,
                        mrope_positions=mrope_positions,
                        mrope_positions_delta=mrope_positions_delta,
                    )
                return True
            except Exception as e:
                logger.error(f"Failed to parse MM_META: {e}")
        return False

    def start_text_thread(self):
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the prefill/text engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            print("start_text_thread")
            while True:
                data = self.server_socket.recv_multipart()
                # print(f"data received on text thread")
                if self.try_handle_mm_metadata(data):
                    continue

                self.try_handle_receiver_data(data)

                # assert len(data) == 3, len(data)
                # bootstrap_room = data[0]
                # status = data[1]
                # prefill_rank = data[2]
                # status = int(status.decode("ascii"))
                # bootstrap_room = int(bootstrap_room.decode("ascii"))
                # prefill_rank = int(prefill_rank.decode("ascii"))
                # logger.debug(
                #     f"text thread received waiting_req_bytes {bootstrap_room=}"
                # )
                # if status == KVPoll.Success:
                #     if bootstrap_room in self.request_status:
                #         self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                #         expected_response_num = (
                #             self.required_prefill_response_num_table[bootstrap_room]
                #         )
                #         arrived_response_num = len(
                #             self.prefill_response_tracker[bootstrap_room]
                #         )
                #         if (
                #             self.is_mla_backend
                #             or arrived_response_num == expected_response_num
                #         ):
                #             self.update_status(bootstrap_room, KVPoll.Success)
                # elif status == KVPoll.Failed:
                #     self.record_failure(
                #         bootstrap_room,
                #         f"Failed to get kvcache from prefill instance, it might be dead",
                #     )
                #     self.update_status(bootstrap_room, status)

        threading.Thread(target=bootstrap_thread).start()

    def try_handle_sender_data(self, data):
        logger.debug(f"try_handle_sender_data receiver received bootstrap info")
        room = data[0].decode("ascii")
        mooncake_session_id = data[3].decode("ascii")
        # print(f"{room=}")
        if room == "None":
            # KVArgsRegisterInfo
            self.decode_kv_args_table[mooncake_session_id] = (
                KVArgsRegisterInfo.from_zmq(data)
            )
            with self.session_lock:
                if mooncake_session_id in self.failed_sessions:
                    self.failed_sessions.remove(mooncake_session_id)
                if mooncake_session_id in self.session_failures:
                    del self.session_failures[mooncake_session_id]
            logger.debug(f"Register KVArgs from {mooncake_session_id} successfully")
            return
        else:
            required_dst_info_num = int(data[6].decode("ascii"))
            room = int(room)
            if room not in self.transfer_infos:
                self.transfer_infos[room] = {}

            self.transfer_infos[room][mooncake_session_id] = TransferInfo.from_zmq(data)
            # NOTE: in ep, only if the dst_indices is not none, bootstrapped is finished
            # NOTE: after bootstrapping we can mark the req as waiting for input
            if len(self.transfer_infos[room]) == required_dst_info_num:
                self.update_status(room, KVPoll.WaitingForInput)
        logger.debug(f"try_handle_sender_data receiver received bootstrap info handled")

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            print("start_prefill_thread")
            while True:
                data = self.server_socket.recv_multipart()

                if self.is_mm_embedding_manager:
                    if self.try_handle_mm_metadata(data):
                        continue
                    self.try_handle_receiver_data(data)
                else:
                    self.try_handle_sender_data(data)

        threading.Thread(target=bootstrap_thread).start()

    def start_encode_thread(self):
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the prefill/text engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                data = self.server_socket.recv_multipart()
                self.try_handle_sender_data(data)

        threading.Thread(target=bootstrap_thread).start()

    def try_handle_receiver_data(self, data):
        logger.debug("try_handle_receiver_data")
        assert len(data) == 3, len(data)
        bootstrap_room = data[0]
        status = data[1]
        prefill_rank = data[2]
        status = int(status.decode("ascii"))
        bootstrap_room = int(bootstrap_room.decode("ascii"))
        prefill_rank = int(prefill_rank.decode("ascii"))

        if status == KVPoll.Success:
            if bootstrap_room in self.request_status:
                self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                expected_response_num = self.required_prefill_response_num_table[
                    bootstrap_room
                ]
                arrived_response_num = len(
                    self.prefill_response_tracker[bootstrap_room]
                )
                if self.is_mla_backend or arrived_response_num == expected_response_num:
                    self.update_status(bootstrap_room, KVPoll.Success)
        elif status == KVPoll.Failed:
            self.record_failure(
                bootstrap_room,
                f"Failed to get kvcache from prefill instance, it might be dead",
            )
            self.update_status(bootstrap_room, status)

    def start_decode_thread(self):
        """
        Receives embedding from encoder (
        """
        self.rank_port = get_free_port()
        self._bind_server_socket()
        print("start_decode_thread")

        def decode_thread():
            while True:
                data = self.server_socket.recv_multipart()
                self.try_handle_receiver_data(data)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_embedding_request(
        self,
        bootstrap_room: int,
        mm_indices: npt.NDArray[np.int32],
    ):
        assert self.disaggregation_mode == DisaggregationMode.ENCODE
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferEmbeddingChunk(
                room=bootstrap_room,
                mm_indices=mm_indices,
            )
        )

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        logger.debug(f"update status of {bootstrap_room=} with {status=}")
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    # -------- MM metadata helpers ---------
    def get_mm_metadata(self, bootstrap_room: int) -> Optional[TransferMMTokenizedData]:
        with _MM_METADATA_LOCK:
            return _MM_METADATA_STORE.get(bootstrap_room)

    def clear_mm_metadata(self, bootstrap_room: int) -> None:
        with _MM_METADATA_LOCK:
            if bootstrap_room in _MM_METADATA_STORE:
                del _MM_METADATA_STORE[bootstrap_room]

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        # print(f"_register_to_bootstrap")
        if self.dist_init_addr:
            if self.dist_init_addr.startswith("["):  # [ipv6]:port or [ipv6]
                if self.dist_init_addr.endswith("]"):
                    host = self.dist_init_addr
                else:
                    host, _ = self.dist_init_addr.rsplit(":", 1)
            else:
                host = socket.gethostbyname(self.dist_init_addr.rsplit(":", 1)[0])
        else:
            host = get_ip()
            host = maybe_wrap_ipv6_address(host)

        bootstrap_server_url = f"{host}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        role_str = self.disaggregation_mode.role_str
        payload = {
            "role": role_str,
            "dp_size": self.dp_size,
        }
        payload.update(
            {
                "tp_size": self.tp_size,
                "dp_size": self.dp_size,
                "rank_ip": self.local_ip,
                "rank_port": self.rank_port,
                "engine_rank": self.kv_args.engine_rank,
            }
        )

        try:
            # print(f"registering to bootstrap with: {url=}")
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug(f"{role_str} successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"{role_str} instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"{role_str} instance failed to register to bootstrap server: {e}"
            )

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_tp_size_table:
                del self.prefill_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class MooncakeKVSender(BaseKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int] = None,
        pp_rank: int = None,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        logger.debug(f"sender {bootstrap_room=}")
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = time.time()
        # inner state
        self.curr_idx = 0

    def init(self, num_kv_indices: Optional[int], aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        logger.debug(f"{self.num_kv_indices=} {self.bootstrap_room=}")
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        logger.debug(f"{len(kv_indices)=}")
        logger.debug(f"{self.curr_idx=}")
        logger.debug(f"{self.num_kv_indices=}")
        logger.debug(f"Sender sending kvcache {is_last=}")

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            # print(f"checked status of {self.bootstrap_room=} with result {status=}, {self.kv_mgr.request_status=}")
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def send_embedding(self, mm_indices: npt.NDArray[np.int32]):
        """
        Send the embedding tensor to the remote server using MooncakeKVManager.
        """
        self.kv_mgr.add_transfer_embedding_request(
            self.bootstrap_room,
            mm_indices,
        )

    def send_mm_metadata(self, data: TransferMMTokenizedData):
        """Send multimodal metadata prior to bootstrap and embedding transfer."""
        logger.debug(f"sending mm metadata, {data=}")

        self.kv_mgr.send_mm_metadata(
            bootstrap_addr=self.bootstrap_server_url,
            bootstrap_room=self.bootstrap_room,
            data=data,
        )


class MooncakeKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        disaggregation_mode: DisaggregationMode,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.init_time = None
        self.data_parallel_rank = data_parallel_rank
        self.disaggregation_mode = disaggregation_mode

        # if self.disaggregation_mode == DisaggregationMode.PREFILL:
        #     # FIXME
        #     self.target_dp_group = 0
        #     self.target_tp_rank = -1
        #     self.target_tp_ranks = [0]
        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = (
                self._get_prefill_parallel_info_from_server()
            )

            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self.bootstrap_infos = []

                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
                self.kv_mgr.prefill_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
        else:
            self.prefill_tp_size = self.kv_mgr.prefill_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]

        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size
        if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.required_prefill_response_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.required_prefill_response_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1
            self.required_prefill_response_num = (
                prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank
            )

        if self.data_parallel_rank is not None:
            logger.debug(f"Targeting DP rank: {self.data_parallel_rank}")
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.prefill_dp_size

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )
        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    if self.kv_mgr.is_mla_backend:
                        # For MLA: target_tp_rank is the selected real rank, others are dummy ranks
                        bootstrap_info["is_dummy"] = not bool(
                            target_tp_rank == self.target_tp_rank
                            or self.target_tp_rank is None
                        )
                    else:
                        # For non-MLA: all target_tp_ranks are selected real ranks
                        bootstrap_info["is_dummy"] = False
                    logger.debug(
                        f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank}"
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

            self.bootstrap_infos = bootstrap_infos
            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
            self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]
        # print(f"{self.bootstrap_infos=}")
        assert len(self.bootstrap_infos) > 0
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                # print(f"{bootstrap_info=}")
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_prefill_parallel_info_from_server(
        self,
    ) -> tuple[int, int] | tuple[None, None]:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"

            response = requests.get(url)
            logger.debug(f"{url=} {response.json()=}")
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return int(prefill_parallel_info["prefill_tp_size"]), int(
                    prefill_parallel_info["prefill_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None

    def _register_kv_args(self):
        tp_rank = self.kv_mgr.kv_args.engine_rank
        tp_size = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        dst_tp_rank = str(tp_rank).encode("ascii")
        dst_tp_size = str(tp_size).encode("ascii")
        logger.debug(f"receiver send_multipart _register_kv_args")

        for bootstrap_info in self.bootstrap_infos:
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                packed_kv_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
                )
                packed_aux_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
                )
                kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
                dst_kv_item_len = str(kv_item_len).encode("ascii")
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                transfer_info = TransferInfoCommon(
                    room=None,
                    endpoint=self.kv_mgr.local_ip,
                    dst_port=self.kv_mgr.rank_port,
                    mooncake_session_id=self.session_id,
                    dst_kv_indices=packed_kv_data_ptrs,
                    dst_aux_index=packed_aux_data_ptrs,
                )
                # KVArgsRegisterInfo
                with lock:
                    sock.send_multipart(
                        transfer_info.to_bytes_list()
                        + [
                            dst_tp_rank,
                            dst_tp_size,
                            dst_kv_item_len,
                        ]
                    )
            elif (
                self.disaggregation_mode == DisaggregationMode.PREFILL
                or self.disaggregation_mode == DisaggregationMode.TEXT
            ):
                print(f"{self.kv_mgr.kv_args.kv_data_ptrs=}")
                packed_kv_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
                )
                # print(f"{len(self.kv_mgr.kv_args.kv_data_ptrs)=}")
                # packed_aux_data_ptrs = b"".join(
                #     struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
                # )
                kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
                # print(f"{self.kv_mgr.kv_args.kv_data_ptrs=}, {kv_item_len=}")
                dst_kv_item_len = str(kv_item_len).encode("ascii")
                logger.debug(
                    f"receiver _register_kv_args, _register_kv_args with {bootstrap_info=} "
                )
                logger.debug(f"{self.kv_mgr.local_ip=}")
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                transfer_info = TransferInfoCommon(
                    room=None,
                    endpoint=self.kv_mgr.local_ip,
                    dst_port=self.kv_mgr.rank_port,
                    mooncake_session_id=self.session_id,
                    dst_kv_indices=packed_kv_data_ptrs,
                    dst_aux_index=b"",
                )
                # print(b"")
                # print(f"{transfer_info.to_bytes_list()[-1]=}")
                logger.debug("send_multipart registering kv args")
                sock.send_multipart(
                    transfer_info.to_bytes_list()
                    + [
                        dst_tp_rank,
                        dst_tp_size,
                        dst_kv_item_len,
                    ]
                )

    @classmethod
    def _connect(cls, endpoint: str, is_ipv6: bool = False):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        ip_address = bootstrap_info["rank_ip"]
        port = bootstrap_info["rank_port"]
        is_ipv6_address = is_valid_ipv6_address(ip_address)
        sock, lock = cls._connect(
            format_tcp_address(ip_address, port), is_ipv6=is_ipv6_address
        )
        return sock, lock

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        logger.debug(f"receiver send_multipart initializing... ")

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"kv receiver init sending: {bootstrap_info=} {self.bootstrap_room=} "
            )
            if kv_indices is None:
                is_dummy = True
                # logger.debug(
                #     f"kv receiver init sending: {kv_indices=}"
                #     f"{len(kv_indices.tobytes())=}"
                #     f"{type(kv_indices)=}"
                # )
                # logger.debug(f"receiver initializing, registering with {bootstrap_info=} ")
                # kv_indices = kv_indices.tobytes() if not is_dummy else b"",
            else:
                ...
                # kv_indices = b""

            with lock:
                transfer_info = TransferInfoCommon(
                    room=self.bootstrap_room,
                    endpoint=self.kv_mgr.local_ip,
                    dst_port=self.kv_mgr.rank_port,
                    mooncake_session_id=self.session_id,
                    dst_kv_indices=kv_indices.tobytes() if not is_dummy else None,
                    dst_aux_index=(
                        str(aux_index).encode("ascii") if not is_dummy else None
                    ),
                )
                assert [
                    str(self.bootstrap_room).encode("ascii"),
                    self.kv_mgr.local_ip.encode("ascii"),
                    str(self.kv_mgr.rank_port).encode("ascii"),
                    self.session_id.encode("ascii"),
                    kv_indices.tobytes() if not is_dummy else b"",
                    str(aux_index).encode("ascii") if not is_dummy else b"",
                ] == transfer_info.to_bytes_list()
                # print(f"{transfer_info.to_bytes_list()[-1]=}")
                sock.send_multipart(
                    transfer_info.to_bytes_list()
                    + [
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status

        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)


class MooncakeKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()
        print(f"bootstrap server started at: {port}")

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        # print(f"handle route: {request=}")
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        # print(f"route put {data=}")
        if role == DisaggregationMode.PREFILL.role_str:
            tp_size = data["tp_size"]
            dp_size = data["dp_size"]
            rank_ip = data["rank_ip"]
            rank_port = int(data["rank_port"])
            engine_rank = int(data["engine_rank"])

            if self.tp_size is None:
                self.tp_size = tp_size

            if self.dp_size is None:
                self.dp_size = dp_size

            tp_size_per_dp_rank = tp_size // dp_size
            if self.tp_size_per_dp_rank is None:
                self.tp_size_per_dp_rank = tp_size_per_dp_rank

            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.port_table:
                    self.port_table[dp_group] = {}

            self.port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )
        elif role == DisaggregationMode.ENCODE.role_str:
            tp_size = data["tp_size"]
            dp_size = data["dp_size"]
            rank_ip = data["rank_ip"]
            rank_port = int(data["rank_port"])
            engine_rank = int(data["engine_rank"])

            if self.tp_size is None:
                self.tp_size = tp_size

            if self.dp_size is None:
                self.dp_size = dp_size

            tp_size_per_dp_rank = tp_size // dp_size
            if self.tp_size_per_dp_rank is None:
                self.tp_size_per_dp_rank = tp_size_per_dp_rank

            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.port_table:
                    self.port_table[dp_group] = {}

            self.port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        # print(f"{engine_rank=}")
        # print(f"{target_dp_group=}")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.tp_size,
                "prefill_dp_size": self.dp_size,
            }
            # print(f"responding get with: {prefill_parallel_info=}")
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.port_table[int(target_dp_group)][int(engine_rank)]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
