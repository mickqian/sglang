from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class EmbeddingArgs:
    pass


@dataclasses.dataclass
class TransferMMTokenizedData:
    input_ids: npt.NDArray[np.int32]
    mm_embedding_lens: npt.NDArray[np.int32]
    mm_offsets: npt.NDArray[np.int32]
    mm_pad_values: npt.NDArray[np.int32]
    mrope_positions: npt.NDArray[np.int32]
    mrope_positions_delta: npt.NDArray[np.int32]

    @staticmethod
    def from_req(req):
        input_ids = np.asarray(getattr(req, "origin_input_ids", []), dtype=np.int32)
        mm_embedding_lens = np.asarray(
            getattr(req, "mm_embedding_lens", []), dtype=np.int32
        )
        mm_offsets = np.asarray(getattr(req, "mm_offsets", []), dtype=np.int32)
        # print(f"{mm_offsets.tolist()=}")
        mm_pad_values = np.asarray(
            getattr(req, "mm_pad_values", []),
            dtype=np.int32,
        )
        # print(f"195 {req.multimodal_inputs.mrope_positions.shape=}")
        mrope_positions = np.asarray(
            getattr(req.multimodal_inputs, "mrope_positions", []), dtype=np.int32
        )
        mrope_positions_delta = np.asarray(
            getattr(req.multimodal_inputs, "mrope_position_delta", []),
            dtype=np.int32,
        )
        # print(f"{mrope_positions_delta=}")

        return TransferMMTokenizedData(
            input_ids,
            mm_embedding_lens,
            mm_offsets,
            mm_pad_values,
            mrope_positions,
            mrope_positions_delta,
        )

    def get_bytes(self):
        input_ids_bytes = (
            self.input_ids.astype(np.int32, copy=False).tobytes()
            if self.input_ids is not None
            else b""
        )
        mm_lens_bytes = (
            self.mm_embedding_lens.astype(np.int32, copy=False).tobytes()
            if self.mm_embedding_lens is not None
            else b""
        )
        mm_offsets_bytes = (
            self.mm_offsets.astype(np.int32, copy=False).tobytes()
            if self.mm_offsets is not None
            else b""
        )

        mm_hashes_bytes = (
            self.mm_pad_values.astype(np.int32, copy=False).tobytes()
            if self.mm_pad_values is not None
            else b""
        )
        mrope_positions_bytes = (
            self.mrope_positions.astype(np.int32, copy=False).tobytes()
            if self.mrope_positions is not None
            else b""
        )
        mrope_positions_delta_bytes = (
            self.mrope_positions_delta.astype(np.int32, copy=False).tobytes()
            if self.mrope_positions_delta is not None
            else b""
        )

        return [
            input_ids_bytes,
            mm_lens_bytes,
            mm_offsets_bytes,
            mm_hashes_bytes,
            mrope_positions_bytes,
            mrope_positions_delta_bytes,
        ]


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    # for different tp
    decode_tp_size: int
    # for pp prefill
    prefill_pp_size: int
    kv_head_num: int
    page_size: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    # waiting for sender-side to finish computing
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfers states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    def get_mm_metadata(self, bootstrap_room: int) -> Optional[TransferMMTokenizedData]:
        pass

    def clear_mm_metadata(self, bootstrap_room: int) -> None:
        pass


class BaseMetadataSender(ABC):
    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ): ...


class ZMQMetadataSender(ABC):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ): ...


class ZMQMetadataReceiver(ABC):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ): ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Notify the decoder server about the kv indices length and aux index
        """
        ...

    @abstractmethod
    def send(self, kv_indices: npt.NDArray[np.int32]):
        """
        Send the kv cache at the given kv indices to the decoder server
        """
        ...

    def send_embedding(self, mm_indices: npt.NDArray[np.int32]):
        """
        Send the concatenated embeddings with each embedding's start token indices
        """
        pass

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...

    def send_mm_metadata(self, data: TransferMMTokenizedData):
        pass


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        """
        Notify the prefill server about the kv indices and aux index
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, port: int): ...
