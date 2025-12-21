"""
PyTorch Implementation of DisaggCommunicator.
"""

import io
import pickle
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

from sglang.multimodal_gen.runtime.distributed.disagg_communicators.base_communicator import (
    DisaggCommunicator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PyTorchDisaggCommunicator(DisaggCommunicator):
    def __init__(self):
        self.non_dit_group = None
        self.dit_group = None
        self.p2p_group = None  # Dedicated P2P group for master-to-master communication
        self.broadcast_group = None
        self.non_dit_ranks = []
        self.dit_ranks = []
        self.role = None  # "non_dit" or "dit"
        self.group_rank = -1
        self.world_rank = -1

        # We assume rank 0 of Non-DiT group communicates with rank 0 of DiT group.
        # These are GLOBAL ranks.
        self.non_dit_master_rank = -1
        self.dit_master_rank = -1

    def initialize_topology(self, server_args: Any) -> None:
        self.world_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # --- Logic from composed_pipeline_base.py (Refactored) ---
        # "do_disaggregation = server_args.num_gpus > 1 and (server_args.num_gpus - 1) % 2 == 0"
        # We should probably make this more explicit in server_args later.
        # For now, let's assume the same logic:
        # Rank 0 is Non-DiT
        # Ranks 1..N are DiT

        non_dit_ranks = list(
            range(world_size - server_args.num_non_dit_ranks, world_size)
        )
        dit_ranks = [r for r in range(world_size) if r not in non_dit_ranks]

        self.non_dit_ranks = non_dit_ranks
        self.dit_ranks = dit_ranks
        self.non_dit_master_rank = non_dit_ranks[0]
        self.dit_master_rank = dit_ranks[0]

        logger.info(
            f"Initializing Disagg Topology: Non-DiT Ranks={non_dit_ranks}, DiT Ranks={dit_ranks}"
        )

        # Create groups
        # Note: new_group requires all processes to call it in the same order
        self.non_dit_group = dist.new_group(ranks=non_dit_ranks)
        self.dit_group = dist.new_group(ranks=dit_ranks)

        # Create dedicated P2P group for master-to-master communication
        # This avoids serialization warnings on the default ProcessGroup
        p2p_ranks = [self.dit_master_rank, self.non_dit_master_rank]
        self.p2p_group = dist.new_group(ranks=p2p_ranks)
        logger.info(
            f"Created P2P group for ranks {p2p_ranks} "
            f"(DiT master={self.dit_master_rank}, Non-DiT master={self.non_dit_master_rank})"
        )

        # Create broadcast group for Non-DiT Master -> All DiT Ranks
        # This includes Non-DiT Master + All DiT Ranks (Master + Workers)
        broadcast_ranks = sorted(list(set([self.non_dit_master_rank] + dit_ranks)))
        self.broadcast_group = dist.new_group(ranks=broadcast_ranks)
        logger.info(
            f"Created Broadcast group for ranks {broadcast_ranks} "
            f"(Source=Non-DiT master {self.non_dit_master_rank})"
        )

        if self.world_rank in non_dit_ranks:
            self.role = "non_dit"
            self.group_rank = non_dit_ranks.index(self.world_rank)
        elif self.world_rank in dit_ranks:
            self.role = "dit"
            self.group_rank = dit_ranks.index(self.world_rank)
        else:
            raise ValueError(f"Rank {self.world_rank} not assigned to any group!")

    def get_my_group(self) -> Optional[dist.ProcessGroup]:
        if self.role == "non_dit":
            return self.non_dit_group
        return self.dit_group

    def is_dit_rank(self) -> bool:
        return self.role == "dit"

    def is_non_dit_rank(self) -> bool:
        return self.role == "non_dit"

    def send_to_dit(
        self, tensor: torch.Tensor, metadata: Optional[Dict] = None
    ) -> None:
        """Called by Non-DiT Master (Rank 0) to send to DiT Master."""
        if self.world_rank != self.non_dit_master_rank:
            return  # Only master sends cross-group

        # P2P Send to DiT Master using dedicated P2P group
        dist.send(tensor, dst=self.dit_master_rank, group=self.p2p_group)

    def recv_from_non_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Called by DiT Ranks.
        DiT Master receives from Non-DiT, then broadcasts to other DiT ranks.
        """
        tensor = torch.empty(shape, dtype=dtype, device="cuda")  # Todo: proper device

        if self.world_rank == self.dit_master_rank:
            dist.recv(tensor, src=self.non_dit_master_rank, group=self.p2p_group)

        # Broadcast within DiT group so all workers have the input
        self.broadcast_in_group(tensor)
        return tensor

    def send_to_non_dit(self, tensor: torch.Tensor) -> None:
        """Called by DiT Master to send result back to Non-DiT."""
        if self.world_rank != self.dit_master_rank:
            return

        dist.send(tensor, dst=self.non_dit_master_rank, group=self.p2p_group)

    def recv_from_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Called by Non-DiT Master to receive result."""
        tensor = torch.empty(shape, dtype=dtype, device="cuda")

        if self.world_rank == self.non_dit_master_rank:
            dist.recv(tensor, src=self.dit_master_rank, group=self.p2p_group)

        return tensor

    def broadcast_in_group(
        self, tensor: torch.Tensor, src_rank_in_group: int = 0
    ) -> None:
        """
        Wraps dist.broadcast using the current role's group.
        src_rank_in_group is relative to the group.
        """
        group = self.get_my_group()
        if group is None:
            return

        # Check group size - if only 1 member, no need to broadcast
        group_size = dist.get_world_size(group=group)
        if group_size == 1:
            return

        # CRITICAL: Ensure tensor is on CUDA for NCCL backend
        backend = dist.get_backend(group)
        if backend == "nccl" and not tensor.is_cuda:
            raise RuntimeError(
                f"[Rank {dist.get_rank()}] Cannot broadcast CPU tensor with NCCL backend. "
                f"Tensor device: {tensor.device}, backend: {backend}, "
                f"tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
            )

        # We need to translate group-relative src rank to global rank for dist.broadcast
        # Wait, dist.broadcast(group=group) usually expects the `src` to be the GLOBAL rank
        # of the broadcaster.

        if self.role == "dit":
            # Assuming linear mapping for now or finding from stored list
            # Simpler: just assume src is always 0 (master) of that group
            # We need to find the global rank of the group master
            global_src = self.dit_master_rank  # Logic for src=0
        else:
            global_src = self.non_dit_master_rank

        try:
            dist.broadcast(tensor, src=global_src, group=group)
        except Exception as e:
            logger.error(
                f"[Rank {dist.get_rank()}] broadcast failed: tensor device={tensor.device}, "
                f"backend={backend}, error={e}"
            )
            raise

    # --- Async Communication Implementation ---

    def batch_isend_to_dit(self, tensors: List[torch.Tensor]) -> List[Work]:
        """
        Batched non-blocking send from Non-DiT to DiT group.
        Uses torch.distributed.batch_isend_irecv to avoid serialization.

        Returns:
            List of Work handles (empty if not master)
        """
        if self.world_rank != self.non_dit_master_rank:
            return []  # Only master sends cross-group

        # Create P2P operation list
        works = []
        for tensor in tensors:
            work = dist.isend(tensor, dst=self.dit_master_rank, group=self.p2p_group)
            works.append(work)

        return works

    def batch_irecv_from_non_dit(
        self, shapes_dtypes: List[tuple[torch.Size, torch.dtype]]
    ) -> tuple[List[torch.Tensor], List[Work]]:
        """
        Batched non-blocking receive from Non-DiT group at DiT group.

        Args:
            shapes_dtypes: List of (shape, dtype) tuples for tensors to receive

        Returns:
            (list of tensors, list of Work handles)
        """
        tensors = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in shapes_dtypes
        ]

        works = []
        if self.world_rank == self.dit_master_rank:
            # Create P2P operation list
            for tensor in tensors:
                work = dist.irecv(
                    tensor, src=self.non_dit_master_rank, group=self.p2p_group
                )
                works.append(work)

        return tensors, works

    def batch_isend_to_non_dit(self, tensors: List[torch.Tensor]) -> List[Work]:
        """Batched non-blocking send from DiT to Non-DiT group."""
        if self.world_rank != self.dit_master_rank:
            return []

        works = []
        for tensor in tensors:
            work = dist.isend(
                tensor, dst=self.non_dit_master_rank, group=self.p2p_group
            )
            works.append(work)

        return works

    def batch_irecv_from_dit(
        self, shapes_dtypes: List[tuple[torch.Size, torch.dtype]]
    ) -> tuple[List[torch.Tensor], List[Work]]:
        """Batched non-blocking receive from DiT group at Non-DiT group."""
        tensors = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in shapes_dtypes
        ]

        works = []
        if self.world_rank == self.non_dit_master_rank:
            for tensor in tensors:
                work = dist.irecv(
                    tensor, src=self.dit_master_rank, group=self.p2p_group
                )
                works.append(work)

        return tensors, works

    # Legacy single-tensor methods (kept for backward compatibility, but should use batched versions)
    def isend_to_dit(
        self, tensor: torch.Tensor, metadata: Optional[Dict] = None
    ) -> Optional[Work | List[Work]]:
        """Send a signal to all DiT ranks from Non-DiT Master."""
        if self.world_rank != self.non_dit_master_rank:
            return None
        
        # Use P2P isend to each DiT rank instead of broadcast.
        # This avoids deadlocks when DiT ranks are waiting but Non-DiT is not ready.
        works = []
        for rank in self.dit_ranks:
            works.append(dist.isend(tensor, dst=rank))
        return works

    def irecv_from_non_dit(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Optional[Work]]:
        """Join the asynchronous signal from Non-DiT Master."""
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        
        # Use P2P irecv from Non-DiT master
        work = dist.irecv(tensor, src=self.non_dit_master_rank)
        return tensor, work

    def isend_to_non_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """Non-blocking send from DiT to Non-DiT group."""
        works = self.batch_isend_to_non_dit([tensor])
        return works[0] if works else None

    def irecv_from_dit(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Optional[Work]]:
        """Non-blocking receive from DiT group at Non-DiT group."""
        tensors, works = self.batch_irecv_from_dit([(shape, dtype)])
        return tensors[0], works[0] if works else None

    def wait_work(self, work: Optional[Work]) -> None:
        """Wait for a Work handle to complete."""
        if work is not None:
            work.wait()

    def wait_all_works(self, works: List[Optional[Work]]) -> None:
        """Wait for multiple Work handles to complete."""
        for work in works:
            if work is not None:
                work.wait()

    def irecv_size_from_non_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """DiT Master listens for signal from Non-DiT Master (P2P)."""
        if self.world_rank == self.dit_master_rank:
            return dist.irecv(tensor, src=self.non_dit_master_rank, group=self.p2p_group)
        return None

    def isend_signal_to_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """Non-DiT Master sends signal to DiT Master (P2P)."""
        if self.world_rank == self.non_dit_master_rank:
            return dist.isend(tensor, dst=self.dit_master_rank, group=self.p2p_group)
        return None

    def broadcast_object_from_non_dit(self, obj: Optional[Any] = None) -> Any:
        """
        Broadcast a complex object (e.g. Req) with tensors from Non-DiT Master to all DiT ranks.
        Optimized for NVLink: separates metadata and tensors.
        """
        # If I am not part of the broadcast group (e.g. Non-DiT Worker), return None
        if self.broadcast_group is None:
            return None

        is_sender = self.world_rank == self.non_dit_master_rank

        # 1. Prepare Metadata and Tensors (Sender)
        meta_tensor = None
        size_tensor = None
        tensor_list = []

        if is_sender:
            assert obj is not None, "Sender must provide an object to broadcast"
            meta_dict = {"__class__": obj.__class__}
            tensor_list = []

            # Extract state
            # We assume the object has a __dict__
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = ("tensor", v.shape, v.dtype)
                        tensor_list.append(v)
                    elif (
                        isinstance(v, list)
                        and len(v) > 0
                        and isinstance(v[0], torch.Tensor)
                    ):
                        shapes = [(t.shape, t.dtype) for t in v]
                        meta_dict[k] = ("tensor_list", shapes)
                        tensor_list.extend(v)
                    else:
                        meta_dict[k] = ("value", v)

            # Serialize metadata
            buffer = io.BytesIO()
            pickle.dump(meta_dict, buffer)
            meta_bytes = buffer.getvalue()

            # Create meta tensors on CUDA (to use NCCL)
            meta_tensor = torch.tensor(
                list(meta_bytes), dtype=torch.uint8, device="cuda"
            )
            size_tensor = torch.tensor(
                [len(meta_bytes)], dtype=torch.long, device="cuda"
            )
        else:
            # Receiver init
            size_tensor = torch.tensor([0], dtype=torch.long, device="cuda")

        # 2. Broadcast Size
        dist.broadcast(
            size_tensor, src=self.non_dit_master_rank, group=self.broadcast_group
        )

        # 3. Broadcast Metadata
        if not is_sender:
            meta_tensor = torch.empty(
                size_tensor.item(), dtype=torch.uint8, device="cuda"
            )

        dist.broadcast(
            meta_tensor, src=self.non_dit_master_rank, group=self.broadcast_group
        )

        # 4. Reconstruct and Prepare Tensors (Receiver)
        if not is_sender:
            meta_bytes = bytes(meta_tensor.cpu().tolist())
            meta_dict = pickle.loads(meta_bytes)

            # Reconstruct object
            cls = meta_dict.pop("__class__", None)
            if cls:
                try:
                    obj = cls.__new__(cls)
                except Exception:
                    # Fallback if __new__ fails or not applicable
                    class ReceivedObject:
                        pass

                    obj = ReceivedObject()
            else:

                class ReceivedObject:
                    pass

                obj = ReceivedObject()

            tensor_list = []

            for k, info in meta_dict.items():
                tag = info[0]
                if tag == "value":
                    setattr(obj, k, info[1])
                elif tag == "tensor":
                    shape, dtype = info[1], info[2]
                    t = torch.empty(shape, dtype=dtype, device="cuda")
                    setattr(obj, k, t)
                    tensor_list.append(t)
                elif tag == "tensor_list":
                    shapes = info[1]
                    t_list = []
                    for s, d in shapes:
                        t = torch.empty(s, dtype=d, device="cuda")
                        t_list.append(t)
                        tensor_list.append(t)
                    setattr(obj, k, t_list)

        # 5. Broadcast Tensors (Direct NVLink)
        for t in tensor_list:
            if is_sender:
                if not t.is_cuda:
                    # Warning: NCCL broadcast requires CUDA tensors.
                    # In high-perf path, we expect CUDA. If CPU, we should move it.
                    # But modifying in-place might be side-effect.
                    # For now, assume CUDA or copy.
                    t_cuda = t.cuda()
                    if not t_cuda.is_contiguous():
                        t_cuda = t_cuda.contiguous()
                    dist.broadcast(
                        t_cuda, src=self.non_dit_master_rank, group=self.broadcast_group
                    )
                else:
                    if not t.is_contiguous():
                        t = t.contiguous()
                    dist.broadcast(
                        t, src=self.non_dit_master_rank, group=self.broadcast_group
                    )
            else:
                # Receiver tensors are already allocated on CUDA
                dist.broadcast(
                    t, src=self.non_dit_master_rank, group=self.broadcast_group
                )

        return obj

    def isend_object_to_non_dit(self, obj: Any) -> List[Work]:
        """
        Non-blocking send of a complex object from DiT Master to Non-DiT Master (P2P).
        Separates metadata and tensors for high performance.
        """
        if self.world_rank != self.dit_master_rank:
            return []

        assert obj is not None
        meta_dict = {"__class__": obj.__class__}
        tensor_list = []

        # Extract state
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if isinstance(v, torch.Tensor):
                    meta_dict[k] = ("tensor", v.shape, v.dtype)
                    tensor_list.append(v)
                elif (
                    isinstance(v, list)
                    and len(v) > 0
                    and isinstance(v[0], torch.Tensor)
                ):
                    shapes = [(t.shape, t.dtype) for t in v]
                    meta_dict[k] = ("tensor_list", shapes)
                    tensor_list.extend(v)
                else:
                    meta_dict[k] = ("value", v)

        # Serialize metadata
        buffer = io.BytesIO()
        pickle.dump(meta_dict, buffer)
        meta_bytes = buffer.getvalue()

        # Create meta tensors on CUDA
        meta_tensor = torch.tensor(list(meta_bytes), dtype=torch.uint8, device="cuda")
        size_tensor = torch.tensor([len(meta_bytes)], dtype=torch.long, device="cuda")

        works = []
        # 1. Send Size
        works.append(
            dist.isend(size_tensor, dst=self.non_dit_master_rank, group=self.p2p_group)
        )
        # 2. Send Metadata
        works.append(
            dist.isend(meta_tensor, dst=self.non_dit_master_rank, group=self.p2p_group)
        )
        # 3. Send Tensors
        for t in tensor_list:
            if not t.is_cuda:
                t = t.cuda()
            if not t.is_contiguous():
                t = t.contiguous()
            works.append(
                dist.isend(t, dst=self.non_dit_master_rank, group=self.p2p_group)
            )

        return works

    def recv_object_from_dit(
        self, known_size_tensor: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Receive a complex object from DiT Master at Non-DiT Master (P2P).
        Optimized for NVLink.
        """
        if self.world_rank != self.non_dit_master_rank:
            return None

        # 1. Receive Size
        if known_size_tensor is not None:
            size_tensor = known_size_tensor
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.recv(size_tensor, src=self.dit_master_rank, group=self.p2p_group)

        # 2. Receive Metadata
        meta_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
        dist.recv(meta_tensor, src=self.dit_master_rank, group=self.p2p_group)

        # 3. Parse Metadata & Alloc Tensors
        meta_bytes = bytes(meta_tensor.cpu().tolist())
        meta_dict = pickle.loads(meta_bytes)

        # Reconstruct object
        cls = meta_dict.pop("__class__", None)
        if cls:
            try:
                obj = cls.__new__(cls)
            except Exception:

                class ReceivedObject:
                    pass

                obj = ReceivedObject()
        else:

            class ReceivedObject:
                pass

            obj = ReceivedObject()

        tensor_list = []
        for k, info in meta_dict.items():
            tag = info[0]
            if tag == "value":
                setattr(obj, k, info[1])
            elif tag == "tensor":
                shape, dtype = info[1], info[2]
                t = torch.empty(shape, dtype=dtype, device="cuda")
                setattr(obj, k, t)
                tensor_list.append(t)
            elif tag == "tensor_list":
                shapes = info[1]
                t_list = []
                for s, d in shapes:
                    t = torch.empty(s, dtype=d, device="cuda")
                    t_list.append(t)
                    tensor_list.append(t)
                setattr(obj, k, t_list)

        # 4. Receive Tensors
        for t in tensor_list:
            dist.recv(t, src=self.dit_master_rank, group=self.p2p_group)

        return obj
