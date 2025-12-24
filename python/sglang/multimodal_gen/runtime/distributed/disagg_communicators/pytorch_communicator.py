"""
PyTorch Implementation of DisaggCommunicator.
"""

import io
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import Work

from sglang.multimodal_gen.runtime.distributed.disagg_communicators.base_communicator import (
    DisaggCommunicator,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class AsyncWork:
    """
    Wraps a torch.distributed.Work handle with its associated tensors
    to prevent them from being garbage collected before the operation completes.
    """

    works: List[Work]
    tensors: List[torch.Tensor]
    callback: Optional[Callable] = None
    metadata: Optional[Dict] = None
    expected_signal_value: Optional[int] = None

    def is_completed(self) -> bool:
        """
        Check if all work handles in this batch are completed.

        """
        statuses = [w.is_completed() for w in self.works]
        all_works_completed = all(statuses)

        if not all_works_completed and self.expected_signal_value is not None:
            if len(self.tensors) == 1 and self.tensors[0].numel() == 1:
                current_value = self.tensors[0].item()
                if current_value == self.expected_signal_value:
                    return True

        return all_works_completed

    def wait(self) -> None:
        """Wait for all work handles to complete."""
        for w in self.works:
            w.wait()


class AsyncWorkRegistry:
    """
    A registry to track and poll asynchronous distributed operations.
    Useful for non-blocking event loops to manage keep-alive tensors and callbacks.
    """

    def __init__(self):
        self.pending_works: List[AsyncWork] = []

    def add(
        self,
        work: Union[Work, List[Work]],
        tensors: Union[torch.Tensor, List[torch.Tensor]],
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None,
        expected_signal_value: Optional[int] = None,
    ) -> AsyncWork:
        """
        Add a new async work to the registry.

        Args:
            expected_signal_value: For signal reception, the expected value after completion.
                                   This is a workaround for Gloo backend not updating work status.
        """
        if not isinstance(work, list):
            work = [work]
        if not isinstance(tensors, list):
            tensors = [tensors]

        aw = AsyncWork(
            works=work,
            tensors=tensors,
            callback=callback,
            metadata=metadata,
            expected_signal_value=expected_signal_value,
        )
        self.pending_works.append(aw)
        return aw

    def poll(self) -> List[AsyncWork]:
        """
        Poll all pending works and return the ones that have completed.
        Completed works are removed from the registry and their callbacks (if any) are executed.
        """
        completed = []
        still_pending = []
        for aw in self.pending_works:
            if aw.is_completed():
                if aw.callback:
                    try:
                        aw.callback(aw)
                    except Exception as e:
                        logger.error(
                            f"[Rank {dist.get_rank()}] Error in AsyncWork callback: {e}",
                            exc_info=True,
                        )
                completed.append(aw)
            else:
                still_pending.append(aw)

        self.pending_works = still_pending
        return completed

    def clear(self):
        """Clear all pending works."""
        self.pending_works = []

    def __len__(self):
        return len(self.pending_works)


class PyTorchDisaggCommunicator(DisaggCommunicator):
    def __init__(self):
        self.dit_group: GroupCoordinator | None = None
        self.p2p_group: GroupCoordinator | None = (
            None  # Dedicated P2P group for master-to-master communication
        )
        self.signal_group: GroupCoordinator | None = (
            None  # Dedicated Gloo group for non-blocking signals
        )
        self.broadcast_group: GroupCoordinator | None = None
        self.non_dit_ranks = []
        self.dit_ranks = []
        self.role = None  # "non_dit" or "dit"
        self.world_rank = -1

        # We assume rank 0 of Non-DiT group communicates with rank 0 of DiT group.
        # These are GLOBAL ranks.
        self.non_dit_master_rank = -1
        self.dit_master_rank = -1

    def initialize_topology(self, server_args: Any) -> None:
        self.world_rank = dist.get_rank()
        world_size = dist.get_world_size()

        non_dit_ranks = list(
            range(world_size - server_args.num_non_dit_ranks, world_size)
        )
        dit_ranks = [r for r in range(world_size) if r not in non_dit_ranks]
        all_ranks = [r for r in range(world_size)]

        self.non_dit_ranks = non_dit_ranks
        self.dit_ranks = dit_ranks
        self.non_dit_master_rank = non_dit_ranks[0]
        self.dit_master_rank = dit_ranks[0]

        logger.info(
            f"Initializing Disagg Topology: Non-DiT Ranks={non_dit_ranks}, DiT Ranks={dit_ranks}"
        )

        # # This avoids serialization warnings on the default ProcessGroup
        p2p_ranks = [self.dit_master_rank, self.non_dit_master_rank]
        non_p2p_ranks = [rank for rank in all_ranks if rank not in p2p_ranks]
        dist.barrier()
        p2p_ranks_all = [p2p_ranks, non_p2p_ranks]
        self.signal_group = GroupCoordinator(
            group_ranks=p2p_ranks_all,
            local_rank=self.world_rank,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            group_name="signal",
        )

        dit_rank_groups = [dit_ranks, non_dit_ranks]
        self.dit_group = GroupCoordinator(
            group_ranks=dit_rank_groups,
            local_rank=self.world_rank,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            group_name="dit",
        )

        self.p2p_ranks = p2p_ranks
        dist.barrier()

        broadcast_ranks = sorted(list(set([self.non_dit_master_rank] + dit_ranks)))
        non_broadcast_ranks = [
            rank for rank in all_ranks if rank not in broadcast_ranks
        ]
        if non_broadcast_ranks:
            broadcast_ranks_all = [broadcast_ranks, non_broadcast_ranks]
        else:
            broadcast_ranks_all = [broadcast_ranks]
        self.broadcast_group = GroupCoordinator(
            group_ranks=broadcast_ranks_all,
            local_rank=self.world_rank,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            group_name="dit",
        )
        self.dispatch_ranks = broadcast_ranks
        dist.barrier()

        if self.world_rank in non_dit_ranks:
            self.role = "non_dit"
        elif self.world_rank in dit_ranks:
            self.role = "dit"
        else:
            raise ValueError(f"Rank {self.world_rank} not assigned to any group!")

        logger.info(
            f"[Rank {self.world_rank}] Disagg Topology initialization COMPLETED. "
            f"Role={self.role}"
        )

    def get_my_group(self) -> Optional[dist.ProcessGroup]:
        return self.dit_group

    def is_dit_rank(self) -> bool:
        return self.role == "dit"

    def is_non_dit_rank(self) -> bool:
        return self.role == "non_dit"

    def wait_all_works(self, works: List[Optional[Work]]) -> None:
        """Wait for multiple Work handles to complete."""
        for work in works:
            if work is not None:
                work.wait()

    def irecv_size_from_non_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """
        All DiT ranks listen for signal from Non-DiT Master via Gloo (CPU).
        This must be truly non-blocking.
        """
        if self.is_dit_rank():
            assert tensor.device.type == "cpu", "Gloo dispatch signal must be on CPU"
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            if self.broadcast_group is None:
                logger.error(f"[Rank {self.world_rank}] dispatch_group is None!")
                return None

            work = dist.irecv(
                tensor,
                src=self.non_dit_master_rank,
                group=self.broadcast_group.cpu_group,
            )
            return work
        return None

    def isend_signal_to_dit(self, tensor: torch.Tensor) -> List[Work]:
        """
        Non-DiT Master sends signal to ALL DiT ranks via Gloo (CPU).
        """
        works = []
        if self.world_rank == self.non_dit_master_rank:
            assert tensor.device.type == "cpu", "Gloo dispatch signal must be on CPU"
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            if self.broadcast_group is None:
                return works

            for rank in self.dit_ranks:
                works.append(
                    dist.isend(tensor, dst=rank, group=self.broadcast_group.cpu_group)
                )
        return works

    def irecv_signal_from_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """Non-DiT Master listens for signal from DiT Master (P2P via Gloo)."""
        if self.world_rank == self.non_dit_master_rank:
            assert tensor.device.type == "cpu", "Gloo signal tensor must be on CPU"
            return dist.irecv(
                tensor, src=self.dit_master_rank, group=self.signal_group.cpu_group
            )
        return None

    def isend_signal_to_non_dit(
        self, tensor: torch.Tensor
    ) -> tuple[Optional[Work], Optional[torch.Tensor]]:
        """DiT Master sends signal to Non-DiT Master (P2P via Gloo)."""
        if self.world_rank == self.dit_master_rank:
            cpu_tensor = tensor.to("cpu")
            # CRITICAL: 使用全局 rank，不是 group rank
            work = dist.isend(
                cpu_tensor,
                dst=self.non_dit_master_rank,
                group=self.signal_group.cpu_group,
            )
            return work, cpu_tensor
        return None, None

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
            size_tensor,
            src=self.non_dit_master_rank,
            group=self.broadcast_group.device_group,
        )

        # 3. Broadcast Metadata
        if not is_sender:
            meta_tensor = torch.empty(
                size_tensor.item(), dtype=torch.uint8, device="cuda"
            )

        dist.broadcast(
            meta_tensor,
            src=self.non_dit_master_rank,
            group=self.broadcast_group.device_group,
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
                        t_cuda,
                        src=self.non_dit_master_rank,
                        group=self.broadcast_group.device_group,
                    )
                else:
                    if not t.is_contiguous():
                        t = t.contiguous()
                    dist.broadcast(
                        t,
                        src=self.non_dit_master_rank,
                        group=self.broadcast_group.device_group,
                    )
            else:
                # Receiver tensors are already allocated on CUDA
                dist.broadcast(
                    t,
                    src=self.non_dit_master_rank,
                    group=self.broadcast_group.device_group,
                )

        return obj

    def isend_object_to_non_dit(
        self, obj: Any
    ) -> tuple[List[Work], List[torch.Tensor]]:
        """
        Non-blocking send of a complex object from DiT Master to Non-DiT Master (P2P).
        Uses Gloo backend with CPU tensors for truly non-blocking communication.
        """
        if self.world_rank != self.dit_master_rank:
            return [], []

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

        # Use CPU tensors for Gloo backend (truly non-blocking)
        meta_tensor = torch.tensor(list(meta_bytes), dtype=torch.uint8, device="cpu")
        size_tensor = torch.tensor([len(meta_bytes)], dtype=torch.long, device="cpu")

        keep_alive_tensors = [meta_tensor, size_tensor]

        # CRITICAL: 使用全局 rank，不是 group rank
        dst_rank = self.non_dit_master_rank

        works = []
        # 1. Send Size (via Gloo)
        works.append(
            dist.isend(size_tensor, dst=dst_rank, group=self.signal_group.cpu_group)
        )
        # 2. Send Metadata (via Gloo)
        works.append(
            dist.isend(meta_tensor, dst=dst_rank, group=self.signal_group.cpu_group)
        )
        # 3. Send Tensors (via Gloo with CPU tensors)
        for t in tensor_list:
            # Move to CPU for Gloo backend
            if t.is_cuda:
                t = t.cpu()
            if not t.is_contiguous():
                t = t.contiguous()

            keep_alive_tensors.append(t)
            works.append(dist.isend(t, dst=dst_rank, group=self.signal_group.cpu_group))

        return works, keep_alive_tensors

    def recv_object_from_dit(
        self, known_size_tensor: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Receive a complex object from DiT Master at Non-DiT Master (P2P).
        Uses Gloo backend with CPU tensors for truly non-blocking communication.
        """
        if self.world_rank != self.non_dit_master_rank:
            return None

        # CRITICAL: 使用全局 rank，不是 group rank
        src_rank = self.dit_master_rank

        # 1. Receive Size (via Gloo, CPU tensor)
        if known_size_tensor is not None:
            size_tensor = known_size_tensor
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
            dist.recv(size_tensor, src=src_rank, group=self.signal_group.cpu_group)

        # 2. Receive Metadata (via Gloo, CPU tensor)
        meta_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cpu")
        dist.recv(meta_tensor, src=src_rank, group=self.signal_group.cpu_group)

        # 3. Parse Metadata & Alloc Tensors
        meta_bytes = bytes(meta_tensor.tolist())
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
        tensor_targets = (
            []
        )  # Store (attribute_name, index_in_list) for later CUDA transfer
        for k, info in meta_dict.items():
            tag = info[0]
            if tag == "value":
                setattr(obj, k, info[1])
            elif tag == "tensor":
                shape, dtype = info[1], info[2]
                # Receive on CPU first, then move to CUDA
                t = torch.empty(shape, dtype=dtype, device="cpu")
                tensor_list.append(t)
                tensor_targets.append((k, None, shape, dtype))
            elif tag == "tensor_list":
                shapes = info[1]
                t_list = []
                for idx, (s, d) in enumerate(shapes):
                    t = torch.empty(s, dtype=d, device="cpu")
                    t_list.append(t)
                    tensor_list.append(t)
                    tensor_targets.append((k, idx, s, d))
                setattr(obj, k, t_list)

        # 4. Receive Tensors (via Gloo, CPU tensors)
        for t in tensor_list:
            dist.recv(t, src=src_rank, group=self.signal_group.cpu_group)

        # 5. Move tensors to CUDA after receiving
        tensor_idx = 0
        for k, list_idx, shape, dtype in tensor_targets:
            cpu_tensor = tensor_list[tensor_idx]
            cuda_tensor = cpu_tensor.cuda()
            if list_idx is None:
                # Single tensor
                setattr(obj, k, cuda_tensor)
            else:
                # Tensor in list
                getattr(obj, k)[list_idx] = cuda_tensor
            tensor_idx += 1

        return obj
