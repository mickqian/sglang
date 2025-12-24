"""
Abstract base class for Disaggregation Communication.

This module defines the interface for communication between different components
(e.g., Non-DiT encoder/decoder and DiT denoiser) in a disaggregated architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from torch.distributed import ProcessGroup, Work


class DisaggCommunicator(ABC):
    """
    Abstract base class for disaggregation communication.

    This class handles:
    1. Topology management (who is Encoder, who is DiT)
    2. Inter-group P2P communication (Send/Recv tensors)
    3. Intra-group collective communication (Broadcast inputs to SP/TP group)
    """

    @abstractmethod
    def initialize_topology(self, server_args: Any) -> None:
        """
        Initialize the distributed topology based on server arguments.

        This should setup:
        - self.non_dit_group: ProcessGroup for Encoder/VAE ranks
        - self.dit_group: ProcessGroup for DiT ranks
        - self.rank_role: "non_dit" or "dit"
        """
        pass

    @abstractmethod
    def get_my_group(self) -> Optional[ProcessGroup]:
        """Return the ProcessGroup this rank belongs to."""
        pass

    @abstractmethod
    def is_dit_rank(self) -> bool:
        """Return True if this rank is part of the DiT group."""
        pass

    @abstractmethod
    def is_non_dit_rank(self) -> bool:
        """Return True if this rank is part of the Non-DiT group."""
        pass

    @abstractmethod
    def wait_all_works(self, works: List[Optional[Work]]) -> None:
        """
        Wait for multiple Work handles to complete.
        Filters out None values automatically.
        """
        pass

    @abstractmethod
    def broadcast_object_from_non_dit(self, obj: Optional[Any] = None) -> Any:
        """
        Broadcast a complex object (e.g. Req) with tensors from Non-DiT Master to all DiT ranks.

        Optimized for NVLink: separates metadata and tensors.
        - Sender (Non-DiT Master): Pass 'obj'.
        - Receiver (DiT Ranks): Pass None. Returns the received object.
        - Other Ranks: Returns None.
        """
        pass

    @abstractmethod
    def isend_object_to_non_dit(
        self, obj: Any
    ) -> tuple[List[Work], List[torch.Tensor]]:
        """
        Non-blocking send of a complex object from DiT Master to Non-DiT Master (P2P).
        Returns (list of Work handles, list of keep-alive tensors).
        """
        pass

    @abstractmethod
    def isend_signal_to_non_dit(
        self, tensor: torch.Tensor
    ) -> tuple[Optional[Work], Optional[torch.Tensor]]:
        """
        Non-blocking send of a signal from DiT Master to Non-DiT Master (P2P).
        Returns (Work handle, persistent tensor).
        """
        pass

    @abstractmethod
    def recv_object_from_dit(
        self, known_size_tensor: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Receive a complex object from DiT Master at Non-DiT Master (P2P).
        If known_size_tensor is provided, skips receiving the size header.
        """
        pass

    def isend_signal_to_dit(self, signal_tensor):
        pass

    def irecv_size_from_non_dit(self, signal_tensor):
        pass

    def irecv_signal_from_dit(self, signal_tensor):
        pass
