from __future__ import annotations

import collections
import dataclasses
import functools
import math
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.managers.mm_utils import tensor_hash
from sglang.srt.utils import is_cuda, print_info_once

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import add_prefix

ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}


@dataclasses.dataclass
class SingletonCache:
    data: Any = None

    def set_data(self, value: Any) -> None:
        self.data = value

    def get_data(self) -> Optional[Any]:
        return self.data

    def empty(self) -> bool:
        return self.get_data() is None


@dataclasses.dataclass
class MultiCache:
    """Simple capped LRU cache for storing multiple tensors keyed by a hashable key.

    Notes:
    - Stores references to tensors. Caller must ensure device/dtype correctness.
    - Uses a small capacity to avoid memory blowup.
    """

    max_entries: int = 16
    _cache: dict = dataclasses.field(default_factory=dict)
    _order: collections.OrderedDict = dataclasses.field(
        default_factory=collections.OrderedDict
    )

    def empty(self) -> bool:
        return len(self._cache) == 0

    def get_data(self, key: Any) -> Optional[torch.Tensor]:
        if key in self._cache:
            # mark as most-recently-used
            self._order.move_to_end(key)
            return self._cache[key]
        return None

    def set_data(self, key: Any, value: torch.Tensor) -> None:
        if key in self._cache:
            self._cache[key] = value
            self._order.move_to_end(key)
            return
        # evict least-recently-used if over capacity
        if len(self._cache) >= self.max_entries:
            old_key, _ = self._order.popitem(last=False)
            self._cache.pop(old_key, None)
        self._cache[key] = value
        self._order[key] = None


def convert_hf_attention_backend_to_sgl_attention_backend(
    attn_implementation: Optional[str] = None,
):
    if attn_implementation is None:
        # softmax_in_single_precision = False
        qkv_backend = None
    elif attn_implementation == "sdpa":
        # softmax_in_single_precision = False
        qkv_backend = "sdpa"
    elif attn_implementation == "flash_attention_2":
        # softmax_in_single_precision = False
        qkv_backend = "triton_attn"
    elif attn_implementation == "eager":
        # softmax_in_single_precision = True
        qkv_backend = "sdpa"
    elif attn_implementation == "flash_attention_3":
        # softmax_in_single_precision = False
        qkv_backend = "fa3"
    return qkv_backend


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flattened_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_size)

    @staticmethod
    def _generate_simple_block_diag_mask_key(
        s: int,
        cu_seqlens: torch.Tensor,
        device_index: int,
        cross_fill_value: bool = True,
        with_batch_dim: bool = False,
    ) -> tuple:
        key = (
            s,
            tensor_hash(cu_seqlens),
            device_index,
            int(cross_fill_value),
            int(with_batch_dim),
        )
        return key

    @staticmethod
    def _generate_block_diag_mask(
        s: int,
        cu_seqlens: torch.Tensor,
        device: torch.device,
        same_sequence_fill_value: bool = True,
        with_batch_dim: bool = False,
    ) -> torch.BoolTensor:
        """
        Build a boolean attention mask on device.

        - Flattened case (with_batch_dim=False):
          Returns [1, s, s]. Each diagonal block (tokens from same sequence) is set to
          not same_sequence_fill_value, while cross-sequence positions are set to same_sequence_fill_value.

        - Non-flattened case (with_batch_dim=True):
          Returns [b, 1, S, S] where S = max_seqlen or max(seq_lens).
          For sample i, valid region [:L_i, :L_i] is set to not same_sequence_fill_value,
          padding/out-of-range is set to same_sequence_fill_value.
        """
        # print("_generate_block_diag_mask")
        if with_batch_dim:
            # Efficient, fully vectorized GPU implementation.
            # True means masked; valid [:L_i, :L_i] region is False when cross_fill_value=True.
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
                dtype=torch.int32, device=device
            )
            batch = int(seq_lens.numel())
            S = (
                int(s)
                if (s is not None)
                else (int(seq_lens.max().item()) if batch > 0 else 0)
            )
            if S == 0 or batch == 0:
                return torch.zeros((batch, 1, S, S), dtype=torch.bool, device=device)

            positions = torch.arange(S, device=device, dtype=torch.int32)
            row_valid = positions.view(1, 1, 1, S) < seq_lens.view(-1, 1, 1, 1)
            col_valid = positions.view(1, 1, S, 1) < seq_lens.view(-1, 1, 1, 1)
            valid_square = row_valid & col_valid  # [b, 1, S, S]
            # True means masked. To match previous behavior, when
            # same_sequence_fill_value=True, valid square should be False.
            mask = ~valid_square if same_sequence_fill_value else valid_square
            return mask

        # Flattened block-diagonal mask (fully vectorized on device)
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        if s == 0 or cu_seqlens.numel() <= 1:
            return torch.zeros((1, 0, 0), dtype=torch.bool, device=device)
        positions = torch.arange(s, device=device, dtype=torch.int32)
        # Positions inside any real interval [cu[i], cu[i+1]) are valid
        valid_pos = positions < cu_seqlens[-1]
        # seq_id for each position: number of sequence ends <= position
        seq_ids = torch.bucketize(positions, cu_seqlens[1:], right=True)
        # same is True only for pairs that are both valid and in the same sequence
        same = (seq_ids.view(1, s, 1) == seq_ids.view(1, 1, s)) & (
            valid_pos.view(1, s, 1) & valid_pos.view(1, 1, s)
        )
        # True means masked. To match previous behavior, set inside-block to not fill_value, others to fill_value
        mask = ~same if same_sequence_fill_value else same
        return mask

    @staticmethod
    def _generate_key_padding_mask(
        seq_lens: torch.Tensor,
        max_seqlen: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        """
        Build key-padding mask with shape [b, 1, 1, s]. True means valid.
        """
        seq_lens = seq_lens.to(device=device, dtype=torch.int32)
        s = int(max_seqlen)
        if s == 0:
            return torch.zeros(
                (seq_lens.numel(), 1, 1, 0), dtype=torch.bool, device=device
            )
        positions = torch.arange(s, device=device, dtype=torch.int32)
        # True indicates masked (padding) positions
        masked = positions.unsqueeze(0) >= seq_lens.view(-1, 1)
        return masked.view(-1, 1, 1, s)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flattened_batch:
            assert bsz == 1, "flatten_batch is True, bsz is expected to be 1"

        assert q.dim() == 3, q.shape

        s = q.shape[0] // bsz

        # Flattened path: build block-diagonal mask once on GPU
        if self.flattened_batch:
            if attention_mask is None:
                if cu_seqlens is None:
                    attention_mask = None
                else:
                    attention_mask = self._generate_block_diag_mask(
                        s=s,
                        cu_seqlens=cu_seqlens,
                        device=q.device,
                        same_sequence_fill_value=True,
                        with_batch_dim=False,
                    )

            if attention_mask is None:
                if self.softmax_in_single_precision:
                    raise RuntimeError("Empty attention mask")
            elif isinstance(attention_mask, SingletonCache):
                # Singleton cache just stores last mask
                if attention_mask.empty():
                    mask_data = self._generate_block_diag_mask(
                        s=s,
                        cu_seqlens=cu_seqlens,
                        device=q.device,
                        same_sequence_fill_value=True,
                        with_batch_dim=False,
                    )
                    attention_mask.set_data(mask_data)
                attention_mask = attention_mask.data
            elif isinstance(attention_mask, MultiCache):
                key = VisionSdpaAttention._generate_simple_block_diag_mask_key(
                    s,
                    cu_seqlens,
                    torch.cuda.current_device(),
                    cross_fill_value=True,
                    with_batch_dim=False,
                )
                mask_data = attention_mask.get_data(key)
                if mask_data is None:
                    mask_data = self._generate_block_diag_mask(
                        s=s,
                        cu_seqlens=cu_seqlens,
                        device=q.device,
                        same_sequence_fill_value=True,
                        with_batch_dim=False,
                    )
                    attention_mask.set_data(key, mask_data)
                attention_mask = mask_data
            else:
                attention_mask = attention_mask.to(device=q.device)

            q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

            if self.softmax_in_single_precision:
                k = rearrange(k, "b h s d -> b h d s")
                attn_weights = torch.matmul(q, k) * self.scale
                del k
                # attention_mask is bool with True=masked, convert to additive
                add_mask = attention_mask * torch.finfo(q.dtype).min
                attn_weights = attn_weights + add_mask
                del add_mask
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights, p=self.dropout, training=False
                )
                output = torch.matmul(attn_weights, v)
                del attn_weights, v
            else:
                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout,
                    is_causal=False,
                )

            output = rearrange(output, "b h s d -> (b s) h d")
            return output

        # Non-flattened path: pack variable-length sequences and use key-padding mask
        if cu_seqlens is None:
            q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
            output = rearrange(output, "b h s d -> (b s) h d")
            return output

        cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=q.device)
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        real_bsz = int(seq_lens.numel())
        max_seqlen = (
            max_seqlen or int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        )

        if max_seqlen == 0:
            return q.new_zeros(q.shape)

        q_chunks = list(q.split([int(l.item()) for l in seq_lens], dim=0))
        k_chunks = list(k.split([int(l.item()) for l in seq_lens], dim=0))
        v_chunks = list(v.split([int(l.item()) for l in seq_lens], dim=0))

        q_padded = torch.nn.utils.rnn.pad_sequence(q_chunks, batch_first=True)
        k_padded = torch.nn.utils.rnn.pad_sequence(k_chunks, batch_first=True)
        v_padded = torch.nn.utils.rnn.pad_sequence(v_chunks, batch_first=True)

        q_padded = q_padded.permute(0, 2, 1, 3).contiguous()
        k_padded = k_padded.permute(0, 2, 1, 3).contiguous()
        v_padded = v_padded.permute(0, 2, 1, 3).contiguous()
        # Build key-padding mask (bool: True=masked)
        key_padding_mask = self._generate_key_padding_mask(
            seq_lens=seq_lens, max_seqlen=max_seqlen, device=q.device
        )

        output = F.scaled_dot_product_attention(
            q_padded,
            k_padded,
            v_padded,
            attn_mask=key_padding_mask,
            dropout_p=self.dropout,
            is_causal=False,
        )

        outs = []
        for i in range(real_bsz):
            Li = int(seq_lens[i].item())
            if Li == 0:
                continue
            outs.append(output[i, :, :Li, :].permute(1, 0, 2).contiguous())
        return torch.cat(outs, dim=0) if len(outs) > 0 else q.new_zeros(q.shape)


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # [b * s, head, head_size]
        output = torch.empty_like(q)
        cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        if max_seqlen is None:
            max_seqlen = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        else:
            max_seqlen = int(max_seqlen)
        if max_seqlen == 0:
            return q.new_zeros(q.shape)

        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens,
            seq_lens,
            max_seqlen,
            is_causal=False,
        )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash3Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[Union[SingletonCache, torch.Tensor]],
        bsz: int,
        seq_len: int,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
                )
            cu_seqlens = cu_seqlens.get_data()

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        if max_seqlen is None:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        return output


MM_ATTENTION_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision
        flattened_batch: (bool, default to False):
            if ``True``, the input tokens should already be flattened in batch dim, and a block_diag mask denoting each batch's tokens will be generated and used for attention.
            Otherwise, a key-padding mask will be used

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        flattened_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ):
        super().__init__()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.tp_size = attn_tp_size
        self.tp_rank = attn_tp_rank
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )
            self.k_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )

        # priority: server_args > passed qkv_backend > sdpa
        if global_server_args_dict["mm_attention_backend"] is None:
            if qkv_backend is None:
                if is_cuda():
                    # Double prefill throughput by setting attn backend to Triton on CUDA
                    qkv_backend = "triton_attn"
                else:
                    qkv_backend = "sdpa"
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        else:
            qkv_backend = global_server_args_dict["mm_attention_backend"]

        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self.qkv_backend = MM_ATTENTION_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flattened_batch,
            softmax_in_single_precision=softmax_in_single_precision,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("proj", prefix),
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn"""
        q = q.flatten(1, 2)
        k = k.flatten(1, 2)

        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        q = q.unflatten(-1, (-1, self.head_size))
        k = k.unflatten(-1, (-1, self.head_size))
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if position_embeddings is not None:
            original_shape = q.shape

            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
                q = q.view(original_shape)
                k = k.view(original_shape)
            else:
                cos, sin = position_embeddings

                # [total_tokens, head, head_size]
                q = q.view(-1, head, self.head_size)
                k = k.view(-1, head, self.head_size)

                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                q = q.view(original_shape)
                k = k.view(original_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output
