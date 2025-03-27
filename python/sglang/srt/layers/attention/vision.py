from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Parameter

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb,
)
from sglang.srt.utils import add_prefix

ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
    "multimodal": apply_multimodal_rotary_pos_emb,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for ViT.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        use_context_forward (bool, default to True):
            if ``True``, a flash_attn style attention will be applied
            Otherwise, a full-sequence attention will be applied.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        use_context_forward: bool = True,
        softmax_in_single_precision: bool = False,
        rotary_embed: Optional[str] = None,
        # TODO: only for mm rotary embed, refactor this
        mrope_section: Optional[int] = None,
        flatten_batch: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.use_context_forward = use_context_forward
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )

        self.rotary_embed = None
        if rotary_embed is not None:
            # if not rotary_embed in ROTARY_EMBED_CLASSES:
            self.rotary_embed = ROTARY_EMBED_CLASSES[rotary_embed]
        self.mrope_section = mrope_section

        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_key_value_heads, world_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        if self.use_context_forward:
            self.qkv_backend = VisionTritonAttention()
        else:
            self.qkv_backend = VisionSdpaAttention(
                head_size=self.head_size,
                num_heads=self.num_attention_heads_per_partition,
                num_kv_heads=self.num_attention_kv_heads_per_partition,
                dropout=dropout,
                flatten_batch=flatten_batch,
                softmax_in_single_precision=softmax_in_single_precision,
            )

        self.use_qkv_parallel = use_qkv_parallel
        num_key_value_heads = num_key_value_heads or num_heads
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_heads,
                total_num_kv_heads=num_key_value_heads,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * projection_size,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        bsz, s, _ = x.shape
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
            # [s, b, head * 3 * head_size] --> [s, b, head, 3 * head_size]
            new_x_shape = qkv.size()[:-1] + (
                head,
                3 * self.hidden_size_per_attention_head,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = dist_utils.split_tensor_along_last_dim(qkv, 3)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if position_embeddings is not None:
            if not self.rotary_embed:
                raise RuntimeError()
            cos, sin = position_embeddings

            original_shape_q = q.shape
            original_numel_k = k.numel()
            original_shape_k = k.shape
            print(f"{original_shape_q=}")
            print(f"{original_shape_k=}")

            # [total_tokens, head, head_size]
            q = q.view(head, -1, self.head_size)
            k = k.view(kv_head, -1, self.head_size)
            print(f"{q.shape=}")
            print(f"{k.shape=}")
            q, k = self.rotary_embed(
                q=q, k=k, cos=cos, sin=sin, mrope_section=self.mrope_section
            )
            # assert original_numel_k == k.nuem(

            print(f"{q.shape=}")
            print(f"{k.shape=}")
            # -> [b * s, head, head_size]
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, kv_head, self.head_size)

        if self.use_qkv_parallel:
            pass
        else:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]

        output = self.qkv_backend.forward(q, k, v, bsz, cu_seqlens, attention_mask)

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

    def load_weights(
        self,
        name: str,
        weight: torch.Tensor,
        shard_id: Optional[str] = None,
    ):
        """Load weights by name into the appropriate layer.

        Args:
            name: The name of the parameter to load (e.g. "qkv_proj" or "proj")
            weight: The weight tensor to load
            shard_id: For QKV projections, can be "q", "k", or "v"
        """
        if name == "qkv_proj":
            if isinstance(self.qkv_proj, QKVParallelLinear):
                # Handle QKV projection weights with possible sharding
                for param_name, param in self.qkv_proj.named_parameters():
                    if "weight" in param_name:
                        self.qkv_proj.weight_loader(param, weight, shard_id)
                    elif "bias" in param_name:
                        self.qkv_proj.weight_loader(param, weight)
            else:
                # Handle ColumnParallelLinear case
                for param_name, param in self.qkv_proj.named_parameters():
                    self.qkv_proj.weight_loader(param, weight)
        elif name == "proj":
            # Handle output projection weights
            for param_name, param in self.proj.named_parameters():
                self.proj.weight_loader(param, weight)
        else:
            raise ValueError(f"Unknown weight name {name} in VisionAttention")

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        """Load weights into both QKV and output projection layers.

        Args:
            param: The parameter to load weights into
            loaded_weight: The weight tensor to load
            loaded_shard_id: For QKV projections, can be "q", "k", or "v"
        """

        if param in self.qkv_proj.parameters():
            # Handle QKV projection weights
            if isinstance(self.qkv_proj, QKVParallelLinear):
                self.qkv_proj.weight_loader(param, loaded_weight, loaded_shard_id)
            else:
                # For ColumnParallelLinear case
                self.qkv_proj.weight_loader(param, loaded_weight)
        elif param in self.proj.parameters():
            # Handle output projection weights
            self.proj.weight_loader(param, loaded_weight)
        else:
            raise ValueError(f"Unknown parameter {param} in VisionAttention")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
    ):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            ...
            # if self.softmax_in_single_precision:
            #     raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        # print(f"396 {q.shape=}")
        # print(f"396 {k.shape=}")
        # print(f"396 {v.shape=}")
        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]
        # print(f"399 {q.shape=}")
        # print(f"400 {k.shape=}")
        # print(f"400 {v.shape=}")
        if self.softmax_in_single_precision:
            scale = self.head_size**-0.5
            num_key_value_groups = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, num_key_value_groups)
            v = repeat_kv(v, num_key_value_groups)
            # print(f"424 {k.shape=}")
            # print(f"426 {v.shape=}")
            k = rearrange(k, "b h s d -> b h d s")
            # print(f"401 {q.shape=}")
            # print(f"401 {k.shape=}")
            attn_weights = torch.matmul(q, k) * scale
            # print(f"401 {attn_weights.shape=}")

            # del k, k
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]
                attn_weights = attn_weights + causal_mask
                del attention_mask
            # attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            # attn_weights = attn_weights + attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            # print(f"451 {attn_weights.shape=}")
            # print(f"451 {v.shape=}")

            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")
        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        _bsz: int,
        cu_seqlens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        # [b * s, head, head_size]
        output = torch.empty_like(q)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens.cuda(),
            seq_lens.cuda(),
            max_seqlen,
            is_causal=False,
        )

        return output
