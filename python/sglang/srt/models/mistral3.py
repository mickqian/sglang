# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Set, TypedDict, TypeVar, Union

import numpy as np
import torch
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from ..configs.pixtral import Mistral3Config, NestedTensors
from ..layers.attention.vision import VisionAttention
from ..layers.logits_processor import LogitsProcessorOutput
from ..layers.quantization import QuantizationConfig
from ..layers.rotary_embedding import rotate_half
from ..managers.image_processor import MultiModalEmbeddings
from ..managers.multi_modality_utils import (
    MultiModalityDataPaddingPatternImageTokens,
    embed_image_inputs,
)
from ..managers.schedule_batch import ImageInputs
from ..model_executor.forward_batch_info import ForwardBatch, ForwardMode
from ..model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from ..utils import add_prefix
from .mistral import MistralForCausalLM

_T = TypeVar("_T")
_U = TypeVar("_U")

JSONTree = Union[
    dict[str, "JSONTree[_T]"], list["JSONTree[_T]"], tuple["JSONTree[_T]", ...], _T
]

T = TypeVar("T")


def flatten_2d_lists(lists: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


def json_map_leaves(
    func: Callable[[_T], _U],
    value: JSONTree[_T],
) -> JSONTree[_U]:
    """Apply a function to each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        return {k: json_map_leaves(func, v) for k, v in value.items()}
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict[str, torch.Tensor]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


def resolve_visual_encoder_outputs(
    encoder_outputs: Union[torch.Tensor, list[torch.Tensor]],
    feature_sample_layers: Optional[list[int]],
    post_layer_norm: Optional[torch.nn.LayerNorm],
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs is a list containing
    # the inputs to the visual encoder, followed by the hidden states
    # of each layer.
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        (
            encoder_outputs[layer_idx]
            if layer_idx >= 0
            else encoder_outputs[layer_idx + offset]
        )
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)


#
# class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):
#
#     def get_dummy_processor_inputs(
#         self,
#         seq_len: int,
#         mm_counts: Mapping[str, int],
#     ) -> ProcessorInputs:
#         num_images = mm_counts.get("image", 0)
#
#         target_width, target_height = self.info.get_image_size_with_most_features()
#
#         mm_data = {
#             "image": self._get_dummy_images(
#                 width=target_width, height=target_height, num_images=num_images
#             )
#         }
#
#         return ProcessorInputs(
#             prompt_text="",
#             mm_data=mm_data,
#         )
#


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders"
        )

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds


class PixtralImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]

    images: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, image_width, image_height)`

    The result of stacking :attr:`ImageEncoding.tokens` from each prompt.
    """

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.

    Shape: `(batch_size, num_images, num_embeds)`
    """

    num_patches: Union[torch.Tensor, list[torch.Tensor]]
    """Shape: `(batch_size, num_images)`"""


from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


class PixtralRotaryEmbedding(nn.Module):
    """
    The key with pixtral embedding is just that you have a frequency for each pixel positions.
    If you have height x width pixels (or embedding pixels), then the frequency used for ROPE
    is given by indexing the pre_computed frequency on the width and height.

    What you output is of dimension (batch, height * width, dim) with dim the embed dim.

    This simply means that for each image hidden state, you are going to add
    a corresponding positional embedding, based on its index in the grid.
    """

    def __init__(self, config, device):
        super().__init__()
        self.rope_type = "default"
        self.dim = config.head_dim
        self.base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        h = torch.arange(max_patches_per_side, device=freqs.device)
        w = torch.arange(max_patches_per_side, device=freqs.device)

        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(
            -1, self.dim // 2
        )  # we reshape to only index on the position indexes, not tuple of indexes
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        # TODO maybe make it torch compatible later on. We can also just slice
        self.register_buffer(
            "inv_freq", torch.cat((inv_freq, inv_freq), dim=-1), persistent=False
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        freqs = self.inv_freq[position_ids]
        # position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            emb = freqs
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=0
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, patches, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Pixtral
class PixtralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Pixtral
class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PixtralAttentionLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig]
    ):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            quant_config=quant_config,
            use_context_forward=False,
            softmax_in_single_precision=True,
            bias=False,
        )
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            x=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class PixtralTransformer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig]
    ):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                PixtralAttentionLayer(config=config, quant_config=quant_config)
            )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embeddings which serve as input to the Transformer.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=[hidden_states],
            attentions=all_attentions,
        )


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device
    )

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


class PixtralVisionModel(PreTrainedModel):
    base_model_prefix = "vision_encoder"

    def __init__(
        self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig]
    ):
        super().__init__(config)
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config=config, quant_config=quant_config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(
            config, device=self.device
        )

    def forward(
        self,
        pixel_values: List[torch.Tensor],
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This means it will be a list of tensors
                because multiple requests batched can have multiple images,
                each with their own shape potentially
        """
        # pass images through initial convolution independently
        # if len(pixel_values) > 1:
        #     raise ValueError("Batching/padding not supported yet!")
        patch_embeds_list = [
            self.patch_conv(pixel_value.unsqueeze(0).to(self.dtype))
            for pixel_value in pixel_values
        ]

        # flatten to a single sequence
        patch_embeds = [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list]

        patch_embeds = torch.cat(patch_embeds, dim=1)

        # patch_embeds = torch.cat(
        #     [p.flatten(1).T for p in patch_embeds_list], dim=0
        # ).unsqueeze(0)

        print(f"flattened patch embeds shape: {patch_embeds.shape}")

        patch_embeds = self.ln_pre(patch_embeds)
        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        ).to(self.device)

        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )
        return self.transformer(patch_embeds, attention_mask, position_embedding)


def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, transformers.)```.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
        to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.dtype == torch.complex64
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def position_meshgrid(
    patch_embeds_list: List[torch.Tensor],
) -> torch.Tensor:
    positions = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(p.shape[-2]),
                    torch.arange(p.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for p in patch_embeds_list
        ]
    )
    return positions


class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mistral3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        print(f"patch merger image_sizes: {image_sizes}")

        print(f"image_features shape: {image_features.shape}")
        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        print(f"{tokens_per_image=}")

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config.vision_config
        self.spatial_merge_size_square = config.spatial_merge_size**2
        self.norm = Mistral3RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)
        self.patch_size = self.config.patch_size
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, images: List[torch.Tensor], image_features: torch.Tensor):
        image_features = self.norm(image_features)
        img_patch_dims = [
            (img.shape[-2] // self.patch_size, img.shape[-1] // self.patch_size)
            for img in images
        ]
        for image in images:
            print(f"mm projector {image.shape=}")
        print(f"{img_patch_dims=}")
        print(f"{self.patch_size=}")
        feature_sizes = [image_feature.shape[0] for image_feature in image_features]
        feature_sizes = [
            feature_size // self.spatial_merge_size_square
            for feature_size in feature_sizes
        ]
        image_features = self.patch_merger(image_features, image_sizes=img_patch_dims)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        # image_embeds = torch.split(hidden_states, feature_sizes)
        image_embeds = hidden_states
        return image_embeds


class PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
        use_mlp_bias: bool = False,
    ) -> None:
        super().__init__()

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = nn.Linear(
            mlp_input_dim,
            vision_encoder_dim,
            bias=use_mlp_bias,
        )

    def forward(
        self, x: torch.Tensor, image_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == len(x)

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2,
        #       vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """

        sub_grids = get_sub_grids(
            x=x, image_sizes=image_sizes, spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor: list[torch.Tensor] = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted_tensor.append(
                grid.view(-1, n_patches).t()
            )  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(
            permuted_tensor, dim=0
        )  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1)[
            None, :, :, :
        ]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(
            image_grid, kernel_size=sub_grid_size, stride=sub_grid_size
        )
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size, -1
        )  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids


class Mistral3ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        *,
        config: Mistral3Config,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        multimodal_config = config.vision_config
        self.multimodal_config = multimodal_config

        self.vision_tower = PixtralVisionModel(
            config.vision_config, quant_config=quant_config
        )

        # self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # init MistralForCausalLM
        self.language_model = MistralForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "language_model"),
        )

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        # self.make_empty_intermediate_tensors = (
        #     self.language_model.make_empty_intermediate_tensors
        # )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[PixtralImagePixelInputs]:
        images = kwargs.pop("images", None)
        if images is None:
            return None

        if not isinstance(images, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. " f"Got type: {type(images)}")

        embed_is_patch = kwargs.pop("embed_is_patch")
        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of embed_is_patch. " f"Got type: {type(embed_is_patch)}"
            )

        num_patches = kwargs.pop("num_patches")
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_patches. " f"Got type: {type(num_patches)}"
            )

        return PixtralImagePixelInputs(
            type="pixel_values",
            images=flatten_bn(images),
            embed_is_patch=embed_is_patch,
            num_patches=num_patches,
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        pattern = MultiModalityDataPaddingPatternImageTokens(
            image_token_id=image_inputs.im_token_id
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def _get_mm_embeds(
        self,
        features: torch.Tensor,  # Shape: (num_patch, d)
        num_patches: torch.Tensor,  # Shape: (num_images,)
        embed_is_patch: torch.Tensor,  # Shape: (num_images, num_embeds)
    ) -> tuple[torch.Tensor, ...]:
        """Scatter the patch features into a contiguous tensor that corresponds
        to the embedding tokens defined by the multimodal processor.

        Mostly copied from `Molmo._get_mm_embeds`. See following fixme comment.
        """
        # Insert columns of nan values according to `embed_is_patch`. This work
        # ideally should be done in `_process_image_input`, but
        # `_process_image_input` is used in both V0 and V1 path. It's safer to
        # put the logic here.
        # FIXME: Move this logic to `_process_image_input` when v0 is
        # deprecated. Merge this function with `Molmo._get_mm_embeds`.
        num_patches_per_image: list[int] = num_patches.tolist()

        embeds_flat = features.new_full(
            (sum(num_patches_per_image), *features.shape[1:]),
            fill_value=torch.nan,
        )
        embeds_flat[embed_is_patch.view(-1)] = features

        return embeds_flat.split(num_patches_per_image)

    def get_multimodal_embeddings(
        self, image_inputs: ImageInputs
    ) -> Optional[MultiModalEmbeddings]:

        image_features = self._process_image_input(image_inputs)
        return image_features

    def get_image_feature(
        self,
        image_input: ImageInputs,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            image_sizes (`torch.Tensor`):
                Tensor containing the image sizes as returned by the processor.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # print(f"pixel_values shape: {image_input.pixel_values.shape}")
        pixel_values = image_input.pixel_values
        if (
            isinstance(pixel_values, list)
            and len(pixel_values) != 0
            and isinstance(pixel_values[0], np.ndarray)
        ):
            pixel_values = [torch.from_numpy(t) for t in pixel_values]

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values]

        assert (
            isinstance(pixel_values, list)
            and len(pixel_values) != 0
            and isinstance(pixel_values[0], torch.Tensor)
        )
        pixel_values_new = []
        for pixel_value in pixel_values:
            if pixel_value.dim() == 5:
                pixel_value = pixel_value.view((-1,) + pixel_value.shape[2:])

            if pixel_value.dim() == 4:
                # pixel_value = pixel_value.view((pixel_value.shape[0] * pixel_value.shape[1],) + pixel_value.shape[2:])
                ps = [p.squeeze(0) for p in pixel_value.split(1, dim=0)]

            if pixel_value.dim() == 3:
                ps = [pixel_value]
            # print(f"shape: {pixel_value.shape}")
            for p in ps:
                assert p.dim() == 3
                p = p.type(self.dtype).cuda()
                pixel_values_new += [p]

        pixel_values = pixel_values_new

        vision_feature_layer = self.config.vision_feature_layer
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=False,
            **kwargs,
        )

        print(f"vision tower out shape: ", image_outputs.hidden_states[0].shape)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)
        print(f"selected_image_featureshape: {selected_image_feature.shape}")

        print(f"image_outputs: {image_outputs=}")
        print(f"image_outputs: {image_outputs.hidden_states=}")
        # print(f"image_outputs shape: {image_outputs.shape}")

        image_features = self.multi_modal_projector(
            images=pixel_values, image_features=selected_image_feature.squeeze(0)
        )

        print(f"image_features: {image_features.shape=}")

        return image_features

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        """Run forward pass for mistral3."""
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            or not forward_batch.contains_image_inputs()
        ):
            inputs_embeds = self.get_input_embeddings()(input_ids)
        else:
            image_inputs = forward_batch.reduce_image_inputs()
            inputs_embeds = embed_image_inputs(
                image_input=image_inputs,
                input_ids=input_ids,
                input_embedding=self.get_input_embeddings(),
                image_embedding_func=self.get_image_feature,
            )
            forward_batch.image_inputs = None
        return self.language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
            get_embedding=get_embedding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # (".gate_up_proj", ".gate_proj", 0),
            # (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            # if "vision_tower" in name and "attention" in name:
            #     loaded = VisionAttention.load_weights(
            #         self, [(name, loaded_weight)]
            #     )
            #     if not loaded:
            #         loaded_weight.add(name)
            #     continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # if name.startswith("model.vision_tower") and name not in params_dict:
            #     continue
            # if self.config.tie_word_embeddings and "lm_head.weight" in name:
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            if "vision_tower" in name and "attention" in name:
                name = name.replace("o_proj.", "proj.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # if "vision_tower" in name:
                #     continue

                if "vision_tower" in name:
                    if (
                        "attention" in name
                        and self.vision_tower.transformer.layers[
                            0
                        ].attention.use_qkv_parallel
                    ):
                        pass
                    else:
                        continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            # print(f"loaded: {name}")
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            pass
            # raise RuntimeError(
            #     f"Some weights are not initialized from checkpoints: {unloaded_params}")
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        return loaded_params


EntryClass = Mistral3ForConditionalGeneration
AutoModel.register(Mistral3Config, Mistral3ForConditionalGeneration, exist_ok=True)
