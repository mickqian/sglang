"""Scheduling helpers for chunked multimodal prefill.

The multimodal manager owns transport, padding, and embedding integration.  The
per-image scheduling path is kept here so that its cache and batching policy can
evolve independently from those concerns.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import is_hip

DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:
    """Return whether the model batches feature materialization internally."""
    owner = getattr(data_embedding_func, "__self__", None)
    if owner is None:
        return False
    if getattr(data_embedding_func, "__name__", None) not in (
        "get_image_feature",
        "get_video_feature",
    ):
        return False
    return owner.__class__.__name__ in {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
    }


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking)."""
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


def _acknowledge_deferred_cuda_ipc_cache_hits(
    items: List[MultimodalDataItem],
) -> None:
    """Release lazy CUDA-IPC slices when a cached embedding skips ViT."""
    parallel = get_parallel()
    if parallel.attn_tp_rank != 0:
        return
    server_args = get_server_args()
    consumer_count = max(getattr(server_args, "tp_size", parallel.attn_tp_size), 1)
    for item in items:
        item.acknowledge_deferred_cuda_ipc_feature(consumer_count)


@dataclass
class PerImageRequestInfo:
    req_idx: int
    items: List[MultimodalDataItem]
    items_offset: List[Tuple[int, int]]
    extend_prefix_len: int
    extend_seq_len: int
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]] = field(
        default_factory=list
    )


def _iter_token_bounded_batches(
    items_with_tokens: List[Tuple[MultimodalDataItem, int]], max_tokens: int
) -> List[List[Tuple[MultimodalDataItem, int]]]:
    """Group image items without exceeding the configured ViT token budget.

    A single image is never split.  If it exceeds the budget it is encoded on
    its own, preserving correctness while preventing other images from joining
    an already-large forward.
    """
    if max_tokens <= 0:
        return [items_with_tokens]

    batches: List[List[Tuple[MultimodalDataItem, int]]] = []
    current: List[Tuple[MultimodalDataItem, int]] = []
    current_tokens = 0
    for item, token_count in items_with_tokens:
        if current and current_tokens + token_count > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((item, token_count))
        current_tokens += token_count
    if current:
        batches.append(current)
    return batches


def _encode_per_image_items(
    data_embedding_func: DataEmbeddingFunc,
    items_with_tokens: List[Tuple[MultimodalDataItem, int]],
    device: torch.device,
    embedding_cache: MultiModalStaticCache,
) -> Dict[int, torch.Tensor]:
    """Encode image misses in token-bounded cross-request batches."""
    hash_to_embedding: Dict[int, torch.Tensor] = {}
    max_tokens = envs.SGLANG_VIT_ENCODE_MAX_TOKENS.get()
    for batch in _iter_token_bounded_batches(items_with_tokens, max_tokens):
        items = [item for item, _ in batch]
        token_counts = [token_count for _, token_count in batch]
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(items, device)
        embedding = data_embedding_func(items)
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                "Per-image multimodal scheduling requires tensor embeddings, "
                f"got {type(embedding).__name__}"
            )
        embedding = embedding.reshape(-1, embedding.shape[-1])
        splits = torch.split(embedding, token_counts, dim=0)
        if len(splits) != len(items):
            raise RuntimeError(
                "ViT returned fewer per-image embeddings than requested: "
                f"{len(splits)} vs {len(items)}"
            )
        for item, item_embedding in zip(items, splits):
            embedding_cache.set(item.hash, EmbeddingResult(embedding=item_embedding))
            hash_to_embedding[item.hash] = item_embedding
    return hash_to_embedding


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
    embedding_cache: MultiModalStaticCache,
    get_embedding_chunk: Callable[..., Tuple[torch.Tensor, int, int]],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Fallback: encode all items at once, cache combined result, extract chunk."""
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        embedding_cache.set(embedding_items_hash, embedding_per_req)
    else:
        _acknowledge_deferred_cuda_ipc_cache_hits(embedding_items_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


def _batch_encode_per_image_misses(
    data_embedding_func: DataEmbeddingFunc,
    per_image_requests: List[PerImageRequestInfo],
    device: torch.device,
    embedding_cache: MultiModalStaticCache,
) -> Dict[int, torch.Tensor]:
    unique_misses: Dict[int, Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[int, torch.Tensor] = {}

    for req_info in per_image_requests:
        chunk_start = req_info.extend_prefix_len
        chunk_end = chunk_start + req_info.extend_seq_len
        if req_info.extend_seq_len > 0:
            req_info.overlapping = [
                (idx, item, start, end)
                for idx, (item, (start, end)) in enumerate(
                    zip(req_info.items, req_info.items_offset)
                )
                if end >= chunk_start and start < chunk_end
            ]

        for _, item, start, end in req_info.overlapping:
            if item.hash in hash_to_embedding:
                _acknowledge_deferred_cuda_ipc_cache_hits([item])
                continue

            cached = embedding_cache.get_single(item.hash)
            if cached is not None:
                hash_to_embedding[item.hash] = cached.embedding
                _acknowledge_deferred_cuda_ipc_cache_hits([item])
            elif item.hash not in unique_misses:
                unique_misses[item.hash] = (item, end - start + 1)
            else:
                _acknowledge_deferred_cuda_ipc_cache_hits([item])

    if unique_misses:
        hash_to_embedding.update(
            _encode_per_image_items(
                data_embedding_func,
                list(unique_misses.values()),
                device,
                embedding_cache,
            )
        )

    return hash_to_embedding


def _assemble_per_image_chunk(
    req_info: PerImageRequestInfo,
    hash_to_embedding: Dict[int, torch.Tensor],
) -> Optional[torch.Tensor]:
    if not req_info.overlapping:
        return None

    chunk_start = req_info.extend_prefix_len
    chunk_end = chunk_start + req_info.extend_seq_len
    chunk_slices = []
    for _, item, start, end in req_info.overlapping:
        embedding = hash_to_embedding[item.hash]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        chunk_slices.append(embedding[overlap_start - start : overlap_end - start + 1])
    return torch.cat(chunk_slices, dim=0)


def _get_chunked_embedding_by_item(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
    embedding_cache: MultiModalStaticCache,
) -> Optional[torch.Tensor]:
    """Encode only per-image items overlapping with the current chunk."""
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len
    if extend_seq_len <= 0:
        return None

    overlapping = []
    for idx, (item, offset) in enumerate(zip(embedding_items_per_req, items_offset)):
        start, end = offset
        if end >= chunk_start and start < chunk_end:
            overlapping.append((idx, item, start, end))
    if not overlapping:
        return None

    cached_embeddings = {}
    miss_items = []
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
            _acknowledge_deferred_cuda_ipc_cache_hits([item])
        else:
            miss_items.append((idx, item, start, end))

    if miss_items:
        encoded = _encode_per_image_items(
            data_embedding_func,
            [(item, end - start + 1) for _, item, start, end in miss_items],
            device,
            embedding_cache,
        )
        for idx, item, _, _ in miss_items:
            cached_embeddings[idx] = encoded[item.hash]

    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        chunk_slices.append(emb[overlap_start - start : overlap_end - start + 1])
    return torch.cat(chunk_slices, dim=0)


def get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
    embedding_cache: MultiModalStaticCache,
    get_embedding_chunk: Callable[..., Tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Encode and schedule chunked-prefill multimodal items by request."""
    device = input_ids.device
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    per_image_requests = []
    full_path_requests = []
    all_chunks: List[Tuple[int, torch.Tensor]] = []

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            _acknowledge_deferred_cuda_ipc_cache_hits(embedding_items_per_req)
            continue
        if extend_seq_len <= 0:
            continue

        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)
        req_info = PerImageRequestInfo(
            req_idx=i,
            items=embedding_items_per_req,
            items_offset=items_offset,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
        )

        if is_per_image:
            if is_hip():
                chunk_embedding = _get_chunked_embedding_by_item(
                    data_embedding_func,
                    embedding_items_per_req,
                    items_offset,
                    extend_prefix_len,
                    extend_seq_len,
                    device,
                    embedding_cache,
                )
                if chunk_embedding is not None:
                    all_chunks.append((i, chunk_embedding))
            else:
                per_image_requests.append(req_info)
        else:
            full_path_requests.append(req_info)

    hash_to_embedding = _batch_encode_per_image_misses(
        data_embedding_func, per_image_requests, device, embedding_cache
    )
    for req_info in per_image_requests:
        chunk_embedding = _assemble_per_image_chunk(req_info, hash_to_embedding)
        if chunk_embedding is not None:
            all_chunks.append((req_info.req_idx, chunk_embedding))

    for req_info in full_path_requests:
        chunk_embedding, input_ids = _get_chunked_embedding_full(
            data_embedding_func,
            req_info.items,
            req_info.items_offset,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
            input_ids,
            device,
            embedding_cache,
            get_embedding_chunk,
        )
        if chunk_embedding is not None:
            all_chunks.append((req_info.req_idx, chunk_embedding))

    if not all_chunks:
        return None, input_ids
    all_chunks.sort(key=lambda item: item[0])
    return torch.concat([chunk for _, chunk in all_chunks], dim=0), input_ids
