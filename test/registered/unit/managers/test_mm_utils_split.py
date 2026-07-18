"""Unit tests for ``get_new_expanded_mm_items`` per-image splitting.

This is the load-bearing behavioral path for multi-image requests: a bundled
``MultimodalDataItem`` (one item carrying N image offsets + a concatenated
feature) must be split back into N per-image items so RadixAttention can cache
each image independently and chunked-prefill can encode them one at a time.

The MoonViT-style models (e.g. nvidia/LocateAnything-3B) carry their per-image
grids under ``image_grid_hws`` rather than ``image_grid_thw``; the splitter must
recognize both keys, fall back cleanly when no usable grid is present, and not
mis-split a degenerate flat grid. No server / GPU / weight loading involved.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.managers.mm_utils import (
    PerImageRequestInfo,
    _assemble_per_image_chunk,
    _batch_encode_per_image_misses,
    _get_chunked_prefill_embedding,
    _mm_features_consumed_by_forward,
    get_new_expanded_mm_items,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bundled_item(grid_key=None, grid=None, feature_len=10, num_images=2):
    """A bundled IMAGE item: `num_images` offsets, one concatenated feature."""
    model_specific_data = {}
    if grid_key is not None:
        model_specific_data[grid_key] = grid
    # Distinct per-row values so slice boundaries are checkable.
    feature = torch.arange(feature_len * 3, dtype=torch.float32).reshape(feature_len, 3)
    offsets = [(0, 5), (5, feature_len)][:num_images]
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class TestGetNewExpandedMMItems(CustomTestCase):
    def test_cross_request_image_misses_share_one_encoder_call(self):
        first = _bundled_item(grid_key="image_grid_hws", grid=[[1, 1]], num_images=1)
        second = _bundled_item(grid_key="image_grid_hws", grid=[[1, 1]], num_images=1)
        first.hash = 11
        second.hash = 22
        requests = [
            PerImageRequestInfo(
                req_idx=0,
                items=[first],
                items_offset=[(0, 5)],
                extend_prefix_len=0,
                extend_seq_len=6,
            ),
            PerImageRequestInfo(
                req_idx=1,
                items=[second],
                items_offset=[(0, 5)],
                extend_prefix_len=0,
                extend_seq_len=6,
            ),
        ]
        cache = SimpleNamespace(get_single=lambda _hash: None, set=lambda *_args: None)
        calls = []

        def encode(items):
            calls.append(items)
            return torch.cat(
                [torch.full((6, 2), float(item.hash)) for item in items], dim=0
            )

        with patch("sglang.srt.managers.mm_utils.embedding_cache", cache), patch(
            "sglang.srt.managers.mm_utils._can_skip_pre_embed_feature_move",
            return_value=True,
        ):
            hash_to_embedding = _batch_encode_per_image_misses(
                encode, requests, torch.device("cpu")
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual([item.hash for item in calls[0]], [11, 22])
        first_chunk = _assemble_per_image_chunk(requests[0], hash_to_embedding)
        second_chunk = _assemble_per_image_chunk(requests[1], hash_to_embedding)
        self.assertTrue(torch.equal(first_chunk, torch.full((6, 2), 11.0)))
        self.assertTrue(torch.equal(second_chunk, torch.full((6, 2), 22.0)))

    def test_mm_features_consumed_by_forward_tracks_chunk_boundaries(self):
        item = _bundled_item(grid_key="image_grid_hws", grid=[[1, 1]], num_images=1)
        mm_inputs = SimpleNamespace(mm_items=[item])

        self.assertTrue(_mm_features_consumed_by_forward([mm_inputs], [0], [6]))
        self.assertTrue(_mm_features_consumed_by_forward([mm_inputs], [6], [0]))
        self.assertFalse(_mm_features_consumed_by_forward([mm_inputs], [0], [3]))

    def test_prefix_covered_item_acknowledges_deferred_ipc_feature(self):
        item = _bundled_item(grid_key="image_grid_hws", grid=[[1, 1]], num_images=1)
        with patch(
            "sglang.srt.managers.mm_utils._acknowledge_deferred_cuda_ipc_cache_hits"
        ) as acknowledge:
            embedding, input_ids = _get_chunked_prefill_embedding(
                lambda _: torch.empty(0, 4),
                [item],
                [0, 1],
                [10],
                [0],
                [[(0, 5)]],
                torch.tensor([1, 2]),
            )

        self.assertIsNone(embedding)
        self.assertEqual(input_ids.tolist(), [1, 2])
        acknowledge.assert_called_once_with([item])

    def test_image_grid_hws_splits_per_image(self):
        # grid rows [[2,3],[4,1]] -> prod = [6, 4] patches -> feature_len 10.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=[[2, 3], [4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertEqual([len(o.offsets) for o in out], [1, 1])
        self.assertEqual(out[0].offsets, [(0, 5)])
        self.assertEqual(out[1].offsets, [(5, 10)])
        # Feature sliced 0:6 and 6:10 along dim-0.
        self.assertEqual(out[0].feature.shape[0], 6)
        self.assertEqual(out[1].feature.shape[0], 4)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))
        # Split items must re-hash (pad value is recomputed per image).
        self.assertTrue(all(o.hash is None for o in out))

    def test_image_grid_hws_tensor_splits_per_image(self):
        # Same as above but the grid arrives as a rank-2 tensor (HF emits these).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([[2, 3], [4, 1]], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_image_grid_thw_still_splits(self):
        # The pre-existing image_grid_thw path must keep working:
        # [[1,2,3],[1,4,1]] -> [6,4].
        item = _bundled_item(
            grid_key="image_grid_thw",
            grid=[[1, 2, 3], [1, 4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_missing_grid_falls_back_to_simple_split(self):
        # No grid, but feature dim-0 == num offsets -> simple per-row split.
        item = _bundled_item(grid_key=None, feature_len=2, num_images=2)
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:1]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[1:2]))

    def test_flat_1d_grid_does_not_mis_split(self):
        # A flat 1-D grid (`tensor([2, 2])`) has length == num_items so it passes
        # the length check, but prod(dim=-1) would collapse it to a scalar and
        # corrupt the slice boundaries. The rank-2 guard must reject it. With
        # feature_len != num_items, the simple-split fallback also declines, so
        # the bundled item is passed through unchanged (never mis-sliced).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([2, 2], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_numpy_grid_splits_per_image(self):
        # image_grid_hws can arrive as a numpy array from the HF image processor.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=np.array([[2, 3], [4, 1]], dtype=np.int64),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_non_bundled_item_passes_through(self):
        # A single-image item (one offset) is not bundled and is returned as-is.
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.arange(18, dtype=torch.float32).reshape(6, 3),
            model_specific_data={"image_grid_hws": [[2, 3]]},
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)


if __name__ == "__main__":
    unittest.main()
