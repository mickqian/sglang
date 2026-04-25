import unittest

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    expand_request_outputs,
    normalize_output_seeds,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


class TestMultiOutputGrouping(unittest.TestCase):
    def test_normalize_output_seeds_from_int(self):
        self.assertEqual(
            normalize_output_seeds(10, num_outputs_per_prompt=3),
            [10, 11, 12],
        )

    def test_normalize_output_seeds_from_per_prompt_list(self):
        self.assertEqual(
            normalize_output_seeds([3, 5], num_outputs_per_prompt=2),
            [3, 5],
        )

    def test_normalize_output_seeds_from_total_list(self):
        self.assertEqual(
            normalize_output_seeds(
                [1, 2, 3, 4],
                num_outputs_per_prompt=2,
                num_prompts=2,
                prompt_index=1,
            ),
            [3, 4],
        )

    def test_normalize_output_seeds_rejects_mismatched_list(self):
        with self.assertRaisesRegex(ValueError, r"seed list length"):
            normalize_output_seeds(
                [1, 2, 3],
                num_outputs_per_prompt=2,
                num_prompts=2,
                prompt_index=0,
            )

    def test_expand_request_outputs_splits_seed_and_output_name(self):
        req = Req(
            sampling_params=SamplingParams(
                request_id="rid",
                prompt="p",
                output_path="/tmp",
                output_file_name="image.png",
                num_outputs_per_prompt=2,
                seed=[100, 101],
            )
        )

        outputs = expand_request_outputs(req)

        self.assertEqual([item.seed for item in outputs], [100, 101])
        self.assertEqual([item.num_outputs_per_prompt for item in outputs], [1, 1])
        self.assertEqual(
            [item.output_file_name for item in outputs],
            ["image_0.png", "image_1.png"],
        )
        self.assertEqual(
            [item.request_id for item in outputs],
            ["rid:0", "rid:1"],
        )

    def test_split_batched_latents_uses_original_batched_tensor(self):
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        src = Req(sampling_params=SamplingParams(prompt="p"))
        dst = Req(sampling_params=SamplingParams(prompt="p"))
        src.latents = torch.tensor([[[1.0]], [[2.0]]])
        src.latent_ids = torch.tensor([[[10.0]], [[20.0]]])

        stage._split_batched_latents(src, [src, dst])

        self.assertTrue(torch.equal(src.latents, torch.tensor([[[1.0]]])))
        self.assertTrue(torch.equal(dst.latents, torch.tensor([[[2.0]]])))
        self.assertTrue(torch.equal(src.latent_ids, torch.tensor([[[10.0]]])))
        self.assertTrue(torch.equal(dst.latent_ids, torch.tensor([[[20.0]]])))


if __name__ == "__main__":
    unittest.main()
