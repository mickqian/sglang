"""
"""

import multiprocessing as mp
import unittest
from io import BytesIO
from typing import Type

import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree


def init_process_group():
    """Initialize the default process group"""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0,
        )


def run_test_class(test_class):
    """Run a single test class in its own process"""
    # Initialize test class
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if not result.wasSuccessful():
        raise RuntimeError(f"Tests failed for {test_class.__name__}")


# Test the logits output between HF and SGLang
class VisionLLMLogitsBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model_path = ""
        cls.chat_template = ""
        cls.processor = ""
        response = requests.get(cls.image_url)
        cls.main_image = Image.open(BytesIO(response.content))
        cls.model_runner = None

    @classmethod
    def tearDownClass(cls):
        if cls.model_runner is not None:
            if dist.is_initialized():
                dist.destroy_process_group()
            cls.model_runner = None

    def compare_outputs(self, sglang_output: torch.Tensor, hf_output: torch.Tensor):
        # Convert to float32 for numerical stability if needed
        hf = hf_output.float()
        sg = sglang_output.float()

        # Basic shape and dtype comparison
        print("\n=== Basic Properties ===")
        print(f"Shapes match: {hf.shape == sg.shape}")
        print(f"HF shape: {hf.shape}, SGLang shape: {sg.shape}")
        print(f"HF dtype: {hf.dtype}, SGLang dtype: {sg.dtype}")

        # Move tensors to CPU for numpy operations
        hf_np = hf.cpu().numpy()
        sg_np = sg.cpu().numpy()

        # Statistical metrics
        print("\n=== Statistical Metrics ===")
        print(f"Mean absolute difference: {torch.mean(torch.abs(hf - sg)).item():.6f}")
        print(f"Max absolute difference: {torch.max(torch.abs(hf - sg)).item():.6f}")
        print(f"Mean squared error: {torch.mean((hf - sg) ** 2).item():.6f}")
        print(
            f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}"
        )

        # Cosine similarity (across feature dimension)
        cos_sim = F.cosine_similarity(hf, sg)
        print(f"Mean cosine similarity: {torch.mean(cos_sim).item():.6f}")
        print(f"Min cosine similarity: {torch.min(cos_sim).item():.6f}")

        # Find largest absolute differences
        print("\n=== Largest Absolute Differences ===")
        diffs = torch.abs(hf - sg)
        flat_diffs = diffs.flatten()

        # Get indices of top 10 differences
        top_k = 10
        top_values, top_flat_indices = torch.topk(flat_diffs, top_k)

        # Convert flat indices to multidimensional indices
        top_indices = np.unravel_index(top_flat_indices.cpu().numpy(), diffs.shape)

        print(f"\nTop {top_k} largest absolute differences:")
        print(
            "Index".ljust(30)
            + "Difference".ljust(15)
            + "HF Value".ljust(15)
            + "SGLang Value"
        )
        print("-" * 75)

        for i in range(top_k):
            # Get the index tuple for this difference
            idx = tuple(dim[i] for dim in top_indices)
        diff_val = top_values[i].item()
        hf_val = hf[idx].item()
        sg_val = sg[idx].item()

        # Format the index tuple and values
        idx_str = str(idx)
        print(f"{idx_str:<30}{diff_val:<15.6f}{hf_val:<15.6f}{sg_val:.6f}")

        np.testing.assert_allclose(hf_np, sg_np)

    def get_processor_output(self):
        json_str = f"""
        {{
  "model": "{self.model_path}",
  "messages": [
    {{
      "role": "user",
      "content": [
        {{
          "type": "image_url",
          "image_url": {{
            "url": "{self.image_url}"
          }}
        }},
        {{
          "type": "text",
          "text": "Whats in this picture?"
        }}
      ]
    }}
  ]
}}
        """

        req = ChatCompletionRequest.model_validate_json(json_str)

        conv = generate_chat_conv(req, template_name=self.chat_template)

        text = conv.get_prompt()

        # Process inputs using processor
        # FIXME: the formal arguments may differ
        inputs = self.processor(
            text=[text],
            images=[self.main_image],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def get_sglang_model(self):
        self.model_runner = ModelRunner(
            model_config=ModelConfig(self.model_path, model_override_args="{}"),
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=12435,
            server_args=ServerArgs(
                model_path=self.model_path,
                disable_cuda_graph=True,
            ),
        )
        return self.model_runner.model


class TestMiniCPMVLogits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-V-2_6"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )

    async def test_vlm_embedding_output(self):
        """
        Compares the embedding output of vlm
        """
        inputs = self.get_processor_output()

        with torch.no_grad():
            # hf
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            (hf_output, _) = self.hf_model.get_vlm_embedding(
                model_inputs,
            )
            hf_output = hf_output.squeeze(0)

            # sglang
            model = self.get_sglang_model()
            input_ids = inputs["input_ids"].to(self.device).flatten()
            sglang_output = embed_mm_inputs(
                mm_input=MultimodalInputs(
                    pixel_values=inputs["pixel_values"][0],
                    tgt_sizes=inputs["tgt_sizes"][0],
                ),
                input_ids=input_ids,
                input_embedding=model.get_input_embeddings(),
                mm_data_embedding_func=model.get_image_features,
                placeholder_token_ids=[
                    self.processor.tokenizer.unk_token_id,
                ],
            )

        self.compare_outputs(sglang_output, hf_output)


class TestMiniCPMOLogits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-o-2_6"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmo"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )

    async def test_encode_output(self):
        inputs = self.get_processor_output()

        with torch.no_grad():
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            (hf_output, _) = self.hf_model.get_vlm_embedding(
                model_inputs,
            )
            hf_output = hf_output.squeeze(0)

        with torch.no_grad():
            model = self.get_sglang_model()
            input_ids = inputs["input_ids"].to(self.device).flatten()
            image_inputs = model.get_image_pixel_inputs(
                input_ids=input_ids,
                **{
                    "pixel_values": [inputs["pixel_values"]],
                    "tgt_sizes": [inputs["tgt_sizes"]],
                    "im_start_id": self.tokenizer.im_start_id,
                    "im_end_id": self.tokenizer.im_end_id,
                    "slice_start_id": self.tokenizer.slice_start_id,
                    "slice_end_id": self.tokenizer.slice_end_id,
                },
            )
            (sglang_output, _) = model.get_vlm_embedding(
                input_ids=input_ids, image_inputs=image_inputs
            )

        self.compare_outputs(sglang_output, hf_output)


def find_vision_llm_test_classes() -> list[Type[VisionLLMLogitsBase]]:
    """Automatically find all test classes inheriting from VisionLLMLogitsBase"""
    test_classes = []
    for name, obj in globals().items():
        if (
            isinstance(obj, type)
            and issubclass(obj, VisionLLMLogitsBase)
            and obj != VisionLLMLogitsBase
        ):
            test_classes.append(obj)
    return test_classes


if __name__ == "__main__":

    # TODO: Use standard unittest
    mp.set_start_method("spawn", force=True)

    # Run each test class in sequence in its own process
    test_classes = find_vision_llm_test_classes()

    for test_class in test_classes:
        print(f"\nRunning tests for {test_class.__name__}")
        process = mp.Process(target=run_test_class, args=(test_class,))
        process.start()
        process.join()
        kill_process_tree(process.pid)

        if process.exitcode != 0:
            print(
                f"Test class {test_class.__name__} failed with exit code {process.exitcode}"
            )
            exit(1)
        print(f"Completed tests for {test_class.__name__}\n")
