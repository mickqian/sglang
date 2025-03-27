from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import List, Union

import torch
from PIL import Image

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.qwen2_5_omni import Qwen2_5OmniModel
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5OmniModel,
    ]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    def process_images_fast(self, images, input_text):
        if isinstance(images, list) and len(images) == 0:
            images = None
        processor = self._processor
        result = processor.__call__(
            text=[input_text],
            images=images,
            padding=True,
            return_tensors="pt",
            device="cuda",
        )

        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "image_grid_thw": getattr(result, "image_grid_thw", None),
            "second_per_grid_ts": getattr(result, "second_per_grid_ts", None),
            "video_grid_thws": getattr(result, "video_grid_thws", None),
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        start = time.time()

        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_mm_data(
            prompt=input_ids,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=image_token),
            max_req_input_len=max_req_input_len,
        )

        def smart_resize(
            height: int,
            width: int,
            factor: int = self.IMAGE_FACTOR,
            min_pixels: int = self.MIN_PIXELS,
            max_pixels: int = self.MAX_PIXELS,
        ) -> tuple[int, int]:
            """
            Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > self.MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
                )
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
            return h_bar, w_bar

        def resize_image(image, size_factor: int = self.IMAGE_FACTOR) -> Image.Image:
            width, height = image.size
            min_pixels = self.MIN_PIXELS
            max_pixels = self.MAX_PIXELS
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            return image

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        async def resize_image_async(image):
            return resize_image(image)

        resize_tasks = [resize_image_async(image) for image in base_output.images]
        resized_images = await asyncio.gather(*resize_tasks)

        res = self.process_images_fast(
            images=resized_images, input_text=base_output.input_text
        )

        items = []

        if len(res["pixel_values"]) != 0:
            image_grid_thws = torch.concat([res["image_grid_thw"]])
            item = MultimodalDataItem(
                pixel_values=res["pixel_values"],
                image_grid_thws=image_grid_thws,
                modality="image",
            )
            items += [item]

        if res["audio_features"] is not None and len(res["audio_features"]) != 0:
            # res["audio_features"] = [res["audio_features"]]
            audio_features = torch.concat([ret["audios_inputs"]])

            item = MultimodalDataItem(
                audio_features=[res["audio_features"]],
                audio_feature_lens=res["audio_feature_lens"],
                modality="audio",
            )
            items += [item]

        video_grid_thws = None

        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            # "items": items,
            "items": items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
        }
