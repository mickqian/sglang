import asyncio
from typing import List, Union

from sglang.srt.managers.image_processors.base_image_processor import (
    BaseImageProcessor as SGLangBaseImageProcessor,
)
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.mistral3 import Mistral3ForConditionalGeneration


class PixtralProcessor(SGLangBaseImageProcessor):
    models = [Mistral3ForConditionalGeneration]

    def __init__(self, hf_config, server_args, processor):
        super().__init__(hf_config, server_args, processor)
        self.image_token = processor.image_token
        self.image_token_id = hf_config.image_token_index

    @staticmethod
    def _process_single_image(input_text, images) -> dict:
        processor = get_global_processor()
        result = processor(text=input_text, prompt=input_text, images=images)

        print(f"{result=}")
        for p in result["pixel_values"]:
            print(f"pixel values shape: {p.shape}")
        return result

    async def _process_images(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                PixtralProcessor._process_single_image,
                input_text,
                images,
            )
        else:
            image_inputs = self._processor(
                images=images, text=input_text, return_tensors="pt"
            )

        return image_inputs

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        print(f"image data: {image_data}")

        if not image_data:
            return None
        if input_ids is None:
            input_ids = []
        if not isinstance(input_ids, list):
            input_ids = [input_ids]
        if image_data is None:
            image_data = []
        if not isinstance(image_data, list):
            image_data = [image_data]
        base_out = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token=self.image_token,
            max_req_input_len=max_req_input_len,
        )
        images = base_out.all_frames
        res = await self._process_images(images=images, input_text=base_out.input_text)
        print(f"image res: {res}")
        print(f"image res: {base_out=}")
        print(f"input ids shape: ", res["input_ids"].numel())

        return {
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=res["pixel_values"],
                    image_sizes=res["image_sizes"],
                    modality=Modality.IMAGE,
                )
            ],
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": res["pixel_values"],
            "im_token_id": self.image_token_id,
        }
