import asyncio
from typing import List, Union

from sglang.srt.managers.image_processors.base_image_processor import (
    BaseImageProcessor as SGLangBaseImageProcessor,
)
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
    stack_nested_list,
)
from sglang.srt.models.mistral3 import Mistral3ForConditionalGeneration


class PixtralProcessor(SGLangBaseImageProcessor):
    def __init__(self, hf_config, server_args, processor):
        super().__init__(hf_config, server_args, processor)
        self.image_token = processor.image_token
        self.image_token_id = hf_config.image_token_index

    @staticmethod
    def _process_single_image(input_text, images):
        processor = get_global_processor()
        print(f"processor: {type(processor)}")

        result = processor.__call__(
            text=input_text, prompt=input_text, images=images, return_tensors="pt"
        )
        return {
            "input_ids": result["input_ids"],
            "pixel_values": result["pixel_values"],
            "image_sizes": result["image_sizes"],
        }

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

        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": stack_nested_list(res["pixel_values"]),
            "image_hashes": base_out.image_hashes,
            "im_token_id": self.image_token_id,
            "image_sizes": res["image_sizes"],
        }


ImageProcessorMapping = {Mistral3ForConditionalGeneration: PixtralProcessor}
