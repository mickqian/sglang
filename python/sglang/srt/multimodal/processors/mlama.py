from typing import List, Union

from sglang.srt.managers.mm_utils import MMDataPaddingStrategy
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self._processor.image_token,
            image_token_id=self._processor.image_token_id,
        ).build(_processor)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        strategy = BaseMultimodalProcessor.build_mm_padding_strategy(
            MMDataPaddingStrategy.Tokens
        )
        return strategy.pad_input_tokens(input_ids, mm_inputs, self.mm_tokens)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_out, self.mm_tokens
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.mm_tokens.image_token_id,
        }
