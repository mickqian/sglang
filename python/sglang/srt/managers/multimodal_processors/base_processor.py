import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import PIL
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.utils import load_audio, load_image, load_video, logger


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[PIL.Image]] = None
    videos: Optional[list[VideoReader]] = None

    # audios
    audios: Optional[list[np.ndarray]] = None

    def normalize(self):
        for field_name in ["image_sizes", "images", "videos", "audios"]:
            field = getattr(self, field_name, None)
            if field is not None and isinstance(field, list) and len(field) == 0:
                setattr(self, field_name, None)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[int, str]] = None
    video_token: Optional[Union[int, str]] = None
    audio_token: Optional[Union[int, str]] = None

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        self.image_token = self.convert_to_str(self.image_token, processor)
        self.video_token = self.convert_to_str(self.video_token, processor)
        self.audio_token = self.convert_to_str(self.audio_token, processor)

    def collect(self) -> list[str]:
        return [
            token
            for token in [self.image_token, self.video_token, self.audio_token]
            if token
        ]


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """
        process multimodal data with transformers AutoProcessor
        """
        if images is not None:
            kwargs["images"] = images
        if videos is not None:
            kwargs["videos"] = videos
        if audios is not None:
            kwargs["audios"] = audios

        processor = self._processor
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        return result

    @abstractmethod
    async def process_mm_data_async(
        self, image_data, input_text, max_req_input_len, **kwargs
    ):
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Before processing inputs
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        max_req_input_len: int,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        multimodal_tokens.convert_to_strs(self._processor)

        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt
        if return_text:
            import re

            pattern = (
                "("
                + "|".join(re.escape(sep) for sep in multimodal_tokens.collect())
                + ")"
            )
            # split text into list of normal text and special tokens
            text_parts = re.split(pattern, prompt)

        image_index, video_index, audio_index = 0, 0, 0
        image_sizes, images, videos, audios = [], [], [], []
        new_text = ""
        print(f"{text_parts=}")
        print(f"{multimodal_tokens=}")
        for index, text_part in enumerate(text_parts):
            text = text_part
            try:
                if text_part == multimodal_tokens.image_token:
                    # image
                    raw_image, _size = load_image(image_data[image_index])
                    if discard_alpha_channel:
                        raw_image = raw_image.convert("RGB")
                    frames = [raw_image]
                    image_sizes += frames[0].size * len(frames)
                    images += frames
                    image_index += 1
                elif text_part == multimodal_tokens.video_token:
                    # load as video
                    video_file = video_data[video_index]
                    video = load_video(video_file)
                    videos += [video]
                    video_index += 1
                elif text_part == multimodal_tokens.audio_token:
                    # load as audio
                    audio_file = audio_data[audio_index]
                    audio = load_audio(audio_file)
                    audios += [audio]
                    audio_index += 1
                else:
                    # normal text
                    ...

                new_text += text

            except Exception as e:
                logger.error(
                    f"An exception occurred while loading multimodal data: {e}"
                )
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )

        out = BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text=new_text,
        )
        out.normalize()
        return out
