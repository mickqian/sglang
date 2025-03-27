import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import PIL
import transformers
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.utils import load_audio, load_image, logger

_global_processor = None


def get_global_processor():
    global _global_processor
    return _global_processor


def init_global_processor(server_args):
    """
    Init the global processor for multimodal models."""
    global _global_processor
    print("initializing global processor")
    transformers.logging.set_verbosity_error()
    _global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[PIL.Image]] = None

    # audios
    audios: Optional[list[np.ndarray]] = None

    def normalize(self):
        for field_name in ["image_sizes", "images", "audios"]:
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
            mp_context=mp.get_context("spawn"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        self.gpu_executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            mp_context=mp.get_context("spawn"),
            initargs=(
                # self,
                server_args,
            ),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

    # self._gpu_queue = asyncio.Queue(maxsize=1)
    # self._gpu_workers = int(os.environ.get("SGLANG_GPU_WORKERS", 2))
    # self._gpu_task_runner = None

    # self.executor = concurrent.futures.ProcessPoolExecutor(
    #     initializer=init_global_processor,
    #     # mp_context=mp.get_context("fork"),
    #     mp_context=mp.get_context("spawn"),
    #     initargs=(
    #         # self,
    #         server_args,
    #     ),
    #     max_workers=int(os.environ.get("SGLANG_CPU_COUNT", os.cpu_count())),
    # )

    def _build_processor(self, server_args):
        """Init the global processor for multi modal models."""
        from sglang.srt.hf_transformers_utils import get_processor

        return get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

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
        if not isinstance(image_data, list):
            return []
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

    @staticmethod
    def encode_video(video_path, frame_count_limit=None):
        if not os.path.exists(video_path):
            logger.error(f"Video {video_path} does not exist")
            return []

        if frame_count_limit == 0:
            return []

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_indices = [i for i in range(0, len(vr), sample_fps)]
        if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
            frame_indices = uniform_sample(frame_indices, frame_count_limit)

        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        return frames

    @staticmethod
    def _load_single_item(
        data, is_video, is_audio, frame_count_limit=None, discard_alpha_channel=True
    ):
        """Static method that can be pickled for multiprocessing"""
        try:
            if is_audio:
                return load_audio(data)
            elif is_video:
                path = data[len("video:") :]
                return BaseMultimodalProcessor.encode_video(path, frame_count_limit)
            else:
                img, _ = load_image(data)
                return img.convert("RGB") if discard_alpha_channel else img
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        discard_alpha_channel: bool = True,
    ):
        """
        load multimodal data parallelly
        """

        # TODO(mick): load from server_args, env, or sampling_params
        MAX_NUM_FRAMES = 30
        # estimated_frames_list = self.get_estimated_frames_list(image_data=image_data)
        # total_frame_count = sum(estimated_frames_list)
        # # a heuristic value, suggesting the maximum fraction of frames to embed from all visual inputs.
        # # e.g., 0.1 suggests that 1 frame out of 10 input frames should be used
        # scaling_factor = min(1.0, MAX_NUM_FRAMES / max(1, total_frame_count))
        #
        # assert len(image_data) == len(estimated_frames_list)
        # Submit all tasks
        futures = []
        task_info = []
        image_index, audio_index = 0, 0

        for text_part in text_parts:
            if text_part == multimodal_tokens.image_token:
                data = image_data[image_index]
                is_video = isinstance(data, str) and data.startswith("video:")
                # estimated_frames = estimated_frames_list[image_index]
                # frame_count_limit = max(1, int(estimated_frames * scaling_factor))
                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        is_video,
                        False,
                        None,
                        discard_alpha_channel,
                    )
                )
                task_info.append(("image", data, None))
                image_index += 1
            elif text_part == multimodal_tokens.audio_token:
                data = audio_data[audio_index]
                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        False,
                        True,
                        None,
                        discard_alpha_channel,
                    )
                )
                task_info.append(("audio", data, None))
                audio_index += 1

        return futures, task_info

    def load_mm_data(
        self,
        prompt: list[int],
        multimodal_tokens: MultimodalSpecialTokens,
        max_req_input_len: int,
        image_data: Optional[list] = None,
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
        # start = time.time()

        multimodal_tokens.convert_to_strs(self._processor)
        print(f"{multimodal_tokens=}")
        print(f"{prompt=}")

        assert isinstance(prompt, str)
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

        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            audio_data=audio_data,
            discard_alpha_channel=discard_alpha_channel,
        )
        # Process results
        image_sizes, images, audios = [], [], []
        new_text = ""
        task_ptr = 0

        for text_part in text_parts:
            if text_part in multimodal_tokens.collect():
                task_type, data, frame_limit = task_info[task_ptr]
                result = futures[task_ptr].result()
                task_ptr += 1

                if task_type == "image":
                    frames = [result] if not isinstance(result, list) else result
                    if frames:
                        image_sizes += frames[0].size * len(frames)
                        images += frames
                        new_text += multimodal_tokens.image_token * len(frames)
                elif task_type == "audio":
                    # audio
                    audios.append(result)
                    new_text += multimodal_tokens.audio_token
                # TODO: handle video
            else:
                new_text += text_part

        out = BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            input_text=new_text,
        )
        out.normalize()
        return out
