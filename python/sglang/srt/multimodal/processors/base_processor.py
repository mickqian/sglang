import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.utils import encode_video, load_audio, load_image, logger


class MultimodalInputFormat(Enum):
    """Enum for different multimodal input formats."""

    RAW_IMAGES = "raw_images"
    PRECOMPUTED_FEATURES = "precomputed_features"
    PIXEL_VALUES = "pixel_values"
    AUDIO = "audio"


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[Union[Image.Image, dict]]] = None

    videos: Optional[list[torch.Tensor]] = None

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = None

    def normalize(self):
        for field_name in ["images", "audios", "videos"]:
            field = getattr(self, field_name, None)
            if field is not None and isinstance(field, list) and len(field) == 0:
                setattr(self, field_name, None)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[int, str, List[str]]] = None
    video_token: Optional[Union[int, str, List[str]]] = None
    audio_token: Optional[Union[int, str, List[str]]] = None

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

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    def __post_init__(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def collect(self) -> re.Pattern:
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        return re.compile(combined, flags)


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.arch = hf_config.architectures[0]
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
        if hasattr(processor, "image_processor") and isinstance(
            processor.image_processor, BaseImageProcessorFast
        ):
            kwargs["device"] = "cuda"
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        if "pixel_values" in result and isinstance(
            result["pixel_values"], torch.Tensor
        ):
            result["pixel_values"] = result["pixel_values"].to("cpu")
        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Lazy import because decord is not available on some arm platforms.
        from decord import VideoReader, cpu

        # Before processing inputs
        if not image_data or len(image_data) == 0:
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
    def _load_single_item(
        data, is_video, is_audio, frame_count_limit=None, discard_alpha_channel=True
    ):
        """Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if is_audio:
                return load_audio(data)
            elif is_video:
                path = data[len("video:") :]
                return encode_video(path, frame_count_limit)
            else:
                img, _ = load_image(data)
                return img.convert("RGB") if discard_alpha_channel else img
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: dict,
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[iter] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
    ):
        """
        load multimodal data parallelly using iterators.
        """
        futures = []
        task_info = []
        # Map token strings to Modality enum for cleaner logic
        token_to_modality = {
            multimodal_tokens.image_token: Modality.IMAGE,
            multimodal_tokens.video_token: Modality.VIDEO,
            multimodal_tokens.audio_token: Modality.AUDIO,
        }

        for text_part in text_parts:
            modality = token_to_modality.get(text_part)
            if modality is not None:
                data_iterator = data_iterators.get(modality)
                if data_iterator is None:
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(data_iterator)
                except StopIteration:
                    raise ValueError(
                        f"Mismatch: More '{text_part}' tokens found than corresponding data items provided."
                    )

                frame_count_limit = None
                if modality == Modality.IMAGE and image_estimated_frames_iter:
                    try:
                        estimated_frames = next(image_estimated_frames_iter)
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )

                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))

        # Check if any iterators still have data left (indicates fewer tokens than data)
        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                # This is expected, the iterator is correctly exhausted
                pass
            except (
                Exception
            ):  # Catch other potential errors from next() if iterators are complex
                pass

        return futures, task_info

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

        assert isinstance(prompt, str)

        if return_text:
            import re

            pattern = (
                "("
                + "|".join(re.escape(sep) for sep in multimodal_tokens.collect())
                + ")"
            )
            # split text into list of normal text and special tokens
            text_parts = re.split(pattern, prompt)

        # collect all data
        data_iterators = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)

        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
        )

        # Process results
        images, videos, audios = [], [], []
        new_text_parts = []
        task_ptr = 0
        multimodal_token_list = multimodal_tokens.collect()
        for text_part in text_parts:
            try:
                if text_part in multimodal_token_list:
                    modality, data, frame_limit = task_info[task_ptr]
                    result = futures[task_ptr].result()
                    task_ptr += 1

                    if modality == Modality.IMAGE:
                        frames = [result] if not isinstance(result, list) else result
                        if frames:
                            # only for minicpmv
                            images += frames
                            new_text_parts += [
                                multimodal_tokens.image_token * len(frames)
                            ]
                    elif modality == Modality.VIDEO:
                        # load as video
                        videos += [result]
                        new_text_parts += [text_part]
                    elif modality == Modality.AUDIO:
                        # audio
                        audios += [result]
                        new_text_parts += [text_part]
                else:
                    # normal text
                    new_text_parts += [text_part]

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
            input_text="".join(new_text_parts),
        )
        out.normalize()
        return out

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id

        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    @staticmethod
    def _extract_processor_features(
        items: List[dict], attr_name: str
    ) -> Optional[torch.Tensor]:
        """
        Helper function to concat extracted attributes from processor output.
        """
        values = [value for item in items if (value := item.get(attr_name)) is not None]
        return torch.cat(values) if values else None

    # When we assume that all the items have the same attributes
    def _extract_processor_features_from_all_attributes(
        self, items: List[dict]
    ) -> dict:
        values = {}
        # Verify all items have the same keys
        first_keys = set(items[0].keys())
        for item in items[1:]:
            if set(item.keys()) != first_keys:
                raise ValueError(
                    f"All items must have the same attributes. "
                    f"First item has {first_keys}, but found {set(item.keys())}"
                )

        # Process each attribute
        for k, v in items[0].items():
            if isinstance(v, list):
                values[k] = self._extract_processor_features(items, k)
            else:
                # Verify all items have the same value for non-list attributes
                for item in items[1:]:
                    if item[k] != v:
                        raise ValueError(
                            f"All items must have the same value for attribute {k}. "
                            f"First item has {v}, but found {item[k]}"
                        )
                values[k] = v
        return values

    def process_and_combine_mm_data(
        self, base_output: BaseMultiModalProcessorOutput
    ) -> Tuple[Optional[MultimodalDataItem], torch.Tensor]:
        """
        Process multimodal data and return the combined multimodal item and input_ids.
        Handles all three input formats at the same abstraction level.

        Returns:
            Tuple of (combined_mm_item, input_ids)
        """

        def tokenize_text(input_text: str) -> torch.Tensor:
            """Tokenize input text."""
            return self._processor.tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        def categorize_mm_inputs(mm_inputs: List) -> MultimodalInputFormat:
            """Categorize multimodal inputs and validate consistency."""
            try:
                has_image = False
                has_pixel_values = False
                has_precomputed_features = False
                has_audio = False

                for mm_input in mm_inputs:
                    if isinstance(mm_input, Image.Image):
                        has_image = True
                    elif isinstance(mm_input, np.ndarray):
                        has_audio = True
                    elif isinstance(mm_input, dict):
                        if mm_input.get("precomputed_features", None) is not None:
                            has_precomputed_features = True
                        elif mm_input.get("pixel_values", None) is not None:
                            has_pixel_values = True
                        else:
                            raise ValueError(
                                f"Invalid multimodal input: {mm_input}, expected dict with pixel_values or precomputed_features"
                            )
                    else:
                        raise ValueError(
                            f"Invalid multimodal input: {mm_input}, expected Image.Image or dict"
                        )

                # Validate format consistency
                format_count = sum(
                    [has_image, has_pixel_values, has_precomputed_features, has_audio]
                )
                if format_count > 1:
                    raise ValueError(
                        "Unsupported: mixture of multimodal input formats. "
                        f"Found formats: image={has_image}, pixel_values={has_pixel_values}, "
                        f"precomputed_features={has_precomputed_features}, audio={has_audio}"
                    )

                if has_image:
                    return MultimodalInputFormat.RAW_IMAGES
                elif has_precomputed_features:
                    return MultimodalInputFormat.PRECOMPUTED_FEATURES
                elif has_pixel_values:
                    return MultimodalInputFormat.PIXEL_VALUES
                elif has_audio:
                    return MultimodalInputFormat.AUDIO
                else:
                    raise ValueError("No valid multimodal input format found")
            except Exception as e:
                raise ValueError(f"Failed to categorize inputs: {e}")

        def process_raw_images(
            base_output: BaseMultiModalProcessorOutput,
        ) -> Tuple[MultimodalDataItem, torch.Tensor]:
            """Process raw Image.Image objects using transformers processor."""
            ret = self.process_mm_data(
                input_text=base_output.input_text,
                images=base_output.images,
            )
            combined_mm_item = MultimodalDataItem(modality=Modality.IMAGE)

            # Copy all fields from processor output except input_ids
            for key, value in ret.items():
                if key != "input_ids" and hasattr(combined_mm_item, key):
                    setattr(combined_mm_item, key, value)

            input_ids = ret["input_ids"].flatten()
            return combined_mm_item, input_ids

        def process_precomputed_features(
            base_output: BaseMultiModalProcessorOutput,
        ) -> Tuple[MultimodalDataItem, torch.Tensor]:
            """Process inputs with precomputed features."""
            combined_mm_item = MultimodalDataItem(modality=Modality.IMAGE)
            combined_mm_item.precomputed_features = self._extract_processor_features(
                base_output.images, "precomputed_features"
            )
            input_ids = tokenize_text(base_output.input_text)
            return combined_mm_item, input_ids

        def process_pixel_values(
            base_output: BaseMultiModalProcessorOutput,
        ) -> Tuple[MultimodalDataItem, torch.Tensor]:
            """Process inputs with pixel values."""
            values = self._extract_processor_features_from_all_attributes(
                base_output.images
            )
            combined_mm_item = MultimodalDataItem.from_dict(values)
            input_ids = tokenize_text(base_output.input_text)
            return combined_mm_item, input_ids

        def process_audio(
            base_output: BaseMultiModalProcessorOutput,
        ) -> Tuple[MultimodalDataItem, torch.Tensor]:
            """Process inputs with audio."""
            ret = self.process_mm_data(
                input_text=base_output.input_text,
                audio=base_output.audios,  # Note: "audio" is for gemma3n only
            )
            combined_mm_item = MultimodalDataItem(modality=Modality.AUDIO)
            for key, value in ret.items():
                if key != "input_ids" and hasattr(combined_mm_item, key):
                    setattr(combined_mm_item, key, value)
            input_ids = ret["input_ids"].flatten()
            return combined_mm_item, input_ids

        def finalize_mm_item(
            combined_mm_item: MultimodalDataItem, input_ids: torch.Tensor
        ) -> MultimodalDataItem:
            """Apply common post-processing to the multimodal item."""
            if combined_mm_item.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]:
                combined_mm_item.offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.IM_TOKEN_ID,
                )
            elif combined_mm_item.modality == Modality.AUDIO:
                combined_mm_item.audio_offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.AUDIO_TOKEN_ID,
                )
            elif combined_mm_item.modality == Modality.VIDEO:
                combined_mm_item.video_offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.VIDEO_TOKEN_ID,
                )
            else:
                raise ValueError(f"Unknown modality: {combined_mm_item.modality}")
            return combined_mm_item

        # Main logic - determine input type and handle text-only case
        mm_inputs = base_output.images or base_output.audios
        if not mm_inputs:
            input_ids = tokenize_text(base_output.input_text)
            return None, input_ids

        # Categorize input formats
        input_format = categorize_mm_inputs(mm_inputs)

        # Process based on format
        if input_format == MultimodalInputFormat.RAW_IMAGES:
            combined_mm_item, input_ids = process_raw_images(base_output)
        elif input_format == MultimodalInputFormat.PRECOMPUTED_FEATURES:
            combined_mm_item, input_ids = process_precomputed_features(base_output)
        elif input_format == MultimodalInputFormat.PIXEL_VALUES:
            combined_mm_item, input_ids = process_pixel_values(base_output)
        elif input_format == MultimodalInputFormat.AUDIO:
            combined_mm_item, input_ids = process_audio(base_output)
        else:
            raise ValueError(f"Unknown input format: {input_format}")

        # Finalize with common processing
        combined_mm_item = finalize_mm_item(combined_mm_item, input_ids)
        return combined_mm_item, input_ids
