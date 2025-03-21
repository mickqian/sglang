from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Literal, Mapping, NamedTuple, Protocol

import cv2
import PIL
import torch
from mistral_common.protocol.instruct.messages import ImageChunk
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder, is_cv2_installed
from PIL import Image
from pydantic import BaseModel, ConfigDict
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BatchFeature,
    PixtralVisionConfig,
    PretrainedConfig,
    ProcessorMixin,
)
from transformers.image_utils import is_valid_image, load_image
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from transformers.processing_utils import (
    ProcessingKwargs,
    _validate_images_text_input_order,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from typing_extensions import TypeVar, Unpack

from sglang.srt.configs.utils import (
    register_image_processor,
    register_processor,
    remove_if_exists,
)
from sglang.srt.utils import print_warning_once
from sglang.utils import logger


def rename_class(new_name):
    def decorator(cls):
        return type(new_name, cls.__bases__, dict(cls.__dict__))

    return decorator


import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import pad, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging
from transformers.utils.import_utils import requires_backends

logger = logging.get_logger(__name__)

if is_vision_available():
    import PIL


@dataclass
class SpecialImageIDs:
    img: int
    img_break: int
    img_end: int

    @staticmethod
    def from_tokenizer(tokenizer) -> "SpecialImageIDs":
        return SpecialImageIDs(
            img=tokenizer.get_control_token(SpecialTokens.img.value),
            img_break=tokenizer.get_control_token(SpecialTokens.img_break.value),
            img_end=tokenizer.get_control_token(SpecialTokens.img_end.value),
        )


class ImageSize(NamedTuple):
    width: int
    height: int


_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)
_T = TypeVar("_T")

_C = TypeVar("_C", bound=PretrainedConfig, default=PretrainedConfig)


#
# @dataclass(frozen=True)
# class InputContext:
#     """
#     Contains information about the model which may be used to
#     modify the inputs.
#     """
#
#     model_config: "ModelConfig"
#     """The configuration of the model."""
#
#     def get_hf_config(
#         self,
#         typ: Union[type[_C], tuple[type[_C], ...]] = PretrainedConfig,
#         /,
#     ) -> _C:
#         """
#         Get the HuggingFace configuration
#         (:class:`transformers.PretrainedConfig`) of the model,
#         additionally checking its type.
#
#         Raises:
#             TypeError: If the configuration is not of the specified type.
#         """
#         hf_config = self.model_config.hf_config
#         if not isinstance(hf_config, typ):
#             raise TypeError(
#                 "Invalid type of HuggingFace config. "
#                 f"Expected type: {typ}, but "
#                 f"found type: {type(hf_config)}"
#             )
#
#         return hf_config
#
#     def get_hf_image_processor_config(self) -> dict[str, Any]:
#         """
#         Get the HuggingFace image processor configuration of the model.
#         """
#         return self.model_config.hf_image_processor_config
#
#     def get_mm_config(self):
#         """
#         Get the multimodal config of the model.
#
#         Raises:
#             RuntimeError: If the model is not a multimodal model.
#         """
#         mm_config = self.model_config.multimodal_config
#         if mm_config is None:
#             raise RuntimeError("Not a multimodal model")
#
#         return mm_config
#
#     def get_hf_processor(
#         self,
#         typ: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
#         /,
#         **kwargs: object,
#     ) -> _P:
#         """
#         Get the HuggingFace processor
#         (:class:`transformers.ProcessorMixin`) of the model,
#         additionally checking its type.
#
#         Raises:
#             TypeError: If the processor is not of the specified type.
#         """
#         return cached_processor_from_config(
#             self.model_config,
#             processor_cls=typ,
#             **kwargs,
#         )
#
#     def init_processor(
#         self,
#         typ: type[_T],
#         /,
#         **kwargs: object,
#     ) -> _T:
#         """
#         Initialize a HuggingFace-like processor class, merging the
#         keyword arguments with those in the model's configuration.
#         """
#         base_kwargs = self.model_config.mm_processor_kwargs
#         if base_kwargs is None:
#             base_kwargs = {}
#
#         merged_kwargs = {**base_kwargs, **kwargs}
#
#         return typ(**merged_kwargs)


def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Optional[Mapping[str, object]],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
    """
    if not overrides:
        return {}

    # Drop any mm_processor_kwargs provided by the user that
    # are not kwargs, unless it can fit it var_kwargs param
    filtered_overrides = {
        kwarg_name: val
        for kwarg_name, val in overrides.items()
        # if supports_kw(callable,
        #                kwarg_name,
        #                requires_kw_only=requires_kw_only,
        #                allow_var_kwargs=allow_var_kwargs
        #                )
    }

    # If anything is dropped, log a warning
    dropped_keys = overrides.keys()
    # dropped_keys = overrides.keys() - filtered_overrides.keys()
    if dropped_keys:
        if requires_kw_only:
            logger.warning(
                "The following intended overrides are not keyword-only args "
                "and will be dropped: %s",
                dropped_keys,
            )
        else:
            logger.warning(
                "The following intended overrides are not keyword args "
                "and will be dropped: %s",
                dropped_keys,
            )

    return filtered_overrides


def resolve_mm_processor_kwargs(
    init_kwargs: Optional[Mapping[str, object]],
    inference_kwargs: Optional[Mapping[str, object]],
    callable: Callable[..., object],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """Applies filtering to eliminate invalid mm_processor_kwargs, i.e.,
    those who are not explicit keywords to the given callable (of one is
    given; otherwise no filtering is done), then merges the kwarg dicts,
    giving priority to inference_kwargs if there are any collisions.

    In the case that no kwarg overrides are provided, returns an empty
    dict so that it can still be kwarg expanded into the callable later on.

    If allow_var_kwargs=True, allows for things that can be expanded into
    kwargs as long as they aren't naming collision for var_kwargs or potential
    positional arguments.
    """
    # Filter inference time multimodal processor kwargs provided
    runtime_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=inference_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Filter init time multimodal processor kwargs provided
    init_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=init_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Merge the final processor kwargs, prioritizing inference
    # time values over the initialization time values.
    mm_processor_kwargs = {**init_mm_kwargs, **runtime_mm_kwargs}
    return mm_processor_kwargs


#
# @dataclass(frozen=True)
# class InputProcessingContext(InputContext):
#     tokenizer: MistralTokenizer
#     """The tokenizer used to tokenize the inputs."""
#
#     def get_hf_processor(
#         self,
#         typ: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
#         /,
#         **kwargs: object,
#     ) -> _P:
#         return super().get_hf_processor(
#             typ,
#             tokenizer=self.tokenizer,
#             **kwargs,
#         )
#
#     def call_hf_processor(
#         self,
#         hf_processor: ProcessorMixin,
#         data: Mapping[str, object],
#         kwargs: Mapping[str, object] = {},
#     ) -> BatchFeature:
#         """
#         Call :code:`hf_processor` on the prompt :code:`data`
#         (text, image, audio...) with configurable options :code:`kwargs`.
#         """
#         assert callable(hf_processor)
#
#         base_kwargs = self.model_config.mm_processor_kwargs
#         if base_kwargs is None:
#             base_kwargs = {}
#
#         merged_kwargs = resolve_mm_processor_kwargs(
#             base_kwargs,
#             kwargs,
#             hf_processor,
#             requires_kw_only=False,
#             allow_var_kwargs=True,
#         )
#
#         try:
#             return hf_processor(**data, **merged_kwargs, return_tensors="pt")
#         except Exception as exc:
#             msg = (
#                 f"Failed to apply {type(hf_processor).__name__} "
#                 f"on data={data} with kwargs={merged_kwargs}"
#             )
#
#             raise RuntimeError(msg) from exc
#
#
# class BaseProcessingInfo:
#     """Base class to provide the information necessary for data processing."""
#
#     def __init__(self, ctx: InputProcessingContext) -> None:
#         super().__init__()
#
#         self.ctx = ctx
#
#     @property
#     def model_id(self) -> str:
#         return self.ctx.model_config.model_path
#
#     def get_tokenizer(self):
#         return self.ctx.tokenizer
#
#     def get_hf_config(self) -> PretrainedConfig:
#         return self.ctx.get_hf_config()
#
#     def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
#         """
#         Subclasses can override this method to handle
#         specific kwargs from model config or user inputs.
#         """
#         return self.ctx.get_hf_processor(**kwargs)
#
#     @abstractmethod
#     def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
#         """
#         Return the maximum supported number of items for each modality.
#
#         A value of `None` means unlimited number of items.
#
#         Omitting a modality from the returned dictionary means that
#         it is not supported at all.
#         """
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_mm_max_tokens_per_item(
#         self,
#         seq_len: int,
#         mm_counts: Mapping[str, int],
#     ) -> Mapping[str, int]:
#         """
#         Get the maximum possible number of tokens per data item
#         for each modality.
#
#         The dictionary returned by this method should have the same
#         keys as that returned by :meth:`get_supported_mm_limits`.
#         """
#         raise NotImplementedError
#

NestedTensors = Union[
    list["NestedTensors"], list[torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]
]


# A drop-in replacement for hf-compatible processor
# transformers support for mistralai/Mistral-Small-3.1-24B-Instruct-2503 was **not throughly tested**
# https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#transformers-untested
@rename_class("PixtralProcessor")
class PixtralProcessorAdapter(ProcessorMixin):
    """
    Provide a HF-compatible interface for
    :class:`mistral_common.tokens.tokenizers.multimodal.ImageEncoder`.
    """

    model_type = "pixtral"

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "spatial_merge_size",
        "image_token",
        "image_break_token",
        "image_end_token",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.mm_encoder = ImageEncoder(
            PretrainedConfig(
                image_patch_size=14, max_image_size=1540, spatial_merge_size=2
            ),
            special_ids=SpecialImageIDs(img=10, img_break=12, img_end=13),
        )
        self.image_token = image_token

    @property
    def image_processor(self) -> ImageEncoder:
        # ImageEncoder
        # image_encoder = self.tokenizer.instruct.mm_encoder
        image_encoder = self.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @cached_property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if not images:
            input_ids = self.tokenizer(text).input_ids

            return {"input_ids": torch.tensor(input_ids)}

        # input_ids = self.tokenizer(text).input_ids
        print(f"input_ids")

        # Allow dummy text, which is used for profiling as well as token inputs
        # if any(len(t) > 0 for t in text):
        #     raise ValueError(
        #         "You've passed text inputs instead of token inputs. "
        #         "Make sure to process your input via `mistral_common`'s "
        #         "tokenizer or pass a chat completion request. "
        #     )
        # prompt_strings = text
        # if images is not None:
        #     # Replace the image token with the expanded image token sequence
        #     image_sizes = iter(image_inputs["image_sizes"])
        #     prompt_strings = []
        #     replace_strings = []
        #
        #     for sample in text:
        #         while self.image_token in sample:
        #             image_size = next(image_sizes)
        #             if isinstance(image_size, list):
        #                 image_size = image_size[0]
        #             height, width = image_size
        #
        #             num_height_tokens = height // (
        #                 self.patch_size * self.spatial_merge_size
        #             )
        #             num_width_tokens = width // (
        #                 self.patch_size * self.spatial_merge_size
        #             )
        #             replace_tokens = [
        #                                  [self.image_token] * num_width_tokens + [self.image_break_token]
        #                              ] * num_height_tokens
        #             # Flatten list
        #             replace_tokens = [
        #                 item for sublist in replace_tokens for item in sublist
        #             ]
        #             replace_tokens[-1] = self.image_end_token
        #             replace_str = "".join(replace_tokens)
        #             replace_strings.append(replace_str)
        #             sample = sample.replace(self.image_token, "<placeholder>", 1)
        #
        #         while "<placeholder>" in sample:
        #             replace_str = replace_strings.pop(0)
        #             sample = sample.replace("<placeholder>", replace_str, 1)
        #         prompt_strings.append(sample)

        image_token_id = self.image_token_id

        images_processed = list[torch.Tensor]()
        images_tokenss = list[torch.Tensor]()
        images_embed_is_patch = list[torch.Tensor]()
        images_num_patches = list[int]()

        for image in images:
            image_inputs = self.image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)
            image_tokens = torch.tensor(image_inputs.tokens)
            print(f"image_tokens: {image_tokens.shape}")

            images_processed.append(image_processed)
            images_tokenss.append(image_tokens)
            images_embed_is_patch.append(image_tokens == image_token_id)
            images_num_patches.append(len(image_tokens))
        print(f"text {len(text)}")
        return {
            "input_ids": torch.cat(images_tokenss)[None].expand(len(text), -1),
            "images": images_processed,
            "embed_is_patch": images_embed_is_patch,
            "num_patches": torch.tensor(images_num_patches),
        }


# class PixtralProcessingInfo(BaseProcessingInfo):
#
#     def get_tokenizer(self) -> MistralTokenizer:
#         # tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
#         tokenizer = get_tokenizer(
#             tokenizer_name=self.ctx.model_config.model_path,
#             torch_dtype=torch.dtype,
#             # FIXME
#             trust_remote_code=True
#         )
#         if not isinstance(tokenizer, MistralTokenizer):
#             raise ValueError("This model requires `--tokenizer-mode mistral`")
#
#         return tokenizer
#
#     def get_hf_processor(self) -> PixtralProcessorAdapter:
#         return PixtralProcessorAdapter(self.get_tokenizer())
#
#     def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
#         return {"image": None}
#
#     def get_mm_max_tokens_per_item(
#         self,
#         seq_len: int,
#         mm_counts: Mapping[str, int],
#     ) -> Mapping[str, int]:
#         return {"image": self.get_max_image_tokens()}
#
#     def get_vision_config(
#         self,
#         processor: Optional[PixtralProcessorAdapter] = None,
#     ):
#         if processor is None:
#             processor = self.get_hf_processor()
#
#         return PixtralVisionConfig(
#             image_size=processor.image_size,
#             patch_size=processor.patch_size,
#         )
#
#     def get_num_image_tokens(
#         self,
#         *,
#         image_width: int,
#         image_height: int,
#         processor: Optional[PixtralProcessorAdapter] = None,
#     ) -> int:
#         if processor is None:
#             processor = self.get_hf_processor()
#
#         ncols, nrows = processor.image_processor._image_to_num_tokens(
#             Image.new("RGB", (image_width, image_height)))
#
#         return (ncols + 1) * nrows
#
#     def get_image_size_with_most_features(self) -> ImageSize:
#         image_processor = self.get_hf_processor().image_processor
#         max_image_size = image_processor.mm_config.max_image_size
#
#         return ImageSize(width=max_image_size, height=max_image_size)
#
#     def get_max_image_tokens(self) -> int:
#         target_width, target_height = self.get_image_size_with_most_features()
#
#         return self.get_num_image_tokens(
#             image_width=target_width,
#             image_height=target_height,
#         )


# class VisionConfig(PretrainedConfig):
#     def __init__(
#         self,
#         adapter_bias=False,
#         add_pre_mm_projector_layer_norm=True,
#         hidden_size=1024,
#         image_break_token_id=12,
#         image_end_token_id=13,
#         image_size=1540,
#         image_token_id=10,
#         intermediate_size=4096,
#         max_image_size=1540,
#         mm_projector_id="patch_merge",
#         num_attention_heads=16,
#         num_channels=3,
#         num_hidden_layers=24,
#         patch_size=14,
#         rope_theta=1e3,
#         spatial_merge_size=2,
#         transformers_version="4.48.3",
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.adapter_bias = adapter_bias
#         self.add_pre_mm_projector_layer_norm = add_pre_mm_projector_layer_norm
#         self.hidden_size = hidden_size
#         self.image_break_token_id = image_break_token_id
#         self.image_end_token_id = image_end_token_id
#         self.image_size = image_size
#         self.image_token_id = image_token_id
#         self.intermediate_size = intermediate_size
#         self.max_image_size = max_image_size
#         self.mm_projector_id = mm_projector_id
#         self.num_attention_heads = num_attention_heads
#         self.num_channels = num_channels
#         self.num_hidden_layers = num_hidden_layers
#         self.patch_size = patch_size
#         self.rope_theta = rope_theta
#         self.spatial_merge_size = spatial_merge_size
#         self.transformers_version = transformers_version
#
#
# # Wrapper class around PretrainedConfig to ensure compatability with mistral
# @dataclass
# class PixtralConfig(PretrainedConfig):
#     model_type = "pixtral"
#
#     def __init__(
#         self,
#         text_config: Optional[LlamaConfig] = None,
#         vision_config=None,
#         **kwargs,
#     ):
#         print(f"601 text_config: {text_config}")
#         print(f"602 vision_config: {vision_config}")
#         super().__init__(**kwargs)
#         if isinstance(text_config, dict):
#             self.text_config = LlamaConfig(**text_config)
#         else:
#             self.text_config = text_config or LlamaConfig()
#
#         if isinstance(vision_config, dict):
#             self.vision_config = VisionConfig(**vision_config)
#         else:
#
#             self.vision_config = vision_config or VisionConfig()
#
#         print(f"578 text_config: {self.text_config}")
#         print(f"602 vision_config: {self.vision_config}")
#         for key, value in kwargs.items():
#             setattr(self, key, value)


class Mistral3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mistral3ForConditionalGeneration`]. It is used to instantiate an
    Mistral3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `PixtralVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MistralConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 10):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_layer (`Union[int, List[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        multimodal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multimodal projector.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The downsampling factor for the spatial merge operation.

    ```"""

    model_type = "mistral3"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        self.vision_feature_layer = vision_feature_layer
        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"]
                if "model_type" in vision_config
                else "pixtral"
            )
            config_cls = CONFIG_MAPPING[vision_config["model_type"]]
            vision_config = config_cls(**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["pixtral"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=1540,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                head_dim=64,
                hidden_act="gelu",
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "mistral"
            )
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["mistral"](
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=5120,
                initializer_range=0.02,
                intermediate_size=32768,
                max_position_embeddings=131072,
                model_type="mistral",
                num_attention_heads=32,
                num_hidden_layers=40,
                num_key_value_heads=8,
                rms_norm_eps=1e-05,
                rope_theta=1000000000.0,
                sliding_window=None,
                use_cache=True,
                vocab_size=131072,
            )

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size


class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


# Copied from https://github.com/huggingface/transformers/blob/706703bba6c920b10aa7e7ee8163b06a8a03c450/src/transformers/models/pixtral/image_processing_pixtral.py
class PixtralProcessor(ProcessorMixin):
    r"""
    Constructs a Pixtral processor which wraps a Pixtral image processor and a Pixtral tokenizer into a single processor.

    [`PixtralProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~PixtralProcessor.__call__`] and [`~PixtralProcessor.decode`] for more information.

    Args:
        image_processor ([`PixtralImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*, defaults to 16):
            Patch size from the vision tower.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The downsampling factor for the spatial merge operation.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"[IMG]"`):
            Special token used to denote image location.
        image_break_token (`str`, *optional*, defaults to `"[IMG_BREAK]"`):
            Special token used to denote the end of a line of pixels in an image.
        image_end_token (`str`, *optional*, defaults to `"[IMG_END]"`):
            Special token used to denote the end of an image input.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "spatial_merge_size",
        "image_token",
        "image_break_token",
        "image_end_token",
    ]
    image_processor_class = "PixtralImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ):
        self.image_processor = image_processor
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        # print("bbbbbbbbbb")
        # print(f"{type(image_processor)}")
        # transformers support for mistralai/Mistral-Small-3.1-24B-Instruct-2503 was **not throughly tested**
        # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#transformers-untested
        use_hf_version_image_processor = True
        # if use_hf_version_image_processor:
        #     ...
        # else:
        #     image_processor = PixtralImageProcessorFromMistralCommon()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.mm_encoder = ImageEncoder(
            PretrainedConfig(
                image_patch_size=14, max_image_size=1540, spatial_merge_size=2
            ),
            special_ids=SpecialImageIDs(img=10, img_break=12, img_end=13),
        )
        self.image_token = image_token

    @property
    def mistral_image_processor(self) -> ImageEncoder:
        # ImageEncoder
        # image_encoder = self.tokenizer.instruct.mm_encoder
        image_encoder = self.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @cached_property
    def image_break_id(self) -> int:
        return self.mistral_image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self.mistral_image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self.mistral_image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.mistral_image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.mistral_image_processor.mm_config.image_patch_size

    def _image_to_num_tokens(
        self, size: Tuple[Union[int, float], Union[int, float]]
    ) -> Tuple[int, int]:
        mm_encoder = self.mm_encoder
        w, h = size
        ratio = max(
            h / mm_encoder.mm_config.max_image_size,
            w / mm_encoder.mm_config.max_image_size,
        )
        if ratio > 1:
            w = round(w / ratio)
            h = round(h / ratio)

        width_tokens = (w - 1) // (
            mm_encoder.mm_config.image_patch_size
            * mm_encoder.mm_config.spatial_merge_size
        ) + 1
        height_tokens = (h - 1) // (
            mm_encoder.mm_config.image_patch_size
            * mm_encoder.mm_config.spatial_merge_size
        ) + 1

        return width_tokens, height_tokens

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[PixtralProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        # check if images and text inputs are reversed for BC
        print(f"text0 {text}")
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PixtralProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_token_id = self.image_token_id

        images_processed = list[torch.Tensor]()
        images_tokens = list[torch.Tensor]()
        images_embed_is_patch = list[torch.Tensor]()
        images_num_embeds = list[int]()
        print(f"images len: {len(images)}")
        for image in images:
            image_inputs = self.mistral_image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)
            image_tokens = torch.tensor(image_inputs.tokens)

            images_processed.append(image_processed)
            images_tokens.append(image_tokens)
            images_embed_is_patch.append(image_tokens == image_token_id)
            images_num_embeds.append(len(image_tokens))

        for image in images_processed:
            print(f"image shape: {image.shape}")

        for img in images_tokens:
            print(f"images_tokens shape: {img.shape}")
            print(f"images_tokens: {img}")
        if images is not None:
            if is_image_or_image_url(images):
                images = [images]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                pass
            elif (
                isinstance(images, (list, tuple))
                and isinstance(images[0], (list, tuple))
                and is_image_or_image_url(images[0][0])
            ):
                images = [image for sublist in images for image in sublist]
            else:
                raise ValueError(
                    "Invalid input images. Please provide a single image, a list of images, or a list of lists of images."
                )
            images = [load_image(im) if isinstance(im, str) else im for im in images]
            image_inputs = self.image_processor(
                images, patch_size=self.patch_size, **output_kwargs["images_kwargs"]
            )

            # image_inputs["image_sizes"] = flatten_nested_list(
            #     image_inputs["image_sizes"]
            # )
            # image_inputs["pixel_values"] = flatten_nested_list(
            #     image_inputs["pixel_values"]
            # )

        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            image_sizes = iter(image_inputs["image_sizes"])
            prompt_strings = []
            replace_strings = []

            for sample in text:
                while self.image_token in sample:
                    image_size = next(image_sizes)
                    if isinstance(image_size, list):
                        image_size = image_size[0]
                    height, width = image_size

                    num_height_tokens_0 = height // (
                        self.patch_size * self.spatial_merge_size
                    )
                    num_width_tokens_0 = width // (
                        self.patch_size * self.spatial_merge_size
                    )

                    (
                        num_width_tokens,
                        num_height_tokens,
                    ) = self._image_to_num_tokens(image_size)
                    # if num_height_tokens != num_height_tokens_0:
                    #     print("1111111111")
                    # if num_width_tokens_0 != num_width_tokens:
                    #     print("2222222")
                    # print(f"{image_size=}")
                    # print(f"{num_width_tokens=}")
                    # print(f"{num_height_tokens=}")

                    replace_tokens = [
                        [self.image_token] * num_width_tokens + [self.image_break_token]
                    ] * num_height_tokens
                    # Flatten list
                    replace_tokens = [
                        item for sublist in replace_tokens for item in sublist
                    ]
                    replace_tokens[-1] = self.image_end_token
                    replace_str = "".join(replace_tokens)
                    # print(f"{image_size=}")
                    # print(f"replace_str: {replace_str}")
                    # print(f"{self.patch_size=}")
                    # print(f"{self.spatial_merge_size=}")
                    replace_strings.append(replace_str)
                    # print(f"before: {sample}")
                    sample = sample.replace(self.image_token, "<placeholder>", 1)
                    # print(f"after: {sample}")

                while "<placeholder>" in sample:
                    replace_str = replace_strings.pop(0)
                    print(f"replacing <placeholder> with: {replace_str}")
                    print(f"before: {sample}")
                    sample = sample.replace("<placeholder>", replace_str, 1)
                    print(f"after: {sample}")

                prompt_strings.append(sample)

        # print(f"prompt_strings: {prompt_strings}")
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        # pixel_values = torch.stack(image_inputs["pixel_values"], dim=0)
        # image_inputs["pixel_values"] = pixel_values
        # input_ids = torch.cat(images_tokens)[None].expand([''], -1),
        image_inputs["pixel_values"] = images_processed
        # image_inputs["input_ids"] = input_ids
        return BatchFeature(
            data={**text_inputs, **image_inputs},
            # tensor_type=output_kwargs["common_kwargs"]["return_tensors"],
            tensor_type=None,
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.mistral_image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# Adapted from function in image_transforms.py to ensure any transparent pixels are converted to white.
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    requires_backends(convert_to_rgb, ["vision"])

    if not isinstance(image, PIL.Image.Image):
        return image

    if image.mode == "RGB":
        return image

    # First we convert to RGBA to set background to white.
    image = image.convert("RGBA")

    # Create a new image with a white background.
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB")
    return new_image


def _num_image_tokens(image_size: Tuple[int, int], patch_size: Tuple[int, int]) -> int:
    """
    Calculate the number of image tokens given the image size and patch size.

    Args:
        image_size (`Tuple[int, int]`):
            The size of the image as `(height, width)`.
        patch_size (`Tuple[int, int]`):
            The patch size as `(height, width)`.

    Returns:
        `int`: The number of image tokens.
    """
    height, width = image_size
    patch_height, patch_width = (
        patch_size
        if isinstance(patch_size, (tuple, list))
        else (patch_size, patch_size)
    )
    num_width_tokens = (width - 1) // patch_width + 1
    num_height_tokens = (height - 1) // patch_height + 1
    return num_height_tokens, num_width_tokens


def get_resize_output_image_size(
    input_image: ImageInput,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    patch_size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`ImageInput`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            Max image size an input image can be. Must be a dictionary with the key "longest_edge".
        patch_size (`int` or `Tuple[int, int]`):
            The patch_size as `(height, width)` to use for resizing the image. If patch_size is an integer, `(patch_size, patch_size)`
            will be used
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    max_height, max_width = size if isinstance(size, (tuple, list)) else (size, size)
    patch_height, patch_width = (
        patch_size
        if isinstance(patch_size, (tuple, list))
        else (patch_size, patch_size)
    )
    height, width = get_image_size(input_image, input_data_format)

    ratio = max(height / max_height, width / max_width)

    if ratio > 1:
        # Original implementation uses `round` which utilises bankers rounding, which can lead to surprising results
        # Here we use floor to ensure the image is always smaller than the given "longest_edge"
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))

    num_height_tokens, num_width_tokens = _num_image_tokens(
        (height, width), (patch_height, patch_width)
    )
    return num_height_tokens * patch_height, num_width_tokens * patch_width


@rename_class("PixtralImageProcessorFast")
class PixtralImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Pixtral image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the maximum dimension of either the height or width dimension of the image. Used to control how
            images are resized. If either the height or width are greater than `size["longest_edge"]` then both the height and width are rescaled by `height / ratio`, `width /ratio` where `ratio = max(height / longest_edge, width / longest_edge)`
        patch_size (`Dict[str, int]` *optional*, defaults to `{"height": 16, "width": 16}`):
            Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        patch_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        print("Using customed sglang ImageProcessor")

        size = size if size is not None else {"longest_edge": 1024}
        patch_size = (
            patch_size if patch_size is not None else {"height": 16, "width": 16}
        )
        patch_size = get_size_dict(patch_size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean
            if image_mean is not None
            else [0.48145466, 0.4578275, 0.40821073]
        )
        self.image_std = (
            image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        )
        self.do_convert_rgb = do_convert_rgb
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "patch_size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        patch_size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dict containing the longest possible edge of the image.
            patch_size (`Dict[str, int]`):
                Patch size used to calculate the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if "longest_edge" in size:
            size = (size["longest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "size must contain either 'longest_edge' or 'height' and 'width'."
            )

        if "height" in patch_size and "width" in patch_size:
            patch_size = (patch_size["height"], patch_size["width"])
        else:
            raise ValueError(
                "patch_size must contain either 'shortest_edge' or 'height' and 'width'."
            )

        output_size = get_resize_output_image_size(
            image,
            size=size,
            patch_size=patch_size,
            input_data_format=input_data_format,
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        image_sizes: List[List[int]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `height`, `width`, `channels`)
            image_sizes (`List[List[int]]`):
                A list of sizes for each image in `pixel_values` in (height, width) format.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.
        Returns:
            List[`np.ndarray`]: The padded images.
        """

        max_shape = (
            max([size[0] for size in image_sizes]),
            max([size[1] for size in image_sizes]),
        )
        pixel_values = [
            pad(
                image,
                padding=((0, max_shape[0] - size[0]), (0, max_shape[1] - size[1])),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image, size in zip(pixel_values, image_sizes)
        ]
        return pixel_values

    @property
    def image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @cached_property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        patch_size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Describes the maximum input dimensions to the model.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Patch size in the model. Used to calculate the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        patch_size = patch_size if patch_size is not None else self.patch_size
        patch_size = get_size_dict(patch_size, default_to_square=True)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self._valid_processor_keys,
        )

        images = make_list_of_images(images)

        if not valid_images(images[0]):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        batch_images = []
        batch_image_sizes = []
        for image in images:
            if do_resize:
                image = self.resize(
                    image=image,
                    size=size,
                    patch_size=patch_size,
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(
                    image=image,
                    scale=rescale_factor,
                    input_data_format=input_data_format,
                )

            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )

            batch_images.append(image)
            batch_image_sizes.append(get_image_size(image, data_format))
        #
        # pixel_values = self._pad_for_batching(
        #     pixel_values=batch_images,
        #     image_sizes=batch_image_sizes,
        #     input_data_format=data_format,
        #     data_format=data_format,
        # )

        image_token_id = self.image_token_id

        images_processed = list[torch.Tensor]()
        images_tokens = list[torch.Tensor]()
        images_embed_is_patch = list[torch.Tensor]()
        images_num_embeds = list[int]()
        for image in images:
            image_inputs = self.image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)
            image_tokens = torch.tensor(image_inputs.tokens)

            images_processed.append(image_processed)
            images_tokens.append(image_tokens)
            images_embed_is_patch.append(image_tokens == image_token_id)
            images_num_embeds.append(len(image_tokens))

        print(f"111111using pixtral image processor: {images_processed[0].shape}")
        return BatchFeature(
            data={"pixel_values": images_processed, "image_sizes": batch_image_sizes},
            tensor_type=return_tensors,
        )


class MistralBase(BaseModel):
    """
    Base class for all Mistral Pydantic models.
    """

    model_config = ConfigDict(
        extra="forbid", validate_default=True, use_enum_values=True
    )


class ImageURL(MistralBase):
    url: str
    detail: Optional[str] = None


class ChunkTypes(str, Enum):
    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(MistralBase):
    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]


class ImageURLChunk(BaseContentChunk):
    """
    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: Union[ImageURL, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url


class ImageURLChunk(BaseContentChunk):
    """
    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: Union[ImageURL, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url


@dataclass
class ImageEncoding:
    tokens: List[int]
    image: np.ndarray


class MultiModalEncoder(Protocol):
    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        """
        Encode the given content.

        Args:
            content (ChunkContent): The content to be encoded.

        Returns:
            ImageEncoding: The encoded image content.
        """
        ...

    @property
    def image_token(self) -> int: ...


# Copied from mistral_common.tokens.tokenizers.multimodal
class ImageEncoder(MultiModalEncoder):
    def __init__(
        self, mm_config: PretrainedConfig, special_ids: SpecialImageIDs
    ) -> None:
        self.mm_config = mm_config
        self.special_ids = special_ids

    def _image_to_num_tokens(self, img: Image.Image) -> Tuple[int, int]:
        w: Union[int, float]
        h: Union[int, float]

        w, h = img.size
        ratio = max(
            h / self.mm_config.max_image_size, w / self.mm_config.max_image_size
        )
        if ratio > 1:
            w = round(w / ratio)
            h = round(h / ratio)

        width_tokens = (w - 1) // (
            self.mm_config.image_patch_size * self.mm_config.spatial_merge_size
        ) + 1
        height_tokens = (h - 1) // (
            self.mm_config.image_patch_size * self.mm_config.spatial_merge_size
        ) + 1

        return width_tokens, height_tokens

    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        """
        Converts ImageChunks to numpy image arrays and image token ids

        Args:
        image (ImageChunk, ImageURLChunk): ImageChunk to be converted

        Returns:
        ImageEncoding containing image token ids and processed image in numpy format
        """
        image = image_from_chunk(content)
        w, h = self._image_to_num_tokens(image)
        assert w > 0
        assert h > 0
        image_tokens = ([self.special_ids.img] * w + [self.special_ids.img_break]) * h
        image_tokens[-1] = self.special_ids.img_end
        new_image_size = (
            w * self.mm_config.image_patch_size * self.mm_config.spatial_merge_size,
            h * self.mm_config.image_patch_size * self.mm_config.spatial_merge_size,
        )
        processed_image = transform_image(image, new_image_size)
        return ImageEncoding(tokens=image_tokens, image=processed_image)

    @property
    def image_token(self) -> int:
        return self.special_ids.img


class SpecialTokens(str, Enum):
    bos = "<s>"
    eos = "</s>"
    begin_inst = "[INST]"
    end_inst = "[/INST]"
    begin_tools = "[AVAILABLE_TOOLS]"
    end_tools = "[/AVAILABLE_TOOLS]"
    begin_tool_results = "[TOOL_RESULTS]"
    end_tool_results = "[/TOOL_RESULTS]"
    tool_calls = "[TOOL_CALLS]"
    img = "[IMG]"
    img_break = "[IMG_BREAK]"
    img_end = "[IMG_END]"
    prefix = "[PREFIX]"
    middle = "[MIDDLE]"
    suffix = "[SUFFIX]"
    begin_system = "[SYSTEM_PROMPT]"
    end_system = "[/SYSTEM_PROMPT]"
    begin_tool_content = "[TOOL_CONTENT]"


def image_from_chunk(chunk: Union[ImageURLChunk, ImageChunk]) -> Image:
    """Get a serializable image from a chunk."""
    if isinstance(chunk, ImageChunk):
        return chunk.image
    if chunk.get_url().startswith("data:image"):
        data = chunk.get_url().split(",")[1]
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    if chunk.get_url().startswith("http"):
        return download_image(chunk.get_url())

    raise RuntimeError(f"Unsupported image url scheme {chunk.get_url()}")


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert a PIL image to RGB.
    We ensure transparent background becomes white.
    """
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    white_bg: Image.Image = Image.new("RGBA", image.size, "WHITE")
    white_bg.paste(image, (0, 0), image)
    return white_bg.convert("RGB")


DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


def normalize(
    np_image: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """
    Normalize a tensor image with mean and standard deviation.

    Args:
    image (np.ndarray): Image to be normalized.
    mean (tuple[float, float, float]): Mean for each channel.
    std (tuple[float, float, float]): Standard deviation for each channel.

    Returns:
    np.ndarray: Normalized image with shape (C, H, W).
    """
    np_image = np_image / 255.0

    assert len(np_image.shape) == 3, f"{np_image.shape=}"
    assert (
        np_image.shape[2] == len(mean) == len(std)
    ), f"{np_image.shape=}, {mean=}, {std=}"

    np_image = (np_image - mean) / std

    return np_image.transpose(2, 0, 1)


def transform_image(image: Image.Image, new_size: Tuple[int, int]) -> np.ndarray:
    if not is_cv2_installed():
        raise ImportError(
            "OpenCV is required for this function. Install it with 'pip install mistral_common[opencv]'"
        )

    np_image = cv2.resize(
        np.array(_convert_to_rgb(image), dtype=np.float32),
        new_size,
        interpolation=cv2.INTER_CUBIC,
    )
    return normalize(np_image, DATASET_MEAN, DATASET_STD)


#
# class PixtralImageProcessorFromMistralCommon(BaseImageProcessor):
#     r"""
#     Constructs a Pixtral image processor.
#
#     Args:
#         do_resize (`bool`, *optional*, defaults to `True`):
#             Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
#             `do_resize` in the `preprocess` method.
#         size (`Dict[str, int]` *optional*, defaults to `{"longest_edge": 1024}`):
#             Size of the maximum dimension of either the height or width dimension of the image. Used to control how
#             images are resized. If either the height or width are greater than `size["longest_edge"]` then both the height and width are rescaled by `height / ratio`, `width /ratio` where `ratio = max(height / longest_edge, width / longest_edge)`
#         patch_size (`Dict[str, int]` *optional*, defaults to `{"height": 16, "width": 16}`):
#             Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.
#         resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
#             Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
#         do_rescale (`bool`, *optional*, defaults to `True`):
#             Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
#             the `preprocess` method.
#         rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
#             Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
#             method.
#         do_normalize (`bool`, *optional*, defaults to `True`):
#             Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
#         image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
#             Mean to use if normalizing the image. This is a float or list of floats the length of the number of
#             channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
#         image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
#             Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
#             number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
#             Can be overridden by the `image_std` parameter in the `preprocess` method.
#         do_convert_rgb (`bool`, *optional*, defaults to `True`):
#             Whether to convert the image to RGB.
#     """
#
#     model_input_names = ["pixel_values"]
#
#     def __init__(
#         self,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         # TODO: remove the hard code
#         self.mm_encoder = ImageEncoder(
#             PretrainedConfig(
#                 image_patch_size=14,
#                 max_image_size=1540,
#                 spatial_merge_size=2
#             ),
#             special_ids=SpecialImageIDs(img=10, img_break=12, img_end=13)
#         )
#
#     def preprocess(
#         self,
#         images: ImageInput,
#         do_resize: bool = None,
#         size: Dict[str, int] = None,
#         patch_size: Dict[str, int] = None,
#         resample: PILImageResampling = None,
#         do_rescale: bool = None,
#         rescale_factor: float = None,
#         do_normalize: bool = None,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         do_convert_rgb: bool = None,
#         return_tensors: Optional[Union[str, TensorType]] = None,
#         data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ) -> PIL.Image.Image:
#         image = image_from_chunk(ImageChunk())
#         mm_encoder = self.mm_encoder
#         w, h = mm_encoder._image_to_num_tokens(image)
#         assert w > 0
#         assert h > 0
#         image_tokens = ([mm_encoder.special_ids.img] * w + [mm_encoder.special_ids.img_break]) * h
#         image_tokens[-1] = mm_encoder.special_ids.img_end
#         new_image_size = (
#             w * mm_encoder.mm_config.image_patch_size * self.mm_config.spatial_merge_size,
#             h * mm_encoder.mm_config.image_patch_size * self.mm_config.spatial_merge_size,
#         )
#         processed_image = transform_image(image, new_image_size)
#         image_inputs = {
#             "image_sizes": new_image_size,
#             "pixel_values": processed_image,
#         }
#         return BatchFeature(
#             data={**text_inputs, **image_inputs},
#             tensor_type=output_kwargs["common_kwargs"]["return_tensors"],
#         )
#         return ImageEncoding(tokens=image_tokens, image=processed_image)
#


# the transformers compatible with sglang is <= 4.50, which has obsolete version of PixtralProcessor
# FIXME: deregister `pixtral` from transformers to enable customized processor
register_image_processor(Mistral3Config, PixtralImageProcessor)
register_image_processor(PixtralVisionConfig, PixtralImageProcessor)

# transformers support for mistralai/Mistral-Small-3.1-24B-Instruct-2503 was **not throughly tested**
# https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#transformers-untested

remove_if_exists(PROCESSOR_MAPPING_NAMES, PixtralVisionConfig.model_type)
remove_if_exists(IMAGE_PROCESSOR_MAPPING_NAMES, PixtralVisionConfig.model_type)

register_processor(Mistral3Config, PixtralProcessor)
# register_processor(PixtralVisionConfig, processor_cls)
# register_image_processor(PixtralVisionConfig, PixtralImageProcessor)
