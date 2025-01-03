# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/deepseek-ai/DeepSeek-VL2
"""Inference-only DeepseekVL2 model."""
import math
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from deepseek_v2 import DeepseekV2ForCausalLM
from timm.layers import (
    AttentionPoolLatent,
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    resample_abs_pos_embed,
)
from timm.models._manipulate import checkpoint_seq, named_apply
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""The original timm.models.layers.weight_init.trunc_normal_ can not handle bfloat16 yet, here we first
    convert the tensor to float32, apply the trunc_normal_() in float32, and then convert it back to its orignal dtype.
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn
    from the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


# 对于inference sample也可以维护input_ids，反正最后不会用到
@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.LongTensor
    target_ids: torch.LongTensor
    images: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_spatial_crop: torch.LongTensor
    num_image_tokens: List[int]

    def __len__(self):
        return len(self.input_ids)


class ImageTransform(object):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [T.ToTensor()]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x


class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):

        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio

        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # must set this，padding side with make a difference in batch inference

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
        print(
            f"Add pad token = ['{pad_token}'] to the tokenizer\n"
            f"{pad_token}:{tokenizer.encode(pad_token, add_special_tokens=False)[0]}"
        )

        # add image token
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        print(
            f"Add image token = ['{image_token}'] to the tokenizer\n"
            f"{image_token}:{tokenizer.encode(image_token, add_special_tokens=False)[0]}"
        )

        # add five special tokens for grounding-related tasks
        # <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
        special_tokens = ["<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(
            f"Add grounding-related tokens = {special_tokens} to the tokenizer with input_ids\n"
            f"<|ref|>:{tokenizer.encode('<|ref|>', add_special_tokens=False)[0]}\n"
            f"<|/ref|>:{tokenizer.encode('<|/ref|>', add_special_tokens=False)[0]}\n"
            f"<|det|>:{tokenizer.encode('<|det|>', add_special_tokens=False)[0]}\n"
            f"<|/det|>:{tokenizer.encode('<|/det|>', add_special_tokens=False)[0]}\n"
            f"<|grounding|>:{tokenizer.encode('<|grounding|>', add_special_tokens=False)[0]}"
        )

        # add special tokens for SFT data
        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(
            f"Add chat tokens = {special_tokens} to the tokenizer with input_ids\n"
            f"<|User|>:{tokenizer.encode('<|User|>', add_special_tokens=False)[0]}\n"
            f"<|Assistant|>:{tokenizer.encode('<|Assistant|>', add_special_tokens=False)[0]}\n"
        )

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            tokenizer,
            **kwargs,
        )

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        return conv

    def format_messages(
        self,
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        Args:
            conversations (List[Dict]): A List of messages.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    def format_messages_v2(self, messages, pil_images, systems=None):
        """play the role of format_messages_v2 and get_images_info in the last version"""
        tokenized_data = []
        masked_tokenized_data = []  # labels
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []
        num_image_tokens = []

        image_index = 0

        conv = get_conv_template(self.sft_format)
        conv_system_message = conv.system_message

        for idx, message in enumerate(messages):
            if idx == 0:
                tokenized_data += [self.bos_id]
                masked_tokenized_data += [self.bos_id]
                images_seq_mask += [False]
                conv.system_message = conv_system_message
            else:
                conv.system_message = ""

            if message["role"] == conv.roles[0] or message["role"] == "user":
                conv.reset_message()
                conv.append_message(conv.roles[0], str(message["content"]).strip())
                conv.append_message(conv.roles[1], "")
                formatted_question = conv.get_prompt()
                tokenized_str, images, seq_mask, spatial_crop, n_image_tokens = (
                    self.tokenize_with_images(
                        formatted_question,
                        pil_images[
                            image_index : image_index
                            + formatted_question.count(self.image_token)
                        ],
                        bos=False,
                        eos=False,
                        cropping=len(pil_images) <= 2,
                    )
                )
                image_index += formatted_question.count(self.image_token)

                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                images_list += images
                images_seq_mask += seq_mask
                images_spatial_crop += spatial_crop
                num_image_tokens += n_image_tokens

            elif message["role"] == conv.roles[1] or message["role"] == "assistant":
                formatted_answer = message["content"].strip()
                assert (
                    formatted_answer.count(self.image_token) == 0
                ), f"there should be no {self.image_token} in the assistant's reply, but got {messages}"
                tokenized_str, images, seq_mask, spatial_crop, n_image_tokens = (
                    self.tokenize_with_images(
                        formatted_answer,
                        [],
                        bos=False,
                        eos=True,
                        cropping=len(pil_images) <= 2,
                    )
                )

                tokenized_data += tokenized_str
                masked_tokenized_data += tokenized_str
                images_seq_mask += seq_mask

            elif message["role"] == "system" or message["role"] == "deepseekapi-sys":
                # 如果message里面有system，那就只允许出现在message的第一句，同时conv原本的system就会失效
                assert (
                    idx == 0
                ), "system information should only exist in the begining of the conversation"
                formatted_system = message["content"].strip()
                tokenized_str = self.encode(formatted_system, bos=False, eos=False)
                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                seq_mask = [False] * len(tokenized_str)
                images_seq_mask += seq_mask

            else:
                assert False, f"Unknown role: {message['role']}"

        assert len(tokenized_data) == len(
            images_seq_mask
        ), f"format_messages_v2: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"
        assert len(images_spatial_crop) == len(
            num_image_tokens
        ), f"image number should be compatible"

        return (
            tokenized_data,
            masked_tokenized_data,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        )

    def format_prompts(
        self,
        prompts: str,
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to prompts.

        Args:
            prompts (str): the non-sft formatted prompt;
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], prompts.strip())
        conv.append_message(conv.roles[1], "")

        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            # apply sft format
            sft_format = self.format_messages(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=system_prompt,
            )
            (
                tokenized_str,
                masked_tokenized_str,
                images_list,
                images_seq_mask,
                images_spatial_crop,
                num_image_tokens,
            ) = self.format_messages_v2(conversations, images)
        else:
            if apply_sft_format:
                sft_format = self.format_prompts(
                    prompts=prompt,
                    sft_format=self.sft_format,
                    system_prompt=system_prompt,
                )
            else:
                sft_format = prompt
            (
                tokenized_str,
                images_list,
                images_seq_mask,
                images_spatial_crop,
                num_image_tokens,
            ) = self.tokenize_with_images(
                sft_format, images, bos=True, eos=True, cropping=len(images) <= 2
            )
            masked_tokenized_str = []
            for token_index in tokenized_str:
                if token_index != self.image_token_id:
                    masked_tokenized_str.append(token_index)
                else:
                    masked_tokenized_str.append(self.ignore_id)

        assert (
            len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)
        ), (
            f"tokenized_str's length {len(tokenized_str)}, input_ids' length {len(masked_tokenized_str)}, "
            f"imags_seq_mask's length {len(images_seq_mask)}, are not equal"
        )

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = (
            self.ignore_id
        )
        input_ids[input_ids < 0] = self.pad_id

        if inference_mode:
            # 去掉结尾的eos token
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            images = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        else:
            images = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            target_ids=target_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            num_image_tokens=num_image_tokens,
        )

        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        force_batchify: bool = True,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            force_batchify (bool): force batchify the inputs;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt,
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def tokenize_with_images(
        self,
        conversation: str,
        images: List[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        """Tokenize text with <image> tags."""
        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            """encode text_sep"""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(
                    image.size, self.candidate_resolutions
                )
            else:
                best_width, best_height = self.image_size, self.image_size
            # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

            """process the global view"""
            global_view = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(
                image,
                (best_width, best_height),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(
                            local_view.crop(
                                (j, i, j + self.image_size, i + self.image_size)
                            )
                        )
                    )

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = (
                best_width // self.image_size,
                best_height // self.image_size,
            )
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image = [self.image_token_id] * h * (w + 1)
            # add a seperator between global and local views
            tokenized_image += [self.image_token_id]
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image += (
                [self.image_token_id]
                * (num_height_tiles * h)
                * (num_width_tiles * w + 1)
            )

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))
            # print(width_crop_num, height_crop_num, len(tokenized_image)) # test the correctness of the number of image-related tokens

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        )

    def batchify(
        self,
        sample_list: List[VLChatProcessorOutput],
        padding: Literal["left", "right"] = "left",
    ) -> BatchCollateOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            sample_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.
            padding (str): The padding method. Defaults to "left".

        Returns:
            BatchCollateOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batched_sft_format = [sample.sft_format for sample in sample_list]
        batched_input_ids = [sample.input_ids for sample in sample_list]
        batched_images_seq_mask = [sample["images_seq_mask"] for sample in sample_list]
        seq_lens = [len(sample) for sample in sample_list]

        """padding input_ids and images_seq_mask"""
        if padding == "left":
            # the tokenizer is default to pad at left
            ## TODO, You're using a LlamaTokenizerFast tokenizer.
            #   Please note that with a fast tokenizer, using the `__call__` method is faster than
            #   using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
            padded_input_ids = self.tokenizer.pad({"input_ids": batched_input_ids})
            batched_input_ids = padded_input_ids["input_ids"]
            batched_images_seq_mask = self.tokenizer.pad(
                {"input_ids": batched_images_seq_mask}
            )["input_ids"]
            batched_images_seq_mask[batched_images_seq_mask == self.pad_id] = False
        else:
            batched_input_ids = pad_sequence(
                batched_input_ids, batch_first=True, padding_value=self.pad_id
            )
            batched_images_seq_mask = pad_sequence(
                batched_images_seq_mask, batch_first=True, padding_value=0
            )

        """padding images to max_patch_num"""
        max_n_patches = max(sample["images"].shape[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample["images"]
            n_pads = max_n_patches - images.shape[0]
            if n_pads > 0:
                pad_images = torch.zeros(
                    (n_pads, *images.shape[1:]), dtype=images.dtype
                )
                images = torch.cat([images, pad_images], dim=0)
            batched_images.append(images)
        batched_images = torch.stack(batched_images, dim=0)

        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(
            sample["images_spatial_crop"].shape[0] for sample in sample_list
        )
        batched_images_spatial_crop = []
        for sample in sample_list:
            images_spatial_crop = sample["images_spatial_crop"]
            n_pads = max_n_images - sample["images_spatial_crop"].shape[0]
            if n_pads > 0:
                pad_images_spatial_crop = torch.full(
                    (n_pads, 2), 0, dtype=images_spatial_crop.dtype
                )
                images_spatial_crop = torch.cat(
                    [images_spatial_crop, pad_images_spatial_crop], dim=0
                )
            batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        batched_samples = BatchCollateOutput(
            input_ids=batched_input_ids,
            images=batched_images,
            images_seq_mask=batched_images_seq_mask,
            images_spatial_crop=batched_images_spatial_crop,
            sft_format=batched_sft_format,
            seq_lens=seq_lens,
        )

        return batched_samples


@dataclass
class BatchCollateOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.LongTensor
    images: torch.Tensor
    images_spatial_crop: torch.LongTensor
    seq_lens: List[int]

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.images_spatial_crop = self.images_spatial_crop.to(device)
        self.images = self.images.to(device=device, dtype=dtype)
        return self


DeepseekVLV2ProcessorInstance: DeepseekVLV2Processor = None
DeepseekVLV2TokenizerInstance: LlamaTokenizerFast = None


def load_model(model_path, dtype=torch.bfloat16):
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    return tokenizer, vl_chat_processor


def fetch_model(dtype=torch.bfloat16):
    global DeepseekVLV2ProcessorInstance, DeepseekVLV2TokenizerInstance

    if DeepseekVLV2ProcessorInstance:
        print(f"has been loaded.")
    else:
        print(f"is loading...")
        DeepseekVLV2ProcessorInstance, DeepseekVLV2TokenizerInstance = load_model(
            model_path, dtype=dtype
        )
        print(f"Load successfully...")

    return model_info


def fetch_processor() -> DeepseekVLV2Processor:
    fetch_model()
    return DeepseekVLV2ProcessorInstance


def fetch_tokenizer() -> LlamaTokenizerFast:
    fetch_model()
    return DeepseekVLV2TokenizerInstance


class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
        deterministic: bool = False,
        num_recomputing_layers: int = 0,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # siglip use PytorchGELUTanh() rather than the vanilla nn.GELU()
        # https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/siglip/configuration_siglip.py#L191
        act_layer = partial(nn.GELU, approximate="tanh")

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head
        self.num_recomputing_layers = num_recomputing_layers

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    deterministic=deterministic,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n
        )

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "is_first_stage", True):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            skip_last = max(1, len(self.blocks) - self.num_recomputing_layers)
            x = checkpoint_seq(self.blocks, x, skip_last=skip_last)
        else:
            x = self.blocks(x)
        if getattr(self, "is_last_stage", True):
            x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if not getattr(self, "is_last_stage", True):
            return x
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x

    def to_pipeline(self, pp_size, pp_rank, pp_splits: Optional[List[int]] = None):
        self.is_first_stage = pp_rank == 0
        self.is_last_stage = pp_rank == pp_size - 1
        if not self.is_first_stage and hasattr(self, "patch_embed"):
            del (
                self.patch_embed,
                self.cls_token,
                self.reg_token,
                self.pos_embed,
                self.pos_drop,
                self.patch_drop,
                self.norm_pre,
            )
        if not self.is_last_stage and hasattr(self, "norm"):
            del self.norm, self.attn_pool, self.fc_norm, self.head_drop, self.head
        if pp_splits is not None:
            assert len(self.blocks) == sum(pp_splits)
            splits = np.cumsum([0] + pp_splits)
            self.blocks = self.blocks[splits[pp_rank] : splits[pp_rank + 1]]
        return self


class DeepseekVLV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers,
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        # image token format 形式
        # FIXME 目前tile tag & global_view_pos的默认取值都是之前的实验策略；后续应当去掉默认取值，改为没有取值就raise error
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed))
                * embed_std
            )
        else:
            raise ValueError(
                f"tile tag should be either 1D or 2D, but got {self.tile_tag}"
            )

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(
            config=language_config, quant_config=quant_config
        )

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **ignore_kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(images[idx, : batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == "2D":

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat(
                        [global_features, new_lines_in_global], dim=1
                    )
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w,
                    )

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(
                        self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h
                    )

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat(
                        [local_features, new_lines_in_local], dim=1
                    )

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [
                                global_features,
                                self.view_seperator[None, :],
                                local_features,
                            ],
                            dim=0,
                        )
                    else:
                        global_local_features = torch.cat(
                            [
                                local_features,
                                self.view_seperator[None, :],
                                global_features,
                            ],
                            dim=0,
                        )

                else:
                    # abandoned，实际上不会走这个逻辑
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )
                    local_features = torch.cat(
                        [
                            self.tile_indicators[1 : num_tiles_in_image + 1].unsqueeze(
                                1
                            ),
                            local_features,
                        ],
                        dim=1,
                    )
                    local_features = rearrange(
                        local_features, "crop_num hw d -> (crop_num hw) d"
                    )

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, local_features], dim=0
                        )
                    else:
                        global_local_features = torch.cat(
                            [local_features, global_features], dim=0
                        )

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(
                    images_seq_mask[idx].unsqueeze(-1), images_in_this_batch
                )

        return input_embeds

    @torch.no_grad()
    def incremental_prefilling(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        chunk_size: int = 1024,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images
            del images_seq_mask
            del images_spatial_crop

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            self._clear_cuda_cache()

        bzs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        # remain the last token for the next forward
        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[:, 0:chunk_end]
            # print(f"start = {chunk_start}, end = {chunk_end}, prefilling_len = {prefilling_len}, seq_len = {seq_len}")

            # compute position_ids
            if past_key_values is not None:
                position_ids = torch.arange(
                    chunk_start,
                    chunk_end,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ).unsqueeze(0)
                past_key_values = self._move_past_key_values_to_gpu(
                    past_key_values, inputs_embeds.device
                )
            else:
                position_ids = None

            # chunk-forward
            with torch.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                # update past_key_values
                past_key_values = outputs.past_key_values
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)

                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append(
                (
                    layer_past[0][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
                    layer_past[1][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
                )
            )

        return inputs_embeds, prefilling_key_values

    # Using the info from forward_batch, run preprocess on image_inputs, then stack tensor for later inference
    def prepare_image_inputs(
        self, forward_batch: ForwardBatch, preprocessor: DeepseekVLV2Processor
    ):
        # Initialize lists to store processed data
        pixel_values_list = []
        images_seq_mask_list = []
        images_spatial_crop_list = []

        # Get max number of images in batch
        max_n_images = (
            max(len(item.images) for item in forward_batch.image_inputs)
            if forward_batch.image_inputs
            else 0
        )

        # Process each image input in the batch
        for i in range(forward_batch.batch_size):
            # Get the prompt and images for this batch item
            prompt = forward_batch.prompts[i]
            images = forward_batch.images[i] if forward_batch.images else None

            # Call preprocessor
            preprocessor_out = preprocessor(
                prompt=prompt, images=images, apply_sft_format=True, inference_mode=True
            )

            # Pad images to max_n_images
            if preprocessor_out.images is not None:
                num_images = preprocessor_out.images.shape[0]
                padding = [(0, max_n_images - num_images)] + [(0, 0)] * (
                    len(preprocessor_out.images.shape) - 1
                )
                padded_images = torch.nn.functional.pad(
                    preprocessor_out.images, padding
                )
                pixel_values_list.append(padded_images)

                # Pad spatial crop
                if preprocessor_out.images_spatial_crop is not None:
                    spatial_crop_padding = [(0, max_n_images - num_images), (0, 0)]
                    padded_spatial_crop = torch.nn.functional.pad(
                        preprocessor_out.images_spatial_crop, spatial_crop_padding
                    )
                    images_spatial_crop_list.append(padded_spatial_crop)

            # Pad sequence mask
            if preprocessor_out.images_seq_mask is not None:
                images_seq_mask_list.append(preprocessor_out.images_seq_mask)

        # Stack tensors along batch dimension
        pixel_values = (
            torch.stack(pixel_values_list, dim=0) if pixel_values_list else None
        )  # [b, max_n_images, 3, H, W]
        images_seq_mask = (
            torch.stack(images_seq_mask_list, dim=0) if images_seq_mask_list else None
        )  # [b, T]
        images_spatial_crop = (
            torch.stack(images_spatial_crop_list, dim=0)
            if images_spatial_crop_list
            else None
        )  # [b, max_n_images, 2]

        return pixel_values, images_seq_mask, images_spatial_crop

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        positions: Optional[torch.LongTensor],
        forward_batch: ForwardBatch,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Get preprocessor
        preprocessor = fetch_processor()

        # Prepare image inputs
        images, images_seq_mask, images_spatial_crop = self.prepare_image_inputs(
            forward_batch, preprocessor
        )

        # Prepare inputs embeddings
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )

        outputs = self.language.forward(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            forward_batch=forward_batch,
        )

        return outputs

    def _clear_cuda_cache(self):
        """clear CUDA memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _move_past_key_values_to_cpu(self, past_key_values):
        # print(f"past_key_values -> cpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.cpu() for t in layer) for layer in past_key_values)

    def _move_past_key_values_to_gpu(self, past_key_values, device="cuda:0"):
        # print(f"past_key_values -> gpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.to(device) for t in layer) for layer in past_key_values)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


EntryClass = [DeepseekVLV2ForCausalLM]
