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
"""Utilities for Huggingface Transformers."""

import contextlib
import enum
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Type, Union

import huggingface_hub
import torch
from huggingface_hub import (
    file_exists,
    hf_hub_download,
    snapshot_download,
    try_to_load_from_cache,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from sglang.srt.configs import (
    ChatGLMConfig,
    DbrxConfig,
    DeepseekVL2Config,
    ExaoneConfig,
    Gemma3Config,
    Gemma3TextConfig,
    Mistral3Config,
    MultiModalityConfig,
    Qwen2_5_VLConfig,
)
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url
from sglang.utils import logger

HF_TOKEN = os.getenv("HF_TOKEN", None)
MISTRAL_CONFIG_NAME = "params.json"
AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    ChatGLMConfig.model_type: ChatGLMConfig,
    DbrxConfig.model_type: DbrxConfig,
    ExaoneConfig.model_type: ExaoneConfig,
    Qwen2_5_VLConfig.model_type: Qwen2_5_VLConfig,
    DeepseekVL2Config.model_type: DeepseekVL2Config,
    MultiModalityConfig.model_type: MultiModalityConfig,
    Gemma3Config.model_type: Gemma3Config,
    Gemma3TextConfig.model_type: Gemma3TextConfig,
    Mistral3Config.model_type: Mistral3Config,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def pour_multimodal_model_text_config(model: str, config: PretrainedConfig):
    # FIXME: Pour contents of mm model's langauge_config to first-level
    text_config = get_hf_text_config(config)
    if text_config is not None:
        for key, val in text_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)
        return


# In offline mode the result can be a false negative
def file_or_path_exists(
    model: Union[str, Path], config_name: str, revision: Optional[str]
) -> bool:
    if (local_path := Path(model)).exists():
        return (local_path / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(
        repo_id=model, filename=config_name, revision=revision
    )
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.

    # Call HF to check if the file exists
    return file_exists(str(model), config_name, revision=revision, token=HF_TOKEN)


class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


def try_get_local_file(
    model: Union[str, Path], file_name: str, revision: Optional[str] = "main"
) -> Optional[Path]:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(
                repo_id=model, filename=file_name, revision=revision
            )
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except HFValidationError:
            ...
    return None


def get_hf_file_to_dict(
    file_name: str, model: Union[str, Path], revision: Optional[str] = "main"
):
    """
    Downloads a file from the Hugging Face Hub and returns
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model.

    Returns:
    - config_dict (dict): A dictionary containing
    the contents of the downloaded file.
    """

    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (
            RepositoryNotFoundError,
            RevisionNotFoundError,
            EntryNotFoundError,
            LocalEntryNotFoundError,
        ) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file "
                "download for '%s':",
                file_name,
                exc_info=e,
            )
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

    return None


def get_config_format(
    is_gguf: bool,
    model: str,
    revision: Optional[str] = None,
) -> ConfigFormat:
    try:
        if file_or_path_exists(model, HF_CONFIG_NAME, revision=revision):
            config_format = ConfigFormat.HF
    except Exception as e:
        error_message = (
            "Invalid repository ID or local directory specified:"
            " '{model}'.\nPlease verify the following requirements:\n"
            "1. Provide a valid Hugging Face repository ID.\n"
            "2. Specify a local directory that contains a recognized "
            "configuration file.\n"
            "   - For Hugging Face models: ensure the presence of a "
            "'config.json'.\n"
            "   - For Mistral models: ensure the presence of a "
            "'params.json'.\n"
        ).format(model=model)

        raise ValueError(error_message) from e
    return config_format


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = config.architectures[0]
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
        setattr(config, "torch_dtype", torch.float16)
        return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    if hasattr(config, "text_config"):
        return config.text_config
    else:
        return config


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        config_format = get_config_format(is_gguf, model, revision)
    if config_format == ConfigFormat.HF:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=revision,
            token=HF_TOKEN,
            **kwargs,
        )
        pour_multimodal_model_text_config(model=model, config=config)

        if config.model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[config.model_type]
            config = config_class.from_pretrained(model, revision=revision)
            # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
            setattr(config, "_name_or_path", model)
    else:
        raise ValueError(f"Unsupported config format: {config_format}")

    if model_override_args:
        config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config, config_format


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    tokenizer: AnyTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    # FIXME
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )

    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
