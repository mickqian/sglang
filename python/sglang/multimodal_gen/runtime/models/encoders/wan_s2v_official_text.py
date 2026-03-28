# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.encoders import T5Config
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder


class WanS2VOfficialTextEncoder(TextEncoder):
    _aliases = ["WanS2VOfficialTextEncoder"]

    def __init__(self, config: T5Config, helper: Any) -> None:
        super().__init__(config)
        self.helper = helper
        self.model = helper.model

    @staticmethod
    def _resolve_existing_path(component_model_path: str, path_value: str | None) -> str:
        if not path_value:
            raise ValueError(
                "WanS2VOfficialTextEncoder config is missing a required path"
            )
        path_value = os.path.expanduser(path_value)
        if os.path.isabs(path_value):
            resolved = path_value
        else:
            resolved = os.path.join(component_model_path, path_value)
        if not os.path.exists(resolved):
            raise ValueError(f"Resolved path does not exist: {resolved}")
        return resolved

    @classmethod
    def from_component_path(
        cls,
        component_model_path: str,
        server_args,
        config: dict[str, Any],
    ):
        del server_args
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        module_configs = importlib.import_module("wan.configs")
        module_t5 = importlib.import_module("wan.modules.t5")
        task_name = str(config.get("wan_task_name", "s2v-14B"))
        config_obj = module_configs.WAN_CONFIGS[task_name]
        target_device = (
            torch.device("cpu")
            if bool(config.get("t5_cpu", False))
            else get_local_torch_device()
        )
        helper = module_t5.T5EncoderModel(
            text_len=config_obj.text_len,
            dtype=config_obj.t5_dtype,
            device=target_device,
            checkpoint_path=os.path.join(checkpoint_root, config_obj.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_root, config_obj.t5_tokenizer),
            shard_fn=None,
        )
        return cls(config=T5Config(), helper=helper)

    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        return self.helper([prompt], device)[0]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ):
        del (
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
            output_hidden_states,
            kwargs,
        )
        raise NotImplementedError(
            "WanS2VOfficialTextEncoder is used via encode_prompt(), not forward()."
        )


EntryClass = WanS2VOfficialTextEncoder
