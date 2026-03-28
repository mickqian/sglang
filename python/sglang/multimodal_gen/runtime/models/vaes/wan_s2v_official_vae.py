# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device


class WanS2VOfficialVAE(nn.Module):
    _aliases = ["WanS2VOfficialVAE"]

    def __init__(self, helper: Any) -> None:
        super().__init__()
        self.helper = helper
        self.dtype = helper.dtype
        self.device_ = helper.device

    @property
    def device(self):
        return self.device_

    @staticmethod
    def _resolve_existing_path(component_model_path: str, path_value: str | None) -> str:
        if not path_value:
            raise ValueError("WanS2VOfficialVAE config is missing a required path")
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
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        module_configs = importlib.import_module("wan.configs")
        module_vae = importlib.import_module("wan.modules.vae2_1")
        task_name = str(config.get("wan_task_name", "s2v-14B"))
        config_obj = module_configs.WAN_CONFIGS[task_name]
        helper = module_vae.Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_root, config_obj.vae_checkpoint),
            device=(
                torch.device("cpu")
                if bool(getattr(server_args, "vae_cpu_offload", False))
                else get_local_torch_device()
            ),
        )
        return cls(helper=helper)

    def to(self, *args, **kwargs):
        return self

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            self.helper.encode(video.to(dtype=self.dtype, device=self.device))
        )

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            self.helper.decode(latents.to(dtype=self.dtype, device=self.device))
        )


EntryClass = WanS2VOfficialVAE
