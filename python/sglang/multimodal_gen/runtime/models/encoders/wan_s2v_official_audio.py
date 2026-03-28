# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.encoders.wan_s2v_audio import (
    WanS2VAudioEncoderConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.encoders.base import AudioEncoder


class WanS2VOfficialAudioEncoder(AudioEncoder):
    _aliases = ["WanS2VOfficialAudioEncoder"]

    def __init__(self, config: WanS2VAudioEncoderConfig, helper: Any) -> None:
        super().__init__(config)
        self.helper = helper

    @staticmethod
    def _resolve_existing_path(component_model_path: str, path_value: str | None) -> str:
        if not path_value:
            raise ValueError(
                "WanS2VOfficialAudioEncoder config is missing a required path"
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
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        module_audio = importlib.import_module("wan.modules.s2v.audio_encoder")
        target_device = (
            torch.device("cpu")
            if bool(getattr(server_args, "audio_encoder_cpu_offload", False))
            else get_local_torch_device()
        )
        helper = module_audio.AudioEncoder(
            device=target_device,
            model_id=os.path.join(checkpoint_root, "wav2vec2-large-xlsr-53-english"),
        )
        return cls(config=WanS2VAudioEncoderConfig(), helper=helper)

    def encode_audio(
        self,
        audio_path: str,
        *,
        infer_frames: int,
        fps: int,
        dtype: torch.dtype,
        m: int = 0,
    ) -> tuple[torch.Tensor, int]:
        embeddings = self.helper.extract_audio_feat(
            audio_path,
            return_all_layers=True,
            dtype=torch.float32,
        )
        bucket, num_repeat = self.helper.get_audio_embed_bucket_fps(
            embeddings,
            fps=fps,
            batch_frames=infer_frames,
            m=m,
        )
        bucket = bucket.to(get_local_torch_device(), dtype).unsqueeze(0)
        if bucket.ndim == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.ndim == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        return bucket, num_repeat

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "WanS2VOfficialAudioEncoder is used via encode_audio(), not forward()."
        )


EntryClass = WanS2VOfficialAudioEncoder
