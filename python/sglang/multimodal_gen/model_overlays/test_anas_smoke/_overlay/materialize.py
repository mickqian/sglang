# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors.torch import save_file


def _read_state_dict(path: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path} did not contain a state dict")
    state = {
        (key[len("model.") :] if key.startswith("model.") else key): value.contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    state.pop("pos_embed", None)
    return state


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _patch_vae_config(output_dir: str) -> None:
    config_path = os.path.join(output_dir, "vae", "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    source_class = config.get("_class_name")
    config["_sglang_source_class_name"] = source_class
    config["_class_name"] = "AutoencoderKLLTX2Video"
    _write_json(config_path, config)


def materialize(
    *,
    overlay_dir: str,
    source_dir: str,
    output_dir: str,
    manifest: dict[str, Any],
) -> None:
    del overlay_dir, manifest

    os.makedirs(os.path.join(output_dir, "text_encoder"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)

    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    _write_json(
        os.path.join(transformer_dir, "config.json"),
        {
            "_class_name": "SanaWM1600M",
            "_diffusers_version": "0.35.0",
            "caption_channels": 2304,
            "hidden_size": 2240,
            "in_channels": 128,
            "model_max_length": 300,
            "num_attention_heads": 20,
            "num_layers": 20,
            "out_channels": 128
        },
    )

    state = _read_state_dict(os.path.join(source_dir, "sana_dit", "model.pt"))
    save_file(
        state,
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
    )
    _patch_vae_config(output_dir)
