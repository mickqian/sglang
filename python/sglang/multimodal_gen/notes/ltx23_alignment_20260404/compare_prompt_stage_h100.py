from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import pack_text_embeds_v2
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2TextConnectors,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

LTX_REPO_ROOT = Path(os.environ.get("LTX_REPO_ROOT", "/tmp/LTX-2"))
CHECKPOINT_GLOBS = (
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-20b-dev.safetensors",
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-22b-dev.safetensors",
)
MODEL_ROOT_GLOB = "/root/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-*"
OUTPUT_PATH = Path(
    os.environ.get("LTX23_PROMPT_COMPARE_JSON", "/tmp/ltx23_prompt_compare_h100.json")
)
PROMPT = os.environ.get("LTX23_PROMPT", "A beautiful sunset over the ocean")
NEGATIVE_PROMPT = os.environ.get("LTX23_NEGATIVE_PROMPT", " ")
COMPARE_MODE = os.environ.get("LTX23_COMPARE_MODE", "full")


def resolve_single_path(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern}, got {matches}")
    return matches[0]


def resolve_first_existing_path(patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Expected at most one match for {pattern}, got {matches}")
    raise RuntimeError(f"Expected one match from {patterns}, got none")


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | list[int]]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = a - b
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1), dim=-1
            ).item()
        ),
    }


def main() -> None:
    checkpoint_path = resolve_first_existing_path(CHECKPOINT_GLOBS)
    model_root = resolve_single_path(MODEL_ROOT_GLOB)
    connectors_root = model_root / "connectors"

    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-core/src"))
    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-pipelines/src"))

    from ltx_core.model.transformer.rope import (
        apply_rotary_emb,
        generate_freq_grid_np,
        generate_freq_grid_pytorch,
        precompute_freqs_cis,
    )
    from ltx_core.utils import rms_norm
    from ltx_pipelines.utils.blocks import PromptEncoder, gpu_model

    device = torch.device("cuda")
    dtype = torch.bfloat16
    set_global_server_args(ServerArgs(model_path=str(model_root)))

    prompt_encoder = PromptEncoder(
        checkpoint_path=str(checkpoint_path),
        gemma_root=str(model_root),
        dtype=dtype,
        device=device,
    )

    with prompt_encoder._text_encoder_ctx(None) as text_encoder:
        raw_outputs = {
            "prompt": text_encoder.encode(PROMPT),
            "negative_prompt": text_encoder.encode(NEGATIVE_PROMPT),
        }

    with open(connectors_root / "config.json") as f:
        connector_cfg = json.load(f)
    connector_cfg.pop("_class_name", None)
    connector_cfg.pop("_diffusers_version", None)
    connector_cfg.pop("_name_or_path", None)
    connectors = LTX2TextConnectors(SimpleNamespace(**connector_cfg)).to(
        device=device, dtype=dtype
    )
    load_result = connectors.load_state_dict(
        safetensors_load_file(str(connectors_root / "model.safetensors")), strict=False
    )
    connectors.eval()

    payload: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "model_root": str(model_root),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "missing_keys": sorted(load_result.missing_keys),
        "unexpected_keys": sorted(load_result.unexpected_keys),
        "results": {},
    }

    with gpu_model(
        prompt_encoder._embeddings_processor_builder.build(device=device, dtype=dtype)
        .to(device)
        .eval()
    ) as embeddings_processor:
        payload.update(
            {
                "video_norm_q_weight": tensor_metrics(
                    embeddings_processor.video_connector.transformer_1d_blocks[
                        0
                    ].attn1.q_norm.weight,
                    connectors.video_connector.transformer_blocks[0].attn1.norm_q.weight,
                ),
                "video_norm_k_weight": tensor_metrics(
                    embeddings_processor.video_connector.transformer_1d_blocks[
                        0
                    ].attn1.k_norm.weight,
                    connectors.video_connector.transformer_blocks[0].attn1.norm_k.weight,
                ),
                "audio_norm_q_weight": tensor_metrics(
                    embeddings_processor.audio_connector.transformer_1d_blocks[
                        0
                    ].attn1.q_norm.weight,
                    connectors.audio_connector.transformer_blocks[0].attn1.norm_q.weight,
                ),
                "audio_norm_k_weight": tensor_metrics(
                    embeddings_processor.audio_connector.transformer_1d_blocks[
                        0
                    ].attn1.k_norm.weight,
                    connectors.audio_connector.transformer_blocks[0].attn1.norm_k.weight,
                ),
            }
        )
        for key, (hidden_states, attention_mask) in raw_outputs.items():
            official_video_features, official_audio_features = (
                embeddings_processor.feature_extractor(
                    hidden_states, attention_mask, "left"
                )
            )
            official_output = None
            if COMPARE_MODE == "full":
                official_output = embeddings_processor.process_hidden_states(
                    hidden_states, attention_mask
                )
            seq_len = official_video_features.shape[1]
            indices_grid = torch.arange(
                seq_len, dtype=torch.float32, device=official_video_features.device
            )[None, None, :]

            stacked_hidden_states = torch.stack(hidden_states, dim=-1)
            packed_hidden_states = pack_text_embeds_v2(
                stacked_hidden_states, attention_mask
            ).to(device=device, dtype=dtype)
            source_dim = connectors.video_aggregate_embed.out_features
            video_hidden_states = packed_hidden_states
            audio_hidden_states = packed_hidden_states
            if video_hidden_states.dtype != connectors.video_aggregate_embed.weight.dtype:
                video_hidden_states = video_hidden_states.to(
                    connectors.video_aggregate_embed.weight.dtype
                )
            if audio_hidden_states.dtype != connectors.audio_aggregate_embed.weight.dtype:
                audio_hidden_states = audio_hidden_states.to(
                    connectors.audio_aggregate_embed.weight.dtype
                )
            ours_video_features = connectors.video_aggregate_embed(
                connectors._rescale_v2_features(
                    video_hidden_states,
                    connectors.video_aggregate_embed.out_features,
                    source_dim,
                )
            )
            ours_audio_features = connectors.audio_aggregate_embed(
                connectors._rescale_v2_features(
                    audio_hidden_states,
                    connectors.audio_aggregate_embed.out_features,
                    source_dim,
                )
            )
            additive_mask = (
                (1 - attention_mask.to(device=device, dtype=dtype)) * -1000000.0
            )
            ours_video = None
            ours_audio = None
            ours_mask = None
            with set_forward_context(current_timestep=None, attn_metadata=None):
                if COMPARE_MODE == "full":
                    ours_video, ours_audio, ours_mask = connectors(
                        packed_hidden_states, additive_mask, additive_mask=True
                    )
                official_video_freqs = precompute_freqs_cis(
                    indices_grid=indices_grid,
                    dim=embeddings_processor.video_connector.inner_dim,
                    out_dtype=official_video_features.dtype,
                    theta=embeddings_processor.video_connector.positional_embedding_theta,
                    max_pos=embeddings_processor.video_connector.positional_embedding_max_pos,
                    num_attention_heads=embeddings_processor.video_connector.num_attention_heads,
                    rope_type=embeddings_processor.video_connector.rope_type,
                    freq_grid_generator=(
                        generate_freq_grid_np
                        if embeddings_processor.video_connector.double_precision_rope
                        else generate_freq_grid_pytorch
                    ),
                )
                ours_video_freqs = connectors.video_connector.rope(
                    official_video_features.shape[0],
                    seq_len,
                    official_video_features.device,
                    dtype=official_video_features.dtype,
                )
                official_audio_freqs = precompute_freqs_cis(
                    indices_grid=indices_grid,
                    dim=embeddings_processor.audio_connector.inner_dim,
                    out_dtype=official_audio_features.dtype,
                    theta=embeddings_processor.audio_connector.positional_embedding_theta,
                    max_pos=embeddings_processor.audio_connector.positional_embedding_max_pos,
                    num_attention_heads=embeddings_processor.audio_connector.num_attention_heads,
                    rope_type=embeddings_processor.audio_connector.rope_type,
                    freq_grid_generator=(
                        generate_freq_grid_np
                        if embeddings_processor.audio_connector.double_precision_rope
                        else generate_freq_grid_pytorch
                    ),
                )
                ours_audio_freqs = connectors.audio_connector.rope(
                    official_audio_features.shape[0],
                    seq_len,
                    official_audio_features.device,
                    dtype=official_audio_features.dtype,
                )
                official_video_block0 = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0](
                        official_video_features,
                        attention_mask=additive_mask,
                        pe=official_video_freqs,
                    )
                )
                ours_video_block0 = connectors.video_connector.transformer_blocks[0](
                    official_video_features,
                    attention_mask=additive_mask,
                    rotary_emb=ours_video_freqs,
                )
                official_audio_block0 = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0](
                        official_audio_features,
                        attention_mask=additive_mask,
                        pe=official_audio_freqs,
                    )
                )
                ours_audio_block0 = connectors.audio_connector.transformer_blocks[0](
                    official_audio_features,
                    attention_mask=additive_mask,
                    rotary_emb=ours_audio_freqs,
                )
                official_video_norm1 = rms_norm(official_video_features)
                ours_video_norm1 = connectors.video_connector.transformer_blocks[0].norm1(
                    official_video_features
                )
                official_audio_norm1 = rms_norm(official_audio_features)
                ours_audio_norm1 = connectors.audio_connector.transformer_blocks[0].norm1(
                    official_audio_features
                )

                official_video_q = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1.to_q(official_video_norm1)
                )
                ours_video_q = connectors.video_connector.transformer_blocks[0].attn1.to_q(
                    ours_video_norm1
                )
                official_video_k = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1.to_k(official_video_norm1)
                )
                ours_video_k = connectors.video_connector.transformer_blocks[0].attn1.to_k(
                    ours_video_norm1
                )
                official_video_v = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1.to_v(official_video_norm1)
                )
                ours_video_v = connectors.video_connector.transformer_blocks[0].attn1.to_v(
                    ours_video_norm1
                )
                official_video_q_norm = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1.q_norm(official_video_q)
                )
                ours_video_q_norm = (
                    connectors.video_connector.transformer_blocks[0].attn1.norm_q(
                        ours_video_q
                    )
                )
                official_video_k_norm = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1.k_norm(official_video_k)
                )
                ours_video_k_norm = (
                    connectors.video_connector.transformer_blocks[0].attn1.norm_k(
                        ours_video_k
                    )
                )
                official_video_q_rope = apply_rotary_emb(
                    official_video_q_norm,
                    official_video_freqs,
                    embeddings_processor.video_connector.rope_type,
                )
                official_video_k_rope = apply_rotary_emb(
                    official_video_k_norm,
                    official_video_freqs,
                    embeddings_processor.video_connector.rope_type,
                )
                if connectors.video_connector.transformer_blocks[0].attn1.rope_type == "interleaved":
                    ours_video_q_rope = connectors.video_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_interleaved_rotary_emb"
                    ](ours_video_q_norm, ours_video_freqs)
                    ours_video_k_rope = connectors.video_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_interleaved_rotary_emb"
                    ](ours_video_k_norm, ours_video_freqs)
                else:
                    ours_video_q_rope = connectors.video_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_split_rotary_emb"
                    ](ours_video_q_norm, ours_video_freqs)
                    ours_video_k_rope = connectors.video_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_split_rotary_emb"
                    ](ours_video_k_norm, ours_video_freqs)
                official_video_attn = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .attn1(official_video_norm1, mask=additive_mask, pe=official_video_freqs)
                )
                ours_video_attn = connectors.video_connector.transformer_blocks[0].attn1(
                    ours_video_norm1,
                    attention_mask=additive_mask,
                    query_rotary_emb=ours_video_freqs,
                )
                official_video_after_attn = official_video_attn + official_video_features
                ours_video_after_attn = ours_video_attn + official_video_features
                official_video_norm2 = rms_norm(official_video_after_attn)
                ours_video_norm2 = connectors.video_connector.transformer_blocks[0].norm2(
                    ours_video_after_attn
                )
                official_video_ff = (
                    embeddings_processor.video_connector.transformer_1d_blocks[0]
                    .ff(official_video_norm2)
                )
                ours_video_ff = connectors.video_connector.transformer_blocks[0].ff(
                    ours_video_norm2
                )

                official_audio_q = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1.to_q(official_audio_norm1)
                )
                ours_audio_q = connectors.audio_connector.transformer_blocks[0].attn1.to_q(
                    ours_audio_norm1
                )
                official_audio_k = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1.to_k(official_audio_norm1)
                )
                ours_audio_k = connectors.audio_connector.transformer_blocks[0].attn1.to_k(
                    ours_audio_norm1
                )
                official_audio_v = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1.to_v(official_audio_norm1)
                )
                ours_audio_v = connectors.audio_connector.transformer_blocks[0].attn1.to_v(
                    ours_audio_norm1
                )
                official_audio_q_norm = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1.q_norm(official_audio_q)
                )
                ours_audio_q_norm = (
                    connectors.audio_connector.transformer_blocks[0].attn1.norm_q(
                        ours_audio_q
                    )
                )
                official_audio_k_norm = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1.k_norm(official_audio_k)
                )
                ours_audio_k_norm = (
                    connectors.audio_connector.transformer_blocks[0].attn1.norm_k(
                        ours_audio_k
                    )
                )
                official_audio_q_rope = apply_rotary_emb(
                    official_audio_q_norm,
                    official_audio_freqs,
                    embeddings_processor.audio_connector.rope_type,
                )
                official_audio_k_rope = apply_rotary_emb(
                    official_audio_k_norm,
                    official_audio_freqs,
                    embeddings_processor.audio_connector.rope_type,
                )
                if connectors.audio_connector.transformer_blocks[0].attn1.rope_type == "interleaved":
                    ours_audio_q_rope = connectors.audio_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_interleaved_rotary_emb"
                    ](ours_audio_q_norm, ours_audio_freqs)
                    ours_audio_k_rope = connectors.audio_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_interleaved_rotary_emb"
                    ](ours_audio_k_norm, ours_audio_freqs)
                else:
                    ours_audio_q_rope = connectors.audio_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_split_rotary_emb"
                    ](ours_audio_q_norm, ours_audio_freqs)
                    ours_audio_k_rope = connectors.audio_connector.transformer_blocks[0].attn1.forward.__globals__[
                        "apply_split_rotary_emb"
                    ](ours_audio_k_norm, ours_audio_freqs)
                official_audio_attn = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .attn1(official_audio_norm1, mask=additive_mask, pe=official_audio_freqs)
                )
                ours_audio_attn = connectors.audio_connector.transformer_blocks[0].attn1(
                    ours_audio_norm1,
                    attention_mask=additive_mask,
                    query_rotary_emb=ours_audio_freqs,
                )
                official_audio_after_attn = official_audio_attn + official_audio_features
                ours_audio_after_attn = ours_audio_attn + official_audio_features
                official_audio_norm2 = rms_norm(official_audio_after_attn)
                ours_audio_norm2 = connectors.audio_connector.transformer_blocks[0].norm2(
                    ours_audio_after_attn
                )
                official_audio_ff = (
                    embeddings_processor.audio_connector.transformer_1d_blocks[0]
                    .ff(official_audio_norm2)
                )
                ours_audio_ff = connectors.audio_connector.transformer_blocks[0].ff(
                    ours_audio_norm2
                )

            payload["results"][key] = {
                "video_rope_cos": tensor_metrics(
                    official_video_freqs[0], ours_video_freqs[0]
                ),
                "video_rope_sin": tensor_metrics(
                    official_video_freqs[1], ours_video_freqs[1]
                ),
                "audio_rope_cos": tensor_metrics(
                    official_audio_freqs[0], ours_audio_freqs[0]
                ),
                "audio_rope_sin": tensor_metrics(
                    official_audio_freqs[1], ours_audio_freqs[1]
                ),
                "video_features": tensor_metrics(
                    official_video_features, ours_video_features
                ),
                "audio_features": tensor_metrics(
                    official_audio_features, ours_audio_features
                ),
                "video_norm1": tensor_metrics(official_video_norm1, ours_video_norm1),
                "audio_norm1": tensor_metrics(official_audio_norm1, ours_audio_norm1),
                "video_q": tensor_metrics(official_video_q, ours_video_q),
                "video_k": tensor_metrics(official_video_k, ours_video_k),
                "video_v": tensor_metrics(official_video_v, ours_video_v),
                "video_q_norm": tensor_metrics(
                    official_video_q_norm, ours_video_q_norm
                ),
                "video_k_norm": tensor_metrics(
                    official_video_k_norm, ours_video_k_norm
                ),
                "video_q_rope": tensor_metrics(
                    official_video_q_rope, ours_video_q_rope
                ),
                "video_k_rope": tensor_metrics(
                    official_video_k_rope, ours_video_k_rope
                ),
                "video_attn": tensor_metrics(official_video_attn, ours_video_attn),
                "video_after_attn": tensor_metrics(
                    official_video_after_attn, ours_video_after_attn
                ),
                "video_norm2": tensor_metrics(official_video_norm2, ours_video_norm2),
                "video_ff": tensor_metrics(official_video_ff, ours_video_ff),
                "audio_q": tensor_metrics(official_audio_q, ours_audio_q),
                "audio_k": tensor_metrics(official_audio_k, ours_audio_k),
                "audio_v": tensor_metrics(official_audio_v, ours_audio_v),
                "audio_q_norm": tensor_metrics(
                    official_audio_q_norm, ours_audio_q_norm
                ),
                "audio_k_norm": tensor_metrics(
                    official_audio_k_norm, ours_audio_k_norm
                ),
                "audio_q_rope": tensor_metrics(
                    official_audio_q_rope, ours_audio_q_rope
                ),
                "audio_k_rope": tensor_metrics(
                    official_audio_k_rope, ours_audio_k_rope
                ),
                "audio_attn": tensor_metrics(official_audio_attn, ours_audio_attn),
                "audio_after_attn": tensor_metrics(
                    official_audio_after_attn, ours_audio_after_attn
                ),
                "audio_norm2": tensor_metrics(official_audio_norm2, ours_audio_norm2),
                "audio_ff": tensor_metrics(official_audio_ff, ours_audio_ff),
                "video_block0": tensor_metrics(
                    official_video_block0, ours_video_block0
                ),
                "audio_block0": tensor_metrics(
                    official_audio_block0, ours_audio_block0
                ),
            }
            if official_output is not None and ours_video is not None and ours_audio is not None and ours_mask is not None:
                payload["results"][key]["video_encoding"] = tensor_metrics(
                    official_output.video_encoding, ours_video
                )
                payload["results"][key]["audio_encoding"] = tensor_metrics(
                    official_output.audio_encoding, ours_audio
                )
                payload["results"][key]["attention_mask_match"] = bool(
                    torch.equal(
                        official_output.attention_mask.detach().cpu(),
                        ours_mask.detach().cpu(),
                    )
                )
                payload["results"][key]["attention_mask_sum_official"] = int(
                    official_output.attention_mask.sum().item()
                )
                payload["results"][key]["attention_mask_sum_sglang"] = int(
                    ours_mask.sum().item()
                )

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
