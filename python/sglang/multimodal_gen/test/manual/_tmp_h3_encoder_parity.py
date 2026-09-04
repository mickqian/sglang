"""Temporary remote-only H3 encoder precision probe.

This file is carried only by a disposable validation ref and is not part of
the implementation branch.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_text_only_ids,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args.server_args import (
    set_global_server_args,
)

_SNAPSHOT = Path(
    "/scratch/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe"
)
_COMPONENT = _SNAPSHOT / "FL2VA/text_encoder"
_NVFP4 = Path(
    "/scratch/hf-cache/hub/models--qtum--MiniMax-H3-Qwen3-VL-NVFP4/"
    "snapshots/9bc1e665e427119f83ed8f7af556992a5c28433a/"
    "qwen3vl_32b_minimax_h3_nvfp4.safetensors"
)
_PROMPT = (
    "A single enormous cobalt-blue cat wearing a small bright yellow top hat "
    "plays a vivid red electric guitar alone on an empty snow-white theater stage."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    component_weights_paths = (
        {"text_encoder": str(_NVFP4)} if args.mode == "nvfp4" else {}
    )
    server_args = ServerArgs.from_kwargs(
        model_path=str(_SNAPSHOT),
        model_variant="fl2va",
        num_gpus=4,
        sp_degree=4,
        encoder_parallel="auto",
        component_weights_paths=component_weights_paths,
        warmup_mode="off",
    )
    set_global_server_args(server_args)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    initialize_model_parallel(
        tensor_parallel_degree=1,
        sequence_parallel_degree=4,
        ulysses_degree=4,
        ring_degree=1,
    )

    start = time.perf_counter()
    model, memory = PipelineComponentLoader.load_component(
        component_name="text_encoder",
        component_type="text_encoder",
        component_model_path=str(_COMPONENT),
        transformers_or_diffusers="transformers",
        server_args=server_args,
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(module, inputs, output):
            del module, inputs
            if rank == 0:
                value = output[0] if isinstance(output, tuple) else output
                captured[name] = value.detach().to("cpu", torch.bfloat16)

        return hook

    handles = [
        model.model.language_model.embed_tokens.register_forward_hook(save("embed"))
    ]
    for index, layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_hook(save(f"layer_{index}")))

    tokenizer = AutoTokenizer.from_pretrained(str(_COMPONENT), trust_remote_code=True)
    input_ids = minimax_h3_text_only_ids(tokenizer, _PROMPT)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        final = model.encode_ids(input_ids).detach().cpu()

    for handle in handles:
        handle.remove()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt": _PROMPT,
                "input_ids": input_ids.cpu(),
                "layers": captured,
                "final": final,
            },
            args.output,
        )
        print(
            "saved",
            args.output,
            "memory",
            memory,
            "tokens",
            input_ids.numel(),
            "layers",
            len(captured),
            "seconds",
            time.perf_counter() - start,
            flush=True,
        )

    torch.distributed.barrier()
    del model
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
