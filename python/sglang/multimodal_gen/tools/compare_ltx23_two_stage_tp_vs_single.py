"""Compare LTX-2.3 two-stage TP output against a single-GPU baseline.

This tool is specialized for native `LTX2TwoStagePipeline` precision alignment.
It runs a single-GPU reference (`tp=1`) and a TP candidate (`tp>1`) on the same
prompt / image / seed / generation settings, then reports:

- final video frame metrics, including aggregate PSNR
- stage1 / stage2 trajectory metrics when available
- audio trajectory metrics when available

Example:

    python -m sglang.multimodal_gen.tools.compare_ltx23_two_stage_tp_vs_single \
        --cases t2v ti2v \
        --output-json /tmp/ltx23_tp_vs_single.json \
        --save-output-dir /tmp/ltx23_tp_vs_single_outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    compute_tensor_metrics,
    extract_result_frames,
    run_variant,
    summarize_output_frame_metrics,
    summarize_trajectory_metrics,
)

T2V_PRESET = {
    "prompt": "A beautiful sunset over the ocean",
    "image_path": None,
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "guidance_scale_2": None,
}

TI2V_PRESET = {
    "prompt": (
        "The man in the picture slowly turns his head, his expression enigmatic "
        "and otherworldly. The camera performs a slow, cinematic dolly out, "
        "focusing on his face. Moody lighting, neon signs glowing in the "
        "background, shallow depth of field."
    ),
    "image_path": (
        "https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/"
        "5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg"
    ),
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "guidance_scale_2": None,
}

CASE_PRESETS = {
    "t2v": T2V_PRESET,
    "ti2v": TI2V_PRESET,
}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=True))


def _extract_stage_trajectory_data(result: Any) -> dict[str, dict[str, Any]]:
    stage_data = getattr(result, "stage_trajectory_data", None)
    if isinstance(stage_data, dict) and stage_data:
        return stage_data

    fallback: dict[str, dict[str, Any]] = {}
    payload: dict[str, Any] = {}
    if getattr(result, "trajectory_timesteps", None) is not None:
        payload["timesteps"] = result.trajectory_timesteps
    if getattr(result, "trajectory_latents", None) is not None:
        payload["video_latents"] = result.trajectory_latents
    if getattr(result, "trajectory_audio_latents", None) is not None:
        payload["audio_latents"] = result.trajectory_audio_latents
    if payload:
        fallback["stage2"] = payload
    return fallback


def summarize_stage_trajectory_data(
    reference_stage_data: dict[str, dict[str, Any]],
    candidate_stage_data: dict[str, dict[str, Any]],
    *,
    step_index: int,
) -> dict[str, Any]:
    reserved_stage_keys = {
        "timesteps",
        "video_latents",
        "audio_latents",
        "block_trace_data",
        "block_trace_export_phase_keys",
        "block_trace_forward_count",
    }
    shared_stages = sorted(set(reference_stage_data).intersection(candidate_stage_data))
    summary: dict[str, Any] = {}
    for stage in shared_stages:
        ref_stage = reference_stage_data[stage]
        cand_stage = candidate_stage_data[stage]
        stage_summary: dict[str, Any] = {}

        ref_timesteps = ref_stage.get("timesteps")
        cand_timesteps = cand_stage.get("timesteps")

        if (
            ref_stage.get("video_latents") is not None
            and cand_stage.get("video_latents") is not None
        ):
            stage_summary["video_trajectory_metrics"] = summarize_trajectory_metrics(
                ref_stage["video_latents"],
                cand_stage["video_latents"],
                reference_timesteps=ref_timesteps,
                candidate_timesteps=cand_timesteps,
                step_index=step_index,
            )
        if (
            ref_stage.get("audio_latents") is not None
            and cand_stage.get("audio_latents") is not None
        ):
            stage_summary["audio_trajectory_metrics"] = summarize_trajectory_metrics(
                ref_stage["audio_latents"],
                cand_stage["audio_latents"],
                reference_timesteps=ref_timesteps,
                candidate_timesteps=cand_timesteps,
                step_index=step_index,
            )
        ref_block_trace = ref_stage.get("block_trace_data")
        cand_block_trace = cand_stage.get("block_trace_data")
        if isinstance(ref_block_trace, dict) and isinstance(cand_block_trace, dict):
            ref_blocks = ref_block_trace.get("blocks")
            cand_blocks = cand_block_trace.get("blocks")
            if isinstance(ref_blocks, dict) and isinstance(cand_blocks, dict):
                shared_blocks = sorted(
                    set(ref_blocks).intersection(cand_blocks),
                    key=lambda value: int(value),
                )
                block_trace_metrics: dict[str, Any] = {}
                ref_step_indices = ref_block_trace.get("step_indices")
                cand_step_indices = cand_block_trace.get("step_indices")
                for block_key in shared_blocks:
                    ref_block = ref_blocks[block_key]
                    cand_block = cand_blocks[block_key]
                    if not isinstance(ref_block, dict) or not isinstance(cand_block, dict):
                        continue
                    module_metrics: dict[str, Any] = {}
                    for module_name in sorted(set(ref_block).intersection(cand_block)):
                        module_metrics[module_name] = summarize_trajectory_metrics(
                            ref_block[module_name],
                            cand_block[module_name],
                            reference_timesteps=ref_step_indices,
                            candidate_timesteps=cand_step_indices,
                            step_index=step_index,
                        )
                    if module_metrics:
                        block_trace_metrics[block_key] = module_metrics
                if block_trace_metrics:
                    stage_summary["block_trace_metrics"] = block_trace_metrics
        extra_tensor_metrics: dict[str, Any] = {}
        for extra_key in sorted(set(ref_stage).intersection(cand_stage)):
            if extra_key in reserved_stage_keys:
                continue
            ref_value = ref_stage[extra_key]
            cand_value = cand_stage[extra_key]
            try:
                extra_tensor_metrics[extra_key] = compute_tensor_metrics(
                    ref_value, cand_value
                )
            except (TypeError, ValueError):
                continue
        if extra_tensor_metrics:
            stage_summary["extra_tensor_metrics"] = extra_tensor_metrics
        if stage_summary:
            summary[stage] = stage_summary
    return summary


def build_server_kwargs(
    args: argparse.Namespace,
    *,
    num_gpus: int,
    tp_size: int,
    variant_index: int,
) -> dict[str, Any]:
    kwargs = {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "backend": args.backend,
        "attention_backend": args.attention_backend,
        "num_gpus": num_gpus,
        "tp_size": tp_size,
        "sp_degree": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "pipeline_class_name": "LTX2TwoStagePipeline",
        "ltx2_two_stage_device_mode": args.ltx2_two_stage_device_mode,
        "dit_cpu_offload": args.dit_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "pin_cpu_memory": args.pin_cpu_memory,
        "enable_cfg_parallel": False,
    }
    if args.port_base is not None:
        kwargs["port"] = args.port_base + variant_index
        kwargs["scheduler_port"] = args.port_base + 100 + variant_index
        kwargs["master_port"] = args.port_base + 200 + variant_index
    return kwargs


def build_sampling_kwargs(
    case_name: str,
    preset: dict[str, Any],
    *,
    seed: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": preset["prompt"],
        "width": preset["width"],
        "height": preset["height"],
        "num_frames": preset["num_frames"],
        "fps": preset["fps"],
        "num_inference_steps": preset["num_inference_steps"],
        "guidance_scale": preset["guidance_scale"],
        "seed": seed,
        "return_frames": True,
        "return_trajectory_latents": True,
        "return_trajectory_decoded": False,
        "save_output": output_dir is not None,
    }
    if preset.get("guidance_scale_2") is not None:
        kwargs["guidance_scale_2"] = preset["guidance_scale_2"]
    if preset.get("image_path") is not None:
        kwargs["image_path"] = preset["image_path"]
    if output_dir is not None:
        kwargs["output_path"] = str(output_dir)
        kwargs["output_file_name"] = f"{case_name}.mp4"
    return kwargs


def summarize_case_result(
    *,
    case_name: str,
    reference_run: dict[str, Any],
    candidate_run: dict[str, Any],
    psnr_threshold: float,
    trajectory_step_index: int,
) -> dict[str, Any]:
    reference = reference_run["result"]
    candidate = candidate_run["result"]

    output_metrics = summarize_output_frame_metrics(
        extract_result_frames(reference),
        extract_result_frames(candidate),
    )
    stage_metrics = summarize_stage_trajectory_data(
        _extract_stage_trajectory_data(reference),
        _extract_stage_trajectory_data(candidate),
        step_index=trajectory_step_index,
    )
    aggregate_psnr = float(output_metrics["all_frames_metrics"]["psnr_db"])

    return {
        "case": case_name,
        "psnr_threshold": psnr_threshold,
        "psnr_passed": aggregate_psnr >= psnr_threshold,
        "aggregate_psnr_db": aggregate_psnr,
        "reference_generation": {
            key: value for key, value in reference_run.items() if key != "result"
        }
        | {"output_file_path": reference.output_file_path},
        "candidate_generation": {
            key: value for key, value in candidate_run.items() if key != "result"
        }
        | {"output_file_path": candidate.output_file_path},
        "output_metrics": output_metrics,
        "stage_metrics": stage_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_PRESETS.keys()),
        default=["t2v", "ti2v"],
    )
    parser.add_argument("--model-path", default="Lightricks/LTX-2.3")
    parser.add_argument("--model-id")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--attention-backend")
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-output-dir")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--trajectory-step-index", type=int, default=-1)
    parser.add_argument("--psnr-threshold", type=float, default=35.0)
    parser.add_argument("--reference-num-gpus", type=int, default=1)
    parser.add_argument("--reference-tp-size", type=int, default=1)
    parser.add_argument("--candidate-num-gpus", type=int, default=2)
    parser.add_argument("--candidate-tp-size", type=int, default=2)
    parser.add_argument(
        "--ltx2-two-stage-device-mode",
        default="original",
        choices=("original", "snapshot", "resident"),
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    args = parser.parse_args()

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    save_root = None
    if args.save_output_dir:
        save_root = Path(args.save_output_dir).expanduser().resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    reference_server_kwargs = build_server_kwargs(
        args,
        num_gpus=args.reference_num_gpus,
        tp_size=args.reference_tp_size,
        variant_index=0,
    )
    candidate_server_kwargs = build_server_kwargs(
        args,
        num_gpus=args.candidate_num_gpus,
        tp_size=args.candidate_tp_size,
        variant_index=1,
    )

    results: dict[str, Any] = {}
    for case_name in args.cases:
        preset = CASE_PRESETS[case_name]
        case_save_root = save_root / case_name if save_root is not None else None
        reference_sampling_kwargs = build_sampling_kwargs(
            case_name,
            preset,
            seed=args.seed,
            output_dir=case_save_root / "reference" if case_save_root else None,
        )
        candidate_sampling_kwargs = build_sampling_kwargs(
            case_name,
            preset,
            seed=args.seed,
            output_dir=case_save_root / "candidate" if case_save_root else None,
        )

        reference_run = run_variant(
            server_kwargs=reference_server_kwargs,
            sampling_kwargs=reference_sampling_kwargs,
            fp4_gemm_backend=None,
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        )
        candidate_run = run_variant(
            server_kwargs=candidate_server_kwargs,
            sampling_kwargs=candidate_sampling_kwargs,
            fp4_gemm_backend=None,
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        )
        results[case_name] = summarize_case_result(
            case_name=case_name,
            reference_run=reference_run,
            candidate_run=candidate_run,
            psnr_threshold=args.psnr_threshold,
            trajectory_step_index=args.trajectory_step_index,
        )

    summary = {
        "model_path": args.model_path,
        "seed": args.seed,
        "psnr_threshold": args.psnr_threshold,
        "server_kwargs": {
            "reference": reference_server_kwargs,
            "candidate": candidate_server_kwargs,
        },
        "cases": results,
    }
    output_json.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    compact = {
        case_name: {
            "aggregate_psnr_db": case_result["aggregate_psnr_db"],
            "psnr_passed": case_result["psnr_passed"],
        }
        for case_name, case_result in results.items()
    }
    print(json.dumps({"output_json": str(output_json), "cases": compact}, indent=2))


if __name__ == "__main__":
    main()
