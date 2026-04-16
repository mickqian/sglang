import argparse
import json
import os
from pathlib import Path
import traceback

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingContext,
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class ProbeComplete(RuntimeError):
    pass


def _tensor_report(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    diff = candidate - reference
    mse = torch.mean(diff.square()).item()
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), candidate.reshape(1, -1), dim=1
    ).item()
    return {
        "shape": list(reference.shape),
        "bit_exact": bool(torch.equal(reference, candidate)),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(mse**0.5),
        "cosine": float(cosine),
    }


def _cfg_report(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    return {
        "video_uncond": _tensor_report(reference[0], candidate[0]),
        "video_cond": _tensor_report(reference[1], candidate[1]),
        "audio_uncond": _tensor_report(reference[2], candidate[2]),
        "audio_cond": _tensor_report(reference[3], candidate[3]),
    }


def _aux_report(
    reference: dict[str, tuple[torch.Tensor, torch.Tensor]],
    candidate: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    report: dict[str, object] = {}
    for key, ref_pair in reference.items():
        cand_pair = candidate[key]
        report[key] = {
            "video": _tensor_report(ref_pair[0], cand_pair[0]),
            "audio": _tensor_report(ref_pair[1], cand_pair[1]),
        }
    return report


def _build_sampling_params(args: argparse.Namespace, server_args: ServerArgs) -> SamplingParams:
    return SamplingParams.from_user_sampling_params_args(
        server_args.model_path,
        server_args=server_args,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image_path,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        save_output=False,
        return_frames=False,
    )


def _merge_model_kwargs(
    *items: dict[str, object],
) -> dict[str, object]:
    if len(items) < 2:
        raise ValueError("Need at least 2 kwargs dicts to merge")
    merged: dict[str, object] = {}
    all_keys = set()
    for d in items:
        all_keys |= d.keys()
    for key in all_keys:
        values = [d.get(key) for d in items]
        if all(
            isinstance(v, torch.Tensor) and v.ndim > 0
            for v in values
        ) and len({v.shape[1:] for v in values}) == 1:
            merged[key] = torch.cat(values, dim=0)
            continue
        merged[key] = values[0]
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe LTX-2.3 two-stage stage1 DiT current_model equivalence between "
            "the current batched path and the sequential reference."
        )
    )
    parser.add_argument("--model-path", default="Lightricks/LTX-2.3")
    parser.add_argument("--pipeline-class-name", default="LTX2TwoStagePipeline")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--prompt", default="A curious raccoon")
    parser.add_argument("--negative-prompt", default="low quality, blurry")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--port", type=int, default=31221)
    parser.add_argument("--scheduler-port", type=int, default=5861)
    parser.add_argument("--master-port", type=int, default=32221)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--force-sdpa-math",
        action="store_true",
        help="Force SDPA math backend to test if kernel selection causes B=3 drift",
    )
    return parser.parse_args()


def _install_probe_hooks() -> dict[str, object]:
    state: dict[str, object] = {
        "cfg": None,
        "aux": None,
        "official_cfg": None,
        "current_stage": None,
        "in_model_forward_probe": False,
        "model_forward_calls": [],
        "sequential_pair": [],
    }
    orig_cfg_batched = LTX2DenoisingStage._run_ltx2_cfg_batched_forward
    orig_cfg_seq = LTX2DenoisingStage._run_ltx2_cfg_sequential_forward
    orig_aux_batched = LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward
    orig_aux_seq = LTX2DenoisingStage._run_ltx2_stage1_aux_forward
    orig_model_forward = LTX2DenoisingStage._run_ltx2_model_forward
    orig_step = LTX2DenoisingStage._run_denoising_step

    def _slice_model_kwargs(
        model_kwargs: dict[str, object], start: int, end: int
    ) -> dict[str, object]:
        sliced: dict[str, object] = {}
        for key, value in model_kwargs.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= end:
                sliced[key] = value[start:end]
            elif isinstance(value, tuple) and value and len(value) >= end:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def patched_cfg_batched(self, **kwargs):
        candidate = orig_cfg_batched(self, **kwargs)
        reference = orig_cfg_seq(self, **kwargs)
        state["cfg"] = _cfg_report(reference, candidate)
        return candidate

    def patched_aux_batched(self, **kwargs):
        pass_specs = kwargs["pass_specs"]
        candidate = orig_aux_batched(self, **kwargs)
        reference: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for pass_name, perturbation_config in pass_specs:
            reference[pass_name] = orig_aux_seq(
                self,
                step=kwargs["step"],
                latent_model_input=kwargs["latent_model_input"],
                audio_latent_model_input=kwargs["audio_latent_model_input"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                audio_encoder_hidden_states=kwargs["audio_encoder_hidden_states"],
                timestep_video=kwargs["timestep_video"],
                timestep_audio=kwargs["timestep_audio"],
                encoder_attention_mask=kwargs["encoder_attention_mask"],
                ctx=kwargs["ctx"],
                batch=kwargs["batch"],
                audio_num_frames_latent=kwargs["audio_num_frames_latent"],
                video_coords=kwargs["video_coords"],
                audio_coords=kwargs["audio_coords"],
                prompt_timestep_video=kwargs["prompt_timestep_video"],
                prompt_timestep_audio=kwargs["prompt_timestep_audio"],
                video_self_attention_mask=kwargs["video_self_attention_mask"],
                audio_self_attention_mask=kwargs["audio_self_attention_mask"],
                a2v_cross_attention_mask=kwargs["a2v_cross_attention_mask"],
                v2a_cross_attention_mask=kwargs["v2a_cross_attention_mask"],
                skip_video_self_attn_blocks=perturbation_config[
                    "skip_video_self_attn_blocks"
                ],
                skip_audio_self_attn_blocks=perturbation_config[
                    "skip_audio_self_attn_blocks"
                ],
                disable_a2v_cross_attn=bool(
                    perturbation_config["skip_a2v_cross_attn"]
                ),
                disable_v2a_cross_attn=bool(
                    perturbation_config["skip_v2a_cross_attn"]
                ),
            )
        state["aux"] = _aux_report(reference, candidate)
        return candidate

    def patched_model_forward(self, *, step, model_kwargs):
        candidate = orig_model_forward(step=step, model_kwargs=model_kwargs)
        hidden_states = model_kwargs.get("hidden_states")
        encoder_hidden_states = model_kwargs.get("encoder_hidden_states")
        if len(state["model_forward_calls"]) < 16:
            state["model_forward_calls"].append(
                {
                    "stage": state["current_stage"],
                    "step_index": int(step.step_index),
                    "hidden_states_shape": (
                        list(hidden_states.shape)
                        if isinstance(hidden_states, torch.Tensor)
                        else None
                    ),
                    "encoder_hidden_states_shape": (
                        list(encoder_hidden_states.shape)
                        if isinstance(encoder_hidden_states, torch.Tensor)
                        else None
                    ),
                    "has_perturbation_configs": model_kwargs.get(
                        "perturbation_configs"
                    )
                    is not None,
                }
            )
        should_probe_official_cfg = (
            state["current_stage"] == "stage1"
            and state["official_cfg"] is None
            and not state["in_model_forward_probe"]
            and isinstance(hidden_states, torch.Tensor)
            and isinstance(encoder_hidden_states, torch.Tensor)
            and model_kwargs.get("perturbation_configs") is None
        )
        if not should_probe_official_cfg:
            return candidate

        if hidden_states.shape[0] == 1 and encoder_hidden_states.shape[0] == 1:
            sequential_pair: list[tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor]]] = state["sequential_pair"]  # type: ignore[assignment]
            sequential_pair.append(
                (
                    {
                        key: value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in model_kwargs.items()
                    },
                    (candidate[0].detach().clone(), candidate[1].detach().clone()),
                )
            )
            if len(sequential_pair) < 2:
                return candidate
            first_kwargs, first_outputs = sequential_pair[0]
            second_kwargs, second_outputs = sequential_pair[1]
            batched_kwargs = _merge_model_kwargs(first_kwargs, second_kwargs)
            state["in_model_forward_probe"] = True
            try:
                batched_outputs = orig_model_forward(
                    step=step,
                    model_kwargs=batched_kwargs,
                )
            finally:
                state["in_model_forward_probe"] = False
            state["official_cfg"] = _cfg_report(
                (
                    first_outputs[0],
                    second_outputs[0],
                    first_outputs[1],
                    second_outputs[1],
                ),
                (
                    batched_outputs[0][0:1],
                    batched_outputs[0][1:2],
                    batched_outputs[1][0:1],
                    batched_outputs[1][1:2],
                ),
            )
            return candidate

        if not (hidden_states.shape[0] == 2 and encoder_hidden_states.shape[0] == 2):
            return candidate

        uncond_kwargs = _slice_model_kwargs(model_kwargs, 0, 1)
        cond_kwargs = _slice_model_kwargs(model_kwargs, 1, 2)
        state["in_model_forward_probe"] = True
        try:
            uncond = orig_model_forward(
                step=step,
                model_kwargs=uncond_kwargs,
            )
            cond = orig_model_forward(
                step=step,
                model_kwargs=cond_kwargs,
            )
        finally:
            state["in_model_forward_probe"] = False

        state["official_cfg"] = _cfg_report(
            (uncond[0], cond[0], uncond[1], cond[1]),
            (
                candidate[0][0:1],
                candidate[0][1:2],
                candidate[1][0:1],
                candidate[1][1:2],
            ),
        )

        # B=3 test: batch [uncond, cond, uncond_dup] and compare each
        # item against individual B=1 references.
        uncond_kwargs_clone = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in uncond_kwargs.items()
        }
        b3_kwargs = _merge_model_kwargs(uncond_kwargs, cond_kwargs, uncond_kwargs_clone)
        state["in_model_forward_probe"] = True
        try:
            b3_out = orig_model_forward(step=step, model_kwargs=b3_kwargs)
        finally:
            state["in_model_forward_probe"] = False
        state["b3_test"] = {
            "item0_vs_ref_uncond": {
                "video": _tensor_report(uncond[0], b3_out[0][0:1]),
                "audio": _tensor_report(uncond[1], b3_out[1][0:1]),
            },
            "item1_vs_ref_cond": {
                "video": _tensor_report(cond[0], b3_out[0][1:2]),
                "audio": _tensor_report(cond[1], b3_out[1][1:2]),
            },
            "item2_vs_ref_uncond_dup": {
                "video": _tensor_report(uncond[0], b3_out[0][2:3]),
                "audio": _tensor_report(uncond[1], b3_out[1][2:3]),
            },
            "item0_vs_item2_self_consistency": {
                "video": _tensor_report(b3_out[0][0:1], b3_out[0][2:3]),
                "audio": _tensor_report(b3_out[1][0:1], b3_out[1][2:3]),
            },
        }

        return candidate

    def patched_run_step(
        self,
        ctx: LTX2DenoisingContext,
        step,
        batch,
        server_args,
    ):
        state["current_stage"] = ctx.stage
        try:
            result = orig_step(self, ctx, step, batch, server_args)
        finally:
            state["current_stage"] = None
        if ctx.stage == "stage1" and int(step.step_index) == 0:
            if state["cfg"] is not None or state["official_cfg"] is not None:
                raise ProbeComplete
        return result

    LTX2DenoisingStage._run_ltx2_cfg_batched_forward = patched_cfg_batched
    LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward = patched_aux_batched
    LTX2DenoisingStage._run_ltx2_model_forward = patched_model_forward
    LTX2DenoisingStage._run_denoising_step = patched_run_step

    return {
        "state": state,
        "orig_cfg_batched": orig_cfg_batched,
        "orig_aux_batched": orig_aux_batched,
        "orig_model_forward": orig_model_forward,
        "orig_step": orig_step,
    }


def _restore_probe_hooks(handles: dict[str, object]) -> None:
    LTX2DenoisingStage._run_ltx2_cfg_batched_forward = handles["orig_cfg_batched"]
    LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward = handles[
        "orig_aux_batched"
    ]
    LTX2DenoisingStage._run_ltx2_model_forward = handles["orig_model_forward"]
    LTX2DenoisingStage._run_denoising_step = handles["orig_step"]


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_LTX2_PROBE_OUTPUT"] = str(output_path)
    if args.force_sdpa_math:
        os.environ["SGLANG_FORCE_SDPA_MATH"] = "1"
    if os.getenv("SGLANG_LTX2_PROBE_DISABLE_BF16_REDUCTION", "0") == "1":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if os.getenv("SGLANG_LTX2_PROBE_DISABLE_TF32", "0") == "1":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    generator = None
    try:
        generator = DiffGenerator.from_pretrained(
            model_path=args.model_path,
            pipeline_class_name=args.pipeline_class_name,
            num_gpus=args.num_gpus,
            log_level=args.log_level,
            attention_backend=args.attention_backend,
            port=args.port,
            scheduler_port=args.scheduler_port,
            master_port=args.master_port,
        )

        sampling_params = _build_sampling_params(args, generator.server_args)
        req = prepare_request(
            server_args=generator.server_args, sampling_params=sampling_params
        )

        try:
            generator._send_to_scheduler_and_wait_for_response([req])
        except ProbeComplete:
            pass

        if not output_path.exists():
            raise RuntimeError("probe did not produce runtime output")
        print(output_path.read_text(encoding="utf-8"), end="", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("SGLANG_LTX2_PROBE_OUTPUT", None)
        if generator is not None:
            if generator.local_scheduler_process:
                for process in generator.local_scheduler_process:
                    process.terminate()
                    process.join(timeout=5)
                generator.local_scheduler_process = None
            if getattr(generator, "owns_scheduler_client", False):
                try:
                    sync_scheduler_client.close()
                except Exception:
                    pass
                generator.owns_scheduler_client = False


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
