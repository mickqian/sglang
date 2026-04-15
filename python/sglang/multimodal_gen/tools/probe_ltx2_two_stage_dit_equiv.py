import argparse
import json
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingContext,
    LTX2DenoisingStage,
)
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
        seconds=args.seconds,
    )


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
    parser.add_argument("--negative-prompt", default=None)
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
    return parser.parse_args()


def _install_probe_hooks() -> dict[str, object]:
    state: dict[str, object] = {"cfg": None, "aux": None}
    orig_cfg_batched = LTX2DenoisingStage._run_ltx2_cfg_batched_forward
    orig_cfg_seq = LTX2DenoisingStage._run_ltx2_cfg_sequential_forward
    orig_aux_batched = LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward
    orig_aux_seq = LTX2DenoisingStage._run_ltx2_stage1_aux_forward
    orig_step = LTX2DenoisingStage._run_denoising_step

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

    def patched_run_step(
        self,
        ctx: LTX2DenoisingContext,
        step,
        batch,
        server_args,
    ):
        result = orig_step(self, ctx, step, batch, server_args)
        if ctx.stage == "stage1" and int(step.step_index) == 0 and state["cfg"] is not None:
            raise ProbeComplete
        return result

    LTX2DenoisingStage._run_ltx2_cfg_batched_forward = patched_cfg_batched
    LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward = patched_aux_batched
    LTX2DenoisingStage._run_denoising_step = patched_run_step

    return {
        "state": state,
        "orig_cfg_batched": orig_cfg_batched,
        "orig_aux_batched": orig_aux_batched,
        "orig_step": orig_step,
    }


def _restore_probe_hooks(handles: dict[str, object]) -> None:
    LTX2DenoisingStage._run_ltx2_cfg_batched_forward = handles["orig_cfg_batched"]
    LTX2DenoisingStage._run_ltx2_stage1_batched_aux_forward = handles[
        "orig_aux_batched"
    ]
    LTX2DenoisingStage._run_denoising_step = handles["orig_step"]


def main() -> None:
    args = _parse_args()
    handles = _install_probe_hooks()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    req = prepare_request(server_args=generator.server_args, sampling_params=sampling_params)

    try:
        generator._send_to_scheduler_and_wait_for_response([req])
    except ProbeComplete:
        pass
    finally:
        _restore_probe_hooks(handles)

    state = handles["state"]
    if state["cfg"] is None:
        raise RuntimeError("probe did not capture stage1 cfg outputs")
    output_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
