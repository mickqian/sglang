import argparse
import os
import traceback
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client


DEFAULT_IMAGE_URL = (
    "https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/"
    "5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe LTX-2.3 two-stage TI2V split guider against the batched "
            "candidate without changing the real runtime path."
        )
    )
    parser.add_argument("--model-path", default="Lightricks/LTX-2.3")
    parser.add_argument("--pipeline-class-name", default="LTX2TwoStagePipeline")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument(
        "--prompt",
        default=(
            "The man in the picture slowly turns his head, his expression "
            "enigmatic and otherworldly. The camera performs a slow, cinematic "
            "dolly out, focusing on his face. Moody lighting, neon signs glowing "
            "in the background, shallow depth of field."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "blurry, out of focus, overexposed, underexposed, low contrast, "
            "washed out colors, excessive noise, grainy texture"
        ),
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--port", type=int, default=31231)
    parser.add_argument("--scheduler-port", type=int, default=5871)
    parser.add_argument("--master-port", type=int, default=32231)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _build_sampling_params(
    args: argparse.Namespace, generator: DiffGenerator
) -> SamplingParams:
    return SamplingParams.from_user_sampling_params_args(
        args.model_path,
        server_args=generator.server_args,
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


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_LTX2_SPLIT_BATCH_PROBE_OUTPUT"] = str(output_path)

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
        sampling_params = _build_sampling_params(args, generator)
        req = prepare_request(
            server_args=generator.server_args, sampling_params=sampling_params
        )
        generator._send_to_scheduler_and_wait_for_response([req])
        if not output_path.exists():
            raise RuntimeError("split/batched probe did not produce runtime output")
        print(output_path.read_text(encoding="utf-8"), end="", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("SGLANG_LTX2_SPLIT_BATCH_PROBE_OUTPUT", None)
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
