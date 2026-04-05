from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

LTX_REPO_ROOT = Path(os.environ.get("LTX_REPO_ROOT", "/tmp/LTX-2-official"))
CHECKPOINT_GLOBS = (
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-20b-dev.safetensors",
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-22b-dev.safetensors",
)
GEMMA_GLOB = (
    "/root/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-*"
)
OUTPUT_PATH = Path(
    os.environ.get("LTX23_OUTPUT_PATH", "/tmp/ltx23_official_one_stage_sunset.mp4")
)
PROMPT = os.environ.get("LTX23_PROMPT", "A beautiful sunset over the ocean")
STREAMING_PREFETCH_COUNT = int(os.environ.get("LTX23_STREAMING_PREFETCH_COUNT", "1"))


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


def main() -> None:
    checkpoint_path = resolve_first_existing_path(CHECKPOINT_GLOBS)
    gemma_root = resolve_single_path(GEMMA_GLOB)

    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-core/src"))
    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-pipelines/src"))

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS
    from ltx_pipelines.utils.media_io import encode_video

    def _patched_prompt_encoder_call(
        self,
        prompts,
        *,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=42,
        streaming_prefetch_count=None,
    ):
        text_encoder = self._text_encoder_builder.build(
            device=torch.device("cpu"),
            dtype=self._dtype,
        ).eval()
        if enhance_first_prompt:
            prompts = list(prompts)
            prompts[0] = generate_enhanced_prompt(
                text_encoder,
                prompts[0],
                enhance_prompt_image,
                seed=enhance_prompt_seed,
            )
        raw_outputs = [text_encoder.encode(p) for p in prompts]
        del text_encoder

        torch.cuda.empty_cache()

        with gpu_model(
            self._embeddings_processor_builder.build(
                device=self._device,
                dtype=self._dtype,
            )
            .to(self._device)
            .eval()
        ) as embeddings_processor:
            return [
                embeddings_processor.process_hidden_states(hs, mask)
                for hs, mask in raw_outputs
            ]

    from ltx_pipelines.utils.blocks import generate_enhanced_prompt, gpu_model

    PromptEncoder.__call__ = _patched_prompt_encoder_call

    params = LTX_2_3_PARAMS
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(checkpoint_path),
        gemma_root=str(gemma_root),
        loras=[],
    )
    video, audio = pipeline(
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=params.seed,
        height=params.stage_1_height,
        width=params.stage_1_width,
        num_frames=params.num_frames,
        frame_rate=params.frame_rate,
        num_inference_steps=params.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=params.video_guider_params.cfg_scale,
            stg_scale=params.video_guider_params.stg_scale,
            rescale_scale=params.video_guider_params.rescale_scale,
            modality_scale=params.video_guider_params.modality_scale,
            skip_step=params.video_guider_params.skip_step,
            stg_blocks=list(params.video_guider_params.stg_blocks),
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=params.audio_guider_params.cfg_scale,
            stg_scale=params.audio_guider_params.stg_scale,
            rescale_scale=params.audio_guider_params.rescale_scale,
            modality_scale=params.audio_guider_params.modality_scale,
            skip_step=params.audio_guider_params.skip_step,
            stg_blocks=list(params.audio_guider_params.stg_blocks),
        ),
        images=[],
        streaming_prefetch_count=STREAMING_PREFETCH_COUNT,
    )
    encode_video(
        video=video,
        fps=params.frame_rate,
        audio=audio,
        output_path=str(OUTPUT_PATH),
        video_chunks_number=1,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
