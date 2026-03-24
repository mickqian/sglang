# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
V = StageValidators


class WanS2VExecutionStage(PipelineStage):
    """Thin execution stage around the official Wan S2V engine."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_path", batch.image_path, V.not_none)
        result.add_check("audio_path", batch.audio_path, V.not_none)
        return result

    def _get_single_path(self, value, field_name: str) -> str:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Wan S2V expects exactly one {field_name}")
            value = value[0]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Wan S2V expects {field_name} as a non-empty string")
        if not os.path.exists(value):
            raise FileNotFoundError(f"{field_name} not found: {value}")
        return value

    def _normalize_output_video(self, output) -> torch.Tensor:
        tensor = output if isinstance(output, torch.Tensor) else torch.as_tensor(output)
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"Official Wan S2V engine returned unexpected output shape: {tuple(tensor.shape)}"
            )
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[1] in (1, 3, 4):
            tensor = tensor.permute(1, 0, 2, 3).contiguous()
        return tensor.unsqueeze(0)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image_path = self._get_single_path(batch.image_path, "image_path")
        audio_path = self._get_single_path(batch.audio_path, "audio_path")
        pose_video_path = batch.pose_video_path
        if pose_video_path is not None:
            pose_video_path = self._get_single_path(pose_video_path, "pose_video_path")

        seed = batch.seed if batch.seed is not None else 42
        prompt = batch.prompt or ""
        output = self.engine.generate(
            prompt=prompt,
            image_path=image_path,
            audio_path=audio_path,
            pose_video_path=pose_video_path,
            num_clip=batch.num_clip,
            num_frames=batch.num_frames,
            guidance_scale=batch.guidance_scale,
            num_inference_steps=batch.num_inference_steps,
            seed=seed,
        )
        if output is None:
            raise RuntimeError("Official Wan S2V engine returned no output")
        batch.output = self._normalize_output_video(output)
        if sf is not None:
            try:
                audio_np, sample_rate = sf.read(
                    audio_path, dtype="float32", always_2d=False
                )
                if audio_np is not None:
                    batch.audio = torch.from_numpy(audio_np)
                    batch.audio_sample_rate = int(sample_rate)
            except Exception as exc:
                logger.warning("Failed to load source audio for output muxing: %s", exc)
        return batch


class WanS2VBeforeDenoisingStage(PipelineStage):
    """Prepare Wan S2V conditions for the standard denoising loop."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def _get_single_path(self, value, field_name: str) -> str:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Wan S2V expects exactly one {field_name}")
            value = value[0]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Wan S2V expects {field_name} as a non-empty string")
        if not os.path.exists(value):
            raise FileNotFoundError(f"{field_name} not found: {value}")
        return value

    def _generate_seed_and_generator(self, batch: Req) -> None:
        seed = batch.seed if batch.seed is not None else 42
        batch.seeds = [seed]
        generator_device = (
            "cpu" if batch.generator_device == "cpu" else current_platform.device_type
        )
        batch.generator = torch.Generator(generator_device).manual_seed(seed)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_path", batch.image_path, V.not_none)
        result.add_check("audio_path", batch.audio_path, V.not_none)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not getattr(self.transformer, "supports_standard_denoising", False):
            raise RuntimeError(
                "Wan S2V standard denoising path requested, but transformer does not support it"
            )

        image_path = self._get_single_path(batch.image_path, "image_path")
        audio_path = self._get_single_path(batch.audio_path, "audio_path")
        pose_video_path = batch.pose_video_path
        if pose_video_path is not None:
            pose_video_path = self._get_single_path(pose_video_path, "pose_video_path")

        self._generate_seed_and_generator(batch)

        prepared = self.transformer.prepare_standard_s2v_inputs(
            prompt=batch.prompt or "",
            negative_prompt=batch.negative_prompt,
            image_path=image_path,
            audio_path=audio_path,
            pose_video_path=pose_video_path,
            num_clip=batch.num_clip,
            num_frames=batch.num_frames,
            seed=batch.seeds[0],
        )
        latents = self.transformer.prepare_standard_s2v_latents(
            latent_shape=prepared["latent_shape"],
            generator=batch.generator,
        )

        batch.prompt_embeds = prepared["prompt_embeds"]
        if batch.do_classifier_free_guidance:
            batch.negative_prompt_embeds = prepared["negative_prompt_embeds"]
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = prepared["height"]
        batch.width = prepared["width"]
        batch.audio = prepared["audio"]
        batch.audio_sample_rate = prepared["audio_sample_rate"]
        batch.extra["wan_s2v"] = {
            "ref_latents": prepared["ref_latents"],
            "motion_latents": prepared["motion_latents"],
            "cond_states": prepared["cond_states"],
            "audio_input": prepared["audio_input"],
            "motion_frames": prepared["motion_frames"],
            "drop_motion_frames": prepared["drop_motion_frames"],
            "infer_frames": prepared["infer_frames"],
        }
        return batch


class WanS2VDecodingStage(PipelineStage):
    """Decode Wan S2V latents using the official Wan VAE."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        result.add_check("wan_s2v_extra", batch.extra.get("wan_s2v"), V.not_none)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        extra = batch.extra["wan_s2v"]
        batch.output = self.transformer.decode_standard_s2v_output(
            latents=batch.latents,
            ref_latents=extra["ref_latents"],
            motion_latents=extra["motion_latents"],
            infer_frames=extra["infer_frames"],
            drop_motion_frames=extra["drop_motion_frames"],
        )
        return batch
