import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2HalveResolutionStage,
    LTX2LoRASwitchStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def build_ltx2_native_sigmas(
    batch: Req,
    server_args: ServerArgs,
    *,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.FloatTensor:
    # Match the official diffusers==0.37.0 LTX2Pipeline stage-1 schedule used by
    # the trusted two-stage sunset baseline:
    # 1. start from a linear sigma ramp [1.0, 1 / steps]
    # 2. apply dynamic exponential shifting with mu=calculate_shift(max_image_seq_len)
    # 3. stretch the schedule to terminate at shift_terminal=0.1
    #
    # For the shipped scheduler config, calculate_shift(max_image_seq_len, ...)
    # equals max_shift, so the stage-1 schedule is resolution-independent.
    sigmas = torch.linspace(
        1.0,
        1.0 / int(batch.num_inference_steps),
        int(batch.num_inference_steps),
        dtype=torch.float32,
    )
    sigmas = math.exp(max_shift) / (math.exp(max_shift) + (1 / sigmas - 1))

    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])
    return sigmas.to(torch.float32)


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"
        sigmas = build_ltx2_native_sigmas(batch, server_args)
        batch.sigmas = sigmas[:-1].tolist()
        return batch


def _add_ltx2_front_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stages(
        [
            InputValidationStage(),
            TextEncodingStage(
                text_encoders=[pipeline.get_module("text_encoder")],
                tokenizers=[pipeline.get_module("tokenizer")],
            ),
            LTX2TextConnectorStage(connectors=pipeline.get_module("connectors")),
        ]
    )


def _add_ltx2_stage1_generation_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stage(LTX2SigmaPreparationStage())
    pipeline.add_standard_timestep_preparation_stage()
    pipeline.add_stages(
        [
            LTX2AVLatentPreparationStage(
                scheduler=pipeline.get_module("scheduler"),
                transformer=pipeline.get_module("transformer"),
                audio_vae=pipeline.get_module("audio_vae"),
            ),
            LTX2AVDenoisingStage(
                transformer=pipeline.get_module("transformer"),
                scheduler=pipeline.get_module("scheduler"),
                vae=pipeline.get_module("vae"),
                audio_vae=pipeline.get_module("audio_vae"),
                pipeline=pipeline,
            ),
        ]
    )


def _add_ltx2_decoding_stage(pipeline: ComposedPipelineBase):
    pipeline.add_stage(
        LTX2AVDecodingStage(
            vae=pipeline.get_module("vae"),
            audio_vae=pipeline.get_module("audio_vae"),
            vocoder=pipeline.get_module("vocoder"),
            pipeline=pipeline,
        )
    )


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        if sigmas is not None and timesteps is None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self.num_inference_steps = len(timesteps)
            self.timesteps = timesteps
            self.sigmas = sigmas
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class _BaseLTX2Pipeline(LoRAPipeline):
    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)


class LTX2Pipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        _add_ltx2_stage1_generation_stages(self)
        _add_ltx2_decoding_stage(self)


class LTX2TwoStagePipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2TwoStagePipeline"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]

    def initialize_pipeline(self, server_args: ServerArgs):
        super().initialize_pipeline(server_args)
        upsampler_path = server_args.component_paths.get("spatial_upsampler")
        if not upsampler_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --spatial-upsampler-path "
                "(component_paths['spatial_upsampler'])."
            )
        upsampler_path = os.path.expanduser(upsampler_path)
        upsampler_root = upsampler_path
        upsampler_subfolder = None
        if os.path.isdir(upsampler_path):
            upsampler_root = os.path.dirname(upsampler_path.rstrip("/"))
            upsampler_subfolder = os.path.basename(upsampler_path.rstrip("/"))

        module = LTX2LatentUpsamplerModel.from_pretrained(
            upsampler_root,
            subfolder=upsampler_subfolder,
            revision=server_args.revision,
            torch_dtype=torch.bfloat16,
        )
        self.modules["spatial_upsampler"] = module
        self.memory_usages["spatial_upsampler"] = 0.0

        distilled_lora_path = server_args.component_paths.get("distilled_lora")
        if not distilled_lora_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --distilled-lora-path "
                "(component_paths['distilled_lora'])."
            )
        self._distilled_lora_path = distilled_lora_path
        self._stage1_lora_path = server_args.lora_path
        self._stage1_lora_scale = float(server_args.lora_scale)
        self._active_lora_phase = None

    def switch_lora_phase(self, phase: str) -> None:
        if phase == self._active_lora_phase:
            return

        if phase == "stage1":
            if self._stage1_lora_path:
                self.set_lora(
                    lora_nickname="ltx2_stage1_base",
                    lora_path=self._stage1_lora_path,
                    target="transformer",
                    strength=self._stage1_lora_scale,
                )
            else:
                self.unmerge_lora_weights(target="transformer")
        elif phase == "stage2":
            lora_nicknames = []
            lora_paths = []
            lora_strengths = []
            lora_targets = []
            if self._stage1_lora_path:
                lora_nicknames.append("ltx2_stage1_base")
                lora_paths.append(self._stage1_lora_path)
                lora_strengths.append(self._stage1_lora_scale)
                lora_targets.append("transformer")
            lora_nicknames.append("ltx2_stage2_distilled")
            lora_paths.append(self._distilled_lora_path)
            lora_strengths.append(1.0)
            lora_targets.append("transformer")
            self.set_lora(
                lora_nickname=lora_nicknames,
                lora_path=lora_paths,
                target=lora_targets,
                strength=lora_strengths,
            )
        else:
            raise ValueError(f"Unknown LTX2 two-stage LoRA phase: {phase}")

        self._active_lora_phase = phase

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        self.add_stage(
            LTX2LoRASwitchStage(pipeline=self, phase="stage1"),
            stage_name="ltx2_lora_switch_stage1",
        )
        _add_ltx2_stage1_generation_stages(self)
        self.add_stages(
            [
                LTX2UpsampleStage(
                    spatial_upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                (
                    LTX2LoRASwitchStage(pipeline=self, phase="stage2"),
                    "ltx2_lora_switch_stage2",
                ),
                LTX2RefinementStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    distilled_sigmas=self.STAGE_2_DISTILLED_SIGMA_VALUES,
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
            ]
        )
        _add_ltx2_decoding_stage(self)


EntryClass = [LTX2Pipeline, LTX2TwoStagePipeline]
