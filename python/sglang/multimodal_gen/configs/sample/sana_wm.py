# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class SanaWMSamplingParams(SamplingParams):
    height: int = 704
    width: int = 1280
    num_frames: int = 961
    fps: int = 16
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    generator_device: str = "cuda"

    action: str | None = None
    actions: list[list[str]] | None = None
    camera_path: str | None = None
    intrinsics_path: str | None = None
    translation_speed: float = 0.04
    rotation_speed_deg: float = 1.2
    denoising_step_list: list[int] = field(
        default_factory=lambda: [1000, 960, 889, 727, 0]
    )
    num_frame_per_block: int = 3
    num_cached_blocks: int = 2
    sink_size: int = 1

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["sana_wm_denoising_step_list"] = list(self.denoising_step_list)
        extra["sana_wm_num_frame_per_block"] = int(self.num_frame_per_block)
        extra["sana_wm_num_cached_blocks"] = int(self.num_cached_blocks)
        extra["sana_wm_sink_size"] = int(self.sink_size)
        extra["sana_wm_translation_speed"] = float(self.translation_speed)
        extra["sana_wm_rotation_speed_deg"] = float(self.rotation_speed_deg)
        return extra

    def _adjust(self, server_args):
        self.adjust_frames = False
        super()._adjust(server_args)
        self.height = 704
        self.width = 1280
        self.adjust_frames = False
        if self.action is not None:
            self.condition_inputs["action"] = self.action
        if self.actions is not None:
            self.condition_inputs["camera_actions"] = self.actions
        if self.camera_path is not None:
            self.condition_inputs["camera_path"] = self.camera_path
        if self.intrinsics_path is not None:
            self.condition_inputs["intrinsics_path"] = self.intrinsics_path
        self.realtime_chunk_size = int(self.num_frame_per_block)
