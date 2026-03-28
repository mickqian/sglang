# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


class WanS2VOfficialScheduler(BaseScheduler):
    _aliases = ["WanS2VOfficialScheduler"]

    def __init__(self, inner) -> None:
        self._inner = inner
        self.config = inner.config
        self.order = inner.order
        self.num_train_timesteps = int(self.config.num_train_timesteps)
        BaseScheduler.__init__(self)

    @staticmethod
    def _resolve_existing_path(component_model_path: str, path_value: str | None) -> str:
        if not path_value:
            raise ValueError("WanS2VOfficialScheduler config is missing a required path")
        path_value = os.path.expanduser(path_value)
        if os.path.isabs(path_value):
            resolved = path_value
        else:
            resolved = os.path.join(component_model_path, path_value)
        if not os.path.exists(resolved):
            raise ValueError(f"Resolved path does not exist: {resolved}")
        return resolved

    @classmethod
    def from_component_path(
        cls,
        component_model_path: str,
        server_args,
        config: dict[str, Any],
    ):
        del server_args
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        module_unipc = importlib.import_module("wan.utils.fm_solvers_unipc")
        scheduler_cls = getattr(module_unipc, "FlowUniPCMultistepScheduler")
        scheduler_kwargs = dict(config)
        scheduler_kwargs.pop("wan_code_root", None)
        scheduler_kwargs.pop("wan_task_name", None)
        return cls(inner=scheduler_cls(**scheduler_kwargs))

    def set_shift(self, shift: float) -> None:
        self.config.shift = shift

    @property
    def begin_index(self):
        return self._inner.begin_index

    @property
    def step_index(self):
        return self._inner.step_index

    @property
    def timesteps(self):
        return self._inner.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._inner.timesteps = value

    @property
    def sigmas(self):
        return self._inner.sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._inner.sigmas = value

    @property
    def num_inference_steps(self):
        return self._inner.num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, value):
        self._inner.num_inference_steps = value

    def set_begin_index(self, begin_index: int = 0):
        return self._inner.set_begin_index(begin_index)

    def set_timesteps(self, *args, **kwargs):
        return self._inner.set_timesteps(*args, **kwargs)

    def scale_model_input(self, *args, **kwargs):
        return self._inner.scale_model_input(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._inner.step(*args, **kwargs)

    def add_noise(self, *args, **kwargs):
        return self._inner.add_noise(*args, **kwargs)

    @property
    def init_noise_sigma(self):
        return self._inner.init_noise_sigma


EntryClass = WanS2VOfficialScheduler
