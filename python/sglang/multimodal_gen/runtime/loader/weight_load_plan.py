from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightLoadPlan:
    """Device plan for checkpoint loading, before runtime residency takes over."""

    checkpoint_load_device: torch.device
    weight_postprocess_device: torch.device | None = None
    defer_component_cpu_offload: bool = False

    @classmethod
    def for_component(
        cls,
        *,
        checkpoint_load_device: torch.device,
        needs_device_weight_postprocess: bool,
        component_cpu_offload: bool,
    ) -> "WeightLoadPlan":
        weight_postprocess_device = (
            checkpoint_load_device if needs_device_weight_postprocess else None
        )
        return cls(
            checkpoint_load_device=checkpoint_load_device,
            weight_postprocess_device=weight_postprocess_device,
            defer_component_cpu_offload=(
                needs_device_weight_postprocess and component_cpu_offload
            ),
        )
