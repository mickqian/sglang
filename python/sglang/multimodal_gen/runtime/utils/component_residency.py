from dataclasses import dataclass
from typing import Any, Sequence

import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class ComponentUse:
    stage_name: str
    component_name: str
    access_kind: str = "forward"
    phase: str | None = None
    preferred_after_request: bool = False
    allow_prefetch: bool = True


@dataclass(slots=True)
class ExecutorResidencyState:
    stages: Sequence[Any] = ()
    stage_index: int = -1
    stage_name: str | None = None
    next_stage_name: str | None = None
    current_use: ComponentUse | None = None
    future_uses: tuple[ComponentUse, ...] = ()
    batch_is_warmup: bool = False
    manager_mode: str = "static"
    dynamic_budget: bool = False
    trace_enabled: bool = False


class ComponentResidencyStrategy:
    name = "resident"

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        self.enter(module)

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        pass

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        self.exit(module)

    def finish_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
        *,
        preferred: bool,
    ) -> None:
        if preferred:
            self.prepare_for_use(module, use, state)
            self.wait_for_use(module, use, state)
        else:
            self.finish_use(module, use, state)

    def enter(self, module: nn.Module) -> None:
        pass

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        pass


class ResidentStrategy(ComponentResidencyStrategy):
    name = "resident"


class VanillaD2HStrategy(ComponentResidencyStrategy):
    name = "vanilla"

    def enter(self, module: nn.Module) -> None:
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            module.to(get_local_torch_device(), non_blocking=True)

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        del next_module
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cuda":
            module.to("cpu")


class LayerwiseOffloadStrategy(ComponentResidencyStrategy):
    name = "layerwise"

    def enter(self, module: nn.Module) -> None:
        if isinstance(module, OffloadableDiTMixin):
            module.prepare_for_next_req()

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        del next_module
        if not isinstance(module, OffloadableDiTMixin):
            return
        for manager in module.layerwise_offload_managers:
            manager.release_all()


class LifecycleAdapterStrategy(ComponentResidencyStrategy):
    """Adapter for model-specific managers that already own residency semantics."""

    name = "adapter"

    def __init__(self, adapter: Any, component_name: str) -> None:
        self.adapter = adapter
        self.component_name = component_name

    def _phase(self, use: ComponentUse) -> str:
        if use.phase in ("stage1", "stage2"):
            return use.phase
        return "stage2" if self.component_name == "transformer_2" else "stage1"

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        del module, state
        enter_phase = getattr(self.adapter, "enter_phase", None)
        if callable(enter_phase):
            enter_phase(self._phase(use))

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        del module, state
        ensure_phase_ready = getattr(self.adapter, "ensure_phase_ready", None)
        if callable(ensure_phase_ready):
            ensure_phase_ready(self._phase(use))

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ExecutorResidencyState,
    ) -> None:
        del module, state
        exit_phase = getattr(self.adapter, "exit_phase", None)
        if callable(exit_phase):
            exit_phase(self._phase(use))


def build_dit_residency_strategy(
    module: nn.Module,
    server_args: ServerArgs,
) -> ComponentResidencyStrategy:
    if (
        isinstance(module, OffloadableDiTMixin)
        and module.layerwise_offload_managers
        and any(manager.enabled for manager in module.layerwise_offload_managers)
    ):
        return LayerwiseOffloadStrategy()
    if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
        return VanillaD2HStrategy()
    return ResidentStrategy()


def is_fsdp_managed_module(module: nn.Module) -> bool:
    return module.__class__.__name__.startswith("FSDP")


def build_component_residency_strategy(
    component_name: str,
    module: nn.Module,
    server_args: ServerArgs,
) -> ComponentResidencyStrategy:
    if component_name in {
        "transformer",
        "transformer_2",
        "video_dit",
        "video_dit_2",
        "audio_dit",
        "dual_tower_bridge",
    }:
        return build_dit_residency_strategy(module, server_args)

    if component_name.startswith("text_encoder"):
        if (
            server_args.text_encoder_cpu_offload
            and not server_args.use_fsdp_inference
            and not is_fsdp_managed_module(module)
        ):
            return VanillaD2HStrategy()
        return ResidentStrategy()

    if component_name == "image_encoder":
        if server_args.image_encoder_cpu_offload and not server_args.use_fsdp_inference:
            return VanillaD2HStrategy()
        return ResidentStrategy()

    if component_name in {
        "vae",
        "video_vae",
        "audio_vae",
        "vocoder",
        "spatial_upsampler",
    }:
        if server_args.vae_cpu_offload and not server_args.use_fsdp_inference:
            return VanillaD2HStrategy()
        return ResidentStrategy()

    return ResidentStrategy()


@dataclass(slots=True)
class SequentialComponent:
    phase: str
    name: str
    module: nn.Module
    strategy: ComponentResidencyStrategy


class SequentialComponentGroup:
    """Lifecycle helper for mutually exclusive components used in phase order."""

    def __init__(self, components: list[SequentialComponent]) -> None:
        self._components_by_phase = {
            component.phase: component for component in components
        }
        self._phase_order = tuple(component.phase for component in components)
        self.active_phase: str | None = None

    def switch_phase(self, phase: str) -> None:
        if phase == self.active_phase:
            return
        if self.active_phase is not None:
            self.exit_phase(self.active_phase, next_phase=phase)
        self.enter_phase(phase)

    def enter_phase(self, phase: str) -> None:
        component = self._components_by_phase[phase]
        component.strategy.enter(component.module)
        self.active_phase = phase

    def exit_phase(self, phase: str, next_phase: str | None = None) -> None:
        component = self._components_by_phase[phase]
        next_component = (
            self._components_by_phase[next_phase] if next_phase is not None else None
        )
        component.strategy.exit(
            component.module,
            next_module=next_component.module if next_component is not None else None,
        )
        if self.active_phase == phase:
            self.active_phase = None

    def finish_request(self, preferred_phase: str | None = None) -> None:
        for phase in self._phase_order:
            component = self._components_by_phase[phase]
            use = ComponentUse(stage_name="", component_name=component.name, phase=phase)
            state = ExecutorResidencyState()
            component.strategy.finish_request(
                component.module,
                use,
                state,
                preferred=phase == preferred_phase,
            )

        self.active_phase = preferred_phase


class PipelineResidencyManager:
    def __init__(self, pipeline: Any, server_args: ServerArgs) -> None:
        self.pipeline = pipeline
        self.server_args = server_args
        self.state = ExecutorResidencyState(
            manager_mode=server_args.component_residency_manager,
            dynamic_budget=server_args.component_residency_dynamic_budget,
            trace_enabled=server_args.component_residency_trace,
        )
        self._stage_names_by_id: dict[int, str] = {}
        self._stage_uses_by_index: list[tuple[ComponentUse, ...]] = []
        self._strategy_cache: dict[str, ComponentResidencyStrategy] = {}
        self._uses_seen: dict[str, ComponentUse] = {}

    @property
    def enabled(self) -> bool:
        return self.server_args.component_residency_manager != "disabled"

    def refresh_pipeline(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._stage_names_by_id = {
            id(stage): name
            for name, stage in getattr(pipeline, "_stage_name_mapping", {}).items()
        }

    def begin_request(
        self,
        stages: Sequence[Any],
        batch: Any,
        server_args: ServerArgs,
    ) -> None:
        if not self.enabled:
            return
        self.server_args = server_args
        self.state = ExecutorResidencyState(
            stages=stages,
            batch_is_warmup=bool(getattr(batch, "is_warmup", False)),
            manager_mode=server_args.component_residency_manager,
            dynamic_budget=server_args.component_residency_dynamic_budget,
            trace_enabled=server_args.component_residency_trace,
        )
        self._stage_uses_by_index = [
            tuple(stage.component_uses(server_args, self.stage_name(stage)))
            for stage in stages
        ]
        self._uses_seen.clear()
        self._trace("request_start", detail=f"stages={len(stages)}")

    def before_stage(
        self,
        stage: Any,
        stage_index: int,
        batch: Any,
        server_args: ServerArgs,
    ) -> None:
        if not self.enabled:
            return
        del batch, server_args
        self.state.stage_index = stage_index
        self.state.stage_name = self.stage_name(stage)
        self.state.next_stage_name = self._next_stage_name(stage_index)
        self.state.future_uses = self._future_uses(stage_index + 1)
        self._trace("stage_enter", detail=f"index={stage_index}")
        for use in self._stage_uses(stage_index):
            self.before_use(use)

    def after_stage(self, stage_index: int) -> None:
        if not self.enabled:
            return
        for use in self._stage_uses(stage_index):
            self.after_use(use)
        self._trace("stage_exit", detail=f"index={stage_index}")
        self.prefetch_future_uses(stage_index + 1)

    def before_use(self, use: ComponentUse) -> None:
        if not self.enabled:
            return
        module = self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        strategy = self.strategy_for(use.component_name, module)
        self.state.current_use = use
        self._uses_seen[use.component_name] = use
        self._trace("prepare", use, strategy, module)
        strategy.prepare_for_use(module, use, self.state)
        self._trace("wait", use, strategy, module)
        strategy.wait_for_use(module, use, self.state)

    def prepare_for_use(self, use: ComponentUse) -> None:
        if not self.enabled or not use.allow_prefetch:
            return
        module = self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        self._uses_seen[use.component_name] = use
        strategy = self.strategy_for(use.component_name, module)
        self._trace("prefetch", use, strategy, module)
        strategy.prepare_for_use(module, use, self.state)

    def after_use(self, use: ComponentUse) -> None:
        if not self.enabled:
            return
        module = self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        if self.state.batch_is_warmup or self._should_keep_after_use(use):
            self._trace(
                "keep",
                use,
                self.strategy_for(use.component_name, module),
                module,
            )
            return
        strategy = self.strategy_for(use.component_name, module)
        self._trace("finish", use, strategy, module)
        strategy.finish_use(module, use, self.state)

    def finish_request(self) -> None:
        if not self.enabled:
            return
        preferred_use = self._preferred_request_end_use()
        for component_name, use in list(self._uses_seen.items()):
            module = self.get_module(component_name)
            if module is None:
                continue
            preferred = (
                preferred_use is not None
                and preferred_use.component_name == component_name
            )
            if not preferred and self._should_keep_for_dynamic_budget(component_name):
                self._trace(
                    "keep",
                    use,
                    self.strategy_for(component_name, module),
                    module,
                    detail="dynamic_budget",
                )
                continue
            strategy = self.strategy_for(component_name, module)
            action = "request_resident" if preferred else "request_finish"
            self._trace(action, use, strategy, module)
            strategy.finish_request(module, use, self.state, preferred=preferred)
        self._trace("request_end")

    def prefetch_future_uses(self, start_index: int) -> None:
        if not self.enabled:
            return
        for index in range(start_index, len(self._stage_uses_by_index)):
            uses = self._stage_uses(index)
            if not uses:
                continue
            for use in uses:
                self.prepare_for_use(use)
            return

    def stage_name(self, stage: Any) -> str:
        return self._stage_names_by_id.get(id(stage), stage.__class__.__name__)

    def component_name_for_module(
        self, module: nn.Module | None, default: str
    ) -> str:
        if module is None:
            return default
        for name, candidate in getattr(self.pipeline, "modules", {}).items():
            if candidate is module:
                return name
        return default

    def get_module(self, component_name: str) -> nn.Module | None:
        module = getattr(self.pipeline, "modules", {}).get(component_name)
        return module if isinstance(module, nn.Module) else None

    def strategy_for(
        self, component_name: str, module: nn.Module
    ) -> ComponentResidencyStrategy:
        strategy = self._strategy_cache.get(component_name)
        if strategy is not None:
            return strategy
        device_manager = getattr(self.pipeline, "_device_manager", None)
        if (
            callable(getattr(device_manager, "enter_phase", None))
            and component_name in ("transformer", "transformer_2")
            and bool(getattr(device_manager, "should_use_premerged", False))
        ):
            strategy = LifecycleAdapterStrategy(device_manager, component_name)
        else:
            strategy = build_component_residency_strategy(
                component_name, module, self.server_args
            )
        self._strategy_cache[component_name] = strategy
        return strategy

    def _stage_uses(self, stage_index: int) -> tuple[ComponentUse, ...]:
        if stage_index < 0 or stage_index >= len(self._stage_uses_by_index):
            return ()
        return self._stage_uses_by_index[stage_index]

    def _next_stage_name(self, stage_index: int) -> str | None:
        next_index = stage_index + 1
        if next_index < 0 or next_index >= len(self.state.stages):
            return None
        return self.stage_name(self.state.stages[next_index])

    def _future_uses(self, start_index: int) -> tuple[ComponentUse, ...]:
        uses: list[ComponentUse] = []
        for index in range(start_index, len(self._stage_uses_by_index)):
            uses.extend(self._stage_uses_by_index[index])
        return tuple(uses)

    def _should_keep_after_use(self, use: ComponentUse) -> bool:
        future_component_names = {
            future.component_name for future in self.state.future_uses
        }
        if use.component_name in future_component_names:
            return True
        return self._should_keep_for_dynamic_budget(use.component_name)

    def _should_keep_for_dynamic_budget(self, component_name: str) -> bool:
        if (
            self.state.manager_mode != "dynamic"
            or not self.state.dynamic_budget
            or not current_platform.is_cuda()
        ):
            return False
        memory_usage = getattr(self.pipeline, "memory_usages", {}).get(component_name)
        if memory_usage is None:
            return False
        return memory_usage + 2.0 < current_platform.get_available_gpu_memory()

    def _preferred_request_end_use(self) -> ComponentUse | None:
        for uses in self._stage_uses_by_index:
            for use in uses:
                if use.preferred_after_request:
                    return use
        for uses in self._stage_uses_by_index:
            if uses:
                return uses[0]
        return None

    def _trace(
        self,
        action: str,
        use: ComponentUse | None = None,
        strategy: ComponentResidencyStrategy | None = None,
        module: nn.Module | None = None,
        *,
        component_name: str | None = None,
        detail: str = "",
    ) -> None:
        if not self.state.trace_enabled:
            return
        if use is not None:
            component_name = use.component_name
        device = self._module_device(module)
        logger.info(
            "[component_residency] action=%s stage=%s next_stage=%s component=%s "
            "strategy=%s phase=%s access=%s device=%s warmup=%s mode=%s %s",
            action,
            self.state.stage_name,
            self.state.next_stage_name,
            component_name,
            strategy.name if strategy is not None else None,
            use.phase if use is not None else None,
            use.access_kind if use is not None else None,
            device,
            self.state.batch_is_warmup,
            self.state.manager_mode,
            detail,
        )

    def _module_device(self, module: nn.Module | None) -> str | None:
        if module is None:
            return None
        param = next(module.parameters(), None)
        if param is not None:
            return param.device.type
        buffer = next(module.buffers(), None)
        return buffer.device.type if buffer is not None else None
