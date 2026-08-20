"""Strict declarative defaults for packaged diffusion artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from sglang.multimodal_gen.runtime.loader.artifact_resolver import (
    ArtifactInventory,
    ArtifactSource,
    materialize_artifact_file,
    parse_artifact_source,
    resolve_artifact_inventory,
)

MANIFEST_FILENAME = "sglang_artifacts.json"
_ENTRY_ROLES = frozenset(("component", "component_weights", "lora"))
_ROOT_KEYS = frozenset(
    (
        "schema_version",
        "base_model",
        "model_variant",
        "entries",
        "request_defaults",
    )
)
_ENTRY_KEYS = frozenset(
    (
        "id",
        "path",
        "role",
        "component",
        "default",
        "checksum",
        "adapter",
        "request_defaults",
    )
)
_ADAPTER_KEYS = frozenset(("alpha", "scale"))
_SHA256_PATTERN = re.compile(r"sha256:([0-9a-fA-F]{64})")


@dataclass(frozen=True)
class ArtifactManifestDefaults:
    model_path: str | None
    model_variant: str | None
    component_paths: dict[str, str]
    component_weights_paths: dict[str, str]
    lora_path: str | None
    lora_alpha: int | None
    lora_scale: float | None
    request_defaults: dict[str, object]


def _reject_unknown_keys(value: dict, allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} fields: {unknown}")


def _manifest_inventory(source: ArtifactSource) -> tuple[ArtifactInventory, str]:
    if source.kind == "local":
        assert source.local_path is not None
        local_path = Path(source.local_path)
        manifest_path = (
            local_path if local_path.is_file() else local_path / MANIFEST_FILENAME
        )
        if manifest_path.name != MANIFEST_FILENAME or not manifest_path.is_file():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        inventory = resolve_artifact_inventory(
            ArtifactSource(
                original=str(manifest_path.parent),
                kind="local",
                local_path=str(manifest_path.parent),
            )
        )
        return inventory, MANIFEST_FILENAME

    inventory = resolve_artifact_inventory(source)
    if source.filename is not None:
        if PurePosixPath(source.filename).name != MANIFEST_FILENAME:
            raise ValueError(f"Artifact manifest must be named {MANIFEST_FILENAME!r}")
        return inventory, source.filename
    manifest_path = (
        f"{source.subfolder.rstrip('/')}/{MANIFEST_FILENAME}"
        if source.subfolder is not None
        else MANIFEST_FILENAME
    )
    if manifest_path not in {item.path for item in inventory.files}:
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
    return inventory, manifest_path


def _entry_inventory(
    manifest_inventory: ArtifactInventory, manifest_path: str
) -> ArtifactInventory:
    source = manifest_inventory.source
    if source.kind == "local" or source.filename is None:
        return manifest_inventory
    assert source.repo_id is not None
    manifest_dir = PurePosixPath(manifest_path).parent
    return resolve_artifact_inventory(
        ArtifactSource(
            original=source.repo_id,
            kind="huggingface",
            repo_id=source.repo_id,
            revision=manifest_inventory.resolved_revision or source.revision,
            subfolder=None if str(manifest_dir) == "." else str(manifest_dir),
        )
    )


def _relative_entry_source(
    inventory: ArtifactInventory,
    manifest_path: str,
    relative_path: str,
    checksum: str | None,
) -> str:
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or not relative_path:
        raise ValueError(f"Artifact manifest paths must be relative: {relative_path!r}")
    manifest_dir = PurePosixPath(manifest_path).parent
    target = pure_path if str(manifest_dir) == "." else manifest_dir / pure_path
    target_path = str(target)

    source = inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        local_target = os.path.join(source.local_path, relative_path)
        if not os.path.exists(local_target):
            raise FileNotFoundError(f"Manifest artifact does not exist: {local_target}")
        is_file = os.path.isfile(local_target)
        resolved_source = local_target
    else:
        paths = {item.path for item in inventory.files}
        is_file = target_path in paths
        is_directory = any(path.startswith(f"{target_path}/") for path in paths)
        if not is_file and not is_directory:
            raise FileNotFoundError(f"Manifest artifact does not exist: {target_path}")
        assert source.repo_id is not None
        revision = inventory.resolved_revision or source.revision or "main"
        action = "resolve" if is_file else "tree"
        resolved_source = (
            f"https://huggingface.co/{source.repo_id}/{action}/"
            f"{quote(revision, safe='')}/{quote(target_path, safe='/')}"
        )

    if checksum is None:
        return resolved_source
    match = _SHA256_PATTERN.fullmatch(checksum)
    if match is None:
        raise ValueError("Artifact checksum must use sha256:<64 hex digits>")
    if not is_file:
        raise ValueError("Artifact checksums require an exact file entry")
    return f"{resolved_source}#sha256={match.group(1).lower()}"


def _read_manifest(
    source: str, revision: str | None
) -> tuple[dict, ArtifactInventory, str]:
    manifest_source = parse_artifact_source(source, revision=revision)
    inventory, manifest_path = _manifest_inventory(manifest_source)
    local_path = materialize_artifact_file(inventory, manifest_path)
    with open(local_path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("Artifact manifest must be a JSON object")
    return value, _entry_inventory(inventory, manifest_path), manifest_path


def load_artifact_manifest_defaults(
    source: str,
    *,
    revision: str | None = None,
    selected_entries: list[str] | tuple[str, ...] | None = None,
) -> ArtifactManifestDefaults:
    """Load versioned, non-executable artifact defaults from a local or Hub manifest."""
    value, inventory, manifest_path = _read_manifest(source, revision)
    _reject_unknown_keys(value, _ROOT_KEYS, "artifact manifest")
    if value.get("schema_version") != 1:
        raise ValueError("Artifact manifest schema_version must be 1")

    entries = value.get("entries", [])
    root_request_defaults = value.get("request_defaults", {})
    if not isinstance(entries, list) or not isinstance(root_request_defaults, dict):
        raise ValueError(
            "Artifact manifest entries must be a list and request_defaults an object"
        )
    request_defaults = dict(root_request_defaults)
    request_default_priorities = {name: -1 for name in request_defaults}

    component_paths: dict[str, str] = {}
    component_weights_paths: dict[str, str] = {}
    lora_path = None
    lora_alpha = None
    lora_scale = None
    entry_ids: set[str] = set()
    selected_ids = set(selected_entries or ())
    matched_selected_ids: set[str] = set()
    component_priorities: dict[tuple[str, str], int] = {}
    lora_priority = -1
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Artifact manifest entry {index} must be an object")
        _reject_unknown_keys(entry, _ENTRY_KEYS, f"artifact entry {index}")
        entry_id = entry.get("id")
        role = entry.get("role")
        path = entry.get("path")
        if not isinstance(entry_id, str) or not entry_id or entry_id in entry_ids:
            raise ValueError(
                f"Artifact entry id must be a unique non-empty string: {entry_id!r}"
            )
        entry_ids.add(entry_id)
        if role not in _ENTRY_ROLES or not isinstance(path, str):
            raise ValueError(f"Artifact entry {entry_id!r} has an invalid role or path")
        is_default = entry.get("default", False)
        if not isinstance(is_default, bool):
            raise ValueError(f"Artifact entry {entry_id!r} default must be boolean")
        is_selected = entry_id in selected_ids
        if is_selected:
            matched_selected_ids.add(entry_id)
        is_active = is_default or is_selected
        priority = 1 if is_selected else 0
        entry_request_defaults = entry.get("request_defaults", {})
        if not isinstance(entry_request_defaults, dict):
            raise ValueError(
                f"Artifact entry {entry_id!r} request_defaults must be an object"
            )
        if is_active:
            for name, request_value in entry_request_defaults.items():
                existing_priority = request_default_priorities.get(name)
                if (
                    existing_priority == priority
                    and request_defaults[name] != request_value
                ):
                    raise ValueError(
                        f"Active artifact entries disagree on request default "
                        f"{name!r}"
                    )
                if existing_priority is None or priority > existing_priority:
                    request_defaults[name] = request_value
                    request_default_priorities[name] = priority
        checksum = entry.get("checksum")
        if checksum is not None and (
            not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None
        ):
            raise ValueError(
                f"Artifact entry {entry_id!r} checksum must use sha256:<64 hex digits>"
            )
        component = entry.get("component")
        if role in ("component", "component_weights"):
            if not isinstance(component, str) or not component:
                raise ValueError(f"Artifact entry {entry_id!r} requires component")
            if "adapter" in entry:
                raise ValueError(
                    f"Artifact entry {entry_id!r} adapter is only valid for LoRA"
                )
            resolved_source = _relative_entry_source(
                inventory, manifest_path, path, checksum
            )
            if not is_active:
                continue
            target = component_paths if role == "component" else component_weights_paths
            target_key = (role, component)
            existing_priority = component_priorities.get(target_key)
            if existing_priority == priority:
                raise ValueError(
                    f"Multiple active {role} artifacts target component {component!r}"
                )
            if existing_priority is None or priority > existing_priority:
                target[component] = resolved_source
                component_priorities[target_key] = priority
        else:
            if component is not None:
                raise ValueError(
                    f"Artifact entry {entry_id!r} LoRA cannot name a component"
                )
            adapter = entry.get("adapter", {})
            if not isinstance(adapter, dict):
                raise ValueError(
                    f"Artifact entry {entry_id!r} adapter must be an object"
                )
            _reject_unknown_keys(adapter, _ADAPTER_KEYS, f"adapter {entry_id!r}")
            alpha = adapter.get("alpha")
            scale = adapter.get("scale")
            if alpha is not None and (not isinstance(alpha, int) or alpha <= 0):
                raise ValueError(f"Artifact entry {entry_id!r} alpha must be positive")
            if scale is not None and (
                not isinstance(scale, (int, float)) or scale <= 0
            ):
                raise ValueError(f"Artifact entry {entry_id!r} scale must be positive")
            resolved_source = _relative_entry_source(
                inventory, manifest_path, path, checksum
            )
            if not is_active:
                continue
            if lora_path is not None and priority == lora_priority:
                raise ValueError("Only one active LoRA manifest entry is supported")
            if priority > lora_priority:
                lora_path, lora_alpha = resolved_source, alpha
                lora_scale = float(scale) if scale is not None else None
                lora_priority = priority

    missing_entries = sorted(selected_ids - matched_selected_ids)
    if missing_entries:
        raise ValueError(f"Unknown artifact manifest entries: {missing_entries}")

    base_model = value.get("base_model")
    model_variant = value.get("model_variant")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("Artifact manifest base_model must be a string")
    if model_variant is not None and not isinstance(model_variant, str):
        raise ValueError("Artifact manifest model_variant must be a string")
    return ArtifactManifestDefaults(
        model_path=base_model,
        model_variant=model_variant,
        component_paths=component_paths,
        component_weights_paths=component_weights_paths,
        lora_path=lora_path,
        lora_alpha=lora_alpha,
        lora_scale=lora_scale,
        request_defaults=dict(request_defaults),
    )
