"""Resolve diffusion artifact sources without constructing model components."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id
from safetensors import safe_open

from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

ArtifactSourceKind = Literal["local", "huggingface"]
ArtifactRole = Literal["pipeline", "component", "component_weights", "lora"]

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")
_DECLARATIVE_SIDECARS = frozenset(
    ("adapter_config.json", "config.json", "sglang_artifacts.json")
)


@dataclass(frozen=True)
class ArtifactSource:
    original: str
    kind: ArtifactSourceKind
    local_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    filename: str | None = None
    expected_sha256: str | None = None


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size: int | None = None
    blob_id: str | None = None


@dataclass(frozen=True)
class ArtifactInventory:
    source: ArtifactSource
    resolved_revision: str | None
    files: tuple[ArtifactFile, ...]


@dataclass(frozen=True)
class ArtifactRequest:
    name: str
    role: ArtifactRole
    source: str
    component: str | None = None
    revision: str | None = None
    weight_name: str | None = None


@dataclass(frozen=True)
class TensorSummary:
    tensor_count: int
    dtypes: tuple[str, ...]
    key_samples: tuple[str, ...]
    metadata: dict[str, str]
    lora_ranks: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedArtifact:
    request: ArtifactRequest
    inventory: ArtifactInventory
    selected_files: tuple[str, ...]
    container_format: str | None
    quantization_method: str | None
    quantization_source: str | None
    tensor_summary: TensorSummary | None
    request_defaults: dict[str, int | float] = field(default_factory=dict)
    request_default_sources: dict[str, str] = field(default_factory=dict)
    lora_alpha: int | None = None
    lora_alpha_source: str | None = None
    lora_adapter_type: str | None = None


_REQUEST_METADATA_KEYS = {
    "num_inference_steps": ("num_inference_steps", "inference_steps", "sampler_steps"),
    "guidance_scale": ("guidance_scale", "cfg_scale"),
    "flow_shift": ("flow_shift",),
}
_LORA_ALPHA_METADATA_KEYS = (
    "lora_alpha",
    "network_alpha",
    "ss_network_alpha",
    "alpha",
)
_TENSOR_QUANTIZATION_METADATA_KEYS = (
    "quantization_config",
    "_quantization_metadata",
)


def _metadata_number(value: str) -> int | float | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(parsed, bool)
        or not isinstance(parsed, (int, float))
        or not math.isfinite(parsed)
    ):
        return None
    return parsed


def infer_request_defaults_from_tensor_metadata(
    metadata: dict[str, str],
) -> tuple[dict[str, int | float], dict[str, str]]:
    """Read only explicit, numeric sampling metadata; never infer from filenames."""
    defaults: dict[str, int | float] = {}
    sources: dict[str, str] = {}
    for field_name, metadata_keys in _REQUEST_METADATA_KEYS.items():
        candidates = {
            key: parsed
            for key in metadata_keys
            if key in metadata
            and (parsed := _metadata_number(metadata[key])) is not None
        }
        values = set(candidates.values())
        if len(values) > 1:
            raise ValueError(
                f"Conflicting tensor metadata for {field_name}: {candidates}"
            )
        if values:
            defaults[field_name] = values.pop()
            sources[field_name] = f"safetensors:{next(iter(candidates))}"
    return defaults, sources


def infer_lora_alpha_from_tensor_metadata(metadata: dict[str, str]) -> int | None:
    candidates = {
        key: parsed
        for key in _LORA_ALPHA_METADATA_KEYS
        if key in metadata and (parsed := _metadata_number(metadata[key])) is not None
    }
    values = set(candidates.values())
    if len(values) > 1:
        raise ValueError(f"Conflicting LoRA alpha tensor metadata: {candidates}")
    if not values:
        return None
    value = values.pop()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"LoRA alpha tensor metadata must be a positive integer: {value}"
        )
    return value


def _validate_relative_hub_path(path: str, field_name: str) -> str:
    normalized = str(PurePosixPath(path))
    pure_path = PurePosixPath(normalized)
    if not path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Invalid Hugging Face {field_name}: {path!r}")
    return normalized


def _merge_revision(url_revision: str | None, revision: str | None) -> str | None:
    if url_revision is not None and revision is not None and url_revision != revision:
        raise ValueError(
            f"Artifact URL pins revision {url_revision!r}, which conflicts with "
            f"--revision {revision!r}"
        )
    return url_revision or revision


def _split_checksum(source: str) -> tuple[str, str | None]:
    parsed = urlparse(source)
    if not parsed.fragment:
        return source, None
    prefix = "sha256="
    if not parsed.fragment.startswith(prefix):
        raise ValueError(f"Unsupported artifact source fragment: {parsed.fragment!r}")
    checksum = parsed.fragment.removeprefix(prefix).lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("Artifact sha256 checksum must contain exactly 64 hex digits")
    return source.rsplit("#", 1)[0], checksum


def _parse_huggingface_url(
    source: str, revision: str | None, expected_sha256: str | None
) -> ArtifactSource:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(
            "Only huggingface.co artifact URLs are supported; use a local path "
            "or an owner/repo reference for other sources"
        )

    raw_parts = [part for part in parsed.path.split("/") if part]
    if raw_parts and raw_parts[0] in ("datasets", "spaces"):
        raise ValueError("Diffusion artifacts must come from a Hugging Face model repo")
    if len(raw_parts) < 2:
        raise ValueError(f"Hugging Face artifact URL has no model repo: {source!r}")

    repo_id = "/".join(unquote(part) for part in raw_parts[:2])
    validate_repo_id(repo_id)
    action = raw_parts[2] if len(raw_parts) > 2 else None
    if action is None:
        return ArtifactSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=revision,
            expected_sha256=expected_sha256,
        )
    if action not in ("tree", "blob", "resolve") or len(raw_parts) < 4:
        raise ValueError(f"Unsupported Hugging Face artifact URL: {source!r}")

    url_revision = unquote(raw_parts[3])
    selected_revision = _merge_revision(url_revision, revision)
    tail = "/".join(unquote(part) for part in raw_parts[4:])
    if action == "tree":
        subfolder = _validate_relative_hub_path(tail, "subfolder") if tail else None
        return ArtifactSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=selected_revision,
            subfolder=subfolder,
            expected_sha256=expected_sha256,
        )
    if not tail:
        raise ValueError(f"Hugging Face file URL has no filename: {source!r}")
    return ArtifactSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=selected_revision,
        filename=_validate_relative_hub_path(tail, "filename"),
        expected_sha256=expected_sha256,
    )


def parse_artifact_source(
    source: str,
    *,
    revision: str | None = None,
) -> ArtifactSource:
    """Parse local paths, Hub repo IDs, subfolders, and exact Hub URLs."""
    base_source, expected_sha256 = _split_checksum(source)
    expanded = os.path.expanduser(base_source)
    parsed = urlparse(base_source)
    if parsed.scheme in ("http", "https"):
        return _parse_huggingface_url(base_source, revision, expected_sha256)

    looks_local = (
        os.path.exists(expanded)
        or os.path.isabs(expanded)
        or base_source.startswith(("./", "../", "~"))
    )
    if looks_local:
        return ArtifactSource(
            original=source,
            kind="local",
            local_path=os.path.abspath(expanded),
            expected_sha256=expected_sha256,
        )

    parts = base_source.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(
            f"Artifact source {source!r} is neither a local path nor an "
            "owner/repo Hugging Face reference"
        )
    repo_id = "/".join(parts[:2])
    validate_repo_id(repo_id)
    tail = "/".join(parts[2:]) or None
    filename = (
        _validate_relative_hub_path(tail, "filename")
        if tail is not None and tail.lower().endswith(_WEIGHT_SUFFIXES)
        else None
    )
    subfolder = tail if filename is None else None
    if subfolder is not None:
        subfolder = _validate_relative_hub_path(subfolder, "subfolder")
    return ArtifactSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        filename=filename,
        expected_sha256=expected_sha256,
    )


def _verify_artifact_checksum(path: str, expected_sha256: str | None) -> str:
    if expected_sha256 is None:
        return path
    if not os.path.isfile(path):
        raise ValueError("Artifact sha256 checksums require one exact file")
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Artifact sha256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return path


def _filter_inventory_files(
    files: tuple[ArtifactFile, ...], source: ArtifactSource
) -> tuple[ArtifactFile, ...]:
    if source.filename is not None:
        exact = tuple(item for item in files if item.path == source.filename)
        if not exact:
            raise FileNotFoundError(
                f"Artifact file {source.filename!r} was not found in {source.repo_id}"
            )
        parent = PurePosixPath(source.filename).parent
        sidecars = tuple(
            item
            for item in files
            if item.path != source.filename
            and PurePosixPath(item.path).parent == parent
            and PurePosixPath(item.path).name in _DECLARATIVE_SIDECARS
        )
        return exact + sidecars
    if source.subfolder is None:
        return files
    prefix = source.subfolder.rstrip("/") + "/"
    selected = tuple(item for item in files if item.path.startswith(prefix))
    if not selected:
        raise FileNotFoundError(
            f"Artifact subfolder {source.subfolder!r} was not found in {source.repo_id}"
        )
    return selected


def resolve_artifact_inventory(source: ArtifactSource) -> ArtifactInventory:
    """List artifact files and pin a remote source to an immutable revision."""
    if source.kind == "local":
        assert source.local_path is not None
        local_path = Path(source.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {local_path}")
        if local_path.is_file():
            files = (
                ArtifactFile(path=local_path.name, size=local_path.stat().st_size),
            )
        else:
            files = tuple(
                ArtifactFile(
                    path=path.relative_to(local_path).as_posix(),
                    size=path.stat().st_size,
                )
                for path in sorted(local_path.rglob("*"))
                if path.is_file()
            )
        return ArtifactInventory(
            source=source,
            resolved_revision=None,
            files=files,
        )

    assert source.repo_id is not None
    model_info = HfApi().model_info(
        source.repo_id,
        revision=source.revision,
        files_metadata=True,
    )
    files = tuple(
        ArtifactFile(
            path=sibling.rfilename,
            size=sibling.size,
            blob_id=sibling.blob_id,
        )
        for sibling in model_info.siblings
    )
    return ArtifactInventory(
        source=source,
        resolved_revision=model_info.sha,
        files=_filter_inventory_files(files, source),
    )


def materialize_artifact(inventory: ArtifactInventory) -> str:
    """Download the pinned artifact selected by an inventory."""
    source = inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        return _verify_artifact_checksum(source.local_path, source.expected_sha256)

    assert source.repo_id is not None
    revision = inventory.resolved_revision or source.revision
    if source.filename is not None:
        local_path = hf_hub_download(
            repo_id=source.repo_id,
            filename=source.filename,
            revision=revision,
        )
        return _verify_artifact_checksum(local_path, source.expected_sha256)
    if source.expected_sha256 is not None:
        raise ValueError("Artifact sha256 checksums require one exact file")
    allow_patterns = None
    if source.subfolder is not None:
        allow_patterns = [f"{source.subfolder}/**", f"{source.subfolder}/*"]
    local_repo = snapshot_download(
        repo_id=source.repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    if source.subfolder is None:
        return local_repo
    return os.path.join(local_repo, source.subfolder)


def _local_inventory_file_path(
    inventory: ArtifactInventory, inventory_path: str
) -> str:
    source = inventory.source
    assert source.local_path is not None
    if os.path.isfile(source.local_path):
        return source.local_path
    return os.path.join(source.local_path, inventory_path)


def materialize_artifact_file(inventory: ArtifactInventory, inventory_path: str) -> str:
    source = inventory.source
    expected_sha256 = source.expected_sha256
    if source.filename is not None and inventory_path != source.filename:
        expected_sha256 = None
    if source.kind == "local":
        return _verify_artifact_checksum(
            _local_inventory_file_path(inventory, inventory_path),
            expected_sha256,
        )
    assert source.repo_id is not None
    local_path = hf_hub_download(
        repo_id=source.repo_id,
        filename=inventory_path,
        revision=inventory.resolved_revision or source.revision,
    )
    return _verify_artifact_checksum(local_path, expected_sha256)


def _read_inventory_json(inventory: ArtifactInventory, inventory_path: str) -> dict:
    local_path = materialize_artifact_file(inventory, inventory_path)
    with open(local_path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Artifact metadata must be an object: {inventory_path}")
    return value


def _select_named_file(
    candidates: tuple[str, ...], weight_name: str
) -> tuple[str, ...]:
    exact = tuple(path for path in candidates if path == weight_name)
    if exact:
        return exact
    basename_matches = tuple(
        path for path in candidates if PurePosixPath(path).name == weight_name
    )
    if len(basename_matches) == 1:
        return basename_matches
    if not basename_matches:
        raise FileNotFoundError(
            f"Requested artifact weight {weight_name!r} was not found"
        )
    raise ValueError(
        f"Artifact weight name {weight_name!r} matches multiple files: "
        f"{list(basename_matches)}"
    )


def select_artifact_weight_files(
    request: ArtifactRequest, inventory: ArtifactInventory
) -> tuple[str, ...]:
    """Select weights deterministically; never guess among independent files."""
    candidates = tuple(
        item.path
        for item in inventory.files
        if item.path.lower().endswith(_WEIGHT_SUFFIXES)
    )
    if inventory.source.filename is not None:
        return tuple(
            item.path
            for item in inventory.files
            if item.path == inventory.source.filename
        )
    if request.weight_name is not None:
        return _select_named_file(candidates, request.weight_name)
    if request.role in ("pipeline", "component"):
        return candidates

    index_files = tuple(
        item.path
        for item in inventory.files
        if item.path.endswith(".safetensors.index.json")
    )
    if len(index_files) == 1:
        weight_map = _read_inventory_json(inventory, index_files[0]).get(
            "weight_map", {}
        )
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Artifact index has no weight_map: {index_files[0]}")
        index_dir = str(PurePosixPath(index_files[0]).parent)
        prefix = "" if index_dir == "." else f"{index_dir}/"
        selected = tuple(
            sorted({f"{prefix}{str(filename)}" for filename in weight_map.values()})
        )
        missing = [path for path in selected if path not in candidates]
        if missing:
            raise FileNotFoundError(
                f"Artifact index {index_files[0]!r} references missing files: {missing}"
            )
        return selected
    if len(index_files) > 1:
        raise ValueError(
            f"Artifact contains multiple weight indexes: {list(index_files)}"
        )
    if len(candidates) == 1:
        return candidates
    if not candidates:
        raise FileNotFoundError("Artifact contains no recognized weight files")
    raise ValueError(
        "Artifact contains multiple independent weight files; select one with "
        f"an exact file URL or weight name. Candidates: {list(candidates)}"
    )


def _lora_rank(key: str, shape: list[int]) -> int | None:
    if len(shape) < 2:
        return None
    if ".lora_A" in key or "lora_down" in key:
        return int(shape[-2])
    if ".lora_B" in key or "lora_up" in key:
        return int(shape[-1])
    return None


def _summarize_local_safetensors(file_path: str) -> TensorSummary:
    with safe_open(file_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        metadata = weights.metadata() or {}
        dtypes: set[str] = set()
        lora_ranks: set[int] = set()
        for key in keys:
            tensor_slice = weights.get_slice(key)
            dtypes.add(str(tensor_slice.get_dtype()))
            rank = _lora_rank(key, tensor_slice.get_shape())
            if rank is not None:
                lora_ranks.add(rank)
    return TensorSummary(
        tensor_count=len(keys),
        dtypes=tuple(sorted(dtypes)),
        key_samples=tuple(keys[:20]),
        metadata=dict(metadata),
        lora_ranks=tuple(sorted(lora_ranks)),
    )


def _summarize_remote_safetensors(
    inventory: ArtifactInventory, inventory_path: str
) -> TensorSummary:
    source = inventory.source
    assert source.repo_id is not None
    metadata = HfApi().parse_safetensors_file_metadata(
        source.repo_id,
        inventory_path,
        revision=inventory.resolved_revision or source.revision,
    )
    keys = list(metadata.tensors)
    lora_ranks = {
        rank
        for key, tensor in metadata.tensors.items()
        if (rank := _lora_rank(key, tensor.shape)) is not None
    }
    return TensorSummary(
        tensor_count=len(keys),
        dtypes=tuple(sorted(metadata.parameter_count)),
        key_samples=tuple(keys[:20]),
        metadata=dict(metadata.metadata or {}),
        lora_ranks=tuple(sorted(lora_ranks)),
    )


def _merge_tensor_summaries(summaries: list[TensorSummary]) -> TensorSummary | None:
    if not summaries:
        return None
    metadata: dict[str, str] = {}
    for summary in summaries:
        metadata.update(summary.metadata)
    return TensorSummary(
        tensor_count=sum(summary.tensor_count for summary in summaries),
        dtypes=tuple(
            sorted({dtype for summary in summaries for dtype in summary.dtypes})
        ),
        key_samples=tuple(key for summary in summaries for key in summary.key_samples)[
            :20
        ],
        metadata=metadata,
        lora_ranks=tuple(
            sorted({rank for summary in summaries for rank in summary.lora_ranks})
        ),
    )


def _resolve_quantization_metadata(
    inventory: ArtifactInventory,
    tensor_summaries: list[TensorSummary],
) -> tuple[str | None, str | None]:
    declarations: set[tuple[str, str]] = set()
    config_files = tuple(
        item.path
        for item in inventory.files
        if PurePosixPath(item.path).name == "config.json"
    )
    if len(config_files) == 1:
        quant_spec = resolve_checkpoint_quant_spec(
            _read_inventory_json(inventory, config_files[0])
        )
        if quant_spec is not None and quant_spec.declared_method is not None:
            declarations.add((quant_spec.declared_method, quant_spec.source))

    for summary in tensor_summaries:
        for metadata_key in _TENSOR_QUANTIZATION_METADATA_KEYS:
            serialized = summary.metadata.get(metadata_key)
            if serialized is None:
                continue
            try:
                quantization_config = json.loads(serialized)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid safetensors {metadata_key}: expected a JSON object"
                ) from error
            if not isinstance(quantization_config, dict):
                raise ValueError(
                    f"Invalid safetensors {metadata_key}: expected a JSON object"
                )
            quant_spec = resolve_checkpoint_quant_spec(
                {"quantization_config": quantization_config}
            )
            if quant_spec is not None and quant_spec.declared_method is not None:
                declarations.add(
                    (quant_spec.declared_method, f"safetensors:{metadata_key}")
                )

    methods = {method for method, _ in declarations}
    if len(methods) > 1:
        raise ValueError(
            f"Conflicting artifact quantization metadata: {sorted(declarations)}"
        )
    if not declarations:
        return None, None
    method = methods.pop()
    sources = ",".join(sorted(source for _, source in declarations))
    return method, sources


def _resolve_lora_adapter_config(
    inventory: ArtifactInventory, selected_files: tuple[str, ...]
) -> tuple[int | None, str | None]:
    if len(selected_files) != 1:
        return None, None
    selected_parent = PurePosixPath(selected_files[0]).parent
    config_path = next(
        (
            item.path
            for item in inventory.files
            if PurePosixPath(item.path).parent == selected_parent
            and PurePosixPath(item.path).name == "adapter_config.json"
        ),
        None,
    )
    if config_path is None:
        return None, None
    config = _read_inventory_json(inventory, config_path)
    use_rslora = config.get("use_rslora", False)
    use_dora = config.get("use_dora", False)
    if not isinstance(use_rslora, bool) or not isinstance(use_dora, bool):
        raise ValueError("PEFT use_rslora and use_dora must be boolean")
    adapter_type = "dora" if use_dora else "rslora" if use_rslora else "lora"
    alpha = config.get("lora_alpha")
    if alpha is None:
        return None, adapter_type
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("PEFT lora_alpha must be a positive integer")
    if not float(alpha).is_integer() or alpha <= 0:
        raise ValueError("PEFT lora_alpha must be a positive integer")
    return int(alpha), adapter_type


def resolve_artifact(request: ArtifactRequest) -> ResolvedArtifact:
    """Resolve one artifact request using the same metadata path as preflight."""
    source = parse_artifact_source(request.source, revision=request.revision)
    inventory = resolve_artifact_inventory(source)
    selected_files = select_artifact_weight_files(request, inventory)
    if source.expected_sha256 is not None and len(selected_files) != 1:
        raise ValueError("Artifact sha256 checksums require one exact weight file")
    summaries = []
    summary_files = () if request.role == "pipeline" else selected_files
    for selected_file in summary_files:
        if not selected_file.endswith(".safetensors"):
            continue
        if source.kind == "local":
            summaries.append(
                _summarize_local_safetensors(
                    _local_inventory_file_path(inventory, selected_file)
                )
            )
        else:
            summaries.append(_summarize_remote_safetensors(inventory, selected_file))
    request_defaults: dict[str, int | float] = {}
    request_default_sources: dict[str, str] = {}
    lora_alphas: set[int] = set()
    for summary in summaries:
        summary_defaults, summary_sources = infer_request_defaults_from_tensor_metadata(
            summary.metadata
        )
        for field_name, value in summary_defaults.items():
            existing = request_defaults.get(field_name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Conflicting artifact metadata for {field_name}: "
                    f"{existing} vs {value}"
                )
            request_defaults[field_name] = value
            request_default_sources[field_name] = summary_sources[field_name]
        lora_alpha = infer_lora_alpha_from_tensor_metadata(summary.metadata)
        if lora_alpha is not None:
            lora_alphas.add(lora_alpha)
    if len(lora_alphas) > 1:
        raise ValueError(f"Conflicting LoRA alpha across artifact files: {lora_alphas}")
    tensor_lora_alpha = next(iter(lora_alphas), None)
    config_lora_alpha = None
    lora_adapter_type = None
    if request.role == "lora":
        config_lora_alpha, lora_adapter_type = _resolve_lora_adapter_config(
            inventory, selected_files
        )
    lora_alpha = (
        config_lora_alpha if config_lora_alpha is not None else tensor_lora_alpha
    )
    lora_alpha_source = (
        "adapter_config.json"
        if config_lora_alpha is not None
        else "safetensors metadata" if tensor_lora_alpha is not None else None
    )
    quantization_method, quantization_source = _resolve_quantization_metadata(
        inventory, summaries
    )
    formats = {PurePosixPath(path).suffix.removeprefix(".") for path in selected_files}
    if len(formats) == 1:
        container_format = next(iter(formats))
    elif formats:
        container_format = "mixed"
    else:
        container_format = None
    return ResolvedArtifact(
        request=request,
        inventory=inventory,
        selected_files=selected_files,
        container_format=container_format,
        quantization_method=quantization_method,
        quantization_source=quantization_source,
        tensor_summary=_merge_tensor_summaries(summaries),
        request_defaults=request_defaults,
        request_default_sources=request_default_sources,
        lora_alpha=lora_alpha,
        lora_alpha_source=lora_alpha_source,
        lora_adapter_type=lora_adapter_type,
    )


def materialize_resolved_artifact(artifact: ResolvedArtifact) -> tuple[str, ...]:
    """Download exactly the files selected by a resolved weights artifact."""
    if artifact.request.role in ("pipeline", "component"):
        return (materialize_artifact(artifact.inventory),)
    return tuple(
        materialize_artifact_file(artifact.inventory, selected_file)
        for selected_file in artifact.selected_files
    )
