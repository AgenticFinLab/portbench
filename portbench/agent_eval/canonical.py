"""Canonical serialization and stage cache-key construction."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from enum import Enum
from typing import Any, Mapping


def _stable_float(value: float) -> str:
    """Serialize a binary64 value independently of decimal formatting."""
    # Preserve non-finite values explicitly instead of relying on non-standard JSON tokens.
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    # Encode raw IEEE-754 bytes so decimal printer differences cannot change a cache key.
    return "f64:" + struct.pack("<d", float(value)).hex()


def _normalize_dataframe(obj: Any) -> Any:
    """Serialize pandas-like tables without display truncation."""
    # Labels are part of the scientific input and must affect the digest.
    columns = [_normalize(value) for value in obj.columns.tolist()]
    index = [_normalize(value) for value in obj.index.tolist()]
    data = [[_normalize(value) for value in row] for row in obj.to_numpy().tolist()]
    return {"__type__": "dataframe", "columns": columns, "index": index, "data": data}


def _normalize_series(obj: Any) -> Any:
    """Serialize pandas-like vectors with their complete index."""
    return {
        "__type__": "series",
        "name": _normalize(obj.name),
        "index": [_normalize(value) for value in obj.index.tolist()],
        "data": [_normalize(value) for value in obj.tolist()],
    }


def _normalize(obj: Any) -> Any:
    """Recursively normalize supported objects for canonical JSON."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return _stable_float(obj)
    if isinstance(obj, Enum):
        return _normalize(obj.value)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _normalize(dataclasses.asdict(obj))
    if obj.__class__.__module__.startswith("numpy") and hasattr(obj, "item"):
        return _normalize(obj.item())
    if obj.__class__.__module__.startswith("pandas"):
        if hasattr(obj, "columns") and hasattr(obj, "to_numpy"):
            return _normalize_dataframe(obj)
        if hasattr(obj, "index") and hasattr(obj, "tolist"):
            return _normalize_series(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
    if isinstance(obj, Mapping):
        # String-only keys avoid ambiguous coercions such as 1 and "1".
        if any(not isinstance(key, str) for key in obj):
            raise TypeError("canonical mappings require string keys")
        return {key: _normalize(obj[key]) for key in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(value) for value in obj]
    if isinstance(obj, set):
        # Sort normalized set members by their canonical JSON representation.
        normalized = [_normalize(value) for value in obj]
        return sorted(normalized, key=lambda value: json.dumps(value, sort_keys=True))
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"unsupported canonical type: {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    """Return compact canonical JSON for supported scientific objects."""
    normalized = _normalize(obj)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Keep a closed field list so new runtime state cannot be omitted silently.
CACHE_KEY_FIELDS = (
    "model",
    "model_revision",
    "provider",
    "stage_id",
    "stage_schema_version",
    "system_prompt_hash",
    "user_prompt_hash",
    "structured_stage_input_hash",
    "generation_config",
    "toolset_hash",
    "tool_result_hash",
    "memory_state_hash",
    "market_snapshot_hash",
    "profile",
    "decision_date",
    "data_version",
    "code_commit",
    "architecture_id",
    "memory_enabled",
    "tools_enabled",
    "intervention_id",
    "agent_id",
    "agent_protocol_version",
    "collaboration_round",
    "inbox_hash",
    "message_bundle_hash",
)


def build_stage_cache_key(**fields: Any) -> str:
    """Build a deterministic key and reject missing or unknown dimensions."""
    # Validate the complete contract before hashing any caller-supplied values.
    provided = set(fields)
    expected = set(CACHE_KEY_FIELDS)
    unknown = provided - expected
    missing = expected - provided
    if unknown:
        raise ValueError(f"unknown cache-key fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing cache-key fields: {sorted(missing)}")
    # Preserve the declared field order for review while canonical JSON stabilizes nested maps.
    payload = {name: fields[name] for name in CACHE_KEY_FIELDS}
    return sha256_hex(canonical_json(payload))


__all__ = [
    "CACHE_KEY_FIELDS",
    "canonical_json",
    "sha256_hex",
    "build_stage_cache_key",
]
