"""Persistent factual and intervention stage cache."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from portbench.agent_eval.contracts import CachedStageResult, ProvenanceSource, ResultProvenance


FACTUAL_NAMESPACE = "factual"
INTERVENTION_NAMESPACE = "intervention"


def intervention_namespace(branch_id: str) -> str:
    """Return an isolated namespace for one intervention branch."""
    # Reject path separators so branch identifiers cannot escape the cache directory.
    clean = str(branch_id).strip()
    if not clean or any(char in clean for char in "/\\:"):
        raise ValueError("branch_id must be a non-empty file-safe string")
    return f"{INTERVENTION_NAMESPACE}__{clean}"


def _jsonable(value: Any) -> Any:
    """Convert cached values into lossless JSON-compatible structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cache output is not JSON serializable: {type(value)!r}")


class ReplayAdapter:
    """Store stage outputs with provenance and optional atomic persistence."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._lock = threading.RLock()
        self._stores: Dict[str, Dict[str, CachedStageResult]] = {
            FACTUAL_NAMESPACE: {},
            INTERVENTION_NAMESPACE: {},
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def _require_ns(self, namespace: str) -> Dict[str, CachedStageResult]:
        if namespace != FACTUAL_NAMESPACE and not namespace.startswith(INTERVENTION_NAMESPACE):
            raise ValueError("namespace must be factual or an intervention namespace")
        return self._stores.setdefault(namespace, {})

    def _path(self, namespace: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("persistent cache path is not configured")
        return self.cache_dir / f"{namespace}.json"

    def _load_all(self) -> None:
        """Load every persisted namespace into memory."""
        if self.cache_dir is None:
            return
        for path in self.cache_dir.glob("*.json"):
            namespace = path.stem
            # Ignore unrelated JSON artifacts stored beside the cache files.
            if namespace != FACTUAL_NAMESPACE and not namespace.startswith(INTERVENTION_NAMESPACE):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            store = self._require_ns(namespace)
            for key, item in payload.items():
                provenance = ResultProvenance(**item["provenance"])
                # A mismatched embedded key indicates corruption or an invalid manual edit.
                if provenance.cache_key != key:
                    raise ValueError(f"cache provenance mismatch for key={key}")
                store[key] = CachedStageResult(
                    output=item["output"],
                    provenance=provenance,
                    schema_version=item["schema_version"],
                )

    def _persist(self, namespace: str) -> None:
        """Atomically replace one namespace file after a cache mutation."""
        if self.cache_dir is None:
            return
        store = self._require_ns(namespace)
        payload = {
            key: {
                "output": _jsonable(entry.output),
                "provenance": asdict(entry.provenance),
                "schema_version": entry.schema_version,
            }
            for key, entry in store.items()
        }
        path = self._path(namespace)
        # Write and fsync a sibling temporary file before the atomic replacement.
        temp_path = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def lookup(
        self, stage_input_key: str, *, namespace: str = FACTUAL_NAMESPACE
    ) -> Optional[CachedStageResult]:
        """Return a cached stage result without mutating provenance."""
        with self._lock:
            return self._require_ns(namespace).get(stage_input_key)

    def put(
        self,
        stage_input_key: str,
        output: Any,
        provenance: ResultProvenance,
        *,
        namespace: str = FACTUAL_NAMESPACE,
        schema_version: str = "pipeline-v1",
    ) -> CachedStageResult:
        """Store one result and require provenance to match its content key."""
        if not stage_input_key:
            raise ValueError("stage_input_key must be non-empty")
        if provenance is None:
            raise ValueError("provenance is required")
        if provenance.cache_key and provenance.cache_key != stage_input_key:
            raise ValueError("provenance.cache_key must match stage_input_key")
        # Bind provenance to the content key before persistence.
        provenance.cache_key = stage_input_key
        entry = CachedStageResult(
            output=output,
            provenance=provenance,
            schema_version=schema_version,
        )
        with self._lock:
            self._require_ns(namespace)[stage_input_key] = entry
            self._persist(namespace)
        return entry

    def complete_from_cache(
        self, stage_input_key: str, *, namespace: str = FACTUAL_NAMESPACE
    ) -> Tuple[Any, ResultProvenance]:
        """Return a cached result and fail loudly on incomplete provenance."""
        hit = self.lookup(stage_input_key, namespace=namespace)
        if hit is None:
            raise KeyError(f"cache miss for key={stage_input_key!r} namespace={namespace!r}")
        if hit.provenance is None or not hit.provenance.cache_key:
            raise ValueError("cached result missing provenance")
        return hit.output, hit.provenance

    def complete_or_call(
        self,
        stage_input_key: str,
        call_fn: Callable[[], Any],
        *,
        namespace: str = FACTUAL_NAMESPACE,
        schema_version: str = "pipeline-v1",
        provenance_factory: Callable[[], ResultProvenance] | None = None,
    ) -> Tuple[Any, ResultProvenance]:
        """Return a cache hit or invoke and persist one new stage call."""
        hit = self.lookup(stage_input_key, namespace=namespace)
        if hit is not None:
            # Copy provenance so marking a hit never rewrites the stored call origin.
            provenance = ResultProvenance(**asdict(hit.provenance))
            provenance.source = ProvenanceSource.CURRENT_CACHE.value
            return hit.output, provenance
        # Persist only after the provider call and provenance construction both succeed.
        output = call_fn()
        provenance = (
            provenance_factory()
            if provenance_factory is not None
            else ResultProvenance(
                source=ProvenanceSource.NEW_API_CALL.value,
                cache_key=stage_input_key,
                schema_version=schema_version,
                request_count=1,
            )
        )
        self.put(
            stage_input_key,
            output,
            provenance,
            namespace=namespace,
            schema_version=schema_version,
        )
        return output, provenance


__all__ = [
    "FACTUAL_NAMESPACE",
    "INTERVENTION_NAMESPACE",
    "intervention_namespace",
    "ReplayAdapter",
]
