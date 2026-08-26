"""Durable LLM call, parse, and score artifacts for restart-safe evaluation."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from portbench.agent_eval.canonical import canonical_json, sha256_hex


FACTUAL_CALL_NAMESPACE = "factual"
INTERVENTION_CALL_PREFIX = "intervention__"


class TerminalCallFailure(RuntimeError):
    """Raised when a call exhausted its persisted automatic attempt budget."""


def _now() -> str:
    """Return a timezone-aware timestamp for an auditable artifact."""
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert supported values into deterministic JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"artifact value is not JSON serializable: {type(value)!r}")


def _validate_namespace(namespace: str) -> str:
    """Validate one factual or isolated intervention namespace."""
    value = str(namespace).strip()
    if value == FACTUAL_CALL_NAMESPACE:
        return value
    if value.startswith(INTERVENTION_CALL_PREFIX) and all(
        char not in value for char in "/\\:"
    ):
        return value
    raise ValueError("namespace must be factual or intervention__{branch_id}")


@dataclass(frozen=True)
class CallRequest:
    """Describe every behavior-bearing field of one provider request."""

    provider: str
    model: str
    model_revision: str
    stage_id: str
    system_prompt: str
    user_prompt: str
    response_schema: Mapping[str, Any]
    generation_config: Mapping[str, Any]
    visible_input: Mapping[str, Any]
    architecture_id: str = "SA"
    memory_mode: str = "none"
    tool_mode: str = "none"
    namespace: str = FACTUAL_CALL_NAMESPACE
    data_version: str = ""
    profile: str = ""
    decision_date: str = ""

    def __post_init__(self) -> None:
        _validate_namespace(self.namespace)
        if self.architecture_id != "SA":
            raise ValueError("SA-only call artifacts require architecture_id='SA'")
        if self.memory_mode != "none" or self.tool_mode != "none":
            raise ValueError("SA-only call artifacts require memory_mode=tool_mode='none'")

    def key_payload(self) -> dict[str, Any]:
        """Return the stable content that determines call reuse."""
        return {
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "stage_id": self.stage_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response_schema": _jsonable(self.response_schema),
            "generation_config": _jsonable(self.generation_config),
            "visible_input": _jsonable(self.visible_input),
            "architecture_id": self.architecture_id,
            "memory_mode": self.memory_mode,
            "tool_mode": self.tool_mode,
            "namespace": self.namespace,
            "data_version": self.data_version,
            "profile": self.profile,
            "decision_date": self.decision_date,
        }

    @property
    def call_key(self) -> str:
        """Return the digest used for exact request reuse."""
        return sha256_hex(canonical_json(self.key_payload()))


@dataclass(frozen=True)
class CallArtifact:
    """Persist a validated raw response without coupling it to a scorer."""

    call_key: str
    request: Mapping[str, Any]
    raw_response: str
    response_hash: str
    completed_at: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseArtifact:
    """Persist one parser-version-specific interpretation of a raw response."""

    call_key: str
    parser_version: str
    response_hash: str
    parsed_output: Any
    parsed_at: str


@dataclass(frozen=True)
class ScoreArtifact:
    """Persist a metric-version-specific score without reissuing a call."""

    call_key: str
    parser_version: str
    metric_version: str
    ground_truth_hash: str
    score_payload: Any
    scored_at: str


class CallArtifactStore:
    """Store atomic call artifacts and append-only provider attempt ledgers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, namespace: str, key: str, suffix: str = ".json") -> Path:
        """Return one artifact path under an isolated namespace."""
        return self.root / kind / _validate_namespace(namespace) / f"{key}{suffix}"

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        """Replace a JSON artifact atomically after flushing its contents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _append_attempt(self, namespace: str, call_key: str, payload: Mapping[str, Any]) -> None:
        """Append one fsynced audit event without rewriting prior attempts."""
        path = self._path("attempts", namespace, call_key, suffix=".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        line = canonical_json(_jsonable(payload)) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def attempts(self, request: CallRequest) -> list[dict[str, Any]]:
        """Return all persisted attempt events for one exact request."""
        path = self._path("attempts", request.namespace, request.call_key, suffix=".jsonl")
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def load_call(self, request: CallRequest) -> Optional[CallArtifact]:
        """Load a validated call artifact when the exact request completed."""
        path = self._path("calls", request.namespace, request.call_key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("call_key") != request.call_key:
            raise ValueError("call artifact key mismatch")
        if raw.get("request") != request.key_payload():
            raise ValueError("call artifact request mismatch")
        return CallArtifact(**raw)

    def _load_linked_recovery(
        self,
        request: CallRequest,
        ledger: list[dict[str, Any]],
    ) -> Optional[CallArtifact]:
        """Load one auditable recovery call that preserves the logical request."""
        expected_core = dict(request.key_payload())
        expected_core.pop("generation_config")
        for event in reversed(ledger):
            if event.get("event") != "recovery_available":
                continue
            recovery_key = event.get("recovery_call_key")
            if not isinstance(recovery_key, str):
                continue
            path = self._path("calls", request.namespace, recovery_key)
            if not path.exists():
                continue
            try:
                artifact = CallArtifact(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if artifact.call_key != recovery_key:
                continue
            recovery_core = dict(artifact.request)
            recovery_core.pop("generation_config", None)
            if recovery_core != expected_core:
                continue
            if artifact.provenance.get("recovery_of_call_key") != request.call_key:
                continue
            return artifact
        return None

    def load_parse(self, request: CallRequest, parser_version: str) -> Optional[ParseArtifact]:
        """Load a parser-version-specific result for one exact raw response."""
        path = self._path("parses", request.namespace, f"{request.call_key}.{parser_version}")
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ParseArtifact(**raw)

    def load_score(
        self,
        request: CallRequest,
        parser_version: str,
        metric_version: str,
        ground_truth: Any,
    ) -> Optional[ScoreArtifact]:
        """Load a score only when it matches the current GT and metric version."""
        key = f"{request.call_key}.{parser_version}.{metric_version}"
        path = self._path("scores", request.namespace, key)
        if not path.exists():
            return None
        artifact = ScoreArtifact(**json.loads(path.read_text(encoding="utf-8")))
        expected = sha256_hex(canonical_json(_jsonable(ground_truth)))
        if artifact.ground_truth_hash != expected:
            return None
        return artifact

    def _record_parse(
        self,
        request: CallRequest,
        parser_version: str,
        raw_response: str,
        parsed_output: Any,
    ) -> ParseArtifact:
        """Persist a parser result after the response satisfies its contract."""
        artifact = ParseArtifact(
            call_key=request.call_key,
            parser_version=parser_version,
            response_hash=sha256_hex(raw_response),
            parsed_output=_jsonable(parsed_output),
            parsed_at=_now(),
        )
        self._atomic_write(
            self._path("parses", request.namespace, f"{request.call_key}.{parser_version}"),
            asdict(artifact),
        )
        return artifact

    def score(
        self,
        request: CallRequest,
        parser_version: str,
        metric_version: str,
        ground_truth: Any,
        score_fn: Callable[[Any], Any],
    ) -> ScoreArtifact:
        """Reuse or persist a derived score without touching the provider."""
        cached = self.load_score(request, parser_version, metric_version, ground_truth)
        if cached is not None:
            return cached
        parsed = self.load_parse(request, parser_version)
        if parsed is None:
            raise KeyError("cannot score a call without a parsed artifact")
        artifact = ScoreArtifact(
            call_key=request.call_key,
            parser_version=parser_version,
            metric_version=metric_version,
            ground_truth_hash=sha256_hex(canonical_json(_jsonable(ground_truth))),
            score_payload=_jsonable(score_fn(parsed.parsed_output)),
            scored_at=_now(),
        )
        key = f"{request.call_key}.{parser_version}.{metric_version}"
        self._atomic_write(self._path("scores", request.namespace, key), asdict(artifact))
        return artifact

    def complete_or_call(
        self,
        request: CallRequest,
        *,
        parser_version: str,
        parse: Callable[[str], Any],
        call_fn: Callable[[], str],
        provenance: Optional[Mapping[str, Any]] = None,
        max_attempts: int = 3,
        backoff_seconds: tuple[float, ...] = (2.0, 8.0),
        retry_failed: bool = False,
    ) -> tuple[Any, CallArtifact, bool]:
        """Return a parsed cache hit or persist a new validated provider response."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self._lock:
            existing = self.load_call(request)
            if existing is not None:
                try:
                    parsed_output = parse(existing.raw_response)
                except Exception as exc:
                    self._append_attempt(
                        request.namespace,
                        request.call_key,
                        {
                            "timestamp": _now(),
                            "event": "reparse_failed",
                            "parser_version": parser_version,
                            "error": str(exc),
                        },
                    )
                else:
                    self._record_parse(request, parser_version, existing.raw_response, parsed_output)
                    return parsed_output, existing, True

            ledger = self.attempts(request)
            recorded_now = not any(item.get("event") == "request_recorded" for item in ledger)
            if recorded_now:
                self._append_attempt(
                    request.namespace,
                    request.call_key,
                    {
                        "timestamp": _now(),
                        "event": "request_recorded",
                        "request_hash": request.call_key,
                        "request": request.key_payload(),
                        "provenance": dict(provenance or {}),
                    },
                )
            for item in reversed(ledger):
                if item.get("event") != "received":
                    continue
                raw_response = item.get("raw_response")
                if not isinstance(raw_response, str):
                    continue
                try:
                    parsed_output = parse(raw_response)
                except Exception:
                    continue
                artifact_provenance = dict(provenance or {})
                artifact_provenance["reparsed_from_attempt"] = item.get("attempt")
                artifact = CallArtifact(
                    call_key=request.call_key,
                    request=request.key_payload(),
                    raw_response=raw_response,
                    response_hash=sha256_hex(raw_response),
                    completed_at=_now(),
                    provenance=artifact_provenance,
                )
                self._atomic_write(
                    self._path("calls", request.namespace, request.call_key), asdict(artifact)
                )
                self._record_parse(request, parser_version, raw_response, parsed_output)
                self._append_attempt(
                    request.namespace,
                    request.call_key,
                    {
                        "timestamp": _now(),
                        "event": "reparsed_validated",
                        "attempt": item.get("attempt"),
                        "parser_version": parser_version,
                    },
                )
                return parsed_output, artifact, True
            recovered = self._load_linked_recovery(request, ledger)
            if recovered is not None:
                try:
                    parsed_output = parse(recovered.raw_response)
                except Exception as exc:
                    self._append_attempt(
                        request.namespace,
                        request.call_key,
                        {
                            "timestamp": _now(),
                            "event": "recovery_reparse_failed",
                            "recovery_call_key": recovered.call_key,
                            "parser_version": parser_version,
                            "error": str(exc),
                        },
                    )
                else:
                    self._record_parse(
                        request,
                        parser_version,
                        recovered.raw_response,
                        parsed_output,
                    )
                    if not any(
                        item.get("event") == "recovery_reused"
                        and item.get("recovery_call_key") == recovered.call_key
                        and item.get("parser_version") == parser_version
                        for item in ledger
                    ):
                        self._append_attempt(
                            request.namespace,
                            request.call_key,
                            {
                                "timestamp": _now(),
                                "event": "recovery_reused",
                                "recovery_call_key": recovered.call_key,
                                "parser_version": parser_version,
                            },
                        )
                    return parsed_output, recovered, True
            completed_attempts = {
                int(item["attempt"])
                for item in ledger
                if item.get("event") in {"received", "provider_error"}
                and isinstance(item.get("attempt"), int)
            }
            started_attempts = {
                int(item["attempt"])
                for item in ledger
                if item.get("event") == "provider_started"
                and isinstance(item.get("attempt"), int)
            }
            interrupted_attempts = {
                int(item["attempt"])
                for item in ledger
                if item.get("event") == "interrupted"
                and isinstance(item.get("attempt"), int)
            }
            if not recorded_now and not started_attempts and not completed_attempts:
                self._append_attempt(
                    request.namespace,
                    request.call_key,
                    {
                        "timestamp": _now(),
                        "event": "interrupted",
                        "attempt": 1,
                        "error": "legacy request had no persisted provider-start event",
                    },
                )
                started_attempts.add(1)
                interrupted_attempts.add(1)
            for stale_attempt in sorted(started_attempts - completed_attempts - interrupted_attempts):
                self._append_attempt(
                    request.namespace,
                    request.call_key,
                    {
                        "timestamp": _now(),
                        "event": "interrupted",
                        "attempt": stale_attempt,
                        "error": "process exited before a provider outcome was persisted",
                    },
                )
            attempted_numbers = started_attempts | completed_attempts
            if len(attempted_numbers) >= max_attempts and not retry_failed:
                raise TerminalCallFailure(
                    f"call {request.call_key} exhausted {max_attempts} persisted attempts"
                )
            attempt_number = max(attempted_numbers, default=0)
            limit = max_attempts + (1 if retry_failed else 0)

            while attempt_number < limit:
                attempt_number += 1
                started = time.perf_counter()
                self._append_attempt(
                    request.namespace,
                    request.call_key,
                    {
                        "timestamp": _now(),
                        "event": "provider_started",
                        "attempt": attempt_number,
                    },
                )
                try:
                    raw_response = str(call_fn())
                except Exception as exc:
                    self._append_attempt(
                        request.namespace,
                        request.call_key,
                        {
                            "timestamp": _now(),
                            "event": "provider_error",
                            "attempt": attempt_number,
                            "error": str(exc),
                            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        },
                    )
                else:
                    self._append_attempt(
                        request.namespace,
                        request.call_key,
                        {
                            "timestamp": _now(),
                            "event": "received",
                            "attempt": attempt_number,
                            "raw_response": raw_response,
                            "response_hash": sha256_hex(raw_response),
                            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        },
                    )
                    try:
                        parsed_output = parse(raw_response)
                    except Exception as exc:
                        self._append_attempt(
                            request.namespace,
                            request.call_key,
                            {
                                "timestamp": _now(),
                                "event": "validation_error",
                                "attempt": attempt_number,
                                "error": str(exc),
                            },
                        )
                    else:
                        artifact = CallArtifact(
                            call_key=request.call_key,
                            request=request.key_payload(),
                            raw_response=raw_response,
                            response_hash=sha256_hex(raw_response),
                            completed_at=_now(),
                            provenance=dict(provenance or {}),
                        )
                        self._atomic_write(
                            self._path("calls", request.namespace, request.call_key), asdict(artifact)
                        )
                        self._record_parse(request, parser_version, raw_response, parsed_output)
                        self._append_attempt(
                            request.namespace,
                            request.call_key,
                            {"timestamp": _now(), "event": "validated", "attempt": attempt_number},
                        )
                        return parsed_output, artifact, False
                if attempt_number < limit:
                    pause = backoff_seconds[min(attempt_number - 1, len(backoff_seconds) - 1)]
                    time.sleep(max(0.0, pause))
        raise TerminalCallFailure(f"call {request.call_key} did not produce a valid response")


__all__ = [
    "FACTUAL_CALL_NAMESPACE",
    "INTERVENTION_CALL_PREFIX",
    "TerminalCallFailure",
    "CallRequest",
    "CallArtifact",
    "ParseArtifact",
    "ScoreArtifact",
    "CallArtifactStore",
]
