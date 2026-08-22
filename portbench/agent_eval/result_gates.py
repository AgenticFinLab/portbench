"""Protocol / provenance validators and ranking schema guards."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from portbench.agent_eval.contracts import (
    STEP_REPLAY_FORBIDDEN_METRICS,
    ProvenanceSource,
    ResultProvenance,
)
from portbench.metrics.plan_outcome_scores import (
    S4_ENV_OUTCOME_KEYS,
    S4_PLAN_QUALITY_KEYS,
    S5_ENV_OUTCOME_KEYS,
    S5_PLAN_QUALITY_KEYS,
)

_FORBIDDEN_LOWER = frozenset(
    m.lower() for m in STEP_REPLAY_FORBIDDEN_METRICS
) | frozenset(
    {
        "sharpe",
        "cagr",
        "cumulative_max_dd",
        "cumulative_nav",
        "max_dd_cumulative",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "total_return",
        "final_nav",
        "calmar_ratio",
    }
)


def _find_forbidden_fields(value: Any, path: str = "") -> List[str]:
    """Return forbidden metric paths from an arbitrarily nested result."""
    hits: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in _FORBIDDEN_LOWER:
                hits.append(item_path)
            hits.extend(_find_forbidden_fields(item, item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(_find_forbidden_fields(item, f"{path}[{index}]"))
    return hits


def validate_step_replay_record(metrics: Mapping[str, Any]) -> None:
    """Reject closed-loop metrics anywhere in a step-replay record."""
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    forbidden_hit = _find_forbidden_fields(metrics)
    if forbidden_hit:
        raise ValueError(
            "step-replay forbids metrics: " + ", ".join(sorted(map(str, forbidden_hit)))
        )


def validate_provenance(p: Any) -> None:
    """Raise ValueError if provenance is missing or incomplete."""
    if p is None:
        raise ValueError("provenance is required")
    if isinstance(p, ResultProvenance):
        source = p.source
        cache_key = p.cache_key
    elif isinstance(p, Mapping):
        source = p.get("source")
        cache_key = p.get("cache_key", "")
    else:
        raise TypeError(f"unsupported provenance type: {type(p)!r}")

    allowed = {s.value for s in ProvenanceSource}
    if source not in allowed:
        raise ValueError(
            f"invalid provenance.source={source!r}; expected one of {sorted(allowed)}"
        )
    if not isinstance(cache_key, str) or not cache_key.strip():
        raise ValueError("provenance.cache_key must be a non-empty string")


class SchemaMixError(ValueError):
    """Raised when ranking rows mix incompatible S4/S5 schema versions."""


def assert_homogeneous_schema_versions(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_key: str = "schema_version",
) -> str:
    """Reject mixing deterministic results with collaborative agentic results.

    Returns the single schema version string when homogeneous.
    """
    versions = {str(r.get(schema_key, "")) for r in rows}
    versions.discard("")
    if not versions:
        raise SchemaMixError("ranking rows missing schema_version")
    if len(versions) > 1:
        raise SchemaMixError(
            f"cannot mix schema versions in one ranking list: {sorted(versions)}"
        )
    return next(iter(versions))


def build_ranking_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    score_keys: Sequence[str],
    schema_key: str = "schema_version",
) -> List[Dict[str, Any]]:
    """Build a ranking table; refuses mixed schema versions.

    Caller must pass a single score family via score_keys (plan or environment).
    """
    rows = [dict(r) for r in records]
    schema = assert_homogeneous_schema_versions(rows, schema_key=schema_key)
    plan_keys = S4_PLAN_QUALITY_KEYS | S5_PLAN_QUALITY_KEYS
    env_keys = S4_ENV_OUTCOME_KEYS | S5_ENV_OUTCOME_KEYS
    sk = set(score_keys)
    if sk & plan_keys and sk & env_keys:
        raise SchemaMixError(
            "ranking score_keys must not mix plan_quality with environment_outcome keys"
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        item = {schema_key: schema, "id": r.get("id")}
        for k in score_keys:
            item[k] = r.get(k)
        out.append(item)
    return out


__all__ = [
    "validate_step_replay_record",
    "validate_provenance",
    "SchemaMixError",
    "assert_homogeneous_schema_versions",
    "build_ranking_rows",
]
