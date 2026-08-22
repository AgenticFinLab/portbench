"""Frozen contracts for PortBench paper-upgrade scaffolding.

Implements plan steps 1–3 contract surface: schema versions, result protocols,
architecture IDs, and shared dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


class SchemaVersion(str, Enum):
    """Pipeline schema versions frozen for the upgrade."""

    PIPELINE_V1 = "pipeline-v1"
    PIPELINE_V1_DETERMINISTIC = "pipeline-v1-deterministic"
    PIPELINE_V2_AGENTIC = "pipeline-v2-agentic"
    PIPELINE_V3_COLLAB = "pipeline-v3-collab"


# Module-level aliases matching plan string literals.
PIPELINE_V1 = SchemaVersion.PIPELINE_V1.value
PIPELINE_V1_DETERMINISTIC = SchemaVersion.PIPELINE_V1_DETERMINISTIC.value
PIPELINE_V2_AGENTIC = SchemaVersion.PIPELINE_V2_AGENTIC.value
PIPELINE_V3_COLLAB = SchemaVersion.PIPELINE_V3_COLLAB.value

# S4/S5 dual-layer schema tags (plan vs deterministic environment).
# Legacy stages map to DETERMINISTIC; agentic upgrade path uses AGENTIC.
S4S5_SCHEMA_DETERMINISTIC = PIPELINE_V1_DETERMINISTIC
S4S5_SCHEMA_AGENTIC = PIPELINE_V3_COLLAB


class ResultProtocol(str, Enum):
    """Result storage / reporting protocol."""

    STEP_REPLAY = "step-replay"
    CLOSED_LOOP = "closed-loop"
    BLACK_BOX = "black-box"


STEP_REPLAY = ResultProtocol.STEP_REPLAY.value
CLOSED_LOOP = ResultProtocol.CLOSED_LOOP.value
BLACK_BOX = ResultProtocol.BLACK_BOX.value


class ProvenanceSource(str, Enum):
    LEGACY_REPLAY = "legacy_replay"
    CURRENT_CACHE = "current_cache"
    NEW_API_CALL = "new_api_call"


class ArchitectureId(str, Enum):
    """2x2x2 controlled architecture IDs (topology × memory × tools).

    SA* = shared-agent; MA* = multi-agent / role-decomposed.
    Suffix -M = memory on; -T = tools on; -MT = both.
    """

    SA = "SA"
    SA_T = "SA-T"
    SA_M = "SA-M"
    SA_MT = "SA-MT"
    MA = "MA"
    MA_T = "MA-T"
    MA_M = "MA-M"
    MA_MT = "MA-MT"


ARCHITECTURE_IDS: tuple[str, ...] = tuple(a.value for a in ArchitectureId)


# Metrics allowed / forbidden by protocol.
STEP_REPLAY_FORBIDDEN_METRICS: frozenset[str] = frozenset(
    {
        "sharpe",
        "Sharpe",
        "cagr",
        "CAGR",
        "cumulative_max_dd",
        "CumulativeMaxDD",
        "max_dd_cumulative",
        "cumulative_nav",
        "CumulativeNAV",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "total_return",
        "final_nav",
        "calmar_ratio",
    }
)

STEP_REPLAY_ALLOWED_METRIC_HINTS: frozenset[str] = frozenset(
    {
        "stage_score",
        "plan_quality",
        "implementation_shortfall",
        "single_period_return",
        "single_period_risk",
        "cost",
        "turnover",
        "intervention_delta",
    }
)

CLOSED_LOOP_ALLOWED_METRIC_HINTS: frozenset[str] = frozenset(
    {
        "sharpe",
        "sortino",
        "cagr",
        "cumulative_return",
        "max_dd",
        "cvar",
        "stress_pass_rate",
        "compliance",
        "total_cost",
        "resource_audit",
    }
)

ALLOWED_METRICS_BY_PROTOCOL: Mapping[str, frozenset[str]] = {
    ResultProtocol.STEP_REPLAY.value: STEP_REPLAY_ALLOWED_METRIC_HINTS,
    ResultProtocol.CLOSED_LOOP.value: CLOSED_LOOP_ALLOWED_METRIC_HINTS,
}


@dataclass
class OrderIntent:
    """Single-asset order intent within an ExecutionPlan.

    Prefer ``target_weight`` when known; otherwise ``delta_weight`` relative to
    current holdings. Direction is ``buy`` | ``sell`` | ``hold``.
    """

    asset: str = ""
    # buy | sell | hold
    direction: str = "hold"
    target_weight: Optional[float] = None
    delta_weight: Optional[float] = None
    order_type: str = "market"
    urgency: str = "normal"
    slip_limit: Optional[float] = None

    def __post_init__(self) -> None:
        if self.direction not in {"buy", "sell", "hold"}:
            raise ValueError(f"invalid order direction: {self.direction!r}")
        if self.order_type not in {"market", "limit"}:
            raise ValueError(f"invalid order type: {self.order_type!r}")
        for name, value in (
            ("target_weight", self.target_weight),
            ("delta_weight", self.delta_weight),
            ("slip_limit", self.slip_limit),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.target_weight is not None and not 0.0 <= float(self.target_weight) <= 1.0:
            raise ValueError("target_weight must be in [0, 1]")
        if self.slip_limit is not None and float(self.slip_limit) < 0.0:
            raise ValueError("slip_limit must be non-negative")


def _coerce_order_intent(item: Any) -> "OrderIntent":
    if isinstance(item, OrderIntent):
        return item
    if isinstance(item, Mapping):
        return OrderIntent(
            asset=str(item.get("asset", "")),
            direction=str(item.get("direction", "hold")),
            target_weight=item.get("target_weight"),
            delta_weight=item.get("delta_weight"),
            order_type=str(item.get("order_type", "market")),
            urgency=str(item.get("urgency", "normal")),
            slip_limit=item.get("slip_limit"),
        )
    raise TypeError(f"Cannot coerce order intent from {type(item)!r}")


@dataclass
class ExecutionPlan:
    """S4-style execution plan payload.

    Backward compatible: legacy fields (``order_direction``, ``target_scale``,
    ``order_type``, ``urgency``, ``slip_limit``) remain. Multi-asset plans
    should populate ``orders`` (list of ``OrderIntent`` or plain dicts).
    """

    order_direction: str = ""
    target_scale: float = 1.0
    order_type: str = "market"
    urgency: str = "normal"
    slip_limit: Optional[float] = None
    orders: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.order_type not in {"market", "limit"}:
            raise ValueError(f"invalid order type: {self.order_type!r}")
        if not math.isfinite(float(self.target_scale)) or float(self.target_scale) < 0.0:
            raise ValueError("target_scale must be finite and non-negative")
        if self.slip_limit is not None and float(self.slip_limit) < 0.0:
            raise ValueError("slip_limit must be non-negative")

    def normalized_orders(self) -> List[OrderIntent]:
        """Return ``orders`` coerced to ``OrderIntent`` instances."""
        return [_coerce_order_intent(o) for o in self.orders]


@dataclass
class ExecutionResult:
    """Deterministic execution outcome for an ExecutionPlan."""

    filled_weights: Dict[str, float] = field(default_factory=dict)
    implementation_shortfall: float = 0.0
    turnover: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskControlDecision:
    """S5-style risk control decision (agent plan layer).

    ``action`` is ``hold`` | ``scale_down`` | ``rebalance``.
    ``alerts`` may be plain strings or structured dicts; ``corrective_weights``
    are applied by the deterministic environment when action is not hold.
    """

    # hold | scale_down | rebalance
    action: str = "hold"
    alerts: List[Any] = field(default_factory=list)
    corrective_weights: Optional[Dict[str, float]] = None
    scale_factor: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {"hold", "scale_down", "rebalance"}:
            raise ValueError(f"invalid risk action: {self.action!r}")
        if self.scale_factor is not None:
            factor = float(self.scale_factor)
            if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
                raise ValueError("scale_factor must be in [0, 1]")
        if self.corrective_weights is not None:
            values = [float(value) for value in self.corrective_weights.values()]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("corrective_weights must be finite and non-negative")
            if values and abs(sum(values) - 1.0) > 1e-4:
                raise ValueError("corrective_weights must sum to one")


@dataclass
class RiskEvaluationResult:
    """Deterministic risk evaluation outcome."""

    var: Optional[float] = None
    cvar: Optional[float] = None
    drawdown: Optional[float] = None
    constraint_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """Per-episode decision trace spanning stages."""

    episode_id: str = ""
    decision_date: str = ""
    architecture_id: str = ArchitectureId.SA.value
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    result_protocol: str = ResultProtocol.STEP_REPLAY.value
    schema_version: str = SchemaVersion.PIPELINE_V1.value
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageReplayKey:
    """Logical key for a stage replay lookup (namespace + cache key)."""

    # factual | intervention
    namespace: str
    cache_key: str


@dataclass
class CachedStageResult:
    """Cached stage output plus forced provenance."""

    output: Any
    provenance: "ResultProvenance"
    schema_version: str = SchemaVersion.PIPELINE_V1.value


@dataclass
class ResultProvenance:
    """Mandatory provenance attached to every stage result."""

    # legacy_replay | current_cache | new_api_call
    source: str
    cache_key: str = ""
    legacy_path: Optional[str] = None
    schema_version: str = SchemaVersion.PIPELINE_V1.value
    prompt_version: Optional[str] = None
    data_version: Optional[str] = None
    code_commit: Optional[str] = None
    memory_hash: Optional[str] = None
    toolset_hash: Optional[str] = None
    token_exact: Optional[float] = None
    token_est: Optional[float] = None
    request_count: int = 0
    tool_call_count: int = 0

    def __post_init__(self) -> None:
        allowed = {s.value for s in ProvenanceSource}
        if self.source not in allowed:
            raise ValueError(
                f"provenance.source must be one of {sorted(allowed)}, got {self.source!r}"
            )
        if not isinstance(self.cache_key, str):
            raise TypeError("provenance.cache_key must be a string")


__all__ = [
    "SchemaVersion",
    "PIPELINE_V1",
    "PIPELINE_V1_DETERMINISTIC",
    "PIPELINE_V2_AGENTIC",
    "PIPELINE_V3_COLLAB",
    "S4S5_SCHEMA_DETERMINISTIC",
    "S4S5_SCHEMA_AGENTIC",
    "ResultProtocol",
    "STEP_REPLAY",
    "CLOSED_LOOP",
    "BLACK_BOX",
    "ProvenanceSource",
    "ArchitectureId",
    "ARCHITECTURE_IDS",
    "STEP_REPLAY_FORBIDDEN_METRICS",
    "STEP_REPLAY_ALLOWED_METRIC_HINTS",
    "CLOSED_LOOP_ALLOWED_METRIC_HINTS",
    "ALLOWED_METRICS_BY_PROTOCOL",
    "OrderIntent",
    "ExecutionPlan",
    "ExecutionResult",
    "RiskControlDecision",
    "RiskEvaluationResult",
    "DecisionTrace",
    "StageReplayKey",
    "CachedStageResult",
    "ResultProvenance",
]
