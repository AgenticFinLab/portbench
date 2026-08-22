"""Simulator interventions over factual stage outputs and closed-loop branches."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from portbench.agent_eval import pit_repair
from portbench.agent_eval.base import (
    EpisodeResult,
    EvalPipeline,
    MarketSnapshot,
    S1Output,
    S2Output,
    S3Output,
    S4Output,
    S5Output,
    StageID,
)
from portbench.agent_eval.contracts import ResultProvenance
from portbench.agent_eval.replay_adapter import ReplayAdapter, intervention_namespace
from portbench.agent_eval.stages import S4ExecutionSimulation, S5RiskMonitoring


STAGE_ORDER = (
    StageID.S1_MARKET_INTERPRETATION,
    StageID.S2_SIGNAL_GENERATION,
    StageID.S3_WEIGHT_OPTIMIZATION,
    StageID.S4_EXECUTION_SIMULATION,
    StageID.S5_RISK_MONITORING,
)
STAGE_BY_TEXT = {stage.value: stage for stage in STAGE_ORDER}


@dataclass
class StageInterventionResult:
    """Store factual and intervened episode outcomes with labeled deltas."""

    stage_id: str
    operator: str
    factual: EpisodeResult
    intervened: EpisodeResult
    score_delta: Dict[str, float]


@dataclass
class ClosedLoopBranchResult:
    """Store one sequential intervention branch without mixing protocols."""

    branch_id: str
    factual_nav: list[float]
    intervention_nav: list[float]
    factual_weights: list[Dict[str, float]]
    intervention_weights: list[Dict[str, float]]
    episode_interventions: list[StageInterventionResult]


def _stage(stage_id: str | StageID) -> StageID:
    """Normalize one stage identifier."""
    if isinstance(stage_id, StageID):
        return stage_id
    if stage_id not in STAGE_BY_TEXT:
        raise ValueError(f"unsupported stage_id={stage_id!r}")
    return STAGE_BY_TEXT[stage_id]


def _plain(output: Any) -> Dict[str, Any]:
    """Serialize one typed stage output for perturbation."""
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        return asdict(output)
    if isinstance(output, Mapping):
        return dict(output)
    raise TypeError(f"unsupported stage output: {type(output)!r}")


def _coerce(stage_id: StageID, payload: Mapping[str, Any]) -> Any:
    """Convert serialized repair or perturb output into pipeline dataclasses."""
    if stage_id is StageID.S1_MARKET_INTERPRETATION:
        return S1Output(
            asset_views={str(key): float(value) for key, value in payload.get("asset_views", {}).items()},
            macro_summary=str(payload.get("macro_summary", "")),
            detected_regime=str(payload.get("detected_regime", "unknown")),
            confidence=float(payload.get("confidence", 0.5)),
        )
    if stage_id is StageID.S2_SIGNAL_GENERATION:
        return S2Output(
            signals={str(key): str(value) for key, value in payload.get("signals", {}).items()},
            strengths={str(key): float(value) for key, value in payload.get("strengths", {}).items()},
            reasoning=str(payload.get("reasoning", "")),
        )
    if stage_id is StageID.S3_WEIGHT_OPTIMIZATION:
        return S3Output(
            weights={str(key): float(value) for key, value in payload.get("weights", {}).items()},
            expected_return=float(payload.get("expected_return", 0.0)),
            expected_vol=float(payload.get("expected_vol", 0.0)),
            sharpe_estimate=float(payload.get("sharpe_estimate", 0.0)),
        )
    if stage_id is StageID.S4_EXECUTION_SIMULATION:
        return S4Output(**{key: value for key, value in payload.items() if key in S4Output.__dataclass_fields__})
    if stage_id is StageID.S5_RISK_MONITORING:
        return S5Output(**{key: value for key, value in payload.items() if key in S5Output.__dataclass_fields__})
    raise ValueError(stage_id)


def apply_repair(
    stage_id: str,
    context: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply one Point-in-Time repair operator and return its serialized output."""
    repair_functions = {
        "S1": pit_repair.repair_s1,
        "S2": pit_repair.repair_s2,
        "S3": pit_repair.repair_s3,
        "S4": pit_repair.repair_s4,
        "S5": pit_repair.repair_s5,
    }
    if stage_id not in repair_functions:
        raise ValueError(f"unsupported stage_id={stage_id!r}")
    # Reject future-bearing inputs before dispatching to any repair implementation.
    pit_repair._raise_if_future(context, **kwargs)
    return repair_functions[stage_id](context, **kwargs)


def apply_perturb(
    stage_id: str,
    factual_output: Mapping[str, Any],
    *,
    perturb_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Perturb a factual stage output without changing the stage input."""
    _stage(stage_id)
    perturbed = perturb_fn(dict(factual_output))
    if not isinstance(perturbed, Mapping):
        raise TypeError("perturb_fn must return a mapping")
    return dict(perturbed)


def _repair_payload(
    stage_id: StageID, snapshot: MarketSnapshot, factual: EpisodeResult
) -> Dict[str, Any]:
    """Build a schema-compatible repair from Point-in-Time snapshot fields."""
    lookback = snapshot.return_data
    if stage_id is StageID.S1_MARKET_INTERPRETATION:
        return pit_repair.repair_s1(lookback_returns=lookback)
    if stage_id is StageID.S2_SIGNAL_GENERATION:
        s1_payload = _plain(factual.stage_outputs[StageID.S1_MARKET_INTERPRETATION])
        return pit_repair.repair_s2(lookback_returns=lookback, s1_output=s1_payload)
    if stage_id is StageID.S3_WEIGHT_OPTIMIZATION:
        return pit_repair.repair_s3(lookback_returns=lookback)
    if stage_id is StageID.S4_EXECUTION_SIMULATION:
        s3_output = factual.stage_outputs[StageID.S3_WEIGHT_OPTIMIZATION]
        repaired = S4ExecutionSimulation()._execute(s3_output, snapshot)
        return _plain(repaired)
    s4_output = factual.stage_outputs[StageID.S4_EXECUTION_SIMULATION]
    repaired = S5RiskMonitoring()._monitor(s4_output.executed_weights, snapshot)
    repaired.final_weights = dict(s4_output.executed_weights)
    return _plain(repaired)


def intervene_from_factual(
    pipeline: EvalPipeline,
    snapshot: MarketSnapshot,
    factual: EpisodeResult,
    stage_id: str | StageID,
    *,
    operator: str = "repair",
    perturb_fn: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> StageInterventionResult:
    """Replace one factual stage output and rerun only its downstream stages."""
    stage = _stage(stage_id)
    if operator == "repair":
        payload = _repair_payload(stage, snapshot, factual)
    elif operator == "perturb" and perturb_fn is not None:
        payload = apply_perturb(stage.value, _plain(factual.stage_outputs[stage]), perturb_fn=perturb_fn)
    else:
        raise ValueError("operator must be repair or perturb with perturb_fn")
    replacement = _coerce(stage, payload)
    stage_index = STAGE_ORDER.index(stage)
    # Preserve factual upstream outputs and rerun only the affected suffix.
    reused = {
        upstream: factual.stage_outputs[upstream]
        for upstream in STAGE_ORDER[:stage_index]
        if upstream in factual.stage_outputs
    }
    intervened = pipeline.run_episode(
        snapshot,
        stage_overrides={stage: replacement},
        reuse_outputs=reused,
    )
    deltas = {
        downstream.value: float(intervened.stage_scores.get(downstream, 0.0))
        - float(factual.stage_scores.get(downstream, 0.0))
        for downstream in STAGE_ORDER[stage_index:]
    }
    return StageInterventionResult(
        stage_id=stage.value,
        operator=operator,
        factual=factual,
        intervened=intervened,
        score_delta=deltas,
    )


def run_stage_intervention(
    pipeline: EvalPipeline,
    snapshot: MarketSnapshot,
    stage_id: str | StageID,
    *,
    operator: str = "repair",
    perturb_fn: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> StageInterventionResult:
    """Run one factual episode followed by a labeled simulator intervention."""
    factual = pipeline.run_episode(snapshot)
    return intervene_from_factual(
        pipeline,
        snapshot,
        factual,
        stage_id,
        operator=operator,
        perturb_fn=perturb_fn,
    )


def _final_weights(result: EpisodeResult) -> Dict[str, float]:
    """Extract the final portfolio state from a completed episode."""
    s5_output = result.stage_outputs.get(StageID.S5_RISK_MONITORING)
    if s5_output is not None and getattr(s5_output, "final_weights", None):
        return dict(s5_output.final_weights)
    s4_output = result.stage_outputs.get(StageID.S4_EXECUTION_SIMULATION)
    if s4_output is not None and s4_output.executed_weights:
        return dict(s4_output.executed_weights)
    return dict(result.stage_outputs[StageID.S3_WEIGHT_OPTIMIZATION].weights)


def _realized_period_return(snapshot: MarketSnapshot, weights: Mapping[str, float]) -> float:
    """Apply realized forward returns only in the post-decision simulator environment."""
    if not snapshot.future_return_data:
        return 0.0
    total = 0.0
    for asset, weight in weights.items():
        values = snapshot.future_return_data.get(asset)
        if values is None:
            continue
        series = values.dropna() if hasattr(values, "dropna") else values
        asset_return = float((1.0 + series).prod() - 1.0) if len(series) else 0.0
        total += float(weight) * asset_return
    return total


def run_closed_loop_intervention(
    factual_pipeline: EvalPipeline,
    intervention_pipeline: EvalPipeline,
    snapshots: Sequence[MarketSnapshot],
    *,
    intervention_index: int,
    stage_id: str | StageID,
    branch_id: str,
) -> ClosedLoopBranchResult:
    """Propagate a stage repair through independent factual and branch states."""
    if not 0 <= intervention_index < len(snapshots):
        raise IndexError("intervention_index is outside the snapshot sequence")
    factual_weights = dict(snapshots[0].current_weights)
    branch_weights = dict(factual_weights)
    factual_nav = float(snapshots[0].portfolio_value)
    branch_nav = factual_nav
    factual_navs = [factual_nav]
    branch_navs = [branch_nav]
    factual_history = [dict(factual_weights)]
    branch_history = [dict(branch_weights)]
    interventions: list[StageInterventionResult] = []
    for index, snapshot in enumerate(snapshots):
        if index == intervention_index:
            factual_runtime = getattr(factual_pipeline, "runtime", None)
            branch_runtime = getattr(intervention_pipeline, "runtime", None)
            if factual_runtime is not None and branch_runtime is not None:
                # Fork memory once, then keep branch writes isolated thereafter.
                branch_runtime.clone_memory_from(factual_runtime)
        factual_snapshot = dataclasses.replace(
            snapshot, current_weights=factual_weights, portfolio_value=factual_nav
        )
        factual_result = factual_pipeline.run_episode(factual_snapshot)
        factual_weights = _final_weights(factual_result)
        if index < intervention_index:
            # Before the intervention, both trajectories share the factual state.
            branch_weights = dict(factual_weights)
            branch_nav = factual_nav
        else:
            branch_snapshot = dataclasses.replace(
                snapshot, current_weights=branch_weights, portfolio_value=branch_nav
            )
            if index == intervention_index:
                branch_intervention = intervene_from_factual(
                    intervention_pipeline,
                    branch_snapshot,
                    factual_result,
                    stage_id,
                    operator="repair",
                )
                branch_result = branch_intervention.intervened
                interventions.append(branch_intervention)
            else:
                branch_result = intervention_pipeline.run_episode(branch_snapshot)
            branch_weights = _final_weights(branch_result)
        # Forward returns are simulator-only outcomes and never enter agent prompts.
        factual_nav *= 1.0 + _realized_period_return(snapshot, factual_weights)
        branch_nav *= 1.0 + _realized_period_return(snapshot, branch_weights)
        factual_navs.append(factual_nav)
        branch_navs.append(branch_nav)
        factual_history.append(dict(factual_weights))
        branch_history.append(dict(branch_weights))
    return ClosedLoopBranchResult(
        branch_id=branch_id,
        factual_nav=factual_navs,
        intervention_nav=branch_navs,
        factual_weights=factual_history,
        intervention_weights=branch_history,
        episode_interventions=interventions,
    )


def compute_descriptive_delta(
    factual: Mapping[str, Any], intervened: Mapping[str, Any], keys: Optional[list[str]] = None
) -> Dict[str, Any]:
    """Compute labeled numeric differences without claiming identification."""
    use_keys = keys or sorted(set(factual) | set(intervened))
    delta: Dict[str, Any] = {}
    for key in use_keys:
        factual_value = factual.get(key)
        intervention_value = intervened.get(key)
        if isinstance(factual_value, (int, float)) and isinstance(intervention_value, (int, float)):
            delta[key] = float(intervention_value) - float(factual_value)
        else:
            delta[key] = {"factual": factual_value, "intervened": intervention_value}
    return delta


def store_intervention_result(
    adapter: ReplayAdapter,
    cache_key: str,
    output: Any,
    provenance: ResultProvenance,
    *,
    branch_id: str = "default",
    schema_version: str = "pipeline-v3-collab",
) -> Any:
    """Store one branch result in an isolated intervention namespace."""
    return adapter.put(
        cache_key,
        output,
        provenance,
        namespace=intervention_namespace(branch_id),
        schema_version=schema_version,
    )


__all__ = [
    "StageInterventionResult",
    "ClosedLoopBranchResult",
    "apply_repair",
    "apply_perturb",
    "intervene_from_factual",
    "run_stage_intervention",
    "run_closed_loop_intervention",
    "compute_descriptive_delta",
    "store_intervention_result",
]
