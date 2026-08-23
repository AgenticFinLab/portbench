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


def episode_from_log(
    episode: Mapping[str, Any],
    *,
    stage_scores: Optional[Mapping[str, float]] = None,
    gt_outputs: Optional[Mapping[StageID, Any]] = None,
) -> EpisodeResult:
    """Rebuild a typed EpisodeResult from one saved episode JSON object."""
    from datetime import date as date_cls

    outputs: Dict[StageID, Any] = {}
    gts: Dict[StageID, Any] = dict(gt_outputs or {})
    scores: Dict[StageID, float] = {}
    for record in episode.get("stages") or []:
        sid_text = str(record.get("stage_id", ""))
        if sid_text not in STAGE_BY_TEXT:
            continue
        sid = STAGE_BY_TEXT[sid_text]
        parsed = record.get("parsed_output") or {}
        if parsed:
            try:
                outputs[sid] = _coerce(sid, parsed)
            except Exception:
                pass
        gt_payload = record.get("ground_truth") or {}
        if sid not in gts and gt_payload:
            try:
                gts[sid] = _coerce(sid, gt_payload)
            except Exception:
                pass
        if stage_scores and sid_text in stage_scores:
            scores[sid] = float(stage_scores[sid_text])
        elif "score" in record:
            scores[sid] = float(record.get("score") or 0.0)
    decision_raw = episode.get("decision_date")
    if hasattr(decision_raw, "isoformat"):
        decision_date = decision_raw
    else:
        decision_date = date_cls.fromisoformat(str(decision_raw))
    return EpisodeResult(
        decision_date=decision_date,
        stage_outputs=outputs,
        gt_outputs=gts,
        stage_scores=scores,
        schema_version=str(episode.get("schema_version") or "pipeline-v1"),
        architecture_id=str(episode.get("architecture_id") or "legacy"),
        result_protocol=str(episode.get("result_protocol") or "closed-loop"),
        ceps_score=float(episode.get("ceps_score") or 0.0),
    )


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


PERTURB_SCALE = 1.1


def _stage_key(stage_id: str | StageID) -> str:
    if isinstance(stage_id, StageID):
        return stage_id.value
    text = str(stage_id)
    return text.split("_", 1)[0] if text.startswith("S") else text


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _amplify_weights(weights: Mapping[str, Any], scale: float = PERTURB_SCALE) -> Dict[str, float]:
    """Boost the largest long-only weight and renormalize."""
    parsed = {str(asset): max(0.0, float(weight)) for asset, weight in dict(weights).items()}
    if not parsed:
        return {}
    top = max(parsed, key=parsed.get)
    parsed[top] = parsed[top] * float(scale)
    total = sum(parsed.values())
    if total <= 0.0:
        equal = 1.0 / len(parsed)
        return {asset: equal for asset in parsed}
    return {asset: weight / total for asset, weight in parsed.items()}


def default_perturb(stage_id: str | StageID, factual_output: Mapping[str, Any]) -> Dict[str, Any]:
    """Amplify the model's own stance by 10% and project back onto stage constraints."""
    out = dict(factual_output)
    key = _stage_key(stage_id)
    if key == "S1":
        views = dict(out.get("asset_views") or {})
        out["asset_views"] = {
            str(asset): _clip(float(view) * PERTURB_SCALE, -1.0, 1.0)
            for asset, view in views.items()
        }
        return out
    if key == "S2":
        strengths = dict(out.get("strengths") or {})
        out["strengths"] = {
            str(asset): _clip(float(strength) * PERTURB_SCALE, 0.0, 1.0)
            for asset, strength in strengths.items()
        }
        return out
    if key == "S3":
        out["weights"] = _amplify_weights(out.get("weights") or {})
        return out
    if key == "S4":
        executed = dict(out.get("executed_weights") or {})
        if executed:
            out["executed_weights"] = _amplify_weights(executed)
        plan = dict(out.get("plan") or {})
        orders = list(plan.get("orders") or [])
        if orders:
            targets = {
                str(order.get("asset")): float(order.get("target_weight") or 0.0)
                for order in orders
                if order.get("asset")
            }
            amplified = _amplify_weights(targets) if targets else {}
            new_orders = []
            for order in orders:
                item = dict(order)
                asset = str(item.get("asset", ""))
                if asset in amplified:
                    item["target_weight"] = amplified[asset]
                new_orders.append(item)
            plan["orders"] = new_orders
            out["plan"] = plan
        return out
    if key == "S5":
        decision = dict(out.get("decision") or {})
        action = str(decision.get("action") or "hold").lower()
        if action == "hold":
            decision["action"] = "scale_down"
            decision["scale_factor"] = 0.9
        elif action == "scale_down":
            current = decision.get("scale_factor")
            factor = 0.9 if current is None else _clip(float(current) * 0.9, 0.0, 1.0)
            decision["scale_factor"] = factor
        else:
            weights = decision.get("corrective_weights")
            if isinstance(weights, Mapping) and weights:
                decision["corrective_weights"] = _amplify_weights(weights)
        out["decision"] = decision
        return out
    raise ValueError(f"unsupported stage_id={stage_id!r}")


def _lookback_returns(snapshot: MarketSnapshot) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for asset, series in (snapshot.return_data or {}).items():
        values = series.dropna() if hasattr(series, "dropna") else series
        out[str(asset)] = [float(item) for item in list(values)]
    return out


def _ceps_from_scores(scores: Mapping[StageID, float], propagation_weight: float = 0.1) -> float:
    from portbench.metrics.ceps import CEPS, StageScore

    ordered = [
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
        StageID.S3_WEIGHT_OPTIMIZATION,
        StageID.S4_EXECUTION_SIMULATION,
        StageID.S5_RISK_MONITORING,
    ]
    payload = [
        StageScore(stage_id=sid.value, stage_name=sid.name, score=float(scores.get(sid, 0.0)))
        for sid in ordered
    ]
    return float(CEPS(propagation_weight).compute(payload).ceps_score)


def _legacy_scorer(stage_id: StageID):
    from portbench.agent_eval.stages import (
        S1MarketInterpretation,
        S2SignalGeneration,
        S3WeightOptimization,
        S4ExecutionSimulation,
        S5RiskMonitoring,
    )

    scorers = {
        StageID.S1_MARKET_INTERPRETATION: S1MarketInterpretation,
        StageID.S2_SIGNAL_GENERATION: S2SignalGeneration,
        StageID.S3_WEIGHT_OPTIMIZATION: S3WeightOptimization,
        StageID.S4_EXECUTION_SIMULATION: S4ExecutionSimulation,
        StageID.S5_RISK_MONITORING: S5RiskMonitoring,
    }
    return scorers[stage_id]()


def _score_outputs(
    pipeline: Optional[EvalPipeline],
    outputs: Mapping[StageID, Any],
    ground_truth: Mapping[StageID, Any],
) -> Dict[StageID, float]:
    from portbench.metrics.plan_outcome_scores import ceps_plan_score

    scores: Dict[StageID, float] = {}
    for sid, actual in outputs.items():
        gt = ground_truth.get(sid)
        plan_scores = getattr(actual, "plan_scores", None) or {}
        if plan_scores:
            scores[sid] = ceps_plan_score(plan_scores, stage=sid.value)
            continue
        try:
            scorer = _legacy_scorer(sid)
            scores[sid] = float(scorer.score(actual, gt)) if gt is not None else 0.0
        except Exception:
            stage = pipeline.stages.get(sid) if pipeline is not None else None
            if stage is None or gt is None:
                scores[sid] = 0.0
            else:
                try:
                    scores[sid] = float(stage.score(actual, gt))
                except Exception:
                    scores[sid] = 0.0
    return scores


def _fill_deterministic_suffix(
    snapshot: MarketSnapshot,
    outputs: Dict[StageID, Any],
    from_index: int,
) -> Dict[StageID, Any]:
    """Fill downstream stages with Point-in-Time repair / deterministic S4/S5."""
    from portbench.agent_eval.stages import S4ExecutionSimulation, S5RiskMonitoring

    lookback = _lookback_returns(snapshot)
    filled = dict(outputs)
    for index, stage in enumerate(STAGE_ORDER):
        if index <= from_index:
            continue
        if stage is StageID.S1_MARKET_INTERPRETATION:
            filled[stage] = _coerce(stage, pit_repair.repair_s1(lookback_returns=lookback))
        elif stage is StageID.S2_SIGNAL_GENERATION:
            s1_payload = _plain(filled[StageID.S1_MARKET_INTERPRETATION])
            filled[stage] = _coerce(
                stage,
                pit_repair.repair_s2(lookback_returns=lookback, s1_output=s1_payload),
            )
        elif stage is StageID.S3_WEIGHT_OPTIMIZATION:
            filled[stage] = _coerce(stage, pit_repair.repair_s3(lookback_returns=lookback))
        elif stage is StageID.S4_EXECUTION_SIMULATION:
            s3_output = filled[StageID.S3_WEIGHT_OPTIMIZATION]
            filled[stage] = S4ExecutionSimulation()._execute(s3_output, snapshot)
        else:
            s4_output = filled[StageID.S4_EXECUTION_SIMULATION]
            monitored = S5RiskMonitoring()._monitor(s4_output.executed_weights, snapshot)
            monitored.final_weights = dict(s4_output.executed_weights)
            filled[stage] = monitored
    return filled


def _is_agentic_factual(factual: EpisodeResult) -> bool:
    if factual.schema_version == "pipeline-v3-collab":
        return True
    for output in factual.stage_outputs.values():
        if getattr(output, "plan_scores", None):
            return True
    return False


def _annotate_suffix_plan_scores(
    snapshot: MarketSnapshot,
    filled: Dict[StageID, Any],
    factual: EpisodeResult,
) -> Dict[StageID, Any]:
    """Attach plan/outcome scores so deterministic S4/S5 stay in the CEPS plan family."""
    if not _is_agentic_factual(factual):
        return filled
    from portbench.agent_eval.s4_s5_bridge import (
        legacy_s4_to_plan_and_result,
        run_s4_deterministic_from_weights,
    )
    from portbench.agent_eval.s4_s5_stages import _reference_risk_decision
    from portbench.metrics.plan_outcome_scores import (
        score_s4_environment_outcome,
        score_s4_plan_quality,
        score_s5_environment_outcome,
        score_s5_plan_quality,
    )
    from portbench.sandbox.risk_control import evaluate_risk, summarize_pre_action_risk

    s3 = filled.get(StageID.S3_WEIGHT_OPTIMIZATION)
    s4 = filled.get(StageID.S4_EXECUTION_SIMULATION)
    if s3 is not None and s4 is not None and getattr(s3, "weights", None):
        nav = float(snapshot.portfolio_value or 0.0)
        ref_plan, ref_result = run_s4_deterministic_from_weights(
            s3.weights,
            snapshot,
            snapshot.current_weights,
            nav,
        )
        plan, result = legacy_s4_to_plan_and_result(s4, snapshot)
        s4.plan = asdict(plan)
        s4.plan_scores = score_s4_plan_quality(plan, ref_plan, target_weights=s3.weights)
        s4.outcome_scores = score_s4_environment_outcome(result, ref_result)
        s4.schema_version = "pipeline-v3-collab"
    s5 = filled.get(StageID.S5_RISK_MONITORING)
    s4_for_s5 = filled.get(StageID.S4_EXECUTION_SIMULATION)
    weights = {}
    if s4_for_s5 is not None and getattr(s4_for_s5, "executed_weights", None):
        weights = dict(s4_for_s5.executed_weights)
    elif s3 is not None:
        weights = dict(s3.weights)
    if s5 is not None and weights:
        pre_risk = summarize_pre_action_risk(weights, snapshot.return_data)
        ref_decision = _reference_risk_decision(weights, pre_risk, -0.02, -0.10, 0.05)
        from portbench.agent_eval.contracts import RiskControlDecision

        existing = dict(getattr(s5, "decision", None) or {})
        if existing.get("action"):
            corrective = existing.get("corrective_weights")
            decision = RiskControlDecision(
                action=str(existing.get("action") or "hold"),
                alerts=list(existing.get("alerts") or []),
                corrective_weights=(
                    {str(k): float(v) for k, v in dict(corrective).items()}
                    if isinstance(corrective, dict)
                    else None
                ),
                scale_factor=existing.get("scale_factor"),
            )
        else:
            # Deterministic suffix has no agent plan; its PiT action is the reference.
            decision = ref_decision
        eval_result = evaluate_risk(decision, weights, snapshot.return_data)
        ref_result = evaluate_risk(ref_decision, weights, snapshot.return_data)
        s5.decision = asdict(decision)
        s5.plan_scores = score_s5_plan_quality(decision, ref_decision)
        s5.outcome_scores = score_s5_environment_outcome(eval_result, ref_result)
        s5.schema_version = "pipeline-v3-collab"
    return filled


def _replacement_payload(
    snapshot: MarketSnapshot,
    factual: EpisodeResult,
    stage: StageID,
    operator: str,
) -> Any:
    if operator == "repair":
        return _coerce(stage, _repair_payload(stage, snapshot, factual))
    if operator == "perturb":
        payload = default_perturb(stage.value, _plain(factual.stage_outputs[stage]))
        return _coerce(stage, payload)
    raise ValueError(f"operator must be repair or perturb, got {operator!r}")


def run_offline_stage_intervention(
    snapshot: MarketSnapshot,
    factual: EpisodeResult,
    stage_id: str | StageID,
    *,
    operator: str = "repair",
    pipeline: Optional[EvalPipeline] = None,
    propagation_weight: float = 0.1,
) -> Dict[str, Any]:
    """Replace one factual stage and continue the suffix without provider calls."""
    stage = _stage(stage_id)
    if stage not in factual.stage_outputs:
        raise KeyError(f"factual result is missing {stage.value}")
    stage_index = STAGE_ORDER.index(stage)
    outputs = {
        upstream: factual.stage_outputs[upstream]
        for upstream in STAGE_ORDER[:stage_index]
        if upstream in factual.stage_outputs
    }
    outputs[stage] = _replacement_payload(snapshot, factual, stage, operator)
    outputs = _fill_deterministic_suffix(snapshot, outputs, stage_index)
    outputs = _annotate_suffix_plan_scores(snapshot, outputs, factual)
    scores = _score_outputs(pipeline, outputs, factual.gt_outputs)
    intervened_ceps = _ceps_from_scores(scores, propagation_weight)
    factual_ceps = _ceps_from_scores(factual.stage_scores, propagation_weight)
    score_delta = {
        sid.value: float(scores.get(sid, 0.0)) - float(factual.stage_scores.get(sid, 0.0))
        for sid in STAGE_ORDER[stage_index:]
        if sid in scores or sid in factual.stage_scores
    }
    return {
        "stage_id": stage.value,
        "operator": operator,
        "mode": "offline",
        "score_delta": score_delta,
        "ceps_factual": factual_ceps,
        "ceps_intervened": intervened_ceps,
        "ceps_delta": intervened_ceps - factual_ceps,
    }


def run_episode_interventions(
    pipeline: EvalPipeline,
    snapshot: MarketSnapshot,
    factual: EpisodeResult,
    *,
    stages: Sequence[str],
    operator: str = "repair",
    mode: str = "offline",
    propagation_weight: float = 0.1,
) -> list[dict[str, Any]]:
    """Run repair/perturb interventions for one factual episode."""
    records: list[dict[str, Any]] = []
    for stage_id in stages:
        if mode == "offline":
            records.append(
                run_offline_stage_intervention(
                    snapshot,
                    factual,
                    stage_id,
                    operator=operator,
                    pipeline=pipeline,
                    propagation_weight=propagation_weight,
                )
            )
            continue
        if mode != "online":
            raise ValueError(f"mode must be offline or online, got {mode!r}")
        perturb_fn = (
            (lambda output, sid=stage_id: default_perturb(sid, output))
            if operator == "perturb"
            else None
        )
        result = intervene_from_factual(
            pipeline,
            snapshot,
            factual,
            stage_id,
            operator=operator,
            perturb_fn=perturb_fn,
        )
        factual_ceps = _ceps_from_scores(factual.stage_scores, propagation_weight)
        intervened_ceps = _ceps_from_scores(result.intervened.stage_scores, propagation_weight)
        records.append(
            {
                "stage_id": result.stage_id,
                "operator": operator,
                "mode": "online",
                "score_delta": {
                    key: float(value) for key, value in result.score_delta.items()
                },
                "ceps_factual": factual_ceps,
                "ceps_intervened": intervened_ceps,
                "ceps_delta": intervened_ceps - factual_ceps,
            }
        )
    return records


def run_offline_closed_loop(
    records: Sequence[tuple[MarketSnapshot, EpisodeResult]],
    *,
    stage_id: str | StageID,
    operator: str = "repair",
    intervention_index: int = 0,
    pipeline: Optional[EvalPipeline] = None,
    propagation_weight: float = 0.1,
) -> Dict[str, Any]:
    """Fork NAV from one rebalance using a deterministic suffix (no LLM)."""
    if not records:
        raise ValueError("closed-loop intervention requires rebalance records")
    if not 0 <= intervention_index < len(records):
        raise IndexError("intervention_index is outside the snapshot sequence")
    stage = _stage(stage_id)
    factual_weights = dict(records[0][0].current_weights)
    branch_weights = dict(factual_weights)
    factual_nav = float(records[0][0].portfolio_value)
    branch_nav = factual_nav
    factual_navs = [factual_nav]
    branch_navs = [branch_nav]
    episode_deltas: list[dict[str, Any]] = []
    for index, (snapshot, factual) in enumerate(records):
        factual_snapshot = dataclasses.replace(
            snapshot, current_weights=factual_weights, portfolio_value=factual_nav
        )
        factual_result = factual
        factual_weights = _final_weights(factual_result)
        if index < intervention_index:
            branch_weights = dict(factual_weights)
            branch_nav = factual_nav
        elif index == intervention_index:
            branch_snapshot = dataclasses.replace(
                snapshot, current_weights=branch_weights, portfolio_value=branch_nav
            )
            offline = run_offline_stage_intervention(
                branch_snapshot,
                factual_result,
                stage,
                operator=operator,
                pipeline=pipeline,
                propagation_weight=propagation_weight,
            )
            episode_deltas.append(offline)
            suffix = _fill_deterministic_suffix(
                branch_snapshot,
                {
                    **{
                        upstream: factual_result.stage_outputs[upstream]
                        for upstream in STAGE_ORDER[: STAGE_ORDER.index(stage)]
                        if upstream in factual_result.stage_outputs
                    },
                    stage: _replacement_payload(
                        branch_snapshot, factual_result, stage, operator
                    ),
                },
                STAGE_ORDER.index(stage),
            )
            s5 = suffix.get(StageID.S5_RISK_MONITORING)
            s4 = suffix.get(StageID.S4_EXECUTION_SIMULATION)
            s3 = suffix.get(StageID.S3_WEIGHT_OPTIMIZATION)
            if s5 is not None and getattr(s5, "final_weights", None):
                branch_weights = dict(s5.final_weights)
            elif s4 is not None and s4.executed_weights:
                branch_weights = dict(s4.executed_weights)
            elif s3 is not None:
                branch_weights = dict(s3.weights)
        else:
            branch_snapshot = dataclasses.replace(
                snapshot, current_weights=branch_weights, portfolio_value=branch_nav
            )
            lookback = _lookback_returns(branch_snapshot)
            suffix = _fill_deterministic_suffix(
                branch_snapshot,
                {
                    StageID.S1_MARKET_INTERPRETATION: _coerce(
                        StageID.S1_MARKET_INTERPRETATION,
                        pit_repair.repair_s1(lookback_returns=lookback),
                    )
                },
                0,
            )
            s5 = suffix.get(StageID.S5_RISK_MONITORING)
            s4 = suffix.get(StageID.S4_EXECUTION_SIMULATION)
            s3 = suffix.get(StageID.S3_WEIGHT_OPTIMIZATION)
            if s5 is not None and getattr(s5, "final_weights", None):
                branch_weights = dict(s5.final_weights)
            elif s4 is not None and s4.executed_weights:
                branch_weights = dict(s4.executed_weights)
            elif s3 is not None:
                branch_weights = dict(s3.weights)
        factual_nav *= 1.0 + _realized_period_return(snapshot, factual_weights)
        branch_nav *= 1.0 + _realized_period_return(snapshot, branch_weights)
        factual_navs.append(factual_nav)
        branch_navs.append(branch_nav)
    return {
        "branch_id": f"offline-{stage.value}",
        "stage_id": stage.value,
        "operator": operator,
        "mode": "offline",
        "result_protocol": "closed-loop",
        "factual_nav": factual_navs,
        "intervention_nav": branch_navs,
        "episode_interventions": episode_deltas,
    }


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
    elif operator == "perturb":
        fn = perturb_fn or (lambda output: default_perturb(stage.value, output))
        payload = apply_perturb(stage.value, _plain(factual.stage_outputs[stage]), perturb_fn=fn)
    else:
        raise ValueError("operator must be repair or perturb")
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
        run_interventions=False,
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
    "default_perturb",
    "intervene_from_factual",
    "run_stage_intervention",
    "run_closed_loop_intervention",
    "run_offline_stage_intervention",
    "run_episode_interventions",
    "run_offline_closed_loop",
    "episode_from_log",
    "compute_descriptive_delta",
    "store_intervention_result",
]
