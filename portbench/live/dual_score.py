"""Score one episode's model outputs under lookback and ex-post oracles."""

from __future__ import annotations

from typing import Any, Optional

from ..agent_eval.base import EpisodeResult, MarketSnapshot, StageID
from ..agent_eval.stages import (
    S1MarketInterpretation,
    S2SignalGeneration,
    S3WeightOptimization,
    S4ExecutionSimulation,
    S5RiskMonitoring,
)
from ..metrics.ceps import CEPS


def score_with_oracle(
    snapshot: MarketSnapshot,
    stage_outputs: dict[StageID, Any],
    *,
    oracle_mode: str,
    profile=None,
    propagation_weight: float = 0.1,
) -> dict:
    """
    Re-score fixed model outputs against GT for the given S3 oracle_mode.

    S4/S5 ground truth is chained from the oracle-aware S3 GT (not the default
    ex_post-only helpers inside S4/S5.compute_ground_truth).
    """
    s1 = S1MarketInterpretation()
    s2 = S2SignalGeneration()
    s3 = S3WeightOptimization(oracle_mode=oracle_mode)
    s3._last_snapshot = snapshot  # needed for correlation-awareness in S3.score
    s4 = S4ExecutionSimulation()
    s5 = S5RiskMonitoring(profile=profile)

    gt = {
        StageID.S1_MARKET_INTERPRETATION: s1.compute_ground_truth(snapshot),
        StageID.S2_SIGNAL_GENERATION: s2.compute_ground_truth(snapshot),
        StageID.S3_WEIGHT_OPTIMIZATION: s3.compute_ground_truth(snapshot),
    }
    gt[StageID.S4_EXECUTION_SIMULATION] = s4._execute(
        gt[StageID.S3_WEIGHT_OPTIMIZATION], snapshot, slippage_rate=0.0
    )
    gt[StageID.S5_RISK_MONITORING] = s5._monitor(
        gt[StageID.S4_EXECUTION_SIMULATION].executed_weights, snapshot
    )

    stages = {
        StageID.S1_MARKET_INTERPRETATION: s1,
        StageID.S2_SIGNAL_GENERATION: s2,
        StageID.S3_WEIGHT_OPTIMIZATION: s3,
        StageID.S4_EXECUTION_SIMULATION: s4,
        StageID.S5_RISK_MONITORING: s5,
    }

    scores: dict[str, float] = {}
    for sid, stage in stages.items():
        actual = stage_outputs.get(sid)
        if actual is None:
            scores[sid.value] = 0.0
            continue
        try:
            scores[sid.value] = float(stage.score(actual, gt[sid]))
        except Exception:
            scores[sid.value] = 0.0

    # Build a temporary EpisodeResult for CEPS
    ep = EpisodeResult(decision_date=snapshot.decision_date)
    ep.stage_outputs = dict(stage_outputs)
    ep.gt_outputs = gt
    ep.stage_scores = {sid: scores[sid.value] for sid in stages}
    ceps = CEPS(propagation_weight).compute(ep.to_stage_score_list())

    return {
        "oracle_mode": oracle_mode,
        "stage_scores": scores,
        "ceps": float(ceps.ceps_score),
        "isolated_avg": float(ceps.isolated_avg),
        "propagation_penalty": float(ceps.propagation_penalty),
        "gt_s3_weights": dict(gt[StageID.S3_WEIGHT_OPTIMIZATION].weights),
    }


def dual_score(
    snapshot: MarketSnapshot,
    stage_outputs: dict[StageID, Any],
    *,
    profile=None,
    propagation_weight: float = 0.1,
    oracle_modes: Optional[list[str]] = None,
) -> dict[str, dict]:
    modes = oracle_modes or ["lookback", "ex_post"]
    return {
        mode: score_with_oracle(
            snapshot,
            stage_outputs,
            oracle_mode=mode,
            profile=profile,
            propagation_weight=propagation_weight,
        )
        for mode in modes
    }
