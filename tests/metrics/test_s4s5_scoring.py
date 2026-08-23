"""Dual-layer scoring key separation and perfect-match tests."""

from __future__ import annotations

from portbench.agent_eval.contracts import (
    ExecutionPlan,
    ExecutionResult,
    OrderIntent,
    RiskControlDecision,
    RiskEvaluationResult,
)
from portbench.metrics.plan_outcome_scores import (
    S4_ENV_OUTCOME_KEYS,
    S4_PLAN_QUALITY_KEYS,
    S5_ENV_OUTCOME_KEYS,
    S5_PLAN_QUALITY_KEYS,
    score_s4_environment_outcome,
    score_s4_plan_quality,
    score_s5_environment_outcome,
    score_s5_plan_quality,
)


def test_plan_and_env_keys_disjoint():
    assert S4_PLAN_QUALITY_KEYS.isdisjoint(S4_ENV_OUTCOME_KEYS)
    assert S5_PLAN_QUALITY_KEYS.isdisjoint(S5_ENV_OUTCOME_KEYS)
    assert S4_PLAN_QUALITY_KEYS.isdisjoint(S5_ENV_OUTCOME_KEYS)
    assert S5_PLAN_QUALITY_KEYS.isdisjoint(S4_ENV_OUTCOME_KEYS)


def test_s4_perfect_match_scores_high():
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.6),
            OrderIntent(asset="TLT", direction="sell", target_weight=0.4),
        ],
        metadata={"target_weights": {"SPY": 0.6, "TLT": 0.4}},
    )
    pq = score_s4_plan_quality(plan, plan, target_weights={"SPY": 0.6, "TLT": 0.4})
    assert set(pq) == S4_PLAN_QUALITY_KEYS
    assert pq["order_legality"] >= 0.99
    assert pq["target_tracking"] >= 0.99
    assert pq["plan_quality"] >= 0.99

    rounded = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.3333),
            OrderIntent(asset="TLT", direction="hold", target_weight=0.3333),
            OrderIntent(asset="BIL", direction="sell", target_weight=0.3331),
        ]
    )
    rounded_pq = score_s4_plan_quality(
        rounded, rounded, target_weights={"SPY": 0.3333, "TLT": 0.3333, "BIL": 0.3331}
    )
    assert rounded_pq["order_legality"] == 1.0

    res = ExecutionResult(
        filled_weights={"SPY": 0.6, "TLT": 0.4},
        implementation_shortfall=0.001,
        turnover=0.2,
        cost=50.0,
    )
    oq = score_s4_environment_outcome(res, res)
    assert set(oq) == S4_ENV_OUTCOME_KEYS
    assert all(v >= 0.99 for v in oq.values())


def test_s5_perfect_match_scores_high():
    dec = RiskControlDecision(
        action="rebalance",
        alerts=["drawdown", "weight_drift"],
        corrective_weights={"SPY": 0.5, "BIL": 0.5},
    )
    pq = score_s5_plan_quality(dec, dec)
    assert set(pq) == S5_PLAN_QUALITY_KEYS
    assert all(v >= 0.99 for v in pq.values())

    ev = RiskEvaluationResult(
        var=-0.01,
        cvar=-0.015,
        drawdown=-0.05,
        constraint_violations=["drawdown"],
        metadata={"period_return": 0.01},
    )
    oq = score_s5_environment_outcome(ev, ev)
    assert set(oq) == S5_ENV_OUTCOME_KEYS
    assert all(v >= 0.99 for v in oq.values())


def test_s5_off_simplex_corrective_scores_zero_compliance():
    dec = RiskControlDecision(
        action="rebalance",
        corrective_weights={"SPY": 0.6, "BIL": 0.6},
    )
    ref = RiskControlDecision(
        action="rebalance",
        corrective_weights={"SPY": 0.5, "BIL": 0.5},
    )
    pq = score_s5_plan_quality(dec, ref)
    assert pq["corrective_compliance"] == 0.0


def test_s4_parse_error_zeros_all_plan_keys():
    plan = ExecutionPlan(orders=[], metadata={"parse_error": "invalid JSON"})
    pq = score_s4_plan_quality(
        plan,
        plan,
        target_weights={"SPY": 0.6, "BIL": 0.4},
    )
    assert pq["order_legality"] == 0.0
    assert pq["target_tracking"] == 0.0
    assert pq["plan_quality"] == 0.0


def test_s4_direction_mismatch_fails_legality():
    current = {"SPY": 0.5, "BIL": 0.5}
    bad = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.4),
            OrderIntent(asset="BIL", direction="sell", target_weight=0.6),
        ]
    )
    pq_bad = score_s4_plan_quality(
        bad,
        target_weights={"SPY": 0.4, "BIL": 0.6},
        current_weights=current,
    )
    assert pq_bad["order_legality"] == 0.0

    good = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="sell", target_weight=0.4),
            OrderIntent(asset="BIL", direction="buy", target_weight=0.6),
        ]
    )
    pq_good = score_s4_plan_quality(
        good,
        target_weights={"SPY": 0.4, "BIL": 0.6},
        current_weights=current,
    )
    assert pq_good["order_legality"] == 1.0

    mixed = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="hold", target_weight=0.7),
            OrderIntent(asset="BIL", direction="sell", target_weight=0.3),
        ]
    )
    pq_mixed = score_s4_plan_quality(
        mixed,
        target_weights={"SPY": 0.7, "BIL": 0.3},
        current_weights=current,
    )
    assert pq_mixed["order_legality"] == 0.5

    # Without current_weights the mismatch is not scored.
    pq_unchecked = score_s4_plan_quality(
        mixed, target_weights={"SPY": 0.7, "BIL": 0.3}
    )
    assert pq_unchecked["order_legality"] == 1.0
