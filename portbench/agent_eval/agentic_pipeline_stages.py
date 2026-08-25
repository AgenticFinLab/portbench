"""PipelineStage adapters for the agent-planned S4 and S5 implementations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from portbench.agent_eval.base import (
    MarketSnapshot,
    PipelineStage,
    RiskAlert,
    S3Output,
    S4Output,
    S5Output,
    StageID,
    TradeOrder,
)
from portbench.agent_eval.contracts import S4S5_SCHEMA_AGENTIC
from portbench.agent_eval.s4_s5_stages import AgenticS4Stage, AgenticS5Stage
from portbench.agent_eval.stages import S3WeightOptimization, S4ExecutionSimulation, S5RiskMonitoring
from portbench.metrics.plan_outcome_scores import ceps_plan_score


class AgenticS4PipelineStage(PipelineStage):
    """Expose agentic execution planning through the standard pipeline API."""

    def __init__(
        self,
        adapter: Any,
        oracle_mode: str = "lookback",
        use_tools: bool = False,
        schema_version: str = S4S5_SCHEMA_AGENTIC,
    ) -> None:
        self.adapter = adapter
        self.oracle_mode = oracle_mode
        self.schema_version = schema_version
        self._stage = AgenticS4Stage(adapter, use_tools=use_tools, schema_version=schema_version)

    @property
    def stage_id(self) -> StageID:
        return StageID.S4_EXECUTION_SIMULATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S4Output:
        s3_ground_truth = S3WeightOptimization(oracle_mode=self.oracle_mode).compute_ground_truth(snapshot)
        return S4ExecutionSimulation()._execute(s3_ground_truth, snapshot, slippage_rate=0.0)

    def run(self, snapshot: MarketSnapshot, prior_output: Optional[S3Output] = None) -> S4Output:
        if prior_output is None or not prior_output.weights:
            raise ValueError("agentic S4 requires S3 target weights")
        bundle = self._stage.run(
            snapshot_like=snapshot,
            current_weights=snapshot.current_weights,
            target_weights=prior_output.weights,
            nav=snapshot.portfolio_value,
        )
        self._last_prompt = self._stage.last_prompt
        orders = [
            TradeOrder(
                asset=str(item["asset"]),
                direction=str(item["direction"]),
                quantity=float(item.get("quantity", 0.0)),
                price=float(item.get("price", 0.0)),
                slippage=float(item.get("slippage", 0.0)),
                commission=float(item.get("commission", 0.0)),
            )
            for item in bundle.result.metadata.get("orders", [])
            if item.get("status") == "filled"
        ]
        return S4Output(
            orders=orders,
            executed_weights=dict(bundle.result.filled_weights),
            total_cost=float(bundle.result.cost),
            turnover=float(bundle.result.turnover),
            raw_llm_output=bundle.raw_response,
            refused=bool((bundle.plan.metadata or {}).get("parse_error")),
            schema_version=self.schema_version,
            plan=asdict(bundle.plan),
            plan_scores=dict(bundle.plan_scores),
            outcome_scores=dict(bundle.outcome_scores),
        )

    def score(self, actual: S4Output, ground_truth: S4Output) -> float:
        return ceps_plan_score(actual.plan_scores, stage="S4")


class AgenticS5PipelineStage(PipelineStage):
    """Expose agentic risk control through the standard pipeline API."""

    def __init__(
        self,
        adapter: Any,
        profile: Any = None,
        use_tools: bool = False,
        schema_version: str = S4S5_SCHEMA_AGENTIC,
    ) -> None:
        self.adapter = adapter
        self.schema_version = schema_version
        self._stage = AgenticS5Stage(adapter, use_tools=use_tools, schema_version=schema_version)
        self._var_limit = profile.var_limit if profile is not None else S5RiskMonitoring.VAR_LIMIT
        self._drawdown_limit = (
            -profile.max_drawdown_tolerance
            if profile is not None
            else S5RiskMonitoring.DRAWDOWN_LIMIT
        )
        self._drift_limit = S5RiskMonitoring.DRIFT_LIMIT

    @property
    def stage_id(self) -> StageID:
        return StageID.S5_RISK_MONITORING

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S5Output:
        return S5RiskMonitoring(profile=None).compute_ground_truth(snapshot)

    def run(self, snapshot: MarketSnapshot, prior_output: Optional[S4Output] = None) -> S5Output:
        if prior_output is None or not prior_output.executed_weights:
            raise ValueError("agentic S5 requires S4 executed weights")
        bundle = self._stage.run(
            weights=prior_output.executed_weights,
            return_data=snapshot.return_data,
            snapshot_like=snapshot,
            var_limit=self._var_limit,
            drawdown_limit=self._drawdown_limit,
            drift_limit=self._drift_limit,
        )
        self._last_prompt = self._stage.last_prompt
        metrics = bundle.eval_result
        alerts = []
        for violation in metrics.constraint_violations:
            if violation == "var_breach":
                alerts.append(RiskAlert(violation, float(metrics.var or 0.0), self._var_limit, "warning", "reduce"))
            elif violation == "drawdown":
                alerts.append(RiskAlert(violation, float(metrics.drawdown or 0.0), self._drawdown_limit, "critical", "rebalance"))
            else:
                drift = float(metrics.metadata.get("weight_drift", 0.0))
                alerts.append(RiskAlert(violation, drift, self._drift_limit, "warning", "rebalance"))
        return S5Output(
            portfolio_var=float(metrics.var or 0.0),
            portfolio_drawdown=float(metrics.drawdown or 0.0),
            weight_drift=float(metrics.metadata.get("weight_drift", 0.0)),
            alerts=alerts,
            rebalance_needed=bundle.decision.action != "hold",
            raw_llm_output=bundle.raw_response,
            final_weights=dict(metrics.metadata.get("final_weights", {})),
            refused=bool((bundle.decision.metadata or {}).get("parse_error")),
            schema_version=self.schema_version,
            decision=asdict(bundle.decision),
            plan_scores=dict(bundle.plan_scores),
            outcome_scores=dict(bundle.outcome_scores),
        )

    def score(self, actual: S5Output, ground_truth: S5Output) -> float:
        return ceps_plan_score(actual.plan_scores, stage="S5")


__all__ = ["AgenticS4PipelineStage", "AgenticS5PipelineStage"]
