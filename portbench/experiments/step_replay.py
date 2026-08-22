"""Step-replay runner for S4/S5 from cached S1-S3 outputs.

Writes results under result_protocol=step-replay with provenance.
Refuses closed-loop metric fields on the output record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from portbench.agent_eval.contracts import (
    STEP_REPLAY,
    ExecutionPlan,
    OrderIntent,
    ProvenanceSource,
    ResultProvenance,
    RiskControlDecision,
)
from portbench.agent_eval.result_gates import validate_step_replay_record
from portbench.sandbox.execution import simulate_execution
from portbench.sandbox.risk_control import evaluate_risk


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _weights_from_cached(s3_like: Mapping[str, Any]) -> Dict[str, float]:
    """Extract target weights from a cached S3-like payload."""
    if "target_weights" in s3_like:
        return {str(k): float(v) for k, v in dict(s3_like["target_weights"]).items()}
    if "weights" in s3_like:
        return {str(k): float(v) for k, v in dict(s3_like["weights"]).items()}
    return {str(k): float(v) for k, v in dict(s3_like).items() if isinstance(v, (int, float))}


def run_step_replay(
    *,
    target_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    snapshot_like: Any,
    nav: float,
    return_data: Optional[Mapping[str, Any]] = None,
    cache_key: str = "",
    source: str = ProvenanceSource.LEGACY_REPLAY.value,
) -> Dict[str, Any]:
    """Run deterministic S4 execution and S5 risk on cached upstream outputs.

    Returns a step-replay record. Closed-loop metrics are never attached.
    """
    if not cache_key:
        raise ValueError("step-replay requires a non-empty source cache_key")
    orders = []
    for asset, tw in dict(target_weights).items():
        curr = float(current_weights.get(asset, 0.0))
        delta = float(tw) - curr
        if abs(delta) < 1e-6:
            direction = "hold"
        else:
            direction = "buy" if delta > 0 else "sell"
        orders.append(
            OrderIntent(
                asset=str(asset),
                direction=direction,
                target_weight=float(tw),
                delta_weight=delta,
            )
        )
    plan = ExecutionPlan(
        orders=orders,
        metadata={"target_weights": dict(target_weights), "source": "step_replay"},
    )
    exec_result = simulate_execution(plan, snapshot_like, current_weights, nav)

    decision = RiskControlDecision(action="hold")
    risk_result = evaluate_risk(
        decision,
        exec_result.filled_weights,
        return_data or {},
    )

    metrics = {
        "implementation_shortfall": exec_result.implementation_shortfall,
        "cost": exec_result.cost,
        "turnover": exec_result.turnover,
        "var": risk_result.var,
        "cvar": risk_result.cvar,
        "drawdown": risk_result.drawdown,
        "constraint_violations": list(risk_result.constraint_violations),
    }
    # Refuse closed-loop fields before persistence.
    validate_step_replay_record(metrics)

    provenance = ResultProvenance(
        source=source,
        cache_key=cache_key,
        schema_version="pipeline-v1-deterministic",
        request_count=0,
    )
    return {
        "result_protocol": STEP_REPLAY,
        "provenance": {
            "source": provenance.source,
            "cache_key": provenance.cache_key,
            "schema_version": provenance.schema_version,
            "request_count": provenance.request_count,
        },
        "filled_weights": dict(exec_result.filled_weights),
        "metrics": metrics,
        "plan_metadata": dict(plan.metadata or {}),
        "risk_metadata": dict(risk_result.metadata or {}),
    }


def run_step_replay_from_fixture(fixture_dir: str | Path, out_dir: str | Path) -> Path:
    """Dry-run step-replay using a mini fixture directory."""
    fixture = Path(fixture_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s3_path = fixture / "S3.json"
    weights_path = fixture / "current_weights.json"
    s3 = _load_json(s3_path) if s3_path.exists() else {"target_weights": {"A": 0.5, "B": 0.5}}
    current = _load_json(weights_path) if weights_path.exists() else {"A": 0.5, "B": 0.5}
    target = _weights_from_cached(s3)
    snapshot = {"price_data": {}, "portfolio_value": 100000.0}
    record = run_step_replay(
        target_weights=target,
        current_weights=current,
        snapshot_like=snapshot,
        nav=float(snapshot["portfolio_value"]),
        cache_key=f"fixture:{fixture.name}",
        source=ProvenanceSource.LEGACY_REPLAY.value,
    )
    out_path = out / "step_replay_result.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for step-replay dry-run."""
    p = argparse.ArgumentParser(description="Step-replay S4/S5 from cached upstream (no LLM)")
    p.add_argument("--fixture", type=str, required=True, help="Fixture episode directory")
    p.add_argument("--out", type=str, default="outputs/step_replay_dryrun", help="Output dir")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry: dry-run step-replay on a fixture."""
    args = build_parser().parse_args(argv)
    path = run_step_replay_from_fixture(args.fixture, args.out)
    print(json.dumps({"wrote": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
