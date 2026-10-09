"""Evaluate the published classical baselines on the saved-allocation replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .replay import MarketData, digest, normalize, read_json, replay, write_json


def run_classical_baselines(source_root: Path, output_dir: Path) -> None:
    """Score 60/40, risk parity, covariance risk parity, minimum variance, and Black--Litterman.

    Equal weight is already written by ``run_saved_allocations``. The 60/40
    sleeve uses eligible equities and bonds. The other four strategies call
    the library baselines on the same eligible universe and execution contract.
    """
    root = Path(source_root)
    output = Path(output_dir)
    from portbench.agent_eval.base import MarketSnapshot
    from portbench.baselines import (
        BlackLittermanBaseline,
        CovarianceRiskParityBaseline,
        MinVarianceBaseline,
        RiskParityBaseline,
    )

    inputs = {}
    support_file = root / "EXPERIMENTS_rebuttal_lookback/monthly/baseline/equal_weight/20260709_083230/balanced/normal_bull_2024/weight_history.csv"
    inputs[str(support_file.relative_to(root))] = digest(support_file)
    assets = list(pd.read_csv(support_file, nrows=0).columns[1:])
    market = MarketData(root, assets, inputs)
    baseline_types = {
        "risk_parity": RiskParityBaseline,
        "cov_risk_parity": CovarianceRiskParityBaseline,
        "min_variance": MinVarianceBaseline,
        "black_litterman": BlackLittermanBaseline,
    }
    for path in (root / "portbench/baselines").glob("*.py"):
        inputs[str(path.relative_to(root))] = digest(path)
    summary = read_json(output / "summary.json")
    coverage = pd.read_csv(output / "coverage.csv")
    results = []
    allocations = []
    for scenario, window in summary["windows"].items():
        dates = sorted(coverage.loc[coverage.scenario.eq(scenario), "decision_date"].unique())
        policies = {name: {} for name in ["sixty_forty", *baseline_types]}
        for date in dates:
            source = market.snapshot(date)
            snapshot = MarketSnapshot(
                decision_date=pd.Timestamp(date).date(),
                price_data={},
                return_data={a: source["returns"][a] for a in source["assets"]},
                correlation_matrix=source["correlation"],
                asset_class_map={a: market.classes[a] for a in source["assets"]},
            )
            equities = [a for a in source["assets"] if market.classes[a] == "equities"]
            bonds = [a for a in source["assets"] if market.classes[a] == "bonds"]
            assert equities and bonds
            policies["sixty_forty"][date] = {
                **{a: .6 / len(equities) for a in equities},
                **{a: .4 / len(bonds) for a in bonds},
            }
            for name, strategy in baseline_types.items():
                weights = normalize(strategy().allocate(snapshot))
                assert set(weights) <= set(source["assets"])
                policies[name][date] = weights
        for name, targets in policies.items():
            metrics, _ = replay(market, targets, window["start"], window["end"])
            results.append(dict(model=name, scenario=scenario, **metrics))
            allocations.extend(
                dict(model=name, scenario=scenario, decision_date=d, asset=a, weight=w)
                for d, target in targets.items()
                for a, w in target.items()
            )
        print(f"Computed five classical baselines for {scenario}.", flush=True)
    pd.DataFrame(results).to_csv(output / "classical_baselines.csv", index=False, float_format="%.12g")
    pd.DataFrame(allocations).to_csv(output / "classical_allocations.csv", index=False, float_format="%.12g")
    write_json(
        output / "classical_manifest.json",
        dict(
            script_sha256=digest(Path(__file__)),
            inputs=inputs,
            shared_replay_script_sha256=digest(Path(__file__).with_name("replay.py")),
            outputs={name: digest(output / name) for name in ["classical_baselines.csv", "classical_allocations.csv"]},
        ),
    )
