"""Replay frozen decisions under the published PortBench evaluation contract.

This path scores archived S1--S3 outputs. It does not call models and does not
use the live sandbox. Ineligible or non-tradable weight, including ^VIX, stays
in cash. Execution is the next close after the decision, with 10 bps slippage
and 5 bps commission on security turnover.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr


PROFILES = {"conservative": (.40, .40, .10), "balanced": (.65, .20, .20), "aggressive": (.90, .05, .35)}
SEED = 20261006
CASH = "CASH"
MODEL_LABELS = {
    "deepseek-v4-flash": "DS-V4-Flash", "deepseek-v4-pro": "DS-V4-Pro",
    "qwen3.6-plus": "Qwen3.6-Plus", "qwen3.7-max": "Qwen3.7-Max",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "glm-5.1": "GLM-5.1",
    "doubao-seed-2-0-lite-260215": "DB-2.0-Lite", "doubao-seed-2-0-pro-260215": "DB-2.0-Pro",
    "hy3-preview": "HY3-Preview", "kimi-k2.6": "Kimi-K2.6",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))


def normalize(weights: dict) -> dict:
    assert weights and all(np.isfinite(x) and x >= 0 for x in weights.values())
    total = sum(weights.values())
    assert total > 0
    return {asset: float(value / total) for asset, value in weights.items()}


def project(weights: dict, eligible: list[str]) -> dict:
    weights = normalize(weights)
    result = {asset: weights.get(asset, 0.0) for asset in eligible}
    result[CASH] = sum(value for asset, value in weights.items() if asset not in result)
    assert abs(sum(result.values()) - 1) < 1e-10 and "^VIX" not in result
    return result


def exposure(weights: dict, classes: dict, profile: str) -> tuple[float, float, bool]:
    cap, floor, _ = PROFILES[profile]
    risk = sum(w for a, w in weights.items() if classes[a] in {"equities", "cryptocurrency"})
    safe = sum(w for a, w in weights.items() if classes[a] in {"bonds", "cash"})
    return risk, safe, risk <= cap + 1e-7 and safe >= floor - 1e-7


class MarketData:
    def __init__(self, root: Path, assets: list[str], inputs: dict):
        path = root / "datasets/processed/asset_class_map.json"
        inputs[str(path.relative_to(root))] = digest(path)
        self.classes = {k.removesuffix("_close"): v for k, v in read_json(path).items()}
        self.classes[CASH] = "cash"
        self.assets = sorted(set(assets) - {"^VIX"})
        self.prices, self.returns = {}, {}
        for cls in sorted({self.classes[a] for a in assets}):
            path = root / f"datasets/processed/{cls}.csv"
            inputs[str(path.relative_to(root))] = digest(path)
            wanted = {"date"} | {f"{a}_{suffix}" for a in assets for suffix in ["close", "return"]}
            frame = pd.read_csv(path, usecols=lambda c: c in wanted, index_col="date", parse_dates=True)
            assert frame.index.is_unique
            for asset in assets:
                if f"{asset}_close" in frame:
                    self.prices[asset] = frame[f"{asset}_close"].dropna()
                    self.returns[asset] = frame[f"{asset}_return"].dropna()
        # Outcome returns use the archived adjusted price sources, independently
        # of the processed statistics supplied to the decision stages.
        self.outcome_prices = {}
        stocks = []
        for asset in self.assets:
            path = root / f"datasets/yahoo/{self.classes[asset]}/{asset}.csv"
            if path.exists():
                inputs[str(path.relative_to(root))] = digest(path)
                frame = pd.read_csv(path, usecols=["date", "close"], index_col="date", parse_dates=True)
                self.outcome_prices[asset] = frame["close"].dropna()
            else:
                stocks.append(asset)
        if stocks:
            path = root / "datasets/kaggle/equities/nasdaq100-historical-data-2000-2026-upvote/NASDAQ100_Historical_Data.csv"
            inputs[str(path.relative_to(root))] = digest(path)
            frame = pd.read_csv(path, usecols=["Ticker", "Date", "Adj Close"], parse_dates=["Date"])
            for asset in stocks:
                selected = frame.loc[frame.Ticker.eq(asset)].set_index("Date")["Adj Close"].sort_index()
                assert not selected.empty and selected.index.is_unique, asset
                self.outcome_prices[asset] = selected.dropna()
        self.calendar = self.outcome_prices["SPY"].index.sort_values()
        close = pd.DataFrame(self.outcome_prices).reindex(self.calendar).ffill(limit=5)
        self.return_frame = close.pct_change(fill_method=None)
        self.return_frame[CASH] = 0.0
        self.cache = {}

    def snapshot(self, date: str) -> dict:
        if date in self.cache:
            return self.cache[date]
        stamp = pd.Timestamp(date)
        start = stamp - pd.Timedelta(days=90)
        series = {a: r.loc[(r.index > start) & (r.index <= stamp)].tail(60)
                  for a, r in self.returns.items() if a != "^VIX"}
        eligible = [a for a in self.assets if len(series[a]) >= 20
                    and not self.prices[a].loc[:stamp].empty
                    and self.prices[a].loc[:stamp].iloc[-1] > 0
                    and not self.outcome_prices[a].loc[:stamp].empty]
        frame = pd.DataFrame({a: series[a] for a in eligible})
        complete = frame.dropna()
        assert len(complete) >= 2, (date, len(complete))
        covariance = complete.cov()
        assert np.linalg.eigvalsh(covariance).min() > -1e-10
        views = {a: float(np.clip(((1 + series[a]).prod() - 1) / .10, -1, 1)) for a in eligible}
        signals = {a: "buy" if v > .2 else "sell" if v < -.2 else "hold" for a, v in views.items()}
        value = dict(date=date, assets=eligible, returns=complete, covariance=covariance,
                     mean=complete.mean(), views=views, signals=signals,
                     correlation=frame.corr(), pairwise_covariance=frame.cov())
        self.cache[date] = value
        return value


def optimize(snapshot: dict, signals: dict, classes: dict, profile: str) -> tuple[dict, dict]:
    """Optimize the selected signal support while always admitting defensive assets."""
    cap, floor, _ = PROFILES[profile]
    assets = [a for a in snapshot["assets"] if signals.get(a) == "buy" or classes[a] in {"bonds", "cash"}]
    defensive = np.array([classes[a] in {"bonds", "cash"} for a in assets], dtype=float)
    risk = np.array([classes[a] in {"equities", "cryptocurrency"} for a in assets], dtype=float)
    assert defensive.sum() > 0
    initial = defensive / defensive.sum()
    other = 1 - defensive
    if other.sum():
        initial = floor * initial + (1 - floor) * other / other.sum()
        excess = max(float(initial @ risk - cap), 0)
        if excess:
            initial[risk.astype(bool)] *= cap / (initial @ risk)
            initial += excess * defensive / defensive.sum()
    mu = snapshot["mean"].reindex(assets).to_numpy()
    covariance = snapshot["covariance"].loc[assets, assets].to_numpy()

    def objective(w):
        variance = w @ covariance @ w + 1e-10
        vol = np.sqrt(variance)
        ret = w @ mu
        return -ret / vol, -mu / vol + ret * (covariance @ w) / (variance * vol)

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))},
        {"type": "ineq", "fun": lambda w: cap - w @ risk, "jac": lambda w: -risk},
        {"type": "ineq", "fun": lambda w: w @ defensive - floor, "jac": lambda w: defensive},
    ]
    solution = minimize(objective, initial, jac=True, method="SLSQP", bounds=[(0, 1)] * len(assets),
                        constraints=constraints, options={"ftol": 1e-9, "maxiter": 500})
    w = solution.x
    feasible = np.isfinite(w).all() and w.min() >= -1e-7 and abs(w.sum() - 1) <= 1e-7
    feasible = feasible and w @ risk <= cap + 1e-7 and w @ defensive >= floor - 1e-7
    accepted = bool(solution.success and feasible and objective(w)[0] <= objective(initial)[0] + 1e-8)
    if not accepted:
        w = initial
    w = np.maximum(w, 0)
    w /= w.sum()
    result = {a: 0.0 for a in snapshot["assets"]}
    result.update(dict(zip(assets, map(float, w))))
    result[CASH] = 0.0
    assert exposure(result, classes, profile)[2]
    info = dict(accepted=accepted, status=int(solution.status), message=str(solution.message),
                n_selected=len(assets), objective=float(-objective(w)[0]))
    return result, info


def correlation_score(weights: dict, snapshot: dict, classes: dict) -> float:
    corr = snapshot["correlation"]
    class_names = sorted({classes[a] for a in snapshot["assets"]})
    totals = {c: sum(w for a, w in weights.items() if classes[a] == c) for c in class_names}
    penalty = numerator = denominator = 0.0
    for c in class_names:
        group = [a for a in corr if classes[a] == c]
        block = corr.loc[group, group].to_numpy()
        values = block[~np.eye(len(group), dtype=bool)]
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) else 0.0
        penalty += totals[c] * max(mean, 0)
        for other in class_names:
            if c == other:
                continue
            group2 = [a for a in corr if classes[a] == other]
            values = corr.loc[group, group2].to_numpy().ravel()
            values = values[np.isfinite(values)]
            if len(values):
                weight = totals[c] * totals[other]
                numerator += weight * values.mean()
                denominator += weight
    inter = (1 - numerator / denominator) / 2 if denominator > 1e-12 else .5
    return float(.5 * np.clip(1 - penalty, 0, 1) + .5 * np.clip(inter, 0, 1))


def turnover(current: dict, target: dict, include_cash: bool = False) -> float:
    return sum(abs(target.get(a, 0) - current.get(a, 0)) for a in set(current) | set(target)
               if include_cash or a != CASH)


def risk_state(weights: dict, snapshot: dict, profile: str) -> tuple[float, float, bool]:
    r = snapshot["returns"].to_numpy() @ np.array([weights.get(a, 0) for a in snapshot["assets"]])
    var, dd = float(np.quantile(r, .05)), drawdown(r)
    drift = max(abs(weights.get(a, 0) - 1 / len(snapshot["assets"])) for a in snapshot["assets"])
    return var, dd, dd < -PROFILES[profile][2] or drift > .05


def stage_scores(stages: dict, target: dict, reference: dict, current: dict,
                 snapshot: dict, classes: dict, profile: str) -> dict:
    views = stages["S1"]["parsed_output"]["asset_views"]
    signals = stages["S2"]["parsed_output"]["signals"]
    s1 = 1 - np.mean([abs(views.get(a, 0) - v) for a, v in snapshot["views"].items()]) / 2
    s2 = np.mean([signals.get(a) == v for a, v in snapshot["signals"].items()])
    similarity = 1 - turnover(target, reference, include_cash=True) / 2
    corr = correlation_score(target, snapshot, classes)
    s3 = .5 * (similarity + corr)
    actual_turnover, reference_turnover = turnover(current, target), turnover(current, reference)
    s4 = 1 - abs(actual_turnover - reference_turnover) / max(actual_turnover, reference_turnover, 1e-4)
    avar, add, areb = risk_state(target, snapshot, profile)
    rvar, rdd, rreb = risk_state(reference, snapshot, profile)
    error = (abs(avar - rvar) / max(abs(rvar), 1e-6) + abs(add - rdd) / max(abs(rdd), 1e-6)) / 2
    s5 = .5 * (areb == rreb) + .5 * np.clip(1 - error, 0, 1)
    values = np.array([s1, s2, s3, s4, s5])
    drops = np.maximum(values[:-1] - values[1:], 0).sum()
    return {**dict(zip(["S1", "S2", "S3", "S4", "S5"], map(float, values))),
            "ceps": float(np.clip(values.mean() - .1 * drops, 0, 1)), "stage_mean": float(values.mean()),
            "reference_similarity": float(similarity), "class_score": corr,
            "turnover": actual_turnover, "reference_turnover": reference_turnover,
            "lookback_var": avar, "lookback_drawdown": add}


def replay(market: MarketData, targets: dict, start: str, end: str,
           callback=None) -> tuple[dict, list[dict]]:
    """Execute dated targets after their information cutoff; cash is explicit."""
    dates = market.calendar[(market.calendar >= start) & (market.calendar <= end)]
    scheduled = {}
    for decision, weights in targets.items():
        later = dates[dates > pd.Timestamp(decision)]
        if len(later):
            execution = later[0]
            assert execution not in scheduled, (decision, execution)
            scheduled[execution] = (decision, weights)
    nav, weights, total_cost, total_turnover = 1.0, {CASH: 1.0}, 0.0, 0.0
    rows = []
    for stamp in dates:
        previous = nav
        observed = market.return_frame.loc[stamp]
        growth = {}
        for asset, weight in weights.items():
            value = observed.get(asset, np.nan)
            if weight > 1e-12:
                assert np.isfinite(value), (stamp, asset, weight)
            growth[asset] = weight * (1 + (value if np.isfinite(value) else 0))
        gross = sum(growth.values())
        nav *= gross
        weights = {a: w / gross for a, w in growth.items()}
        cost = 0.0
        if stamp in scheduled:
            decision, target = scheduled[stamp]
            if callback:
                callback(decision, dict(weights), str(stamp.date()))
            amount = turnover(weights, target)
            cost = .0015 * amount
            total_cost += nav * cost
            total_turnover += amount
            nav *= 1 - cost
            weights = dict(target)
        assert abs(sum(weights.values()) - 1) < 1e-8 and nav > 0
        rows.append(dict(date=str(stamp.date()), nav=nav, daily_return=nav / previous - 1, cost_fraction=cost))
    returns = np.array([r["daily_return"] for r in rows])
    vol = float(returns.std(ddof=1) * np.sqrt(252))
    metrics = dict(total_return=nav - 1, volatility=vol,
                   sharpe=float((252 * returns.mean() - .04) / vol) if vol > 0 else np.nan,
                   max_drawdown=drawdown(returns), cost_initial_nav=total_cost,
                   turnover=total_turnover, trading_days=len(dates), executed_decisions=len(scheduled),
                   unexecuted_terminal_decisions=len(targets) - len(scheduled))
    assert np.isfinite(list(metrics.values())).all()
    return metrics, rows


def covariance_metrics(weights: dict, snapshot: dict, classes: dict, estimator: str) -> dict:
    assets = snapshot["assets"]
    covariance = snapshot["covariance" if estimator == "complete_case" else "pairwise_covariance"].to_numpy()
    w = np.array([weights.get(a, 0) for a in assets])
    same = np.equal.outer([classes[a] for a in assets], [classes[a] for a in assets])
    weighted = np.outer(w, w) * covariance
    variance = float(weighted.sum())
    diagonal = float(np.trace(weighted))
    within = float(weighted[same].sum() - diagonal)
    cross = float(weighted[~same].sum())
    scales = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = float((w @ scales) ** 2 - diagonal)
    assert variance > 0 and denominator > 0
    return dict(variance=variance, diagonal_variance=diagonal, within_class_covariance=within,
                cross_class_covariance=cross, volatility_annual=float(np.sqrt(252 * variance)),
                weighted_pair_correlation=(variance - diagonal) / denominator,
                positive_holdings=int((w > 1e-10).sum()))


def matched_weights(weights: dict, classes: dict, all_assets: bool = False) -> dict:
    result = dict(weights)
    for cls in sorted(set(classes[a] for a in weights if a != CASH)):
        selected = [a for a in weights if a != CASH and classes[a] == cls and (all_assets or weights[a] > 1e-10)]
        if selected:
            amount = sum(weights[a] for a in selected) / len(selected)
            result.update({a: amount for a in selected})
    assert result.get(CASH, 0) == weights.get(CASH, 0)
    return result


def bootstrap_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    indices = np.random.default_rng(SEED).integers(0, len(x), (10000, len(x)))
    a, b = rankdata(x[indices], axis=1), rankdata(y[indices], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    values = (a * b).sum(axis=1)[denominator > 0] / denominator[denominator > 0]
    return float(spearmanr(x, y).statistic), np.quantile(values, [.025, .975]).tolist()


def verify_accounting(market: MarketData) -> dict:
    date = "2024-02-01"
    metrics, nav = replay(market, {date: {"SPY": 1.0, CASH: 0.0}}, date, "2024-02-09")
    execution = market.calendar[market.calendar > date][0]
    after = market.return_frame.loc[(market.calendar > execution) & (market.calendar <= "2024-02-09"), "SPY"]
    expected = .9985 * (1 + after).prod()
    assert np.isclose(nav[-1]["nav"], expected, rtol=1e-12)
    assert nav[0]["nav"] == 1 and metrics["executed_decisions"] == 1
    projected = project({"SPY": .7, "^VIX": .3}, ["SPY"])
    assert projected == {"SPY": .7, CASH: .3}
    return dict(single_asset_path_matches=True, excluded_weight_stays_cash=True,
                execution_strictly_after_decision=True, date_returns_applied_once=True)


def run_saved_allocations(source_root: Path, output_dir: Path) -> dict:
    """Replay frozen decisions and write the published evaluation tables."""
    root = Path(source_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    canonical = root / "EXPERIMENTS_SA_UPGRADE/analysis/s3_tv_v1"
    manifest = read_json(canonical / "manifest.json")
    index_path = canonical / "main_episodes.csv"
    assert digest(index_path) == manifest["outputs"][index_path.name]
    inputs[str(index_path.relative_to(root))] = digest(index_path)
    index = pd.read_csv(index_path)
    assert len(index) == 2790 and len(index.groupby(["model", "profile", "scenario"])) == 120
    assert not index.duplicated(["model", "profile", "scenario", "decision_date"]).any()
    episodes = {}
    support = set()
    for row in index.itertuples(index=False):
        path = root / row.episode_path
        content = path.read_bytes()
        inputs[row.episode_path] = hashlib.sha256(content).hexdigest()
        episode = json.loads(content)
        stages = {s["stage_id"]: {"parsed_output": s["parsed_output"], "ground_truth": s["ground_truth"],
                                    "prompt": "[INVESTOR PROFILE]" if "[INVESTOR PROFILE]" in s.get("prompt", "") else ""}
                  for s in episode["stages"]}
        support.update(stages["S3"]["ground_truth"]["weights"])
        episodes[(row.model, row.profile, row.scenario, row.decision_date)] = (stages, path)
    print(f"Loaded {len(episodes)} canonical decisions; raw support {len(support)}.", flush=True)
    market = MarketData(root, sorted(support), inputs)
    checks = verify_accounting(market)
    references, optimizer_rows, reference_rows = {}, [], []
    for date in sorted(index.decision_date.unique()):
        snapshot = market.snapshot(date)
        for profile in PROFILES:
            weights, info = optimize(snapshot, snapshot["signals"], market.classes, profile)
            references[(date, profile)] = weights
            optimizer_rows.append(dict(kind="reference", model="reference", profile=profile, decision_date=date, **info))
            reference_rows.extend(dict(profile=profile, decision_date=date, asset=a, weight=w) for a, w in weights.items())
    print(f"Computed {len(references)} profile-constrained references.", flush=True)
    financial, decision_rows, nav_rows, covariance_rows, coverage, hybrid_rows = [], [], [], [], [], []
    windows = {}
    for (model, profile, scenario), group in index.groupby(["model", "profile", "scenario"], sort=True):
        first_path = root / group.episode_path.iloc[0]
        result_path = first_path.parents[3] / "backtest_result.json"
        inputs[str(result_path.relative_to(root))] = digest(result_path)
        result = read_json(result_path)
        start, end = result["start_date"], result["end_date"]
        windows[scenario] = (start, end, sorted(group.decision_date))
        targets, hybrid, stage_by_date = {}, {}, {}
        for row in group.itertuples(index=False):
            stages, path = episodes[(model, profile, scenario, row.decision_date)]
            snapshot = market.snapshot(row.decision_date)
            target = project(stages["S3"]["parsed_output"]["weights"], snapshot["assets"])
            targets[row.decision_date] = target
            stage_by_date[row.decision_date] = stages
            risk, safe, compliant = exposure(target, market.classes, profile)
            coverage.append(dict(model=model, profile=profile, scenario=scenario, decision_date=row.decision_date,
                                 eligible_assets=len(snapshot["assets"]), cash=target[CASH], risk_exposure=risk,
                                 safe_exposure=safe, exposure_compliant=compliant,
                                 s3_profile_explicit="[INVESTOR PROFILE]" in stages["S3"].get("prompt", "")))
            if scenario == "normal_bull_2024":
                weights, info = optimize(snapshot, stages["S2"]["parsed_output"]["signals"], market.classes, profile)
                hybrid[row.decision_date] = weights
                optimizer_rows.append(dict(kind="llm_signals", model=model, profile=profile, decision_date=row.decision_date, **info))
                hybrid_rows.extend(dict(model=model, profile=profile, decision_date=row.decision_date, asset=a, weight=w)
                                   for a, w in weights.items())
                control = matched_weights(target, market.classes)
                for estimator in ["complete_case", "pairwise"]:
                    left = covariance_metrics(target, snapshot, market.classes, estimator)
                    right = covariance_metrics(control, snapshot, market.classes, estimator)
                    assert left["positive_holdings"] == right["positive_holdings"]
                    covariance_rows.append(dict(model=model, profile=profile, decision_date=row.decision_date,
                                                estimator=estimator, **{k + "_model": v for k, v in left.items()},
                                                **{k + "_control": v for k, v in right.items()}))

        def record_scores(date, current, execution_date):
            snapshot = market.snapshot(date)
            scores = stage_scores(stage_by_date[date], targets[date], references[(date, profile)],
                                  current, snapshot, market.classes, profile)
            decision_rows.append(dict(model=model, profile=profile, scenario=scenario, decision_date=date,
                                      execution_date=execution_date, **scores))

        metrics, rows = replay(market, targets, start, end, callback=record_scores)
        financial.append(dict(model=model, profile=profile, scenario=scenario, strategy="saved_llm", **metrics))
        nav_rows.extend(dict(model=model, profile=profile, scenario=scenario, strategy="saved_llm", **r) for r in rows)
        if hybrid:
            metrics, rows = replay(market, hybrid, start, end)
            financial.append(dict(model=model, profile=profile, scenario=scenario, strategy="llm_optimizer", **metrics))
            nav_rows.extend(dict(model=model, profile=profile, scenario=scenario, strategy="llm_optimizer", **r) for r in rows)
        print(f"Replayed {model} / {profile} / {scenario}.", flush=True)

    # Deterministic strategies use the identical calendar and execution contract.
    for scenario, (start, end, dates) in windows.items():
        eqw = {date: {**{a: 1 / len(market.snapshot(date)["assets"]) for a in market.snapshot(date)["assets"]}, CASH: 0}
               for date in dates}
        for profile in PROFILES:
            for name, targets in [("equal_weight", eqw), ("rule_optimizer", {d: references[(d, profile)] for d in dates})]:
                metrics, rows = replay(market, targets, start, end)
                financial.append(dict(model=name, profile=profile, scenario=scenario, strategy=name, **metrics))
                nav_rows.extend(dict(model=name, profile=profile, scenario=scenario, strategy=name, **r) for r in rows)

    financial = pd.DataFrame(financial)
    coverage = pd.DataFrame(coverage)
    decisions = pd.DataFrame(decision_rows)
    covariance = pd.DataFrame(covariance_rows)
    outcomes = financial[financial.strategy.eq("saved_llm")].copy()
    equal = financial[financial.strategy.eq("equal_weight")][["profile", "scenario", "sharpe"]]
    outcomes = outcomes.merge(equal.rename(columns={"sharpe": "eqw_sharpe"}), on=["profile", "scenario"], validate="many_to_one")
    outcomes["delta_sharpe"] = outcomes.sharpe - outcomes.eqw_sharpe
    outcomes["gate"] = [-dd <= PROFILES[p][2] for dd, p in zip(outcomes.max_drawdown, outcomes.profile)]
    outcomes = outcomes.merge(decisions.groupby(["model", "profile", "scenario"]).mean(numeric_only=True).reset_index(),
                              on=["model", "profile", "scenario"], validate="one_to_one", suffixes=("", "_decision"))
    compliance = coverage.groupby(["model", "profile", "scenario"]).exposure_compliant.mean().rename("exposure_compliance").reset_index()
    outcomes = outcomes.merge(compliance, on=["model", "profile", "scenario"], validate="one_to_one")
    assert len(outcomes) == 120 and len(covariance) == 660
    qa_path = root / "EXPERIMENTS_SA_UPGRADE/analysis/paper_evidence_20260929/qa_model_scores.csv"
    inputs[str(qa_path.relative_to(root))] = digest(qa_path)
    qa = pd.read_csv(qa_path)
    qa_scores = qa.pivot(index="model", columns="condition", values="mean").reindex(MODEL_LABELS)
    qa_scores["all_seven"] = qa_scores[[f"T{i}" for i in range(1, 8)]].mean(axis=1)
    qa_scores["without_t3_t4"] = qa_scores[["T1", "T2", "T5", "T6", "T7"]].mean(axis=1)
    associations = []
    for (scenario, profile), cell in outcomes.groupby(["scenario", "profile"]):
        cell = cell.set_index("model").reindex(qa_scores.index)
        for variant in ["all_seven", "without_t3_t4"]:
            rho, ci = bootstrap_correlation(qa_scores[variant].to_numpy(), cell.sharpe.to_numpy())
            associations.append(dict(scenario=scenario, profile=profile, variant=variant, rho=rho, ci_low=ci[0], ci_high=ci[1], n_models=10))

    covariance["reduction"] = 1 - np.sqrt(covariance.variance_control / covariance.variance_model)
    covariance["correlation_gap"] = covariance.weighted_pair_correlation_model - covariance.weighted_pair_correlation_control
    correlation_summary = []
    for estimator, group in covariance.groupby("estimator"):
        means = group.groupby("model").reduction.mean().to_numpy()
        samples = np.random.default_rng(SEED).integers(0, 10, (10000, 10))
        ci = np.quantile(means[samples].mean(axis=1), [.025, .975])
        differences = {c: float((group[c + "_model"] - group[c + "_control"]).mean())
                       for c in ["variance", "diagonal_variance", "within_class_covariance", "cross_class_covariance"]}
        correlation_summary.append(dict(estimator=estimator, n=330, reduction=float(group.reduction.mean()),
                                        ci_low=float(ci[0]), ci_high=float(ci[1]), lower_count=int((group.reduction > 1e-10).sum()),
                                        correlation_gap=float(group.correlation_gap.mean()), **differences))
    normal = outcomes[outcomes.scenario.eq("normal_bull_2024")]
    process_associations = []
    for profile, cell in normal.groupby("profile"):
        for metric in ["sharpe", "volatility", "max_drawdown"]:
            values = -cell[metric].to_numpy() if metric == "max_drawdown" else cell[metric].to_numpy()
            rho, ci = bootstrap_correlation(cell.ceps.to_numpy(), values)
            process_associations.append(dict(profile=profile, metric=metric, rho=rho, ci_low=ci[0], ci_high=ci[1]))
    artifacts = {
        "financial.csv": financial, "outcomes.csv": outcomes, "stage_scores.csv": decisions,
        "nav.csv": pd.DataFrame(nav_rows), "coverage.csv": coverage, "references.csv": pd.DataFrame(reference_rows),
        "hybrid_allocations.csv": pd.DataFrame(hybrid_rows), "optimizer_status.csv": pd.DataFrame(optimizer_rows),
        "correlation_decisions.csv": covariance, "qa_scores.csv": qa_scores.reset_index(),
        "qa_associations.csv": pd.DataFrame(associations), "process_associations.csv": pd.DataFrame(process_associations),
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False, float_format="%.12g")
    summary = dict(models=10, cells=120, canonical_decisions=len(index), scored_executed_decisions=len(decisions),
                   projected_decisions=len(coverage), nontradable_assets=["^VIX"], raw_support=len(support),
                   tradable_support=len(market.assets), eligible_min=int(coverage.eligible_assets.min()),
                   eligible_max=int(coverage.eligible_assets.max()), wins=int((outcomes.delta_sharpe > 0).sum()),
                   normal_wins=int((normal.delta_sharpe > 0).sum()),
                   wins_by_window=outcomes.groupby("scenario").delta_sharpe.apply(lambda x: int((x > 0).sum())).to_dict(),
                   normal_profile_compliance=normal.groupby("profile").exposure_compliance.mean().to_dict(),
                   optimizer_fallbacks=int((~pd.DataFrame(optimizer_rows).accepted).sum()), optimizer_problems=len(optimizer_rows),
                   correlation=correlation_summary,
                   qa_intervals_including_zero=int(sum(r["ci_low"] <= 0 <= r["ci_high"] for r in associations)),
                   s3_explicit_profile_count=int(coverage.s3_profile_explicit.sum()),
                   windows={k: dict(start=v[0], end=v[1], decisions=len(v[2])) for k, v in windows.items()})
    write_json(output / "summary.json", summary)
    checks.update(all_cells_retained=len(outcomes) == 120, all_decisions_retained=len(coverage) == 2790,
                  references_profile_feasible=True, finite_outcomes=True, covariance_pairs=330,
                  no_model_calls=True, no_source_mutations=True)
    write_json(output / "validation.json", checks)
    write_json(output / "manifest.json", dict(source_root=str(root), script_sha256=digest(Path(__file__)),
                                           seed=SEED, inputs=inputs,
                                           outputs={p.name: digest(p) for p in output.iterdir() if p.suffix in {".csv", ".json"} and p.name != "manifest.json"}))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
