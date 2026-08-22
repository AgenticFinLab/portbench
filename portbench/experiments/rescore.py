"""
rescore_ceps: recompute CEPS scores for all completed runs using the updated
S3 ground truth (max-Sharpe optimal weights instead of equal-weight buy signals).

Does NOT re-call any LLM — reads actual stage outputs from pipeline_logs,
rebuilds MarketSnapshots from the data provider to get future_return_data,
then rescores S3/S4/S5 and recomputes CEPS per episode.

Usage:
    python -m portbench.experiments --rescore --rebalance monthly
    python -m portbench.experiments --rescore --rebalance monthly --config configs/experiments/default.yaml
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to reconstruct typed stage outputs from episode JSON dicts
# ---------------------------------------------------------------------------


def _s3_from_dict(d: dict):
    from ..agent_eval.base import S3Output

    return S3Output(weights={k: float(v) for k, v in d.get("weights", {}).items()})


def _s4_from_dict(d: dict):
    from ..agent_eval.base import S4Output

    return S4Output(
        executed_weights={
            k: float(v) for k, v in d.get("executed_weights", {}).items()
        },
        total_cost=float(d.get("total_cost", 0.0)),
        turnover=float(d.get("turnover", 0.0)),
    )


def _s5_from_dict(d: dict):
    from ..agent_eval.base import S5Output

    return S5Output(
        portfolio_var=float(d.get("portfolio_var", 0.0)),
        portfolio_drawdown=float(d.get("portfolio_drawdown", 0.0)),
        weight_drift=float(d.get("weight_drift", 0.0)),
        rebalance_needed=bool(d.get("rebalance_needed", False)),
    )


# ---------------------------------------------------------------------------
# Per-episode rescore
# ---------------------------------------------------------------------------


def _rescore_episode(
    episode: dict,
    snapshot_builder,
    forward_days: int,
    propagation_weight: float,
    profile=None,
    current_weights: dict = None,
) -> Optional[tuple[float, dict]]:
    """
    Rescore one episode using updated S3 GT.

    Returns (ceps_score, {S1: score, ..., S5: score}), or None if insufficient data.
    Accepts optional `current_weights` so gt_turnover is computed from the correct
    starting portfolio rather than an empty one.
    """
    from ..agent_eval.stages import (
        S3WeightOptimization,
        S4ExecutionSimulation,
        S5RiskMonitoring,
    )
    from ..metrics.ceps import CEPS, StageScore

    stages_by_id = {s["stage_id"]: s for s in episode.get("stages", [])}

    # Require at least S3 parsed output to do anything useful
    s3_log = stages_by_id.get("S3")
    if not s3_log or not s3_log.get("parsed_output", {}).get("weights"):
        return None

    dec_date = date.fromisoformat(episode["decision_date"])

    # Rebuild snapshot with future_return_data and (optionally) correct current_weights
    try:
        snapshot = snapshot_builder.build(
            dec_date,
            current_weights=current_weights or {},
            nav=1_000_000.0,  # fixed; only affects cost_drag (< 0.1%)
            forward_days=forward_days,
        )
    except Exception as exc:
        log.debug("snapshot rebuild failed for %s: %s", dec_date, exc)
        return None

    if not snapshot.return_data:
        return None

    # Instantiate scorer stages (no LLM adapter — only GT computation and score())
    s3_stage = S3WeightOptimization()
    s3_stage._last_snapshot = snapshot  # needed for correlation-awareness scoring
    s4_stage = S4ExecutionSimulation()
    s5_stage = S5RiskMonitoring(profile=profile)

    # Compute new GTs for S3, S4, S5
    try:
        s3_gt = s3_stage.compute_ground_truth(snapshot)
        s4_gt = s4_stage.compute_ground_truth(snapshot)
        s5_gt = s5_stage.compute_ground_truth(snapshot)
    except Exception as exc:
        log.debug("GT computation failed for %s: %s", dec_date, exc)
        return None

    # Reconstruct actual outputs from episode JSON
    s3_actual = _s3_from_dict(s3_log.get("parsed_output", {}))
    s4_log = stages_by_id.get("S4", {})
    s5_log = stages_by_id.get("S5", {})
    s4_actual = _s4_from_dict(s4_log.get("parsed_output", {})) if s4_log else None
    s5_actual = _s5_from_dict(s5_log.get("parsed_output", {})) if s5_log else None

    # Re-score S3, S4, S5; keep S1/S2 scores from episode file
    s1_score = float(stages_by_id.get("S1", {}).get("score", 0.0))
    s2_score = float(stages_by_id.get("S2", {}).get("score", 0.0))
    s3_score = s3_stage.score(s3_actual, s3_gt)
    s4_score = (
        s4_stage.score(s4_actual, s4_gt)
        if s4_actual
        else float(stages_by_id.get("S4", {}).get("score", 0.0))
    )
    s5_score = (
        s5_stage.score(s5_actual, s5_gt)
        if s5_actual
        else float(stages_by_id.get("S5", {}).get("score", 0.0))
    )

    stage_scores = [
        StageScore(
            stage_id="S1", stage_name="S1_MARKET_INTERPRETATION", score=s1_score
        ),
        StageScore(stage_id="S2", stage_name="S2_SIGNAL_GENERATION", score=s2_score),
        StageScore(stage_id="S3", stage_name="S3_WEIGHT_OPTIMIZATION", score=s3_score),
        StageScore(stage_id="S4", stage_name="S4_EXECUTION_SIMULATION", score=s4_score),
        StageScore(stage_id="S5", stage_name="S5_RISK_MONITORING", score=s5_score),
    ]

    ceps_score = CEPS(propagation_weight).compute(stage_scores).ceps_score
    per_stage = {
        "S1": s1_score, "S2": s2_score, "S3": s3_score,
        "S4": s4_score, "S5": s5_score,
    }
    return ceps_score, per_stage


# ---------------------------------------------------------------------------
# Per-profile rescore
# ---------------------------------------------------------------------------


def _load_weight_history(p_dir: Path) -> "dict[str, dict[str, float]]":
    """Load weight_history.csv → {date_str: {asset: weight}}. Returns {} on missing file."""
    path = p_dir / "weight_history.csv"
    if not path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index).normalize()
        return {
            str(idx.date()): {col: float(val) for col, val in row.items() if not np.isnan(val)}
            for idx, row in df.iterrows()
        }
    except Exception:
        return {}


def _prev_weights(weight_history: "dict[str, dict[str, float]]", dec_date: date) -> dict:
    """Return the most recent portfolio weights strictly before dec_date."""
    candidates = [d for d in weight_history if d < str(dec_date)]
    if not candidates:
        return {}
    return weight_history[max(candidates)]


def _rescore_profile(
    p_dir: Path,
    snapshot_builder,
    forward_days: int,
    propagation_weight: float,
    profile_name: Optional[str] = None,
) -> Optional[tuple[list[float], dict[str, float]]]:
    """
    Rescore all episodes in a profile's pipeline_logs.

    Returns (per_step_ceps, mean_stage_scores) or None if pipeline_logs are missing/empty.
    Reads weight_history.csv to supply correct current_weights for each episode so that
    S4 gt_turnover is computed from the actual prior portfolio, not an empty one.
    """
    from ..agent_eval.investor_profiles import PROFILES

    profile = PROFILES.get(profile_name) if profile_name else None
    logs_root = p_dir / "pipeline_logs"
    if not logs_root.exists():
        return None

    episode_files = sorted(logs_root.glob("*/episodes/*.json"))
    if not episode_files:
        return None

    weight_history = _load_weight_history(p_dir)

    new_ceps: list[float] = []
    stage_accum: dict[str, list[float]] = {}

    for ep_path in episode_files:
        try:
            episode = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dec_date = date.fromisoformat(episode["decision_date"])
        cur_w = _prev_weights(weight_history, dec_date)
        result = _rescore_episode(
            episode, snapshot_builder, forward_days, propagation_weight,
            profile=profile, current_weights=cur_w,
        )
        if result is not None:
            ceps, per_stage = result
            new_ceps.append(ceps)
            for sid, sc in per_stage.items():
                stage_accum.setdefault(sid, []).append(sc)

    if not new_ceps:
        return None

    mean_stage = {
        sid: round(float(np.mean(scores)), 4)
        for sid, scores in stage_accum.items()
    }
    return new_ceps, mean_stage


# ---------------------------------------------------------------------------
# Update run_summary.json
# ---------------------------------------------------------------------------


def _update_run_summary(
    r_dir: Path,
    profile_name: str,
    new_per_step_ceps: list[float],
    mean_stage_scores: dict[str, float] = None,
) -> None:
    """Patch the CEPS fields in run_summary.json for a single profile."""
    summary_path = r_dir / "run_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_data = summary.get("profiles", {}).get(profile_name)
    if profile_data is None or profile_data.get("normal") is None:
        return

    mean_ceps = float(np.mean(new_per_step_ceps))
    std_ceps = float(np.std(new_per_step_ceps))

    profile_data["normal"]["per_step_ceps"] = [round(v, 6) for v in new_per_step_ceps]
    profile_data["normal"]["mean_ceps"] = round(mean_ceps, 6)
    profile_data["normal"]["std_ceps"] = round(std_ceps, 6)
    if mean_stage_scores:
        profile_data["normal"]["mean_stage_scores"] = {
            k: round(v, 4) for k, v in mean_stage_scores.items()
        }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


def _update_run_summary_stress(
    r_dir: Path,
    profile_name: str,
    scenario: str,
    per_step_ceps: list[float],
    mean_stage_scores: dict[str, float] = None,
) -> None:
    """Patch stress CEPS fields in run_summary.json for a profile/scenario."""
    summary_path = r_dir / "run_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_data = summary.get("profiles", {}).get(profile_name)
    if profile_data is None:
        return

    sr = profile_data.get("stress_results")
    # Convert legacy list format to dict format
    if isinstance(sr, list):
        sr = {item["scenario"]: item for item in sr}
        profile_data["stress_results"] = sr
    if sr is None:
        sr = {}
        profile_data["stress_results"] = sr

    entry = sr.setdefault(scenario, {})
    entry["per_step_ceps"] = [round(v, 6) for v in per_step_ceps]
    entry["mean_ceps"] = round(float(np.mean(per_step_ceps)), 6)
    if mean_stage_scores:
        entry["mean_stage_scores"] = {
            k: round(v, 4) for k, v in mean_stage_scores.items()
        }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def rescore_ceps(
    rebalance: str = "monthly",
    output_root: str = "EXPERIMENTS",
    config_path: Optional[str] = None,
    propagation_weight: float = 0.1,
    logger=None,
) -> dict:
    """
    Rescore CEPS for all completed runs under EXPERIMENTS/{rebalance}/.

    Args:
        rebalance:          Rebalance frequency directory to scan.
        output_root:        Root experiment directory.
        config_path:        Path to YAML config (to read data_provider settings).
                            If None, uses defaults (mock data provider, seed=42).
        propagation_weight: Cascade penalty weight for CEPS (must match original).

    Returns:
        Dict with counts: {rescored, skipped, errors}.
    """
    from ..sandbox.snapshot_builder import SnapshotBuilder
    from ..sandbox.engine import _REBALANCE_FORWARD_DAYS
    from . import paths

    _log = logger.info if logger else print

    # ── Load config for data provider settings ───────────────────────────────
    data_provider_kind = "mock"
    data_dir = "datasets/processed"
    sec_dir = "datasets/sec"
    seed = 42
    lookback_days = 60
    asset_class_map_path = None

    if config_path:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        data_provider_kind = raw.get("data_provider", data_provider_kind)
        data_dir = raw.get("data_dir", data_dir)
        sec_dir = raw.get("sec_dir", sec_dir)
        seed = raw.get("seed", seed)
        lookback_days = raw.get("lookback_days", lookback_days)

    # ── Build data provider ───────────────────────────────────────────────────
    if data_provider_kind == "processed":
        from ..qa_builder.processed_data import ProcessedDataProvider

        provider = ProcessedDataProvider(data_dir=data_dir, sec_dir=sec_dir)
        acmap_file = Path(data_dir) / "asset_class_map.json"
        asset_class_map = (
            json.loads(acmap_file.read_text(encoding="utf-8"))
            if acmap_file.exists()
            else None
        )
    else:
        from ..qa_builder.mock_data import MockDataProvider

        provider = MockDataProvider(seed=seed)
        asset_class_map = None

    assets = provider.list_assets()
    forward_days = _REBALANCE_FORWARD_DAYS.get(rebalance, 21)

    snapshot_builder = SnapshotBuilder(
        provider=provider,
        assets=assets,
        lookback_days=lookback_days,
        asset_class_map=asset_class_map,
    )

    # ── Scan rebalance dir ───────────────────────────────────────────────────
    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    if not rebal_dir.exists():
        raise FileNotFoundError(f"No results found at {rebal_dir}")

    n_rescored = n_skipped = n_errors = 0

    for prov_dir in sorted(rebal_dir.iterdir()):
        if (
            not prov_dir.is_dir()
            or prov_dir.name.startswith("_")
            or prov_dir.name == "comparison_figures"
        ):
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            # Find the best (most complete) run
            ts = paths.find_best_run(
                output_root, rebalance, prov_dir.name, m_dir.name, []
            )
            if not ts:
                continue
            r_dir = paths.run_dir(output_root, rebalance, prov_dir.name, m_dir.name, ts)
            summary_path = r_dir / "run_summary.json"
            if not summary_path.exists():
                n_skipped += 1
                continue

            model_key = f"{prov_dir.name}/{m_dir.name}"
            _log(f"rescore: {model_key} ts={ts}")

            for p_dir in sorted(r_dir.iterdir()):
                if not p_dir.is_dir() or p_dir.name.startswith(("stress_", "_")):
                    continue
                # Normal profile directories: conservative | balanced | aggressive
                profile_name = p_dir.name
                if profile_name not in ("conservative", "balanced", "aggressive"):
                    continue
                # Look for pipeline_logs under normal/ sub-directory
                normal_p_dir = p_dir / "normal"
                if not normal_p_dir.exists():
                    n_skipped += 1
                    continue

                try:
                    result = _rescore_profile(
                        normal_p_dir,
                        snapshot_builder,
                        forward_days,
                        propagation_weight,
                        profile_name=profile_name,
                    )
                    if result is None:
                        _log(f"  {profile_name}: no pipeline_logs — skipped")
                        n_skipped += 1
                        continue
                    new_ceps, mean_stage_scores = result
                    _update_run_summary(r_dir, profile_name, new_ceps, mean_stage_scores)
                    mean_new = float(np.mean(new_ceps))
                    _log(
                        f"  {profile_name}: {len(new_ceps)} episodes rescored, mean_ceps={mean_new:.4f}"
                    )
                    n_rescored += 1
                    # Rescore stress scenarios for this profile
                    for stress_dir in sorted(p_dir.iterdir()):
                        if not stress_dir.is_dir() or not stress_dir.name.startswith("stress_"):
                            continue
                        scenario = stress_dir.name[len("stress_"):]
                        try:
                            stress_result = _rescore_profile(
                                stress_dir,
                                snapshot_builder,
                                forward_days,
                                propagation_weight,
                                profile_name=profile_name,
                            )
                            if stress_result is not None:
                                s_ceps, s_stage = stress_result
                                _update_run_summary_stress(
                                    r_dir, profile_name, scenario, s_ceps, s_stage
                                )
                                _log(f"    stress/{scenario}: {len(s_ceps)} episodes rescored")
                        except Exception as exc:
                            _log(f"    stress/{scenario}: ERROR — {exc}")
                except Exception as exc:
                    _log(f"  {profile_name}: ERROR — {exc}")
                    n_errors += 1

    _log(
        f"rescore complete: rescored={n_rescored} skipped={n_skipped} errors={n_errors}"
    )

    # ── Propagate updated CEPS back to backtest_result.json ──────────────────
    _sync_backtest_results(rebal_dir, _log)

    # ── Regenerate all figures + analysis report ─────────────────────────────
    _regenerate_figures(rebal_dir, output_root, rebalance, _log)

    return {"rescored": n_rescored, "skipped": n_skipped, "errors": n_errors}


# ---------------------------------------------------------------------------
# Post-rescore helpers
# ---------------------------------------------------------------------------


def _sync_backtest_results(rebal_dir: Path, log) -> None:
    """
    Copy updated per_step_ceps / mean_ceps from run_summary.json into the
    individual backtest_result.json files so comparison figures read fresh values.
    """
    n = 0
    for prov_dir in sorted(rebal_dir.iterdir()):
        if (
            not prov_dir.is_dir()
            or prov_dir.name.startswith("_")
            or prov_dir.name == "comparison_figures"
        ):
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            for ts_dir in sorted(m_dir.iterdir()):
                if not ts_dir.is_dir():
                    continue
                summary_f = ts_dir / "run_summary.json"
                if not summary_f.exists():
                    continue
                try:
                    summary = json.loads(summary_f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for profile_name, payload in summary.get("profiles", {}).items():
                    normal = payload.get("normal")
                    if normal:
                        br_path = ts_dir / profile_name / "normal" / "backtest_result.json"
                        if br_path.exists():
                            try:
                                br = json.loads(br_path.read_text(encoding="utf-8"))
                                per_step = normal.get("per_step_ceps", [])
                                br["per_step_ceps"] = per_step
                                br["mean_ceps"] = (
                                    round(float(np.mean(per_step)), 6) if per_step else 0.0
                                )
                                br_path.write_text(
                                    json.dumps(br, indent=2, default=str), encoding="utf-8"
                                )
                                n += 1
                            except Exception:
                                pass
                    # Sync stress backtest_results
                    sr = payload.get("stress_results", {})
                    if isinstance(sr, list):
                        sr = {item["scenario"]: item for item in sr}
                    for scenario, s_data in sr.items():
                        br_path = ts_dir / profile_name / f"stress_{scenario}" / "backtest_result.json"
                        if not br_path.exists():
                            continue
                        try:
                            br = json.loads(br_path.read_text(encoding="utf-8"))
                            per_step = s_data.get("per_step_ceps", [])
                            br["per_step_ceps"] = per_step
                            br["mean_ceps"] = (
                                round(float(np.mean(per_step)), 6) if per_step else 0.0
                            )
                            br_path.write_text(
                                json.dumps(br, indent=2, default=str), encoding="utf-8"
                            )
                            n += 1
                        except Exception:
                            pass
    log(f"synced backtest_result.json: {n} files updated")


def _regenerate_figures(rebal_dir: Path, output_root: str, rebalance: str, log) -> None:
    """Regenerate all comparison figures and the analysis report."""
    from .figures import render_batch_comparison_figures
    from .analysis import analyze_runs

    # Cross-model NAV / metrics / stress drawdown figures
    try:
        render_batch_comparison_figures(
            rebal_dir,
            run_timestamps={},  # auto-discover all best runs
            output_root=output_root,
            rebalance=rebalance,
        )
        log(f"comparison figures regenerated → {rebal_dir / 'comparison_figures'}")
    except Exception as exc:
        log(f"comparison figures failed: {exc}")

    # Analysis figures (rankings, stress_gate, ceps_breakdown, risk_return_scatter)
    # + copies them to comparison_figures + writes analysis_report.md + figure_index.md
    try:
        report = analyze_runs(rebalance=rebalance, output_root=output_root, logger=None)
        log(f"analysis report → {report}")
    except Exception as exc:
        log(f"analysis report failed: {exc}")


# ---------------------------------------------------------------------------
# σ Ablation: rescore S3 for multiple sigma values
# ---------------------------------------------------------------------------


def rescore_sigma_ablation(
    sigma_values: list[float] = None,
    rebalance: str = "monthly",
    output_root: str = "EXPERIMENTS",
    config_path: Optional[str] = None,
    propagation_weight: float = 0.1,
    logger=None,
) -> Path:
    """
    Rescore S3 for each σ value in sigma_values, collecting per-(model, profile, σ) CEPS.

    Reads pipeline_logs from existing runs (no LLM re-calls). Results are written to
    {output_root}/{rebalance}/sigma_ablation/results.json and a figure is generated.

    Returns the path to results.json.
    """
    from ..agent_eval.stages import S3WeightOptimization
    from ..metrics.ceps import CEPS, StageScore
    from ..sandbox.snapshot_builder import SnapshotBuilder
    from ..sandbox.engine import _REBALANCE_FORWARD_DAYS
    from ..agent_eval.investor_profiles import PROFILES
    from . import paths

    if sigma_values is None:
        sigma_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    sigma_values = [float(s) for s in sigma_values]

    _log = logger.info if logger else print

    # ── Build data provider (same as rescore_ceps) ───────────────────────────
    data_provider_kind = "mock"
    data_dir = "datasets/processed"
    sec_dir = "datasets/sec"
    seed = 42
    lookback_days = 60

    if config_path:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        data_provider_kind = raw.get("data_provider", data_provider_kind)
        data_dir = raw.get("data_dir", data_dir)
        sec_dir = raw.get("sec_dir", sec_dir)
        seed = raw.get("seed", seed)
        lookback_days = raw.get("lookback_days", lookback_days)
        sigma_values = [
            float(s) for s in raw.get("sigma_ablation_values", sigma_values)
        ]

    if data_provider_kind == "processed":
        from ..qa_builder.processed_data import ProcessedDataProvider

        provider = ProcessedDataProvider(data_dir=data_dir, sec_dir=sec_dir)
        acmap_file = Path(data_dir) / "asset_class_map.json"
        asset_class_map = (
            json.loads(acmap_file.read_text(encoding="utf-8"))
            if acmap_file.exists()
            else None
        )
    else:
        from ..qa_builder.mock_data import MockDataProvider

        provider = MockDataProvider(seed=seed)
        asset_class_map = None

    assets = provider.list_assets()
    forward_days = _REBALANCE_FORWARD_DAYS.get(rebalance, 21)
    snapshot_builder = SnapshotBuilder(
        provider=provider,
        assets=assets,
        lookback_days=lookback_days,
        asset_class_map=asset_class_map,
    )

    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    if not rebal_dir.exists():
        raise FileNotFoundError(f"No results found at {rebal_dir}")

    # results[model_key][profile_name][sigma_str] = mean_ceps
    results: dict[str, dict[str, dict[str, float]]] = {}

    for prov_dir in sorted(rebal_dir.iterdir()):
        if (
            not prov_dir.is_dir()
            or prov_dir.name.startswith("_")
            or prov_dir.name == "comparison_figures"
        ):
            continue
        if prov_dir.name == "baseline":
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            model_key = f"{prov_dir.name}/{m_dir.name}"
            ts = paths.find_best_run(
                output_root, rebalance, prov_dir.name, m_dir.name, []
            )
            if not ts:
                continue
            r_dir = paths.run_dir(output_root, rebalance, prov_dir.name, m_dir.name, ts)

            results[model_key] = {}
            for profile_name in ("conservative", "balanced", "aggressive"):
                p_dir = r_dir / profile_name / "normal"
                if not p_dir.exists():
                    continue
                episode_files = (
                    sorted((p_dir / "pipeline_logs").glob("*/episodes/*.json"))
                    if (p_dir / "pipeline_logs").exists()
                    else []
                )
                if not episode_files:
                    continue

                episodes_data = []
                for ep_path in episode_files:
                    try:
                        episodes_data.append(
                            json.loads(ep_path.read_text(encoding="utf-8"))
                        )
                    except Exception:
                        continue
                if not episodes_data:
                    continue

                profile_obj = PROFILES.get(profile_name)
                results[model_key][profile_name] = {}

                for sigma in sigma_values:
                    sigma_str = str(sigma)
                    ceps_scores = []
                    for episode in episodes_data:
                        stages_by_id = {
                            s["stage_id"]: s for s in episode.get("stages", [])
                        }
                        s3_log = stages_by_id.get("S3")
                        if not s3_log or not s3_log.get("parsed_output", {}).get(
                            "weights"
                        ):
                            continue
                        dec_date = date.fromisoformat(episode["decision_date"])
                        try:
                            snapshot = snapshot_builder.build(
                                dec_date,
                                current_weights={},
                                nav=1_000_000.0,
                                forward_days=forward_days,
                            )
                        except Exception:
                            continue
                        if not snapshot.return_data:
                            continue

                        from ..agent_eval.base import S3Output, S4Output, S5Output

                        s3_stage = S3WeightOptimization(sigma=sigma)
                        s3_stage._last_snapshot = snapshot
                        s3_gt = s3_stage.compute_ground_truth(snapshot)
                        s3_actual = S3Output(
                            weights={
                                k: float(v)
                                for k, v in s3_log.get("parsed_output", {})
                                .get("weights", {})
                                .items()
                            }
                        )
                        s3_score = s3_stage.score(s3_actual, s3_gt)

                        s1_score = float(stages_by_id.get("S1", {}).get("score", 0.0))
                        s2_score = float(stages_by_id.get("S2", {}).get("score", 0.0))
                        s4_score = float(stages_by_id.get("S4", {}).get("score", 0.0))
                        s5_score = float(stages_by_id.get("S5", {}).get("score", 0.0))

                        stage_scores = [
                            StageScore("S1", "S1", s1_score),
                            StageScore("S2", "S2", s2_score),
                            StageScore("S3", "S3", s3_score),
                            StageScore("S4", "S4", s4_score),
                            StageScore("S5", "S5", s5_score),
                        ]
                        ceps_scores.append(
                            CEPS(propagation_weight).compute(stage_scores).ceps_score
                        )

                    if ceps_scores:
                        results[model_key][profile_name][sigma_str] = round(
                            float(np.mean(ceps_scores)), 4
                        )
                    _log(
                        f"  σ={sigma} {model_key}/{profile_name}: {len(ceps_scores)} eps, mean={results[model_key][profile_name].get(sigma_str, 0):.4f}"
                    )

    # ── Write results.json ───────────────────────────────────────────────────
    ablation_dir = rebal_dir / "sigma_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    results_path = ablation_dir / "results.json"
    payload = {
        "sigma_values": sigma_values,
        "rebalance": rebalance,
        "models": results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"sigma ablation results → {results_path}")

    # ── Generate figure ──────────────────────────────────────────────────────
    try:
        from ..visualization.ablation_plots import plot_sigma_ablation
        from ..visualization.style import save_figure

        fig = plot_sigma_ablation(payload, title=f"σ Ablation — {rebalance}")
        save_figure(fig, str(ablation_dir / "sigma_ablation.png"), formats=("png",))
        _log(f"sigma ablation figure → {ablation_dir / 'sigma_ablation.png'}")
    except Exception as exc:
        _log(f"sigma ablation figure failed: {exc}")

    return results_path


# ---------------------------------------------------------------------------
# λ (Propagation Weight) Sweep: CEPS ranking stability
# ---------------------------------------------------------------------------


def rescore_lambda_sweep(
    lambda_values: list[float] = None,
    rebalance: str = "monthly",
    output_root: str = "EXPERIMENTS",
    config_path: Optional[str] = None,
    logger=None,
) -> Path:
    """
    Rescore CEPS for multiple propagation_weight (λ) values.

    Reads pipeline_logs from existing runs (no LLM re-calls) and recomputes
    CEPS at each λ. Reports ranking stability via Kendall W.

    Output written to {output_root}/{rebalance}/lambda_sweep/results.json.

    Returns path to results.json.
    """
    import numpy as np
    from scipy.stats import kendalltau

    from ..metrics.ceps import CEPS, StageScore
    from ..agent_eval.base import StageID
    from . import paths

    if lambda_values is None:
        lambda_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    lambda_values = [float(lam) for lam in lambda_values]

    _log = logger.info if logger else print

    rebal_dir = Path(output_root) / rebalance

    # --- Collect episodes from pipeline_logs ---
    all_models: dict[str, list[list[StageScore]]] = {}
    for provider_dir in sorted(rebal_dir.glob("*")):
        if not provider_dir.is_dir():
            continue
        for model_dir in sorted(provider_dir.glob("*")):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.glob("*"), reverse=True):
                if not run_dir.is_dir():
                    continue
                model_key = f"{provider_dir.name}/{model_dir.name}"
                episodes: list[list[StageScore]] = []
                for profile_dir in sorted(run_dir.glob("*")):
                    if not profile_dir.is_dir():
                        continue
                    for scenario_dir in sorted(profile_dir.glob("*")):
                        if scenario_dir.name in ("figures", "snapshots", "step_cache"):
                            continue
                        log_dir = scenario_dir / "pipeline_logs"
                        if not log_dir.is_dir():
                            continue
                        # Find the latest run subdir under pipeline_logs
                        run_dirs = sorted(log_dir.glob("*"))
                        if not run_dirs:
                            continue
                        log_subdir = run_dirs[-1]
                        episodes_dir = log_subdir / "episodes"
                        if not episodes_dir.is_dir():
                            continue
                        for ep_file in sorted(episodes_dir.glob("*.json")):
                            try:
                                data = json.loads(ep_file.read_text(encoding="utf-8"))
                                stages = data.get("stages", [])
                                scores = []
                                for s in stages:
                                    score = float(s.get("score", 0.0))
                                    scores.append(
                                        StageScore(
                                            stage_id=str(s.get("stage_id", "")),
                                            stage_name=str(s.get("stage_id", "")),
                                            score=score,
                                        )
                                    )
                                if scores:
                                    episodes.append(scores)
                            except Exception:
                                continue
                if episodes:
                    all_models[model_key] = episodes
                break  # Only use the latest run per model

    if not all_models:
        _log("No pipeline_logs found. Run experiments first with save_pipeline_logs: true.")
        empty_path = rebal_dir / "lambda_sweep" / "results.json"
        empty_path.parent.mkdir(parents=True, exist_ok=True)
        empty_path.write_text(json.dumps({"error": "no data found"}, indent=2))
        return empty_path

    _log(f"Found pipeline_logs for {len(all_models)} models")

    # --- Compute CEPS at each λ for each model ---
    model_ceps: dict[str, dict[float, float]] = {}
    for model_key, episodes in all_models.items():
        model_ceps[model_key] = {}
        for lam in lambda_values:
            ceps = CEPS(propagation_weight=lam)
            batch = ceps.compute_batch(episodes)
            model_ceps[model_key][lam] = batch["mean_ceps"]

    # --- Ranking stability (Kendall W) ---
    model_list = sorted(model_ceps.keys())
    if len(model_list) < 3:
        _log("Need ≥3 models for ranking stability; skipping Kendall W")
    else:
        rank_matrix = np.zeros((len(model_list), len(lambda_values)))
        for j, lam in enumerate(lambda_values):
            scores = [model_ceps[m][lam] for m in model_list]
            order = np.argsort(np.argsort(-np.array(scores)))
            rank_matrix[:, j] = order

        tau_values = []
        for j1 in range(len(lambda_values)):
            for j2 in range(j1 + 1, len(lambda_values)):
                tau, _ = kendalltau(rank_matrix[:, j1], rank_matrix[:, j2])
                tau_values.append(tau)
        mean_tau = float(np.mean(tau_values))
        _log(f"Mean pairwise Kendall τ across λ: {mean_tau:.4f}")
        _log(f"(τ > 0.8 → ranks stable; τ < 0.5 → λ-sensitive)")

    # --- Output ---
    result = {
        "lambda_values": lambda_values,
        "model_ceps": model_ceps,
        "n_models": len(model_list),
    }
    if len(model_list) >= 3:
        result["mean_kendall_tau"] = mean_tau

    sweep_dir = rebal_dir / "lambda_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"λ sweep results → {results_path}")

    return results_path
