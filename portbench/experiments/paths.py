"""
Output path conventions + shared persistence helpers for batch experiments.

New layout (created lazily as experiments run):

  EXPERIMENTS/
    _dataset_figures/              # dataset-level correlation figures (shared)
    {rebalance}/                   # monthly | weekly | quarterly
      comparison_figures/          # cross-model comparison figures
      {provider}/                  # ark | tencent | baseline | mock
        {model}/                   # doubao-seed-2-0-pro-260215 | equal_weight
          {timestamp}/             # 20260510_042407  (one run = all profiles)
            run_config.yaml        # config snapshot for this run
            run_summary.json       # aggregated results across all profiles
            env_meta.json          # git hash, python version, timestamps
            errors.jsonl           # per-profile errors
            checkpoint.json        # completed profile names
            {profile}/             # conservative | balanced | aggressive
              experiment.log
              figures/
              normal/
                backtest_result.json
                nav_curve.csv
                weight_history.csv
                trade_history.json
                summary.txt
                snapshots/
                pipeline_logs/
              stress_{scenario}/
                (same structure as normal/)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..sandbox.result import BacktestResult


# ---------------------------------------------------------------------------
# Directory builders
# ---------------------------------------------------------------------------


def rebalance_dir(output_root: str, rebalance: str) -> Path:
    return Path(output_root) / rebalance


def provider_dir(output_root: str, rebalance: str, provider: str) -> Path:
    return rebalance_dir(output_root, rebalance) / provider


def model_dir(output_root: str, rebalance: str, provider: str, model_name: str) -> Path:
    return provider_dir(output_root, rebalance, provider) / model_name


def run_dir(
    output_root: str, rebalance: str, provider: str, model_name: str, timestamp: str
) -> Path:
    return model_dir(output_root, rebalance, provider, model_name) / timestamp


def profile_dir(
    output_root: str,
    rebalance: str,
    provider: str,
    model_name: str,
    timestamp: str,
    profile: str,
) -> Path:
    return run_dir(output_root, rebalance, provider, model_name, timestamp) / profile


def stress_dir(
    output_root: str,
    rebalance: str,
    provider: str,
    model_name: str,
    timestamp: str,
    profile: str,
    scenario: str,
) -> Path:
    return (
        profile_dir(output_root, rebalance, provider, model_name, timestamp, profile)
        / f"stress_{scenario}"
    )


def normal_dir(
    output_root: str,
    rebalance: str,
    provider: str,
    model_name: str,
    timestamp: str,
    profile: str,
) -> Path:
    return (
        profile_dir(output_root, rebalance, provider, model_name, timestamp, profile)
        / "normal"
    )


def figures_dir(
    output_root: str,
    rebalance: str,
    provider: str,
    model_name: str,
    timestamp: str,
    profile: str,
) -> Path:
    return (
        profile_dir(output_root, rebalance, provider, model_name, timestamp, profile)
        / "figures"
    )


def comparison_figures_dir(output_root: str, rebalance: str) -> Path:
    return rebalance_dir(output_root, rebalance) / "comparison_figures"


def dataset_figures_dir(output_root: str) -> Path:
    return Path(output_root) / "_dataset_figures"


def checkpoint_file(
    output_root: str, rebalance: str, provider: str, model_name: str, timestamp: str
) -> Path:
    return (
        run_dir(output_root, rebalance, provider, model_name, timestamp)
        / "checkpoint.json"
    )


# ---------------------------------------------------------------------------
# Run selection: find best (most complete, then latest) existing timestamp
# ---------------------------------------------------------------------------


def count_completed_profiles(ts_dir: Path, expected_profiles: list[str]) -> int:
    """Count how many expected profiles have at least experiment.log in ts_dir."""
    return sum(1 for p in expected_profiles if (ts_dir / p / "experiment.log").exists())


def get_completed_profiles(ts_dir: Path, expected_profiles: list[str]) -> list[str]:
    """
    Return the subset of expected_profiles recorded as completed in checkpoint.json.

    Only checkpoint.json is authoritative — the experiment.log fallback was removed
    because that file is created at logger init (before the profile runs) and therefore
    cannot reliably signal completion.
    """
    ckpt = ts_dir / "checkpoint.json"
    if not ckpt.exists():
        return []
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    done = set(data.get("completed", []))
    return [p for p in expected_profiles if p in done]


def find_best_run(
    output_root: str,
    rebalance: str,
    provider: str,
    model_name: str,
    expected_profiles: list[str],
) -> Optional[str]:
    """
    Find the most complete (then latest) timestamp directory for a model.

    Returns the timestamp string (e.g. '20260510_042407') or None if no run exists.
    """
    m_dir = model_dir(output_root, rebalance, provider, model_name)
    if not m_dir.exists():
        return None

    best_ts: Optional[str] = None
    best_score = -1

    for child in m_dir.iterdir():
        if not child.is_dir():
            continue
        ts = child.name
        score = count_completed_profiles(child, expected_profiles)
        if score > best_score or (
            score == best_score and (best_ts is None or ts > best_ts)
        ):
            best_score = score
            best_ts = ts

    return best_ts


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_backtest_result(
    result: BacktestResult, out_dir: Path, extra_fields: dict = None
) -> None:
    """Persist BacktestResult fully: JSON + summary + nav/weight CSV + trades."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dict = result.to_dict()
    if extra_fields:
        result_dict.update(extra_fields)
    (out_dir / "backtest_result.json").write_text(
        json.dumps(result_dict, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "summary.txt").write_text(result.summary() + "\n", encoding="utf-8")

    if result.nav_curve is not None and len(result.nav_curve):
        nav_df = result.nav_curve.reset_index()
        nav_df.columns = ["date", "nav"]
        nav_df.to_csv(out_dir / "nav_curve.csv", index=False)
        (
            result.weight_history.reset_index()
            .rename(columns={"index": "date"})
            .to_csv(out_dir / "weight_history.csv", index=False)
        )

    if result.trade_history:
        (out_dir / "trade_history.json").write_text(
            json.dumps(result.trade_history, indent=2, default=str), encoding="utf-8"
        )


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def write_error(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
