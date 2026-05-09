"""
Output path conventions + shared persistence helpers for batch experiments.

Layout (created lazily as experiments run):

  EXPERIMENTS/{batch_id}/
    batch_config.yaml
    batch_summary.json
    errors.jsonl
    {model_label}/
      profile_comparison.json
      {profile}/
        experiment.log
        error.json
        figures/
        stress_{scenario}/
          backtest_result.json
          summary.txt
          nav_curve.csv
          weight_history.csv
          trade_history.json
          snapshots/
          pipeline_logs/
        normal/
          (same structure as stress_*)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..sandbox.result import BacktestResult


def batch_dir(output_root: str, batch_id: str) -> Path:
    return Path(output_root) / batch_id


def model_dir(output_root: str, batch_id: str, model_label: str) -> Path:
    return batch_dir(output_root, batch_id) / model_label


def profile_dir(
    output_root: str, batch_id: str, model_label: str, profile: str
) -> Path:
    return model_dir(output_root, batch_id, model_label) / profile


def stress_dir(
    output_root: str, batch_id: str, model_label: str, profile: str, scenario: str
) -> Path:
    return (
        profile_dir(output_root, batch_id, model_label, profile) / f"stress_{scenario}"
    )


def normal_dir(output_root: str, batch_id: str, model_label: str, profile: str) -> Path:
    return profile_dir(output_root, batch_id, model_label, profile) / "normal"


def figures_dir(
    output_root: str, batch_id: str, model_label: str, profile: str
) -> Path:
    return profile_dir(output_root, batch_id, model_label, profile) / "figures"


def checkpoint_file(output_root: str, batch_id: str) -> Path:
    """Path to checkpoint.json tracking completed (model, profile) pairs."""
    return batch_dir(output_root, batch_id) / "checkpoint.json"


def save_backtest_result(result: BacktestResult, out_dir: Path) -> None:
    """Persist BacktestResult fully: JSON + summary + nav/weight CSV + trades."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
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
