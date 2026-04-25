"""
CEPS vs PnL Correlation Report for PortBench Sandbox runs.

Reads a completed sandbox run directory, extracts per-rebalance CEPS scores
from pipeline_logs, aligns them with the realized return for the subsequent
period, then computes the Pearson correlation and saves diagnostic charts.

Usage:
    python examples/sandbox/ceps_pnl_report.py <run_dir>

    # Example:
    python examples/sandbox/ceps_pnl_report.py \\
        outputs/evaluation_results/sandbox_results/20260422_142246_qwen-plus

Output (written into <run_dir>/):
    ceps_pnl_report.json   — per-period table + Pearson r + p-value
    ceps_pnl_scatter.png   — scatter plot: CEPS vs next-period return
    ceps_pnl_timeseries.png — dual-axis: CEPS and cumulative return over time
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.metrics.ceps import CEPS, StageScore


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_episodes(run_dir: Path) -> list[dict]:
    """
    Load all episode JSON files from pipeline_logs subdirectory.

    Searches for pipeline_logs/{inner_run_id}/episodes/*.json recursively.
    Returns list of raw episode dicts, sorted by decision_date.
    """
    episodes = []
    for ep_file in sorted(run_dir.glob("pipeline_logs/**/episodes/*.json")):
        with open(ep_file) as f:
            episodes.append(json.load(f))
    episodes.sort(key=lambda e: e["decision_date"])
    return episodes


def _compute_ceps_for_episode(ep: dict) -> float:
    """
    Recompute CEPS from raw stage scores stored in the episode log.

    Uses portbench.metrics.ceps.CEPS — same formula as the live evaluation.
    """
    stage_scores = [
        StageScore(
            stage_id=s["stage_id"],
            stage_name=s["stage_id"],
            score=float(s.get("score", 0.0)),
        )
        for s in ep.get("stages", [])
    ]
    result = CEPS().compute(stage_scores)
    return result.ceps_score


def _load_nav_curve(run_dir: Path) -> pd.Series:
    """
    Load nav_curve.csv and return a date-indexed NAV series.
    """
    nav_path = run_dir / "nav_curve.csv"
    df = pd.read_csv(nav_path, parse_dates=["date"])
    df = df.set_index("date")["nav"].sort_index()
    return df


# ---------------------------------------------------------------------------
# Alignment: map each rebalance date to next-period return
# ---------------------------------------------------------------------------

def _build_aligned_table(episodes: list[dict], nav: pd.Series) -> pd.DataFrame:
    """
    Build a DataFrame aligning each rebalance's CEPS score with the
    realized return of the subsequent holding period.

    The "next-period return" is defined as the return from this rebalance
    date to the next rebalance date (or end of backtest for the last period).

    Returns a DataFrame with columns:
        decision_date   : rebalance date
        ceps            : CEPS score [0, 1]
        s1, s2, s3      : individual stage scores
        period_return   : realized return over the subsequent holding period
    """
    rebalance_dates = [pd.Timestamp(ep["decision_date"]) for ep in episodes]

    rows = []
    for i, ep in enumerate(episodes):
        d = pd.Timestamp(ep["decision_date"])
        ceps = _compute_ceps_for_episode(ep)

        # Extract individual stage scores
        stage_map = {s["stage_id"]: float(s.get("score", 0.0)) for s in ep.get("stages", [])}

        # Next period end: next rebalance date or last NAV date
        if i + 1 < len(rebalance_dates):
            period_end = rebalance_dates[i + 1]
        else:
            period_end = nav.index[-1]

        # Realized return: NAV at period_end / NAV at rebalance date - 1
        nav_start = nav.asof(d)
        nav_end = nav.asof(period_end)
        if nav_start and nav_start > 0:
            period_ret = (nav_end - nav_start) / nav_start
        else:
            period_ret = float("nan")

        rows.append({
            "decision_date": d,
            "ceps": ceps,
            "s1": stage_map.get("S1", float("nan")),
            "s2": stage_map.get("S2", float("nan")),
            "s3": stage_map.get("S3", float("nan")),
            "s4": stage_map.get("S4", float("nan")),
            "s5": stage_map.get("S5", float("nan")),
            "period_return": period_ret,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_correlation(df: pd.DataFrame) -> dict:
    """
    Compute Pearson r (and p-value) between CEPS and next-period return.

    Returns a dict with r, p_value, n, and an interpretation string.
    Requires at least 3 valid paired observations.
    """
    valid = df[["ceps", "period_return"]].dropna()
    n = len(valid)
    if n < 3:
        return {
            "pearson_r": None,
            "p_value": None,
            "n_observations": n,
            "note": f"Insufficient data ({n} periods). Need ≥3 rebalances for correlation.",
        }

    r, p = stats.pearsonr(valid["ceps"], valid["period_return"])
    if abs(r) < 0.2:
        interp = "negligible"
    elif abs(r) < 0.4:
        interp = "weak"
    elif abs(r) < 0.6:
        interp = "moderate"
    else:
        interp = "strong"

    direction = "positive" if r > 0 else "negative"
    return {
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "n_observations": n,
        "interpretation": f"{interp} {direction} correlation",
        "significant_at_05": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_scatter(df: pd.DataFrame, corr: dict, out_path: Path) -> None:
    """Scatter plot: CEPS score (x) vs next-period return (y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[["ceps", "period_return", "decision_date"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(valid["ceps"], valid["period_return"] * 100,
               color="#4a6fa5", s=80, alpha=0.8, zorder=3)

    # Annotate each point with its date
    for _, row in valid.iterrows():
        ax.annotate(
            row["decision_date"].strftime("%b %Y"),
            (row["ceps"], row["period_return"] * 100),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color="#1e3d6e",
        )

    # OLS trend line (only if ≥3 points)
    if len(valid) >= 3:
        m, b = np.polyfit(valid["ceps"], valid["period_return"] * 100, 1)
        xs = np.linspace(valid["ceps"].min(), valid["ceps"].max(), 100)
        ax.plot(xs, m * xs + b, "--", color="#1e3d6e", linewidth=1.5,
                label=f"OLS fit (r={corr.get('pearson_r', 'n/a')})")
        ax.legend(fontsize=9)

    ax.axhline(0, color="#c0c0c0", linewidth=0.8, linestyle=":")
    ax.set_xlabel("CEPS Score (decision quality)", fontsize=11)
    ax.set_ylabel("Next-Period Return (%)", fontsize=11)
    ax.set_title("CEPS vs Realized Next-Period Return", fontsize=13, fontweight="bold")

    r_str = str(corr.get("pearson_r", "n/a"))
    p_str = str(corr.get("p_value", "n/a"))
    note = corr.get("interpretation", corr.get("note", ""))
    ax.text(0.02, 0.97, f"Pearson r = {r_str}  (p = {p_str})\n{note}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4e4f7", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter plot → {out_path}")


def _plot_timeseries(df: pd.DataFrame, nav: pd.Series, out_path: Path) -> None:
    """Dual-axis time series: CEPS per rebalance + cumulative NAV return."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[["decision_date", "ceps"]].dropna()
    cumret = (nav / nav.iloc[0] - 1) * 100  # cumulative return in %

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: cumulative return
    ax1.fill_between(cumret.index, cumret.values, alpha=0.15, color="#4a6fa5")
    ax1.plot(cumret.index, cumret.values, color="#4a6fa5", linewidth=1.5,
             label="Cumulative Return (%)")
    ax1.set_ylabel("Cumulative Return (%)", color="#4a6fa5", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#4a6fa5")

    # Right axis: CEPS per rebalance
    ax2 = ax1.twinx()
    ax2.bar(valid["decision_date"], valid["ceps"],
            width=12, alpha=0.6, color="#1e3d6e", label="CEPS Score")
    ax2.set_ylabel("CEPS Score", color="#1e3d6e", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#1e3d6e")
    ax2.set_ylim(0, 1.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("CEPS Score per Rebalance vs Portfolio Cumulative Return",
                  fontsize=13, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved time series plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(run_dir: Path) -> None:
    print(f"Loading run: {run_dir}")

    episodes = _load_episodes(run_dir)
    if not episodes:
        print("  ERROR: No episode logs found under pipeline_logs/. "
              "Was the sandbox run with --no-pipeline?")
        sys.exit(1)
    print(f"  Episodes found: {len(episodes)}")

    nav = _load_nav_curve(run_dir)
    df = _build_aligned_table(episodes, nav)
    corr = _compute_correlation(df)

    # --- Print summary ---
    print("\nPer-rebalance table:")
    print(df[["decision_date", "ceps", "s1", "s2", "s3", "period_return"]].to_string(index=False))
    print(f"\nCorrelation: Pearson r = {corr.get('pearson_r', 'n/a')}, "
          f"p = {corr.get('p_value', 'n/a')}")
    if "note" in corr:
        print(f"  Note: {corr['note']}")

    # --- Save JSON report ---
    report = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(),
        "n_rebalances": len(episodes),
        "correlation": corr,
        "per_period": [
            {k: (v.isoformat() if hasattr(v, "isoformat") else
                 (None if (isinstance(v, float) and np.isnan(v)) else v))
             for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ],
    }
    report_path = run_dir / "ceps_pnl_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved report → {report_path}")

    # --- Charts --- saved to run_dir AND mirrored to figures/
    try:
        fig_dir = Path("figures") / "sandbox" / run_dir.name
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_scatter(df, corr, run_dir / "ceps_pnl_scatter.png")
        _plot_scatter(df, corr, fig_dir / "ceps_pnl_scatter.png")
        _plot_timeseries(df, nav, run_dir / "ceps_pnl_timeseries.png")
        _plot_timeseries(df, nav, fig_dir / "ceps_pnl_timeseries.png")
    except ImportError:
        print("  matplotlib not installed; skipping charts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEPS vs PnL correlation report")
    parser.add_argument("run_dir", type=str,
                        help="Path to sandbox run directory (contains nav_curve.csv + pipeline_logs/)")
    args = parser.parse_args()
    generate_report(Path(args.run_dir))
