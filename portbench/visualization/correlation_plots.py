"""
Cross-asset correlation visualization.

Three figures, each surfacing a different layer of the multi-asset correlation
structure that PortBench's S3 score depends on:

  plot_correlation_heatmap        — full asset × asset matrix, grouped & boxed
                                    by asset class so intra/inter blocks are
                                    visually distinct.
  plot_inter_class_correlation    — asset-class × asset-class matrix (averaged
                                    cross-class pairs) + per-class intra-class
                                    average correlation bar chart.
  plot_correlation_evolution      — rolling cross-asset / inter-class
                                    correlation through a backtest, built from
                                    per-rebalance MarketSnapshot dumps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import apply_paper_style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_by_class(
    corr: pd.DataFrame, asset_class_map: dict[str, str]
) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """
    Reorder corr rows/cols so assets in the same class are contiguous.
    Returns (reordered_corr, [(class, start_idx, end_idx_exclusive), ...]).
    """
    classes = sorted({asset_class_map.get(a, "_unknown") for a in corr.columns})
    order: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for ac in classes:
        members = [a for a in corr.columns if asset_class_map.get(a, "_unknown") == ac]
        if not members:
            continue
        start = len(order)
        order.extend(members)
        spans.append((ac, start, len(order)))
    reordered = corr.loc[order, order]
    return reordered, spans


def _intra_class_avgs(
    corr: pd.DataFrame, asset_class_map: dict[str, str]
) -> dict[str, float]:
    """Average off-diagonal correlation within each class with >= 2 members."""
    groups: dict[str, list[str]] = {}
    for a in corr.columns:
        ac = asset_class_map.get(a)
        if ac:
            groups.setdefault(ac, []).append(a)
    out = {}
    for ac, members in groups.items():
        if len(members) < 2:
            continue
        sub = corr.loc[members, members].values
        off = sub[~np.eye(len(members), dtype=bool)]
        if len(off):
            out[ac] = float(np.nanmean(off))
    return out


def _inter_class_matrix(
    corr: pd.DataFrame, asset_class_map: dict[str, str]
) -> pd.DataFrame:
    """class × class matrix from averaged cross-class pairwise correlations."""
    groups: dict[str, list[str]] = {}
    for a in corr.columns:
        ac = asset_class_map.get(a)
        if ac:
            groups.setdefault(ac, []).append(a)
    classes = sorted(groups)
    if len(classes) < 2:
        return pd.DataFrame()
    out = pd.DataFrame(index=classes, columns=classes, dtype=float)
    for ci in classes:
        for cj in classes:
            vals = []
            for a in groups[ci]:
                for b in groups[cj]:
                    if a == b:
                        continue
                    vals.append(float(corr.loc[a, b]))
            out.loc[ci, cj] = float(np.nanmean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# 1. Full asset × asset heatmap, grouped by class
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    asset_class_map: Optional[dict[str, str]] = None,
    *,
    annotate: bool = False,
    figsize: tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """
    Asset × asset Pearson correlation heatmap.

    When asset_class_map is provided, rows/cols are reordered so each class is
    contiguous and a coloured rectangle is drawn around each intra-class block;
    this makes the intra-class (within rectangles) vs inter-class (off-block)
    structure read at a glance.
    """
    apply_paper_style()
    if correlation_matrix.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        ax.axis("off")
        return fig

    if asset_class_map:
        cm, spans = _group_by_class(correlation_matrix, asset_class_map)
    else:
        cm, spans = correlation_matrix, []

    # Mask upper triangle (keep only lower triangle + diagonal)
    mask = np.triu(np.ones_like(cm.values, dtype=bool), k=1)
    masked = np.ma.masked_where(mask, cm.values)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=4)
    ax.set_yticklabels(cm.index, fontsize=4)
    ax.tick_params(axis="both", which="major", pad=8)

    if annotate and len(cm) <= 20:
        for i in range(len(cm)):
            for j in range(len(cm)):
                v = cm.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{v:+.2f}",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black",
                    fontsize=6,
                )

    # Draw bottom + left edges only (lower-triangle portion of each class block)
    for ac, start, end in spans:
        n = end - start
        x0, y0 = start - 0.5, start - 0.5
        x1, y1 = end - 0.5, end - 0.5
        # Bottom edge
        ax.plot([x0, x1], [y1, y1], color="#1e3d6e", linewidth=1.6)
        # Left edge
        ax.plot([x0, x0], [y0, y1], color="#1e3d6e", linewidth=1.6)
        # Class label at the midpoint of the diagonal (hypotenuse), offset to the right
        mid_x = start + n / 2 - 0.5
        mid_y = start + n / 2 - 0.5
        ax.text(
            mid_x + 0.6, mid_y, ac,
            ha="left", va="center", fontsize=8,
            color="#1e3d6e", fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Inter-class matrix + intra-class bars
# ---------------------------------------------------------------------------


def plot_inter_class_correlation(
    correlation_matrix: pd.DataFrame,
    asset_class_map: dict[str, str],
    *,
    figsize: tuple[float, float] = (11.0, 4.5),
) -> Figure:
    """
    Two-panel figure mirroring the S3 score's two 15% components:

      Left  — class × class heatmap (averaged cross-class pairwise corr).
              Off-diagonal entries drive the *inter-class hedging* term.
      Right — bar chart of each class's average internal correlation.
              Drives the *intra-class concentration penalty* term.
    """
    apply_paper_style()
    inter = _inter_class_matrix(correlation_matrix, asset_class_map)
    intra = _intra_class_avgs(correlation_matrix, asset_class_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1.0]})

    if inter.empty:
        ax1.text(0.5, 0.5, "No inter-class data", ha="center", va="center")
        ax1.axis("off")
    else:
        im = ax1.imshow(inter.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax1.set_xticks(range(len(inter.columns)))
        ax1.set_yticks(range(len(inter.index)))
        ax1.set_xticklabels(inter.columns, rotation=45, ha="right", fontsize=8)
        ax1.set_yticklabels(inter.index, fontsize=8)
        for i in range(len(inter)):
            for j in range(len(inter)):
                v = inter.values[i, j]
                if np.isnan(v):
                    continue
                ax1.text(
                    j, i, f"{v:+.2f}",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.04)

    if not intra:
        ax2.text(0.5, 0.5, "No intra-class data", ha="center", va="center")
        ax2.axis("off")
    else:
        classes = sorted(intra)
        vals = [intra[c] for c in classes]
        colors = ["#4a6fa5" if v < 0.6 else "#e67e22" for v in vals]
        ax2.barh(classes, vals, color=colors, edgecolor="#1e3d6e", linewidth=0.8)
        ax2.axvline(0.0, color="#888", linewidth=0.6)
        ax2.set_xlim(-0.2, 1.05)
        ax2.set_xlabel("Avg pairwise correlation", fontsize=9)
        for i, v in enumerate(vals):
            ax2.text(v + 0.02, i, f"{v:+.2f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Correlation evolution from per-rebalance snapshots
# ---------------------------------------------------------------------------


def _load_snapshot_returns(snapshot_dir: Path) -> Optional[pd.DataFrame]:
    """
    Snapshots dumped by BacktestEngine carry only trailing-return scalars,
    not full series, so we approximate cross-asset correlation via the
    cross-snapshot panel of trailing returns. This is coarse but reflects
    the trajectory of correlation regime as time advances.
    """
    rows = []
    for f in sorted(snapshot_dir.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = payload.get("decision_date")
        tr = payload.get("trailing_returns") or {}
        if d and tr:
            row = {"date": d, **tr}
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def plot_correlation_evolution(
    snapshot_dir: Path,
    asset_class_map: Optional[dict[str, str]] = None,
    *,
    window: int = 6,
    title: str = "Correlation Evolution",
    figsize: tuple[float, float] = (10.0, 5.0),
) -> Optional[Figure]:
    """
    Rolling cross-asset average correlation through a backtest run.

    Reads per-rebalance JSON snapshots from snapshot_dir (written by
    `BacktestEngine(snapshot_dump_dir=...)`) and plots:

      Top    — overall cross-asset average correlation (rolling window of
               `window` rebalance dates).
      Bottom — per-class internal average correlation, when an asset_class_map
               is provided.

    Returns None if the snapshot directory is empty / unreadable.
    """
    apply_paper_style()
    panel = _load_snapshot_returns(snapshot_dir)
    if panel is None or panel.shape[0] < window or panel.shape[1] < 2:
        return None

    rolling_avg = []
    intra_rolling: dict[str, list[float]] = {}
    classes_present: list[str] = []
    if asset_class_map:
        classes_present = sorted({asset_class_map.get(a) for a in panel.columns if asset_class_map.get(a)})
        intra_rolling = {c: [] for c in classes_present}
    dates: list[pd.Timestamp] = []

    for end in range(window, len(panel) + 1):
        win = panel.iloc[end - window : end]
        cm = win.corr()
        n = cm.shape[0]
        if n < 2:
            continue
        off = cm.values[~np.eye(n, dtype=bool)]
        rolling_avg.append(float(np.nanmean(off)))
        dates.append(panel.index[end - 1])
        for c in classes_present:
            members = [a for a in cm.columns if asset_class_map.get(a) == c]
            if len(members) < 2:
                intra_rolling[c].append(float("nan"))
                continue
            sub = cm.loc[members, members].values
            sub_off = sub[~np.eye(len(members), dtype=bool)]
            intra_rolling[c].append(float(np.nanmean(sub_off)))

    if not rolling_avg:
        return None

    rows = 2 if intra_rolling else 1
    fig, axes = plt.subplots(rows, 1, figsize=figsize, sharex=True)
    if rows == 1:
        axes = [axes]

    axes[0].plot(dates, rolling_avg, color="#1e3d6e", linewidth=1.8, marker="o", markersize=3)
    axes[0].axhline(0.0, color="#888", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel(f"Avg pairwise corr\n(rolling {window} rebalances)", fontsize=9)
    axes[0].set_title(title)
    axes[0].set_ylim(-0.2, 1.0)

    if intra_rolling:
        palette = ["#4a6fa5", "#e67e22", "#1abc9c", "#9b59b6", "#e74c3c", "#7a9fc5"]
        for i, (c, ys) in enumerate(intra_rolling.items()):
            axes[1].plot(
                dates, ys, label=c, color=palette[i % len(palette)],
                linewidth=1.4, marker=".", markersize=3,
            )
        axes[1].axhline(0.0, color="#888", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel("Intra-class corr", fontsize=9)
        axes[1].set_xlabel("Rebalance date", fontsize=9)
        axes[1].set_ylim(-0.2, 1.05)
        axes[1].legend(fontsize=7, ncol=min(len(intra_rolling), 4), loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: load processed-dir artifacts
# ---------------------------------------------------------------------------


def load_processed_correlation(
    processed_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[dict[str, str]]]:
    """
    Load datasets/processed/{correlation_matrix.csv, asset_class_map.json}.
    Returns (corr_df, asset_class_map). Either can be None if missing.
    """
    corr_path = processed_dir / "correlation_matrix.csv"
    map_path = processed_dir / "asset_class_map.json"
    corr = (
        pd.read_csv(corr_path, index_col=0)
        if corr_path.exists()
        else None
    )
    amap = (
        json.loads(map_path.read_text(encoding="utf-8"))
        if map_path.exists()
        else None
    )
    return corr, amap
