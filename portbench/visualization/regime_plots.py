"""
Asset return distribution by market regime.

Figure 7 — plot_regime_distributions: KDE plots of daily returns per asset class,
    colored by regime (bull/bear/crisis/sideways).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import apply_paper_style, REGIME_COLORS


_ASSET_CLASS_MAP = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "EEM": "Equities",
    "TLT": "Bonds",
    "IEF": "Bonds",
    "LQD": "Bonds",
    "GLD": "Commodities",
    "USO": "Commodities",
    "VNQ": "Real Estate",
    "BTC": "Crypto",
    "ETH": "Crypto",
    "BIL": "Cash",
}

_CLASS_ORDER = ["Equities", "Bonds", "Commodities", "Real Estate", "Crypto", "Cash"]


def plot_regime_distributions(
    return_data: dict[str, dict[str, pd.Series]],
    title: str = "Asset Return Distributions by Market Regime",
    figsize: tuple = (12, 7),
    bandwidth: float = 0.5,
) -> Figure:
    """
    Multi-panel KDE: one subplot per asset class, each regime a colored curve.

    Args:
        return_data: {regime_name: {asset_ticker: pd.Series of daily returns}}
            regime_name ∈ {"bull", "bear", "crisis", "sideways"}
        title:    Figure title.
        figsize:  Figure size.
        bandwidth: KDE bandwidth (std multiplier).

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=False)
    axes = axes.flatten()

    # Group assets by class
    class_assets: dict[str, list[str]] = {cls: [] for cls in _CLASS_ORDER}
    all_assets = set()
    for regime_returns in return_data.values():
        all_assets.update(regime_returns.keys())
    for asset in all_assets:
        cls = _ASSET_CLASS_MAP.get(asset, "Other")
        if cls in class_assets:
            class_assets[cls].append(asset)

    for ax_idx, cls in enumerate(_CLASS_ORDER):
        ax = axes[ax_idx]
        assets = sorted(class_assets.get(cls, []))
        plotted = False

        for regime, regime_returns in return_data.items():
            # Pool returns from all assets in this class for this regime
            pooled = []
            for asset in assets:
                if asset in regime_returns:
                    vals = regime_returns[asset].dropna().values
                    pooled.extend(vals.tolist())
            if not pooled:
                continue

            pooled = np.array(pooled)
            # Clip extreme outliers for display
            p1, p99 = np.percentile(pooled, 1), np.percentile(pooled, 99)
            pooled = pooled[(pooled >= p1) & (pooled <= p99)]

            # KDE via scipy if available, else histogram
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(pooled, bw_method=bandwidth * pooled.std())
                x = np.linspace(pooled.min(), pooled.max(), 200)
                ax.plot(
                    x,
                    kde(x),
                    color=REGIME_COLORS.get(regime, "gray"),
                    linewidth=2,
                    label=regime.capitalize(),
                )
                ax.fill_between(
                    x, kde(x), alpha=0.15, color=REGIME_COLORS.get(regime, "gray")
                )
            except ImportError:
                ax.hist(
                    pooled,
                    bins=40,
                    density=True,
                    alpha=0.4,
                    color=REGIME_COLORS.get(regime, "gray"),
                    label=regime.capitalize(),
                )
            plotted = True

        ax.set_title(cls, fontsize=10, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        if plotted:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def build_regime_data_from_mock(
    n_days: int = 252,
    seed: int = 42,
) -> dict[str, dict[str, pd.Series]]:
    """
    Generate synthetic regime-labeled return data using MockDataProvider.

    Returns:
        {regime: {asset: pd.Series}}
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from portbench.qa_builder.mock_data import MockDataProvider
    from datetime import date, timedelta

    provider = MockDataProvider(seed=seed)
    assets = provider.list_assets()
    start = date(2020, 1, 1)

    regime_data: dict[str, dict[str, pd.Series]] = {
        r: {} for r in ["bull", "bear", "crisis", "sideways"]
    }

    # Sample many dates and bucket by regime
    rng = np.random.default_rng(seed)
    test_dates = [start + timedelta(days=int(d)) for d in range(0, n_days * 3, 3)]

    for d in test_dates:
        try:
            regime = provider.get_regime(d).value.lower()
        except Exception:
            continue
        if regime not in regime_data:
            continue
        for asset in assets:
            try:
                rets = provider.get_return_series(asset, d, d + timedelta(days=30))
                if not rets.empty:
                    if asset not in regime_data[regime]:
                        regime_data[regime][asset] = rets
                    else:
                        regime_data[regime][asset] = pd.concat(
                            [regime_data[regime][asset], rets]
                        ).drop_duplicates()
            except Exception:
                continue

    return regime_data
