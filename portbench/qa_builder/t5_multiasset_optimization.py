"""
T5 – Multi-Asset Optimization
Compute the maximum-Sharpe portfolio weights for 3–6 assets using numerical optimization.
Complexity level 3.
"""

import random
from datetime import date

import numpy as np

from .base import (
    ComplexityLevel, ContextWindow, MarketRegime,
    QABuilder, QAConfig, QAPair, Split,
)
from ..metrics.base import MetricsConfig


class T5MultiAssetOptimization(QABuilder):
    """
    Template T5: Maximum-Sharpe Portfolio Optimization.

    Uses scipy.optimize.minimize with SLSQP to solve:
        max  SR(w) = (w'μ - rf) / sqrt(w'Σw)
        s.t. sum(w) = 1, w_i >= 0

    Mean returns (μ) and covariance matrix (Σ) are estimated from the context
    return history (lookback window only — PiT safe).

    For stability, at least 30 observations are required.
    """

    def __init__(self, provider, config: QAConfig, n_assets: int = 4):
        """
        Args:
            n_assets: Number of assets to include in the optimization (3–6).
        """
        super().__init__(provider, config)
        self.n_assets = max(3, min(6, n_assets))

    @property
    def template_id(self) -> str:
        return "T5"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_3

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        all_classes = ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]
        rng = random.Random(hash(decision_date) + 4)
        chosen_classes = rng.sample(all_classes, min(self.n_assets, len(all_classes)))
        assets = []
        for cls in chosen_classes:
            candidates = self.provider.list_assets(cls)
            if candidates:
                assets.append(rng.choice(candidates))
        if len(assets) < 2:
            fallback = self.provider.list_assets("equities")
            assets += rng.sample(fallback, min(self.n_assets - len(assets), len(fallback)))
        return assets[:self.n_assets]

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        from scipy.optimize import minimize

        assets = context.assets
        d = context.decision_date
        ann = 252

        # Build aligned return matrix
        return_matrix = {}
        for asset in assets:
            r = context.returns_history[asset].dropna()
            return_matrix[asset] = r

        # Find common date index across all assets
        common_idx = None
        for r in return_matrix.values():
            common_idx = r.index if common_idx is None else common_idx.intersection(r.index)

        if common_idx is None or len(common_idx) < 30:
            raise ValueError(f"Insufficient aligned returns for T5 at {d}: {len(common_idx) if common_idx is not None else 0} rows")

        R = np.array([return_matrix[a].reindex(common_idx).values for a in assets]).T
        mu = R.mean(axis=0) * ann          # Annualized mean returns
        cov = np.cov(R.T) * ann            # Annualized covariance matrix
        rf = 0.04                          # Risk-free rate

        n = len(assets)

        def neg_sharpe(w):
            port_ret = w @ mu - rf
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            return -port_ret / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 500})

        if not result.success:
            # Fall back to equal weight if optimizer fails
            w_opt = np.ones(n) / n
        else:
            w_opt = result.x
            w_opt = np.clip(w_opt, 0, 1)
            w_opt = w_opt / w_opt.sum()

        weights = {a: round(float(w), 4) for a, w in zip(assets, w_opt)}

        port_ret = float(w_opt @ mu)
        port_vol = float(np.sqrt(w_opt @ cov @ w_opt))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

        weights_str = ", ".join(f"w_{a}={w:.4f}" for a, w in weights.items())

        context_summary = (
            f"{n}-asset optimization. Max-Sharpe: {sharpe:.3f}. "
            f"Portfolio: return={port_ret:.2%}, vol={port_vol:.2%}. "
            f"Weights: {weights_str}."
        )

        means_str = ", ".join(f"{a}:{mu[i]:.4f}" for i, a in enumerate(assets))
        question = (
            f"Assets: {', '.join(assets)}\n"
            f"Annualized mean returns: {means_str}\n"
            f"Covariance matrix (annualized):\n{np.round(cov, 6).tolist()}\n"
            f"Risk-free rate: {rf:.2%}\n"
            f"Constraints: weights sum to 1, all weights ≥ 0\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Compute the portfolio weights that maximize the Sharpe Ratio for the above assets."
        )

        explanation = (
            f"Solved max-Sharpe via SLSQP numerical optimization.\n"
            f"Optimal weights: {weights_str}\n"
            f"Portfolio annualized return: {port_ret:.2%}, volatility: {port_vol:.2%}\n"
            f"Sharpe ratio: ({port_ret:.4f} - {rf:.4f}) / {port_vol:.4f} = {sharpe:.4f}"
        )

        split = self.config.get_split(d) or Split.TRAIN
        regime = context.market_regime or MarketRegime.SIDEWAYS

        return QAPair(
            qa_id=self._make_id(d, seq),
            template_id=self.template_id,
            complexity=self.complexity,
            split=split,
            market_regime=regime,
            asset_class=self.asset_class,
            assets=assets,
            decision_date=d,
            context_summary=context_summary,
            question=question,
            answer=weights_str,
            answer_numeric=sharpe,
            explanation=explanation,
            metadata={
                "weights": weights,
                "sharpe_ratio": round(sharpe, 4),
                "portfolio_return": round(port_ret, 6),
                "portfolio_vol": round(port_vol, 6),
                "n_assets": n,
                "optimizer_success": bool(result.success),
            },
        )
