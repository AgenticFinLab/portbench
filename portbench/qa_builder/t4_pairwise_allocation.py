"""
T4 – Pairwise Allocation
Compute the minimum-variance portfolio weights for two assets.
Complexity level 2.
"""

import random
from datetime import date

import numpy as np

from .base import (
    ComplexityLevel, ContextWindow, MarketRegime,
    QABuilder, QAConfig, QAPair, Split,
)


class T4PairwiseAllocation(QABuilder):
    """
    Template T4: Pairwise Minimum-Variance Allocation.

    Analytic solution for two-asset minimum-variance portfolio:
        w1* = (σ2² - σ12) / (σ1² + σ2² - 2*σ12)
        w2* = 1 - w1*

    where σ12 = ρ * σ1 * σ2.

    If the unconstrained solution gives a negative weight (short), it is
    clamped to 0 and the other asset gets weight 1 (long-only constraint).
    """

    @property
    def template_id(self) -> str:
        return "T4"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_2

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        # Always include at least one text-bearing class (equities or crypto)
        text_classes = ["equities", "cryptocurrency"]
        other_classes = ["bonds", "commodities", "real_estate", "cash"]
        rng = random.Random(hash(decision_date) + 3)
        cls1 = rng.choice(text_classes)
        # Second class: 50% another text class, 50% from non-text classes
        if rng.random() < 0.5:
            cls2 = rng.choice([c for c in text_classes if c != cls1] + other_classes)
        else:
            cls2 = rng.choice(other_classes)
        c1 = self.provider.list_assets(cls1) or self.provider.list_assets("equities")
        c2 = self.provider.list_assets(cls2) or self.provider.list_assets("bonds")
        return [rng.choice(c1), rng.choice(c2)]

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        a1, a2 = context.assets[0], context.assets[1]
        d = context.decision_date

        r1 = context.returns_history[a1].dropna()
        r2 = context.returns_history[a2].dropna()

        # Align on common dates
        aligned = np.array([r1, r2]).T
        # Use only rows where both are available
        mask = ~(np.isnan(aligned[:, 0]) | np.isnan(aligned[:, 1]))
        aligned = aligned[mask]

        if len(aligned) < 10:
            raise ValueError(f"Insufficient aligned history for T4: {a1}/{a2} at {d}")

        s1 = aligned[:, 0].std()
        s2 = aligned[:, 1].std()
        cov_12 = np.cov(aligned[:, 0], aligned[:, 1])[0, 1]
        corr = cov_12 / (s1 * s2) if s1 * s2 > 0 else 0.0

        # Minimum-variance weights (unconstrained)
        denom = s1 ** 2 + s2 ** 2 - 2 * cov_12
        if abs(denom) < 1e-12:
            w1, w2 = 0.5, 0.5  # Degenerate: equal weights
        else:
            w1 = (s2 ** 2 - cov_12) / denom
            w2 = 1.0 - w1

        # Long-only constraint: clamp and renormalize
        w1 = max(0.0, w1)
        w2 = max(0.0, w2)
        total = w1 + w2
        if total > 0:
            w1, w2 = w1 / total, w2 / total
        else:
            w1, w2 = 0.5, 0.5

        w1, w2 = round(w1, 4), round(w2, 4)

        context_summary = (
            f"{a1} σ={s1:.4f}, {a2} σ={s2:.4f}, ρ={corr:.3f}. "
            f"Min-variance weights: {a1}={w1:.3f}, {a2}={w2:.3f}."
        )

        question = (
            f"Assets: {a1}, {a2}\n"
            f"{a1} – std={s1:.4f}, mean={aligned[:,0].mean():.4f}\n"
            f"{a2} – std={s2:.4f}, mean={aligned[:,1].mean():.4f}\n"
            f"Covariance({a1},{a2}) = {cov_12:.6f}, Correlation = {corr:.3f}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Compute the minimum-variance portfolio weights for {a1} and {a2} "
            f"(long-only: weights ≥ 0, sum to 1). Report as w_{a1}, w_{a2}."
        )

        explanation = (
            f"Analytic min-variance formula:\n"
            f"  w1* = (σ2² - σ12) / (σ1² + σ2² - 2σ12)\n"
            f"  = ({s2**2:.6f} - {cov_12:.6f}) / ({s1**2:.6f} + {s2**2:.6f} - {2*cov_12:.6f})\n"
            f"  Unconstrained: w_{a1}={((s2**2-cov_12)/denom if abs(denom)>1e-12 else 0.5):.4f}\n"
            f"  After long-only clamp: w_{a1}={w1:.4f}, w_{a2}={w2:.4f}."
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
            assets=[a1, a2],
            decision_date=d,
            context_summary=context_summary,
            question=question,
            answer=f"w_{a1}={w1:.4f}, w_{a2}={w2:.4f}",
            answer_numeric=w1,  # Primary answer: weight of first asset
            explanation=explanation,
            metadata={
                "weights": {a1: w1, a2: w2},
                "sigma_1": round(s1, 6),
                "sigma_2": round(s2, 6),
                "covariance": round(float(cov_12), 6),
                "correlation": round(corr, 4),
            },
        )
