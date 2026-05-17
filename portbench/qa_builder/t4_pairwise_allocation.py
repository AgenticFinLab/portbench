"""
T4 – Pairwise Allocation (Constrained)
Compute the minimum-variance portfolio weights for two assets subject to a
minimum expected return constraint.
Complexity level 2.
"""

import random
from datetime import date

import numpy as np

from .base import (
    ComplexityLevel,
    ContextWindow,
    MarketRegime,
    QABuilder,
    QAPair,
    Split,
)


class T4PairwiseAllocation(QABuilder):
    """
    Template T4: Pairwise Constrained Allocation.

    Solve:
        min  w'Σw
        s.t. w'μ ≥ μ_floor,  sum(w)=1,  w≥0

    When the constraint is non-binding → solution is the unconstrained min-var portfolio.
    When the constraint is binding    → solution is on the iso-return line w'μ = μ_floor.

    The explicit covariance/correlation values are withheld from the question;
    only individual volatilities and mean returns are provided. The model must
    reason about the risk-return tradeoff without a plug-in formula.
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
        text_classes = ["equities", "cryptocurrency"]
        other_classes = ["bonds", "commodities", "real_estate", "cash"]
        rng = random.Random(hash(decision_date) + 3)
        cls1 = rng.choice(text_classes)
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

        aligned = np.array([r1, r2]).T
        mask = ~(np.isnan(aligned[:, 0]) | np.isnan(aligned[:, 1]))
        aligned = aligned[mask]

        if len(aligned) < 10:
            raise ValueError(f"Insufficient aligned history for T4: {a1}/{a2} at {d}")

        s1 = float(aligned[:, 0].std())
        s2 = float(aligned[:, 1].std())
        cov_12 = float(np.cov(aligned[:, 0], aligned[:, 1])[0, 1])
        mu1 = float(aligned[:, 0].mean() * 252)  # annualized
        mu2 = float(aligned[:, 1].mean() * 252)

        # Unconstrained min-variance solution
        denom = s1**2 + s2**2 - 2 * cov_12
        if abs(denom) < 1e-12:
            w1_mv, w2_mv = 0.5, 0.5
        else:
            w1_mv = max(0.0, min(1.0, (s2**2 - cov_12) / denom))
            w2_mv = 1.0 - w1_mv
        mv_return = w1_mv * mu1 + w2_mv * mu2

        # Set return floor: alternate binding / non-binding by seq parity
        rng = np.random.default_rng(abs(hash(d)) + seq)
        mu_max = max(mu1, mu2)
        mu_min = min(mu1, mu2)
        binding = (seq % 2 == 0) and (mu_max > mv_return)
        if binding:
            # Floor strictly above the unconstrained return → constraint must be active
            mu_floor = float(rng.uniform(
                mv_return + 1e-6,
                mu_max - 1e-6 if mu_max > mv_return + 2e-6 else mv_return + 1e-4
            ))
        else:
            # Floor at or below mv_return → constraint satisfied by min-var itself
            mu_floor = float(rng.uniform(
                mu_min,
                mv_return,
            ))
        mu_floor = round(mu_floor, 4)

        # GT: constrained optimal weights
        if binding and abs(mu1 - mu2) > 1e-9:
            # On the iso-return hyperplane: w1*mu1 + (1-w1)*mu2 = mu_floor
            w1 = (mu_floor - mu2) / (mu1 - mu2)
            w1 = max(0.0, min(1.0, w1))
        else:
            w1 = w1_mv
        w2 = 1.0 - w1
        w1, w2 = round(w1, 4), round(w2, 4)

        portfolio_return = round(w1 * mu1 + w2 * mu2, 4)
        portfolio_vol = round(float(np.sqrt(
            w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * cov_12
        )), 6)

        context_summary = (
            f"{a1} σ={s1:.4f} μ={mu1:.4f}, {a2} σ={s2:.4f} μ={mu2:.4f}. "
            f"Return floor={mu_floor:.4f}. "
            f"Optimal: w_{a1}={w1:.4f}, w_{a2}={w2:.4f}."
        )

        question = (
            f"Assets: {a1}, {a2}\n"
            f"{a1}: annualized_mean_return={mu1:.4f}, daily_std={s1:.4f}\n"
            f"{a2}: annualized_mean_return={mu2:.4f}, daily_std={s2:.4f}\n"
            f"Minimum required portfolio return (annualized): {mu_floor:.4f}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Compute portfolio weights (w_{a1}, w_{a2}) that minimize portfolio variance "
            f"while satisfying the minimum return constraint. "
            f"Constraints: all weights ≥ 0, weights sum to 1. "
            f"Report as: w_{a1}=X.XXXX, w_{a2}=X.XXXX"
        )

        explanation = (
            f"Unconstrained min-var: w_{a1}={w1_mv:.4f}, w_{a2}={w2_mv:.4f}, "
            f"expected_return={mv_return:.4f}.\n"
            f"Return floor: {mu_floor:.4f}. "
            f"Constraint {'binding' if binding else 'non-binding'}.\n"
            f"Optimal: w_{a1}={w1:.4f}, w_{a2}={w2:.4f}, "
            f"portfolio_return={portfolio_return:.4f}, portfolio_vol={portfolio_vol:.6f}."
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
            answer_numeric=w1,
            explanation=explanation,
            metadata={
                "weights": {a1: w1, a2: w2},
                "mu_floor": mu_floor,
                "constraint_binding": binding,
                "mv_return": round(mv_return, 6),
                "portfolio_return": portfolio_return,
                "sigma_1": round(s1, 6),
                "sigma_2": round(s2, 6),
                "covariance": round(float(cov_12), 6),
            },
        )


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
        denom = s1**2 + s2**2 - 2 * cov_12
        if abs(denom) < 1e-12:
            w1, w2 = 0.5, 0.5  # Degenerate: equal weights
        else:
            w1 = (s2**2 - cov_12) / denom
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
