"""
T6 – Rebalancing Decision
Determine whether to rebalance a portfolio given current vs. target weights,
transaction costs, and a threshold rule.
Complexity level 3.
"""

from datetime import date

import numpy as np

from .base import (
    ComplexityLevel,
    ContextWindow,
    MarketRegime,
    QABuilder,
    QAConfig,
    QAPair,
    Split,
)


class T6RebalancingDecision(QABuilder):
    """
    Template T6: Rebalancing Decision.

    Rebalancing is triggered when the maximum absolute weight deviation from
    target exceeds a threshold AND the expected transaction cost is below the
    expected improvement in risk-adjusted return.

    Simplified rule used here:
        Rebalance if max(|w_current_i - w_target_i|) > rebal_threshold

    The answer is "yes" (rebalance) or "no" (hold).
    """

    def __init__(
        self,
        provider,
        config: QAConfig,
        rebal_threshold: float = 0.05,
        transaction_cost_rate: float = 0.001,
    ):
        """
        Args:
            rebal_threshold:      Trigger threshold for max weight deviation (e.g., 0.05 = 5%).
            transaction_cost_rate: Round-trip transaction cost per unit traded (e.g., 0.1%).
        """
        super().__init__(provider, config)
        self.rebal_threshold = rebal_threshold
        self.transaction_cost_rate = transaction_cost_rate

    @property
    def template_id(self) -> str:
        return "T6"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_3

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        import random

        # Always include equities + crypto for text coverage; add 2 more from others.
        text_classes = ["equities", "cryptocurrency"]
        other_classes = ["bonds", "commodities", "real_estate", "cash"]
        rng = random.Random(hash(decision_date) + 5)
        chosen = text_classes + rng.sample(other_classes, 2)
        assets = []
        for cls in chosen:
            candidates = self.provider.list_assets(cls)
            if candidates:
                assets.append(rng.choice(candidates))
        if not assets:
            assets = self.provider.list_assets("equities")[:4]
        return assets

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        assets = context.assets
        d = context.decision_date
        n = len(assets)

        rng = np.random.default_rng(hash(d) + 5)

        # Simulate current weights (drifted from equal-weight target due to price changes)
        target_weights = np.ones(n) / n  # Equal-weight target

        # Drift: apply random recent returns to distort weights
        drifts = rng.normal(0, 0.08, n)  # Simulate asset-level drift
        drifted = target_weights * (1 + drifts)
        current_weights = np.clip(drifted / drifted.sum(), 0, 1)

        max_deviation = float(np.max(np.abs(current_weights - target_weights)))
        total_turnover = float(np.sum(np.abs(current_weights - target_weights)))
        total_cost = total_turnover * self.transaction_cost_rate

        # Decision rule: rebalance if max deviation exceeds threshold
        should_rebalance = max_deviation > self.rebal_threshold
        answer = "yes" if should_rebalance else "no"

        current_str = ", ".join(
            f"w_{a}={current_weights[i]:.4f}" for i, a in enumerate(assets)
        )
        target_str = ", ".join(
            f"w_{a}={target_weights[i]:.4f}" for i, a in enumerate(assets)
        )
        dev_str = ", ".join(
            f"Δ{a}={current_weights[i]-target_weights[i]:+.4f}"
            for i, a in enumerate(assets)
        )

        context_summary = (
            f"Max weight deviation: {max_deviation:.4f}, threshold: {self.rebal_threshold}. "
            f"Decision: {answer} (rebalance)."
        )

        question = (
            f"Current portfolio weights: {current_str}\n"
            f"Target portfolio weights: {target_str}\n"
            f"Weight deviations: {dev_str}\n"
            f"Rebalancing threshold: {self.rebal_threshold:.2%}\n"
            f"Transaction cost rate (round-trip): {self.transaction_cost_rate:.2%}\n"
            f"Estimated total turnover if rebalanced: {total_turnover:.4f}\n"
            f"Estimated transaction cost: {total_cost:.4f}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Should the portfolio be rebalanced? Answer: yes or no."
        )

        explanation = (
            f"Maximum absolute deviation: {max_deviation:.4f}.\n"
            f"Rebalancing threshold: {self.rebal_threshold:.4f}.\n"
            f"{max_deviation:.4f} {'>' if should_rebalance else '<='} {self.rebal_threshold:.4f} "
            f"→ decision: '{answer}'.\n"
            f"If rebalanced: total turnover = {total_turnover:.4f}, "
            f"transaction cost = {total_cost:.6f} (negligible vs. drift)."
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
            answer=answer,
            answer_numeric=float(should_rebalance),
            explanation=explanation,
            metadata={
                "current_weights": {
                    a: round(float(current_weights[i]), 4) for i, a in enumerate(assets)
                },
                "target_weights": {
                    a: round(float(target_weights[i]), 4) for i, a in enumerate(assets)
                },
                "max_deviation": round(max_deviation, 6),
                "total_turnover": round(total_turnover, 6),
                "transaction_cost": round(total_cost, 6),
                "threshold": self.rebal_threshold,
                "should_rebalance": should_rebalance,
            },
        )
