"""
T6 – Rebalancing Decision
Determine whether to rebalance and identify the most critical trade.
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
    Template T6: Rebalancing Decision with Trade Identification.

    Two-part question:
      Part A: Should the portfolio be rebalanced?
              Decision rule: rebalance if max(|w_current_i - w_target_i|) > threshold.
      Part B: If rebalancing, which asset has the largest deviation from target
              and what is the required trade size (as fraction of portfolio)?

    Design vs. original:
      - Pre-computed max_deviation and turnover are NOT provided; the model must
        identify the most deviated asset itself (tests analytical ability).
      - Drift is calibrated so that seq % 2 == 0 → rebalance needed,
        seq % 2 == 1 → hold, giving exactly 50/50 balance by construction.
      - The two-part answer tests both decision reasoning and quantitative trade sizing.
    """

    def __init__(
        self,
        provider,
        config: QAConfig,
        rebal_threshold: float = 0.05,
        transaction_cost_rate: float = 0.001,
    ):
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

        rng = np.random.default_rng(abs(hash(d)) + seq)
        target_weights = np.ones(n) / n

        # 50/50 balance by construction: even seq → rebalance case, odd → hold case
        if seq % 2 == 0:
            # Force at least one asset well above threshold
            drifts = rng.normal(0, 0.18, n)
            for _ in range(10):
                drifted = target_weights * (1 + drifts)
                cw = np.clip(drifted / drifted.sum(), 0, 1)
                if np.max(np.abs(cw - target_weights)) > self.rebal_threshold:
                    break
                drifts *= 1.5
        else:
            # Keep all deviations strictly below threshold
            drifts = rng.uniform(
                -self.rebal_threshold * 0.65,
                 self.rebal_threshold * 0.65,
                n,
            )

        drifted = target_weights * (1 + drifts)
        current_weights = np.clip(drifted / drifted.sum(), 0, 1)

        deviations = current_weights - target_weights
        max_dev_idx = int(np.argmax(np.abs(deviations)))
        max_deviation = float(np.abs(deviations[max_dev_idx]))
        should_rebalance = max_deviation > self.rebal_threshold

        primary_asset = assets[max_dev_idx]
        primary_trade = round(float(deviations[max_dev_idx]), 4)
        trade_direction = "sell" if primary_trade > 0 else "buy"
        trade_magnitude = round(abs(primary_trade), 4)

        holdings_str = "\n".join(
            f"  {a}: current={current_weights[i]:.4f}, target={target_weights[i]:.4f}"
            for i, a in enumerate(assets)
        )

        context_summary = (
            f"Max weight deviation: {max_deviation:.4f}, threshold: {self.rebal_threshold}. "
            f"Decision: {'yes' if should_rebalance else 'no'}."
        )

        question = (
            f"Portfolio holdings (current vs. target weights):\n{holdings_str}\n\n"
            f"Rebalancing threshold: {self.rebal_threshold:.2%}\n"
            f"Transaction cost (round-trip): {self.transaction_cost_rate:.2%}\n"
            f"Market regime: "
            f"{context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Part A: Should this portfolio be rebalanced? (yes or no)\n"
            f"Part B: If yes — identify the asset with the largest deviation from its "
            f"target weight and specify the required trade as a fraction of portfolio "
            f"(e.g., 'buy 0.0300 of ASSET' or 'sell 0.0500 of ASSET')."
        )

        if should_rebalance:
            answer = f"yes; {trade_direction} {trade_magnitude:.4f} of {primary_asset}"
        else:
            answer = "no"

        explanation = (
            "Weight deviations: "
            + ", ".join(f"{a}={deviations[i]:+.4f}" for i, a in enumerate(assets))
            + f"\nMax |deviation|={max_deviation:.4f} vs threshold={self.rebal_threshold}.\n"
            + (
                f"Rebalance: {trade_direction} {trade_magnitude:.4f} of {primary_asset}."
                if should_rebalance
                else "All deviations within threshold — hold."
            )
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
                "current_weights": {a: round(float(current_weights[i]), 4) for i, a in enumerate(assets)},
                "target_weights": {a: round(float(target_weights[i]), 4) for i, a in enumerate(assets)},
                "max_deviation": round(max_deviation, 6),
                "threshold": self.rebal_threshold,
                "should_rebalance": should_rebalance,
                "primary_asset": primary_asset,
                "primary_trade": primary_trade,
            },
        )
