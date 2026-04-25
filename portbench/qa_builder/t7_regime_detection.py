"""
T7 – Regime Detection
Identify the current market regime (bull/bear/sideways/crisis) and recommend
an allocation adjustment direction for each asset class.
Complexity level 4 (full portfolio).
"""

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


# Canonical allocation adjustments per regime.
# Values are directional shifts from a neutral equal-weight baseline.
# "increase" / "decrease" / "neutral" are the answer vocabulary.
_REGIME_ALLOCATION: dict[MarketRegime, dict[str, str]] = {
    MarketRegime.BULL: {
        "equities": "increase",
        "bonds": "decrease",
        "commodities": "neutral",
        "real_estate": "increase",
        "cryptocurrency": "increase",
        "cash": "decrease",
    },
    MarketRegime.BEAR: {
        "equities": "decrease",
        "bonds": "increase",
        "commodities": "neutral",
        "real_estate": "decrease",
        "cryptocurrency": "decrease",
        "cash": "increase",
    },
    MarketRegime.SIDEWAYS: {
        "equities": "neutral",
        "bonds": "neutral",
        "commodities": "neutral",
        "real_estate": "neutral",
        "cryptocurrency": "decrease",
        "cash": "increase",
    },
    MarketRegime.CRISIS: {
        "equities": "decrease",
        "bonds": "increase",
        "commodities": "increase",  # Gold as safe haven
        "real_estate": "decrease",
        "cryptocurrency": "decrease",
        "cash": "increase",
    },
}


class T7RegimeDetection(QABuilder):
    """
    Template T7: Regime Detection and Allocation Adjustment.

    Two-part question:
      Part A: Identify the market regime from price and macro data.
      Part B: For each of the six asset classes, recommend increasing,
              decreasing, or maintaining the current allocation.

    Ground truth:
      - Regime: from DataProvider.get_regime()
      - Allocation direction: from the _REGIME_ALLOCATION lookup table
        (canonical expert rules, consistent with portfolio theory)
    """

    @property
    def template_id(self) -> str:
        return "T7"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_4

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        # T7 uses a single equity (SPY/index) as the regime indicator asset
        return self.provider.list_assets("equities")[:1]

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        d = context.decision_date
        regime = context.market_regime or MarketRegime.SIDEWAYS
        asset = context.assets[0]

        # Compute trailing return statistics for context
        prices = context.price_history[asset]
        returns = context.returns_history[asset].dropna()
        trailing_return = (
            float(prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) > 1 else 0.0
        )
        vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        # Allocation recommendations from expert rules
        allocation = _REGIME_ALLOCATION[regime]
        alloc_str = ", ".join(
            f"{cls}: {direction}" for cls, direction in allocation.items()
        )

        # Format macro context
        macro_str = ", ".join(f"{k}={v:.4f}" for k, v in context.macro_context.items())

        context_summary = (
            f"Regime: {regime.value}. {asset} trailing return: {trailing_return:+.1%}, "
            f"annualized vol: {vol:.1%}. Allocation: {alloc_str}."
        )

        question = (
            f"Market indicator: {asset}\n"
            f"Price history ({self.config.lookback_days} days): "
            f"start={prices.iloc[0]:.2f}, end={prices.iloc[-1]:.2f}, "
            f"trailing_return={trailing_return:+.2%}\n"
            f"Annualized volatility: {vol:.2%}\n"
            f"Macro indicators: {macro_str}\n\n"
            f"Part A: Identify the current market regime. "
            f"Choose from: bull, bear, sideways, crisis.\n\n"
            f"Part B: For each of the six asset classes "
            f"(Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash), "
            f"recommend whether to increase, decrease, or maintain the current allocation."
        )

        explanation = (
            f"Regime classification:\n"
            f"  Trailing {self.config.lookback_days}-day return: {trailing_return:+.2%}\n"
            f"  Annualized volatility: {vol:.2%}\n"
            f"  Macro: fed_funds={context.macro_context.get('fed_funds_rate', 'N/A'):.4f}, "
            f"vix={context.macro_context.get('vix', 'N/A'):.1f}\n"
            f"  → Regime: {regime.value}\n\n"
            f"Allocation rationale for {regime.value} regime:\n"
            + "\n".join(
                f"  {cls.title()}: {direction} "
                f"({'risk-on' if direction == 'increase' and cls in ('equities','real_estate','cryptocurrency') else 'defensive' if direction == 'increase' and cls in ('bonds','cash') else ''})"
                for cls, direction in allocation.items()
            )
        )

        answer_part_a = regime.value
        answer = f"Regime: {answer_part_a}; Allocations: {alloc_str}"

        split = self.config.get_split(d) or Split.TRAIN

        return QAPair(
            qa_id=self._make_id(d, seq),
            template_id=self.template_id,
            complexity=self.complexity,
            split=split,
            market_regime=regime,
            asset_class=self.asset_class,
            assets=context.assets,
            decision_date=d,
            context_summary=context_summary,
            question=question,
            answer=answer,
            answer_numeric=None,
            explanation=explanation,
            metadata={
                "detected_regime": regime.value,
                "allocation_adjustments": allocation,
                "trailing_return": round(trailing_return, 6),
                "annualized_vol": round(vol, 6),
            },
        )
