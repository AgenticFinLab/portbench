"""
T1 – Return Prediction
Predict whether asset return over the next horizon_days will be positive,
negative, or flat (±1% band).
Complexity level 1 (single asset).
"""

import random
from datetime import date, timedelta

import numpy as np

from .base import (
    ComplexityLevel,
    ContextWindow,
    MarketRegime,
    QABuilder,
    QAPair,
    Split,
)


class T1ReturnPrediction(QABuilder):
    """
    Template T1: Return Prediction.

    Question format:
        "Given the past {lookback} days of prices and news for {asset}, predict
         whether its return over the next {horizon} trading days will be
         positive (>+1%), negative (<-1%), or flat (within ±1%)."

    Ground truth: computed from actual future prices (forward-looking only at
    evaluation time; strictly PiT at inference time).
    """

    @property
    def template_id(self) -> str:
        return "T1"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_1

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        # Bias toward text-bearing classes (equities/crypto) to maximize text coverage,
        # but still rotate across all six asset classes so T1 covers the full universe.
        text_classes = ["equities", "cryptocurrency"]
        other_classes = ["bonds", "commodities", "real_estate", "cash"]
        rng = random.Random(hash(decision_date))
        cls = (
            rng.choice(text_classes)
            if rng.random() < 0.8
            else rng.choice(other_classes)
        )
        candidates = self.provider.list_assets(cls)
        if not candidates:
            candidates = self.provider.list_assets("equities")
        return [rng.choice(candidates)]

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        asset = context.assets[0]
        d = context.decision_date
        horizon = self.config.horizon_days

        # --- Compute forward return (ground truth, uses future data) ---
        future_end = d + timedelta(days=int(horizon * 1.5))  # extra buffer for weekends
        future_prices = self.provider.get_price_series(asset, d, future_end)

        if len(future_prices) < 2:
            raise ValueError(f"Insufficient future data for {asset} at {d}")

        # Use the horizon-th trading day after decision_date
        future_return = float(
            future_prices.iloc[min(horizon, len(future_prices) - 1)]
            / future_prices.iloc[0]
            - 1
        )

        # Classify direction
        if future_return > 0.01:
            answer = "positive"
        elif future_return < -0.01:
            answer = "negative"
        else:
            answer = "flat"

        # --- Build context summary ---
        prices = context.price_history[asset]
        hist_return = (
            float(prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) > 1 else 0.0
        )
        vol = (
            float(context.returns_history[asset].std() * np.sqrt(252))
            if len(context.returns_history[asset]) > 1
            else 0.0
        )

        context_summary = (
            f"{asset} over past {self.config.lookback_days} days: "
            f"cumulative return {hist_return:+.1%}, annualized vol {vol:.1%}. "
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}."
        )

        question = (
            f"Asset: {asset}\n"
            f"Historical prices (past {self.config.lookback_days} trading days): "
            f"start={prices.iloc[0]:.2f}, end={prices.iloc[-1]:.2f}, "
            f"cumulative_return={hist_return:+.1%}, annualized_volatility={vol:.1%}\n"
            f"Macro context: {context.macro_context}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n"
            + (
                f"Recent filing/news:\n{context.news_text}\n"
                if context.news_text
                else ""
            )
            + f"\nPredict whether the return of {asset} over the next {horizon} trading days "
            f"will be: positive (>+1%), negative (<-1%), or flat (within ±1%)."
        )

        explanation = (
            f"The actual {horizon}-day forward return for {asset} starting {d} was "
            f"{future_return:+.2%}, which classifies as '{answer}'."
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
            assets=[asset],
            decision_date=d,
            context_summary=context_summary,
            question=question,
            answer=answer,
            answer_numeric=round(future_return, 6),
            explanation=explanation,
            metadata={
                "future_return": round(future_return, 6),
                "horizon_days": horizon,
                "hist_return": round(hist_return, 6),
                "annualized_vol": round(vol, 6),
            },
        )
