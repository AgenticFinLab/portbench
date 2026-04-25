"""
T3 – Position Sizing
Determine the maximum position size (as a fraction of portfolio) given a
maximum drawdown constraint, using the fixed-fractional method.
Complexity level 1.
"""

from datetime import date


from .base import (
    ComplexityLevel,
    ContextWindow,
    MarketRegime,
    QABuilder,
    QAConfig,
    QAPair,
    Split,
)
from ..metrics.risk_metrics import max_drawdown, var
from ..metrics.base import MetricsConfig


class T3PositionSizing(QABuilder):
    """
    Template T3: Position Sizing.

    Uses a simplified fixed-fractional / Kelly-inspired rule:
        f* = max_acceptable_drawdown / expected_max_single_period_loss
    where expected_max_single_period_loss is approximated as |VaR(99%)|.

    The answer is capped at 1.0 (100% of portfolio) and floored at 0.0.

    Question format:
        "Given a maximum acceptable drawdown of {threshold}% and the following
         return history for {asset}, determine the maximum position size as a
         fraction of portfolio."
    """

    def __init__(
        self, provider, config: QAConfig, max_drawdown_threshold: float = 0.10
    ):
        """
        Args:
            max_drawdown_threshold: Maximum acceptable portfolio drawdown (e.g., 0.10 = 10%).
        """
        super().__init__(provider, config)
        self.max_drawdown_threshold = max_drawdown_threshold

    @property
    def template_id(self) -> str:
        return "T3"

    @property
    def complexity(self) -> ComplexityLevel:
        return ComplexityLevel.LEVEL_1

    @property
    def asset_class(self) -> str:
        return "all"

    def _select_assets(self, decision_date: date) -> list[str]:
        import random

        text_classes = ["equities", "cryptocurrency"]
        other_classes = ["bonds", "commodities", "real_estate", "cash"]
        rng = random.Random(hash(decision_date) + 2)
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
        returns = context.returns_history[asset].dropna()

        if len(returns) < 20:
            raise ValueError(f"Insufficient history for T3: {asset} at {d}")

        # Approximate max single-period loss as |VaR(99%)|
        metrics_cfg = MetricsConfig(var_confidence=0.99)
        var_99 = var(returns, metrics_cfg)
        expected_loss = abs(var_99)  # Positive number (magnitude of potential loss)

        # Fixed-fractional: position size = threshold / expected_loss
        if expected_loss == 0:
            position_size = 1.0
        else:
            position_size = min(1.0, self.max_drawdown_threshold / expected_loss)

        position_size = round(position_size, 4)

        pct_threshold = int(self.max_drawdown_threshold * 100)
        context_summary = (
            f"{asset}: {len(returns)}-day history, VaR(99%)={var_99:.4f}, "
            f"max drawdown threshold={pct_threshold}%."
        )

        question = (
            f"Asset: {asset}\n"
            f"Daily returns (past {len(returns)} days): "
            f"mean={returns.mean():.4f}, std={returns.std():.4f}, "
            f"worst_day={returns.min():.4f}\n"
            f"Maximum acceptable portfolio drawdown: {pct_threshold}%\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n"
            + (
                f"Recent filing/news:\n{context.news_text}\n"
                if context.news_text
                else ""
            )
            + f"\nUsing the fixed-fractional position sizing method, determine the maximum "
            f"position size in {asset} as a fraction of total portfolio. "
            f"(Approximate worst-case single-period loss as |VaR(99%)|.)"
        )

        explanation = (
            f"Step 1: Compute |VaR(99%)| from historical returns = {expected_loss:.4f} "
            f"(i.e., a {expected_loss:.2%} loss in the worst 1% of days).\n"
            f"Step 2: Fixed-fractional formula: f* = {pct_threshold}% / {expected_loss:.4f} "
            f"= {self.max_drawdown_threshold / expected_loss:.4f}, capped at 1.0.\n"
            f"Maximum position size = {position_size:.4f} ({position_size:.1%} of portfolio)."
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
            answer=f"{position_size:.4f}",
            answer_numeric=position_size,
            explanation=explanation,
            metadata={
                "var_99": round(float(var_99), 6),
                "expected_loss": round(expected_loss, 6),
                "max_drawdown_threshold": self.max_drawdown_threshold,
                "position_size": position_size,
            },
        )
