"""
T2 – Risk Assessment
Compute historical-simulation VaR at a given confidence level for a single asset.
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
from ..metrics.risk_metrics import var, cvar
from ..metrics.base import MetricsConfig


class T2RiskAssessment(QABuilder):
    """
    Template T2: Risk Assessment.

    Question format:
        "Given {lookback} days of daily returns for {asset}, compute the
         1-day Value-at-Risk at {confidence}% confidence level using the
         historical simulation method."

    Ground truth: historical VaR computed from the context return series.
    This is fully PiT-safe because it uses only past returns.
    """

    def __init__(self, provider, config: QAConfig, confidence: float = 0.95):
        super().__init__(provider, config)
        self.confidence = confidence

    @property
    def template_id(self) -> str:
        return "T2"

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
        rng = random.Random(hash(decision_date) + 1)
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
            raise ValueError(f"Insufficient return history for T2: {asset} at {d}")

        metrics_cfg = MetricsConfig(var_confidence=self.confidence)
        var_value = var(returns, metrics_cfg)
        cvar_value = cvar(returns, metrics_cfg)

        pct_conf = int(self.confidence * 100)
        context_summary = (
            f"{asset}: {len(returns)}-day return history, "
            f"mean={returns.mean():.4f}, std={returns.std():.4f}."
        )

        question = (
            f"Asset: {asset}\n"
            f"Daily returns (past {len(returns)} days): "
            f"mean={returns.mean():.4f}, std={returns.std():.4f}, "
            f"min={returns.min():.4f}, max={returns.max():.4f}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n"
            + (
                f"Recent filing/news:\n{context.news_text}\n"
                if context.news_text
                else ""
            )
            + f"\nUsing the historical simulation method, compute the 1-day VaR at "
            f"{pct_conf}% confidence level for {asset}. Express as a decimal (e.g., -0.02)."
        )

        explanation = (
            f"Historical simulation VaR at {pct_conf}%: sort the {len(returns)} daily returns "
            f"and take the {100 - pct_conf}th percentile. "
            f"VaR({pct_conf}%) = {var_value:.4f} (i.e., on a bad day with {100-pct_conf}% probability, "
            f"the loss exceeds {abs(var_value):.2%}). "
            f"CVaR({pct_conf}%) = {cvar_value:.4f}."
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
            answer=f"{var_value:.4f}",
            answer_numeric=round(var_value, 6),
            explanation=explanation,
            metadata={
                "var": round(var_value, 6),
                "cvar": round(cvar_value, 6),
                "confidence": self.confidence,
                "n_returns": len(returns),
            },
        )
