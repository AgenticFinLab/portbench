"""
T3 – Position Sizing
Determine the maximum position size (as a fraction of portfolio) given a
maximum drawdown constraint, using the fixed-fractional method.
Complexity level 1.

When ``redesign=True`` (YAML ``qa.t3t4_redesign``): strip VaR from context,
vary the drawdown threshold, and require a short explanation in the answer.
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
from ..metrics.risk_metrics import var
from ..metrics.base import MetricsConfig
from .constraint_v2 import TEMPLATE_VERSION, source_snapshot_provenance, t3_solution

_REDESIGN_THRESHOLDS = (0.05, 0.08, 0.10, 0.15)


class T3PositionSizing(QABuilder):
    """
    Template T3: Position Sizing.

    Uses a simplified fixed-fractional / Kelly-inspired rule:
        f* = max_acceptable_drawdown / expected_max_single_period_loss
    where expected_max_single_period_loss is approximated as |VaR(99%)|.
    """

    def __init__(
        self,
        provider,
        config: QAConfig,
        max_drawdown_threshold: float = 0.10,
        redesign: bool = False,
        template_version: str = "legacy",
    ):
        super().__init__(provider, config)
        self.max_drawdown_threshold = max_drawdown_threshold
        self.redesign = redesign
        self.template_version = template_version

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
        rng = random.Random(int(decision_date.strftime("%Y%m%d")) + 2)
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
        if self.template_version == TEMPLATE_VERSION:
            return self._build_one_constraint_v2(context, seq)
        if self.redesign:
            return self._build_one_redesign(context, seq)
        return self._build_one_legacy(context, seq)

    def _build_one_constraint_v2(self, context: ContextWindow, seq: int) -> QAPair:
        """Build a solvable multi-constraint sizing problem without GT leakage."""
        asset = context.assets[0]
        decision_date = context.decision_date
        returns = context.returns_history[asset].dropna()
        if len(returns) < 20:
            raise ValueError(f"Insufficient history for T3: {asset} at {decision_date}")

        losses = sorted([-float(value) for value in returns])
        tail_size = max(1, int(len(losses) * 0.05))
        unit_var = max(1e-6, losses[int(len(losses) * 0.95)])
        unit_es = max(unit_var, sum(losses[-tail_size:]) / tail_size)
        cumulative = (1.0 + returns).cumprod()
        unit_drawdown = max(1e-6, float((1.0 - cumulative / cumulative.cummax()).max()))
        constraint_patterns = (
            {"var": 0.35, "es": 0.70, "drawdown": 0.80, "liquidity": 0.90},
            {"var": 0.80, "es": 0.40, "drawdown": 0.70, "liquidity": 0.90},
            {"var": 0.80, "es": 0.90, "drawdown": 0.50, "liquidity": 0.90},
            {"var": 0.80, "es": 0.90, "drawdown": 0.90, "liquidity": 0.60},
            {"var": 1.20, "es": 1.10, "drawdown": 1.30, "liquidity": 1.20},
        )
        limits = constraint_patterns[seq % len(constraint_patterns)]
        target_limits = {name: limits[name] for name in ("var", "es", "drawdown")}
        liquidity_cap = limits["liquidity"]
        unit_risk = {
            "var": round(unit_var, 6),
            "es": round(unit_es, 6),
            "drawdown": round(unit_drawdown, 6),
        }
        budgets = {
            name: round(unit_risk[name] * limit, 6)
            for name, limit in target_limits.items()
        }
        solution = t3_solution(unit_risk, budgets, liquidity_cap)
        context_summary = (
            f"{asset}: {len(returns)} Point-in-Time daily observations. "
            "Use the visible per-unit risk estimates and limits below."
        )
        question = (
            f"Asset: {asset}\n"
            f"Per-unit VaR(95%): {unit_var:.6f}\n"
            f"Per-unit ES(95%): {unit_es:.6f}\n"
            f"Per-unit drawdown proxy: {unit_drawdown:.6f}\n"
            f"VaR budget: {budgets['var']:.6f}\n"
            f"ES budget: {budgets['es']:.6f}\n"
            f"Drawdown budget: {budgets['drawdown']:.6f}\n"
            f"Liquidity cap: {liquidity_cap:.4f}\n\n"
            "Choose the largest position_size in [0,1] that satisfies every constraint. "
            "Use budget / per-unit-risk for VaR, ES, and drawdown. "
            "Reply with JSON only: {\"position_size\": <decimal>, \"binding_constraint\": "
            "\"var|es|drawdown|liquidity|full_allocation\", \"constraint_margins\": "
            "{\"var\": <decimal>, \"es\": <decimal>, \"drawdown\": <decimal>, "
            "\"liquidity\": <decimal>, \"full_allocation\": <decimal>}, \"rationale\": \"...\"}."
        )
        split = self.config.get_split(decision_date) or Split.TRAIN
        regime = context.market_regime or MarketRegime.SIDEWAYS
        provenance = source_snapshot_provenance(context)
        return QAPair(
            qa_id=self._make_id(decision_date, seq),
            template_id=self.template_id,
            complexity=self.complexity,
            split=split,
            market_regime=regime,
            asset_class=self.asset_class,
            assets=[asset],
            decision_date=decision_date,
            context_summary=context_summary,
            question=question,
            answer=str(solution["position_size"]),
            answer_numeric=float(solution["position_size"]),
            explanation="Constraint-v2 reference solution retained only for offline scoring.",
            metadata={
                "template_version": TEMPLATE_VERSION,
                "generator_version": "qa-v2-constraint-20260825",
                "seed": self.config.random_seed,
                **provenance,
                "constraint_v2": {
                    "unit_risk": unit_risk,
                    "budgets": budgets,
                    "liquidity_cap": liquidity_cap,
                    **solution,
                },
            },
        )

    def _compute_position(
        self, returns, threshold: float
    ) -> tuple[float, float, float]:
        metrics_cfg = MetricsConfig(var_confidence=0.99)
        var_99 = float(var(returns, metrics_cfg))
        expected_loss = abs(var_99)
        if expected_loss == 0:
            position_size = 1.0
        else:
            position_size = min(1.0, threshold / expected_loss)
        return round(position_size, 4), var_99, expected_loss

    def _build_one_legacy(self, context: ContextWindow, seq: int) -> QAPair:
        asset = context.assets[0]
        d = context.decision_date
        returns = context.returns_history[asset].dropna()

        if len(returns) < 20:
            raise ValueError(f"Insufficient history for T3: {asset} at {d}")

        position_size, var_99, expected_loss = self._compute_position(
            returns, self.max_drawdown_threshold
        )
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
            + f"\nDetermine the maximum fraction of total portfolio capital that should be "
            f"allocated to {asset}, given the {pct_threshold}% drawdown constraint. "
            f"Report as a decimal between 0.00 and 1.00 (e.g., 0.25 = 25%; "
            f"maximum is 1.00 = 100% of portfolio)."
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
                "t3t4_redesign": False,
            },
        )

    def _build_one_redesign(self, context: ContextWindow, seq: int) -> QAPair:
        asset = context.assets[0]
        d = context.decision_date
        returns = context.returns_history[asset].dropna()

        if len(returns) < 20:
            raise ValueError(f"Insufficient history for T3: {asset} at {d}")

        threshold = _REDESIGN_THRESHOLDS[seq % len(_REDESIGN_THRESHOLDS)]
        position_size, var_99, expected_loss = self._compute_position(returns, threshold)
        pct_threshold = int(threshold * 100)

        # Do not put VaR into context_summary (eval prompt would leak it).
        context_summary = (
            f"{asset}: {len(returns)}-day history, "
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
            + f"\nDetermine the maximum fraction of total portfolio capital that should be "
            f"allocated to {asset}, given the {pct_threshold}% drawdown constraint. "
            f"Report JSON: "
            f'{{"answer": <decimal in [0,1]>, "explanation": "<1-3 sentences>"}}.\n'
            f"In the explanation, state how you estimate downside risk, the "
            f"position-sizing rule you use, and whether the size is capped at 1.0."
        )

        explanation = (
            f"Estimate downside risk as |VaR(99%)| = {expected_loss:.4f}. "
            f"Fixed-fractional: f* = {threshold:.2f} / {expected_loss:.4f}, capped at 1.0. "
            f"Maximum position size = {position_size:.4f}."
        )
        keypoints = [
            "var",
            "drawdown",
            "capped" if position_size >= 0.999 else "position size",
        ]

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
                "max_drawdown_threshold": threshold,
                "position_size": position_size,
                "t3t4_redesign": True,
                "explanation_keypoints": keypoints,
            },
        )
