"""
T4 – Pairwise Allocation (Constrained)
Compute the minimum-variance portfolio weights for two assets subject to a
minimum expected return constraint.
Complexity level 2.

When ``redesign=False`` (default): legacy question exposes covariance (current paper).
When ``redesign=True``: withhold covariance, add return floor, require explanation.
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
from .constraint_v2 import (
    TEMPLATE_VERSION,
    source_snapshot_provenance,
    t4_metrics,
    t4_solution,
)


class T4PairwiseAllocation(QABuilder):
    """Template T4: Pairwise Constrained Allocation."""

    def __init__(
        self,
        provider,
        config,
        redesign: bool = False,
        template_version: str = "legacy",
    ):
        super().__init__(provider, config)
        self.redesign = redesign
        self.template_version = template_version

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
        rng = random.Random(int(decision_date.strftime("%Y%m%d")) + 3)
        cls1 = rng.choice(text_classes)
        if rng.random() < 0.5:
            cls2 = rng.choice([c for c in text_classes if c != cls1] + other_classes)
        else:
            cls2 = rng.choice(other_classes)
        c1 = self.provider.list_assets(cls1) or self.provider.list_assets("equities")
        c2 = self.provider.list_assets(cls2) or self.provider.list_assets("bonds")
        return [rng.choice(c1), rng.choice(c2)]

    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        if self.template_version == TEMPLATE_VERSION:
            return self._build_one_constraint_v2(context, seq)
        if self.redesign:
            return self._build_one_redesign(context, seq)
        return self._build_one_legacy(context, seq)

    def _build_one_constraint_v2(self, context: ContextWindow, seq: int) -> QAPair:
        """Build a candidate-allocation decision with all required data visible."""
        a1, a2, decision_date, _aligned, s1, s2, cov_12, _corr, mu1, mu2 = self._aligned_stats(
            context
        )
        assets = [a1, a2]
        expected_returns = {a1: round(mu1, 6), a2: round(mu2, 6)}
        covariance = [
            [round(s1 * s1 * 252.0, 8), round(cov_12 * 252.0, 8)],
            [round(cov_12 * 252.0, 8), round(s2 * s2 * 252.0, 8)],
        ]
        current_weights = ({a1: 0.7, a2: 0.3} if seq % 2 == 0 else {a1: 0.3, a2: 0.7})
        candidates = []
        for index, weight_1 in enumerate((0.1, 0.3, 0.5, 0.7, 0.9), start=1):
            weights = {a1: weight_1, a2: round(1.0 - weight_1, 4)}
            candidates.append(
                {
                    "candidate_id": f"C{index}",
                    "weights": weights,
                    "metrics": t4_metrics(
                        weights,
                        assets,
                        expected_returns,
                        covariance,
                        current_weights,
                    ),
                }
            )

        return_values = sorted({float(item["metrics"]["expected_return"]) for item in candidates})
        turnover_values = sorted({float(item["metrics"]["turnover"]) for item in candidates})
        selected_problem = None
        for return_floor in return_values:
            for turnover_cap in turnover_values:
                try:
                    solution = t4_solution(
                        candidates,
                        return_floor=return_floor,
                        turnover_cap=turnover_cap,
                    )
                except ValueError:
                    continue
                feasibility = []
                for candidate in candidates:
                    metrics = candidate["metrics"]
                    feasibility.append(
                        float(metrics["expected_return"]) >= return_floor - 1e-8
                        and float(metrics["turnover"]) <= turnover_cap + 1e-8
                    )
                has_return_decoy = any(
                    float(item["metrics"]["expected_return"]) < return_floor - 1e-8
                    for item in candidates
                )
                has_turnover_decoy = any(
                    float(item["metrics"]["turnover"]) > turnover_cap + 1e-8
                    for item in candidates
                )
                if sum(feasibility) >= 2 and has_return_decoy and has_turnover_decoy:
                    selected_problem = (return_floor, turnover_cap, solution)
                    break
            if selected_problem is not None:
                break
        if selected_problem is None:
            raise ValueError("Could not construct a discriminative T4 constraint-v2 item")
        return_floor, turnover_cap, _solution = selected_problem
        return_floor = round(return_floor, 6)
        turnover_cap = round(turnover_cap, 6)
        solution = t4_solution(
            candidates,
            return_floor=return_floor,
            turnover_cap=turnover_cap,
        )
        candidate_lines = "\n".join(
            f"{item['candidate_id']}: {a1}={item['weights'][a1]:.4f}, {a2}={item['weights'][a2]:.4f}"
            for item in candidates
        )
        context_summary = (
            f"Pairwise allocation decision for {a1} and {a2}. "
            "All candidate weights and constraints are in the question."
        )
        question = (
            f"Assets: {a1}, {a2}\n"
            f"Annualized expected returns: {a1}={expected_returns[a1]:.6f}, {a2}={expected_returns[a2]:.6f}\n"
            f"Annualized covariance matrix in [{a1}, {a2}] order: {covariance}\n"
            f"Current weights: {a1}={current_weights[a1]:.4f}, {a2}={current_weights[a2]:.4f}\n"
            f"Return floor: {return_floor:.6f}\n"
            f"Turnover cap: {turnover_cap:.6f}\n"
            f"Candidates:\n{candidate_lines}\n\n"
            "For each candidate, compute expected return w^T mu, variance w^T Sigma w, "
            "and turnover 0.5 * sum(abs(w-current)). Keep only long-only candidates whose weights sum to 1, "
            "meet the return floor, and do not exceed the turnover cap. Select the feasible candidate with "
            "minimum variance. Break ties by higher return, lower turnover, then candidate ID. "
            "Reply with JSON only: {\"candidate_id\": \"C#\", \"weights\": {\"ASSET\": <decimal>}, "
            "\"calculated_metrics\": {\"expected_return\": <decimal>, \"variance\": <decimal>, "
            "\"turnover\": <decimal>}, \"binding_constraints\": [\"return_floor|turnover_cap\"], "
            "\"rationale\": \"...\"}."
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
            assets=assets,
            decision_date=decision_date,
            context_summary=context_summary,
            question=question,
            answer=str(solution["candidate_id"]),
            answer_numeric=None,
            explanation="Constraint-v2 reference solution retained only for offline scoring.",
            metadata={
                "template_version": TEMPLATE_VERSION,
                "generator_version": "qa-v2-constraint-20260825",
                "seed": self.config.random_seed,
                **provenance,
                "constraint_v2": {
                    "assets": assets,
                    "expected_returns": expected_returns,
                    "covariance": covariance,
                    "current_weights": current_weights,
                    "return_floor": round(return_floor, 6),
                    "turnover_cap": round(turnover_cap, 6),
                    "candidates": candidates,
                    **solution,
                },
            },
        )

    def _aligned_stats(self, context: ContextWindow):
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
        corr = cov_12 / (s1 * s2) if s1 * s2 > 0 else 0.0
        mu1 = float(aligned[:, 0].mean() * 252)
        mu2 = float(aligned[:, 1].mean() * 252)
        return a1, a2, d, aligned, s1, s2, cov_12, corr, mu1, mu2

    def _build_one_legacy(self, context: ContextWindow, seq: int) -> QAPair:
        a1, a2, d, aligned, s1, s2, cov_12, corr, _mu1, _mu2 = self._aligned_stats(
            context
        )

        denom = s1**2 + s2**2 - 2 * cov_12
        if abs(denom) < 1e-12:
            w1, w2 = 0.5, 0.5
        else:
            w1 = (s2**2 - cov_12) / denom
            w2 = 1.0 - w1

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
            answer_numeric=w1,
            explanation=explanation,
            metadata={
                "weights": {a1: w1, a2: w2},
                "sigma_1": round(s1, 6),
                "sigma_2": round(s2, 6),
                "covariance": round(float(cov_12), 6),
                "correlation": round(corr, 4),
                "t3t4_redesign": False,
            },
        )

    def _build_one_redesign(self, context: ContextWindow, seq: int) -> QAPair:
        a1, a2, d, _aligned, s1, s2, cov_12, _corr, mu1, mu2 = self._aligned_stats(
            context
        )

        denom = s1**2 + s2**2 - 2 * cov_12
        if abs(denom) < 1e-12:
            w1_mv, w2_mv = 0.5, 0.5
        else:
            w1_mv = max(0.0, min(1.0, (s2**2 - cov_12) / denom))
            w2_mv = 1.0 - w1_mv
        mv_return = w1_mv * mu1 + w2_mv * mu2

        rng = np.random.default_rng(int(d.strftime("%Y%m%d")) + seq)
        mu_max = max(mu1, mu2)
        mu_min = min(mu1, mu2)
        binding = (seq % 2 == 0) and (mu_max > mv_return)
        if binding:
            mu_floor = float(
                rng.uniform(
                    mv_return + 1e-6,
                    mu_max - 1e-6 if mu_max > mv_return + 2e-6 else mv_return + 1e-4,
                )
            )
        else:
            mu_floor = float(rng.uniform(mu_min, mv_return))
        mu_floor = round(mu_floor, 4)

        if binding and abs(mu1 - mu2) > 1e-9:
            w1 = (mu_floor - mu2) / (mu1 - mu2)
            w1 = max(0.0, min(1.0, w1))
        else:
            w1 = w1_mv
        w2 = 1.0 - w1
        w1, w2 = round(w1, 4), round(w2, 4)

        portfolio_return = round(w1 * mu1 + w2 * mu2, 4)
        portfolio_vol = round(
            float(np.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * cov_12)),
            6,
        )

        context_summary = (
            f"{a1} σ={s1:.4f} μ={mu1:.4f}, {a2} σ={s2:.4f} μ={mu2:.4f}. "
            f"Return floor={mu_floor:.4f}. "
            f"Optimal: w_{a1}={w1:.4f}, w_{a2}={w2:.4f}."
        )
        # Covariance withheld from the question.
        question = (
            f"Assets: {a1}, {a2}\n"
            f"{a1}: annualized_mean_return={mu1:.4f}, daily_std={s1:.4f}\n"
            f"{a2}: annualized_mean_return={mu2:.4f}, daily_std={s2:.4f}\n"
            f"Minimum required portfolio return (annualized): {mu_floor:.4f}\n"
            f"Market regime: {context.market_regime.value if context.market_regime else 'unknown'}\n\n"
            f"Compute portfolio weights (w_{a1}, w_{a2}) that minimize portfolio variance "
            f"while satisfying the minimum return constraint. "
            f"Constraints: all weights ≥ 0, weights sum to 1.\n"
            f"Report JSON: "
            f'{{"answer": "w_{a1}=X.XXXX, w_{a2}=X.XXXX", '
            f'"explanation": "<1-3 sentences>"}}.\n'
            f"In the explanation, state the variance objective, whether the return "
            f"floor is binding, and the long-only / sum-to-one constraints."
        )

        binding_word = "binding" if binding else "non-binding"
        explanation = (
            f"Minimize portfolio variance. Return floor {mu_floor:.4f} is {binding_word}. "
            f"Long-only weights sum to 1. "
            f"Optimal: w_{a1}={w1:.4f}, w_{a2}={w2:.4f}, "
            f"portfolio_return={portfolio_return:.4f}, portfolio_vol={portfolio_vol:.6f}."
        )
        keypoints = [
            "variance",
            binding_word,
            "sum to 1",
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
                "t3t4_redesign": True,
                "explanation_keypoints": keypoints,
            },
        )
