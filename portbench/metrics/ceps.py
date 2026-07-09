"""
Cross-Stage Error Propagation Score (CEPS) for end-to-end pipeline evaluation.

CEPS measures how much an error at an upstream stage amplifies as it propagates
through the five-stage portfolio management pipeline:

    S1 (Market Interpretation)
      → S2 (Signal Generation)
        → S3 (Weight Optimization)
          → S4 (Execution Simulation)
            → S5 (Risk Monitoring)

Each stage receives a score in [0, 1] (1 = perfect, 0 = complete failure).
CEPS aggregates the stage scores into a single propagation-aware metric that
penalizes cascading errors more heavily than isolated ones.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage score container
# ---------------------------------------------------------------------------


@dataclass
class StageScore:
    """
    Score for a single pipeline stage.

    Attributes:
        stage_id:       Stage identifier, e.g. "S1" through "S5".
        stage_name:     Human-readable name.
        score:          Accuracy / agreement with ground truth in [0, 1].
        ground_truth:   The ideal output for this stage (typed as Any for flexibility).
        actual_output:  The LLM's actual output for this stage.
        error_details:  Free-form dict describing what went wrong (for debugging).
    """

    stage_id: str
    stage_name: str
    score: float  # 0 = total failure, 1 = perfect
    ground_truth: object = None
    actual_output: object = None
    error_details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CEPS computation
# ---------------------------------------------------------------------------


@dataclass
class CEPSResult:
    """
    Full CEPS result for one evaluation episode.

    Attributes:
        stage_scores:           List of StageScore, one per stage (S1→S5).
        ceps_score:             Aggregate CEPS in [0, 1]. Higher = better.
        propagation_penalty:    Extra penalty for error cascades (0 = no cascade).
        isolated_avg:           Simple mean of stage scores (no cascade penalty).
    """

    stage_scores: list[StageScore] = field(default_factory=list)
    ceps_score: float = 0.0
    propagation_penalty: float = 0.0
    isolated_avg: float = 0.0


class CEPS:
    """
    Computes the Cross-Stage Error Propagation Score.

    Design rationale:
    ─────────────────
    A model that scores [1, 1, 0.5, 0.5, 0.5] (fails at S3 and propagates the
    error) should score lower than one that scores [0.7, 0.7, 0.7, 0.7, 0.7]
    (uniform mediocrity). The propagation_weight controls how strongly cascade
    amplification is penalized.

    Formula:
        isolated_avg  = mean(stage_scores)
        cascade_drop  = max drop between consecutive stage scores (if scores fall)
        propagation_penalty = propagation_weight * sum(max(s[i] - s[i+1], 0))
        ceps_score    = isolated_avg - propagation_penalty   (clipped to [0, 1])

    Args:
        propagation_weight: Penalty coefficient for cascade drops (default 0.1).
                            Set to 0 to recover the simple mean (no cascade awareness).
    """

    def __init__(self, propagation_weight: float = 0.1, variant: str = "full"):
        """
        Args:
            propagation_weight: Penalty coefficient for cascade drops (default 0.1).
            variant: Aggregation variant:
                     "full"      — S1-S5 equal weight (default)
                     "core"      — S1-S3 only (LLM-involved stages)
                     "weighted"  — S1-S3 each 0.25, S4-S5 each 0.125
                     "geometric" — geometric mean instead of arithmetic
        """
        self.propagation_weight = propagation_weight
        if variant not in ("full", "core", "weighted", "geometric"):
            raise ValueError(
                f"Unknown CEPS variant {variant!r}. "
                f"Expected: full, core, weighted, geometric"
            )
        self.variant = variant

    def compute(self, stage_scores: list[StageScore]) -> CEPSResult:
        """
        Compute CEPS from a list of per-stage scores.

        Args:
            stage_scores: List of StageScore ordered S1→S5. Missing stages are
                          treated as score 0 (critical failure).

        Returns:
            CEPSResult with ceps_score and diagnostic breakdown.
        """
        if not stage_scores:
            return CEPSResult()

        scores = [s.score for s in stage_scores]

        # Apply variant-specific aggregation
        if self.variant == "core":
            scores = scores[:3]
            isolated_avg = sum(scores) / max(len(scores), 1)
            cascade_drops = sum(
                max(scores[i] - scores[i + 1], 0.0) for i in range(len(scores) - 1)
            )
        elif self.variant == "weighted":
            weights_arr = [0.25, 0.25, 0.25, 0.125, 0.125]
            isolated_avg = sum(s * w for s, w in zip(scores, weights_arr))
            cascade_drops = sum(
                max(scores[i] - scores[i + 1], 0.0) * weights_arr[i]
                for i in range(len(scores) - 1)
            )
        elif self.variant == "geometric":
            import numpy as np

            eps = 1e-8
            isolated_avg = float(
                np.prod([max(s, eps) for s in scores]) ** (1.0 / len(scores))
            )
            cascade_drops = sum(
                max(scores[i] - scores[i + 1], 0.0) for i in range(len(scores) - 1)
            )
        else:  # "full" (default)
            isolated_avg = sum(scores) / len(scores)
            cascade_drops = sum(
                max(scores[i] - scores[i + 1], 0.0) for i in range(len(scores) - 1)
            )

        penalty = self.propagation_weight * cascade_drops
        ceps_score = max(0.0, min(1.0, isolated_avg - penalty))

        return CEPSResult(
            stage_scores=stage_scores,
            ceps_score=round(ceps_score, 4),
            propagation_penalty=round(penalty, 4),
            isolated_avg=round(isolated_avg, 4),
        )

    def compute_batch(
        self,
        episodes: list[list[StageScore]],
        stage_mask: list[bool] = None,
    ) -> dict:
        """
        Compute CEPS over a batch of evaluation episodes and return summary stats.

        Args:
            episodes:   List of episode stage-score lists.
            stage_mask: Optional bool mask to filter stages (e.g.
                        [True, True, True, False, False] for S1-S3 only).

        Returns:
            Dict with mean_ceps, std_ceps, per_stage_mean, and individual results.
        """
        import numpy as np

        if stage_mask is not None:
            episodes = [
                [ss for ss, keep in zip(ep, stage_mask) if keep]
                for ep in episodes
            ]

        results = [self.compute(ep) for ep in episodes]
        ceps_scores = [r.ceps_score for r in results]

        # Per-stage mean across episodes (pad missing stages with 0)
        max_stages = max(len(ep) for ep in episodes) if episodes else 0
        per_stage: list[list[float]] = [[] for _ in range(max_stages)]
        for ep in episodes:
            for i, ss in enumerate(ep):
                per_stage[i].append(ss.score)

        per_stage_mean = {
            f"S{i+1}": round(float(np.mean(vals)), 4)
            for i, vals in enumerate(per_stage)
            if vals
        }

        return {
            "mean_ceps": round(float(np.mean(ceps_scores)), 4) if ceps_scores else 0.0,
            "std_ceps": round(float(np.std(ceps_scores)), 4) if ceps_scores else 0.0,
            "per_stage_mean": per_stage_mean,
            "n_episodes": len(results),
            "individual_results": results,
        }

    def compute_batch_with_variants(
        self, episodes: list[list[StageScore]]
    ) -> dict[str, dict]:
        """
        Compute CEPS for all four variants in a single pass.

        Args:
            episodes: List of episode stage-score lists.

        Returns:
            Dict mapping variant name → summary dict (from compute_batch).
        """
        variants = {
            "full": CEPS(self.propagation_weight, variant="full"),
            "core": CEPS(self.propagation_weight, variant="core"),
            "weighted": CEPS(self.propagation_weight, variant="weighted"),
            "geometric": CEPS(self.propagation_weight, variant="geometric"),
        }
        return {name: ceps.compute_batch(episodes) for name, ceps in variants.items()}
