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
from typing import Optional


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
    score: float                              # 0 = total failure, 1 = perfect
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

    def __init__(self, propagation_weight: float = 0.1):
        self.propagation_weight = propagation_weight

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
        isolated_avg = sum(scores) / len(scores)

        # Cascade penalty: sum of drops between consecutive stages
        cascade_drops = sum(
            max(scores[i] - scores[i + 1], 0.0)
            for i in range(len(scores) - 1)
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
        self, episodes: list[list[StageScore]]
    ) -> dict:
        """
        Compute CEPS over a batch of evaluation episodes and return summary stats.

        Args:
            episodes: List of episode stage-score lists.

        Returns:
            Dict with mean_ceps, std_ceps, per_stage_mean, and individual results.
        """
        import numpy as np

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
