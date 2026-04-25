"""
Abstract base classes for PortBench baseline strategies.

A BaselineStrategy is both an AgentAdapter (so it can be plugged into
EvalPipeline) and a self-contained portfolio allocator.  The key contract:

    allocate(snapshot) -> dict[str, float]

returns a weight dict that sums to 1.0.  The AgentAdapter.complete()
method is implemented here to serialize the weight dict so that S3
(weight optimization) stages can parse it; other stages fall back to
rule-based ground-truth (same as MockAgentAdapter).
"""

import json
from abc import abstractmethod
from dataclasses import dataclass, field

from ..agent_eval.base import AgentAdapter, MarketSnapshot


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """
    Holds the output of a single baseline allocation.

    Attributes:
        weights:     Asset → weight mapping (sums to 1.0).
        model_name:  Identifier of the baseline that produced this result.
        notes:       Optional free-text rationale (useful for debugging).
    """

    weights: dict[str, float] = field(default_factory=dict)
    model_name: str = "unknown_baseline"
    notes: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaselineStrategy(AgentAdapter):
    """
    Abstract base class for all PortBench baseline strategies.

    Subclasses must implement:
      - allocate(snapshot) → dict[str, float]   # the core weight computation
      - model_name (property)

    The complete(prompt) method is intentionally thin — baselines bypass
    LLM calls entirely and produce weights directly from market data.
    """

    @abstractmethod
    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Compute target portfolio weights from the given market snapshot.

        Args:
            snapshot: Current market data, macro indicators, and current weights.

        Returns:
            Dict mapping asset identifier → weight in [0, 1].
            Weights must sum to approximately 1.0 (tolerance 1e-4).
        """
        pass

    def complete(self, prompt: str) -> str:
        """
        Stub implementation of AgentAdapter.complete().

        Baselines don't interpret free-text prompts.  This returns a JSON
        placeholder so that any stage that reads raw_llm_output can still
        proceed without error.
        """
        return json.dumps({"baseline": self.model_name, "prompt_ignored": True})

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        """
        Normalize a weight dict so values sum to exactly 1.0.

        Zero or negative total → returns equal weights.
        """
        total = sum(weights.values())
        if total <= 0:
            n = max(len(weights), 1)
            return {a: round(1.0 / n, 6) for a in weights}
        return {a: round(w / total, 6) for a, w in weights.items()}
