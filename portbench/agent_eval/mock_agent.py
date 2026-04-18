"""
Mock agent adapter for development and unit testing.

The mock agent does not call any LLM. Instead it returns rule-based outputs
that are intentionally slightly noisy (not perfect) so that CEPS scores are
non-trivial and the pipeline can be tested end-to-end.

Noise level controls how far the mock deviates from ground truth:
  0.0 = always returns ground truth (perfect agent)
  1.0 = returns maximally wrong answers
"""

import numpy as np

from .base import AgentAdapter


class MockAgentAdapter(AgentAdapter):
    """
    Deterministic rule-based mock agent adapter.

    Used in tests and examples to exercise the full pipeline without requiring
    real LLM API keys.

    Args:
        noise_level: Float in [0, 1]. Higher = worse mock answers.
        seed:        Random seed for reproducibility.
    """

    def __init__(self, noise_level: float = 0.2, seed: int = 0):
        self.noise_level = noise_level
        self._rng = np.random.default_rng(seed)

    @property
    def model_name(self) -> str:
        return f"mock_agent(noise={self.noise_level})"

    def complete(self, prompt: str) -> str:
        """
        Return a stub response.
        Concrete stage implementations call self.adapter.complete(prompt) and
        then parse the text; for the mock pipeline the stages bypass this
        and use compute_ground_truth() + noise injection directly.
        """
        return f"[MOCK RESPONSE] noise_level={self.noise_level}"
