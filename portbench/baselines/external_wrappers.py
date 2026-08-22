"""Black-box baseline wrappers for TradingAgents / FinCon.

Citation-only by default: runnable_gate() returns False until an external
repo is installed and explicitly enabled. No hard import of TradingAgents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class ExternalRunResult:
    """Weights plus native cost dict from an external baseline."""

    weights: Dict[str, float] = field(default_factory=dict)
    native_cost: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class _BaseExternalWrapper:
    """Shared interface for external multi-agent trading baselines."""

    name: str = "external"
    citation: str = ""

    def runnable_gate(self) -> Tuple[bool, str]:
        """Return (ok, reason). Default is citation-only / not runnable."""
        return (
            False,
            f"{self.name} is citation-only until installed and enabled; see {self.citation}",
        )

    def run(self, snapshot_like: Any) -> ExternalRunResult:
        """Refuse execution unless runnable_gate passes."""
        ok, reason = self.runnable_gate()
        if not ok:
            raise RuntimeError(reason)
        return self._run_impl(snapshot_like)

    def _run_impl(self, snapshot_like: Any) -> ExternalRunResult:
        raise NotImplementedError


class TradingAgentsWrapper(_BaseExternalWrapper):
    """Wrapper interface for TradingAgents (citation-only stub)."""

    name = "TradingAgents"
    citation = "TradingAgents paper / upstream repo (citation-only path)"

    def __init__(
        self,
        enabled: bool = False,
        runner: Optional[Callable[[Any], ExternalRunResult]] = None,
    ) -> None:
        self.enabled = enabled
        self.runner = runner

    def runnable_gate(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "TradingAgentsWrapper.enabled is False (citation-only)"
        if self.runner is None:
            return False, "TradingAgents native portfolio runner is not configured"
        return True, "ok"

    def _run_impl(self, snapshot_like: Any) -> ExternalRunResult:
        if self.runner is None:
            raise RuntimeError("TradingAgents native portfolio runner is not configured")
        return self.runner(snapshot_like)


class FinConWrapper(_BaseExternalWrapper):
    """Wrapper interface for FinCon (citation-only stub)."""

    name = "FinCon"
    citation = "FinCon paper / upstream repo (citation-only path)"

    def __init__(
        self,
        enabled: bool = False,
        runner: Optional[Callable[[Any], ExternalRunResult]] = None,
    ) -> None:
        self.enabled = enabled
        self.runner = runner

    def runnable_gate(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "FinConWrapper.enabled is False (citation-only)"
        if self.runner is None:
            return False, "FinCon native portfolio runner is not configured"
        return True, "ok"

    def _run_impl(self, snapshot_like: Any) -> ExternalRunResult:
        if self.runner is None:
            raise RuntimeError("FinCon native portfolio runner is not configured")
        return self.runner(snapshot_like)


__all__ = [
    "ExternalRunResult",
    "TradingAgentsWrapper",
    "FinConWrapper",
]
