"""External baseline wrappers are citation-only by default."""

from __future__ import annotations

import pytest

from portbench.baselines.external_wrappers import FinConWrapper, TradingAgentsWrapper


def test_runnable_gate_false_by_default():
    ta = TradingAgentsWrapper()
    ok, reason = ta.runnable_gate()
    assert ok is False
    assert "citation" in reason.lower() or "enabled" in reason.lower()
    with pytest.raises(RuntimeError):
        ta.run({"current_weights": {"A": 1.0}})

    fc = FinConWrapper()
    ok2, _ = fc.runnable_gate()
    assert ok2 is False