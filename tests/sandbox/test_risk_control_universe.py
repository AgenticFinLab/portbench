"""Risk controls must keep final holdings inside the tradable universe."""

import pandas as pd

from portbench.agent_eval.contracts import RiskControlDecision
from portbench.sandbox.risk_control import evaluate_risk


def test_risk_hold_materializes_residual_in_tradable_cash_proxy():
    result = evaluate_risk(
        RiskControlDecision(action="hold"),
        weights={"SPY": 0.6, "BIL": 0.3996},
        return_data={
            "SPY": pd.Series([0.001] * 32),
            "BIL": pd.Series([0.0001] * 32),
        },
    )

    final = result.metadata["final_weights"]
    assert set(final) == {"SPY", "BIL"}
    assert final["BIL"] == 0.4
