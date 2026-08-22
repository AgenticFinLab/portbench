"""pit_repair_v1 must refuse future_return_data."""

import pytest

from portbench.agent_eval import pit_repair


LOOKBACK = {"SPY": [0.01, -0.02, 0.005], "TLT": [0.0, 0.001, -0.001]}


@pytest.mark.parametrize(
    "fn",
    [
        pit_repair.repair_s1,
        pit_repair.repair_s2,
        pit_repair.repair_s3,
        pit_repair.repair_s4,
        pit_repair.repair_s5,
    ],
)
def test_future_return_data_kwarg_raises(fn):
    with pytest.raises((PermissionError, ValueError)):
        fn(lookback_returns=LOOKBACK, future_return_data={"SPY": [0.99]})


@pytest.mark.parametrize(
    "fn",
    [
        pit_repair.repair_s1,
        pit_repair.repair_s2,
        pit_repair.repair_s3,
        pit_repair.repair_s4,
        pit_repair.repair_s5,
    ],
)
def test_future_return_data_in_context_raises(fn):
    with pytest.raises((PermissionError, ValueError)):
        fn({"lookback_returns": LOOKBACK, "future_return_data": {"SPY": [1.0]}})


def test_s1_s2_deterministic_without_future():
    a = pit_repair.repair_s1(lookback_returns=LOOKBACK)
    b = pit_repair.repair_s1(lookback_returns=LOOKBACK)
    assert a == b
    assert a["repair_version"] == pit_repair.VERSION
    s2 = pit_repair.repair_s2(lookback_returns=LOOKBACK)
    assert s2["repair_version"] == pit_repair.VERSION
    assert set(s2["signals"]) == set(LOOKBACK)
