"""Architecture registry and MA frozen message schema."""

from __future__ import annotations

import pytest

from portbench.agent_eval.architectures import (
    ARCHITECTURE_REGISTRY,
    ArchitectureRuntime,
    BudgetExceeded,
    IsoTokenBudget,
    MARoleMessage,
    MemoryStore,
    ResourceLedger,
    build_ma_role_adapters,
    get_architecture,
)
from portbench.agent_eval.base import AgentAdapter
from portbench.agent_eval.contracts import ARCHITECTURE_IDS


class NoCallAdapter(AgentAdapter):
    """Provide an adapter identity for runtime-only tests."""

    @property
    def model_name(self) -> str:
        return "no-call"

    def complete(self, prompt: str) -> str:
        raise AssertionError("provider call is not expected")


def test_eight_architectures_registered():
    assert len(ARCHITECTURE_REGISTRY) == 8
    assert set(ARCHITECTURE_REGISTRY) == set(ARCHITECTURE_IDS)
    sa_mt = get_architecture("SA-MT")
    assert sa_mt.shared_agent is True
    assert sa_mt.memory_enabled is True
    assert sa_mt.tools_enabled is True
    ma = get_architecture("MA")
    assert ma.shared_agent is False
    assert ma.memory_enabled is False
    assert ma.tools_enabled is False


def test_ma_roles_and_frozen_schema():
    adapters = build_ma_role_adapters()
    assert set(adapters) == {"analyst", "signal", "optimizer", "executor", "risk"}
    msg = adapters["analyst"].emit(
        episode_id="ep1",
        decision_date="2020-01-02",
        payload={"view": "risk_on"},
        confidence=0.7,
        rationale_tags=["momentum"],
    )
    assert msg["role"] == "analyst"
    assert "payload" in msg
    with pytest.raises(ValueError):
        MARoleMessage(role="debater", episode_id="x", decision_date="y").to_dict()


def test_iso_token_budget_contract():
    budget = IsoTokenBudget()
    assert budget.config_version == "iso-token-v3"
    assert budget.max_tokens_per_episode == 32000
    assert budget.max_requests_per_episode == 24


def test_resource_ledger_keeps_exact_and_estimated_separate():
    ledger = ResourceLedger(IsoTokenBudget(max_tokens_per_episode=10))
    ledger.begin_episode()
    ledger.record(token_exact=4, token_est=5, request_count=1)
    assert ledger.usage.token_exact == 4
    assert ledger.usage.token_est == 5
    with pytest.raises(BudgetExceeded, match="episode tokens 11"):
        ledger.record(token_est=2)


def test_resource_ledger_zero_ceiling_is_unlimited():
    ledger = ResourceLedger(IsoTokenBudget(max_tokens_per_episode=0, max_requests_per_episode=0))
    ledger.begin_episode()
    ledger.record(token_exact=50000, token_est=1, request_count=100)
    assert ledger.usage.token_exact == 50000
    assert ledger.usage.request_count == 100


def test_memory_branch_is_copy_on_write(tmp_path):
    factual_path = tmp_path / "factual.json"
    factual = MemoryStore(factual_path)
    factual.append("S1", "2024-01-01", "risk off")
    original_hash = factual.state_hash()

    branch = factual.clone(path=tmp_path / "branch.json")
    branch.append("S2", "2024-01-01", "hold")
    assert factual.state_hash() == original_hash
    assert len(factual.entries) == 1
    assert len(branch.entries) == 2

    frozen = factual.clone(read_only=True)
    with pytest.raises(PermissionError, match="read-only"):
        frozen.append("S3", "2024-01-01", "weights")


def test_intervention_branch_temporarily_overrides_max_tokens():
    adapter = NoCallAdapter()
    adapter._max_tokens = 4096
    runtime = ArchitectureRuntime(adapter, "SA", schema_version="pipeline-v4-sa-causal")

    with runtime.intervention_branch("s1_repair", max_tokens=16384):
        assert runtime.cache_namespace == "intervention__s1_repair"
        assert adapter._max_tokens == 16384

    assert runtime.cache_namespace == "factual"
    assert adapter._max_tokens == 4096

def test_ma_nodes_have_private_copy_on_write_memories(tmp_path):
    factual = ArchitectureRuntime(
        NoCallAdapter(),
        "MA-M",
        memory_path=tmp_path / "factual.json",
    )
    branch = ArchitectureRuntime(
        NoCallAdapter(),
        "MA-M",
        memory_path=tmp_path / "branch.json",
    )
    factual.agent_nodes["risk"].memory.append("S3-risk", "2024-01-01", "reduce SPY")
    factual.agent_nodes["optimizer"].memory.append("S3", "2024-01-01", "balanced")
    branch.clone_memory_from(factual)
    branch.agent_nodes["risk"].memory.append("S5", "2024-01-02", "hold")
    assert len(factual.agent_nodes["risk"].memory.entries) == 1
    assert len(branch.agent_nodes["risk"].memory.entries) == 2
    assert factual.agent_nodes["risk"].memory.state_hash() != branch.agent_nodes["risk"].memory.state_hash()
    assert factual.agent_nodes["risk"].memory.state_hash() != factual.agent_nodes["optimizer"].memory.state_hash()


def test_message_changes_inbox_hash():
    runtime = ArchitectureRuntime(NoCallAdapter(), "MA")
    runtime.begin_episode("ep1", "2024-01-01", "snapshot")
    before = runtime.message_bus.inbox_hash("risk")
    runtime.publish(
        sender="optimizer",
        recipient="risk",
        stage_id="S3",
        round_id="proposal",
        message_type="candidate",
        payload={"weights": {"SPY": 1.0}},
    )
    assert runtime.message_bus.inbox_hash("risk") != before
    assert runtime.message_bus.inbox_hash("executor") == before
