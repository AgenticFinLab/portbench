"""Tool-enabled stage calls go through ReplayAdapter."""

from __future__ import annotations

from portbench.agent_eval.architectures import ArchitectureRuntime, IsoTokenBudget
from portbench.agent_eval.base import AgentAdapter
from portbench.agent_eval.tools import get_tools


class CountingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "count"

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return "plain"

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        self.calls += 1
        return "with-tools"


def test_tool_complete_second_call_hits_replay_adapter(tmp_path):
    base = CountingAdapter()
    runtime = ArchitectureRuntime(
        base,
        "SA-T",
        cache_dir=tmp_path,
        budget=IsoTokenBudget(max_tokens_per_episode=10000, max_requests_per_episode=10),
        provider="test",
        data_version="v",
        code_commit="c",
    )
    runtime.begin_episode("ep", "2024-01-02", "snap-hash")
    stage = runtime.stage_adapter("S4")
    tools = get_tools()
    first = stage._complete("prompt-a", tools=tools)
    second = stage._complete("prompt-a", tools=tools)
    assert first == "with-tools"
    assert second == "with-tools"
    assert base.calls == 1
