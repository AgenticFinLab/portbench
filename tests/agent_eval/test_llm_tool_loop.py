"""Tool-loop helpers ignore finish_reason / stop_reason when calls are present."""

from __future__ import annotations

from types import SimpleNamespace

from portbench.agent_eval.llm_adapters import (
    _anthropic_tool_use_blocks,
    _choice_text,
    _message_tool_calls,
)


def test_message_tool_calls_ignores_stop_finish_reason():
    tool_call = SimpleNamespace(id="1")
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="{}", tool_calls=[tool_call]),
    )
    assert _message_tool_calls(choice) == [tool_call]


def test_choice_without_tool_calls_returns_text():
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content='{"action":"hold"}', tool_calls=None),
    )
    assert _message_tool_calls(choice) == []
    assert _choice_text(choice) == '{"action":"hold"}'


def test_anthropic_reads_tool_use_on_end_turn():
    tool_block = SimpleNamespace(type="tool_use", id="t", name="calculator", input={})
    text_block = SimpleNamespace(type="text", text="")
    response = SimpleNamespace(stop_reason="end_turn", content=[tool_block, text_block])
    assert _anthropic_tool_use_blocks(response) == [tool_block]
