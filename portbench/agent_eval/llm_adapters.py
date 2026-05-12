"""
LLM-based AgentAdapter implementations for PortBench.

Provides concrete adapters for:
  - AnthropicAdapter   (Claude models via anthropic SDK)
  - OpenAIAdapter      (GPT-4o etc. via openai SDK)
  - LiteLLMAdapter     (any model via litellm — unified interface)

All adapters implement the same AgentAdapter ABC as MockAgentAdapter, so they
can be dropped into EvalPipeline with zero code changes.

Usage:
    from portbench.agent_eval.llm_adapters import AnthropicAdapter, LiteLLMAdapter

    # Claude
    adapter = AnthropicAdapter(model="claude-opus-4-6")
    pipeline = build_default_pipeline(adapter)

    # GPT-4o
    adapter = OpenAIAdapter(model="gpt-4o")

    # Any model via litellm (most flexible)
    adapter = LiteLLMAdapter(model="anthropic/claude-opus-4-6")
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    adapter = LiteLLMAdapter(model="ollama/llama3")
"""

import os
import time
import json
from typing import Optional

from dotenv import load_dotenv

from .base import AgentAdapter


load_dotenv()


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------


def _retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry fn() with exponential backoff on transient API errors.

    Retries on: RateLimitError, APIConnectionError, ServiceUnavailableError.
    Raises on: AuthenticationError, InvalidRequestError (non-retryable).
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(type(e).__name__).lower()
            # Non-retryable errors — fail immediately
            if any(k in err_str for k in ("auth", "invalid", "permission", "notfound")):
                raise
            if attempt < max_retries - 1:
                print(
                    f"  API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.0f}s…"
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ---------------------------------------------------------------------------
# Anthropic (Claude) Adapter
# ---------------------------------------------------------------------------


class AnthropicAdapter(AgentAdapter):
    """
    AgentAdapter backed by the Anthropic SDK (Claude models).

    Requires:
        pip install anthropic
        ANTHROPIC_API_KEY set in .env or environment

    Args:
        model:          Anthropic model ID (e.g., "claude-opus-4-6", "claude-sonnet-4-6").
        max_tokens:     Maximum tokens to generate in the response.
        temperature:    Sampling temperature (0.0 = deterministic).
        system_prompt:  Optional system-level instruction prepended to every call.
        max_retries:    Number of retry attempts on transient API errors.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        max_retries: int = 3,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return f"anthropic/{self._model}"

    def complete(self, prompt: str) -> str:
        """
        Call the Claude API and return the raw text response.

        Args:
            prompt: Full user-turn prompt (stage-specific instructions + market data).

        Returns:
            Model's response as a plain string.
        """

        def _call():
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        return _retry(_call, max_retries=self._max_retries)

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        """
        Call Claude with native tool use. Runs a multi-turn loop until end_turn
        or no more tool_use blocks, executing each tool call in Python.
        """
        from .tools import dispatch_tool

        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
        messages = [{"role": "user", "content": prompt}]

        for _ in range(10):  # max 10 tool rounds

            def _call(msgs=messages):
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=self._system_prompt,
                    tools=anthropic_tools,
                    messages=msgs,
                )

            response = _retry(_call, max_retries=self._max_retries)

            if response.stop_reason == "end_turn":
                # Return the last text block
                for block in reversed(response.content):
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if response.stop_reason != "tool_use":
                # Unexpected stop — return whatever text we have
                for block in reversed(response.content):
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # Execute tool calls and feed results back
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = dispatch_tool(block.name, block.input, tools)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            }
                        )
                    except Exception as exc:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Error: {exc}",
                                "is_error": True,
                            }
                        )

            # Append assistant turn + tool results to messages
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        # Fallback after max rounds
        return self.complete(prompt)


# ---------------------------------------------------------------------------
# OpenAI (GPT) Adapter
# ---------------------------------------------------------------------------


class OpenAIAdapter(AgentAdapter):
    """
    AgentAdapter backed by the OpenAI SDK.

    Works with any provider that exposes an OpenAI-compatible chat completions
    endpoint — including OpenAI, Qwen (DashScope), Kimi (Moonshot), DeepSeek, etc.

    Requires:
        pip install openai
        Appropriate API key set in environment (see api_key_env below)

    Args:
        model:          Model ID as accepted by the provider.
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        system_prompt:  System-level instruction for every call.
        max_retries:    Number of retry attempts on transient errors.
        base_url:       Custom API base URL for non-OpenAI providers.
                        e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1" (Qwen)
                             "https://api.moonshot.cn/v1"                         (Kimi)
                             "https://api.deepseek.com/v1"                        (DeepSeek)
        api_key_env:    Name of the environment variable holding the API key.
                        Defaults to "OPENAI_API_KEY".
        timeout:        Per-request timeout in seconds. Prevents hangs when an
                        upstream provider accepts the connection but never
                        returns. Default 60s.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        max_retries: int = 3,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        kwargs = {"api_key": os.environ.get(api_key_env), "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        """Call the OpenAI Chat Completions API and return the response text."""

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content

        return _retry(_call, max_retries=self._max_retries)

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        """
        Call GPT with native function/tool calling. Runs a multi-turn loop
        until the model produces a final text response with no pending tool calls.
        """
        from .tools import dispatch_tool

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        for _ in range(10):  # max 10 tool rounds

            def _call(msgs=messages):
                return self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    tools=openai_tools,
                    messages=msgs,
                )

            response = _retry(_call, max_retries=self._max_retries)
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason != "tool_calls":
                return choice.message.content or ""

            # Execute tool calls
            messages.append(choice.message.model_dump())
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = dispatch_tool(tc.function.name, args, tools)
                    content = str(result)
                except Exception as exc:
                    content = f"Error: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )

        return self.complete(prompt)


# ---------------------------------------------------------------------------
# LiteLLM Adapter (unified interface for 100+ models)
# ---------------------------------------------------------------------------


class LiteLLMAdapter(AgentAdapter):
    """
    AgentAdapter backed by litellm — a unified interface supporting 100+ LLM providers.

    Supports any model string that litellm accepts, e.g.:
      "anthropic/claude-opus-4-6"
      "openai/gpt-4o"
      "gemini/gemini-1.5-pro"
      "ollama/llama3"           (local Ollama instance)
      "huggingface/mistralai/Mistral-7B-Instruct-v0.1"

    Requires:
        pip install litellm
        Appropriate API key in environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

    Args:
        model:          litellm model string.
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        system_prompt:  System-level instruction.
        max_retries:    Number of retry attempts.
        api_base:       Optional custom API base URL (for local models / proxies).
        extra_params:   Additional kwargs forwarded to litellm.completion().
    """

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        max_retries: int = 3,
        api_base: Optional[str] = None,
        extra_params: Optional[dict] = None,
    ):
        try:
            import litellm

            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm package required. Install with: pip install litellm"
            )

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries
        self._api_base = api_base
        self._extra_params = extra_params or {}

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        """Call the model via litellm and return the response text."""
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            **self._extra_params,
        )
        if self._api_base:
            kwargs["api_base"] = self._api_base

        def _call():
            response = self._litellm.completion(**kwargs)
            return response.choices[0].message.content

        return _retry(_call, max_retries=self._max_retries)

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        """
        Call the model via litellm with tool-calling. Follows the OpenAI tool-call
        message format (supported by litellm across providers).
        """
        from .tools import dispatch_tool

        litellm_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        for _ in range(10):
            kwargs = dict(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tools=litellm_tools,
                **self._extra_params,
            )
            if self._api_base:
                kwargs["api_base"] = self._api_base

            def _call(kw=kwargs):
                return self._litellm.completion(**kw)

            response = _retry(_call, max_retries=self._max_retries)
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason != "tool_calls":
                return choice.message.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        tc.model_dump() for tc in (choice.message.tool_calls or [])
                    ],
                }
            )
            for tc in choice.message.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments)
                    result = dispatch_tool(tc.function.name, args, tools)
                    content = str(result)
                except Exception as exc:
                    content = f"Error: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )

        return self.complete(prompt)
