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
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}. "
                      f"Retrying in {delay:.0f}s…")
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
        max_tokens: int = 2048,
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

        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
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


# ---------------------------------------------------------------------------
# OpenAI (GPT) Adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(AgentAdapter):
    """
    AgentAdapter backed by the OpenAI SDK (GPT-4o, GPT-4, etc.).

    Requires:
        pip install openai
        OPENAI_API_KEY set in .env or environment

    Args:
        model:          OpenAI model ID (e.g., "gpt-4o", "gpt-4-turbo").
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        system_prompt:  System-level instruction for every call.
        max_retries:    Number of retry attempts on transient errors.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
                             "Respond with structured JSON as instructed.",
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

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
        max_tokens: int = 2048,
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
