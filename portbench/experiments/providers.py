"""
Provider registry: build LLM adapters from .env without touching adapter source.

Each provider entry maps a short key (e.g. "tencent") to:
  - env_prefix      e.g. "TENCENT" → reads TENCENT_API_KEY / TENCENT_BASE_URL / TENCENT_MODEL
  - adapter_class   the AgentAdapter subclass to instantiate
  - kind            "openai_compat" | "anthropic" — picks construction kwargs

To add a new provider:
  1. Add an entry to PROVIDER_REGISTRY below.
  2. Add three vars to .env: NEWPROV_API_KEY / NEWPROV_BASE_URL / NEWPROV_MODEL.
No adapter code changes required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from ..agent_eval.base import AgentAdapter
from ..agent_eval.llm_adapters import AnthropicAdapter, OpenAIAdapter
from ..agent_eval.mock_agent import MockAgentAdapter
from ..baselines import (
    CovarianceRiskParityBaseline,
    EqualWeightBaseline,
    RiskParityBaseline,
    SixtyFortyBaseline,
)


load_dotenv()


@dataclass(frozen=True)
class ProviderSpec:
    env_prefix: str
    kind: str  # "openai_compat" | "anthropic"


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "dashscope": ProviderSpec("DASHSCOPE", "openai_compat"),
    "tencent": ProviderSpec("TENCENT", "openai_compat"),
    "deepseek": ProviderSpec("DEEPSEEK", "openai_compat"),
    "glm": ProviderSpec("GLM", "openai_compat"),
    "kimi": ProviderSpec("KIMI", "openai_compat"),
    "minimax": ProviderSpec("MINIMAX", "openai_compat"),
    "ark": ProviderSpec("ARK", "openai_compat"),
    "openai": ProviderSpec("OPENAI", "openai_compat"),
    "anthropic": ProviderSpec("ANTHROPIC", "anthropic"),
    "google": ProviderSpec("GOOGLE", "openai_compat"),
}


BASELINE_REGISTRY = {
    "equal_weight": EqualWeightBaseline,
    "sixty_forty": SixtyFortyBaseline,
    "risk_parity": RiskParityBaseline,
    "cov_risk_parity": CovarianceRiskParityBaseline,
}


def _env(prefix: str, suffix: str) -> Optional[str]:
    val = os.environ.get(f"{prefix}_{suffix}")
    return val.strip() if val else None


def build_adapter(
    provider: str,
    model: Optional[str] = None,
    **adapter_kwargs,
) -> AgentAdapter:
    """
    Build an AgentAdapter for the given provider key.

    Reads from .env:
        {PREFIX}_API_KEY   — required
        {PREFIX}_BASE_URL  — optional (anthropic ignores it)
        {PREFIX}_MODEL     — fallback when `model` arg is None

    Raises RuntimeError if the API key is missing — no silent fallback.
    """
    key = provider.lower()
    if key not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Known: {sorted(PROVIDER_REGISTRY)}. "
            f"Add it to PROVIDER_REGISTRY in portbench/experiments/providers.py."
        )
    spec = PROVIDER_REGISTRY[key]

    api_key = _env(spec.env_prefix, "API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing {spec.env_prefix}_API_KEY in environment / .env. "
            f"Provider '{provider}' cannot be constructed."
        )

    resolved_model = model or _env(spec.env_prefix, "MODEL")
    if not resolved_model:
        raise RuntimeError(
            f"No model specified for provider '{provider}': pass model=... or "
            f"set {spec.env_prefix}_MODEL in .env."
        )

    if spec.kind == "openai_compat":
        base_url = _env(spec.env_prefix, "BASE_URL")
        kwargs = dict(
            model=resolved_model,
            api_key_env=f"{spec.env_prefix}_API_KEY",
        )
        if base_url:
            kwargs["base_url"] = base_url
        kwargs.update(adapter_kwargs)
        return OpenAIAdapter(**kwargs)

    if spec.kind == "anthropic":
        # AnthropicAdapter currently reads ANTHROPIC_API_KEY; if user mapped
        # to a different prefix, copy it through so the SDK picks it up.
        if spec.env_prefix != "ANTHROPIC":
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        return AnthropicAdapter(model=resolved_model, **adapter_kwargs)

    raise RuntimeError(f"Unhandled provider kind: {spec.kind}")


def build_baseline(name: str) -> AgentAdapter:
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. Known: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[name]()


def build_mock(noise: float = 0.2, seed: int = 42) -> AgentAdapter:
    return MockAgentAdapter(noise_level=noise, seed=seed)


def model_label(provider: str, model: Optional[str]) -> str:
    """Human-readable label used as a directory name in batch outputs."""
    spec = PROVIDER_REGISTRY[provider.lower()]
    resolved = model or _env(spec.env_prefix, "MODEL") or "default"
    safe = resolved.replace("/", "_").replace(":", "_")
    return f"{provider.lower()}-{safe}"


def spec_provider_name(spec) -> str:
    """Return the Level-2 directory name for a ModelSpec: provider key or 'baseline'/'mock'."""
    if spec.baseline:
        return "baseline"
    if spec.mock:
        return "mock"
    return spec.provider.lower()


def spec_model_name(spec, resolved_model: Optional[str] = None) -> str:
    """Return the Level-3 directory name for a ModelSpec: model identifier string."""
    if spec.baseline:
        return spec.baseline
    if spec.mock:
        return "mock"
    model = resolved_model or spec.model
    if not model:
        prov_spec = PROVIDER_REGISTRY[spec.provider.lower()]
        model = _env(prov_spec.env_prefix, "MODEL") or "default"
    return model.replace("/", "_").replace(":", "_")
