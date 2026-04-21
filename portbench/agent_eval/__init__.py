"""PortBench agent evaluation module."""

from .base import (
    AgentAdapter,
    EpisodeResult,
    EvalPipeline,
    MarketSnapshot,
    PipelineStage,
    RiskAlert,
    S1Output, S2Output, S3Output, S4Output, S5Output,
    StageID,
    TradeOrder,
)
from .mock_agent import MockAgentAdapter
from .llm_adapters import AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter
from .local_adapter import VLLMAdapter, OllamaAdapter, HuggingFaceAdapter
from .eval_logger import EvalLogger, EpisodeLog, StageLog
from .stages import (
    S1MarketInterpretation,
    S2SignalGeneration,
    S3WeightOptimization,
    S4ExecutionSimulation,
    S5RiskMonitoring,
)
from .stress_scenarios import ScenarioInjector, StressScenario, STRESS_SCENARIOS


from .tools import ToolSpec, get_tools, dispatch_tool, BUILTIN_TOOLS


def build_default_pipeline(adapter: AgentAdapter = None, use_tools: bool = False) -> EvalPipeline:
    """
    Construct a default five-stage EvalPipeline with the given adapter.

    Args:
        adapter:    AgentAdapter to use. Defaults to MockAgentAdapter if None.
        use_tools:  If True, S1/S2/S3 stages call complete_with_tools() instead of
                    complete(), enabling multi-turn tool execution for cloud adapters.

    Returns:
        EvalPipeline ready to call run_episode().
    """
    if adapter is None:
        adapter = MockAgentAdapter()

    stages = [
        S1MarketInterpretation(adapter, use_tools=use_tools),
        S2SignalGeneration(adapter, use_tools=use_tools),
        S3WeightOptimization(adapter, use_tools=use_tools),
        S4ExecutionSimulation(adapter),
        S5RiskMonitoring(adapter),
    ]
    return EvalPipeline(stages)


__all__ = [
    # Base
    "StageID", "MarketSnapshot",
    "S1Output", "S2Output", "S3Output", "S4Output", "S5Output",
    "TradeOrder", "RiskAlert",
    "PipelineStage", "AgentAdapter",
    "EpisodeResult", "EvalPipeline",
    # Cloud adapters
    "MockAgentAdapter",
    "AnthropicAdapter", "OpenAIAdapter", "LiteLLMAdapter",
    # Local adapters
    "VLLMAdapter", "OllamaAdapter", "HuggingFaceAdapter",
    # Stages
    "S1MarketInterpretation", "S2SignalGeneration", "S3WeightOptimization",
    "S4ExecutionSimulation", "S5RiskMonitoring",
    # Stress testing
    "StressScenario", "ScenarioInjector", "STRESS_SCENARIOS",
    # Logging
    "EvalLogger", "EpisodeLog", "StageLog",
    # Tools
    "ToolSpec", "get_tools", "dispatch_tool", "BUILTIN_TOOLS",
    # Factory
    "build_default_pipeline",
]
