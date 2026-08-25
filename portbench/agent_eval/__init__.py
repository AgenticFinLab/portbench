"""PortBench agent evaluation module."""

from .base import (
    AgentAdapter,
    EpisodeResult,
    EvalPipeline,
    MarketSnapshot,
    PipelineStage,
    RiskAlert,
    S1Output,
    S2Output,
    S3Output,
    S4Output,
    S5Output,
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
from .investor_profiles import (
    InvestorProfile,
    PROFILES,
    ProfileAlignmentScorer,
    ProfiledPipeline,
)


from .tools import ToolSpec, get_tools, dispatch_tool, BUILTIN_TOOLS


def build_default_pipeline(
    adapter: AgentAdapter = None,
    use_tools: bool = False,
    profile=None,
    oracle_mode: str = "ex_post",
    architecture_id: str | None = None,
    cache_dir: str | None = None,
    memory_path: str | None = None,
    budget=None,
    provider: str = "",
    profile_name: str = "",
    data_version: str = "",
    code_commit: str = "",
    call_max_attempts: int = 3,
    retry_failed_calls: bool = False,
    schema_version: str = "pipeline-v3-collab",
    call_artifact_dir: str | None = None,
) -> EvalPipeline:
    """
    Construct a default five-stage EvalPipeline with the given adapter.

    Args:
        adapter:    AgentAdapter to use. REQUIRED — passing None raises ValueError
                    so production runs cannot silently fall back to MockAgentAdapter.
        use_tools:  If True, S1/S2/S3 stages call complete_with_tools() instead of
                    complete(), enabling multi-turn tool execution for cloud adapters.
        profile:    Optional InvestorProfile. When provided, S5 uses the profile's
                    var_limit and max_drawdown_tolerance as alert thresholds instead
                    of the class-level conservative defaults.
        oracle_mode: S3 ground-truth oracle mode:
                     "ex_post" — max-Sharpe using future returns (default)
                     "lookback" — max-Sharpe using only historical returns
                     "equal_weight" — equal-weight baseline oracle

    Returns:
        EvalPipeline ready to call run_episode().
    """
    if adapter is None:
        raise ValueError(
            "build_default_pipeline requires an explicit AgentAdapter. "
            "Mock fallback is disabled — pass MockAgentAdapter() yourself if "
            "that is genuinely what you want."
        )

    if architecture_id is None:
        stages = [
            S1MarketInterpretation(adapter, use_tools=use_tools),
            S2SignalGeneration(adapter, use_tools=use_tools),
            S3WeightOptimization(adapter, use_tools=use_tools, oracle_mode=oracle_mode),
            S4ExecutionSimulation(adapter),
            S5RiskMonitoring(adapter, profile=profile),
        ]
        return EvalPipeline(stages)

    from .agentic_pipeline_stages import AgenticS4PipelineStage, AgenticS5PipelineStage
    from .architectures import ArchitectureRuntime

    runtime = ArchitectureRuntime(
        adapter,
        architecture_id,
        cache_dir=cache_dir,
        memory_path=memory_path,
        budget=budget,
        provider=provider,
        profile=profile_name,
        data_version=data_version,
        code_commit=code_commit,
        call_max_attempts=call_max_attempts,
        retry_failed_calls=retry_failed_calls,
        schema_version=schema_version,
        call_artifact_dir=call_artifact_dir,
    )
    tools_enabled = runtime.spec.tools_enabled
    if runtime.spec.shared_agent:
        s3_stage = S3WeightOptimization(
            runtime.stage_adapter("S3"),
            use_tools=tools_enabled,
            oracle_mode=oracle_mode,
        )
    else:
        from .collaboration import CollaborativeS3WeightOptimization

        s3_stage = CollaborativeS3WeightOptimization(runtime, oracle_mode=oracle_mode)
    stages = [
        S1MarketInterpretation(runtime.stage_adapter("S1"), use_tools=tools_enabled),
        S2SignalGeneration(runtime.stage_adapter("S2"), use_tools=tools_enabled),
        s3_stage,
        AgenticS4PipelineStage(
            runtime.stage_adapter("S4"),
            oracle_mode=oracle_mode,
            use_tools=tools_enabled,
            schema_version=runtime.schema_version,
        ),
        AgenticS5PipelineStage(
            runtime.stage_adapter("S5"),
            profile=profile,
            use_tools=tools_enabled,
            schema_version=runtime.schema_version,
        ),
    ]
    return EvalPipeline(stages, runtime=runtime)


__all__ = [
    # Base
    "StageID",
    "MarketSnapshot",
    "S1Output",
    "S2Output",
    "S3Output",
    "S4Output",
    "S5Output",
    "TradeOrder",
    "RiskAlert",
    "PipelineStage",
    "AgentAdapter",
    "EpisodeResult",
    "EvalPipeline",
    # Cloud adapters
    "MockAgentAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    # Local adapters
    "VLLMAdapter",
    "OllamaAdapter",
    "HuggingFaceAdapter",
    # Stages
    "S1MarketInterpretation",
    "S2SignalGeneration",
    "S3WeightOptimization",
    "S4ExecutionSimulation",
    "S5RiskMonitoring",
    # Stress testing
    "StressScenario",
    "ScenarioInjector",
    "STRESS_SCENARIOS",
    # Investor profiles
    "InvestorProfile",
    "PROFILES",
    "ProfileAlignmentScorer",
    "ProfiledPipeline",
    # Logging
    "EvalLogger",
    "EpisodeLog",
    "StageLog",
    # Tools
    "ToolSpec",
    "get_tools",
    "dispatch_tool",
    "BUILTIN_TOOLS",
    # Factory
    "build_default_pipeline",
]
