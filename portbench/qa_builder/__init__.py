"""PortBench QA dataset builder module."""

from .base import (
    ComplexityLevel,
    ContextWindow,
    DataProvider,
    MarketRegime,
    QABuilder,
    QAConfig,
    QAPair,
    Split,
)
from .mock_data import MockDataProvider
from .processed_data import ProcessedDataProvider
from .t1_return_prediction import T1ReturnPrediction
from .t2_risk_assessment import T2RiskAssessment
from .t3_position_sizing import T3PositionSizing
from .t4_pairwise_allocation import T4PairwiseAllocation
from .t5_multiasset_optimization import T5MultiAssetOptimization
from .t6_rebalancing import T6RebalancingDecision
from .t7_regime_detection import T7RegimeDetection


def get_all_builders(
    provider: DataProvider,
    config: QAConfig,
) -> list[QABuilder]:
    """
    Instantiate all seven template builders with the given provider and config.

    Returns a list ordered T1 → T7.
    """
    return [
        T1ReturnPrediction(provider, config),
        T2RiskAssessment(provider, config),
        T3PositionSizing(provider, config),
        T4PairwiseAllocation(provider, config),
        T5MultiAssetOptimization(provider, config),
        T6RebalancingDecision(provider, config),
        T7RegimeDetection(provider, config),
    ]


__all__ = [
    # Base
    "MarketRegime", "Split", "ComplexityLevel",
    "ContextWindow", "QAPair", "QAConfig",
    "DataProvider", "QABuilder",
    # Data providers
    "MockDataProvider",
    "ProcessedDataProvider",
    # Template builders
    "T1ReturnPrediction",
    "T2RiskAssessment",
    "T3PositionSizing",
    "T4PairwiseAllocation",
    "T5MultiAssetOptimization",
    "T6RebalancingDecision",
    "T7RegimeDetection",
    # Convenience
    "get_all_builders",
]
