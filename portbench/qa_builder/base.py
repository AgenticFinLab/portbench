"""
Base classes and data structures for the PortBench QA dataset builder.

The QA builder generates structured question-answer pairs from historical (or mock)
market data using seven question templates (T1–T7) at four complexity levels.

Each QA pair contains:
  - A point-in-time context window (price history + macro + news)
  - A question text
  - A ground-truth answer with explanation
  - Metadata: template_id, complexity_level, market_regime, split, asset_class

Design principles:
  - PiT safety: context is strictly constructed from data before decision_date
  - Modular: each template is a separate subclass of QABuilder
  - Mock-friendly: all builders accept a DataProvider interface so real and
    synthetic data can be swapped without code changes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Market regime labels used for stratifying the QA dataset."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class Split(Enum):
    """Train / validation / test split."""
    TRAIN = "train"       # 2015-01-01 – 2022-12-31
    VAL = "val"           # 2023-01-01 – 2023-12-31
    TEST = "test"         # 2024-01-01 – 2025-12-31


class ComplexityLevel(Enum):
    """
    Complexity levels as defined in docs/project-overview.md §2.2.
      1 = single asset  (T1, T2, T3)
      2 = pairwise      (T4)
      3 = multi-asset   (T5, T6)
      4 = full portfolio with regime detection (T7)
    """
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

@dataclass
class ContextWindow:
    """
    A point-in-time snapshot of market information available at decision_date.

    All data in this object must be strictly prior to decision_date (PiT constraint).

    Attributes:
        decision_date:    The date at which the investment decision is made.
        assets:           List of asset identifiers included in the context.
        price_history:    Dict mapping asset -> pd.Series of daily close prices
                          (index = date, values = price). Ends strictly before decision_date.
        returns_history:  Dict mapping asset -> pd.Series of daily returns.
        macro_context:    Dict of macro indicator name -> scalar value at decision_date.
        news_text:        Raw news/filing text available before decision_date. May be empty.
        market_regime:    Detected regime at decision_date (if pre-labeled).
        correlation_matrix: pd.DataFrame of pairwise return correlations (assets × assets),
                          computed from returns_history. None if fewer than 2 assets.
                          Captures the cross-asset correlation structure for T4/T5/T7 templates
                          and the new CorrelationMetrics evaluation dimension.
    """

    decision_date: date
    assets: list[str]
    price_history: dict[str, pd.Series]       # asset -> close price series
    returns_history: dict[str, pd.Series]     # asset -> return series
    macro_context: dict[str, float] = field(default_factory=dict)
    news_text: str = ""
    market_regime: Optional[MarketRegime] = None
    correlation_matrix: Optional["pd.DataFrame"] = None   # assets × assets

    def compute_correlation(self) -> "pd.DataFrame":
        """
        Compute the pairwise Pearson correlation matrix from returns_history.

        Aligns all return series on a common date index before computing,
        to handle different trading calendars or missing data.

        Returns:
            pd.DataFrame (n_assets × n_assets) with asset names as index/columns.
            Diagonal = 1.0, off-diagonal = Pearson correlation in [-1, 1].
        """
        df = pd.DataFrame({a: s for a, s in self.returns_history.items()})
        return df.corr(method="pearson")

    def validate_pit(self) -> None:
        """
        Assert that no data in this context leaks beyond decision_date.
        Raises ValueError if a look-ahead violation is detected.
        """
        cutoff = pd.Timestamp(self.decision_date)
        for asset, series in self.price_history.items():
            if not series.empty and series.index.max() >= cutoff:
                raise ValueError(
                    f"PiT violation: price_history['{asset}'] contains data "
                    f"on or after decision_date {self.decision_date}."
                )
        for asset, series in self.returns_history.items():
            if not series.empty and series.index.max() >= cutoff:
                raise ValueError(
                    f"PiT violation: returns_history['{asset}'] contains data "
                    f"on or after decision_date {self.decision_date}."
                )


# ---------------------------------------------------------------------------
# QA pair
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A single question-answer pair in the PortBench QA dataset.

    Attributes:
        qa_id:            Unique identifier, e.g. "T1_equities_20200315_001".
        template_id:      Template used: "T1" … "T7".
        complexity:       ComplexityLevel enum.
        split:            Train / val / test split.
        market_regime:    Market regime at decision_date.
        asset_class:      Primary asset class (for indexing).
        assets:           List of assets involved in the question.
        decision_date:    Date of the decision scenario.
        context_summary:  Short human-readable description of the market context.
        question:         Full question text (as would be presented to an LLM).
        answer:           Ground-truth answer (string representation).
        answer_numeric:   Numeric ground-truth when applicable (e.g., 0.42 for a weight).
        explanation:      Step-by-step derivation of the ground-truth answer.
        metadata:         Arbitrary extra data (e.g., raw numbers used in computation).
    """

    qa_id: str
    template_id: str
    complexity: ComplexityLevel
    split: Split
    market_regime: MarketRegime
    asset_class: str
    assets: list[str]
    decision_date: date
    context_summary: str
    question: str
    answer: str
    answer_numeric: Optional[float] = None
    explanation: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict."""
        return {
            "id": self.qa_id,
            "template": self.template_id,
            "complexity": self.complexity.value,
            "split": self.split.value,
            "market_regime": self.market_regime.value,
            "asset_class": self.asset_class,
            "assets": self.assets,
            "decision_date": str(self.decision_date),
            "context_summary": self.context_summary,
            "question": self.question,
            "answer": self.answer,
            "answer_numeric": self.answer_numeric,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QAConfig:
    """
    Configuration for the QA dataset builder.

    Attributes:
        lookback_days:   Number of trading days of price history to include in context.
        horizon_days:    Forecast horizon for return-prediction templates (T1, T6).
        samples_per_template: Target number of QA pairs per template per split.
        random_seed:     Reproducibility seed.
        train_start/end, val_start/end, test_start/end: Split boundaries.
            Set these to None to trigger auto-computation from the data timeline
            (see QAConfig.from_date_range()).

    Dynamic split computation:
        Use QAConfig.from_date_range(start, end) to derive split boundaries
        automatically. The heuristic is: train=70%, val=15%, test=15% of the
        total date range (rounded to whole calendar years at year boundaries).
    """

    lookback_days: int = 60               # ~3 months of trading history
    horizon_days: int = 21                # ~1 month forward horizon
    samples_per_template: int = 100       # Per split per template
    random_seed: int = 42

    train_start: str = "2015-01-01"
    train_end:   str = "2022-12-31"
    val_start:   str = "2023-01-01"
    val_end:     str = "2023-12-31"
    test_start:  str = "2024-01-01"
    test_end:    str = "2025-12-31"

    @classmethod
    def from_date_range(
        cls,
        data_start: date,
        data_end: date,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        **kwargs,
    ) -> "QAConfig":
        """
        Build a QAConfig with split boundaries derived from the actual data timeline.

        The split is computed as a fraction of the total calendar span:
          train = first train_frac of the span
          val   = next val_frac
          test  = remainder (1 - train_frac - val_frac)

        Boundaries are snapped to year-start to produce clean cutoffs.

        Args:
            data_start:  Earliest date in the merged multi-asset dataset.
            data_end:    Latest date in the merged multi-asset dataset.
            train_frac:  Fraction of the total span allocated to training (default 0.70).
            val_frac:    Fraction allocated to validation (default 0.15).
            **kwargs:    Additional QAConfig fields (lookback_days, etc.).

        Returns:
            QAConfig with train/val/test boundaries computed from the date range.

        Example:
            # Dataset spans 2010-01-01 to 2025-12-31 (15 years)
            # → train: 2010–2020 (10.5y), val: 2021–2022 (2.25y), test: 2023–2025 (2.25y)
            config = QAConfig.from_date_range(date(2010, 1, 1), date(2025, 12, 31))
        """
        import math
        total_days = (data_end - data_start).days
        if total_days < 365 * 3:
            raise ValueError(
                f"Date range {data_start} – {data_end} is too short ({total_days} days). "
                "Need at least 3 years for a meaningful train/val/test split."
            )

        train_end_raw = data_start + timedelta(days=int(total_days * train_frac))
        val_end_raw   = data_start + timedelta(days=int(total_days * (train_frac + val_frac)))

        # Snap to year boundaries (first day of the following year) for clean cutoffs
        def snap_year_end(d: date) -> date:
            return date(d.year, 12, 31)

        def snap_year_start(d: date) -> date:
            return date(d.year + 1, 1, 1)

        train_end   = snap_year_end(train_end_raw)
        val_start   = snap_year_start(train_end_raw)
        val_end     = snap_year_end(val_end_raw)
        test_start  = snap_year_start(val_end_raw)
        test_end    = data_end

        return cls(
            train_start=str(data_start),
            train_end=str(train_end),
            val_start=str(val_start),
            val_end=str(val_end),
            test_start=str(test_start),
            test_end=str(test_end),
            **kwargs,
        )

    def get_split(self, d: date) -> Optional[Split]:
        """Return the Split for a given date, or None if outside all splits."""
        ds = str(d)
        if self.train_start <= ds <= self.train_end:
            return Split.TRAIN
        if self.val_start <= ds <= self.val_end:
            return Split.VAL
        if self.test_start <= ds <= self.test_end:
            return Split.TEST
        return None

    def describe(self) -> str:
        """Return a human-readable description of the split boundaries."""
        return (
            f"Train: {self.train_start} – {self.train_end}  |  "
            f"Val: {self.val_start} – {self.val_end}  |  "
            f"Test: {self.test_start} – {self.test_end}"
        )


# ---------------------------------------------------------------------------
# Data provider interface
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """
    Abstract interface for providing market data to QA builders.

    Concrete implementations:
      - MockDataProvider (portbench/qa_builder/mock_data.py) — synthetic data
      - ProcessedDataProvider (future) — reads from datasets/processed/*.csv

    This abstraction lets templates be developed and tested with mock data
    before real datasets are finalized.
    """

    @abstractmethod
    def get_price_series(
        self,
        asset: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        Return daily close prices for `asset` from `start` to `end` (inclusive).
        Index = pd.DatetimeIndex, values = float price.
        """
        pass

    @abstractmethod
    def get_return_series(
        self,
        asset: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        Return daily simple returns for `asset` from `start` to `end` (inclusive).
        """
        pass

    @abstractmethod
    def get_macro(self, d: date) -> dict[str, float]:
        """Return macro indicator snapshot available at date d."""
        pass

    @abstractmethod
    def get_regime(self, d: date, asset: str = "SPY") -> MarketRegime:
        """Return the market regime label at date d."""
        pass

    @abstractmethod
    def list_assets(self, asset_class: Optional[str] = None) -> list[str]:
        """List available asset identifiers, optionally filtered by asset class."""
        pass

    def get_news(self, asset: str, before_date: date) -> str:
        """
        Return the most recent news/filing text available strictly before before_date.

        Default implementation returns empty string (no text data available).
        Subclasses with text sources (e.g., SEC filings) should override this.

        Returns:
            Plain-text snippet. Empty string if no text is available.
        """
        return ""

    def has_text(self, asset: str, before_date: date) -> bool:
        """
        Return True if get_news() would return non-empty text for this asset/date.

        Default returns False. ProcessedDataProvider overrides with a fast
        column-presence check used for text-priority ranking during QA build.
        """
        return False

    def get_volume_series(self, asset: str, start: date, end: date) -> pd.Series:
        """
        Return daily trading volume for asset in [start, end].

        Default returns empty Series. Override in subclasses with volume data.
        """
        return pd.Series(dtype=float)

    def get_ohlc_series(
        self, asset: str, start: date, end: date
    ) -> "pd.DataFrame":
        """
        Return daily OHLC DataFrame for asset in [start, end].

        Columns: open, high, low, close (float). Index: DatetimeIndex.
        Default returns empty DataFrame. Override in subclasses with OHLC data.
        """
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    def get_asset_metadata(self, asset: str) -> dict:
        """
        Return static fundamental/metadata about an asset (e.g. market cap,
        launch year, sector). Returns empty dict if not available.
        """
        return {}

    def build_context(
        self,
        decision_date: date,
        assets: list[str],
        lookback_days: int,
    ) -> ContextWindow:
        """
        Build a PiT-safe ContextWindow for the given decision_date.

        The context window covers [decision_date - lookback_days, decision_date - 1].
        Raises ValueError if any PiT violation is detected.
        """
        end = decision_date - timedelta(days=1)
        # Walk back to get approximately lookback_days of trading days
        start = end - timedelta(days=int(lookback_days * 1.5))

        price_history = {}
        returns_history = {}
        min_required = max(1, lookback_days // 2)  # require at least half the window
        for asset in assets:
            prices = self.get_price_series(asset, start, end)
            prices = prices.iloc[-lookback_days:]
            # Drop leading NaN but keep trailing ones (they will be forward-filled downstream)
            prices = prices.dropna()
            if len(prices) < min_required:
                raise ValueError(
                    f"Insufficient price data for '{asset}' ending {end}: "
                    f"got {len(prices)} non-NaN rows, need at least {min_required} "
                    f"(half of lookback_days={lookback_days}). "
                    f"Consider skipping this decision date or choosing a different asset."
                )
            price_history[asset] = prices
            rets = self.get_return_series(asset, start, end)
            returns_history[asset] = rets.iloc[-lookback_days:].dropna()

        ctx = ContextWindow(
            decision_date=decision_date,
            assets=assets,
            price_history=price_history,
            returns_history=returns_history,
            macro_context=self.get_macro(end),
            market_regime=self.get_regime(end),
            news_text=self.get_news(assets[0], decision_date),
        )
        # Compute cross-asset correlation matrix (PiT-safe: uses only lookback data)
        if len(assets) >= 2:
            ctx.correlation_matrix = ctx.compute_correlation()
        ctx.validate_pit()  # Enforce PiT: raises if any data >= decision_date
        return ctx


# ---------------------------------------------------------------------------
# Abstract QA builder
# ---------------------------------------------------------------------------

class QABuilder(ABC):
    """
    Abstract base class for a single question template builder.

    Subclasses implement one template (T1–T7) and are responsible for:
      1. Selecting appropriate assets and decision dates
      2. Building a ContextWindow via the DataProvider
      3. Generating the question text
      4. Computing the ground-truth answer and explanation

    Each subclass should override:
      - template_id:     e.g., "T1"
      - complexity:      ComplexityLevel enum value
      - asset_class:     Primary asset class string
      - build_one():     Generate a single QAPair for given context + date
    """

    def __init__(self, provider: DataProvider, config: QAConfig):
        self.provider = provider
        self.config = config

    @property
    @abstractmethod
    def template_id(self) -> str:
        """Template identifier, e.g. 'T1'."""
        pass

    @property
    @abstractmethod
    def complexity(self) -> ComplexityLevel:
        """Complexity level for this template."""
        pass

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Primary asset class this template targets."""
        pass

    @abstractmethod
    def build_one(self, context: ContextWindow, seq: int) -> QAPair:
        """
        Generate one QA pair from the given ContextWindow.

        Args:
            context: Point-in-time market context (already PiT-validated).
            seq:     Sequence number for unique ID generation.

        Returns:
            A fully populated QAPair.
        """
        pass

    def build(self, n: int, decision_dates: list[date]) -> list[QAPair]:
        """
        Generate up to n QA pairs over the provided list of candidate decision_dates.

        This is **adaptive**: returns however many pairs the data supports, capped
        at n. If feasibility is low (e.g., T5/T6 needing 3+ aligned assets), the
        returned list may be much shorter than n. To bias toward text-rich pairs,
        order decision_dates so text-bearing dates come first.

        Dates that fall outside all configured splits, or for which the provider
        cannot supply sufficient history, are silently skipped.

        Args:
            n:               Max number of QA pairs to generate (cap, not target).
            decision_dates:  Candidate dates, ideally ordered so text-bearing
                             dates appear first.

        Returns:
            List of QAPair, length <= n.
        """
        pairs: list[QAPair] = []
        seq = 0
        for d in decision_dates:
            if len(pairs) >= n:
                break
            if self.config.get_split(d) is None:
                continue  # Date outside all configured splits
            try:
                assets = self._select_assets(d)
                context = self.provider.build_context(d, assets, self.config.lookback_days)
                pair = self.build_one(context, seq)
                # Inject text metadata so stats.json can track text coverage
                pair.metadata["has_text"] = bool(context.news_text)
                pair.metadata["text_chars"] = len(context.news_text)
                pairs.append(pair)
                seq += 1
            except Exception:
                # Skip dates where data is unavailable or computation fails
                continue
        return pairs

    def _make_id(self, decision_date: date, seq: int) -> str:
        """Generate a unique QA pair ID."""
        return f"{self.template_id}_{self.asset_class}_{decision_date.strftime('%Y%m%d')}_{seq:04d}"

    def _select_assets(self, decision_date: date) -> list[str]:
        """
        Select assets for this template. Default: all assets for asset_class.
        Subclasses may override to pick a fixed number (e.g., 2 for T4).
        """
        return self.provider.list_assets(self.asset_class)
