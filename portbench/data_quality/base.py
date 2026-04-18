"""Base classes for data quality assessment."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


class QualityLevel(Enum):
    """Overall quality level for a dataset or check."""

    PASS = "pass"        # Meets all thresholds
    WARN = "warn"        # Minor issues, usable with caution
    FAIL = "fail"        # Critical issues, not suitable for benchmark


@dataclass
class CheckResult:
    """Result of a single quality check."""

    check_name: str
    level: QualityLevel
    value: Optional[float]          # Numeric measurement (e.g., coverage rate)
    threshold: Optional[float]      # Pass threshold for reference
    message: str                    # Human-readable explanation
    details: dict = field(default_factory=dict)  # Extra structured data


@dataclass
class DatasetQualityReport:
    """Quality report for a single dataset (one asset class)."""

    asset_class: str
    source: str                     # "yahoo", "fred", "kaggle", "sec", or "processed"
    dataset_id: str
    assessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def level(self) -> QualityLevel:
        """Aggregate level: worst of all checks."""
        if any(c.level == QualityLevel.FAIL for c in self.checks):
            return QualityLevel.FAIL
        if any(c.level == QualityLevel.WARN for c in self.checks):
            return QualityLevel.WARN
        return QualityLevel.PASS

    @property
    def summary(self) -> dict:
        failed = [c for c in self.checks if c.level == QualityLevel.FAIL]
        warned = [c for c in self.checks if c.level == QualityLevel.WARN]
        return {
            "level": self.level.value,
            "total_checks": len(self.checks),
            "failed": len(failed),
            "warned": len(warned),
            "passed": len(self.checks) - len(failed) - len(warned),
        }


@dataclass
class BenchmarkQualityReport:
    """Top-level quality report aggregating all asset classes."""

    assessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_reports: list[DatasetQualityReport] = field(default_factory=list)

    @property
    def level(self) -> QualityLevel:
        if any(r.level == QualityLevel.FAIL for r in self.dataset_reports):
            return QualityLevel.FAIL
        if any(r.level == QualityLevel.WARN for r in self.dataset_reports):
            return QualityLevel.WARN
        return QualityLevel.PASS

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'='*60}")
        print(f"PortBench Data Quality Report  [{self.assessed_at[:19]}]")
        print(f"Overall: {self.level.value.upper()}")
        print(f"{'='*60}")
        for r in self.dataset_reports:
            s = r.summary
            icon = {"pass": "✓", "warn": "!", "fail": "✗"}[r.level.value]
            print(
                f"  [{icon}] {r.asset_class}/{r.dataset_id} ({r.source})"
                f"  — {s['passed']}/{s['total_checks']} checks passed"
                + (f", {s['failed']} failed" if s["failed"] else "")
                + (f", {s['warned']} warned" if s["warned"] else "")
            )
            for c in r.checks:
                if c.level != QualityLevel.PASS:
                    icon2 = "!" if c.level == QualityLevel.WARN else "✗"
                    print(f"       [{icon2}] {c.check_name}: {c.message}")
        print(f"{'='*60}\n")

    def save(self, path: str) -> None:
        """Save the full report to a JSON file."""
        def _serial(obj):
            if isinstance(obj, QualityLevel):
                return obj.value
            if isinstance(obj, CheckResult):
                d = asdict(obj)
                d["level"] = obj.level.value
                return d
            if isinstance(obj, DatasetQualityReport):
                d = asdict(obj)
                d["level"] = obj.level.value
                d["summary"] = obj.summary
                d["checks"] = [_serial(c) for c in obj.checks]
                return d
            return obj

        report_dict = {
            "assessed_at": self.assessed_at,
            "overall_level": self.level.value,
            "dataset_reports": [_serial(r) for r in self.dataset_reports],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)


@dataclass
class QualityConfig:
    """Thresholds and settings for quality checks."""

    datasets_dir: str = "datasets"
    output_dir: str = "outputs/quality_reports"

    # Temporal coverage
    min_coverage_rate: float = 0.95       # At least 95% of trading days must have data
    max_ffill_gap_days: int = 5           # Gaps longer than this are flagged

    # Stress period completeness — must have data for these windows
    # Note: 2008_crisis (2008-09-01 to 2009-03-31) is excluded because the
    # benchmark data starts in 2015 and cannot cover that period.
    stress_periods: list[tuple[str, str, str]] = field(default_factory=lambda: [
        ("2020_covid",     "2020-02-01", "2020-05-31"),
        ("2022_crypto",    "2022-05-01", "2022-12-31"),
    ])
    min_stress_coverage: float = 0.90    # 90% coverage required inside stress periods

    # Train / val / test splits
    train_start: str = "2015-01-01"
    train_end:   str = "2022-12-31"
    val_start:   str = "2023-01-01"
    val_end:     str = "2023-12-31"
    test_start:  str = "2024-01-01"
    test_end:    str = "2025-12-31"

    # Statistical sanity for price data
    max_single_day_return: float = 0.50  # Flag daily returns > 50%
    min_price: float = 0.0               # Prices must be positive

    # Cross-source consistency
    min_cross_source_corr: float = 0.95  # Overlapping columns should correlate > 0.95

    # Text data
    min_avg_text_length: int = 50        # Average document must be > 50 chars
    min_document_count: int = 10         # At least 10 documents per asset


class DataQualityChecker(ABC):
    """Abstract base class for data quality checkers."""

    def __init__(self, config: QualityConfig):
        self.config = config
        self.datasets_dir = Path(config.datasets_dir)

    @property
    @abstractmethod
    def checker_name(self) -> str:
        """Identifier for this checker (e.g., 'numeric', 'text', 'cross_asset')."""
        pass

    @abstractmethod
    def check(self, *args, **kwargs) -> DatasetQualityReport:
        """Run all checks and return a report."""
        pass

    # ------------------------------------------------------------------ helpers

    def _make_check(
        self,
        name: str,
        value: Optional[float],
        threshold: Optional[float],
        level: QualityLevel,
        message: str,
        details: dict = None,
    ) -> CheckResult:
        return CheckResult(
            check_name=name,
            level=level,
            value=value,
            threshold=threshold,
            message=message,
            details=details or {},
        )

    def _level_by_threshold(
        self,
        value: float,
        warn_threshold: float,
        fail_threshold: float,
        higher_is_better: bool = True,
    ) -> QualityLevel:
        """Convert a metric value into a QualityLevel using two thresholds."""
        if higher_is_better:
            if value >= warn_threshold:
                return QualityLevel.PASS
            if value >= fail_threshold:
                return QualityLevel.WARN
            return QualityLevel.FAIL
        else:
            if value <= warn_threshold:
                return QualityLevel.PASS
            if value <= fail_threshold:
                return QualityLevel.WARN
            return QualityLevel.FAIL

    def _load_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a CSV file; return None and log if not found."""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            return None

    def _trading_days(self, start: str, end: str) -> pd.DatetimeIndex:
        """Return approximate trading days (Mon-Fri) between start and end."""
        return pd.bdate_range(start=start, end=end)
