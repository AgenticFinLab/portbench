"""Text data quality checker for SEC filings and news documents."""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    DataQualityChecker,
    DatasetQualityReport,
    QualityConfig,
    QualityLevel,
)


class TextQualityChecker(DataQualityChecker):
    """
    Quality checks for text datasets: SEC filings, news articles.

    Runs the following checks:
      1. document_count     — enough documents exist
      2. avg_text_length    — average document is substantive (not truncated stubs)
      3. empty_doc_rate     — fraction of empty / whitespace-only documents
      4. encoding_issues    — documents with non-UTF-8 characters or replacement chars
      5. temporal_coverage  — text documents span the expected date range
      6. duplicate_rate     — fraction of near-duplicate documents (exact match)
    """

    @property
    def checker_name(self) -> str:
        return "text"

    def check(self, *args, **kwargs) -> DatasetQualityReport:
        """
        Dispatch to check_file_list() or check_dataframe() based on first argument type.

        Accepts the same signatures as those two methods.
        """
        if args and isinstance(args[0], list):
            return self.check_file_list(*args, **kwargs)
        if args and isinstance(args[0], pd.DataFrame):
            return self.check_dataframe(*args, **kwargs)
        # Keyword-only call
        if "file_paths" in kwargs:
            return self.check_file_list(**kwargs)
        return self.check_dataframe(**kwargs)

    def check_file_list(
        self,
        file_paths: list[Path],
        asset_class: str,
        dataset_id: str,
        source: str,
        date_range: Optional[tuple[str, str]] = None,
    ) -> DatasetQualityReport:
        """
        Run quality checks on a list of text files (HTML or plain text).

        Args:
            file_paths:  List of paths to text documents.
            asset_class: Asset class string.
            dataset_id:  Dataset identifier.
            source:      Data source.
            date_range:  Expected (start, end) date strings for coverage check.

        Returns:
            DatasetQualityReport with all check results.
        """
        report = DatasetQualityReport(
            asset_class=asset_class,
            source=source,
            dataset_id=dataset_id,
        )

        if not file_paths:
            report.checks.append(
                self._make_check(
                    "document_count",
                    value=0.0,
                    threshold=float(self.config.min_document_count),
                    level=QualityLevel.FAIL,
                    message="No text files found.",
                )
            )
            return report

        texts = []
        for p in file_paths:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                texts.append(content)
            except Exception:
                texts.append("")

        report.checks.append(self._check_document_count(texts))
        report.checks.append(self._check_avg_length(texts))
        report.checks.append(self._check_empty_doc_rate(texts))
        report.checks.append(self._check_encoding_issues(texts))
        report.checks.append(self._check_duplicate_rate(texts))

        return report

    def check_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        date_col: str,
        asset_class: str,
        dataset_id: str,
        source: str,
    ) -> DatasetQualityReport:
        """
        Run quality checks on a DataFrame column containing text documents.

        Args:
            df:          DataFrame with at least a text column and a date column.
            text_col:    Column name containing document text.
            date_col:    Column name containing dates.
            asset_class: Asset class string.
            dataset_id:  Dataset identifier.
            source:      Data source.

        Returns:
            DatasetQualityReport with all check results.
        """
        report = DatasetQualityReport(
            asset_class=asset_class,
            source=source,
            dataset_id=dataset_id,
        )

        if df is None or df.empty or text_col not in df.columns:
            report.checks.append(
                self._make_check(
                    "document_count",
                    value=0.0,
                    threshold=float(self.config.min_document_count),
                    level=QualityLevel.FAIL,
                    message=f"DataFrame empty or column '{text_col}' not found.",
                )
            )
            return report

        texts = df[text_col].fillna("").astype(str).tolist()

        report.checks.append(self._check_document_count(texts))
        report.checks.append(self._check_avg_length(texts))
        report.checks.append(self._check_empty_doc_rate(texts))
        report.checks.append(self._check_encoding_issues(texts))
        report.checks.append(self._check_duplicate_rate(texts))

        # Temporal coverage when date column is available
        if date_col in df.columns:
            report.checks.append(
                self._check_text_temporal_coverage(df, date_col)
            )

        return report

    # ------------------------------------------------------------------ checks

    def _check_document_count(self, texts: list[str]) -> object:
        count = len(texts)
        threshold = self.config.min_document_count
        level = self._level_by_threshold(
            count,
            warn_threshold=threshold,
            fail_threshold=threshold // 2,
            higher_is_better=True,
        )
        return self._make_check(
            "document_count",
            value=float(count),
            threshold=float(threshold),
            level=level,
            message=f"{count} documents found (minimum: {threshold}).",
        )

    def _check_avg_length(self, texts: list[str]) -> object:
        cleaned = [self._strip_html(t) for t in texts]
        lengths = [len(t.strip()) for t in cleaned]
        avg = sum(lengths) / max(len(lengths), 1)
        threshold = self.config.min_avg_text_length

        level = self._level_by_threshold(
            avg,
            warn_threshold=threshold,
            fail_threshold=threshold // 2,
            higher_is_better=True,
        )
        return self._make_check(
            "avg_text_length",
            value=round(avg, 1),
            threshold=float(threshold),
            level=level,
            message=f"Average document length: {avg:.0f} chars (minimum: {threshold}).",
            details={"min_length": min(lengths), "max_length": max(lengths)},
        )

    def _check_empty_doc_rate(self, texts: list[str]) -> object:
        empty = sum(1 for t in texts if len(self._strip_html(t).strip()) == 0)
        rate = empty / max(len(texts), 1)

        level = self._level_by_threshold(
            rate,
            warn_threshold=0.05,
            fail_threshold=0.20,
            higher_is_better=False,
        )
        return self._make_check(
            "empty_doc_rate",
            value=round(rate, 4),
            threshold=0.05,
            level=level,
            message=f"{rate:.1%} of documents are empty ({empty}/{len(texts)}).",
            details={"empty_count": empty, "total": len(texts)},
        )

    def _check_encoding_issues(self, texts: list[str]) -> object:
        # Count documents containing Unicode replacement character (U+FFFD)
        bad = sum(1 for t in texts if "\ufffd" in t)
        rate = bad / max(len(texts), 1)

        level = self._level_by_threshold(
            rate,
            warn_threshold=0.05,
            fail_threshold=0.20,
            higher_is_better=False,
        )
        return self._make_check(
            "encoding_issues",
            value=round(rate, 4),
            threshold=0.05,
            level=level,
            message=f"{rate:.1%} of documents have encoding issues ({bad}/{len(texts)}).",
            details={"bad_encoding_count": bad},
        )

    def _check_duplicate_rate(self, texts: list[str]) -> object:
        # Exact-match deduplication on stripped lowercased text (first 500 chars)
        seen: set[str] = set()
        dupes = 0
        for t in texts:
            key = t.strip().lower()[:500]
            if key in seen:
                dupes += 1
            else:
                seen.add(key)

        rate = dupes / max(len(texts), 1)
        level = self._level_by_threshold(
            rate,
            warn_threshold=0.10,
            fail_threshold=0.30,
            higher_is_better=False,
        )
        return self._make_check(
            "duplicate_rate",
            value=round(rate, 4),
            threshold=0.10,
            level=level,
            message=f"{rate:.1%} duplicate documents ({dupes}/{len(texts)}).",
            details={"duplicate_count": dupes, "unique_count": len(seen)},
        )

    def _check_text_temporal_coverage(
        self, df: pd.DataFrame, date_col: str
    ) -> object:
        """Check whether text documents span train through test periods."""
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        except Exception:
            return self._make_check(
                "text_temporal_coverage",
                value=None,
                threshold=None,
                level=QualityLevel.WARN,
                message=f"Could not parse dates in column '{date_col}'.",
            )

        if dates.empty:
            return self._make_check(
                "text_temporal_coverage",
                value=0.0,
                threshold=1.0,
                level=QualityLevel.FAIL,
                message="No valid dates found in text dataset.",
            )

        earliest = dates.min()
        latest = dates.max()
        train_start = pd.Timestamp(self.config.train_start)
        test_end = pd.Timestamp(self.config.test_end)

        # Score: fraction of the full benchmark window covered
        window = (test_end - train_start).days
        covered = (min(latest, test_end) - max(earliest, train_start)).days
        coverage = max(covered, 0) / max(window, 1)

        level = self._level_by_threshold(
            coverage,
            warn_threshold=0.50,
            fail_threshold=0.20,
            higher_is_better=True,
        )
        return self._make_check(
            "text_temporal_coverage",
            value=round(coverage, 4),
            threshold=0.50,
            level=level,
            message=(
                f"Text documents span {earliest.date()} – {latest.date()} "
                f"({coverage:.1%} of benchmark window)."
            ),
            details={
                "earliest": str(earliest.date()),
                "latest": str(latest.date()),
                "benchmark_start": self.config.train_start,
                "benchmark_end": self.config.test_end,
            },
        )

    # ------------------------------------------------------------------ util

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]+>", " ", text)
