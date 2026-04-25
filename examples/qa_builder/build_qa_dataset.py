"""
Build a QA dataset using all seven templates with real processed data.

Usage:
    python examples/qa_builder/build_qa_dataset.py

Output:
    outputs/qa_dataset/
        all_pairs.jsonl          — full dataset (all templates, all splits)
        train.jsonl              — training split
        val.jsonl                — validation split
        test.jsonl               — test split
        stats.json               — dataset statistics by template, regime, and text coverage

Split boundaries are computed dynamically from the actual data date range
(70% train / 15% val / 15% test, snapped to year boundaries).

Requires datasets/processed/ to be populated. Run:
    python examples/data_collect/get_all.py
    python examples/data_preprocess/preprocess_all.py
first if needed.
"""

import json
import sys
from datetime import date
from pathlib import Path

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.qa_builder.processed_data import ProcessedDataProvider
from portbench.qa_builder.base import QAConfig, Split
from portbench.qa_builder.t1_return_prediction import T1ReturnPrediction
from portbench.qa_builder.t2_risk_assessment import T2RiskAssessment
from portbench.qa_builder.t3_position_sizing import T3PositionSizing
from portbench.qa_builder.t4_pairwise_allocation import T4PairwiseAllocation
from portbench.qa_builder.t5_multiasset_optimization import T5MultiAssetOptimization
from portbench.qa_builder.t6_rebalancing import T6RebalancingDecision
from portbench.qa_builder.t7_regime_detection import T7RegimeDetection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("datasets/qa_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

provider = ProcessedDataProvider(
    data_dir="datasets/processed",
    sec_dir="datasets/sec",
)

# Derive split boundaries from the actual data date range in portbench.csv
portbench_csv = Path("datasets/processed/portbench.csv")
_dates = pd.read_csv(portbench_csv, usecols=["date"], low_memory=False)["date"]
_dates = pd.to_datetime(_dates, errors="coerce").dropna()
DATA_START = _dates.min().date()
DATA_END   = _dates.max().date()
print(f"Data range: {DATA_START} – {DATA_END}")

config = QAConfig.from_date_range(
    data_start=DATA_START,
    data_end=DATA_END,
    train_frac=0.70,
    val_frac=0.15,
    lookback_days=60,
    horizon_days=21,
    samples_per_template=1000,  # max per template (adaptive — actual count may be lower)
)
print(f"Split boundaries: {config.describe()}")

# Build candidate decision dates per split, then prioritize text-bearing dates
# so the first dates the build loop hits are most likely to yield text context.
import random
TRAIN_DATES = pd.bdate_range(config.train_start, config.train_end).date.tolist()
VAL_DATES   = pd.bdate_range(config.val_start,   config.val_end).date.tolist()
TEST_DATES  = pd.bdate_range(config.test_start,  config.test_end).date.tolist()

def rank_text_first(dates, seed):
    """Split dates into text-bearing vs not (using SPY/BTC-USD as flagship probes),
    shuffle each bucket deterministically, then concat text-first."""
    rng = random.Random(seed)
    text_dates, plain_dates = [], []
    for d in dates:
        if provider.has_text("SPY", d) or provider.has_text("BTC-USD", d):
            text_dates.append(d)
        else:
            plain_dates.append(d)
    rng.shuffle(text_dates)
    rng.shuffle(plain_dates)
    return text_dates + plain_dates

train_ranked = rank_text_first(TRAIN_DATES, config.random_seed)
val_ranked   = rank_text_first(VAL_DATES,   config.random_seed + 1)
test_ranked  = rank_text_first(TEST_DATES,  config.random_seed + 2)

# Round-robin interleave so each split gets representation even if the per-template
# cap is hit before exhausting all candidates.
SAMPLE_DATES = []
iters = [iter(train_ranked), iter(val_ranked), iter(test_ranked)]
exhausted = [False, False, False]
while not all(exhausted):
    for i, it in enumerate(iters):
        if exhausted[i]:
            continue
        try:
            SAMPLE_DATES.append(next(it))
        except StopIteration:
            exhausted[i] = True

print(f"Candidate decision dates: {len(SAMPLE_DATES)} "
      f"(train={len(TRAIN_DATES)}, val={len(VAL_DATES)}, test={len(TEST_DATES)})")


# ---------------------------------------------------------------------------
# Build each template
# ---------------------------------------------------------------------------

BUILDERS = [
    T1ReturnPrediction(provider, config),
    T2RiskAssessment(provider, config),
    T3PositionSizing(provider, config),
    T4PairwiseAllocation(provider, config),
    T5MultiAssetOptimization(provider, config),
    T6RebalancingDecision(provider, config),
    T7RegimeDetection(provider, config),
]

all_pairs = []
stats = {}

for builder in BUILDERS:
    print(f"Building {builder.template_id} ({builder.__class__.__name__})...")
    pairs = builder.build(n=config.samples_per_template, decision_dates=SAMPLE_DATES)
    all_pairs.extend(pairs)

    # Per-template statistics
    with_text = [p for p in pairs if p.metadata.get("has_text")]
    text_lengths = [p.metadata.get("text_chars", 0) for p in with_text]
    stats[builder.template_id] = {
        "n_total": len(pairs),
        "by_split": {
            split.value: sum(1 for p in pairs if p.split == split)
            for split in Split
        },
        "by_regime": {},
        "text": {
            "n_with_text": len(with_text),
            "pct_with_text": round(len(with_text) / len(pairs) * 100, 1) if pairs else 0,
            "avg_chars": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0,
            "max_chars": max(text_lengths) if text_lengths else 0,
            "min_chars": min(text_lengths) if text_lengths else 0,
        },
    }
    for p in pairs:
        regime = p.market_regime.value if hasattr(p.market_regime, 'value') else (p.market_regime or "unknown")
        stats[builder.template_id]["by_regime"][regime] = (
            stats[builder.template_id]["by_regime"].get(regime, 0) + 1
        )

    n_text = len(with_text)
    print(f"  Generated {len(pairs)} QA pairs ({n_text} with text)")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_jsonl(pairs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict(), default=str) + "\n")


# Full dataset
save_jsonl(all_pairs, OUTPUT_DIR / "all_pairs.jsonl")

# Split files
for split in Split:
    split_pairs = [p for p in all_pairs if p.split == split]
    save_jsonl(split_pairs, OUTPUT_DIR / f"{split.value}.jsonl")
    print(f"Split {split.value}: {len(split_pairs)} pairs")

# Statistics (include split boundary metadata)
all_with_text = [p for p in all_pairs if p.metadata.get("has_text")]
all_text_lengths = [p.metadata.get("text_chars", 0) for p in all_with_text]
stats["_meta"] = {
    "split_boundaries": {
        "train": [config.train_start, config.train_end],
        "val":   [config.val_start,   config.val_end],
        "test":  [config.test_start,  config.test_end],
    },
    "data_range": [str(DATA_START), str(DATA_END)],
    "text_overall": {
        "n_total": len(all_pairs),
        "n_with_text": len(all_with_text),
        "pct_with_text": round(len(all_with_text) / len(all_pairs) * 100, 1) if all_pairs else 0,
        "avg_chars": round(sum(all_text_lengths) / len(all_text_lengths), 1) if all_text_lengths else 0,
        "max_chars": max(all_text_lengths) if all_text_lengths else 0,
        "min_chars": min(all_text_lengths) if all_text_lengths else 0,
    },
}
with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, default=str)

print(f"\nTotal QA pairs: {len(all_pairs)}")
print(f"Output saved to: {OUTPUT_DIR}")
