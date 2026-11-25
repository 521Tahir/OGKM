"""
Experiment 001 — Data Profiling

Runs the full data profiling pipeline on the FULL combined dataset (~100K records).
Outputs:
  - data/reports/data_profile.json
  - data/reports/data_profile_report.md
  - data/processed/cleaned_records.jsonl
  - data/splits/{train,val,test}.jsonl
  - artifacts/reports/dataset_statistics.json
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import save_json, ensure_dir
from src.data.loaders import TourismDataLoader
from src.data.validators import DatasetValidator
from src.data.profiling import DataProfiler
from src.data.cleaners import DataCleaner
from src.data.splitters import DataSplitter

logger = get_logger("exp001")


def main() -> None:
    set_seed(42)
    setup_logging(log_level="INFO", log_file="artifacts/runs/exp001_profile.log")
    ensure_dir("artifacts/reports")

    DATA_PATH = PROJECT_ROOT / "data" / "jeju_full" / "jeju_full_dataset.jsonl"
    REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

    logger.info("=== Experiment 001: Data Profiling ===")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    loader = TourismDataLoader(DATA_PATH)
    records = loader.load()
    logger.info(f"Loaded {len(records):,} records.")

    # ── 2. Validate ──────────────────────────────────────────────────────────
    validator = DatasetValidator()
    val_result = validator.validate_all(records)
    logger.info(
        f"Validation: {val_result.valid}/{val_result.total} valid "
        f"({val_result.validity_rate:.1%})"
    )

    # ── 3. Profile ───────────────────────────────────────────────────────────
    profiler = DataProfiler(output_dir=REPORTS_DIR)
    profile = profiler.profile(records)
    profile["validation"] = val_result.to_dict()
    paths = profiler.save_profile(profile, prefix="data_profile")
    logger.info(f"Profile saved: {paths}")

    # ── 4. Clean ─────────────────────────────────────────────────────────────
    cleaner = DataCleaner()
    cleaned = cleaner.clean_all(records)
    unique_records, removed = cleaner.deduplicate(cleaned)
    logger.info(f"After dedup: {len(unique_records):,} unique records ({len(removed)} removed).")

    # Save cleaned JSONL
    ensure_dir(PROCESSED_DIR)
    from src.utils.io import save_jsonl
    save_jsonl([r.to_dict() for r in unique_records], PROCESSED_DIR / "cleaned_records.jsonl")
    save_json({"removed_duplicates": removed[:50]}, PROCESSED_DIR / "dedup_report.json")

    # ── 5. Split ─────────────────────────────────────────────────────────────
    splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train, val, test = splitter.split(unique_records)
    split_paths = splitter.save_splits(train, val, test, output_dir=SPLITS_DIR)

    # ── 6. Save artifact summary ─────────────────────────────────────────────
    statistics = {
        "total_raw": len(records),
        "total_after_dedup": len(unique_records),
        "duplicates_removed": len(removed),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "profile_paths": paths,
        "split_paths": split_paths,
        "validation_summary": val_result.to_dict(),
        "task_taxonomy": profile["task_taxonomy"],
        "top_entities": {
            k: list(v["top_20"].items())[:10]
            for k, v in profile["entity_statistics"].items()
        },
    }
    stats_path = ARTIFACTS_DIR / "dataset_statistics.json"
    save_json(statistics, stats_path)
    logger.info(f"Dataset statistics saved → {stats_path}")

    # ── 7. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 001 — DATA PROFILING COMPLETE")
    print("=" * 60)
    print(f"  Total records:          {len(records):,}")
    print(f"  After deduplication:    {len(unique_records):,}")
    print(f"  Train / Val / Test:     {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"\n  Profile JSON:           {paths['json']}")
    print(f"  Profile Markdown:       {paths['markdown']}")
    print(f"  Cleaned JSONL:          {PROCESSED_DIR / 'cleaned_records.jsonl'}")
    print(f"  Dataset Statistics:     {stats_path}")
    print("\n  Task Categories:")
    for cat, info in list(profile["task_taxonomy"].items())[:8]:
        print(f"    {cat:35s}: {info['count']:5d} ({info['pct']}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
