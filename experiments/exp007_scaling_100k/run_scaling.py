"""
exp007 — 100K Scaling Runner with Checkpoint/Resume (Week 16-17)

Runs chunked LLM extraction on the full dataset, saving a checkpoint after
every chunk so the job can be killed and resumed safely.

Flow
----
  train.jsonl (49,570 records)
    → chunk 0 (5,000 records) → LLM extract → merge → checkpoint
    → chunk 1 (5,000 records) → LLM extract → merge → checkpoint
    …
    → save final ontology version

Checkpoint format: artifacts/runs/exp007_checkpoint.json
  {
    "target_version": "0.8.0",
    "chunk_size": 5000,
    "chunks_completed": 3,
    "records_processed": 15000,
    "n_concepts": 1234,
    "n_relations": 5678,
    "elapsed_s": 1200.0,
    "timestamp": "2026-08-01T12:00:00"
  }

Usage
-----
  # Full run (all records, produces v0.8.0)
  set -a && source .env && set +a
  python -m experiments.exp007_scaling_100k.run_scaling

  # Pilot: first 10,000 records only
  python -m experiments.exp007_scaling_100k.run_scaling --max-records 10000

  # Resume interrupted run
  python -m experiments.exp007_scaling_100k.run_scaling --resume

  # Custom version / chunk size
  python -m experiments.exp007_scaling_100k.run_scaling \\
      --target-version 0.8.0 --chunk-size 5000 --max-records 20000

  # Dry-run: show plan without running
  python -m experiments.exp007_scaling_100k.run_scaling --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

from src.utils.io import load_jsonl, save_json, ensure_dir
from src.utils.logging_utils import get_logger
from src.ontology.offline_extractor import OfflineExtractor
from src.ontology.versioning import OntologyVersionManager
from src.graph.graph_store import TourismGraphStore

logger = get_logger("exp007.run_scaling")

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = ROOT / "artifacts" / "runs" / "exp007_checkpoint.json"
PILOT_LOG       = ROOT / "artifacts" / "runs" / "exp007_pilot.log"
FULL_LOG        = ROOT / "artifacts" / "runs" / "exp007_full.log"
ONT_DIR         = ROOT / "artifacts" / "ontologies"
DATA_TRAIN      = ROOT / "data" / "splits" / "train.jsonl"


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}


def _save_checkpoint(data: dict) -> None:
    data["timestamp"] = datetime.utcnow().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))


def _clear_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ── LLM extractor (optional, falls back to offline) ──────────────────────────

def _make_llm_client():
    """Return LLMClient if HF_TOKEN + LOCAL_MODEL_KEY are set, else None."""
    try:
        from src.llm.client import LLMClient
        client = LLMClient(
            load_in_4bit=True,
            gpu_ids=[int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
                     if g.strip().isdigit()][:1],  # single GPU for chunked extraction
            max_tokens=512,
            temperature=0.1,
        )
        if client.is_available:
            return client
    except Exception as exc:
        logger.warning(f"LLM client init failed: {exc}")
    return None


# ── Per-chunk extraction ──────────────────────────────────────────────────────

def _extract_chunk(
    records: list[dict],
    llm_client,
    offline_extractor: OfflineExtractor,
) -> tuple[list[dict], list[dict]]:
    """
    Extract concepts and relations from a list of records.

    Uses LLM if available, otherwise falls back to offline rule-based extraction.
    Returns (concepts, relations).
    """
    all_concepts: list[dict] = []
    all_relations: list[dict] = []

    for rec in records:
        text = rec.get("Output", rec.get("text", ""))
        if not text:
            continue
        try:
            result = offline_extractor.extract(text)
            all_concepts.extend(result.get("concepts", []))
            all_relations.extend(result.get("relations", []))
        except Exception:
            pass

    return all_concepts, all_relations


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _merge_concepts(existing: list[dict], new: list[dict], policy: str = "priority") -> list[dict]:
    """Merge new concepts into existing, deduplicating by name."""
    index = {c["name"].lower(): c for c in existing}
    added = merged = 0
    for c in new:
        key = c.get("name", "").lower()
        if not key:
            continue
        if key not in index:
            index[key] = c
            added += 1
        else:
            # Keep higher-confidence version
            existing_conf = float(index[key].get("confidence", 0))
            new_conf      = float(c.get("confidence", 0))
            if new_conf > existing_conf:
                index[key] = c
                merged += 1
    return list(index.values()), added, merged


def _merge_relations(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new relations, deduplicating by (subject, predicate, object)."""
    seen = {(r["subject"].lower(), r["predicate"], r["object"].lower())
            for r in existing}
    added = 0
    result = list(existing)
    for r in new:
        key = (r.get("subject","").lower(), r.get("predicate",""), r.get("object","").lower())
        if key not in seen and all(key):
            seen.add(key)
            result.append(r)
            added += 1
    return result, added


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    target_version: str = "0.8.0",
    base_version:   str = "0.7.0",
    chunk_size:     int = 5000,
    max_records:    int | None = None,
    resume:         bool = False,
    dry_run:        bool = False,
) -> dict:
    """
    Main entry point for chunked extraction.

    Returns summary dict with final counts and elapsed time.
    """
    ensure_dir(ROOT / "artifacts" / "runs")
    ensure_dir(ONT_DIR / target_version)

    # ── Load base ontology ────────────────────────────────────────────────────
    logger.info(f"Loading base ontology v{base_version}...")
    vm       = OntologyVersionManager(ONT_DIR)
    base     = vm.load_version(base_version)
    concepts  = list(base["concepts"])
    relations = list(base["relations"])
    rules     = list(base.get("rules", []))
    schema    = base.get("schema", {})
    logger.info(f"  Base: {len(concepts):,} concepts, {len(relations):,} relations")

    # ── Load training records ─────────────────────────────────────────────────
    logger.info(f"Loading training data from {DATA_TRAIN}...")
    all_records = load_jsonl(DATA_TRAIN)
    if max_records:
        all_records = all_records[:max_records]
    total = len(all_records)
    logger.info(f"  {total:,} records to process")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_chunk = 0
    t_offset    = 0.0
    if resume:
        ck = _load_checkpoint()
        if ck and ck.get("target_version") == target_version:
            start_chunk = ck.get("chunks_completed", 0)
            t_offset    = ck.get("elapsed_s", 0.0)
            done_records = start_chunk * chunk_size
            all_records  = all_records[done_records:]
            logger.info(
                f"  Resuming from chunk {start_chunk} "
                f"({done_records:,} records already done)"
            )
        else:
            logger.info("  No valid checkpoint — starting from scratch")

    # ── Chunk plan ────────────────────────────────────────────────────────────
    n_chunks = (len(all_records) + chunk_size - 1) // chunk_size
    total_chunks = start_chunk + n_chunks

    print(f"\n{'='*60}")
    print(f"  exp007 Scaling Runner — v{base_version} → v{target_version}")
    print(f"{'='*60}")
    print(f"  Records     : {total:,} total, {len(all_records):,} remaining")
    print(f"  Chunk size  : {chunk_size:,}")
    print(f"  Chunks      : {n_chunks} remaining ({total_chunks} total)")
    print(f"  Resume      : chunk {start_chunk}")
    print(f"  Dry-run     : {dry_run}")
    print(f"{'='*60}\n")

    if dry_run:
        logger.info("Dry-run mode — no extraction performed")
        return {
            "status":    "dry_run",
            "records":   total,
            "n_chunks":  n_chunks,
            "base":      base_version,
            "target":    target_version,
        }

    # ── Initialise extractors ─────────────────────────────────────────────────
    offline_extractor = OfflineExtractor()
    llm_client        = _make_llm_client()
    mode = "llm" if llm_client else "offline"
    logger.info(f"Extraction mode: {mode.upper()}")

    # ── Process chunks ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    total_added_concepts  = 0
    total_added_relations = 0

    for ci, chunk_start in enumerate(range(0, len(all_records), chunk_size)):
        chunk_idx     = start_chunk + ci
        chunk_records = all_records[chunk_start: chunk_start + chunk_size]
        chunk_n       = len(chunk_records)

        t_chunk = time.perf_counter()
        logger.info(
            f"Chunk {chunk_idx+1}/{total_chunks}: "
            f"{chunk_n:,} records (records {chunk_start:,}–{chunk_start+chunk_n:,})"
        )

        # Extract
        new_concepts, new_relations = _extract_chunk(
            chunk_records, llm_client, offline_extractor
        )

        # Merge
        concepts, added_c, merged_c = _merge_concepts(concepts, new_concepts)
        relations, added_r = _merge_relations(relations, new_relations)
        total_added_concepts  += added_c
        total_added_relations += added_r

        chunk_elapsed = round(time.perf_counter() - t_chunk, 1)
        total_elapsed = round(time.perf_counter() - t0 + t_offset, 1)
        rph = round(chunk_n / max(chunk_elapsed, 0.001) * 3600)

        print(
            f"  Chunk {chunk_idx+1}/{total_chunks} | "
            f"+{added_c:,} concepts | +{added_r:,} relations | "
            f"{chunk_elapsed:.1f}s | {rph:,} rec/hr"
        )

        # Save checkpoint after every chunk
        _save_checkpoint({
            "target_version":    target_version,
            "base_version":      base_version,
            "chunk_size":        chunk_size,
            "chunks_completed":  chunk_idx + 1,
            "records_processed": chunk_start + chunk_n + (start_chunk * chunk_size),
            "n_concepts":        len(concepts),
            "n_relations":       len(relations),
            "elapsed_s":         total_elapsed,
            "extraction_mode":   mode,
        })

    # ── Save final ontology version ───────────────────────────────────────────
    elapsed = round(time.perf_counter() - t0 + t_offset, 1)
    logger.info(
        f"Saving ontology v{target_version}: "
        f"{len(concepts):,} concepts, {len(relations):,} relations"
    )
    vm.save_version(target_version, concepts, relations, rules, schema)
    _clear_checkpoint()

    summary = {
        "experiment":          "exp007_scaling_run",
        "base_version":        base_version,
        "target_version":      target_version,
        "records_processed":   total,
        "chunk_size":          chunk_size,
        "n_chunks":            total_chunks,
        "n_concepts_final":    len(concepts),
        "n_relations_final":   len(relations),
        "added_concepts":      total_added_concepts,
        "added_relations":     total_added_relations,
        "extraction_mode":     mode,
        "elapsed_s":           elapsed,
        "records_per_hour":    round(total / max(elapsed, 1) * 3600),
    }

    out_path = ROOT / "artifacts" / "reports" / "exp007_run_summary.json"
    save_json(summary, out_path)

    print(f"\n{'='*60}")
    print(f"  Run complete in {elapsed:.1f}s")
    print(f"  Final ontology  : v{target_version}")
    print(f"  Concepts        : {len(concepts):,} (+{total_added_concepts:,} new)")
    print(f"  Relations       : {len(relations):,} (+{total_added_relations:,} new)")
    print(f"  Records/hour    : {summary['records_per_hour']:,}")
    print(f"  Mode            : {mode.upper()}")
    print(f"  Saved to        : {out_path.name}")
    print(f"{'='*60}\n")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp007 — Chunked LLM extraction with checkpoint/resume"
    )
    p.add_argument("--target-version", default="0.8.0",
                   help="New ontology version to produce (default: 0.8.0)")
    p.add_argument("--base-version", default="0.7.0",
                   help="Base ontology to start from (default: 0.7.0)")
    p.add_argument("--chunk-size", type=int, default=5000,
                   help="Records per chunk (default: 5000)")
    p.add_argument("--max-records", type=int, default=None,
                   help="Stop after N records (pilot mode)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if one exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without running")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        target_version=args.target_version,
        base_version=args.base_version,
        chunk_size=args.chunk_size,
        max_records=args.max_records,
        resume=args.resume,
        dry_run=args.dry_run,
    )
