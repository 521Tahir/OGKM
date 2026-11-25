"""
Experiment 005 — Constraint Enforcement Analysis

Tests the effectiveness of ontology constraints in the generation pipeline:
  - Compares constrained vs unconstrained (onto_rag) outputs
  - Measures violation counts by type
  - Identifies which constraint categories fire most often
  - Produces a constraint violation breakdown report

Output:
  artifacts/reports/exp005_constraints_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_store import TourismGraphStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.constrained_generator import ConstrainedGenerator
from src.evaluation.benchmark import TourismBenchmark
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp005")

ONTOLOGIES_DIR = PROJECT_ROOT / "artifacts" / "ontologies"
DATA_DIR       = PROJECT_ROOT / "data"
REPORTS_DIR    = PROJECT_ROOT / "artifacts" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp005: Constraint enforcement analysis")
    parser.add_argument("--ontology-version", default="0.3.0")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(
        log_level=args.log_level,
        log_file=str(PROJECT_ROOT / "artifacts" / "runs" / "exp005_constraints.log"),
    )

    logger.info("=== Experiment 005: Constraint Enforcement Analysis ===")

    # ── Setup ─────────────────────────────────────────────────────────────────
    graph = TourismGraphStore.from_version(
        args.ontology_version, ontology_dir=ONTOLOGIES_DIR
    )

    test_path = DATA_DIR / "splits" / "test.jsonl"
    if not test_path.exists():
        test_path = DATA_DIR / "splits" / "test.json"
    test_records: list[dict] = []
    if test_path.exists():
        if test_path.suffix == ".jsonl":
            with open(test_path) as f:
                test_records = [json.loads(line) for line in f if line.strip()]
        else:
            test_records = load_json(test_path)

    # Load or build benchmark
    bench_path = REPORTS_DIR / "benchmark.json"
    if bench_path.exists():
        bench = TourismBenchmark.load(bench_path)
    else:
        concepts = load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")
        bench    = TourismBenchmark.build(
            test_path, concepts, n=args.n_queries, seed=args.seed
        )
        bench.save(bench_path)

    items = bench.items[: args.n_queries]

    retriever = HybridRetriever(graph, records=test_records)

    import os
    from src.llm.client import LLMClient
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    client = LLMClient(
        backend="local" if hf_token else "auto",
        model_key=local_key, load_in_4bit=True, hf_token=hf_token, max_tokens=512,
    )
    generator = ConstrainedGenerator(client if client.is_available else None, graph)

    # ── Run onto_rag vs constrained ───────────────────────────────────────────
    comparison: list[dict] = []
    violation_type_counts: dict[str, int] = defaultdict(int)
    violations_reduced = 0

    logger.info(f"Running constraint comparison on {len(items)} queries …")
    for item in items:
        ctx = retriever.retrieve(item.query, mode="hybrid")

        onto_result   = generator.generate(item.query, ctx, mode="onto_rag")
        const_result  = generator.generate(item.query, ctx, mode="constrained")

        for v in const_result.violations:
            violation_type_counts[v.violation_type] += 1

        # Count how many violations the constraint mode detected
        # (violations are only computed for "constrained" mode)
        n_v = len(const_result.violations)

        comparison.append({
            "item_id":        item.item_id,
            "query":          item.query[:100],
            "onto_answer_len": len(onto_result.answer.split()),
            "const_answer_len": len(const_result.answer.split()),
            "n_violations":   n_v,
            "violation_types": [v.violation_type for v in const_result.violations],
            "is_grounded":    const_result.is_grounded,
        })

    # Stats
    n_with_violations = sum(1 for c in comparison if c["n_violations"] > 0)
    total_violations  = sum(c["n_violations"] for c in comparison)
    hallucination_rate = n_with_violations / len(comparison) if comparison else 0.0

    # ── Save ──────────────────────────────────────────────────────────────────
    report = {
        "experiment":       "exp005_constraints",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "ontology_version": args.ontology_version,
        "n_queries":        len(items),
        "summary": {
            "n_with_violations":   n_with_violations,
            "total_violations":    total_violations,
            "hallucination_rate":  round(hallucination_rate, 3),
            "avg_violations_per_answer": round(total_violations / max(len(comparison), 1), 3),
            "violation_type_counts": dict(violation_type_counts),
        },
        "per_item": comparison,
    }
    report_path = REPORTS_DIR / "exp005_constraints_report.json"
    save_json(report, report_path)

    print("\n" + "=" * 60)
    print("EXPERIMENT 005 — CONSTRAINT ENFORCEMENT COMPLETE")
    print("=" * 60)
    print(f"  Queries analysed  : {len(items)}")
    print(f"  With violations   : {n_with_violations} ({hallucination_rate:.1%})")
    print(f"  Total violations  : {total_violations}")
    print(f"\n  Violation types:")
    for vtype, cnt in sorted(violation_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {vtype:<30} {cnt}")
    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
