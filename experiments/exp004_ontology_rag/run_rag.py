"""
Experiment 004 — Ontology-RAG Pipeline

Runs all four generation conditions on a sample of queries from the test split
and produces a comparison report.

Conditions:
  1. no_rag        — LLM with no retrieval (parametric knowledge only)
  2. doc_rag       — LLM + text passage retrieval (BM25 over records)
  3. onto_rag      — LLM + ontology subgraph context
  4. constrained   — onto_rag + post-generation constraint checking

Output:
  artifacts/reports/exp004_rag_report.json
  artifacts/reports/exp004_rag_results.jsonl  (per-item results)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_store import TourismGraphStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.constrained_generator import ConstrainedGenerator
from src.generation.explanation_builder import ExplanationBuilder
from src.evaluation.benchmark import TourismBenchmark
from src.evaluation.metrics import EvaluationMetrics
from src.utils.io import load_json, save_json, save_jsonl, ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp004")

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
ONTOLOGIES_DIR = PROJECT_ROOT / "artifacts" / "ontologies"
DATA_DIR       = PROJECT_ROOT / "data"
REPORTS_DIR    = PROJECT_ROOT / "artifacts" / "reports"

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp004: Ontology-RAG pipeline")
    parser.add_argument("--ontology-version", default="0.3.0",
                        help="Ontology version to use (default: 0.3.0)")
    parser.add_argument("--n-queries", type=int, default=50,
                        help="Number of benchmark queries (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=MODES,
                        choices=MODES, help="Which conditions to run")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(
        log_level=args.log_level,
        log_file=str(PROJECT_ROOT / "artifacts" / "runs" / "exp004_rag.log"),
    )

    logger.info("=== Experiment 004: Ontology-RAG Pipeline ===")
    logger.info(f"  Ontology version : {args.ontology_version}")
    logger.info(f"  Queries          : {args.n_queries}")
    logger.info(f"  Modes            : {args.modes}")

    # ── Load graph ────────────────────────────────────────────────────────────
    graph = TourismGraphStore.from_version(
        args.ontology_version, ontology_dir=ONTOLOGIES_DIR
    )
    stats = graph.stats()
    logger.info(
        f"Graph loaded: {stats['n_concepts']:,} concepts, "
        f"{stats['n_relations']:,} relations"
    )

    # ── Load test records for doc retrieval ───────────────────────────────────
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
        logger.info(f"Loaded {len(test_records):,} test records for doc retrieval")
    else:
        logger.warning(f"Test split not found at {test_path} — doc_rag will use empty corpus")

    # ── Build benchmark ───────────────────────────────────────────────────────
    bench_path = REPORTS_DIR / "benchmark.json"
    if bench_path.exists():
        bench = TourismBenchmark.load(bench_path)
        logger.info(f"Loaded existing benchmark ({len(bench)} items)")
    else:
        if not test_path.exists():
            logger.error("No test split found. Cannot build benchmark.")
            sys.exit(1)
        ontology_concepts = load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")
        bench = TourismBenchmark.build(
            test_path, ontology_concepts, n=args.n_queries, seed=args.seed
        )
        bench.save(bench_path)

    # Limit to n_queries
    items = bench.items[: args.n_queries]

    # ── Setup pipeline ────────────────────────────────────────────────────────
    retriever = HybridRetriever(graph, records=test_records)

    # LLM client — load Llama 3.1-8B in 4-bit on GPU 2,3
    import os
    from src.llm.client import LLMClient
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    # Force local backend when token available; fall back to offline if not
    if hf_token:
        backend = "local"
        logger.info(f"HF_TOKEN detected — loading {local_key} in 4-bit")
    else:
        backend = "auto"
        logger.warning("HF_TOKEN not set — will try auto-detect or use offline fallback")
    client = LLMClient(
        backend=backend,
        model_key=local_key,
        load_in_4bit=True,
        hf_token=hf_token,
        max_tokens=512,
    )
    if client.is_available:
        logger.info(f"LLM backend: {client.backend.upper()} | model: {getattr(client._client, 'model_id', '?')}")
    else:
        logger.info("No LLM available — using offline fallback")

    generator    = ConstrainedGenerator(client if client.is_available else None, graph)
    exp_builder  = ExplanationBuilder(graph)
    concept_names = {c["name"] for c in load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")}
    evaluator    = EvaluationMetrics(concept_names)

    # ── Run conditions ────────────────────────────────────────────────────────
    all_results:      list[dict] = []
    mode_item_metrics: dict[str, list] = {m: [] for m in args.modes}

    for mode in args.modes:
        logger.info(f"Running condition: {mode} ({len(items)} queries) …")

        # Map mode to retrieval mode
        retrieval_mode = {
            "no_rag":      "ontology_only",  # retriever not used
            "doc_rag":     "text_only",
            "onto_rag":    "ontology_only",
            "constrained": "hybrid",
        }.get(mode, "hybrid")

        for item in items:
            # Retrieve context
            if mode == "no_rag":
                ctx = None
            else:
                ctx = retriever.retrieve(item.query, mode=retrieval_mode)

            # Generate
            gen_result = generator.generate(item.query, ctx, mode=mode)

            # Explain
            subgraph_ctx = ctx.subgraph if ctx else None
            trace = exp_builder.build(gen_result, subgraph_ctx)

            # Score
            item_met = evaluator.score_item(gen_result, trace, item)
            mode_item_metrics[mode].append(item_met)

            # Collect
            all_results.append({
                "mode":    mode,
                "item_id": item.item_id,
                "query":   item.query,
                "answer":  gen_result.answer,
                "metrics": item_met.to_dict(),
                "trace":   trace.to_dict(),
            })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregates = {
        mode: evaluator.aggregate(mode_item_metrics[mode], mode).to_dict()
        for mode in args.modes
    }

    # ── Save results ──────────────────────────────────────────────────────────
    ensure_dir(REPORTS_DIR)
    results_path = REPORTS_DIR / "exp004_rag_results.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "experiment":       "exp004_ontology_rag",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "ontology_version": args.ontology_version,
        "n_queries":        len(items),
        "modes":            args.modes,
        "graph_stats":      stats,
        "aggregate_metrics": aggregates,
        "results_path":     str(results_path),
    }
    report_path = REPORTS_DIR / "exp004_rag_report.json"
    save_json(report, report_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXPERIMENT 004 — ONTOLOGY-RAG PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Ontology v{args.ontology_version} | {stats['n_concepts']:,} concepts | "
          f"{stats['n_relations']:,} relations")
    print(f"  Queries: {len(items)}\n")
    header = f"  {'Mode':<14} {'Recall':>8} {'Precision':>10} {'Halluc%':>9} {'Grounding':>10}"
    print(header)
    print("  " + "-" * 53)
    for mode in args.modes:
        a = aggregates[mode]
        print(
            f"  {mode:<14} {a['entity_recall']:>8.1%} {a['entity_precision']:>10.1%} "
            f"{a['hallucination_rate']:>9.1%} {a['avg_grounding']:>10.1%}"
        )
    print(f"\n  Report  : {report_path}")
    print(f"  Results : {results_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
