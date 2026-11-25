"""
Experiment 006 — Full Benchmark Evaluation

Evaluates all four pipeline conditions on the full benchmark (default 200 items)
and produces the final comparison tables and figures data for the research paper.

Output:
  artifacts/reports/exp006_evaluation_report.json
  artifacts/figures/exp006_comparison_table.md
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
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp006")

ONTOLOGIES_DIR = PROJECT_ROOT / "artifacts" / "ontologies"
DATA_DIR       = PROJECT_ROOT / "data"
REPORTS_DIR    = PROJECT_ROOT / "artifacts" / "reports"
FIGURES_DIR    = PROJECT_ROOT / "artifacts" / "figures"

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp006: Full benchmark evaluation")
    parser.add_argument("--ontology-version", default="0.3.0")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--tag", default="", help="suffix for output files (parallel shards)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    try:
        from src.utils.proctitle import set_proctitle
        set_proctitle()  # tahir_Ont-code (standing project requirement)
    except Exception:
        pass

    setup_logging(
        log_level=args.log_level,
        log_file=str(PROJECT_ROOT / "artifacts" / "runs" / "exp006_evaluation.log"),
    )

    logger.info("=== Experiment 006: Full Benchmark Evaluation ===")
    logger.info(f"  Queries : {args.n_queries}  |  Modes : {args.modes}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    graph = TourismGraphStore.from_version(
        args.ontology_version, ontology_dir=ONTOLOGIES_DIR
    )
    stats = graph.stats()

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
        if len(bench) < args.n_queries and test_path.exists():
            concepts = load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")
            bench = TourismBenchmark.build(
                test_path, concepts, n=args.n_queries, seed=args.seed
            )
            bench.save(bench_path)
    else:
        if not test_path.exists():
            logger.error("No test split and no saved benchmark found. Aborting.")
            sys.exit(1)
        concepts = load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")
        bench = TourismBenchmark.build(
            test_path, concepts, n=args.n_queries, seed=args.seed
        )
        bench.save(bench_path)

    items = bench.items[: args.n_queries]
    logger.info(f"Benchmark: {len(items)} items")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    retriever = HybridRetriever(graph, records=test_records)

    import os
    from src.llm.client import LLMClient
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    client = LLMClient(
        backend="local" if hf_token else "auto",
        model_key=local_key, load_in_4bit=True, hf_token=hf_token, max_tokens=512,
    )
    logger.info(f"LLM backend: {client.backend.upper() if client.is_available else 'OFFLINE'}")

    generator   = ConstrainedGenerator(client if client.is_available else None, graph)
    exp_builder = ExplanationBuilder(graph)
    concept_names = {c["name"] for c in load_json(ONTOLOGIES_DIR / args.ontology_version / "concepts.json")}
    evaluator   = EvaluationMetrics(concept_names)

    retrieval_mode_map = {
        "no_rag":      None,
        "doc_rag":     "text_only",
        "onto_rag":    "ontology_only",
        "constrained": "hybrid",
    }

    # ── Evaluation loop ───────────────────────────────────────────────────────
    mode_item_metrics: dict[str, list] = {m: [] for m in args.modes}
    all_results: list[dict] = []

    for mode in args.modes:
        rmode = retrieval_mode_map[mode]
        logger.info(f"  [{mode}] evaluating {len(items)} queries …")
        for item in items:
            ctx = retriever.retrieve(item.query, mode=rmode) if rmode else None
            gen = generator.generate(item.query, ctx, mode=mode)
            trace = exp_builder.build(gen, ctx.subgraph if ctx else None)
            m = evaluator.score_item(gen, trace, item)
            mode_item_metrics[mode].append(m)
            all_results.append({
                "mode": mode, "item_id": item.item_id,
                "query": item.query[:80], "answer": gen.answer[:200],
                "metrics": m.to_dict(),
            })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregates = {
        mode: evaluator.aggregate(mode_item_metrics[mode], mode).to_dict()
        for mode in args.modes
    }

    # ── Markdown comparison table ─────────────────────────────────────────────
    md_lines = [
        "# Experiment 006 — Benchmark Comparison",
        f"\n**Ontology:** v{args.ontology_version} | "
        f"**Queries:** {len(items)} | "
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "\n| Mode | Entity Recall | Entity Precision | Halluc% | Grounding | Avg Len |",
        "|------|:-------------:|:----------------:|:-------:|:---------:|:-------:|",
    ]
    for mode in args.modes:
        a = aggregates[mode]
        md_lines.append(
            f"| {mode:<12} | {a['entity_recall']:.1%} | {a['entity_precision']:.1%} | "
            f"{a['hallucination_rate']:.1%} | {a['avg_grounding']:.1%} | "
            f"{a['avg_answer_length']:.0f} |"
        )
    md_lines.extend([
        "\n## Research Question Targets",
        "- **RQ3** (onto-RAG vs doc-RAG): Compare entity_recall + grounding rows",
        "- **RQ4** (constraints): Compare halluc% rows for onto_rag vs constrained",
        "- **RQ5** (explanation): Grounding score column",
    ])
    md_table = "\n".join(md_lines)

    ensure_dir(FIGURES_DIR)
    table_path = FIGURES_DIR / f"exp006_comparison_table{args.tag}.md"
    with open(table_path, "w") as f:
        f.write(md_table)

    # ── Save full report ──────────────────────────────────────────────────────
    # per-query metric arrays per mode → enables real significance testing (§11.9)
    per_query_metrics = {
        mode: [mm.to_dict() for mm in mode_item_metrics[mode]]
        for mode in args.modes
    }
    report = {
        "experiment":        "exp006_evaluation",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "ontology_version":  args.ontology_version,
        "n_queries":         len(items),
        "modes":             args.modes,
        "graph_stats":       stats,
        "aggregate_metrics": aggregates,
        "per_query_metrics": per_query_metrics,
        "table_path":        str(table_path),
    }
    report_path = REPORTS_DIR / f"exp006_evaluation_report{args.tag}.json"
    save_json(report, report_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 006 — BENCHMARK EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Ontology v{args.ontology_version} | {stats['n_concepts']:,} concepts | "
          f"{stats['n_relations']:,} relations")
    print(f"  Queries: {len(items)} | Modes: {args.modes}\n")
    print(md_table)
    print(f"\n  Report : {report_path}")
    print(f"  Table  : {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
