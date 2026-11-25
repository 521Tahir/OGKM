"""
Unified Benchmark Runner — runs the full 7-category evaluation suite.

Evaluation categories (maps to paper §11):
  1. Ontology Quality      (OntologyMetricsCalculator)
  2. Retrieval Quality     (RetrievalMetricsCalculator)
  3. Response Quality      (ResponseMetricsCalculator)
  4. Hallucination         (HallucinationEvaluator)
  5. Explainability        (ExplainabilityEvaluator)
  6. Robustness            (RobustnessEvaluator)
  7. Efficiency            (latency / throughput stub)

Usage:
    cd /home/tahirahmad/Ontology_Project
    set -a && source .env && set +a

    # Full offline run (no LLM):
    python -m src.evaluation.benchmark_runner \
        --ontology-version 0.7.0 --n-queries 200

    # With real LLM (GPU 7):
    CUDA_VISIBLE_DEVICES=7 python -m src.evaluation.benchmark_runner \
        --ontology-version 0.7.0 --n-queries 200 --use-llm

Output:
    artifacts/reports/full_evaluation_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.ontology_metrics    import OntologyMetricsCalculator
from src.evaluation.retrieval_metrics   import RetrievalMetricsCalculator
from src.evaluation.response_metrics    import ResponseMetricsCalculator
from src.evaluation.hallucination_eval  import HallucinationEvaluator
from src.evaluation.explainability_eval import ExplainabilityEvaluator
from src.evaluation.robustness_eval     import RobustnessEvaluator
from src.graph.graph_store              import TourismGraphStore
from src.utils.io                       import load_json, save_json, ensure_dir
from src.utils.logging_utils            import get_logger, setup_logging

logger = get_logger("evaluation.benchmark_runner")

ONTOLOGIES_DIR = ROOT / "artifacts" / "ontologies"
DATA_DIR       = ROOT / "data"
REPORTS_DIR    = ROOT / "artifacts" / "reports"
FIGURES_DIR    = ROOT / "artifacts" / "figures"

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]


# ── Category runners ──────────────────────────────────────────────────────────

def run_ontology_quality(
    concepts: list[dict],
    relations: list[dict],
    rules: list[dict],
    hierarchy: list[dict],
    test_records: list[dict],
    version: str,
) -> dict:
    """Category 1: Ontology quality metrics."""
    logger.info("Category 1: Ontology Quality ...")
    calc = OntologyMetricsCalculator(concepts, relations, rules, version)
    snap = calc.compute(test_records=test_records, hierarchy=hierarchy)
    return snap.to_dict()


def run_retrieval_quality(
    graph: TourismGraphStore,
    test_records: list[dict],
    benchmark_items: list,
    concept_names: set[str],
    n_queries: int,
) -> dict:
    """Category 2: Retrieval quality metrics (offline, no LLM)."""
    logger.info("Category 2: Retrieval Quality ...")
    from src.evaluation.retrieval_metrics import RetrievalMetricsCalculator
    from src.retrieval.concept_linker     import ConceptLinker

    calc   = RetrievalMetricsCalculator(concept_names)
    linker = ConceptLinker(graph)

    results: dict[str, list] = {m: [] for m in ["onto_only", "no_retrieval"]}

    items = benchmark_items[:n_queries]
    for i, item in enumerate(items):
        q = item.query if hasattr(item, "query") else item.get("query", "")
        expected = item.expected_concepts if hasattr(item, "expected_concepts") else item.get("expected_concepts", [])

        qid = getattr(item, "item_id", i)

        # no_retrieval
        res_none = calc.score_item(
            query_id=qid, query=q, mode="no_retrieval",
            retrieved_concepts=[], retrieved_relations=[], relevant_concepts=expected,
        )
        results["no_retrieval"].append(res_none)

        # onto_only via concept linker
        linked = linker.link(q)
        retrieved_concepts = [lc["concept"] for lc in linked if lc["concept"].get("name")]
        res_onto = calc.score_item(
            query_id=qid, query=q, mode="onto_only",
            retrieved_concepts=retrieved_concepts, retrieved_relations=[],
            relevant_concepts=expected,
        )
        results["onto_only"].append(res_onto)

    aggs = {
        mode: calc.aggregate(results[mode], mode).to_dict()
        for mode in results
    }
    return aggs


def run_response_quality(
    generator,
    retriever,
    benchmark_items: list,
    n_queries: int,
    concept_names: set[str],
) -> dict:
    """Category 3: Response quality metrics."""
    logger.info("Category 3: Response Quality ...")
    calc  = ResponseMetricsCalculator(ontology_names=concept_names)
    items = benchmark_items[:n_queries]

    results: dict[str, list] = {m: [] for m in MODES}
    retrieval_mode_map = {
        "no_rag": None, "doc_rag": "text_only",
        "onto_rag": "ontology_only", "constrained": "hybrid",
    }

    for mode in MODES:
        rmode = retrieval_mode_map[mode]
        for i, item in enumerate(items):
            q = item.query if hasattr(item, "query") else item.get("query", "")
            expected = (item.expected_concepts
                        if hasattr(item, "expected_concepts")
                        else item.get("expected_concepts", []))
            ctx = retriever.retrieve(q, mode=rmode) if rmode else None
            gen = generator.generate(q, ctx, mode=mode)
            m = calc.score(
                query_id=i, query=q, answer=gen.answer, mode=mode,
                expected_concepts=expected,
            )
            results[mode].append(m)

    aggs = {
        mode: calc.aggregate(results[mode], mode).to_dict()
        for mode in MODES
    }
    return aggs


def run_hallucination(
    generator,
    retriever,
    benchmark_items: list,
    n_queries: int,
    concepts: list[dict],
    relations: list[dict],
) -> dict:
    """Category 4: Hallucination evaluation."""
    logger.info("Category 4: Hallucination ...")
    evaluator = HallucinationEvaluator(concepts, relations)
    items = benchmark_items[:n_queries]
    retrieval_mode_map = {
        "no_rag": None, "doc_rag": "text_only",
        "onto_rag": "ontology_only", "constrained": "hybrid",
    }

    results: dict[str, list] = {m: [] for m in MODES}
    for mode in MODES:
        rmode = retrieval_mode_map[mode]
        for item in items:
            q = item.query if hasattr(item, "query") else item.get("query", "")
            ctx = retriever.retrieve(q, mode=rmode) if rmode else None
            gen = generator.generate(q, ctx, mode=mode)
            ctx_concepts  = ctx.subgraph.concepts  if ctx and ctx.subgraph else []
            ctx_relations = ctx.subgraph.relations if ctx and ctx.subgraph else []
            report = evaluator.evaluate(q, gen.answer, mode,
                                        context_concepts=ctx_concepts,
                                        context_relations=ctx_relations)
            results[mode].append(report)

    aggs = {
        mode: evaluator.summarize(results[mode], mode).to_dict()
        for mode in MODES
    }
    return aggs


def run_explainability(
    generator,
    retriever,
    benchmark_items: list,
    n_queries: int,
    concepts: list[dict],
    relations: list[dict],
    graph=None,
) -> dict:
    """Category 5: Explainability evaluation."""
    logger.info("Category 5: Explainability ...")
    from src.generation.explanation_builder import ExplanationBuilder
    from src.graph.graph_store import TourismGraphStore

    evaluator   = ExplainabilityEvaluator(concepts, relations)
    exp_builder = ExplanationBuilder(retriever.graph if hasattr(retriever, "graph") else graph)
    items = benchmark_items[:n_queries]
    retrieval_mode_map = {
        "no_rag": None, "doc_rag": "text_only",
        "onto_rag": "ontology_only", "constrained": "hybrid",
    }

    results: dict[str, list] = {m: [] for m in MODES}
    for mode in MODES:
        rmode = retrieval_mode_map[mode]
        for i, item in enumerate(items):
            q = item.query if hasattr(item, "query") else item.get("query", "")
            ctx = retriever.retrieve(q, mode=rmode) if rmode else None
            gen = generator.generate(q, ctx, mode=mode)
            ctx_concepts  = ctx.subgraph.concepts  if ctx and ctx.subgraph else []
            ctx_relations = ctx.subgraph.relations if ctx and ctx.subgraph else []
            trace = exp_builder.build(gen, ctx.subgraph if ctx else None)
            report = evaluator.evaluate(
                query_id=i, query=q, answer=gen.answer, mode=mode,
                context_concepts=ctx_concepts,
                context_relations=ctx_relations,
                explanation_trace=trace.to_dict() if hasattr(trace, "to_dict") else {},
            )
            results[mode].append(report)

    aggs = {
        mode: evaluator.aggregate(results[mode], mode).to_dict()
        for mode in MODES
    }
    return aggs


def run_robustness(
    generator,
    retriever,
    n_queries: int,
    concepts: list[dict],
) -> dict:
    """Category 6: Robustness evaluation using the robustness benchmark."""
    logger.info("Category 6: Robustness ...")
    evaluator = RobustnessEvaluator(concepts)

    # Load or build robustness benchmark
    bench_path = REPORTS_DIR / "robustness_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
    else:
        # Use hallucination benchmark as proxy
        hall_path = REPORTS_DIR / "hallucination_benchmark.json"
        if hall_path.exists():
            hall = json.loads(hall_path.read_text())
            bench = {"tests": [
                {"type": "adversarial_resistance",
                 "id": q["id"],
                 "query": q["query"],
                 "fake_entity": q.get("fake_entity", "")}
                for q in hall["categories"]["non_existent_places"]["queries"]
            ] + [
                {"type": "incomplete_query",
                 "id": q["id"],
                 "query": q["query"]}
                for q in hall["categories"]["ambiguous"]["queries"]
            ]}
        else:
            bench = {"tests": []}

    tests = bench.get("tests", [])[:n_queries]
    if not tests:
        logger.warning("No robustness tests found — returning empty results")
        return {}

    retrieval_mode_map = {
        "no_rag": None, "doc_rag": "text_only",
        "onto_rag": "ontology_only", "constrained": "hybrid",
    }

    results: dict[str, list] = {m: [] for m in MODES}
    for mode in MODES:
        rmode = retrieval_mode_map[mode]
        for test in tests:
            q   = test.get("query", "")
            ctx = retriever.retrieve(q, mode=rmode) if rmode else None
            gen = generator.generate(q, ctx, mode=mode)
            ans = gen.answer

            ttype = test.get("type", "incomplete_query")
            if ttype == "adversarial_resistance":
                report = evaluator.adversarial_resistance(
                    test["id"], mode, test.get("fake_entity", q), ans
                )
            elif ttype == "incomplete_query":
                report = evaluator.incomplete_query_score(test["id"], mode, q, ans)
            elif ttype == "paraphrase_consistency":
                report = evaluator.paraphrase_consistency(
                    test["id"], mode, ans,
                    test.get("answer2", ans),
                )
            elif ttype == "constraint_conflict":
                report = evaluator.constraint_conflict_score(
                    test["id"], mode, ans,
                    test.get("constraints", []),
                )
            else:
                report = evaluator.unseen_destination_score(
                    test["id"], mode, test.get("destination", q), ans
                )
            results[mode].append(report)

    aggs = {
        mode: evaluator.aggregate(results[mode], mode).to_dict()
        for mode in MODES
    }
    return aggs


def run_efficiency(
    generator,
    retriever,
    benchmark_items: list,
    n_sample: int = 20,
) -> dict:
    """Category 7: Efficiency stub — latency and throughput."""
    logger.info("Category 7: Efficiency ...")
    items = benchmark_items[:n_sample]
    retrieval_mode_map = {
        "no_rag": None, "doc_rag": "text_only",
        "onto_rag": "ontology_only", "constrained": "hybrid",
    }

    results = {}
    for mode in MODES:
        rmode = retrieval_mode_map[mode]
        latencies = []
        for item in items:
            q  = item.query if hasattr(item, "query") else item.get("query", "")
            t0 = time.perf_counter()
            ctx = retriever.retrieve(q, mode=rmode) if rmode else None
            gen = generator.generate(q, ctx, mode=mode)
            latencies.append(time.perf_counter() - t0)

        results[mode] = {
            "mode":             mode,
            "n_samples":        len(latencies),
            "avg_latency_s":    round(mean(latencies), 3) if latencies else 0,
            "p50_latency_s":    round(sorted(latencies)[len(latencies)//2], 3) if latencies else 0,
            "p95_latency_s":    round(sorted(latencies)[int(len(latencies)*0.95)], 3) if latencies else 0,
            "throughput_qps":   round(1.0 / mean(latencies), 2) if latencies else 0,
        }
    return results


# ── Full evaluation report ────────────────────────────────────────────────────

class FullEvaluationReport:
    """Container for all 7 evaluation categories."""

    def __init__(
        self,
        ontology_version: str,
        n_queries: int,
        ontology_quality:  dict,
        retrieval_quality: dict,
        response_quality:  dict,
        hallucination:     dict,
        explainability:    dict,
        robustness:        dict,
        efficiency:        dict,
        elapsed_s:         float,
    ):
        self.ontology_version = ontology_version
        self.n_queries        = n_queries
        self.timestamp        = datetime.now(timezone.utc).isoformat()
        self.elapsed_s        = elapsed_s
        self.categories = {
            "ontology_quality":  ontology_quality,
            "retrieval_quality": retrieval_quality,
            "response_quality":  response_quality,
            "hallucination":     hallucination,
            "explainability":    explainability,
            "robustness":        robustness,
            "efficiency":        efficiency,
        }

    def to_dict(self) -> dict:
        return {
            "experiment":       "full_evaluation",
            "timestamp":        self.timestamp,
            "ontology_version": self.ontology_version,
            "n_queries":        self.n_queries,
            "elapsed_s":        round(self.elapsed_s),
            "categories":       self.categories,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        logger.info("Full evaluation report saved → %s", path)

    def markdown_summary(self) -> str:
        lines = [
            f"# Full Evaluation Report — v{self.ontology_version}",
            f"\n**Date:** {self.timestamp[:10]} | "
            f"**Queries:** {self.n_queries} | "
            f"**Elapsed:** {self.elapsed_s/60:.1f} min",
        ]

        # Hallucination summary
        hall = self.categories.get("hallucination", {})
        if hall:
            lines += [
                "\n## Hallucination Rates",
                "| Mode | Halluc% | Entity% | Numeric% | Type Confusion% |",
                "|------|:-------:|:-------:|:--------:|:---------------:|",
            ]
            for mode in MODES:
                if mode in hall:
                    h = hall[mode]
                    lines.append(
                        f"| {mode} "
                        f"| {h.get('hallucination_rate',0)*100:.1f}% "
                        f"| {h.get('entity_hallucination_rate',0)*100:.1f}% "
                        f"| {h.get('numeric_hallucination_rate',0)*100:.1f}% "
                        f"| {h.get('type_confusion_rate',0)*100:.1f}% |"
                    )

        # Response quality summary
        rq = self.categories.get("response_quality", {})
        if rq:
            lines += [
                "\n## Response Quality",
                "| Mode | Factual | Complete | Instruct | Constraint% |",
                "|------|:-------:|:--------:|:--------:|:-----------:|",
            ]
            for mode in MODES:
                if mode in rq:
                    r = rq[mode]
                    lines.append(
                        f"| {mode} "
                        f"| {r.get('avg_factual_correctness',0):.3f} "
                        f"| {r.get('avg_completeness',0):.3f} "
                        f"| {r.get('avg_instruction_satisfaction',0):.3f} "
                        f"| {r.get('constraint_satisfaction_rate',0)*100:.1f}% |"
                    )

        # Robustness summary
        rob = self.categories.get("robustness", {})
        if rob:
            lines += [
                "\n## Robustness Pass Rate",
                "| Mode | Pass% |",
                "|------|:-----:|",
            ]
            for mode in MODES:
                if mode in rob:
                    lines.append(
                        f"| {mode} | {rob[mode].get('overall_pass_rate',0)*100:.1f}% |"
                    )

        return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full 7-category evaluation suite")
    parser.add_argument("--ontology-version", default="0.7.0")
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Queries for LLM-based categories (default 200)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use real LLM for generation (default: offline mode)")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["ontology", "retrieval", "response",
                                 "hallucination", "explainability", "robustness", "efficiency"],
                        help="Categories to skip")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=str(ROOT / "artifacts/runs/full_evaluation.log"))
    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Full Benchmark Evaluation — v%s | %d queries | LLM=%s",
                args.ontology_version, args.n_queries, args.use_llm)
    logger.info("=" * 60)

    # ── Load ontology ─────────────────────────────────────────────────────────
    vdir      = ONTOLOGIES_DIR / args.ontology_version
    concepts  = load_json(vdir / "concepts.json")  if (vdir / "concepts.json").exists()  else []
    relations = load_json(vdir / "relations.json") if (vdir / "relations.json").exists() else []
    rules     = load_json(vdir / "rules.json")     if (vdir / "rules.json").exists()     else []
    hierarchy = load_json(vdir / "hierarchy.json") if (vdir / "hierarchy.json").exists() else []

    logger.info("Ontology v%s: %d concepts, %d relations", args.ontology_version, len(concepts), len(relations))

    # ── Load test data ────────────────────────────────────────────────────────
    test_path = DATA_DIR / "splits" / "test.jsonl"
    test_records = []
    if test_path.exists():
        with open(test_path) as f:
            test_records = [json.loads(l) for l in f if l.strip()]

    # ── Load benchmark ────────────────────────────────────────────────────────
    from src.evaluation.benchmark import TourismBenchmark
    bench_path = REPORTS_DIR / "benchmark.json"
    if bench_path.exists():
        bench = TourismBenchmark.load(bench_path)
    else:
        bench = TourismBenchmark.build(test_path, concepts, n=args.n_queries, seed=42)
        bench.save(bench_path)
    benchmark_items = bench.items[: args.n_queries]

    # ── Graph + pipeline ──────────────────────────────────────────────────────
    graph = TourismGraphStore.from_version(args.ontology_version, ontology_dir=ONTOLOGIES_DIR)
    concept_names = {c["name"] for c in concepts if c.get("name")}

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.constrained_generator import ConstrainedGenerator
    retriever = HybridRetriever(graph, records=test_records)

    client = None
    if args.use_llm:
        from src.llm.client import LLMClient
        hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
        client = LLMClient(
            backend="local", model_key=local_key,
            load_in_4bit=True, hf_token=hf_token, max_tokens=512,
        )
        if not client.is_available:
            logger.warning("LLM not available — running in offline mode")
            client = None

    generator = ConstrainedGenerator(client, graph)

    # ── Run categories ────────────────────────────────────────────────────────
    skip = set(args.skip)

    ontology_q = (
        run_ontology_quality(concepts, relations, rules, hierarchy, test_records, args.ontology_version)
        if "ontology" not in skip else {}
    )
    retrieval_q = (
        run_retrieval_quality(graph, test_records, benchmark_items, concept_names, args.n_queries)
        if "retrieval" not in skip else {}
    )
    response_q = (
        run_response_quality(generator, retriever, benchmark_items, args.n_queries, concept_names)
        if "response" not in skip else {}
    )
    hallucination = (
        run_hallucination(generator, retriever, benchmark_items, args.n_queries, concepts, relations)
        if "hallucination" not in skip else {}
    )
    explainability = (
        run_explainability(generator, retriever, benchmark_items, args.n_queries, concepts, relations, graph=graph)
        if "explainability" not in skip else {}
    )
    robustness = (
        run_robustness(generator, retriever, args.n_queries, concepts)
        if "robustness" not in skip else {}
    )
    efficiency = (
        run_efficiency(generator, retriever, benchmark_items, n_sample=min(20, args.n_queries))
        if "efficiency" not in skip else {}
    )

    # ── Compile report ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    report  = FullEvaluationReport(
        ontology_version=args.ontology_version,
        n_queries=args.n_queries,
        ontology_quality=ontology_q,
        retrieval_quality=retrieval_q,
        response_quality=response_q,
        hallucination=hallucination,
        explainability=explainability,
        robustness=robustness,
        efficiency=efficiency,
        elapsed_s=elapsed,
    )
    out_path = REPORTS_DIR / "full_evaluation_report.json"
    report.save(out_path)

    md = report.markdown_summary()
    md_path = FIGURES_DIR / "full_evaluation_summary.md"
    md_path.write_text(md)

    print("\n" + "=" * 60)
    print("FULL EVALUATION COMPLETE")
    print("=" * 60)
    print(md)
    print(f"\nReport : {out_path}")
    print(f"Summary: {md_path}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
