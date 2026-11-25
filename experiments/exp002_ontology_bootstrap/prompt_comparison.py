"""
Experiment 002b — Prompt Strategy Comparison

Compares 4 prompting strategies on a sample of 50 records:
  1. naive
  2. cot (chain-of-thought)
  3. graph_of_thought
  4. metacognitive

Measures per strategy:
  - avg concepts per record
  - avg relations per record
  - schema validity rate
  - avg confidence
  - parse success rate
  - validation pass rate

Outputs:
  - artifacts/reports/prompt_comparison.json
  - artifacts/reports/prompt_comparison.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import load_jsonl, save_json, ensure_dir
from src.data.loaders import TourismRecord
from src.ontology.offline_extractor import OfflineExtractor
from src.ontology.validator import OntologyValidator
from src.ontology.schema import TourismOntologySchema

logger = get_logger("exp002b")
SCHEMA = TourismOntologySchema


def evaluate_strategy_offline(
    records: list[TourismRecord],
    extractor: OfflineExtractor,
    validator: OntologyValidator,
    strategy_name: str,
) -> dict:
    """Run offline extractor and collect quality metrics."""
    metrics = defaultdict(list)
    start = time.perf_counter()

    for rec in records:
        combined = f"{rec.input_text}\n{rec.output_text}"
        concepts = extractor.extract_concepts(combined, rec.record_id)
        relations = extractor.extract_relations(combined, concepts, rec.record_id)

        from src.ontology.schema import OntologyCandidate
        candidate = OntologyCandidate(
            record_id=rec.record_id,
            concepts=concepts,
            relations=relations,
        )
        result = validator.validate_candidate(candidate)

        metrics["n_concepts"].append(len(concepts))
        metrics["n_relations"].append(len(relations))
        metrics["pass"].append(1 if result.status == "pass" else 0)
        metrics["warning"].append(1 if result.status == "warning" else 0)
        metrics["fail"].append(1 if result.status == "fail" else 0)

        if concepts:
            metrics["avg_confidence"].append(
                mean(c.provenance.confidence for c in concepts)
            )
        # Schema validity: fraction of concepts with valid class
        valid_cls = sum(1 for c in concepts if SCHEMA.is_valid_class(c.concept_class))
        if concepts:
            metrics["schema_valid_pct"].append(valid_cls / len(concepts))

    elapsed = time.perf_counter() - start
    n = len(records)
    return {
        "strategy": strategy_name,
        "mode": "offline",
        "n_records": n,
        "avg_concepts_per_record": round(mean(metrics["n_concepts"]), 2),
        "avg_relations_per_record": round(mean(metrics["n_relations"]), 2),
        "avg_confidence": round(mean(metrics["avg_confidence"]) if metrics["avg_confidence"] else 0, 3),
        "schema_validity_rate": round(mean(metrics["schema_valid_pct"]) if metrics["schema_valid_pct"] else 0, 3),
        "validation_pass_rate": round(mean(metrics["pass"]), 3),
        "validation_warning_rate": round(mean(metrics["warning"]), 3),
        "validation_fail_rate": round(mean(metrics["fail"]), 3),
        "total_elapsed_sec": round(elapsed, 2),
        "avg_sec_per_record": round(elapsed / n, 4),
    }


def run_llm_strategy(
    records: list[TourismRecord],
    strategy: str,
    client,
    pm,
    validator: OntologyValidator,
) -> dict:
    """Run LLM-based extractor for a given strategy."""
    from src.ontology.concept_extractor import ConceptExtractor
    from src.ontology.relation_extractor import RelationExtractor
    from src.ontology.schema import OntologyCandidate

    ce = ConceptExtractor(client, pm, strategy=strategy)
    re_ = RelationExtractor(client, pm, strategy=strategy)
    metrics = defaultdict(list)
    start = time.perf_counter()

    for rec in records:
        combined = f"{rec.input_text}\n{rec.output_text}"
        try:
            concepts = ce.extract(combined, rec.record_id)
            concept_dicts = [{"name": c.name, "concept_class": c.concept_class} for c in concepts]
            relations = re_.extract(combined, concept_dicts, rec.record_id)
            candidate = OntologyCandidate(rec.record_id, concepts, relations)
            result = validator.validate_candidate(candidate)

            metrics["n_concepts"].append(len(concepts))
            metrics["n_relations"].append(len(relations))
            metrics["pass"].append(1 if result.status == "pass" else 0)
            metrics["parse_success"].append(1)
            if concepts:
                metrics["avg_confidence"].append(mean(c.provenance.confidence for c in concepts))
                valid = sum(1 for c in concepts if SCHEMA.is_valid_class(c.concept_class))
                metrics["schema_valid_pct"].append(valid / len(concepts))
        except Exception as e:
            logger.error(f"  Record {rec.record_id} failed: {e}")
            metrics["n_concepts"].append(0)
            metrics["n_relations"].append(0)
            metrics["pass"].append(0)
            metrics["parse_success"].append(0)

    elapsed = time.perf_counter() - start
    n = len(records)
    return {
        "strategy": strategy,
        "mode": "llm",
        "n_records": n,
        "avg_concepts_per_record": round(mean(metrics["n_concepts"]), 2),
        "avg_relations_per_record": round(mean(metrics["n_relations"]), 2),
        "avg_confidence": round(mean(metrics["avg_confidence"]) if metrics["avg_confidence"] else 0, 3),
        "schema_validity_rate": round(mean(metrics["schema_valid_pct"]) if metrics["schema_valid_pct"] else 0, 3),
        "validation_pass_rate": round(mean(metrics["pass"]), 3),
        "parse_success_rate": round(mean(metrics["parse_success"]), 3),
        "total_elapsed_sec": round(elapsed, 2),
        "avg_sec_per_record": round(elapsed / n, 4),
    }


def save_comparison_report(results: list[dict], output_dir: Path) -> None:
    """Save JSON and Markdown comparison report."""
    save_json({"results": results}, output_dir / "prompt_comparison.json")

    lines = [
        "# Prompt Strategy Comparison Report",
        "",
        "| Strategy | Mode | Concepts/rec | Relations/rec | Confidence | Schema Valid | Pass Rate | Sec/rec |",
        "|----------|------|-------------|---------------|-----------|--------------|-----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['strategy']} | {r['mode']} | {r['avg_concepts_per_record']} | "
            f"{r['avg_relations_per_record']} | {r['avg_confidence']} | "
            f"{r['schema_validity_rate']:.1%} | {r['validation_pass_rate']:.1%} | "
            f"{r['avg_sec_per_record']} |"
        )
    lines += ["", "## Notes", "- Offline strategies use vocabulary+NER (no LLM cost).",
              "- LLM strategies require ANTHROPIC_API_KEY."]
    (output_dir / "prompt_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparison report saved → {output_dir / 'prompt_comparison.md'}")


def main() -> None:
    set_seed(42)
    setup_logging(log_level="INFO", log_file="artifacts/runs/exp002b_comparison.log")
    ensure_dir("artifacts/reports")

    logger.info("=== Exp 002b: Prompt Strategy Comparison ===")

    SAMPLE_SIZE = 50
    records_raw = load_jsonl(PROJECT_ROOT / "data" / "processed" / "cleaned_records.jsonl")
    records = [
        TourismRecord(r["record_id"], r["Input"], r["CoT"], r["Output"])
        for r in records_raw[:SAMPLE_SIZE]
    ]
    logger.info(f"Sample: {len(records)} records")

    validator = OntologyValidator()
    results: list[dict] = []

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    strategies_to_run = ["naive", "cot", "metacognitive", "graph_of_thought"]

    if api_key:
        from src.llm.client import LLMClient
        from src.llm.prompt_manager import PromptManager
        client = LLMClient(model="claude-haiku-4-5-20251001")
        pm = PromptManager(prompts_root=PROJECT_ROOT / "prompts")

        for strategy in strategies_to_run:
            logger.info(f"Running LLM strategy: {strategy} ...")
            result = run_llm_strategy(records, strategy, client, pm, validator)
            results.append(result)
            logger.info(f"  {strategy}: {result['avg_concepts_per_record']} concepts/rec, "
                        f"pass={result['validation_pass_rate']:.1%}")
    else:
        logger.info("No API key — running offline evaluation for all strategies")
        extractor = OfflineExtractor()
        # Offline: same extractor, but we tag each run with a strategy name
        # to show the report framework works end-to-end
        for strategy in strategies_to_run:
            result = evaluate_strategy_offline(records, extractor, validator, strategy)
            results.append(result)
            logger.info(f"  {strategy} (offline): "
                        f"{result['avg_concepts_per_record']} concepts/rec, "
                        f"pass={result['validation_pass_rate']:.1%}")

    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    save_comparison_report(results, report_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT 002b — PROMPT STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Concepts':>8} {'Relations':>10} {'Conf':>6} {'Pass%':>7}")
    print("-" * 55)
    for r in results:
        print(f"  {r['strategy']:<18} {r['avg_concepts_per_record']:>8} "
              f"{r['avg_relations_per_record']:>10} {r['avg_confidence']:>6.3f} "
              f"{r['validation_pass_rate']:>7.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
