"""
Experiment 002c — LLM Prompt Strategy Comparison

Compares 5 prompting strategies using the local Llama 3.1-8B model on a pilot
sample of records and ranks them on extraction quality metrics.

Strategies:
  1. naive
  2. cot (chain-of-thought)
  3. graph_of_thought
  4. metacognitive
  5. repair_after_failure

Metrics (per strategy):
  - avg_concepts_per_record
  - avg_relations_per_record
  - schema_validity_rate    (fraction of concepts with valid class)
  - avg_confidence
  - parse_success_rate      (fraction of records that produced valid JSON)
  - validation_pass_rate    (fraction that pass OntologyValidator)
  - avg_sec_per_record

Outputs:
  artifacts/reports/prompt_comparison_llm.json
  artifacts/reports/prompt_comparison_llm.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import load_jsonl, save_json, ensure_dir
from src.data.loaders import TourismRecord
from src.ontology.validator import OntologyValidator
from src.ontology.schema import TourismOntologySchema, OntologyCandidate

logger = get_logger("exp002c")
SCHEMA = TourismOntologySchema

STRATEGIES = ["naive", "cot", "graph_of_thought", "metacognitive", "repair_after_failure"]


# ── Per-strategy LLM run ──────────────────────────────────────────────────────

def run_strategy(
    records: list[TourismRecord],
    strategy: str,
    client,
    pm,
    validator: OntologyValidator,
) -> dict:
    """Run concept + relation extraction for one strategy and collect metrics."""
    from src.ontology.concept_extractor import ConceptExtractor
    from src.ontology.relation_extractor import RelationExtractor

    ce  = ConceptExtractor(client, pm, strategy=strategy)
    re_ = RelationExtractor(client, pm, strategy=strategy)

    metrics: dict[str, list] = defaultdict(list)
    start = time.perf_counter()

    for i, rec in enumerate(records):
        combined = f"{rec.input_text}\n{rec.output_text}"
        try:
            concepts  = ce.extract(combined, rec.record_id)
            c_dicts   = [{"name": c.name, "concept_class": c.concept_class} for c in concepts]
            relations = re_.extract(combined, c_dicts, rec.record_id)

            candidate = OntologyCandidate(rec.record_id, concepts, relations)
            result    = validator.validate_candidate(candidate)

            metrics["n_concepts"].append(len(concepts))
            metrics["n_relations"].append(len(relations))
            metrics["pass"].append(1 if result.status == "pass" else 0)
            metrics["parse_success"].append(1)

            if concepts:
                metrics["avg_conf"].append(mean(c.provenance.confidence for c in concepts))
                valid = sum(1 for c in concepts if SCHEMA.is_valid_class(c.concept_class))
                metrics["schema_valid"].append(valid / len(concepts))

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{strategy}] {i+1}/{len(records)} — "
                    f"concepts={metrics['n_concepts'][-1]}, "
                    f"relations={metrics['n_relations'][-1]}, "
                    f"pass={bool(metrics['pass'][-1])}"
                )

        except Exception as e:
            logger.error(f"  [{strategy}] record {rec.record_id} failed: {e}")
            metrics["n_concepts"].append(0)
            metrics["n_relations"].append(0)
            metrics["pass"].append(0)
            metrics["parse_success"].append(0)

    elapsed = time.perf_counter() - start
    n = len(records)

    return {
        "strategy":                  strategy,
        "mode":                      "llm",
        "n_records":                 n,
        "avg_concepts_per_record":   round(mean(metrics["n_concepts"]),    2),
        "avg_relations_per_record":  round(mean(metrics["n_relations"]),   2),
        "avg_confidence":            round(mean(metrics["avg_conf"])        if metrics["avg_conf"]        else 0.0, 3),
        "schema_validity_rate":      round(mean(metrics["schema_valid"])    if metrics["schema_valid"]    else 0.0, 3),
        "validation_pass_rate":      round(mean(metrics["pass"]),           3),
        "parse_success_rate":        round(mean(metrics["parse_success"]),  3),
        "total_elapsed_sec":         round(elapsed, 2),
        "avg_sec_per_record":        round(elapsed / max(n, 1), 4),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def save_report(results: list[dict], report_dir: Path) -> None:
    """Save JSON and Markdown reports."""
    json_path = report_dir / "prompt_comparison_llm.json"
    md_path   = report_dir / "prompt_comparison_llm.md"

    save_json({"results": results}, json_path)
    logger.info(f"JSON report → {json_path}")

    # Rank by combined score: schema_validity * parse_success * validation_pass
    for r in results:
        r["_score"] = (
            r["schema_validity_rate"]
            * r["parse_success_rate"]
            * (1 + r["avg_concepts_per_record"] / 10)
        )
    ranked = sorted(results, key=lambda x: x["_score"], reverse=True)

    lines = [
        "# LLM Prompt Strategy Comparison (Llama 3.1-8B)",
        "",
        "| Rank | Strategy | Concepts/rec | Relations/rec | Confidence | Schema Valid | Pass% | Parse% | Sec/rec |",
        "|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {r['strategy']} | {r['avg_concepts_per_record']} | "
            f"{r['avg_relations_per_record']} | {r['avg_confidence']:.3f} | "
            f"{r['schema_validity_rate']:.1%} | {r['validation_pass_rate']:.1%} | "
            f"{r['parse_success_rate']:.1%} | {r['avg_sec_per_record']:.2f}s |"
        )

    best = ranked[0]["strategy"]
    lines += [
        "",
        f"**Best strategy: `{best}`** (highest schema_validity × parse_success × concept_density)",
        "",
        "## Recommendation",
        f"Re-run exp002 with `--strategy {best} --version 0.6.0 --llm-records 2000` to produce ontology v0.6.0.",
        "",
        "## Notes",
        "- All strategies use `meta-llama/Llama-3.1-8B-Instruct` in 4-bit quantization.",
        f"- Sample size: {results[0]['n_records']} records.",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report → {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exp002c: LLM Prompt Strategy Comparison")
    p.add_argument("--n-records",   type=int, default=100,
                   help="Number of records to test per strategy (default: 100)")
    p.add_argument("--strategies",  nargs="+", default=STRATEGIES,
                   choices=STRATEGIES, help="Which strategies to compare")
    p.add_argument("--gpus",        default="2,3",
                   help="Comma-separated GPU IDs (default: 2,3)")
    p.add_argument("--load-in-4bit", action="store_true", default=True,
                   help="Use 4-bit quantization (default: True)")
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    try:
        from src.utils.proctitle import set_proctitle
        set_proctitle()  # tahir_Ont-code (standing project requirement)
    except Exception:
        pass
    ensure_dir("artifacts/runs")
    ensure_dir("artifacts/reports")
    setup_logging(
        log_level=args.log_level,
        log_file="artifacts/runs/exp002c_prompt_comparison.log",
    )

    logger.info("=== Experiment 002c: LLM Prompt Strategy Comparison ===")
    logger.info(f"  Strategies  : {args.strategies}")
    logger.info(f"  Records     : {args.n_records}")
    logger.info(f"  GPUs        : {args.gpus}")
    logger.info(f"  4-bit quant : {args.load_in_4bit}")

    # ── Load records ──────────────────────────────────────────────────────────
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_records.jsonl"
    records_raw = load_jsonl(data_path)
    records = [
        TourismRecord(
            record_id=r["record_id"],
            input_text=r["Input"],
            cot=r["CoT"],
            output_text=r["Output"],
            source_file=r.get("source_file", ""),
        )
        for r in records_raw[: args.n_records]
    ]
    logger.info(f"Loaded {len(records)} pilot records")

    # ── Load LLM ─────────────────────────────────────────────────────────────
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager

    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    gpu_ids   = [int(g) for g in args.gpus.split(",") if g.strip()]

    if not hf_token:
        logger.error("HF_TOKEN not set. Run: set -a && source .env && set +a")
        sys.exit(1)

    client = LLMClient(
        backend="local",
        model_key=local_key,
        load_in_4bit=args.load_in_4bit,
        gpu_ids=gpu_ids,
        hf_token=hf_token,
        max_tokens=512,
    )
    if not client.is_available:
        logger.error("LLM client not available after init.")
        sys.exit(1)

    logger.info(f"LLM ready: {getattr(client._client, 'model_id', local_key)}")

    pm        = PromptManager(prompts_root=PROJECT_ROOT / "prompts")
    validator = OntologyValidator()
    results: list[dict] = []

    # ── Run each strategy ─────────────────────────────────────────────────────
    for strategy in args.strategies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running strategy: {strategy}")
        logger.info(f"{'='*50}")
        result = run_strategy(records, strategy, client, pm, validator)
        results.append(result)
        logger.info(
            f"  {strategy}: concepts={result['avg_concepts_per_record']}, "
            f"relations={result['avg_relations_per_record']}, "
            f"schema={result['schema_validity_rate']:.1%}, "
            f"parse={result['parse_success_rate']:.1%}, "
            f"pass={result['validation_pass_rate']:.1%}, "
            f"speed={result['avg_sec_per_record']:.1f}s/rec"
        )

    # ── Save reports ──────────────────────────────────────────────────────────
    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    save_report(results, report_dir)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 002c — PROMPT STRATEGY COMPARISON (Llama 3.1-8B)")
    print("=" * 70)
    print(f"{'Strategy':<22} {'Concepts':>8} {'Rels':>6} {'Conf':>6} {'Schema':>7} {'Pass%':>6} {'Parse%':>7} {'s/rec':>6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["schema_validity_rate"], reverse=True):
        print(
            f"  {r['strategy']:<20} {r['avg_concepts_per_record']:>8} "
            f"{r['avg_relations_per_record']:>6} {r['avg_confidence']:>6.3f} "
            f"{r['schema_validity_rate']:>7.1%} {r['validation_pass_rate']:>6.1%} "
            f"{r['parse_success_rate']:>7.1%} {r['avg_sec_per_record']:>6.1f}s"
        )
    print("=" * 70)
    best_strategy = max(results, key=lambda x: x["schema_validity_rate"] * x["parse_success_rate"])
    print(f"\nRecommended: --strategy {best_strategy['strategy']}")
    print(f"  → Re-run exp002 with this strategy to produce ontology v0.6.0")
    print("=" * 70)


if __name__ == "__main__":
    main()
