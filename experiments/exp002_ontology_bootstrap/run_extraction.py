"""
Experiment 002 — Ontology Bootstrap Extraction

Runs the full ontology extraction pipeline on the complete cleaned dataset.
Produces ontology v0.1.0 with concepts, relations, hierarchy, and rules.

Outputs:
  - artifacts/ontologies/0.1.0/
  - artifacts/ontologies/0.1.0/concepts.json
  - artifacts/ontologies/0.1.0/relations.json
  - artifacts/ontologies/0.1.0/rules.json
  - artifacts/ontologies/0.1.0/hierarchy.json
  - artifacts/ontologies/0.1.0/tourism_v0.1.0.ttl
  - artifacts/ontologies/0.1.0/tourism_v0.1.0.jsonld
  - artifacts/reports/ontology_v0_summary.json
"""
from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import load_jsonl, save_json, ensure_dir
from src.data.loaders import TourismDataLoader, TourismRecord
from src.ontology.ontology_builder import OntologyBuilder

logger = get_logger("exp002")


def parse_args():
    p = argparse.ArgumentParser(description="Exp002: Ontology Bootstrap")
    p.add_argument("--max-records", type=int, default=0,
                   help="Max records to process (0=all)")
    p.add_argument("--llm-records", type=int, default=1000,
                   help="Number of records to process with LLM; rest use offline extraction")
    p.add_argument("--strategy", default="cot",
                   choices=["naive", "cot", "metacognitive", "graph_of_thought"],
                   help="Prompting strategy (LLM mode only)")
    p.add_argument("--version", default="0.1.0", help="Ontology version string")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load local LLM in 4-bit quantization (faster, less VRAM)")
    p.add_argument("--gpus", default="6,7,8,9",
                   help="Comma-separated GPU IDs to use for local LLM (default: 6,7,8,9)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    ensure_dir("artifacts/runs")
    setup_logging(
        log_level=args.log_level,
        log_file="artifacts/runs/exp002_extraction.log",
    )
    ensure_dir("artifacts/reports")

    logger.info("=== Experiment 002: Ontology Bootstrap Extraction ===")
    logger.info(f"  Strategy    : {args.strategy}")
    logger.info(f"  Version     : {args.version}")
    logger.info(f"  Max recs    : {args.max_records or 'ALL'}")
    logger.info(f"  LLM records : {args.llm_records}")
    logger.info(f"  4-bit quant : {args.load_in_4bit}")

    # ── Load data ─────────────────────────────────────────────────────────────
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
        for r in records_raw
    ]
    logger.info(f"Loaded {len(records):,} records from {data_path}")

    # ── Build ontology ────────────────────────────────────────────────────────
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    logger.info(f"  GPU IDs     : {gpu_ids}")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    client = LLMClient(
        backend="local" if hf_token else "auto",
        model_key=local_key,
        load_in_4bit=args.load_in_4bit,
        gpu_ids=gpu_ids,
        hf_token=hf_token,
    )
    pm = PromptManager(prompts_root=PROJECT_ROOT / "prompts")

    if client.is_available:
        logger.info(f"LLM backend: {client.backend.upper()}")
    else:
        logger.info("No LLM backend — running fully offline extraction")

    total_records = args.max_records or len(records)
    llm_n = min(args.llm_records, total_records) if client.is_available else 0
    logger.info(
        f"Hybrid mode: {llm_n:,} LLM records + {total_records - llm_n:,} offline records"
    )

    builder = OntologyBuilder(
        client=client if client.is_available else None,
        prompt_manager=pm if client.is_available else None,
        concept_strategy=args.strategy,
        relation_strategy=args.strategy,
        output_dir=PROJECT_ROOT / "artifacts" / "ontologies",
        checkpoint_dir=PROJECT_ROOT / "artifacts" / "runs" / "checkpoints",
        version=args.version,
    )

    summary = builder.process_records(
        records,
        max_records=total_records,
        save_every_n=args.save_every,
        llm_max_records=llm_n if client.is_available else None,
    )
    summary["llm_records"] = llm_n
    summary["offline_records"] = total_records - llm_n

    paths = builder.save_ontology()

    # ── Save run report ───────────────────────────────────────────────────────
    report = {**summary, **paths, "args": vars(args)}
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "ontology_v0_summary.json"
    save_json(report, report_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 002 — ONTOLOGY BOOTSTRAP COMPLETE")
    print("=" * 60)
    print(f"  Extraction mode   : {summary['extraction_mode'].upper()}")
    print(f"  Records processed : {summary['total_records']:,}")
    print(f"  Concepts          : {summary['total_concepts']:,}")
    print(f"  Relations         : {summary['total_relations']:,}")
    print(f"  Hierarchy nodes   : {summary['total_hierarchy_nodes']:,}")
    print(f"  Rules             : {summary['total_rules']:,}")
    print(f"\n  Validation:")
    for k, v in summary["validation"].items():
        print(f"    {k:10s}: {v:,}")
    print(f"\n  Ontology v{args.version} saved:")
    print(f"    Turtle  : {paths.get('turtle')}")
    print(f"    JSON-LD : {paths.get('jsonld')}")
    print(f"    Report  : {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
