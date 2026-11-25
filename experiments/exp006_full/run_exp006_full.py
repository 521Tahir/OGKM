"""
exp006_full — Full Benchmark Evaluation with Real LLM (6,197 test queries)

Runs all 4 RAG modes on the FULL test split (6,197 queries) using real
Llama 3.1-8B-Instruct. Parallelises across free GPUs in the 5–9 range:

  - Detects which of GPUs 5-9 are free (< 8 GB used)
  - Splits test queries equally across free GPUs
  - Each GPU runs all 4 modes on its shard (separate process)
  - Results are merged and aggregated into final report

Usage (from project root, after sourcing .env):
    set -a && source .env && set +a
    python -m experiments.exp006_full.run_exp006_full

Outputs:
    artifacts/reports/exp006_full_report.json
    artifacts/reports/exp006_full_results.jsonl
    artifacts/figures/exp006_full_comparison_table.md

Estimated runtime with N free GPUs (at ~20 tok/s, ~300 tok/answer):
    1 GPU  →  ~26 h    (6197 × 4 modes × 15 s / 3600)
    2 GPUs →  ~13 h
    3 GPUs →   ~9 h
    5 GPUs →   ~5 h
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_json, save_json, save_jsonl, ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp006_full")

ONTOLOGIES_DIR = PROJECT_ROOT / "artifacts" / "ontologies"
DATA_DIR       = PROJECT_ROOT / "data"
REPORTS_DIR    = PROJECT_ROOT / "artifacts" / "reports"
FIGURES_DIR    = PROJECT_ROOT / "artifacts" / "figures"
RUNS_DIR       = PROJECT_ROOT / "artifacts" / "runs"
SHARD_DIR      = REPORTS_DIR / "exp006_shards"

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]
CANDIDATE_GPUS = [5, 6, 7, 8, 9]   # GPUs assigned to this project
FREE_MEM_THRESHOLD_MB = 8_000       # GPU is "free" if < 8 GB used


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _query_gpu_mem() -> dict[int, int]:
    """Return {physical_gpu_id: used_MiB} for all GPUs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        result = {}
        for line in out.strip().splitlines():
            idx_str, mem_str = line.split(",")
            result[int(idx_str.strip())] = int(mem_str.strip())
        return result
    except Exception:
        return {}


def detect_free_gpus(candidate_gpus: list[int] = CANDIDATE_GPUS) -> list[int]:
    """Return subset of candidate_gpus that are currently free."""
    mem = _query_gpu_mem()
    free = [g for g in candidate_gpus if mem.get(g, 0) < FREE_MEM_THRESHOLD_MB]
    return free


def wait_for_free_gpus(
    n_needed: int,
    candidate_gpus: list[int] = CANDIDATE_GPUS,
    poll_interval: int = 60,
) -> list[int]:
    """Block until at least n_needed GPUs in candidate list are free."""
    while True:
        free = detect_free_gpus(candidate_gpus)
        if len(free) >= n_needed:
            return free[:n_needed]
        print(
            f"  Waiting for GPUs {candidate_gpus} — "
            f"currently free: {free} ({len(free)}/{n_needed} needed) "
            f"— retrying in {poll_interval}s …"
        )
        time.sleep(poll_interval)


# ── Shard worker (runs inside a subprocess on one GPU) ────────────────────────

SHARD_WORKER_CODE = '''
"""Worker script — runs on a single GPU, evaluates one shard of queries."""
import json, os, sys, time
from pathlib import Path

GPU_ID   = int(sys.argv[1])
SHARD_ID = int(sys.argv[2])
SHARD_FILE = Path(sys.argv[3])
OUT_FILE   = Path(sys.argv[4])
ONT_VER    = sys.argv[5]
MODES      = sys.argv[6].split(",")
ROOT       = Path(sys.argv[7])   # project root passed explicitly

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

sys.path.insert(0, str(ROOT))

from src.graph.graph_store import TourismGraphStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.constrained_generator import ConstrainedGenerator
from src.generation.explanation_builder import ExplanationBuilder
from src.evaluation.metrics import EvaluationMetrics
from src.utils.io import load_json
from src.llm.client import LLMClient

print(f"[GPU {GPU_ID} | Shard {SHARD_ID}] Loading ontology v{ONT_VER} ...")
ONTOLOGIES_DIR = ROOT / "artifacts" / "ontologies"
DATA_DIR       = ROOT / "data"

graph = TourismGraphStore.from_version(ONT_VER, ontology_dir=ONTOLOGIES_DIR)

test_path = DATA_DIR / "splits" / "test.jsonl"
test_records = []
if test_path.exists():
    with open(test_path) as f:
        test_records = [json.loads(l) for l in f if l.strip()]

retriever   = HybridRetriever(graph, records=test_records)
hf_token    = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
local_key   = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
client      = LLMClient(
    backend="local", model_key=local_key,
    load_in_4bit=True, hf_token=hf_token, max_tokens=512,
    gpu_ids=[0],   # remapped by CUDA_VISIBLE_DEVICES → physical GPU_ID
)
print(f"[GPU {GPU_ID}] LLM backend: {client.backend.upper() if client.is_available else 'OFFLINE'}")
generator   = ConstrainedGenerator(client if client.is_available else None, graph)
exp_builder = ExplanationBuilder(graph)
concept_names = {c["name"] for c in load_json(ONTOLOGIES_DIR / ONT_VER / "concepts.json")}
evaluator   = EvaluationMetrics(concept_names)

retrieval_mode_map = {
    "no_rag":      None,
    "doc_rag":     "text_only",
    "onto_rag":    "ontology_only",
    "constrained": "hybrid",
}

shard_items = json.loads(SHARD_FILE.read_text())
results = []
t0 = time.time()

for mode in MODES:
    rmode = retrieval_mode_map[mode]
    for i, item in enumerate(shard_items):
        query = item["query"]
        ctx = retriever.retrieve(query, mode=rmode) if rmode else None
        gen = generator.generate(query, ctx, mode=mode)

        # Build a minimal benchmark item for scoring
        class _Item:
            item_id = item["item_id"]
            expected_concepts = item.get("expected_concepts", [])
        trace = exp_builder.build(gen, ctx.subgraph if ctx else None)
        m = evaluator.score_item(gen, trace, _Item())

        results.append({
            "shard_id": SHARD_ID,
            "gpu_id":   GPU_ID,
            "mode":     mode,
            "item_id":  item["item_id"],
            "query":    query[:120],
            "answer":   gen.answer[:400],
            "metrics":  m.to_dict(),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(shard_items) - i - 1) / rate if rate > 0 else 0
            print(
                f"[GPU {GPU_ID} | {mode}] {i+1}/{len(shard_items)} "
                f"({rate:.1f} q/s, ~{remaining/60:.0f} min left)"
            )

OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"[GPU {GPU_ID} | Shard {SHARD_ID}] Done — {len(results)} results → {OUT_FILE}")
'''


# ── Main orchestrator ─────────────────────────────────────────────────────────

def _load_all_test_items(ontology_version: str) -> list[dict]:
    """Load all 6,197 test queries with expected concepts."""
    from src.evaluation.benchmark import TourismBenchmark
    from src.utils.io import load_json

    test_path = DATA_DIR / "splits" / "test.jsonl"
    concepts_path = ONTOLOGIES_DIR / ontology_version / "concepts.json"

    bench_path = REPORTS_DIR / f"benchmark_{ontology_version}_full.json"
    if bench_path.exists():
        bench = TourismBenchmark.load(bench_path)
    else:
        concepts = load_json(concepts_path)
        n_test = sum(1 for _ in open(test_path))
        bench = TourismBenchmark.build(test_path, concepts, n=n_test, seed=42)
        bench.save(bench_path)

    return [
        {
            "item_id": item.item_id,
            "query": item.query,
            "expected_concepts": item.expected_concepts if hasattr(item, "expected_concepts") else [],
        }
        for item in bench.items
    ]


def _shard_items(items: list[dict], n_shards: int) -> list[list[dict]]:
    shard_size = math.ceil(len(items) / n_shards)
    return [items[i * shard_size: (i + 1) * shard_size] for i in range(n_shards)]


def _merge_shard_results(shard_files: list[Path]) -> list[dict]:
    merged = []
    for f in shard_files:
        if f.exists():
            merged.extend(json.loads(f.read_text()))
    return merged


def _aggregate(all_results: list[dict], modes: list[str]) -> dict:
    agg = {}
    for mode in modes:
        items = [r for r in all_results if r["mode"] == mode]
        if not items:
            continue
        m_list = [r["metrics"] for r in items]
        agg[mode] = {
            "mode":               mode,
            "n_items":            len(items),
            "entity_recall":      round(mean(m["entity_recall"] for m in m_list), 3),
            "entity_precision":   round(mean(m["entity_precision"] for m in m_list), 3),
            "hallucination_rate": round(mean(m["hallucination_rate"] for m in m_list), 3),
            "avg_grounding":      round(mean(m.get("avg_grounding", m.get("grounding_score", 0)) for m in m_list), 3),
            "avg_answer_length":  round(mean(m.get("answer_length", m.get("avg_answer_length", 0)) for m in m_list), 1),
        }
    return agg


def _build_markdown(aggregates: dict, ontology_version: str, n_queries: int) -> str:
    lines = [
        "# Experiment 006 Full — Benchmark Evaluation (Real LLM)",
        f"\n**Ontology:** v{ontology_version} | **Queries:** {n_queries:,} | "
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | "
        f"**Model:** Llama 3.1-8B-Instruct (4-bit)",
        "\n| Mode | Entity Recall | Entity Precision | Halluc% | Grounding | Avg Len |",
        "|------|:-------------:|:----------------:|:-------:|:---------:|:-------:|",
    ]
    for mode in MODES:
        if mode not in aggregates:
            continue
        a = aggregates[mode]
        lines.append(
            f"| {mode:<12} | {a['entity_recall']:.1%} | {a['entity_precision']:.1%} | "
            f"{a['hallucination_rate']:.1%} | {a['avg_grounding']:.1%} | "
            f"{a['avg_answer_length']:.0f} |"
        )
    lines += [
        "\n## Research Questions",
        "- **RQ3** (onto-RAG vs doc-RAG): Entity recall + grounding columns",
        "- **RQ4** (constraint satisfaction): Halluc% — onto_rag vs constrained",
        "- **RQ5** (response quality): Grounding score column",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp006_full: Full 6197-query evaluation")
    parser.add_argument("--ontology-version", default="0.7.0")
    parser.add_argument("--n-queries", type=int, default=6197,
                        help="Number of test queries (default: full 6197)")
    parser.add_argument("--gpus", nargs="+", type=int, default=CANDIDATE_GPUS,
                        help="Candidate GPU IDs to use (default: 5 6 7 8 9)")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Force N GPUs even if more are free")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--wait-for-gpus", action="store_true",
                        help="Block until enough GPUs are free")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    ensure_dir(RUNS_DIR)
    setup_logging(
        log_level=args.log_level,
        log_file=str(RUNS_DIR / "exp006_full.log"),
    )
    ensure_dir(SHARD_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)

    print("=" * 70)
    print("exp006_full — Full Benchmark Evaluation (Real LLM)")
    print("=" * 70)
    print(f"  Ontology : v{args.ontology_version}")
    print(f"  Queries  : {args.n_queries:,}")
    print(f"  Modes    : {args.modes}")
    print(f"  GPU pool : {args.gpus}")

    # ── Detect free GPUs ──────────────────────────────────────────────────────
    if args.wait_for_gpus:
        n_needed = args.n_gpus or 1
        print(f"\nWaiting for ≥{n_needed} free GPUs in {args.gpus} ...")
        free_gpus = wait_for_free_gpus(n_needed, candidate_gpus=args.gpus)
    else:
        free_gpus = detect_free_gpus(candidate_gpus=args.gpus)
        if not free_gpus:
            print(
                f"\n⚠  No free GPUs detected in {args.gpus}.\n"
                f"   All GPUs in range are busy. Options:\n"
                f"   1. Run with --wait-for-gpus to block until a GPU frees up\n"
                f"   2. Wait for other jobs to finish (check: nvidia-smi)\n"
                f"   Current GPU memory usage:"
            )
            mem = _query_gpu_mem()
            for g in args.gpus:
                print(f"     GPU {g}: {mem.get(g, '?')} MiB used")
            sys.exit(1)

    if args.n_gpus:
        free_gpus = free_gpus[: args.n_gpus]

    print(f"\n  Using GPUs: {free_gpus} ({len(free_gpus)} workers)")

    # ── Load test items ───────────────────────────────────────────────────────
    print(f"\nLoading test items (n={args.n_queries:,}) ...")
    all_items = _load_all_test_items(args.ontology_version)[: args.n_queries]
    print(f"  Loaded {len(all_items):,} items")

    # ── Write shard worker script ─────────────────────────────────────────────
    worker_script = SHARD_DIR / "_shard_worker.py"
    worker_script.write_text(SHARD_WORKER_CODE)

    # ── Shard and dispatch ────────────────────────────────────────────────────
    shards = _shard_items(all_items, len(free_gpus))
    shard_out_files = []
    processes = []
    t_start = time.time()

    for shard_id, (gpu_id, shard) in enumerate(zip(free_gpus, shards)):
        shard_file = SHARD_DIR / f"shard_{shard_id:02d}_input.json"
        shard_file.write_text(json.dumps(shard, ensure_ascii=False))
        out_file = SHARD_DIR / f"shard_{shard_id:02d}_results.json"
        shard_out_files.append(out_file)

        cmd = [
            sys.executable,
            str(worker_script),
            str(gpu_id),
            str(shard_id),
            str(shard_file),
            str(out_file),
            args.ontology_version,
            ",".join(args.modes),
            str(PROJECT_ROOT),   # arg 7: explicit project root for worker
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTHONUNBUFFERED": "1"}
        log_file = RUNS_DIR / f"exp006_shard_{shard_id:02d}_gpu{gpu_id}.log"

        print(
            f"  Launching shard {shard_id} on GPU {gpu_id}: "
            f"{len(shard):,} items → log: {log_file.name}"
        )
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
        processes.append((shard_id, gpu_id, p, log_file))

    # ── Monitor ───────────────────────────────────────────────────────────────
    print(f"\n  {len(processes)} workers launched. Monitoring ...")
    done = set()
    while len(done) < len(processes):
        time.sleep(30)
        for shard_id, gpu_id, p, log_file in processes:
            if shard_id in done:
                continue
            ret = p.poll()
            if ret is not None:
                done.add(shard_id)
                status = "✓" if ret == 0 else f"✗ (rc={ret})"
                print(
                    f"  Shard {shard_id} (GPU {gpu_id}): {status} "
                    f"— elapsed {(time.time()-t_start)/60:.1f} min"
                )
            else:
                # Show last log line
                try:
                    with open(log_file) as lf:
                        lines = lf.read().splitlines()
                    last = next((l for l in reversed(lines) if l.strip()), "")
                    if last:
                        print(f"  [GPU {gpu_id}] {last[:100]}")
                except Exception:
                    pass

    elapsed = time.time() - t_start
    print(f"\n  All shards complete in {elapsed/3600:.2f} h")

    # ── Merge + aggregate ─────────────────────────────────────────────────────
    print("\nMerging shard results ...")
    all_results = _merge_shard_results(shard_out_files)
    print(f"  Total results: {len(all_results):,}")

    aggregates = _aggregate(all_results, args.modes)
    md_table = _build_markdown(aggregates, args.ontology_version, len(all_items))

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Full JSONL
    jsonl_path = REPORTS_DIR / "exp006_full_results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report JSON
    report = {
        "experiment":        "exp006_full",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "ontology_version":  args.ontology_version,
        "n_queries":         len(all_items),
        "modes":             args.modes,
        "gpus_used":         free_gpus,
        "elapsed_seconds":   round(elapsed),
        "aggregate_metrics": aggregates,
    }
    report_path = REPORTS_DIR / "exp006_full_report.json"
    save_json(report, report_path)

    # Markdown table
    table_path = FIGURES_DIR / "exp006_full_comparison_table.md"
    table_path.write_text(md_table)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("exp006_full — COMPLETE")
    print("=" * 70)
    print(md_table)
    print(f"\n  Report  : {report_path}")
    print(f"  Results : {jsonl_path}")
    print(f"  Table   : {table_path}")
    print(f"  Elapsed : {elapsed/3600:.2f} h")
    print("=" * 70)


if __name__ == "__main__":
    main()
