"""
exp012 — LLM-as-Judge Evaluation (Week 10)

Uses real Llama 3.1-8B-Instruct as judge to score 200 responses
(50 per mode × 4 modes) on factual correctness, constraint satisfaction,
and fluency/coherence.

Outputs:
  artifacts/reports/llm_judge_scores.json
  artifacts/figures/llm_judge_comparison.md

Usage (real LLM, GPU 7):
  CUDA_VISIBLE_DEVICES=7 python -m experiments.exp012_llm_judge.run_llm_judge \\
    --use-llm --n-per-mode 50

Usage (offline/heuristic, for structure testing):
  python -m experiments.exp012_llm_judge.run_llm_judge --n-per-mode 50
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.llm_judge import LLMJudge, JudgeScore
from src.evaluation.benchmark import TourismBenchmark
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("exp012")

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SPLIT  = ROOT / "data" / "splits" / "test.jsonl"
ONT_PATH    = ROOT / "artifacts" / "ontologies" / "0.7.0" / "concepts.json"
OUT_SCORES  = ROOT / "artifacts" / "reports" / "llm_judge_scores.json"
OUT_MD      = ROOT / "artifacts" / "figures" / "llm_judge_comparison.md"

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]

# ── Offline answer templates (same as exp011, mode-differentiated) ────────────
_KNOWN_ENTITIES = [
    "Jeju Island", "Hallasan National Park", "Seongsan Ilchulbong",
    "Manjanggul Cave", "Jeju Black Pork", "Jeju Folk Village",
    "Dongmun Traditional Market", "Olle Trail", "Udo Island",
]
_FAKE_ENTITIES = [
    "Sunrise Valley Spa", "Dragon Peak Observatory",
    "Lava Rock Café", "Baram Beach Club", "Crystal Cave Zipline",
    "Tangerine Resort", "Seaside Yurt Village",
]

def _simulate_answer(mode: str, query: str) -> str:
    """Simulate answer per mode for offline testing."""
    import random as _rnd
    _rnd.seed(hash(query + mode) % (2**31))
    if mode == "no_rag":
        entities = _rnd.sample(_KNOWN_ENTITIES, 2) + _rnd.sample(_FAKE_ENTITIES, 2)
        return (
            f"Based on general knowledge: {', '.join(entities)}. "
            f"These are popular options for tourists visiting Jeju Island."
        )
    elif mode == "doc_rag":
        entities = _rnd.sample(_KNOWN_ENTITIES, 3) + _rnd.sample(_FAKE_ENTITIES, 1)
        return (
            f"According to retrieved tourism documents: {', '.join(entities)} "
            f"are well-reviewed destinations that match your query about {query[:60]}."
        )
    elif mode == "onto_rag":
        entities = _rnd.sample(_KNOWN_ENTITIES, 4)
        return (
            f"Based on verified ontology knowledge: {', '.join(entities)}. "
            f"These entries are confirmed tourism resources in the Jeju knowledge base."
        )
    else:  # constrained
        entities = _rnd.sample(_KNOWN_ENTITIES, 3)
        return (
            f"Using only verified ontology entries: {', '.join(entities)}. "
            f"All recommendations are strictly grounded in the tourism ontology v0.7.0. "
            f"Only confirmed entities are included."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="exp012 — LLM-as-Judge evaluation")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use real Llama 3.1-8B judge (default: offline heuristic)")
    parser.add_argument("--n-per-mode", type=int, default=50,
                        help="Queries to judge per mode (default 50 → 200 total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0],
                        help="Local GPU indices for LLM judge (after CUDA_VISIBLE_DEVICES remapping)")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load benchmark queries ────────────────────────────────────────────────
    logger.info("Loading ontology concepts …")
    ontology_concepts = load_json(ONT_PATH)

    logger.info(f"Building benchmark (n={args.n_per_mode} per mode) …")
    bench = TourismBenchmark.build(
        test_split_path=TEST_SPLIT,
        ontology_concepts=ontology_concepts,
        n=args.n_per_mode,
        seed=args.seed,
    )
    items = bench.items
    logger.info(f"Benchmark: {len(items)} items")

    # ── Load LLM judge (optional) ─────────────────────────────────────────────
    llm_client = None
    if args.use_llm:
        logger.info(f"Loading Llama 3.1-8B judge on GPU(s) {args.gpu_ids} …")
        from src.llm.local_client import LocalLLMClient
        import os
        llm_client = LocalLLMClient(
            model_key="llama3.1-8b",
            gpu_ids=args.gpu_ids,
            load_in_4bit=True,
            hf_token=os.environ.get("HF_TOKEN"),
        )
        llm_client.load()
        logger.info("LLM judge loaded")

    judge = LLMJudge(llm_client=llm_client, max_tokens=80)

    # ── System prompts per mode for answer generation ─────────────────────────
    _MODE_SYSTEM: dict[str, str] = {
        "no_rag": (
            "You are a helpful tourism assistant for Jeju Island, South Korea. "
            "Answer the user's question using your general knowledge."
        ),
        "doc_rag": (
            "You are a tourism assistant with access to Jeju Island travel documents. "
            "Provide a helpful, informative answer grounded in Jeju tourism knowledge."
        ),
        "onto_rag": (
            "You are a tourism assistant backed by a Jeju Island knowledge ontology. "
            "Base your answer on verified ontology concepts: attractions, restaurants, "
            "transportation, culture, and accommodation in Jeju."
        ),
        "constrained": (
            "You are a strictly factual tourism assistant for Jeju Island. "
            "RULES: 1) Only mention verified Jeju tourism entities. "
            "2) Do not fabricate names, prices, or statistics. "
            "3) If unsure, say so explicitly. "
            "4) Keep response under 150 words. "
            "5) Cite only confirmed Jeju attractions and services."
        ),
    }

    # ── Judge all modes ───────────────────────────────────────────────────────
    all_scores: list[JudgeScore] = []
    agg_by_mode = {}

    for mode in MODES:
        logger.info(f"Judging mode={mode} ({len(items)} items) …")

        # Generate answers with real LLM or simulate
        if args.use_llm and llm_client is not None:
            system_prompt = _MODE_SYSTEM.get(mode, _MODE_SYSTEM["no_rag"])
            answers = []
            for i, item in enumerate(items):
                if i % 10 == 0:
                    logger.info(f"  Generating {mode} answer {i+1}/{len(items)} …")
                ans = llm_client.call(
                    prompt=item.query,
                    system=system_prompt,
                    max_tokens=200,
                    temperature=0.0,
                )
                answers.append(ans)
        else:
            answers = [_simulate_answer(mode, item.query) for item in items]

        judge_items = [
            {
                "query_id": item.item_id,
                "query":    item.query,
                "answer":   answers[i],
                "context":  " | ".join(item.expected_concepts[:5]),
            }
            for i, item in enumerate(items)
        ]
        scores = judge.judge_batch(judge_items, mode, log_every=10)
        all_scores.extend(scores)
        agg_by_mode[mode] = judge.aggregate(scores, mode)

        agg = agg_by_mode[mode]
        logger.info(
            f"  {mode}: factual={agg.avg_factual_correctness:.2f} "
            f"constraint={agg.avg_constraint_satisfaction:.2f} "
            f"fluency={agg.avg_fluency_coherence:.2f} "
            f"overall={agg.avg_overall:.2f}"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    output = {
        "experiment":    "exp012_llm_judge",
        "use_llm":       args.use_llm,
        "n_per_mode":    args.n_per_mode,
        "n_total":       len(all_scores),
        "modes":         MODES,
        "elapsed_s":     round(elapsed, 1),
        "aggregates":    {m: agg_by_mode[m].to_dict() for m in MODES},
        "scores":        [s.to_dict() for s in all_scores],
    }
    ensure_dir(OUT_SCORES.parent)
    save_json(output, OUT_SCORES)
    logger.info(f"Judge scores saved → {OUT_SCORES}")

    # ── Markdown summary ──────────────────────────────────────────────────────
    md_lines = [
        "# exp012 — LLM-as-Judge Results",
        "",
        f"- Mode: {'real Llama 3.1-8B' if args.use_llm else 'offline heuristic'}",
        f"- N per mode: {args.n_per_mode} | Total scored: {len(all_scores)}",
        "",
        judge.compare_modes(agg_by_mode),
        "",
        "> Scale: 1 (poor) – 5 (excellent)",
        "> Factual correctness: verifiable claims vs. context",
        "> Constraint satisfaction: query requirements met",
        "> Fluency: response coherence and structure",
    ]
    md = "\n".join(md_lines)
    ensure_dir(OUT_MD.parent)
    OUT_MD.write_text(md)
    logger.info(f"Markdown saved → {OUT_MD}")

    print("\n" + md)
    print(f"\n[exp012] Done in {elapsed:.1f}s")
    print(f"  Scores: {OUT_SCORES}")


if __name__ == "__main__":
    main()
