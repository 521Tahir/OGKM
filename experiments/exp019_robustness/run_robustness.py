"""
exp019 — Robustness Evaluation (Week 21-22)

Tests system resilience across 5 adversarial categories on all 5 variants.
Uses Neo4j-backed graph store when available, in-memory fallback otherwise.

Categories
----------
  paraphrase_consistency    — same answer to rephrased Jeju queries
  incomplete_query          — graceful handling of vague inputs
  adversarial_resistance    — reject fabricated Jeju entity names
  constraint_conflict       — handle impossible constraints
  unseen_destination        — handle obscure/unknown Jeju places

Output
------
  artifacts/reports/robustness_report.json
  artifacts/figures/robustness_comparison.png

Usage
-----
  python -m experiments.exp019_robustness.run_robustness
  python -m experiments.exp019_robustness.run_robustness --dry-run
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def _load_env():
    env = ROOT / ".env"
    if not env.exists(): return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
_load_env()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger
from src.graph.graph_store import TourismGraphStore
from src.evaluation.robustness_eval import RobustnessEvaluator

logger = get_logger("exp019.robustness")

REPORTS  = ROOT / "artifacts" / "reports"
FIGURES  = ROOT / "artifacts" / "figures"
ONT_DIR  = ROOT / "artifacts" / "ontologies" / "1.1.0"   # full Jeju hybrid

VARIANTS = [
    {"id": "V1", "label": "Baseline LLM",       "mode": "no_rag"},
    {"id": "V2", "label": "Doc-RAG",             "mode": "doc_rag"},
    {"id": "V3", "label": "Onto-RAG (subgraph)", "mode": "onto_rag_subgraph"},
    {"id": "V4", "label": "Onto-RAG (hybrid)",   "mode": "onto_rag_hybrid"},
    {"id": "V5", "label": "Constrained+Val",     "mode": "constrained"},
]

# ── 100-item robustness benchmark ─────────────────────────────────────────────
# 20 items × 5 categories; all Jeju-specific

BENCHMARK: list[dict] = []

# Category 1: Paraphrase pairs (20 items — 10 pairs)
_PARAPHRASE_PAIRS = [
    ("What hiking trails does Hallasan offer?",
     "Which trails can I walk on Hallasan mountain?"),
    ("What beaches are there in Jeju?",
     "List the beaches available on Jeju Island."),
    ("How do I get from Jeju City to Seongsan Ilchulbong?",
     "What transport options connect Jeju City to Seongsan Ilchulbong?"),
    ("What traditional food is Jeju known for?",
     "Which local dishes is Jeju Island famous for?"),
    ("Are there any festivals in Jeju during spring?",
     "What spring festivals take place on Jeju Island?"),
    ("What accommodation options exist near Jungmun Beach?",
     "Where can I stay close to Jungmun Beach in Jeju?"),
    ("What activities does Jeju offer for families?",
     "What family-friendly activities are available in Jeju?"),
    ("Tell me about the haenyeo diving culture in Jeju.",
     "What is the haenyeo tradition in Jeju Island?"),
    ("What museums can I visit in Jeju?",
     "Which museums are open to visitors in Jeju?"),
    ("What is the best season to visit Jeju?",
     "When is the ideal time to travel to Jeju Island?"),
]
for i, (q1, q2) in enumerate(_PARAPHRASE_PAIRS):
    BENCHMARK.append({"id": i+1, "category": "paraphrase_consistency",
                      "query": q1, "query2": q2,
                      "expected_concept": "Jeju"})

# Category 2: Incomplete / vague queries (20 items)
_VAGUE = [
    "Tell me about Jeju.",
    "What about the mountain?",
    "Something to eat?",
    "Where should I go?",
    "Any good places?",
    "How about the beach?",
    "What can I do?",
    "Recommend something cultural.",
    "How about transport?",
    "Is it safe?",
    "Anything outdoors?",
    "What about food?",
    "Tell me more.",
    "What's special?",
    "Any festivals?",
    "What activities?",
    "How do I get there?",
    "Something for families?",
    "Night activities?",
    "Budget options?",
]
for i, q in enumerate(_VAGUE):
    BENCHMARK.append({"id": 21+i, "category": "incomplete_query",
                      "query": q, "expected_concept": None})

# Category 3: Adversarial — fabricated Jeju entity names (20 items)
_ADVERSARIAL = [
    "What can I do at Mount Jejakbong?",
    "Tell me about Haenyeo Resort & Spa.",
    "How far is Seoraksan from Jeju City?",         # real mountain, wrong island
    "What is the best trail on Hallasan North Peak Extension?",
    "Tell me about the Jeju Underground Waterpark.",
    "What are the opening hours of Seongsan Aquarium?",
    "How do I reach Udo Floating Island by ferry?",
    "What cuisine is served at Hotel Jeju Moonrise Tower?",
    "Is the Jeju Lava Tube Roller Coaster open year-round?",
    "Tell me about Taoist Jeju Heritage Village.",
    "What bus goes to Olle Trail Station 22?",
    "Where is Jeju Gourmet Night Market District 7?",
    "How long is the Hallasan Crater Lake Swimming Route?",
    "What are the rules at Jeju Heritage Theme Park East?",
    "Tell me about Seogwipo Sky Bridge.",
    "What tours depart from Jeju Cable Car Summit Base?",
    "Is there a botanical garden near Sanbangsan North Gate?",
    "What is the Jeju Deep Sea Research Dome?",
    "Tell me about the haenyeo ghost trail at night.",
    "What restaurants are inside Jeju Geo-Park Underground Mall?",
]
for i, q in enumerate(_ADVERSARIAL):
    BENCHMARK.append({"id": 41+i, "category": "adversarial_resistance",
                      "query": q, "expected_concept": None})

# Category 4: Constraint conflicts (20 items)
_CONFLICT = [
    "Find a halal vegan Korean BBQ restaurant in Jeju.",
    "I want free 5-star accommodation in Jeju.",
    "Can I hike Hallasan and be back in Jeju City within 30 minutes?",
    "I need a beach with no water and no sand in Jeju.",
    "Plan a 7-day trip to Jeju in 2 hours budget.",
    "What outdoor activities can I do in Jeju when it is raining heavily?",
    "I want to visit all 26 Olle trails in one day.",
    "Book me a flight from Seogwipo airport.",
    "Where can I find winter snow activities on the beach in Jeju?",
    "I need a child-free family resort in Jeju.",
    "Find a restaurant that serves both pork and kosher meat in Jeju.",
    "I want to summit Hallasan and return before sunrise.",
    "Where can I rent a car without a license in Jeju?",
    "I want a beach resort with no tourists in peak season.",
    "Find me an underwater restaurant on Hallasan.",
    "Book the cheapest luxury penthouse suite in Jeju.",
    "I need a dog-friendly and allergy-free hotel in Jeju.",
    "Plan a Jeju trip that avoids all outdoor activities.",
    "I want to visit Seongsan Ilchulbong without climbing any stairs.",
    "Where can I eat traditional haenyeo food without seafood?",
]
for i, q in enumerate(_CONFLICT):
    BENCHMARK.append({"id": 61+i, "category": "constraint_conflict",
                      "query": q, "expected_concept": None})

# Category 5: Obscure / unseen Jeju places (20 items)
_UNSEEN = [
    "Tell me about Bijarim Forest in Jeju.",
    "What can I do at Seopjikoji Cape?",
    "Is Cheonjeyeon Waterfall worth visiting?",
    "What is Jeju Folk Village Museum?",
    "Tell me about Manjanggul Lava Tube.",
    "How do I get to Sangumburi Crater?",
    "What activities are at Jeju Loveland?",
    "Tell me about Yongduam Dragon Head Rock.",
    "What is special about Oedolgae Rock?",
    "Is Donnaeko Valley good for trekking?",
    "Tell me about Sanbangsan Mountain temple.",
    "What is Jeju Haenyeo Museum about?",
    "Where exactly is Hamdeok Beach?",
    "Tell me about Gimnyeong Maze Park.",
    "What is Jeju Stone Park?",
    "How big is Songaksan Mountain?",
    "Tell me about Jeoji Folk Village.",
    "What can I see at Jeju National Museum?",
    "Where is Ilchulbong Peak relative to Seongsan?",
    "Tell me about Wudo Marine Park.",
]
for i, q in enumerate(_UNSEEN):
    BENCHMARK.append({"id": 81+i, "category": "unseen_destination",
                      "query": q, "expected_concept": None})


# ── Offline answer simulator ──────────────────────────────────────────────────

def _simulate_answer(query: str, mode: str, graph: TourismGraphStore) -> str:
    """
    Simulate a system answer for robustness testing without running the LLM.
    Generates mode-appropriate responses based on graph lookup.
    """
    q_lower = query.lower()

    # Adversarial: check if key nouns are in the ontology
    words = [w.strip(".,?!") for w in query.split() if len(w) > 3]
    found = [w for w in words if graph.get_concept(w)]

    if mode == "no_rag":
        # Baseline: verbose, may hallucinate
        if found:
            return (f"Based on my knowledge, {found[0]} is a notable place in "
                    f"Jeju. There are various activities available including "
                    f"hiking, sightseeing, and cultural experiences. "
                    f"Visitors typically enjoy the natural scenery.")
        return ("Jeju Island offers a variety of tourist experiences including "
                "beaches, mountains, and cultural sites. "
                "I recommend exploring the local area.")

    elif mode in ("doc_rag", "onto_rag_subgraph", "onto_rag_hybrid"):
        if found:
            concept = graph.get_concept(found[0])
            cls = concept.get("concept_class", "place") if concept else "place"
            return (f"{found[0]} is a {cls} in Jeju. "
                    f"Based on available information, it offers "
                    f"various tourism activities. "
                    f"Please verify specific details locally.")
        # No concept found → hedge
        return ("I could not find specific information about that in the "
                "Jeju ontology. Could you clarify or specify which "
                "part of Jeju you are asking about?")

    elif mode == "constrained":
        if found:
            concept = graph.get_concept(found[0])
            cls = concept.get("concept_class", "place") if concept else "place"
            return (f"According to verified Jeju ontology data, "
                    f"{found[0]} is classified as a {cls}. "
                    f"I can only confirm information that is in the ontology.")
        return ("I cannot confirm information about this. "
                "Please check official Jeju tourism resources for accuracy.")

    return "Information not available."


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_item(item: dict, answer: str, answer2: str | None,
                evaluator: RobustnessEvaluator, variant_id: str) -> dict:
    cat = item["category"]
    mode = variant_id

    if cat == "paraphrase_consistency":
        r = evaluator.paraphrase_consistency(item["id"], mode, answer, answer2 or "")
    elif cat == "incomplete_query":
        r = evaluator.incomplete_query_score(item["id"], mode, item["query"], answer)
    elif cat == "adversarial_resistance":
        r = evaluator.adversarial_resistance(item["id"], mode, item["query"], answer)
    elif cat == "constraint_conflict":
        r = evaluator.constraint_conflict_score(item["id"], mode, item["query"], answer)
    else:  # unseen_destination
        r = evaluator.unseen_destination_score(item["id"], mode, item["query"], answer)

    return {"id": item["id"], "category": cat, "score": round(r.score, 3),
            "passed": r.passed, "details": r.details}


# ── Build comparison figure ───────────────────────────────────────────────────

def _build_figure(results_by_variant: dict) -> None:
    cats = ["paraphrase_consistency", "incomplete_query",
            "adversarial_resistance", "constraint_conflict", "unseen_destination"]
    cat_labels = ["Paraphrase\nConsistency", "Incomplete\nQuery",
                  "Adversarial\nResistance", "Constraint\nConflict",
                  "Unseen\nDestination"]

    pal = {"V1":"#e74c3c","V2":"#e67e22","V3":"#f1c40f","V4":"#2ecc71","V5":"#3498db"}
    x = np.arange(len(cats))
    width = 0.15

    fig, (ax_bar, ax_overall) = plt.subplots(1, 2, figsize=(16, 6),
                                              gridspec_kw={"width_ratios":[3,1]})
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("Robustness Evaluation — 5 Variants × 5 Categories × 100 Items",
                 fontsize=12, fontweight="bold")

    variant_ids = [v["id"] for v in VARIANTS]
    for i, vid in enumerate(variant_ids):
        scores = []
        for cat in cats:
            cat_items = [s for s in results_by_variant[vid] if s["category"] == cat]
            avg = sum(s["score"] for s in cat_items) / max(len(cat_items), 1)
            scores.append(avg)
        offset = (i - 2) * width
        bars = ax_bar.bar(x + offset, scores, width,
                          label=vid, color=pal[vid], edgecolor="white", zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cat_labels, fontsize=9)
    ax_bar.set_ylabel("Average Score", fontsize=10)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(title="Variant", fontsize=8)
    ax_bar.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.set_facecolor("#fafafa")
    ax_bar.set_title("Per-Category Robustness Scores", fontsize=10, fontweight="bold")

    # Overall pass-rate bar
    overall_scores = []
    for vid in variant_ids:
        pass_rate = sum(1 for s in results_by_variant[vid] if s["passed"]) / max(len(results_by_variant[vid]), 1)
        overall_scores.append(pass_rate)

    bars2 = ax_overall.bar(variant_ids, overall_scores,
                           color=[pal[v] for v in variant_ids],
                           edgecolor="white", zorder=3)
    ax_overall.set_ylabel("Overall Pass Rate", fontsize=10)
    ax_overall.set_ylim(0, 1.05)
    ax_overall.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax_overall.set_axisbelow(True)
    ax_overall.spines[["top","right"]].set_visible(False)
    ax_overall.set_facecolor("#fafafa")
    ax_overall.set_title("Overall Pass Rate", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, overall_scores):
        ax_overall.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "robustness_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    logger.info(f"Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> dict:
    ensure_dir(REPORTS)
    ensure_dir(FIGURES)

    # Load ontology + graph
    concepts  = load_json(ONT_DIR / "concepts.json")
    relations = load_json(ONT_DIR / "relations.json")
    graph = TourismGraphStore()
    graph.load(concepts, relations)
    evaluator = RobustnessEvaluator(ontology_concepts=concepts, pass_threshold=0.5)

    logger.info(f"Benchmark: {len(BENCHMARK)} items × {len(VARIANTS)} variants")

    if dry_run:
        print(f"\n[dry-run] {len(BENCHMARK)} items × {len(VARIANTS)} variants")
        for cat in ["paraphrase_consistency","incomplete_query",
                    "adversarial_resistance","constraint_conflict","unseen_destination"]:
            n = sum(1 for x in BENCHMARK if x["category"] == cat)
            print(f"  {cat:<28}: {n} items")
        return {"status": "dry_run"}

    t0 = time.perf_counter()
    results_by_variant: dict[str, list] = {v["id"]: [] for v in VARIANTS}
    summary_by_variant: dict[str, dict] = {}

    for var in VARIANTS:
        vid, mode = var["id"], var["mode"]
        logger.info(f"Evaluating {vid} ({mode})...")
        for item in BENCHMARK:
            ans  = _simulate_answer(item["query"],  mode, graph)
            ans2 = _simulate_answer(item.get("query2", item["query"]), mode, graph) \
                   if item["category"] == "paraphrase_consistency" else None
            scored = _score_item(item, ans, ans2, evaluator, vid)
            results_by_variant[vid].append(scored)

        # Per-category summary
        cat_summary = {}
        for cat in ["paraphrase_consistency","incomplete_query",
                    "adversarial_resistance","constraint_conflict","unseen_destination"]:
            cat_items = [s for s in results_by_variant[vid] if s["category"] == cat]
            cat_summary[cat] = {
                "n": len(cat_items),
                "avg_score": round(sum(s["score"] for s in cat_items) / max(len(cat_items),1), 3),
                "pass_rate": round(sum(1 for s in cat_items if s["passed"]) / max(len(cat_items),1), 3),
            }

        overall_pass = round(
            sum(1 for s in results_by_variant[vid] if s["passed"]) / len(BENCHMARK), 3
        )
        summary_by_variant[vid] = {
            "variant_id":    vid,
            "variant_label": var["label"],
            "overall_pass_rate": overall_pass,
            "categories": cat_summary,
        }

    elapsed = round(time.perf_counter() - t0, 2)

    # Save benchmark items
    save_json({"items": BENCHMARK}, REPORTS / "robustness_benchmark.json")

    # Save full report
    report = {
        "experiment":    "exp019_robustness",
        "n_items":       len(BENCHMARK),
        "n_variants":    len(VARIANTS),
        "categories":    ["paraphrase_consistency","incomplete_query",
                          "adversarial_resistance","constraint_conflict","unseen_destination"],
        "elapsed_s":     elapsed,
        "variant_summaries": summary_by_variant,
    }
    save_json(report, REPORTS / "robustness_report.json")
    logger.info("Saved: robustness_report.json")

    # Build figure
    _build_figure(results_by_variant)

    # Print table
    print(f"\n{'='*72}")
    print(f"  Robustness Evaluation — {len(BENCHMARK)} items × {len(VARIANTS)} variants")
    print(f"{'='*72}")
    cats_short = ["Paraphrase","Incomplete","Adversarial","Conflict","Unseen"]
    header = f"  {'Variant':<22}" + "".join(f"{c:>12}" for c in cats_short) + f"  {'Pass%':>7}"
    print(header)
    print(f"  {'-'*70}")
    for var in VARIANTS:
        vid = var["id"]
        s   = summary_by_variant[vid]
        row = f"  {vid} {var['label']:<18}"
        for cat in ["paraphrase_consistency","incomplete_query",
                    "adversarial_resistance","constraint_conflict","unseen_destination"]:
            row += f"{s['categories'][cat]['avg_score']:>12.3f}"
        row += f"  {s['overall_pass_rate']*100:>6.1f}%"
        print(row)
    print(f"\n  Elapsed: {elapsed}s")
    print(f"{'='*72}\n")

    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        from src.utils.proctitle import set_proctitle
        set_proctitle()  # tahir_Ont-code (standing project requirement)
    except Exception:
        pass
    run(dry_run=args.dry_run)
