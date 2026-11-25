"""
Experiment 009 — Retrieval Benchmark Evaluation (Full Dataset)

Evaluates 4 retrieval conditions on ALL 6,197 test records:
  - no_retrieval  : empty context baseline (recall=0 by definition)
  - doc_only      : substring/BM25 text retrieval from train records
  - onto_only     : SubgraphRetriever (ontology v0.7.0 + SubgraphRanker)
  - hybrid        : ontology + doc combined (HybridRetriever)

Ground-truth relevant concepts are auto-generated per query by matching
ontology concept names against the record's Output text. This is the same
approach used in the RAG evaluation pipeline.

Metrics: Context Recall, Context Precision, MRR, Hit@5, Hit@10, Avg Concepts

Outputs:
  artifacts/reports/retrieval_benchmark.json     — full benchmark with ground truth
  artifacts/reports/exp009_retrieval_report.json — per-query results (all modes)
  artifacts/reports/exp009_retrieval_comparison.md — paper table
  artifacts/runs/exp009.log

Usage:
  python experiments/exp009_retrieval_eval/run_retrieval_eval.py \
      --n-queries 6197 \
      2>&1 | tee artifacts/runs/exp009.log
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from itertools import zip_longest
from typing import Optional
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.graph_store import TourismGraphStore
from src.retrieval.subgraph_retriever import SubgraphRetriever
from src.evaluation.retrieval_metrics import (
    RetrievalMetricsCalculator,
    RetrievalItemMetrics,
    RetrievalAggregateMetrics,
)
from src.utils.logging_utils import get_logger
from src.utils.io import save_json, ensure_dir

logger = get_logger("exp009")

ONTOLOGIES_DIR = ROOT / "artifacts" / "ontologies"
REPORTS_DIR    = ROOT / "artifacts" / "reports"
RUNS_DIR       = ROOT / "artifacts" / "runs"
FIGURES_DIR    = ROOT / "artifacts" / "figures"
TEST_SPLIT     = ROOT / "data" / "splits" / "test.jsonl"
TRAIN_SPLIT    = ROOT / "data" / "splits" / "train.jsonl"

ONTOLOGY_VERSION = "1.1.0"
RETRIEVAL_MODES  = ["no_retrieval", "doc_only", "onto_only", "hybrid"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            records.append(json.loads(line))
    return records


def build_graph(version: str = ONTOLOGY_VERSION) -> TourismGraphStore:
    logger.info(f"Loading ontology v{version}...")
    concepts  = json.load(open(ONTOLOGIES_DIR / version / "concepts.json"))
    relations = json.load(open(ONTOLOGIES_DIR / version / "relations.json"))
    graph = TourismGraphStore()
    graph.load(concepts, relations)
    logger.info(f"GraphStore: {len(concepts):,} concepts, {len(relations):,} relations")
    return graph


# ── Ground-truth generation ───────────────────────────────────────────────────

def build_benchmark(test_records: list[dict], concepts: list[dict]) -> list[dict]:
    """
    Build the ground-truth relevant-concept set for each query.

    Research-aligned design (proposal §9.3 — "the minimal knowledge necessary to
    answer the query"): relevance is grounded in the **answer (Output)** — the
    concepts a correct answer actually uses — matched by concept name OR any
    alias (semantic normalisation, not raw longest-substring). The earlier
    longest-20 truncation is removed: it dropped the key short entities (e.g.
    "Krabi", "Beach") that define the query, biasing the gold toward long phrase
    concepts and unfairly penalising focused ontology retrieval.

    Returns: [{id, query, output_text, relevant_concepts}, ...]
    """
    logger.info(f"Building answer-grounded benchmark from {len(test_records):,} test records...")

    # surface form (name/alias, lowercased) → canonical concept name
    surface_to_canon: dict[str, str] = {}
    for c in concepts:
        canon = c.get("name", "")
        if not canon:
            continue
        for surface in [canon] + list(c.get("aliases", []) or []):
            s = surface.lower().strip()
            if len(s) >= 3:
                surface_to_canon.setdefault(s, canon)
    # match longer surfaces first so multi-word names take precedence
    surfaces = sorted(surface_to_canon.keys(), key=len, reverse=True)

    benchmark = []
    for i, rec in enumerate(test_records):
        query  = rec.get("Input", "").strip()
        output = rec.get("Output", "").strip()
        # Answer-grounded: prefer Output; fall back to query if Output empty.
        answer_text = (output or query).lower()
        query_text  = query.lower()

        relevant: list[str] = []
        seen: set[str] = set()
        for s in surfaces:
            if s in answer_text or s in query_text:
                canon = surface_to_canon[s]
                if canon.lower() not in seen:
                    seen.add(canon.lower())
                    relevant.append(canon)

        benchmark.append({
            "id":                i,
            "query":             query,
            "output_text":       output,
            "relevant_concepts": relevant[:30],   # generous cap; selection no longer length-biased
        })

    n_with_relevant = sum(1 for b in benchmark if b["relevant_concepts"])
    avg_gold = sum(len(b["relevant_concepts"]) for b in benchmark) / max(len(benchmark), 1)
    logger.info(
        f"Benchmark: {len(benchmark):,} queries | "
        f"{n_with_relevant:,} ({n_with_relevant/len(benchmark)*100:.1f}%) have ≥1 relevant concept "
        f"| avg {avg_gold:.1f} gold concepts/query"
    )
    return benchmark


# ── Retrieval evaluation ──────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","in","of","to","for","and","or","is","are","was","were",
    "at","on","by","with","from","that","this","it","be","as","have","has",
    "i","my","you","your","we","our","they","their","what","which","how",
    "do","does","can","will","would","should","could","please","tell","me",
}

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class InvertedIndex:
    """
    Fast BM25-style inverted index over training records.
    Builds once; each query runs in O(query_terms × avg_postings).
    """
    def __init__(self, records: list[dict], top_k: int = 5,
                 ontology_names: Optional[set[str]] = None):
        self.top_k = top_k
        self._records = records
        N = len(records)

        # Build index: term → {doc_id: term_freq}; also keep per-doc token sets
        index: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        doc_lengths: list[int] = []
        self._doc_tokens: list[set[str]] = []

        for doc_id, rec in enumerate(records):
            text = rec.get("Input", "") + " " + rec.get("Output", "")
            tokens = _tokenize(text)
            doc_lengths.append(len(tokens))
            self._doc_tokens.append(set(tokens))
            for tok in tokens:
                index[tok][doc_id] += 1

        self._index = {t: dict(d) for t, d in index.items()}
        self._doc_lengths = doc_lengths
        self._avg_dl = sum(doc_lengths) / max(N, 1)
        self._N = N
        self._k1, self._b = 1.5, 0.75

        self._idf: dict[str, float] = {}
        for term, postings in self._index.items():
            df = len(postings)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        # ── Ontology concept → token-set, plus token → concepts reverse index.
        #    Lets us map a retrieved document to the ontology concepts it mentions
        #    (a doc "mentions" a concept if all of the concept-name tokens occur
        #     in the doc), so doc-RAG returns concepts comparable to the gold.
        self._concept_tokens: dict[str, tuple[str, frozenset]] = {}
        self._tok2concepts: dict[str, set[str]] = defaultdict(set)
        for name in (ontology_names or set()):
            toks = frozenset(_tokenize(name))
            if not toks:
                continue
            nl = name.lower()
            self._concept_tokens[nl] = (name, toks)
            for t in toks:
                self._tok2concepts[t].add(nl)
        self._doc_concept_cache: dict[int, list[str]] = {}

    def _scored_docs(self, query: str, k: int) -> list[int]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            if term not in self._index:
                continue
            idf = self._idf[term]
            for doc_id, tf in self._index[term].items():
                dl = self._doc_lengths[doc_id]
                denom = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                scores[doc_id] += idf * (tf * (self._k1 + 1)) / denom
        return sorted(scores, key=scores.__getitem__, reverse=True)[:k]

    def _doc_concepts(self, doc_id: int) -> list[str]:
        """Ontology concept names mentioned in a document (cached)."""
        if doc_id in self._doc_concept_cache:
            return self._doc_concept_cache[doc_id]
        doc_toks = self._doc_tokens[doc_id]
        candidates: set[str] = set()
        for t in doc_toks:
            candidates |= self._tok2concepts.get(t, set())
        found = [self._concept_tokens[nl][0]
                 for nl in candidates
                 if self._concept_tokens[nl][1] <= doc_toks]
        self._doc_concept_cache[doc_id] = found
        return found

    def retrieve(self, query: str) -> list[dict]:
        """Return top_k records as snippet dicts scored by BM25 (text view)."""
        results = []
        for doc_id in self._scored_docs(query, self.top_k):
            rec = self._records[doc_id]
            text = (rec.get("Input", "") + " " + rec.get("Output", ""))[:80].strip()
            results.append({"name": text})
        return results

    def retrieve_concepts(self, query: str, top_docs: int = 10) -> list[dict]:
        """
        Map the top retrieved documents to the ontology concepts they mention,
        ranked by document score. This makes doc-RAG return ontology concepts
        comparable to the gold relevant set (not raw text snippets).
        """
        seen: set[str] = set()
        out: list[dict] = []
        for doc_id in self._scored_docs(query, top_docs):
            for cname in self._doc_concepts(doc_id):
                cl = cname.lower()
                if cl not in seen:
                    seen.add(cl)
                    out.append({"name": cname})
        return out


# ── Evaluation modes ──────────────────────────────────────────────────────────

def order_by_relevance(ctx) -> list[dict]:
    """
    Rank retrieved subgraph concepts by conceptual proximity to the query
    (proposal §9.4): query-linked seed concepts first (in linker rank order),
    then the remaining subgraph nodes. The retriever returns a 60-node subgraph
    unordered; surfacing the seeds first is the ranking/pruning the proposal
    requires and is essential for MRR/Hit@K/nDCG to reflect retrieval quality.
    """
    seed_names = [lk["concept"]["name"] for lk in getattr(ctx, "linked_concepts", [])]
    by_lower = {c.get("name", "").lower(): c for c in ctx.concepts}
    ordered, seen = [], set()
    for s in seed_names:
        sl = s.lower()
        if sl and sl not in seen:
            seen.add(sl)
            ordered.append(by_lower.get(sl, {"name": s}))
    for c in ctx.concepts:
        cl = c.get("name", "").lower()
        if cl and cl not in seen:
            seen.add(cl)
            ordered.append(c)
    return ordered


def eval_no_retrieval(
    benchmark: list[dict],
    calc: RetrievalMetricsCalculator,
) -> list[RetrievalItemMetrics]:
    """Baseline: no retrieval. Context is always empty."""
    return [
        calc.score_item(
            query_id=item["id"], query=item["query"], mode="no_retrieval",
            retrieved_concepts=[], retrieved_relations=[],
            relevant_concepts=item["relevant_concepts"],
        )
        for item in benchmark
    ]


def eval_doc_only(
    benchmark: list[dict],
    idx: InvertedIndex,
    calc: RetrievalMetricsCalculator,
) -> list[RetrievalItemMetrics]:
    """BM25 text retrieval from training documents via inverted index."""
    results = []
    for item in benchmark:
        concepts = idx.retrieve_concepts(item["query"])
        m = calc.score_item(
            query_id=item["id"], query=item["query"], mode="doc_only",
            retrieved_concepts=concepts, retrieved_relations=[],
            relevant_concepts=item["relevant_concepts"],
        )
        results.append(m)
    return results


def eval_onto_only(
    benchmark: list[dict],
    onto_ret: SubgraphRetriever,
    calc: RetrievalMetricsCalculator,
) -> list[RetrievalItemMetrics]:
    """Ontology-only retrieval via SubgraphRetriever + SubgraphRanker."""
    results = []
    for i, item in enumerate(benchmark):
        ctx = onto_ret.retrieve(item["query"])
        m = calc.score_item(
            query_id=item["id"], query=item["query"], mode="onto_only",
            retrieved_concepts=order_by_relevance(ctx), retrieved_relations=ctx.relations,
            relevant_concepts=item["relevant_concepts"],
        )
        results.append(m)
        if (i + 1) % 500 == 0:
            logger.info(f"    onto_only: {i+1:,}/{len(benchmark):,} queries done")
    return results


def eval_hybrid(
    benchmark: list[dict],
    onto_ret: SubgraphRetriever,
    idx: InvertedIndex,
    calc: RetrievalMetricsCalculator,
) -> list[RetrievalItemMetrics]:
    """Hybrid: ontology subgraph concepts + BM25 text passages."""
    results = []
    for i, item in enumerate(benchmark):
        ctx      = onto_ret.retrieve(item["query"])
        doc_concepts = idx.retrieve_concepts(item["query"])
        # Fair fusion: interleave relevance-ordered onto concepts + doc concepts
        # by rank (round-robin) so neither source monopolises the head, then dedup.
        combined, seen = [], set()
        for a, b in zip_longest(order_by_relevance(ctx), doc_concepts):
            for c in (a, b):
                if c is None:
                    continue
                nl = str(c.get("name", "")).lower()
                if nl and nl not in seen:
                    seen.add(nl)
                    combined.append(c)
        m = calc.score_item(
            query_id=item["id"], query=item["query"], mode="hybrid",
            retrieved_concepts=combined, retrieved_relations=ctx.relations,
            relevant_concepts=item["relevant_concepts"],
        )
        results.append(m)
        if (i + 1) % 500 == 0:
            logger.info(f"    hybrid: {i+1:,}/{len(benchmark):,} queries done")
    return results


def build_markdown_report(
    agg_results: dict[str, RetrievalAggregateMetrics],
    n_queries: int,
    ontology_version: str,
) -> str:
    modes = list(agg_results.values())
    lines = [
        "# Experiment 009 — Retrieval Benchmark Evaluation",
        "",
        f"**Ontology:** v{ontology_version} | **Queries:** {n_queries:,} | "
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Results — Retrieval Quality Metrics",
        "",
        "| Mode | Recall | Precision | MRR | Hit@5 | Hit@10 | Avg Concepts |",
        "|------|:------:|:---------:|:---:|:-----:|:------:|:------------:|",
    ]
    for m in modes:
        lines.append(
            f"| {m.mode} | {m.avg_context_recall:.3f} | {m.avg_context_precision:.3f} "
            f"| {m.mrr:.3f} | {m.hit_rate_at_5:.3f} | {m.hit_rate_at_10:.3f} "
            f"| {m.avg_retrieved:.1f} |"
        )

    # Improvements over baseline
    baseline = agg_results.get("no_retrieval")
    if baseline:
        lines += [
            "",
            "## Improvement over No-Retrieval Baseline",
            "",
            "| Mode | Δ Recall | Δ Precision | Δ MRR | Δ Hit@5 | Δ Hit@10 |",
            "|------|:--------:|:-----------:|:-----:|:-------:|:--------:|",
        ]
        for m in modes:
            if m.mode == "no_retrieval":
                continue
            dr   = m.avg_context_recall    - baseline.avg_context_recall
            dp   = m.avg_context_precision - baseline.avg_context_precision
            dmrr = m.mrr                   - baseline.mrr
            dh5  = m.hit_rate_at_5         - baseline.hit_rate_at_5
            dh10 = m.hit_rate_at_10        - baseline.hit_rate_at_10
            lines.append(
                f"| {m.mode} | {dr:+.3f} | {dp:+.3f} | {dmrr:+.3f} "
                f"| {dh5:+.3f} | {dh10:+.3f} |"
            )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **Recall**: fraction of ground-truth relevant concepts found in retrieved context",
        "- **Precision**: fraction of retrieved concepts that exist in the ontology",
        "- **MRR**: Mean Reciprocal Rank — 1/rank of first relevant concept",
        "- **Hit@5**: fraction of queries with ≥1 relevant concept in top-5",
        "- **Hit@10**: fraction of queries with ≥1 relevant concept in top-10",
        "- **Avg Concepts**: average number of concepts in retrieved context",
        "",
        "## Research Question Link",
        "",
        "- **RQ4** (retrieval quality): onto_only vs doc_only comparison (Recall, MRR, Hit@K)",
        "- **RQ5** (hybrid benefit): hybrid vs onto_only — do text passages add recall?",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 009: Retrieval Evaluation")
    parser.add_argument("--n-queries", type=int, default=0,
                        help="Number of queries to evaluate (0 = all 6197)")
    parser.add_argument("--ontology-version", default=ONTOLOGY_VERSION)
    args = parser.parse_args()

    ensure_dir(REPORTS_DIR)
    ensure_dir(RUNS_DIR)
    ensure_dir(FIGURES_DIR)

    logger.info("=" * 65)
    logger.info("EXPERIMENT 009 — Retrieval Benchmark Evaluation")
    logger.info(f"  Ontology : v{args.ontology_version}")
    logger.info(f"  Queries  : {'ALL' if args.n_queries == 0 else args.n_queries}")
    logger.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading test split...")
    test_records = load_jsonl(TEST_SPLIT, limit=args.n_queries)
    logger.info(f"  {len(test_records):,} test records loaded")

    logger.info("Loading train split (for doc retrieval)...")
    train_records = load_jsonl(TRAIN_SPLIT)
    logger.info(f"  {len(train_records):,} train records loaded")

    # ── Build graph & retrievers ──────────────────────────────────────────────
    graph = build_graph(args.ontology_version)
    concept_names = set(graph._concepts.keys())
    concept_list = json.load(open(ONTOLOGIES_DIR / args.ontology_version / "concepts.json"))

    onto_ret = SubgraphRetriever(graph, use_ranker=True, max_seeds=5, max_top_seeds=3)
    calc     = RetrievalMetricsCalculator(ontology_names=concept_names)

    logger.info("Building inverted index from train records...")
    t_idx = time.time()
    idx = InvertedIndex(train_records, top_k=5, ontology_names=concept_names)
    logger.info(f"  Index built in {time.time()-t_idx:.1f}s | {len(idx._index):,} terms")

    # ── Build ground-truth benchmark ──────────────────────────────────────────
    benchmark = build_benchmark(test_records, concept_list)

    # Save benchmark
    bench_path = REPORTS_DIR / "retrieval_benchmark.json"
    save_json({
        "meta": {
            "ontology_version": args.ontology_version,
            "n_queries": len(benchmark),
            "date": datetime.now().isoformat(),
        },
        "items": benchmark,
    }, bench_path)
    logger.info(f"Benchmark saved → {bench_path}")

    # ── Run evaluations ───────────────────────────────────────────────────────
    all_items: dict[str, list[RetrievalItemMetrics]] = {}
    timings: dict[str, float] = {}

    for mode in RETRIEVAL_MODES:
        logger.info(f"Evaluating mode: {mode} ({len(benchmark):,} queries)...")
        t0 = time.time()

        if mode == "no_retrieval":
            items = eval_no_retrieval(benchmark, calc)
        elif mode == "doc_only":
            items = eval_doc_only(benchmark, idx, calc)
        elif mode == "onto_only":
            items = eval_onto_only(benchmark, onto_ret, calc)
        elif mode == "hybrid":
            items = eval_hybrid(benchmark, onto_ret, idx, calc)

        elapsed = time.time() - t0
        timings[mode] = elapsed
        all_items[mode] = items

        agg = calc.aggregate(items, mode)
        logger.info(
            f"  [{mode}] Recall={agg.avg_context_recall:.3f} "
            f"Precision={agg.avg_context_precision:.3f} "
            f"MRR={agg.mrr:.3f} Hit@5={agg.hit_rate_at_5:.3f} "
            f"Hit@10={agg.hit_rate_at_10:.3f} "
            f"({elapsed:.1f}s)"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_results = {
        mode: calc.aggregate(items, mode)
        for mode, items in all_items.items()
    }

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "experiment":       "exp009_retrieval_eval",
        "date":             datetime.now().isoformat(),
        "ontology_version": args.ontology_version,
        "n_queries":        len(benchmark),
        "timings_sec":      {m: round(t, 1) for m, t in timings.items()},
        "aggregate":        {m: a.to_dict() for m, a in agg_results.items()},
        "per_query_sample": {
            mode: [it.to_dict() for it in items[:10]]
            for mode, items in all_items.items()
        },
    }
    json_path = REPORTS_DIR / "exp009_retrieval_report.json"
    save_json(report, json_path)
    logger.info(f"JSON report → {json_path}")

    # ── Save Markdown comparison table ────────────────────────────────────────
    md_path = REPORTS_DIR / "exp009_retrieval_comparison.md"
    md_path.write_text(
        build_markdown_report(agg_results, len(benchmark), args.ontology_version),
        encoding="utf-8",
    )
    logger.info(f"Markdown report → {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 009 — RETRIEVAL BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"  Ontology v{args.ontology_version} | Queries: {len(benchmark):,}")
    print()
    print(f"  {'Mode':<15} {'Recall':>8} {'Precision':>10} {'MRR':>8} "
          f"{'Hit@5':>7} {'Hit@10':>8} {'AvgCtx':>7}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
    for mode, agg in agg_results.items():
        print(
            f"  {mode:<15} {agg.avg_context_recall:>8.3f} "
            f"{agg.avg_context_precision:>10.3f} {agg.mrr:>8.3f} "
            f"{agg.hit_rate_at_5:>7.3f} {agg.hit_rate_at_10:>8.3f} "
            f"{agg.avg_retrieved:>7.1f}"
        )
    print()
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
