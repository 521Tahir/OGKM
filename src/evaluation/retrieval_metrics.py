"""
Retrieval Quality Metrics — evaluate the RAG retrieval pipeline.

Metrics (addresses RQ4 from the paper):
  1. Context recall    — fraction of expected concepts retrieved in context
  2. Context precision — fraction of retrieved concepts relevant to query
  3. MRR              — Mean Reciprocal Rank of first relevant concept
  4. Hit@K            — fraction of queries where relevant concept in top-K
  5. Context richness — avg number of concepts/relations per retrieved context
  6. Retrieval latency — (optional) ms per query

Paper table: Table 4 — Retrieval Quality Metrics
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.retrieval_metrics")


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalItemMetrics:
    """Metrics for a single retrieval result."""
    query_id:          int
    query:             str
    mode:              str
    n_retrieved:       int         # total concepts in context
    n_relations_retrieved: int
    context_recall:    float       # fraction of expected concepts found (all retrieved)
    context_precision: float       # fraction of retrieved concepts that exist in ontology
    reciprocal_rank:   float       # 1/rank of first relevant concept (0 if none)
    hit_at_5:          bool        # relevant concept in top 5
    hit_at_10:         bool        # relevant concept in top 10
    # Proposal §10B additions
    recall_at_5:       float = 0.0  # fraction of relevant in top-5
    recall_at_10:      float = 0.0  # fraction of relevant in top-10
    precision_at_5:    float = 0.0  # fraction of top-5 that are relevant
    precision_at_10:   float = 0.0
    ndcg_at_10:        float = 0.0  # normalised discounted cumulative gain
    subgraph_relevance: float = 0.0  # fraction of retrieved concepts that are relevant
    evidence_coverage: float = 0.0  # fraction of relevant concepts supported by a retrieved relation

    def to_dict(self) -> dict:
        return {
            "query_id":            self.query_id,
            "mode":                self.mode,
            "n_retrieved":         self.n_retrieved,
            "n_relations":         self.n_relations_retrieved,
            "context_recall":      round(self.context_recall, 3),
            "context_precision":   round(self.context_precision, 3),
            "reciprocal_rank":     round(self.reciprocal_rank, 3),
            "hit_at_5":            self.hit_at_5,
            "hit_at_10":           self.hit_at_10,
            "recall_at_5":         round(self.recall_at_5, 3),
            "recall_at_10":        round(self.recall_at_10, 3),
            "precision_at_5":      round(self.precision_at_5, 3),
            "precision_at_10":     round(self.precision_at_10, 3),
            "ndcg_at_10":          round(self.ndcg_at_10, 3),
            "subgraph_relevance":  round(self.subgraph_relevance, 3),
            "evidence_coverage":   round(self.evidence_coverage, 3),
        }


@dataclass
class RetrievalAggregateMetrics:
    """Aggregate retrieval metrics across all queries for one mode."""
    mode:                str
    n_queries:           int
    avg_context_recall:  float
    avg_context_precision: float
    mrr:                 float   # mean reciprocal rank
    hit_rate_at_5:       float
    hit_rate_at_10:      float
    avg_retrieved:       float   # avg concepts per context
    avg_relations:       float
    # Proposal §10B additions
    recall_at_5:         float = 0.0
    recall_at_10:        float = 0.0
    precision_at_5:      float = 0.0
    precision_at_10:     float = 0.0
    ndcg_at_10:          float = 0.0
    subgraph_relevance:  float = 0.0
    evidence_coverage:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode":                   self.mode,
            "n_queries":              self.n_queries,
            "avg_context_recall":     round(self.avg_context_recall, 3),
            "avg_context_precision":  round(self.avg_context_precision, 3),
            "mrr":                    round(self.mrr, 3),
            "hit_rate_at_5":          round(self.hit_rate_at_5, 3),
            "hit_rate_at_10":         round(self.hit_rate_at_10, 3),
            "recall_at_5":            round(self.recall_at_5, 3),
            "recall_at_10":           round(self.recall_at_10, 3),
            "precision_at_5":         round(self.precision_at_5, 3),
            "precision_at_10":        round(self.precision_at_10, 3),
            "ndcg_at_10":             round(self.ndcg_at_10, 3),
            "subgraph_relevance":     round(self.subgraph_relevance, 3),
            "evidence_coverage":      round(self.evidence_coverage, 3),
            "avg_concepts_retrieved": round(self.avg_retrieved, 1),
            "avg_relations_retrieved": round(self.avg_relations, 1),
        }


# ── Main class ────────────────────────────────────────────────────────────────

class RetrievalMetricsCalculator:
    """
    Compute retrieval quality metrics given context and relevance labels.

    Parameters
    ----------
    ontology_names : set[str]
        Full set of ontology concept names (for precision computation).
    """

    def __init__(self, ontology_names: Optional[set[str]] = None):
        self._onto_lower = {n.lower() for n in ontology_names} if ontology_names else set()

    def score_item(
        self,
        query_id: int,
        query: str,
        mode: str,
        retrieved_concepts: list[dict],
        retrieved_relations: list[dict],
        relevant_concepts: list[str],   # ground-truth relevant concept names
    ) -> RetrievalItemMetrics:
        """
        Compute per-query retrieval metrics.

        Parameters
        ----------
        retrieved_concepts : list[dict]
            Ordered list of concepts returned by the retriever (rank 1 = first).
        relevant_concepts : list[str]
            Ground-truth relevant concept names for this query.
        """
        retrieved_names = [c.get("name", "") for c in retrieved_concepts]
        retrieved_lower = [n.lower() for n in retrieved_names]
        relevant_lower  = {r.lower() for r in relevant_concepts}

        n_retrieved = len(retrieved_concepts)

        # Context recall
        if relevant_lower:
            found = sum(1 for r in relevant_lower if r in retrieved_lower)
            context_recall = found / len(relevant_lower)
        else:
            context_recall = 1.0

        # Context precision (retrieved concepts that exist in ontology)
        if retrieved_lower and self._onto_lower:
            in_onto = sum(1 for r in retrieved_lower if r in self._onto_lower)
            context_precision = in_onto / len(retrieved_lower)
        else:
            context_precision = 1.0

        # MRR — rank of first relevant concept
        reciprocal_rank = 0.0
        for rank, name in enumerate(retrieved_lower, start=1):
            if name in relevant_lower:
                reciprocal_rank = 1.0 / rank
                break

        # Hit@K
        top5_lower  = set(retrieved_lower[:5])
        top10_lower = set(retrieved_lower[:10])
        hit_at_5  = bool(relevant_lower & top5_lower)
        hit_at_10 = bool(relevant_lower & top10_lower)

        # Recall@K / Precision@K
        n_rel = len(relevant_lower)
        recall_at_5   = (len(relevant_lower & top5_lower)  / n_rel) if n_rel else 0.0
        recall_at_10  = (len(relevant_lower & top10_lower) / n_rel) if n_rel else 0.0
        precision_at_5  = (len(relevant_lower & top5_lower)  / 5)  if retrieved_lower else 0.0
        precision_at_10 = (len(relevant_lower & top10_lower) / 10) if retrieved_lower else 0.0

        # nDCG@10 — binary gains over the retrieved ranking
        dcg = 0.0
        for rank, name in enumerate(retrieved_lower[:10], start=1):
            if name in relevant_lower:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(n_rel, 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcg_at_10 = (dcg / idcg) if idcg > 0 else 0.0

        # Subgraph relevance — fraction of retrieved concepts that are relevant
        retrieved_set = set(retrieved_lower)
        subgraph_relevance = (
            len(retrieved_set & relevant_lower) / len(retrieved_set)
        ) if retrieved_set else 0.0

        # Evidence coverage — fraction of relevant concepts that appear as an
        # endpoint of at least one retrieved relation (i.e. backed by evidence)
        rel_endpoints = set()
        for r in retrieved_relations:
            rel_endpoints.add(str(r.get("subject", r.get("head", ""))).lower())
            rel_endpoints.add(str(r.get("object", r.get("tail", ""))).lower())
        evidence_coverage = (
            len(relevant_lower & rel_endpoints) / n_rel
        ) if n_rel else 0.0

        return RetrievalItemMetrics(
            query_id=query_id,
            query=query,
            mode=mode,
            n_retrieved=n_retrieved,
            n_relations_retrieved=len(retrieved_relations),
            context_recall=context_recall,
            context_precision=context_precision,
            reciprocal_rank=reciprocal_rank,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            ndcg_at_10=ndcg_at_10,
            subgraph_relevance=subgraph_relevance,
            evidence_coverage=evidence_coverage,
        )

    def aggregate(
        self,
        items: list[RetrievalItemMetrics],
        mode: str,
    ) -> RetrievalAggregateMetrics:
        n = len(items)
        if n == 0:
            return RetrievalAggregateMetrics(
                mode=mode, n_queries=0,
                avg_context_recall=0, avg_context_precision=0,
                mrr=0, hit_rate_at_5=0, hit_rate_at_10=0,
                avg_retrieved=0, avg_relations=0,
            )

        def _mean(vals) -> float:
            return sum(vals) / n

        return RetrievalAggregateMetrics(
            mode=mode,
            n_queries=n,
            avg_context_recall=_mean([i.context_recall for i in items]),
            avg_context_precision=_mean([i.context_precision for i in items]),
            mrr=_mean([i.reciprocal_rank for i in items]),
            hit_rate_at_5=_mean([1.0 if i.hit_at_5 else 0.0 for i in items]),
            hit_rate_at_10=_mean([1.0 if i.hit_at_10 else 0.0 for i in items]),
            avg_retrieved=_mean([i.n_retrieved for i in items]),
            avg_relations=_mean([i.n_relations_retrieved for i in items]),
            recall_at_5=_mean([i.recall_at_5 for i in items]),
            recall_at_10=_mean([i.recall_at_10 for i in items]),
            precision_at_5=_mean([i.precision_at_5 for i in items]),
            precision_at_10=_mean([i.precision_at_10 for i in items]),
            ndcg_at_10=_mean([i.ndcg_at_10 for i in items]),
            subgraph_relevance=_mean([i.subgraph_relevance for i in items]),
            evidence_coverage=_mean([i.evidence_coverage for i in items]),
        )

    def compare_modes(
        self,
        mode_results: dict[str, RetrievalAggregateMetrics],
    ) -> str:
        """Return a markdown table comparing retrieval metrics across modes."""
        header = (
            "| Mode | Recall | Precision | MRR | Hit@5 | Hit@10 | Avg Concepts |\n"
            "|------|--------|-----------|-----|-------|--------|---------------|\n"
        )
        rows = []
        for mode, agg in mode_results.items():
            rows.append(
                f"| {mode} | {agg.avg_context_recall:.3f} | {agg.avg_context_precision:.3f} "
                f"| {agg.mrr:.3f} | {agg.hit_rate_at_5:.3f} | {agg.hit_rate_at_10:.3f} "
                f"| {agg.avg_retrieved:.1f} |"
            )
        return header + "\n".join(rows)
