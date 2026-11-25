"""
Subgraph Ranker — re-ranks subgraph contexts by evidence quality.

Given a list of SubgraphContext objects (from SubgraphRetriever) and a query,
this module scores each subgraph and returns them in quality order.

Ranking criteria (weights sum to 1.0):
  - avg_confidence  (0.30): mean confidence of concepts/relations in subgraph
  - centrality      (0.25): fraction of high-degree concepts (authority nodes)
  - query_match     (0.25): token overlap between query and concept names
  - density         (0.15): relation-to-concept ratio (connectivity)
  - diversity       (0.05): number of distinct concept classes covered

Usage:
    ranker = SubgraphRanker(graph)
    ranked = ranker.rank(subgraphs, query="best beaches in Jeju")
    top    = ranked[0]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from .graph_store import TourismGraphStore
from ..utils.logging_utils import get_logger

logger = get_logger("graph.subgraph_ranker")

# ── Weights ───────────────────────────────────────────────────────────────────

_W_CONFIDENCE = 0.30
_W_CENTRALITY = 0.25
_W_QUERY      = 0.25
_W_DENSITY    = 0.15
_W_DIVERSITY  = 0.05

assert abs(_W_CONFIDENCE + _W_CENTRALITY + _W_QUERY + _W_DENSITY + _W_DIVERSITY - 1.0) < 1e-9


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RankedSubgraphResult:
    """A subgraph with its quality score breakdown."""
    subgraph:          dict                    # original subgraph dict from retriever
    total_score:       float = 0.0
    confidence_score:  float = 0.0
    centrality_score:  float = 0.0
    query_score:       float = 0.0
    density_score:     float = 0.0
    diversity_score:   float = 0.0
    n_concepts:        int   = 0
    n_relations:       int   = 0
    n_classes:         int   = 0
    top_concept_names: list  = field(default_factory=list)

    def score_breakdown(self) -> dict:
        return {
            "total":      round(self.total_score,      4),
            "confidence": round(self.confidence_score, 4),
            "centrality": round(self.centrality_score, 4),
            "query":      round(self.query_score,      4),
            "density":    round(self.density_score,    4),
            "diversity":  round(self.diversity_score,  4),
        }


# ── Ranker ────────────────────────────────────────────────────────────────────

class SubgraphRanker:
    """
    Re-ranks subgraph contexts by evidence quality relative to the query.

    Args:
        graph:             TourismGraphStore — used for in-degree lookup
        high_degree_pct:   concepts with degree >= this percentile are "authority" nodes
    """

    def __init__(
        self,
        graph: TourismGraphStore,
        high_degree_threshold: int = 5,
    ) -> None:
        self.graph = graph
        self._high_degree_threshold = high_degree_threshold

        # Pre-compute in-degree for every concept (used for centrality scoring)
        self._in_degree: dict[str, int] = {}
        for concept_name in graph._concepts:
            self._in_degree[concept_name] = len(graph._in_edges.get(concept_name, []))

    # ── Public API ────────────────────────────────────────────────────────────

    def rank(
        self,
        subgraphs: list[dict],
        query: str = "",
        top_k: Optional[int] = None,
    ) -> list[RankedSubgraphResult]:
        """
        Score and rank a list of subgraph dicts.

        Args:
            subgraphs: list of dicts with keys: concepts, relations, seeds, hops
            query:     user query string for relevance scoring
            top_k:     return only the top-k results (None = return all)

        Returns:
            List of RankedSubgraphResult sorted by total_score descending
        """
        if not subgraphs:
            return []

        query_tokens = self._tokenize(query)
        results = []
        for sg in subgraphs:
            result = self._score_subgraph(sg, query_tokens)
            results.append(result)

        results.sort(key=lambda r: r.total_score, reverse=True)
        if top_k is not None:
            results = results[:top_k]

        logger.debug(
            f"SubgraphRanker: ranked {len(results)} subgraphs | "
            f"top score={results[0].total_score:.3f}" if results else "no results"
        )
        return results

    def rank_single(self, subgraph: dict, query: str = "") -> RankedSubgraphResult:
        """Score a single subgraph dict."""
        return self._score_subgraph(subgraph, self._tokenize(query))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_subgraph(
        self,
        subgraph: dict,
        query_tokens: set[str],
    ) -> RankedSubgraphResult:
        concepts  = subgraph.get("concepts",  []) or []
        relations = subgraph.get("relations", []) or []

        n_concepts  = len(concepts)
        n_relations = len(relations)

        result = RankedSubgraphResult(
            subgraph=subgraph,
            n_concepts=n_concepts,
            n_relations=n_relations,
        )

        if n_concepts == 0:
            return result  # all scores = 0.0

        # 1. Confidence score — mean confidence of concepts + relations
        conf_vals = []
        for c in concepts:
            conf = c.get("provenance", {}).get("confidence", c.get("confidence", 0.0))
            conf_vals.append(float(conf))
        for r in relations:
            conf_vals.append(float(r.get("confidence", 0.0)))
        result.confidence_score = mean(conf_vals) if conf_vals else 0.0

        # 2. Centrality score — fraction of authority (high in-degree) concepts
        high_degree_count = sum(
            1 for c in concepts
            if self._in_degree.get(c.get("name", ""), 0) >= self._high_degree_threshold
        )
        result.centrality_score = high_degree_count / n_concepts

        # 3. Query match score — token overlap between query and concept names
        if query_tokens:
            overlap_scores = []
            for c in concepts:
                c_tokens = self._tokenize(c.get("name", ""))
                if c_tokens:
                    overlap = len(query_tokens & c_tokens) / len(query_tokens)
                    overlap_scores.append(overlap)
            result.query_score = mean(overlap_scores) if overlap_scores else 0.0
        else:
            result.query_score = 0.5  # neutral when no query

        # 4. Density score — relations per concept (capped at 1.0)
        raw_density = n_relations / n_concepts if n_concepts > 0 else 0.0
        result.density_score = min(raw_density / 3.0, 1.0)  # 3 rels/concept = max

        # 5. Diversity score — distinct concept classes (normalised to [0,1])
        classes = {c.get("concept_class", "") for c in concepts if c.get("concept_class")}
        result.n_classes = len(classes)
        result.diversity_score = min(len(classes) / 10.0, 1.0)  # 10 classes = max

        # Top concept names for display
        sorted_concepts = sorted(
            concepts,
            key=lambda c: float(c.get("provenance", {}).get("confidence", c.get("confidence", 0.0))),
            reverse=True,
        )
        result.top_concept_names = [c.get("name", "") for c in sorted_concepts[:5]]

        # Weighted total
        result.total_score = (
            _W_CONFIDENCE * result.confidence_score
            + _W_CENTRALITY * result.centrality_score
            + _W_QUERY      * result.query_score
            + _W_DENSITY    * result.density_score
            + _W_DIVERSITY  * result.diversity_score
        )

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = set(re.split(r"\W+", text.lower()))
        tokens.discard("")
        # Remove stopwords that don't contribute to tourism matching
        _STOPWORDS = {"the", "a", "an", "in", "of", "to", "for", "and", "or", "is", "are"}
        return tokens - _STOPWORDS

    def compare(
        self,
        subgraphs: list[dict],
        query: str = "",
    ) -> str:
        """Return a markdown table comparing subgraph scores."""
        results = self.rank(subgraphs, query)
        lines = [
            "| Rank | Concepts | Relations | Classes | Conf | Central | Query | Density | Total |",
            "|------|----------|-----------|---------|------|---------|-------|---------|-------|",
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"| {i} | {r.n_concepts} | {r.n_relations} | {r.n_classes} | "
                f"{r.confidence_score:.2f} | {r.centrality_score:.2f} | "
                f"{r.query_score:.2f} | {r.density_score:.2f} | **{r.total_score:.3f}** |"
            )
        return "\n".join(lines)
