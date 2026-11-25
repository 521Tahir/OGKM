"""
Reranker — re-scores and prunes retrieved subgraph nodes before context assembly.

Scoring factors (all normalised to [0, 1]):
  1. Confidence   — relation/concept extraction confidence from exp002
  2. Graph dist   — BFS distance from seed nodes (closer = higher score)
  3. Authority    — in-degree (well-connected concepts rank higher)
  4. Query match  — token overlap between concept name/description and query keywords
  5. Theme boost  — concepts matching detected themes get a bonus

Final score = weighted sum of above; weights tunable via constructor.

Used by: constrained_generator.py, run_rag.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .query_understanding import QueryIntent
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.reranker")


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class ScoredConcept:
    concept: dict
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredRelation:
    relation: dict
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RankedSubgraph:
    """Reranked subgraph ready for context assembly."""
    query: str
    concepts: list[ScoredConcept]
    relations: list[ScoredRelation]
    seed_concepts: list[str]       # concept names used as BFS seeds

    def top_concepts(self, n: int = 20) -> list[dict]:
        return [sc.concept for sc in self.concepts[:n]]

    def top_relations(self, n: int = 30) -> list[dict]:
        return [sr.relation for sr in self.relations[:n]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token_overlap(text: str, keywords: list[str]) -> float:
    """Jaccard-style token overlap between text tokens and query keywords."""
    if not keywords or not text:
        return 0.0
    text_tokens = set(text.lower().split())
    kw_set = set(k.lower() for k in keywords)
    intersect = text_tokens & kw_set
    union = text_tokens | kw_set
    return len(intersect) / len(union) if union else 0.0


def _sigmoid_norm(x: float, scale: float = 5.0) -> float:
    """Map count-like values to [0, 1] via sigmoid."""
    return 1.0 / (1.0 + math.exp(-x / scale))


# ── Theme → class/predicate mapping ──────────────────────────────────────────

_THEME_CLASSES: dict[str, set[str]] = {
    "nature":    {"Beach", "Mountain", "NaturalSite", "NationalPark", "Island", "Trail"},
    "culture":   {"Museum", "CulturalSite", "Festival", "Heritage", "TouristAttraction"},
    "food":      {"Restaurant", "LocalFood", "Cuisine", "Cafe"},
    "adventure": {"Activity", "Sport", "AdventureSport", "WaterSport"},
    "history":   {"Monument", "HistoricSite", "Heritage", "CulturalSite"},
    "shopping":  {"ShoppingCenter", "Market", "DutyFree"},
    "wellness":  {"Spa", "HotSpring", "WellnessCenter"},
}


# ── Main class ────────────────────────────────────────────────────────────────

class Reranker:
    """
    Re-score concepts and relations from a subgraph retrieval result.

    Parameters
    ----------
    w_confidence : float   Weight for extraction confidence score
    w_dist       : float   Weight for inverse graph distance
    w_authority  : float   Weight for in-degree authority
    w_query      : float   Weight for query token overlap
    w_theme      : float   Weight for theme class bonus
    """

    def __init__(
        self,
        w_confidence: float = 0.25,
        w_dist:       float = 0.25,
        w_authority:  float = 0.15,
        w_query:      float = 0.25,
        w_theme:      float = 0.10,
    ):
        total = w_confidence + w_dist + w_authority + w_query + w_theme
        self.w_conf  = w_confidence / total
        self.w_dist  = w_dist       / total
        self.w_auth  = w_authority  / total
        self.w_query = w_query      / total
        self.w_theme = w_theme      / total

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        subgraph_context: Any,            # SubgraphContext
        query_intent: Optional[QueryIntent] = None,
        graph_store: Optional[Any] = None, # TourismGraphStore (for in-degree lookup)
    ) -> RankedSubgraph:
        """
        Rerank the concepts and relations in subgraph_context.

        Returns a RankedSubgraph with concepts and relations sorted by score (desc).
        """
        concepts  = list(getattr(subgraph_context, "concepts", []) or [])
        relations = list(getattr(subgraph_context, "relations", []) or [])
        seeds     = list(getattr(subgraph_context, "seeds", []) or [])
        query     = getattr(subgraph_context, "query", "")
        keywords  = query_intent.keywords if query_intent else query.lower().split()
        themes    = query_intent.constraints.themes if query_intent else []

        # Build distance map from seed concept names
        dist_map = self._compute_distances(concepts, seeds)
        # Build authority (in-degree) from relations
        indegree = self._compute_indegree(relations)
        # Max indegree for normalization
        max_deg = max(indegree.values(), default=1)

        scored_concepts: list[ScoredConcept] = []
        for c in concepts:
            sc = self._score_concept(c, keywords, themes, dist_map, indegree, max_deg)
            scored_concepts.append(sc)
        scored_concepts.sort(key=lambda x: x.score, reverse=True)

        scored_relations: list[ScoredRelation] = []
        for r in relations:
            sr = self._score_relation(r, keywords, scored_concepts)
            scored_relations.append(sr)
        scored_relations.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "Reranked %d concepts, %d relations (seeds=%s, themes=%s)",
            len(scored_concepts), len(scored_relations), seeds, themes,
        )
        return RankedSubgraph(
            query=query,
            concepts=scored_concepts,
            relations=scored_relations,
            seed_concepts=seeds,
        )

    def rerank_from_lists(
        self,
        query: str,
        concepts: list[dict],
        relations: list[dict],
        seeds: list[str],
        query_intent: Optional[QueryIntent] = None,
    ) -> RankedSubgraph:
        """Convenience wrapper that takes plain lists instead of SubgraphContext."""

        class _FakeCtx:
            pass

        ctx = _FakeCtx()
        ctx.concepts  = concepts   # type: ignore[attr-defined]
        ctx.relations = relations  # type: ignore[attr-defined]
        ctx.seeds     = seeds      # type: ignore[attr-defined]
        ctx.query     = query      # type: ignore[attr-defined]
        return self.rerank(ctx, query_intent)

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_concept(
        self,
        c: dict,
        keywords: list[str],
        themes: list[str],
        dist_map: dict[str, int],
        indegree: dict[str, int],
        max_deg: int,
    ) -> ScoredConcept:
        name = c.get("name", "")
        cls  = c.get("class", "")
        desc = c.get("description", "")
        conf = float(c.get("confidence", 0.8))

        # Confidence (already 0-1 from extractor)
        s_conf = max(0.0, min(1.0, conf))

        # Inverse distance (seeds have dist=0 → score=1, each step halves)
        dist = dist_map.get(name, 5)
        s_dist = 1.0 / (1.0 + dist)

        # Authority (in-degree normalized)
        deg = indegree.get(name, 0)
        s_auth = deg / max_deg if max_deg > 0 else 0.0

        # Query token overlap on name + description
        search_text = f"{name} {desc}"
        s_query = _token_overlap(search_text, keywords)

        # Theme class bonus
        s_theme = 0.0
        for theme in themes:
            theme_cls_set = _THEME_CLASSES.get(theme, set())
            if cls in theme_cls_set:
                s_theme = 1.0
                break

        score = (
            self.w_conf  * s_conf
            + self.w_dist  * s_dist
            + self.w_auth  * s_auth
            + self.w_query * s_query
            + self.w_theme * s_theme
        )

        return ScoredConcept(
            concept=c,
            score=score,
            breakdown={
                "confidence": s_conf,
                "dist": s_dist,
                "authority": s_auth,
                "query_match": s_query,
                "theme": s_theme,
            },
        )

    def _score_relation(
        self,
        r: dict,
        keywords: list[str],
        scored_concepts: list[ScoredConcept],
    ) -> ScoredRelation:
        """
        Score a relation based on the scores of its subject/object concepts
        and the predicate's relevance to the query keywords.
        """
        subj = r.get("subject", r.get("head", ""))
        obj  = r.get("object",  r.get("tail", ""))
        pred = r.get("predicate", r.get("relation", ""))
        conf = float(r.get("confidence", 0.7))

        # Average score of endpoint concepts
        concept_score_map = {sc.concept.get("name", ""): sc.score for sc in scored_concepts}
        s_endpoints = (concept_score_map.get(subj, 0.3) + concept_score_map.get(obj, 0.3)) / 2.0

        # Predicate overlap with keywords
        s_pred = _token_overlap(pred.replace("_", " "), keywords)

        # Confidence
        s_conf = max(0.0, min(1.0, conf))

        score = 0.4 * s_endpoints + 0.3 * s_pred + 0.3 * s_conf

        return ScoredRelation(
            relation=r,
            score=score,
            breakdown={"endpoints": s_endpoints, "pred_match": s_pred, "confidence": s_conf},
        )

    # ── Graph utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_distances(concepts: list[dict], seeds: list[str]) -> dict[str, int]:
        """Assign BFS distance from seed nodes (approximate, using name matching)."""
        seed_set = {s.lower() for s in seeds}
        dist: dict[str, int] = {}
        for c in concepts:
            name = c.get("name", "")
            if name.lower() in seed_set:
                dist[name] = 0
            else:
                dist[name] = 3  # default if no BFS available
        return dist

    @staticmethod
    def _compute_indegree(relations: list[dict]) -> dict[str, int]:
        """Count in-degree for each concept appearing as object in relations."""
        deg: dict[str, int] = {}
        for r in relations:
            obj = r.get("object", r.get("tail", ""))
            if obj:
                deg[obj] = deg.get(obj, 0) + 1
        return deg
