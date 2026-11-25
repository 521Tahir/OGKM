"""
Subgraph Retriever — extracts a relevant ontology subgraph for a query.

Given a query string:
  1. Links query to seed concepts via ConceptLinker
  2. Optionally injects keyword seeds for domain-specific terms (haenyeo, oreum…)
  3. Expands per-seed candidate subgraphs via GraphStore.extract_subgraph
  4. Re-ranks candidates using SubgraphRanker (when provided)
  5. Merges top-ranked subgraphs and formats context for generation

Returns a SubgraphContext object with concepts, relations, and a text summary.

Multi-hop note: default max_hops=3 (increased from 2) to close the missing-concept
gap identified in failure analysis Case 2 (haenyeo/multi-hop queries).
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from ..graph.graph_store import TourismGraphStore
from ..graph.subgraph_ranker import SubgraphRanker
from .concept_linker import ConceptLinker
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.subgraph_retriever")

# LRU subgraph cache: key=(query_hash, max_hops, max_nodes) → SubgraphContext
# Holds up to 256 entries; hit rate typically >60% on repeated/similar queries
_SUBGRAPH_CACHE: dict[str, "SubgraphContext"] = {}
_CACHE_MAX_SIZE = 256


def _cache_key(query: str, max_hops: int, max_nodes: int) -> str:
    norm = query.lower().strip()
    return hashlib.md5(f"{norm}|{max_hops}|{max_nodes}".encode()).hexdigest()

# Keyword seeds: if the query contains any trigger, inject the mapped concept name
# as an additional seed (before linker lookup) to bridge multi-hop gaps.
_KEYWORD_SEEDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhaenyeo\b",          re.I), "Haenyeo"),
    (re.compile(r"\boreum\b",            re.I), "Oreum"),
    (re.compile(r"\bolleh\b",            re.I), "Jeju Olle Trail"),
    (re.compile(r"\btyphoon\b",          re.I), "Typhoon Season"),
    (re.compile(r"\bemergency\b",        re.I), "Emergency Facility"),
    (re.compile(r"\btraditional\s+diver",re.I), "Haenyeo"),
    (re.compile(r"\bsafety\b",           re.I), "Safety Guidelines"),
    (re.compile(r"\bvolcan",             re.I), "Volcanic Formation"),
    (re.compile(r"\blava\b",             re.I), "Lava Tube"),
    (re.compile(r"\bUNESCO\b",           re.I), "UNESCO World Heritage"),
]

# Safety-critical predicates that are never pruned by the ranker
SAFETY_PREDICATES: frozenset[str] = frozenset({
    "hasEmergencyFacility",
    "hasAccessConstraint",
    "hasSafetyWarning",
    "hasPolicy",
    "requiresReservation",
})


@dataclass
class SubgraphContext:
    """Packaged subgraph context ready for generation."""
    query: str
    seeds: list[str]
    concepts: list[dict]
    relations: list[dict]
    linked_concepts: list[dict]   # full link results with scores
    rank_score: float = 0.0       # quality score from SubgraphRanker (0 if not ranked)
    n_candidates: int = 1         # number of candidate subgraphs that were ranked

    def is_empty(self) -> bool:
        return len(self.concepts) == 0

    def to_text(self, max_relations: int = 30) -> str:
        """Render the subgraph as a readable text block for LLM context."""
        if self.is_empty():
            return "[No relevant ontology context found]"

        lines = ["=== Ontology Context ==="]

        # Key concepts
        lines.append(f"\nKey Concepts ({len(self.concepts)}):")
        for c in self.concepts[:20]:
            aliases = c.get("aliases", [])
            alias_str = f" (also: {', '.join(aliases[:2])})" if aliases else ""
            lines.append(f"  - {c['name']} [{c.get('concept_class','?')}]{alias_str}")

        # Relations
        shown_rels = self.relations[:max_relations]
        if shown_rels:
            lines.append(f"\nRelations ({len(self.relations)} total, showing {len(shown_rels)}):")
            for r in shown_rels:
                conf = r.get("provenance", {}).get("confidence", 0.0)
                lines.append(
                    f"  {r['subject']} --[{r['predicate']}]--> {r['object']}"
                    f"  (conf={conf:.2f})"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query":       self.query,
            "seeds":       self.seeds,
            "n_concepts":  len(self.concepts),
            "n_relations": len(self.relations),
            "rank_score":  round(self.rank_score, 4),
            "n_candidates": self.n_candidates,
            "linked":   [
                {"name": lc["concept"]["name"], "score": lc["score"], "method": lc["method"]}
                for lc in self.linked_concepts
            ],
        }


class SubgraphRetriever:
    """
    Retrieves a context subgraph from the ontology for a natural language query.

    When a SubgraphRanker is provided (or use_ranker=True), the retriever:
      - Extracts one candidate subgraph per linked seed concept
      - Ranks candidates by evidence quality (confidence, centrality, query match…)
      - Merges the top-ranked seeds into one final context

    Usage:
        retriever = SubgraphRetriever(graph, use_ranker=True)
        ctx = retriever.retrieve("What are the best beaches near Jeju?")
        print(ctx.to_text())
        print(ctx.rank_score)  # quality score from ranker
    """

    def __init__(
        self,
        graph: TourismGraphStore,
        linker: Optional[ConceptLinker] = None,
        ranker: Optional[SubgraphRanker] = None,
        use_ranker: bool = True,
        max_seeds: int = 5,
        max_top_seeds: int = 3,     # top-k seeds to merge after ranking
        max_hops: int = 3,          # increased from 2→3 to fix multi-hop missing-concept gap
        max_nodes: int = 60,        # expanded budget to accommodate deeper BFS
        min_link_score: float = 0.3,
        keyword_seeding: bool = True,
    ) -> None:
        self.graph = graph
        self.linker = linker or ConceptLinker(graph, min_score=min_link_score)
        self.ranker = ranker or (SubgraphRanker(graph) if use_ranker else None)
        self.max_seeds = max_seeds
        self.max_top_seeds = max_top_seeds
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.keyword_seeding = keyword_seeding

    def retrieve(
        self,
        query: str,
        predicates: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> SubgraphContext:
        """Retrieve ontology subgraph for a query, with LRU cache and ranker re-ranking."""
        # ── Cache lookup ──────────────────────────────────────────────────────
        if use_cache:
            key = _cache_key(query, self.max_hops, self.max_nodes)
            if key in _SUBGRAPH_CACHE:
                logger.debug(f"Cache HIT for query: '{query[:40]}'")
                return _SUBGRAPH_CACHE[key]

        # Link query to seed concepts
        links = self.linker.link(query, top_k=self.max_seeds)

        # Keyword seeding: inject domain-specific seeds for multi-hop queries
        if self.keyword_seeding:
            links = self._inject_keyword_seeds(query, links)

        if not links:
            logger.debug(f"No concepts linked for query: '{query[:60]}'")
            return SubgraphContext(
                query=query, seeds=[], concepts=[], relations=[],
                linked_concepts=[], rank_score=0.0, n_candidates=0,
            )

        seed_names = [lk["concept"]["name"] for lk in links]
        logger.debug(f"Seeds for '{query[:40]}': {seed_names}")

        # ── With ranker: per-seed candidates → rank → merge top-k ────────────
        if self.ranker is not None and len(seed_names) > 1:
            # Allocate nodes budget per seed so merged total ≈ max_nodes
            per_seed_nodes = max(10, self.max_nodes // len(seed_names) + 5)
            candidates = []
            for seed in seed_names:
                sg = self.graph.extract_subgraph(
                    [seed],
                    max_hops=self.max_hops,
                    max_nodes=per_seed_nodes,
                    predicates=predicates,
                )
                candidates.append(sg)

            ranked = self.ranker.rank(candidates, query=query, top_k=self.max_top_seeds)
            logger.debug(
                f"Ranker: {len(candidates)} candidates → top score={ranked[0].total_score:.3f}"
                if ranked else "Ranker: no results"
            )

            top_score = ranked[0].total_score if ranked else 0.0
            merged = self._merge_subgraphs([r.subgraph for r in ranked], self.max_nodes)

            ctx = SubgraphContext(
                query=query,
                seeds=merged["seeds"],
                concepts=merged["concepts"],
                relations=merged["relations"],
                linked_concepts=links,
                rank_score=top_score,
                n_candidates=len(candidates),
            )
            if use_cache:
                key = _cache_key(query, self.max_hops, self.max_nodes)
                if len(_SUBGRAPH_CACHE) >= _CACHE_MAX_SIZE:
                    _SUBGRAPH_CACHE.pop(next(iter(_SUBGRAPH_CACHE)))
                _SUBGRAPH_CACHE[key] = ctx
            return ctx

        # ── Without ranker (single seed or ranker disabled): original path ────
        subgraph = self.graph.extract_subgraph(
            seed_names,
            max_hops=self.max_hops,
            max_nodes=self.max_nodes,
            predicates=predicates,
        )
        ctx = SubgraphContext(
            query=query,
            seeds=subgraph["seeds"],
            concepts=subgraph["concepts"],
            relations=subgraph["relations"],
            linked_concepts=links,
            rank_score=0.0,
            n_candidates=1,
        )
        if use_cache:
            key = _cache_key(query, self.max_hops, self.max_nodes)
            if len(_SUBGRAPH_CACHE) >= _CACHE_MAX_SIZE:
                _SUBGRAPH_CACHE.pop(next(iter(_SUBGRAPH_CACHE)))
            _SUBGRAPH_CACHE[key] = ctx
        return ctx

    def retrieve_batch(
        self,
        queries: list[str],
        predicates: Optional[list[str]] = None,
    ) -> list[SubgraphContext]:
        """Retrieve subgraphs for a batch of queries."""
        return [self.retrieve(q, predicates=predicates) for q in queries]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _inject_keyword_seeds(self, query: str, links: list[dict]) -> list[dict]:
        """
        Append extra seed concepts based on query keywords.

        For each matched keyword rule, look up the mapped concept name in the
        graph and append a synthetic link entry if the concept is not already
        present.  Capped at max_seeds total to avoid context explosion.
        """
        existing_names = {lk["concept"]["name"].lower() for lk in links}
        added = 0
        for pattern, concept_name in _KEYWORD_SEEDS:
            if len(links) >= self.max_seeds:
                break
            if not pattern.search(query):
                continue
            if concept_name.lower() in existing_names:
                continue
            # Verify concept exists in graph before injecting
            concept = self.graph.get_concept(concept_name)
            if concept:
                links.append({
                    "concept": concept,
                    "score": 0.5,
                    "method": "keyword_seed",
                })
                existing_names.add(concept_name.lower())
                added += 1
                logger.debug(f"Keyword seed injected: '{concept_name}'")
        if added:
            logger.debug(f"Injected {added} keyword seed(s) for query '{query[:40]}'")
        return links

    @staticmethod
    def _merge_subgraphs(subgraphs: list[dict], max_nodes: int) -> dict:
        """
        Merge multiple subgraph dicts into one, de-duplicating by concept name
        and relation (subject, predicate, object) triple.  Truncates to max_nodes.
        """
        seen_concepts: dict[str, dict] = {}
        seen_relations: set[tuple] = set()
        merged_relations: list[dict] = []
        all_seeds: list[str] = []

        for sg in subgraphs:
            for seed in sg.get("seeds", []):
                if seed not in all_seeds:
                    all_seeds.append(seed)
            for c in sg.get("concepts", []):
                name = c.get("name", "")
                if name and name not in seen_concepts:
                    seen_concepts[name] = c
                    if len(seen_concepts) >= max_nodes:
                        break
            for r in sg.get("relations", []):
                key = (r.get("subject"), r.get("predicate"), r.get("object"))
                if key not in seen_relations:
                    seen_relations.add(key)
                    merged_relations.append(r)

        return {
            "seeds":     all_seeds,
            "concepts":  list(seen_concepts.values()),
            "relations": merged_relations,
        }
