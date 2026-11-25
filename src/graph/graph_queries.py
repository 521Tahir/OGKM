"""
Graph query layer for the tourism ontology.

Provides high-level query operations on top of TourismGraphStore:
  - lookup_concept()              — exact + fuzzy concept lookup
  - find_aliases()                — alternative names for a concept
  - shortest_explanation_path()   — BFS shortest path with relation labels
  - confidence_ranked_neighbours()— neighbours filtered + ranked by confidence
  - provenance_filter()           — filter by extraction source
  - concepts_by_class()           — all concepts of a class, confidence-ranked
  - related_concepts()            — concepts sharing a predicate with a seed
  - path_exists()                 — quick reachability check
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .graph_store import TourismGraphStore
from ..utils.logging_utils import get_logger

logger = get_logger("graph.queries")


# ── Result types ──────────────────────────────────────────────────────────────

class ConceptMatch:
    """A concept lookup result with a match score."""
    __slots__ = ("concept", "score", "match_type")

    def __init__(self, concept: dict, score: float, match_type: str) -> None:
        self.concept    = concept      # full concept dict
        self.score      = score        # 1.0 = exact, <1.0 = fuzzy
        self.match_type = match_type   # "exact" | "alias" | "fuzzy"

    def __repr__(self) -> str:
        return f"ConceptMatch({self.concept.get('name')!r}, score={self.score:.2f}, type={self.match_type!r})"


class PathStep:
    """One step in an explanation path."""
    __slots__ = ("concept", "relation", "direction")

    def __init__(self, concept: dict, relation: Optional[dict], direction: str) -> None:
        self.concept   = concept    # concept dict at this node
        self.relation  = relation   # relation dict that led here (None for seed)
        self.direction = direction  # "forward" | "backward" | "seed"

    def __repr__(self) -> str:
        name = self.concept.get("name", "?")
        if self.relation:
            pred = self.relation.get("predicate", "?")
            return f"PathStep({name!r} via {pred!r} [{self.direction}])"
        return f"PathStep({name!r} [seed])"


class ExplanationPath:
    """BFS shortest path between two concepts."""

    def __init__(self, source: str, target: str, steps: list[PathStep]) -> None:
        self.source = source
        self.target = target
        self.steps  = steps   # ordered seed → target

    @property
    def length(self) -> int:
        return max(0, len(self.steps) - 1)

    def to_text(self) -> str:
        """Human-readable path description."""
        if not self.steps:
            return f"No path from '{self.source}' to '{self.target}'"
        parts = []
        for step in self.steps:
            name = step.concept.get("name", "?")
            if step.relation:
                pred = step.relation.get("predicate", "?")
                arrow = "→" if step.direction == "forward" else "←"
                parts.append(f"—[{pred}]—{arrow} {name}")
            else:
                parts.append(name)
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "length": self.length,
            "path":   self.to_text(),
            "steps":  [
                {
                    "concept":   s.concept.get("name"),
                    "relation":  s.relation.get("predicate") if s.relation else None,
                    "direction": s.direction,
                }
                for s in self.steps
            ],
        }

    def __repr__(self) -> str:
        return f"ExplanationPath(len={self.length}, {self.to_text()!r})"


# ── Main query class ──────────────────────────────────────────────────────────

class GraphQueries:
    """
    High-level query interface over TourismGraphStore.

    Usage:
        gq = GraphQueries(graph)
        match = gq.lookup_concept("Jeju Island")
        path  = gq.shortest_explanation_path("Jeju Island", "Beach")
        nbrs  = gq.confidence_ranked_neighbours("Jeju Island", min_conf=0.7)
    """

    def __init__(self, graph: TourismGraphStore) -> None:
        self.graph = graph

    # ── Concept lookup ────────────────────────────────────────────────────────

    def lookup_concept(
        self,
        name: str,
        top_k: int = 5,
    ) -> list[ConceptMatch]:
        """
        Find concepts matching the given name.

        Returns a list of ConceptMatch objects ordered by score:
          1. Exact case-insensitive match (score=1.0)
          2. Alias match (score=0.9)
          3. Fuzzy token overlap (score=overlap/max_tokens)
        """
        results: list[ConceptMatch] = []
        seen: set[str] = set()

        # 1. Exact
        exact = self.graph.get_concept(name)
        if exact:
            results.append(ConceptMatch(exact, 1.0, "exact"))
            seen.add(exact["name"])

        # 2. Alias match
        name_lower = name.lower()
        for cname, concept in self.graph._concepts.items():
            if cname in seen:
                continue
            aliases = concept.get("aliases", []) or []
            if any(a.lower() == name_lower for a in aliases):
                results.append(ConceptMatch(concept, 0.9, "alias"))
                seen.add(cname)

        # 3. Fuzzy token overlap
        if len(results) < top_k:
            fuzzy = self.graph.search_concepts(name, top_k=top_k * 2)
            import re
            query_tokens = set(re.split(r"\W+", name_lower))
            query_tokens.discard("")
            for concept in fuzzy:
                cname = concept.get("name", "")
                if cname in seen:
                    continue
                name_tokens = set(re.split(r"\W+", cname.lower()))
                name_tokens.discard("")
                if not (query_tokens and name_tokens):
                    continue
                overlap = len(query_tokens & name_tokens)
                score   = overlap / max(len(query_tokens), len(name_tokens))
                if score > 0:
                    results.append(ConceptMatch(concept, round(score, 3), "fuzzy"))
                    seen.add(cname)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ── Alias lookup ──────────────────────────────────────────────────────────

    def find_aliases(self, name: str) -> list[str]:
        """
        Return all known aliases for a concept.
        Also returns names of concepts that list this concept as an alias.
        """
        concept = self.graph.get_concept(name)
        if not concept:
            return []

        aliases: list[str] = list(concept.get("aliases", []) or [])

        # Reverse: find concepts that alias back to this name
        name_lower = name.lower()
        for cname, c in self.graph._concepts.items():
            if cname == concept["name"]:
                continue
            if any(a.lower() == name_lower for a in (c.get("aliases", []) or [])):
                if cname not in aliases:
                    aliases.append(cname)

        return aliases

    # ── Shortest explanation path ─────────────────────────────────────────────

    def shortest_explanation_path(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
        bidirectional: bool = True,
    ) -> ExplanationPath:
        """
        Find the shortest path between source and target using BFS.

        Args:
            source:        start concept name
            target:        end concept name
            max_hops:      maximum path length
            bidirectional: also traverse backward edges

        Returns:
            ExplanationPath with ordered steps (empty steps if no path found)
        """
        src_concept = self.graph.get_concept(source)
        tgt_concept = self.graph.get_concept(target)

        if not src_concept or not tgt_concept:
            logger.debug(f"shortest_path: concept not found ({source!r} or {target!r})")
            return ExplanationPath(source, target, [])

        src_name = src_concept["name"]
        tgt_name = tgt_concept["name"]

        if src_name == tgt_name:
            return ExplanationPath(
                source, target,
                [PathStep(src_concept, None, "seed")]
            )

        # BFS: state = (concept_name, path_so_far)
        # path is list of (concept_name, relation_dict|None, direction)
        queue: deque[tuple[str, list]] = deque()
        queue.append((src_name, [(src_name, None, "seed")]))
        visited: set[str] = {src_name}

        while queue:
            node, path = queue.popleft()
            if len(path) > max_hops + 1:
                continue

            directions = [("out", "forward")]
            if bidirectional:
                directions.append(("in", "backward"))

            for direction_str, label in directions:
                for rel in self.graph.neighbours(node, direction=direction_str):
                    neighbour = (
                        rel.get("object", "") if direction_str == "out"
                        else rel.get("subject", "")
                    )
                    if not neighbour or neighbour in visited:
                        continue

                    new_path = path + [(neighbour, rel, label)]

                    if neighbour == tgt_name:
                        # Build PathStep list
                        steps = []
                        for i, (cname, rel_dict, dir_str) in enumerate(new_path):
                            c = self.graph._concepts.get(cname, {"name": cname})
                            steps.append(PathStep(c, rel_dict, dir_str))
                        return ExplanationPath(source, target, steps)

                    visited.add(neighbour)
                    queue.append((neighbour, new_path))

        return ExplanationPath(source, target, [])

    # ── Confidence-ranked neighbours ──────────────────────────────────────────

    def confidence_ranked_neighbours(
        self,
        name: str,
        min_conf: float = 0.0,
        direction: str = "both",
        top_k: int = 20,
        predicates: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Return neighbours of a concept ranked by relation confidence.

        Args:
            name:       concept name to expand
            min_conf:   minimum confidence threshold (0.0–1.0)
            direction:  "out" | "in" | "both"
            top_k:      max results to return
            predicates: filter to specific predicate types

        Returns:
            List of dicts: {"concept": ..., "relation": ..., "confidence": ...}
        """
        rels = self.graph.neighbours(name, direction=direction, predicates=predicates)
        results = []
        for rel in rels:
            conf = float(rel.get("confidence", 0.0))
            if conf < min_conf:
                continue
            # Get the neighbour concept
            if direction == "out" or (direction == "both" and rel.get("subject", "").lower() == name.lower()):
                neighbour_name = rel.get("object", "")
            else:
                neighbour_name = rel.get("subject", "")

            concept = self.graph.get_concept(neighbour_name)
            if concept:
                results.append({
                    "concept":    concept,
                    "relation":   rel,
                    "confidence": conf,
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    # ── Provenance filter ─────────────────────────────────────────────────────

    def provenance_filter(
        self,
        name: str,
        source: str,
        direction: str = "both",
    ) -> list[dict]:
        """
        Return relations for a concept filtered by extraction source.

        Args:
            name:      concept name
            source:    source tag to filter on, e.g. "llm", "offline", "manual"
            direction: "out" | "in" | "both"

        Returns:
            List of relation dicts whose provenance matches source
        """
        rels = self.graph.neighbours(name, direction=direction)
        source_lower = source.lower()
        return [
            r for r in rels
            if source_lower in str(r.get("provenance", {}).get("source", "")).lower()
            or source_lower in str(r.get("source", "")).lower()
        ]

    # ── Concepts by class ─────────────────────────────────────────────────────

    def concepts_by_class(
        self,
        concept_class: str,
        min_conf: float = 0.0,
        top_k: int = 50,
    ) -> list[dict]:
        """
        Return all concepts of a given class, ranked by confidence.

        Args:
            concept_class: e.g. "Beach", "Restaurant", "Destination"
            min_conf:      minimum concept confidence
            top_k:         max results

        Returns:
            List of concept dicts sorted by confidence descending
        """
        concepts = self.graph.get_concepts_by_class(concept_class)
        filtered = [
            c for c in concepts
            if float(c.get("provenance", {}).get("confidence", c.get("confidence", 0.0))) >= min_conf
        ]
        filtered.sort(
            key=lambda c: float(c.get("provenance", {}).get("confidence", c.get("confidence", 0.0))),
            reverse=True,
        )
        return filtered[:top_k]

    # ── Related concepts ──────────────────────────────────────────────────────

    def related_concepts(
        self,
        name: str,
        predicate: str,
        direction: str = "out",
        min_conf: float = 0.0,
    ) -> list[dict]:
        """
        Return concepts connected to name via a specific predicate.

        Example:
            gq.related_concepts("Jeju Island", "has_activity") → [Hiking, Surfing, ...]
        """
        rels = self.graph.neighbours(name, direction=direction, predicates=[predicate])
        results = []
        for rel in rels:
            if float(rel.get("confidence", 0.0)) < min_conf:
                continue
            neighbour = rel.get("object" if direction == "out" else "subject", "")
            concept = self.graph.get_concept(neighbour)
            if concept:
                results.append(concept)
        return results

    # ── Reachability check ────────────────────────────────────────────────────

    def path_exists(self, source: str, target: str, max_hops: int = 3) -> bool:
        """Quick check whether a path exists between two concepts within max_hops."""
        path = self.shortest_explanation_path(source, target, max_hops=max_hops)
        return len(path.steps) > 0

    # ── Summary ───────────────────────────────────────────────────────────────

    def concept_summary(self, name: str) -> dict:
        """
        Return a compact summary of a concept: metadata + top neighbours.

        Useful for building explanation text or context snippets.
        """
        concept = self.graph.get_concept(name)
        if not concept:
            return {}

        out_rels = self.graph.neighbours(name, direction="out")
        in_rels  = self.graph.neighbours(name, direction="in")

        top_out = sorted(out_rels, key=lambda r: float(r.get("confidence", 0)), reverse=True)[:5]
        top_in  = sorted(in_rels,  key=lambda r: float(r.get("confidence", 0)), reverse=True)[:5]

        return {
            "name":        concept.get("name"),
            "class":       concept.get("concept_class"),
            "aliases":     concept.get("aliases", []),
            "confidence":  concept.get("provenance", {}).get("confidence", concept.get("confidence")),
            "out_edges":   [{"predicate": r["predicate"], "object": r["object"]} for r in top_out],
            "in_edges":    [{"predicate": r["predicate"], "subject": r["subject"]} for r in top_in],
            "degree":      len(out_rels) + len(in_rels),
        }
