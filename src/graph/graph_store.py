"""
In-memory tourism ontology graph store.

Loads concepts and relations from the versioned ontology JSON files
and supports:
  - Concept lookup by name (exact and fuzzy)
  - Neighbour traversal (one-hop and multi-hop)
  - Subgraph extraction for a seed set of concept names
  - Class-based concept filtering
  - Relation lookup by predicate
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from ..utils.io import load_json
from ..utils.logging_utils import get_logger

logger = get_logger("graph.store")


class TourismGraphStore:
    """
    Adjacency-list graph over tourism ontology concepts and relations.

    Nodes : concept names (str)
    Edges : relations  { subject, predicate, object, provenance }

    The graph is directed (subject → object) with edges indexed both ways
    for efficient bidirectional traversal.
    """

    def __init__(self) -> None:
        # node metadata: name → concept dict
        self._concepts: dict[str, dict] = {}
        # forward edges:  subject → list[relation_dict]
        self._out_edges: dict[str, list[dict]] = defaultdict(list)
        # backward edges: object  → list[relation_dict]
        self._in_edges:  dict[str, list[dict]] = defaultdict(list)
        # class index: class_name → set of concept names
        self._class_index: dict[str, set[str]] = defaultdict(set)
        # predicate index: predicate → list[relation_dict]
        self._pred_index: dict[str, list[dict]] = defaultdict(list)
        # lowercase name → canonical name (for case-insensitive lookup)
        self._name_lower: dict[str, str] = {}

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_version(
        cls,
        version: str,
        ontology_dir: str | Path = "artifacts/ontologies",
    ) -> "TourismGraphStore":
        """Load graph from a versioned ontology directory."""
        ont_dir = Path(ontology_dir) / version
        concepts  = load_json(ont_dir / "concepts.json")
        relations = load_json(ont_dir / "relations.json")
        store = cls()
        store.load(concepts, relations)
        logger.info(
            f"GraphStore loaded v{version}: "
            f"{len(store._concepts):,} concepts, "
            f"{sum(len(v) for v in store._out_edges.values()):,} relations"
        )
        return store

    @classmethod
    def from_current(
        cls,
        ontology_dir: str | Path = "artifacts/ontologies",
    ) -> "TourismGraphStore":
        """Load graph from the currently active ontology version."""
        ont_dir = Path(ontology_dir)
        current = load_json(ont_dir / "current.json")
        version = current["current_version"]
        return cls.from_version(version, ontology_dir)

    def load(self, concepts: list[dict], relations: list[dict]) -> None:
        """Populate the graph from raw concept/relation dicts."""
        for c in concepts:
            name = c.get("name", "")
            if not name:
                continue
            self._concepts[name] = c
            self._name_lower[name.lower()] = name
            cls = c.get("concept_class", "")
            if cls:
                self._class_index[cls].add(name)

        for r in relations:
            subj = r.get("subject", "")
            obj  = r.get("object", "")
            pred = r.get("predicate", "")
            if not (subj and obj and pred):
                continue
            self._out_edges[subj].append(r)
            self._in_edges[obj].append(r)
            self._pred_index[pred].append(r)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_concept(self, name: str) -> Optional[dict]:
        """Exact lookup by concept name (case-insensitive)."""
        canonical = self._name_lower.get(name.lower())
        if canonical:
            return self._concepts[canonical]
        return None

    def get_concepts_by_class(self, concept_class: str) -> list[dict]:
        """Return all concepts of a given class."""
        return [self._concepts[n] for n in self._class_index.get(concept_class, set())]

    def get_relations_by_predicate(self, predicate: str) -> list[dict]:
        """Return all relations with the given predicate."""
        return self._pred_index.get(predicate, [])

    def neighbours(
        self,
        name: str,
        direction: str = "out",   # "out" | "in" | "both"
        predicates: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return one-hop neighbour relations for a concept."""
        canonical = self._name_lower.get(name.lower(), name)
        rels: list[dict] = []
        if direction in ("out", "both"):
            rels.extend(self._out_edges.get(canonical, []))
        if direction in ("in", "both"):
            rels.extend(self._in_edges.get(canonical, []))
        if predicates:
            pred_set = set(predicates)
            rels = [r for r in rels if r.get("predicate") in pred_set]
        return rels

    # ── Subgraph extraction ───────────────────────────────────────────────────

    def extract_subgraph(
        self,
        seed_names: list[str],
        max_hops: int = 2,
        max_nodes: int = 50,
        predicates: Optional[list[str]] = None,
    ) -> dict:
        """
        BFS from seed_names up to max_hops, collecting at most max_nodes nodes.

        Returns dict with:
          concepts  : list of concept dicts in the subgraph
          relations : list of relation dicts within the subgraph
          seeds     : the resolved seed names
          hops      : number of hops expanded
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        # Resolve seeds (case-insensitive)
        resolved_seeds: list[str] = []
        for raw in seed_names:
            canonical = self._name_lower.get(raw.lower())
            if canonical:
                resolved_seeds.append(canonical)
                queue.append((canonical, 0))
                visited.add(canonical)

        if not resolved_seeds:
            return {"concepts": [], "relations": [], "seeds": [], "hops": 0}

        max_hop_reached = 0
        while queue and len(visited) < max_nodes:
            node, hop = queue.popleft()
            if hop >= max_hops:
                max_hop_reached = max(max_hop_reached, hop)
                continue
            max_hop_reached = max(max_hop_reached, hop)
            for rel in self.neighbours(node, direction="both", predicates=predicates):
                for neighbour in (rel.get("subject", ""), rel.get("object", "")):
                    if neighbour and neighbour not in visited:
                        visited.add(neighbour)
                        queue.append((neighbour, hop + 1))
                        if len(visited) >= max_nodes:
                            break

        # Collect concepts and internal relations
        subgraph_concepts = [
            self._concepts[n] for n in visited if n in self._concepts
        ]
        subgraph_relations = [
            r for n in visited
            for r in self._out_edges.get(n, [])
            if r.get("object", "") in visited
        ]

        return {
            "concepts":  subgraph_concepts,
            "relations": subgraph_relations,
            "seeds":     resolved_seeds,
            "hops":      max_hop_reached,
        }

    # ── Fuzzy search ──────────────────────────────────────────────────────────

    def search_concepts(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return up to top_k concepts whose name contains any query token.
        Ranked by token overlap count (descending).
        """
        tokens = set(re.split(r"\W+", query.lower()))
        tokens.discard("")
        scores: list[tuple[int, dict]] = []
        for name_lower, canonical in self._name_lower.items():
            name_tokens = set(re.split(r"\W+", name_lower))
            overlap = len(tokens & name_tokens)
            if overlap > 0:
                scores.append((overlap, self._concepts[canonical]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scores[:top_k]]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_concepts":  len(self._concepts),
            "n_relations": sum(len(v) for v in self._out_edges.values()),
            "n_classes":   len(self._class_index),
            "n_predicates": len(self._pred_index),
            "class_counts": {k: len(v) for k, v in sorted(
                self._class_index.items(), key=lambda x: len(x[1]), reverse=True
            )},
        }
