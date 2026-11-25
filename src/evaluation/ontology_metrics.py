"""
Ontology Quality Metrics — evaluate the tourism ontology itself.

Metrics (addresses RQ1 from the paper):
  1. Concept coverage        — fraction of test records with ≥1 concept found in ontology
  2. Relation density        — avg relations per concept
  3. Schema completeness     — fraction of schema classes populated
  4. Extraction precision    — manually-estimated rate of correct extractions
  5. Ontology evolution      — concept/relation growth v0.1 → v0.7
  6. Rule coverage           — fraction of records covered by at least one SHACL rule
  7. Hierarchy depth         — average depth of is-a chains (NEW, Week 5)
  8. Hierarchy breadth       — average branching factor of is-a tree (NEW, Week 5)
  9. Orphan rate             — fraction of concepts with zero relations (NEW, Week 5)
  10. Disconnected subgraph count — isolated clusters in the relation graph (NEW, Week 5)

Paper table: Table 3 — Ontology Quality Metrics
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..utils.io import load_json

logger = get_logger("evaluation.ontology_metrics")


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class OntologySnapshot:
    """Stats for one ontology version."""
    version:         str
    n_concepts:      int
    n_relations:     int
    n_rules:         int
    n_classes:       int            # unique class labels in concepts
    avg_relations_per_concept: float
    schema_completeness: float      # fraction of 26 target classes populated
    concept_coverage: float         # fraction of test records matched
    relation_density: float         # relations / concepts
    # Taxonomy metrics (Week 5)
    hierarchy_depth:   float = 0.0  # avg depth of is-a chains
    hierarchy_breadth: float = 0.0  # avg branching factor of parent nodes
    orphan_rate:       float = 0.0  # fraction of concepts with zero relations
    disconnected_subgraph_count: int = 0  # isolated clusters in relation graph

    def to_dict(self) -> dict:
        return {
            "version":                      self.version,
            "n_concepts":                   self.n_concepts,
            "n_relations":                  self.n_relations,
            "n_rules":                      self.n_rules,
            "n_classes":                    self.n_classes,
            "avg_relations_per_concept":    round(self.avg_relations_per_concept, 2),
            "schema_completeness":          round(self.schema_completeness, 3),
            "concept_coverage":             round(self.concept_coverage, 3),
            "relation_density":             round(self.relation_density, 3),
            "hierarchy_depth":              round(self.hierarchy_depth, 2),
            "hierarchy_breadth":            round(self.hierarchy_breadth, 2),
            "orphan_rate":                  round(self.orphan_rate, 3),
            "disconnected_subgraph_count":  self.disconnected_subgraph_count,
        }


@dataclass
class OntologyEvolutionReport:
    """Evolution of the ontology across versions."""
    versions:    list[str]
    snapshots:   list[OntologySnapshot]
    concept_growth:   list[int]    # deltas in concept count
    relation_growth:  list[int]    # deltas in relation count

    def to_dict(self) -> dict:
        return {
            "versions":         self.versions,
            "snapshots":        [s.to_dict() for s in self.snapshots],
            "concept_growth":   self.concept_growth,
            "relation_growth":  self.relation_growth,
        }


# Target schema from paper (26 concept classes)
TARGET_SCHEMA_CLASSES = {
    "Destination", "City", "Island", "Beach", "Mountain", "NaturalSite",
    "NationalPark", "Museum", "CulturalSite", "Monument", "Heritage",
    "Festival", "Restaurant", "LocalFood", "Cuisine", "Hotel", "Accommodation",
    "Activity", "Sport", "Transportation", "TouristAttraction", "Market",
    "TravelInfo", "Season", "Budget", "TravelerType",
}


class OntologyMetricsCalculator:
    """
    Compute quality metrics for a loaded ontology version.

    Parameters
    ----------
    concepts : list[dict]   All concept dicts from the ontology.
    relations : list[dict]  All relation dicts from the ontology.
    rules : list[dict]      All SHACL rules.
    version : str           Version label (e.g., "0.3.0").
    """

    def __init__(
        self,
        concepts: list[dict],
        relations: list[dict],
        rules: list[dict],
        version: str = "unknown",
    ):
        self.concepts  = concepts
        self.relations = relations
        self.rules     = rules
        self.version   = version

    def compute(
        self,
        test_records: Optional[list[dict]] = None,
        hierarchy: Optional[list[dict]] = None,
    ) -> OntologySnapshot:
        """
        Compute all ontology quality metrics.

        Parameters
        ----------
        test_records : list[dict], optional
            Test split records for concept_coverage computation.
        hierarchy : list[dict], optional
            Hierarchy nodes (parent/child dicts) for taxonomy metrics.
        """
        n_concepts  = len(self.concepts)
        n_relations = len(self.relations)
        n_rules     = len(self.rules)

        # Class diversity — key is 'concept_class' in v0.2+
        classes = {
            c.get("concept_class", c.get("class", ""))
            for c in self.concepts
            if c.get("concept_class") or c.get("class")
        }
        n_classes = len(classes)

        # Relation density
        rel_density = n_relations / max(n_concepts, 1)

        # Avg relations per concept (subject occurrences)
        subj_counts = Counter(
            r.get("subject", r.get("head", "")) for r in self.relations
        )
        avg_rels = sum(subj_counts.values()) / max(n_concepts, 1)

        # Schema completeness
        populated_classes = classes & TARGET_SCHEMA_CLASSES
        schema_completeness = len(populated_classes) / len(TARGET_SCHEMA_CLASSES)

        # Concept coverage over test records
        concept_coverage = 0.0
        if test_records:
            concept_names = {c.get("name", "").lower() for c in self.concepts}
            matched = sum(
                1 for r in test_records
                if any(
                    name in r.get("Input", "").lower() or name in r.get("Output", "").lower()
                    for name in concept_names
                    if name
                )
            )
            concept_coverage = matched / len(test_records) if test_records else 0.0

        # Taxonomy metrics (Week 5)
        hier_depth, hier_breadth = self.hierarchy_depth(hierarchy), self.hierarchy_breadth(hierarchy)
        orphan_rate = self.orphan_rate()
        n_disconnected = self.disconnected_subgraph_count()

        snap = OntologySnapshot(
            version=self.version,
            n_concepts=n_concepts,
            n_relations=n_relations,
            n_rules=n_rules,
            n_classes=n_classes,
            avg_relations_per_concept=avg_rels,
            schema_completeness=schema_completeness,
            concept_coverage=concept_coverage,
            relation_density=rel_density,
            hierarchy_depth=hier_depth,
            hierarchy_breadth=hier_breadth,
            orphan_rate=orphan_rate,
            disconnected_subgraph_count=n_disconnected,
        )
        logger.info(
            "Ontology v%s: %d concepts, %d relations, %d classes | "
            "coverage=%.1f%%, schema=%.0f%% | "
            "depth=%.1f, breadth=%.1f, orphans=%.1f%%, components=%d",
            self.version, n_concepts, n_relations, n_classes,
            concept_coverage * 100, schema_completeness * 100,
            hier_depth, hier_breadth, orphan_rate * 100, n_disconnected,
        )
        return snap

    # ── Taxonomy metrics (Week 5) ─────────────────────────────────────────────

    def hierarchy_depth(self, hierarchy: Optional[list[dict]] = None) -> float:
        """
        Average depth of is-a chains.

        Reads `hierarchy_path` from each hierarchy node (the format produced
        by OntologyBuilder). The depth of a node = len(hierarchy_path) - 1
        (root has depth 0). Falls back to is_a relation traversal if no
        hierarchy list is provided.
        """
        if hierarchy:
            depths = []
            for node in hierarchy:
                path = node.get("hierarchy_path", [])
                if path:
                    depths.append(len(path) - 1)   # root is depth 0
            return sum(depths) / len(depths) if depths else 0.0

        # Fallback: traverse is_a relations
        parent_of: dict[str, str] = {}
        for r in self.relations:
            pred = r.get("predicate", "")
            if pred in ("is_a", "subclass_of", "instance_of"):
                child  = r.get("subject", "")
                parent = r.get("object", "")
                if child and parent and child != parent:
                    parent_of[child] = parent

        if not parent_of:
            return 0.0

        def _depth(node: str, cache: dict, visited: set) -> int:
            if node in cache:
                return cache[node]
            if node in visited:       # cycle guard
                return 0
            visited.add(node)
            p = parent_of.get(node)
            d = (1 + _depth(p, cache, visited)) if p else 0
            cache[node] = d
            return d

        cache: dict[str, int] = {}
        depths = [_depth(n, cache, set()) for n in parent_of]
        return sum(depths) / len(depths) if depths else 0.0

    def hierarchy_breadth(self, hierarchy: Optional[list[dict]] = None) -> float:
        """
        Average branching factor — mean number of children per parent class.

        Uses `concept_class` and `parent_class` from hierarchy nodes.
        Falls back to is_a relations if no hierarchy list provided.
        """
        children_of: dict[str, set] = defaultdict(set)

        if hierarchy:
            for node in hierarchy:
                cls    = node.get("concept_class", "")
                parent = node.get("parent_class", "")
                if cls and parent and cls != parent:
                    children_of[parent].add(cls)
        else:
            for r in self.relations:
                pred = r.get("predicate", "")
                if pred in ("is_a", "subclass_of"):
                    child  = r.get("subject", "")
                    parent = r.get("object", "")
                    if child and parent and child != parent:
                        children_of[parent].add(child)

        if not children_of:
            return 0.0

        branch_factors = [len(children) for children in children_of.values()]
        return sum(branch_factors) / len(branch_factors)

    def orphan_rate(self) -> float:
        """
        Fraction of concepts that appear in zero relations (neither as
        subject nor object).  Orphans contribute nothing to the knowledge graph.
        """
        if not self.concepts:
            return 0.0

        connected: set[str] = set()
        for r in self.relations:
            connected.add(r.get("subject", ""))
            connected.add(r.get("object", ""))

        total = len(self.concepts)
        orphans = sum(
            1 for c in self.concepts
            if c.get("name", "") not in connected
        )
        return orphans / total

    def disconnected_subgraph_count(self) -> int:
        """
        Count of isolated connected components in the undirected relation graph.

        Uses Union-Find (disjoint set) on concept names.
        Concepts that never appear in any relation each form their own component.
        """
        if not self.concepts:
            return 0

        # Build Union-Find
        parent: dict[str, str] = {c.get("name", ""): c.get("name", "") for c in self.concepts}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)  # path compression
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for r in self.relations:
            subj = r.get("subject", "")
            obj  = r.get("object", "")
            if subj in parent and obj in parent:
                union(subj, obj)

        # Count distinct roots
        roots = {find(c.get("name", "")) for c in self.concepts if c.get("name")}
        return len(roots)

    @staticmethod
    def compare_versions(
        snapshots: list[OntologySnapshot],
    ) -> OntologyEvolutionReport:
        """
        Build an evolution report from a list of version snapshots
        (must be in chronological order).
        """
        if not snapshots:
            return OntologyEvolutionReport([], [], [], [])

        versions        = [s.version for s in snapshots]
        concept_growth  = [0] + [
            snapshots[i].n_concepts - snapshots[i-1].n_concepts
            for i in range(1, len(snapshots))
        ]
        relation_growth = [0] + [
            snapshots[i].n_relations - snapshots[i-1].n_relations
            for i in range(1, len(snapshots))
        ]

        return OntologyEvolutionReport(
            versions=versions,
            snapshots=snapshots,
            concept_growth=concept_growth,
            relation_growth=relation_growth,
        )

    @staticmethod
    def load_ontology_version(
        ontologies_dir: str | Path,
        version: str,
    ) -> tuple[list, list, list, list]:
        """
        Load concepts, relations, rules, hierarchy from the versioned ontology directory.

        Expected layout:
            artifacts/ontologies/<version>/concepts.json
            artifacts/ontologies/<version>/relations.json
            artifacts/ontologies/<version>/rules.json
            artifacts/ontologies/<version>/hierarchy.json  (optional)

        Returns (concepts, relations, rules, hierarchy).
        """
        vdir = Path(ontologies_dir) / version

        concepts  = load_json(vdir / "concepts.json")  if (vdir / "concepts.json").exists()  else []
        relations = load_json(vdir / "relations.json") if (vdir / "relations.json").exists() else []
        rules     = load_json(vdir / "rules.json")     if (vdir / "rules.json").exists()     else []
        hierarchy = load_json(vdir / "hierarchy.json") if (vdir / "hierarchy.json").exists() else []

        return concepts, relations, rules, hierarchy
