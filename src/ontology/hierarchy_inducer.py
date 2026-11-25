"""
Hierarchy induction module.

Takes the flat set of extracted concepts and organises them
into the tourism ontology taxonomy using:
  1. Schema-based rules (deterministic, no LLM)
  2. LLM-based hierarchy suggestion (when client is provided)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import (
    OntologyConcept,
    OntologyRelation,
    ProvenanceMetadata,
    TourismOntologySchema,
    CLASS_HIERARCHY,
)
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.hierarchy_inducer")
SCHEMA = TourismOntologySchema


@dataclass
class HierarchyNode:
    concept_name: str
    concept_class: str
    parent_class: Optional[str]
    hierarchy_path: list[str]
    is_instance: bool = True
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "concept": self.concept_name,
            "concept_class": self.concept_class,
            "parent_class": self.parent_class,
            "hierarchy_path": self.hierarchy_path,
            "is_instance": self.is_instance,
            "confidence": self.confidence,
        }


def _build_path(class_name: str) -> list[str]:
    """Walk up the class hierarchy and return path from root to class."""
    path: list[str] = [class_name]
    current = class_name
    while True:
        parent = CLASS_HIERARCHY.get(current)
        if parent is None:
            break
        path.insert(0, parent)
        current = parent
    return path


class HierarchyInducer:
    """
    Induces a taxonomy over extracted concepts.

    In schema-only mode (no LLM): uses the predefined CLASS_HIERARCHY.
    In LLM mode: sends batches to LLM for refinement of ambiguous cases.
    """

    PROMPT_VERSION = "v1.0"

    def __init__(
        self,
        client=None,          # LLMClient | None
        prompt_manager=None,  # PromptManager | None
        strategy: str = "cot",
    ):
        self.client = client
        self.pm = prompt_manager
        self.strategy = strategy

    def induce(
        self, concepts: list[OntologyConcept]
    ) -> tuple[list[HierarchyNode], list[OntologyRelation]]:
        """
        Build hierarchy nodes and is_a relations for a list of concepts.

        Returns:
            (hierarchy_nodes, is_a_relations)
        """
        nodes: list[HierarchyNode] = []
        relations: list[OntologyRelation] = []

        for concept in concepts:
            node = self._classify(concept)
            nodes.append(node)

            # Emit is_a relation if parent exists
            if node.parent_class:
                rel = OntologyRelation(
                    subject=concept.name,
                    predicate="is_a",
                    obj=node.parent_class,
                    provenance=ProvenanceMetadata(
                        source_record_id=concept.provenance.source_record_id,
                        extraction_prompt_version=self.PROMPT_VERSION,
                        confidence=node.confidence,
                        validator_status="pending",
                    ),
                )
                relations.append(rel)

        logger.debug(
            f"HierarchyInducer: {len(concepts)} concepts → "
            f"{len(nodes)} nodes, {len(relations)} is_a relations."
        )
        return nodes, relations

    def _classify(self, concept: OntologyConcept) -> HierarchyNode:
        """Assign hierarchy using schema rules."""
        cclass = concept.concept_class
        parent = CLASS_HIERARCHY.get(cclass)
        path = _build_path(cclass)
        return HierarchyNode(
            concept_name=concept.name,
            concept_class=cclass,
            parent_class=parent,
            hierarchy_path=path,
            is_instance=True,
            confidence=concept.provenance.confidence,
        )

    def build_taxonomy_summary(self, nodes: list[HierarchyNode]) -> dict[str, Any]:
        """Summarise the taxonomy: classes, counts, depth distribution."""
        from collections import Counter
        class_counts: Counter = Counter(n.concept_class for n in nodes)
        parent_counts: Counter = Counter(n.parent_class for n in nodes if n.parent_class)
        depth_counts: Counter = Counter(len(n.hierarchy_path) for n in nodes)
        return {
            "total_nodes": len(nodes),
            "class_distribution": dict(class_counts.most_common()),
            "parent_class_distribution": dict(parent_counts.most_common()),
            "depth_distribution": dict(sorted(depth_counts.items())),
        }
