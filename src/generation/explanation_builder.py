"""
Explanation Builder — constructs human-readable provenance traces for answers.

For each answer, traces which ontology facts were used:
  - Which concepts from the subgraph were referenced
  - Which relations formed the reasoning chain
  - What the confidence level of each supporting fact is

Produces structured ExplanationTrace objects and renders them as text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..retrieval.subgraph_retriever import SubgraphContext
from ..generation.constrained_generator import GenerationResult
from ..graph.graph_store import TourismGraphStore
from ..utils.logging_utils import get_logger

logger = get_logger("generation.explanation")


@dataclass
class SupportingFact:
    """A single ontology fact that supports the answer."""
    fact_type: str      # "concept" | "relation"
    description: str    # human-readable description
    confidence: float
    predicate:  str = ""
    subject:    str = ""
    obj:        str = ""

    def to_dict(self) -> dict:
        return {
            "fact_type":   self.fact_type,
            "description": self.description,
            "confidence":  round(self.confidence, 3),
            "predicate":   self.predicate,
            "subject":     self.subject,
            "object":      self.obj,
        }


@dataclass
class ExplanationTrace:
    """Full explanation for a generated answer."""
    query:           str
    answer:          str
    supporting_facts: list[SupportingFact] = field(default_factory=list)
    average_confidence: float = 0.0
    n_ontology_facts:   int   = 0
    grounding_score:    float = 0.0  # fraction of answer concepts grounded in ontology

    def to_text(self) -> str:
        lines = [
            f"Question: {self.query}",
            f"Answer:   {self.answer[:200]}{'…' if len(self.answer) > 200 else ''}",
            f"\nGrounding score : {self.grounding_score:.1%}",
            f"Avg confidence  : {self.average_confidence:.2f}",
            f"Supporting facts: {self.n_ontology_facts}",
        ]
        if self.supporting_facts:
            lines.append("\nTop supporting facts:")
            for f in self.supporting_facts[:10]:
                lines.append(f"  [{f.fact_type}] {f.description} (conf={f.confidence:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query":               self.query,
            "answer":              self.answer,
            "grounding_score":     round(self.grounding_score, 3),
            "average_confidence":  round(self.average_confidence, 3),
            "n_ontology_facts":    self.n_ontology_facts,
            "supporting_facts":    [f.to_dict() for f in self.supporting_facts],
        }


class ExplanationBuilder:
    """
    Builds explanation traces linking generated answers to ontology facts.

    Usage:
        builder = ExplanationBuilder(graph)
        trace = builder.build(result, subgraph_ctx)
    """

    def __init__(self, graph: TourismGraphStore) -> None:
        self.graph = graph

    def build(
        self,
        result: GenerationResult,
        context: Optional[SubgraphContext] = None,
    ) -> ExplanationTrace:
        """Build an explanation trace for a generation result."""
        answer = result.answer
        facts: list[SupportingFact] = []

        if context and not context.is_empty():
            facts = self._match_facts(answer, context)

        avg_conf = (
            sum(f.confidence for f in facts) / len(facts) if facts else 0.0
        )

        # Grounding score: fraction of capitalized answer entities found in ontology
        grounding = self._grounding_score(answer)

        trace = ExplanationTrace(
            query=result.query,
            answer=answer,
            supporting_facts=facts,
            average_confidence=avg_conf,
            n_ontology_facts=len(facts),
            grounding_score=grounding,
        )
        return trace

    def build_batch(
        self,
        results: list[GenerationResult],
        contexts: Optional[list[Optional[SubgraphContext]]] = None,
    ) -> list[ExplanationTrace]:
        if contexts is None:
            contexts = [None] * len(results)
        return [self.build(r, ctx) for r, ctx in zip(results, contexts)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _match_facts(
        self, answer: str, context: SubgraphContext
    ) -> list[SupportingFact]:
        """Find which context concepts/relations are mentioned in the answer."""
        answer_lower = answer.lower()
        facts: list[SupportingFact] = []

        # Match concepts
        for c in context.concepts:
            name = c.get("name", "")
            if name.lower() in answer_lower:
                conf = c.get("provenance", {}).get("confidence", 1.0)
                facts.append(SupportingFact(
                    fact_type="concept",
                    description=f"{name} [{c.get('concept_class','?')}]",
                    confidence=conf,
                ))

        # Match relations
        for r in context.relations:
            subj = r.get("subject", "")
            obj  = r.get("object", "")
            pred = r.get("predicate", "")
            if subj.lower() in answer_lower or obj.lower() in answer_lower:
                conf = r.get("provenance", {}).get("confidence", 0.9)
                facts.append(SupportingFact(
                    fact_type="relation",
                    description=f"{subj} --[{pred}]--> {obj}",
                    confidence=conf,
                    predicate=pred,
                    subject=subj,
                    obj=obj,
                ))

        # Sort by confidence descending
        facts.sort(key=lambda x: x.confidence, reverse=True)
        return facts

    def _grounding_score(self, answer: str) -> float:
        """
        Fraction of capitalized named entities in the answer that exist in the ontology.
        """
        candidates = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", answer)
        if not candidates:
            return 1.0  # no entities → assume fully grounded
        grounded = sum(1 for e in candidates if self.graph.get_concept(e))
        return grounded / len(candidates)
