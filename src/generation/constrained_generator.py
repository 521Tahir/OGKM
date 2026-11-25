"""
Constrained Generator — produces answers grounded in ontology context.

Modes:
  no_rag        : LLM answers from parametric knowledge only
  doc_rag       : LLM answers from retrieved text passages
  onto_rag      : LLM answers from ontology subgraph context
  constrained   : onto_rag + post-generation constraint checks via ResponseValidator
                  + uncertainty scoring via UncertaintyHandler

All modes return a GenerationResult with the answer, source context, and
any constraint violations detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

from ..retrieval.hybrid_retriever import HybridContext
from ..retrieval.query_understanding import QueryUnderstanding, QueryIntent
from ..retrieval.context_assembler import ContextAssembler
from ..retrieval.reranker import Reranker
from ..graph.graph_store import TourismGraphStore
from ..utils.logging_utils import get_logger
from .response_validator import ResponseValidator
from .uncertainty_handler import UncertaintyHandler

logger = get_logger("generation.constrained")


@dataclass
class ConstraintViolation:
    """A detected constraint violation in a generated answer."""
    violation_type: str   # "hallucinated_entity" | "invalid_relation" | "contradicts_ontology"
    detail: str
    evidence: str = ""    # the span in the answer that triggered this

    def to_dict(self) -> dict:
        return {
            "type":     self.violation_type,
            "detail":   self.detail,
            "evidence": self.evidence,
        }


@dataclass
class GenerationResult:
    """Result from constrained generation."""
    query:       str
    answer:      str
    mode:        str   # "no_rag" | "doc_rag" | "onto_rag" | "constrained"
    context_used: str  # the context string passed to the LLM
    violations:  list[ConstraintViolation] = field(default_factory=list)
    is_grounded: bool = True
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query":         self.query,
            "answer":        self.answer,
            "mode":          self.mode,
            "is_grounded":   self.is_grounded,
            "violations":    [v.to_dict() for v in self.violations],
            "n_violations":  len(self.violations),
            "metadata":      self.metadata,
        }


class ConstrainedGenerator:
    """
    Generates answers under four conditions for ablation study.

    Usage:
        gen = ConstrainedGenerator(llm_client, graph)
        result = gen.generate(query, context, mode="constrained")
    """

    def __init__(
        self,
        client: Optional[Any],       # LLMClient instance
        graph: TourismGraphStore,
        prompt_manager: Optional[Any] = None,
        ontology_names: Optional[set[str]] = None,
        rules: Optional[list[dict]] = None,
    ) -> None:
        self.client    = client
        self.graph     = graph
        self.pm        = prompt_manager
        self._qu       = QueryUnderstanding()
        self._assembler = ContextAssembler()
        self._reranker  = Reranker()
        self._validator = ResponseValidator(
            ontology_names=ontology_names or set(graph._concepts.keys()),
            rules=rules or [],
            fail_on_hallucination=False,
            max_ungrounded_fraction=0.6,
        )
        self._uncertainty = UncertaintyHandler()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: Optional[HybridContext] = None,
        mode: str = "onto_rag",
        query_intent: Optional[QueryIntent] = None,
    ) -> GenerationResult:
        """Generate an answer for the given query and mode."""
        # Parse query intent for smarter context assembly
        if query_intent is None:
            query_intent = self._qu.parse(query)

        # Rerank + assemble context
        context_text = self._build_context_text(context, mode, query_intent)

        # Optionally inject calibration instructions for constrained mode
        prompt_override = None
        if mode == "constrained":
            sys_prompt = "You are a tourism expert."
            sys_prompt = self._uncertainty.inject_calibration_prompt(sys_prompt)
            prompt_override = sys_prompt

        answer = self._call_llm(query, context_text, mode, system_prompt=prompt_override)

        # Post-generation validation
        violations: list[ConstraintViolation] = []
        uncertainty_report = None
        if mode in ("constrained", "onto_rag"):
            validation = self._validator.validate(answer, query=query)
            for issue in validation.issues:
                if issue.check == "hallucination":
                    violations.append(ConstraintViolation(
                        violation_type="hallucinated_entity",
                        detail=issue.message,
                        evidence=issue.span or "",
                    ))
                elif issue.check == "constraint":
                    violations.append(ConstraintViolation(
                        violation_type="constraint_violation",
                        detail=issue.message,
                        evidence=issue.span or "",
                    ))
        else:
            violations = self._check_constraints(answer)

        if mode == "constrained":
            uncertainty_report = self._uncertainty.analyze(answer)

        unc_score = uncertainty_report.confidence if uncertainty_report else 1.0
        n_concepts = 0
        n_relations = 0
        if context and context.subgraph:
            n_concepts  = len(getattr(context.subgraph, "concepts", []) or [])
            n_relations = len(getattr(context.subgraph, "relations", []) or [])

        return GenerationResult(
            query=query,
            answer=answer,
            mode=mode,
            context_used=context_text[:500] if context_text else "",
            violations=violations,
            is_grounded=len(violations) == 0,
            metadata={
                "n_concepts_in_context": n_concepts,
                "n_relations_in_context": n_relations,
                "uncertainty_confidence": unc_score,
                "task_type": query_intent.task_type if query_intent else "general",
                "themes": query_intent.constraints.themes if query_intent else [],
            },
        )

    def generate_batch(
        self,
        queries: list[str],
        contexts: Optional[list[HybridContext]] = None,
        mode: str = "onto_rag",
    ) -> list[GenerationResult]:
        if contexts is None:
            contexts = [None] * len(queries)
        return [
            self.generate(q, ctx, mode=mode)
            for q, ctx in zip(queries, contexts)
        ]

    # ── Context building ──────────────────────────────────────────────────────

    def _build_context_text(
        self,
        context: Optional[HybridContext],
        mode: str,
        query_intent: Optional[QueryIntent] = None,
    ) -> str:
        if mode == "no_rag" or context is None:
            return ""

        # Rerank subgraph first
        if context.subgraph and (
            getattr(context.subgraph, "concepts", None) or
            getattr(context.subgraph, "relations", None)
        ):
            ranked = self._reranker.rerank(context.subgraph, query_intent)
            # Build a temporary proxy so assembler sees the reranked lists
            class _RankedProxy:
                def __init__(self, r, orig):
                    self.concepts  = r.top_concepts(20)
                    self.relations = r.top_relations(30)
                    self.seeds     = r.seed_concepts
                    self.query     = orig.query if hasattr(orig, "query") else ""
            ctx_proxy = type("HybridProxy", (), {
                "subgraph": _RankedProxy(ranked, context.subgraph),
                "text_passages": getattr(context, "text_passages", []),
                "mode": mode,
            })()
        else:
            ctx_proxy = context

        assembled = self._assembler.assemble(ctx_proxy, query_intent)
        return assembled.text

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        query: str,
        context_text: str,
        mode: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build prompt and call LLM. Falls back to template answer if LLM unavailable."""
        prompt = self._build_prompt(query, context_text, mode, system_prompt)

        if self.client is not None and getattr(self.client, "is_available", False):
            try:
                response = self.client.call(prompt)
                return response.strip()
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")

        # Offline fallback: template-based answer from context
        return self._offline_answer(query, context_text)

    def _build_prompt(
        self,
        query: str,
        context: str,
        mode: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build the generation prompt."""
        sys = system_prompt or "You are a tourism expert."

        if mode == "no_rag":
            return f"{sys} Answer the following question accurately.\n\nQuestion: {query}\n\nAnswer:"

        constraint_note = (
            "\nIMPORTANT: Only state facts that are explicitly present in the context above. "
            "Do not introduce destinations, places, or attributes not mentioned in the context."
            if mode == "constrained" else ""
        )

        return (
            f"{sys} Use the ontology context below to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {query}{constraint_note}\n\nAnswer:"
        )

    def _offline_answer(self, query: str, context_text: str) -> str:
        """Generate a simple template answer from context when LLM is not available."""
        if not context_text or context_text == "[No context retrieved]":
            return f"[Offline] No ontology context available for: {query}"
        # Extract a few concept names from the context for a basic answer
        import re
        concepts = re.findall(r"- (.+?) \[", context_text)
        if concepts:
            top = ", ".join(concepts[:5])
            return f"[Offline] Based on ontology context, relevant entities include: {top}."
        return f"[Offline] Ontology context retrieved but LLM unavailable."

    # ── Constraint checking ───────────────────────────────────────────────────

    def _check_constraints(self, answer: str) -> list[ConstraintViolation]:
        """
        Post-generation constraint checks:
          - Detect named entities in the answer that are NOT in the ontology
            (potential hallucinations).
        """
        violations: list[ConstraintViolation] = []
        import re

        # Simple NER-like heuristic: capitalized multi-word spans
        candidate_entities = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", answer)
        seen: set[str] = set()
        for entity in candidate_entities:
            if entity in seen or len(entity) < 3:
                continue
            seen.add(entity)
            # Check if entity exists in ontology
            if not self.graph.get_concept(entity):
                # Check if it's a substring of any known concept
                partial = any(
                    entity.lower() in name.lower()
                    for name in self.graph._concepts
                )
                if not partial:
                    violations.append(ConstraintViolation(
                        violation_type="hallucinated_entity",
                        detail=f"'{entity}' not found in ontology",
                        evidence=entity,
                    ))

        return violations
