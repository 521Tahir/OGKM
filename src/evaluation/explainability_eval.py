"""
Explainability Evaluation — measures how well the system explains its answers.

Metrics (addresses RQ6 from the paper):
  1. evidence_trace_coverage   — fraction of context facts cited in the answer
  2. explanation_faithfulness  — cited facts not contradicted by the answer
  3. path_interpretability     — clarity of reasoning chain (hop count, labels)
  4. source_attribution_rate   — fraction of claims with traceable ontology source
  5. grounding_precision       — fraction of cited entities found in ontology

Paper table: Table 7 — Explainability Metrics
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.explainability")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExplainabilityReport:
    """Explainability metrics for a single generated response."""
    query_id:                  int
    mode:                      str
    evidence_trace_coverage:   float   # fraction of context facts cited
    explanation_faithfulness:  float   # 0-1; cited facts not contradicted
    path_interpretability:     float   # 0-1; reasoning chain clarity
    source_attribution_rate:   float   # fraction of claims with ontology source
    grounding_precision:       float   # fraction of cited entities in ontology
    n_hops:                    int     # reasoning chain length
    n_cited_concepts:          int     # distinct ontology concepts cited

    def to_dict(self) -> dict:
        return {
            "query_id":                 self.query_id,
            "mode":                     self.mode,
            "evidence_trace_coverage":  round(self.evidence_trace_coverage, 3),
            "explanation_faithfulness": round(self.explanation_faithfulness, 3),
            "path_interpretability":    round(self.path_interpretability, 3),
            "source_attribution_rate":  round(self.source_attribution_rate, 3),
            "grounding_precision":      round(self.grounding_precision, 3),
            "n_hops":                   self.n_hops,
            "n_cited_concepts":         self.n_cited_concepts,
        }


@dataclass
class ExplainabilityAggregate:
    """Aggregated explainability metrics across all responses for one mode."""
    mode:                          str
    n_responses:                   int
    avg_evidence_coverage:         float
    avg_faithfulness:              float
    avg_path_interpretability:     float
    avg_source_attribution:        float
    avg_grounding_precision:       float
    avg_n_hops:                    float
    avg_cited_concepts:            float

    def to_dict(self) -> dict:
        return {
            "mode":                        self.mode,
            "n_responses":                 self.n_responses,
            "avg_evidence_coverage":       round(self.avg_evidence_coverage, 3),
            "avg_faithfulness":            round(self.avg_faithfulness, 3),
            "avg_path_interpretability":   round(self.avg_path_interpretability, 3),
            "avg_source_attribution":      round(self.avg_source_attribution, 3),
            "avg_grounding_precision":     round(self.avg_grounding_precision, 3),
            "avg_n_hops":                  round(self.avg_n_hops, 2),
            "avg_cited_concepts":          round(self.avg_cited_concepts, 1),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_CAPS_RE      = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")
_CONTRAST_RE  = re.compile(
    r"\b(however|but|although|despite|whereas|on the other hand|in contrast)\b", re.I
)
_STEP_RE      = re.compile(
    r"\b(first|second|third|then|next|finally|because|therefore|since|thus|"
    r"step \d|reason|evidence|based on|according to)\b", re.I
)
_FALSE_CAPS   = {
    "The", "A", "An", "In", "On", "At", "For", "And", "Or", "But", "If",
    "Of", "To", "By", "With", "From", "This", "That", "It", "I", "We",
    "You", "They", "He", "She",
}


class ExplainabilityEvaluator:
    """
    Compute explainability metrics for generated answers.

    Parameters
    ----------
    ontology_concepts : list[dict]
        All concept dicts from the ontology (need 'name' key).
    ontology_relations : list[dict]
        All relation dicts (need 'subject', 'predicate', 'object').
    """

    def __init__(
        self,
        ontology_concepts: Optional[list[dict]] = None,
        ontology_relations: Optional[list[dict]] = None,
    ):
        self._concepts  = ontology_concepts or []
        self._relations = ontology_relations or []

        # Lowercase name set for fast lookup
        self._concept_names: set[str] = {
            c.get("name", "").lower()
            for c in self._concepts if c.get("name")
        }

        # Relation triple set
        self._relation_set: set[tuple] = {
            (
                r.get("subject", "").lower(),
                r.get("predicate", ""),
                r.get("object", "").lower(),
            )
            for r in self._relations
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query_id: int,
        query: str,
        answer: str,
        mode: str,
        context_concepts: Optional[list[dict]] = None,
        context_relations: Optional[list[dict]] = None,
        explanation_trace: Optional[dict] = None,
    ) -> ExplainabilityReport:
        """
        Compute all explainability metrics for one answer.

        Parameters
        ----------
        query_id : int
        query : str
        answer : str
        mode : str
        context_concepts : list[dict]
            Concepts in retrieval context (for coverage computation).
        context_relations : list[dict]
            Relations in retrieval context.
        explanation_trace : dict, optional
            ExplanationBuilder output dict with 'cited_concepts', 'reasoning_path', etc.
        """
        context_concepts  = context_concepts or []
        context_relations = context_relations or []

        coverage      = self.evidence_trace_coverage(answer, context_concepts, context_relations)
        faithfulness  = self.explanation_faithfulness(answer, context_concepts, context_relations)
        interpretability = self.path_interpretability(answer, explanation_trace)
        attribution   = self.source_attribution_rate(answer, context_concepts)
        grounding     = self.grounding_precision(answer)
        n_hops        = self._count_hops(explanation_trace)
        cited         = self._count_cited_concepts(answer)

        return ExplainabilityReport(
            query_id=query_id,
            mode=mode,
            evidence_trace_coverage=coverage,
            explanation_faithfulness=faithfulness,
            path_interpretability=interpretability,
            source_attribution_rate=attribution,
            grounding_precision=grounding,
            n_hops=n_hops,
            n_cited_concepts=cited,
        )

    def evaluate_batch(
        self,
        items: list[dict],
        mode: str,
    ) -> list[ExplainabilityReport]:
        """
        Score a batch of {query_id, query, answer, context_concepts?,
        context_relations?, explanation_trace?} dicts.
        """
        return [
            self.evaluate(
                query_id=item.get("query_id", i),
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                mode=mode,
                context_concepts=item.get("context_concepts"),
                context_relations=item.get("context_relations"),
                explanation_trace=item.get("explanation_trace"),
            )
            for i, item in enumerate(items)
        ]

    def aggregate(
        self,
        reports: list[ExplainabilityReport],
        mode: str,
    ) -> ExplainabilityAggregate:
        """Aggregate per-query reports into mode-level summary."""
        n = len(reports)
        if n == 0:
            return ExplainabilityAggregate(
                mode=mode, n_responses=0,
                avg_evidence_coverage=0, avg_faithfulness=0,
                avg_path_interpretability=0, avg_source_attribution=0,
                avg_grounding_precision=0, avg_n_hops=0, avg_cited_concepts=0,
            )
        return ExplainabilityAggregate(
            mode=mode,
            n_responses=n,
            avg_evidence_coverage=mean(r.evidence_trace_coverage for r in reports),
            avg_faithfulness=mean(r.explanation_faithfulness for r in reports),
            avg_path_interpretability=mean(r.path_interpretability for r in reports),
            avg_source_attribution=mean(r.source_attribution_rate for r in reports),
            avg_grounding_precision=mean(r.grounding_precision for r in reports),
            avg_n_hops=mean(r.n_hops for r in reports),
            avg_cited_concepts=mean(r.n_cited_concepts for r in reports),
        )

    def compare_modes(
        self,
        mode_results: dict[str, ExplainabilityAggregate],
    ) -> str:
        """Return a markdown table comparing all modes."""
        lines = [
            "| Mode | Coverage | Faithful | Interpret | Attribution | Grounding | Hops |",
            "|------|:--------:|:--------:|:---------:|:-----------:|:---------:|:----:|",
        ]
        for mode, agg in mode_results.items():
            lines.append(
                f"| {mode} "
                f"| {agg.avg_evidence_coverage:.3f} "
                f"| {agg.avg_faithfulness:.3f} "
                f"| {agg.avg_path_interpretability:.3f} "
                f"| {agg.avg_source_attribution:.3f} "
                f"| {agg.avg_grounding_precision:.3f} "
                f"| {agg.avg_n_hops:.1f} |"
            )
        return "\n".join(lines)

    # ── Individual metrics ────────────────────────────────────────────────────

    def evidence_trace_coverage(
        self,
        answer: str,
        context_concepts: list[dict],
        context_relations: list[dict],
    ) -> float:
        """
        Fraction of context concepts/relations whose names appear in the answer.
        Returns 1.0 if no context provided.
        """
        if not context_concepts and not context_relations:
            return 1.0

        answer_lower = answer.lower()
        total = 0
        found = 0

        for c in context_concepts:
            name = c.get("name", "").lower()
            if name:
                total += 1
                if name in answer_lower:
                    found += 1

        # Relations: check if subject or object appears in answer
        for r in context_relations:
            subj = r.get("subject", "").lower()
            obj  = r.get("object", "").lower()
            if subj or obj:
                total += 1
                if (subj and subj in answer_lower) or (obj and obj in answer_lower):
                    found += 1

        return found / total if total > 0 else 1.0

    def explanation_faithfulness(
        self,
        answer: str,
        context_concepts: list[dict],
        context_relations: list[dict],
    ) -> float:
        """
        Fraction of context facts mentioned in the answer that are not
        contradicted by surrounding text (no negation or contradiction nearby).

        Score: 1.0 = all mentions faithful, 0.0 = all contradicted.
        """
        if not context_concepts:
            return 1.0

        answer_lower = answer.lower()
        _NEGATION_RE = re.compile(
            r"\b(not|no|never|incorrect|wrong|false|doesn't|isn't|aren't|"
            r"wasn't|weren't|doesn't exist|does not exist|cannot find|"
            r"unable to find|not found|not a)\b",
            re.I,
        )

        total = 0
        faithful = 0

        for c in context_concepts:
            name = c.get("name", "").lower()
            if not name or name not in answer_lower:
                continue
            total += 1
            # Check for negation within 60 chars before or after the mention
            for m in re.finditer(re.escape(name), answer_lower):
                window = answer_lower[max(0, m.start() - 60): m.end() + 60]
                if not _NEGATION_RE.search(window):
                    faithful += 1
                    break
            else:
                pass  # name in answer but loop didn't break = negated

        return faithful / total if total > 0 else 1.0

    def path_interpretability(
        self,
        answer: str,
        explanation_trace: Optional[dict] = None,
    ) -> float:
        """
        Heuristic score [0,1] for how clearly the answer shows its reasoning.

        - Checks for transition/reasoning words (because, therefore, since, first, …)
        - Checks for structured formatting (numbered lists, bullet points)
        - If explanation_trace provided, uses its reasoning_path length as signal
        """
        score = 0.0
        answer_lower = answer.lower()

        # Reasoning connectors
        step_matches = len(_STEP_RE.findall(answer_lower))
        score += min(0.4, step_matches * 0.08)

        # Structured output markers
        if re.search(r"(\n\s*[-•*]|\n\s*\d+\.)", answer):
            score += 0.2

        # Contrast/comparison (shows nuanced reasoning)
        if _CONTRAST_RE.search(answer_lower):
            score += 0.1

        # Length signal (longer structured answers tend to be more interpretable)
        words = len(answer.split())
        score += min(0.2, words / 500)

        # Bonus if explanation trace has reasoning path
        if explanation_trace:
            path = explanation_trace.get("reasoning_path", [])
            if isinstance(path, list) and len(path) >= 2:
                score += 0.1

        return min(1.0, score)

    def source_attribution_rate(
        self,
        answer: str,
        context_concepts: list[dict],
    ) -> float:
        """
        Fraction of named entities in the answer that can be traced to a
        context concept (i.e., the answer cites concepts from its context).
        Returns 1.0 if no context provided (can't check).
        """
        if not context_concepts:
            return 1.0

        answer_lower = answer.lower()
        # Extract named entities from answer
        spans = [
            m.group(1) for m in _CAPS_RE.finditer(answer)
            if m.group(1) not in _FALSE_CAPS and len(m.group(1)) >= 3
        ]
        if not spans:
            return 1.0

        context_names = {c.get("name", "").lower() for c in context_concepts}
        attributed = sum(
            1 for s in spans
            if s.lower() in context_names
            or any(s.lower() in k or k in s.lower() for k in context_names)
        )
        return attributed / len(spans)

    def grounding_precision(self, answer: str) -> float:
        """
        Fraction of capitalised named entities in the answer that appear
        in the full ontology (same logic as factual_correctness_score).
        Returns 1.0 if no ontology loaded.
        """
        if not self._concept_names:
            return 1.0

        spans = [
            m.group(1) for m in _CAPS_RE.finditer(answer)
            if m.group(1) not in _FALSE_CAPS and len(m.group(1)) >= 3
        ]
        if not spans:
            return 1.0

        grounded = sum(
            1 for s in spans
            if s.lower() in self._concept_names
            or any(s.lower() in k or k in s.lower() for k in self._concept_names)
        )
        return grounded / len(spans)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _count_hops(self, trace: Optional[dict]) -> int:
        """Number of reasoning hops from the explanation trace."""
        if not trace:
            return 0
        path = trace.get("reasoning_path", trace.get("evidence_chain", []))
        if isinstance(path, list):
            return len(path)
        return 0

    def _count_cited_concepts(self, answer: str) -> int:
        """Count distinct ontology concepts mentioned in the answer."""
        answer_lower = answer.lower()
        return sum(1 for name in self._concept_names if name and name in answer_lower)
