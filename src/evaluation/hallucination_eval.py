"""
Hallucination Evaluation — comprehensive hallucination detection and analysis.

Goes beyond simple entity checking to provide:
  1. Entity-level hallucination rate (named entities not in ontology)
  2. Relation-level hallucination (stated relations contradicted by ontology)
  3. Numeric/factual hallucination (numbers/dates not in context)
  4. Type confusion (entities misclassified — e.g., "Beach" called a "Restaurant")
  5. Aggregated hallucination scores with confidence calibration

Paper metric: Table 6 — Hallucination Rate across 5 system variants
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.hallucination")


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationInstance:
    """A single hallucination event detected in a response."""
    hallucination_type: str   # "entity" | "relation" | "numeric" | "type_confusion"
    severity: str             # "high" | "medium" | "low"
    span: str                 # the text span that is hallucinated
    reason: str               # explanation of why it's flagged

    def to_dict(self) -> dict:
        return {
            "type":     self.hallucination_type,
            "severity": self.severity,
            "span":     self.span,
            "reason":   self.reason,
        }


@dataclass
class HallucinationReport:
    """Full hallucination analysis for one generated answer."""
    query:       str
    answer:      str
    mode:        str
    instances:   list[HallucinationInstance] = field(default_factory=list)

    @property
    def is_hallucinated(self) -> bool:
        return any(i.severity in ("high", "medium") for i in self.instances)

    @property
    def hallucination_score(self) -> float:
        """0 = no hallucination, 1 = severe hallucination."""
        if not self.instances:
            return 0.0
        weights = {"high": 1.0, "medium": 0.5, "low": 0.1}
        total = sum(weights.get(i.severity, 0.1) for i in self.instances)
        return min(1.0, total / max(1, len(self.instances)))

    @property
    def entity_hallucinations(self) -> list[HallucinationInstance]:
        return [i for i in self.instances if i.hallucination_type == "entity"]

    def to_dict(self) -> dict:
        return {
            "query":               self.query[:100],
            "mode":                self.mode,
            "is_hallucinated":     self.is_hallucinated,
            "hallucination_score": round(self.hallucination_score, 3),
            "n_instances":         len(self.instances),
            "instances":           [i.to_dict() for i in self.instances],
        }


@dataclass
class HallucinationSummary:
    """Aggregate hallucination statistics across all responses."""
    mode:                   str
    n_responses:            int
    hallucination_rate:     float   # fraction with is_hallucinated=True
    avg_hallucination_score: float
    entity_halluc_rate:     float
    relation_halluc_rate:   float
    numeric_halluc_rate:    float
    type_confusion_rate:    float

    def to_dict(self) -> dict:
        return {
            "mode":                      self.mode,
            "n_responses":               self.n_responses,
            "hallucination_rate":        round(self.hallucination_rate, 3),
            "avg_hallucination_score":   round(self.avg_hallucination_score, 3),
            "entity_hallucination_rate": round(self.entity_halluc_rate, 3),
            "relation_hallucination_rate": round(self.relation_halluc_rate, 3),
            "numeric_hallucination_rate":  round(self.numeric_halluc_rate, 3),
            "type_confusion_rate":         round(self.type_confusion_rate, 3),
        }


# ── Regex patterns ─────────────────────────────────────────────────────────────

_NAMED_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")

_NUMERIC_RE = re.compile(
    r"\b(\d{1,4}(?:[,\.]\d+)?)\s*"
    r"(km|meters?|m|km²|acres?|won|usd|\$|€|hours?|mins?|minutes?|days?|years?|%)\b",
    re.I
)

_RELATION_CLAIM_RE = re.compile(
    r"\b(\w[\w\s]{1,30})\s+(is located|is found|is part of|belongs to|contains|includes|"
    r"is near|is in|is a type of)\s+(\w[\w\s]{1,30})\b",
    re.I
)

_FALSE_POSITIVE_CAPS = {
    "The", "A", "An", "In", "On", "At", "For", "And", "Or", "But", "If",
    "Of", "To", "By", "With", "From", "This", "That", "It", "I", "We",
    "You", "They", "He", "She", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November",
    "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "North", "South", "East", "West",
}


# ── Main evaluator ─────────────────────────────────────────────────────────────

class HallucinationEvaluator:
    """
    Multi-dimensional hallucination evaluator.

    Parameters
    ----------
    ontology_concepts : list[dict]
        All concept dicts from the ontology (need 'name' and 'class' keys).
    ontology_relations : list[dict]
        All relation dicts from the ontology (need 'subject', 'predicate', 'object').
    """

    def __init__(
        self,
        ontology_concepts: Optional[list[dict]] = None,
        ontology_relations: Optional[list[dict]] = None,
    ):
        self._concepts = ontology_concepts or []
        self._relations = ontology_relations or []

        # Name → class lookup
        self._name_to_class: dict[str, str] = {
            c.get("name", "").lower(): c.get("class", "")
            for c in self._concepts if c.get("name")
        }

        # Relation triples set (subject_lower, predicate, object_lower)
        self._relation_set: set[tuple[str, str, str]] = {
            (
                r.get("subject", r.get("head", "")).lower(),
                r.get("predicate", r.get("relation", "")),
                r.get("object", r.get("tail", "")).lower(),
            )
            for r in self._relations
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        answer: str,
        mode: str,
        context_concepts: Optional[list[dict]] = None,
        context_relations: Optional[list[dict]] = None,
    ) -> HallucinationReport:
        """
        Run all hallucination checks on the generated answer.

        Parameters
        ----------
        query : str        The original query.
        answer : str       The generated response.
        mode : str         Generation mode label.
        context_concepts : list[dict], optional
            Concepts that were in the retrieval context (used to whitelist).
        context_relations : list[dict], optional
            Relations that were in the context.
        """
        # Whitelist names from context (allowed in answer)
        whitelist: set[str] = set()
        for c in (context_concepts or []):
            name = c.get("name", "")
            if name:
                whitelist.add(name.lower())
        for r in (context_relations or []):
            whitelist.add(r.get("subject", "").lower())
            whitelist.add(r.get("object", "").lower())

        instances: list[HallucinationInstance] = []
        instances.extend(self._check_entity_hallucinations(answer, whitelist))
        instances.extend(self._check_numeric_hallucinations(answer, context_concepts or []))
        instances.extend(self._check_type_confusion(answer))
        # Relation check is expensive; only do it if context provided
        if context_relations:
            instances.extend(self._check_relation_hallucinations(answer, context_relations))

        return HallucinationReport(
            query=query,
            answer=answer,
            mode=mode,
            instances=instances,
        )

    def evaluate_batch(
        self,
        queries: list[str],
        answers: list[str],
        mode: str,
        contexts: Optional[list[dict]] = None,
    ) -> list[HallucinationReport]:
        return [
            self.evaluate(q, a, mode)
            for q, a in zip(queries, answers)
        ]

    def summarize(
        self,
        reports: list[HallucinationReport],
        mode: str,
    ) -> HallucinationSummary:
        n = len(reports)
        if n == 0:
            return HallucinationSummary(
                mode=mode, n_responses=0,
                hallucination_rate=0, avg_hallucination_score=0,
                entity_halluc_rate=0, relation_halluc_rate=0,
                numeric_halluc_rate=0, type_confusion_rate=0,
            )

        def _rate(check_type: Optional[str] = None) -> float:
            if check_type is None:
                return sum(1 for r in reports if r.is_hallucinated) / n
            return sum(
                1 for r in reports
                if any(i.hallucination_type == check_type for i in r.instances)
            ) / n

        return HallucinationSummary(
            mode=mode,
            n_responses=n,
            hallucination_rate=_rate(),
            avg_hallucination_score=sum(r.hallucination_score for r in reports) / n,
            entity_halluc_rate=_rate("entity"),
            relation_halluc_rate=_rate("relation"),
            numeric_halluc_rate=_rate("numeric"),
            type_confusion_rate=_rate("type_confusion"),
        )

    # ── Check implementations ──────────────────────────────────────────────────

    def _check_entity_hallucinations(
        self, answer: str, whitelist: set[str]
    ) -> list[HallucinationInstance]:
        instances = []
        for m in _NAMED_ENTITY_RE.finditer(answer):
            span = m.group(1)
            if span in _FALSE_POSITIVE_CAPS:
                continue
            if len(span) < 3:
                continue
            lower = span.lower()
            # Skip if in whitelist or in full ontology
            if lower in whitelist or lower in self._name_to_class:
                continue
            # Check if partial match exists (substring of known concept)
            partial = any(lower in k or k in lower for k in self._name_to_class)
            severity = "low" if partial else "medium"
            instances.append(HallucinationInstance(
                hallucination_type="entity",
                severity=severity,
                span=span,
                reason=f"'{span}' not found in ontology" + (" (partial match)" if partial else ""),
            ))
        return instances

    def _check_numeric_hallucinations(
        self, answer: str, context_concepts: list[dict]
    ) -> list[HallucinationInstance]:
        """Flag specific numeric claims not mentioned in context."""
        if not context_concepts:
            return []  # can't check without context

        context_text = " ".join(
            str(v) for c in context_concepts for v in c.values()
        ).lower()

        instances = []
        for m in _NUMERIC_RE.finditer(answer):
            num_str = m.group(0).lower()
            if num_str not in context_text:
                instances.append(HallucinationInstance(
                    hallucination_type="numeric",
                    severity="low",
                    span=m.group(0),
                    reason=f"Numeric claim '{m.group(0)}' not found in retrieval context.",
                ))
        return instances

    def _check_relation_hallucinations(
        self, answer: str, context_relations: list[dict]
    ) -> list[HallucinationInstance]:
        """Detect stated relations that contradict the retrieved context."""
        instances = []
        context_triples = {
            (
                r.get("subject", "").lower(),
                r.get("predicate", ""),
                r.get("object", "").lower(),
            )
            for r in context_relations
        }

        for m in _RELATION_CLAIM_RE.finditer(answer):
            subj  = m.group(1).strip().lower()
            pred  = m.group(2).strip()
            obj   = m.group(3).strip().lower()

            # Check if any context triple partially confirms this claim
            confirmed = any(
                (subj in s or s in subj) and (obj in o or o in obj)
                for s, _, o in context_triples
            )
            if not confirmed and subj in self._name_to_class:
                instances.append(HallucinationInstance(
                    hallucination_type="relation",
                    severity="medium",
                    span=m.group(0),
                    reason=f"Relation claim not found in context: '{subj}' {pred} '{obj}'",
                ))
        return instances

    def _check_type_confusion(self, answer: str) -> list[HallucinationInstance]:
        """
        Detect entities referred to with wrong class labels.
        E.g., if 'Manjanggul' (Cave) is called a 'beach' in the answer.
        """
        instances = []
        answer_lower = answer.lower()

        # Check each ontology concept name appearing in the answer
        for name_lower, cls in self._name_to_class.items():
            if name_lower not in answer_lower or not cls:
                continue
            # Check for contradictory class label nearby
            cls_lower = cls.lower()
            # Simple check: concept appears near a different class label
            contradictions: dict[str, list[str]] = {
                "beach":      ["restaurant", "museum", "mountain", "cave"],
                "mountain":   ["beach", "restaurant", "market", "hotel"],
                "restaurant": ["beach", "mountain", "cave", "waterfall"],
                "museum":     ["beach", "restaurant", "trail", "hotel"],
                "cave":       ["beach", "restaurant", "mountain", "park"],
            }
            wrong_labels = contradictions.get(cls_lower, [])
            for wrong in wrong_labels:
                # Find the concept mention and check if wrong label is nearby (±80 chars)
                for m in re.finditer(re.escape(name_lower), answer_lower):
                    window = answer_lower[max(0, m.start()-80): m.end()+80]
                    if wrong in window:
                        instances.append(HallucinationInstance(
                            hallucination_type="type_confusion",
                            severity="medium",
                            span=name_lower,
                            reason=f"'{name_lower}' (class={cls}) described as '{wrong}' near mention.",
                        ))
                        break  # one per concept
        return instances
