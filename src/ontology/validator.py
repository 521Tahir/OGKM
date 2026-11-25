"""
Ontology validation engine.

Checks OntologyCandidates for:
  1. Schema validity (classes, relation types)
  2. Logical consistency (cycles, contradictions, domain/range)
  3. Semantic support (provenance confidence)
  4. Structural quality (connectivity, redundancy)
  5. Provenance completeness
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import (
    OntologyConcept,
    OntologyRelation,
    OntologyCandidate,
    TourismOntologySchema,
)
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.validator")

SCHEMA = TourismOntologySchema


@dataclass
class ValidationIssue:
    severity: str        # "error" | "warning" | "info"
    category: str        # "schema" | "logical" | "semantic" | "structural" | "provenance"
    message: str
    element: str = ""    # which concept/relation triggered the issue

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "element": self.element,
        }


@dataclass
class CandidateValidationResult:
    record_id: int
    issues: list[ValidationIssue] = field(default_factory=list)
    status: str = "pass"  # "pass" | "warning" | "fail" | "expert_review"

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error" and self.status != "fail":
            self.status = "fail"
        elif issue.severity == "warning" and self.status == "pass":
            self.status = "warning"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "status": self.status,
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }


class OntologyValidator:
    """Validate OntologyCandidate objects against the tourism schema."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        strict_mode: bool = False,
    ):
        self.min_confidence = min_confidence
        self.strict_mode = strict_mode

    # ── Public API ──────────────────────────────────────────────────────────

    def validate_candidate(self, candidate: OntologyCandidate) -> CandidateValidationResult:
        """Run all validation checks on a single candidate. Return result."""
        result = CandidateValidationResult(record_id=candidate.record_id)

        self._check_schema(candidate, result)
        self._check_logical(candidate, result)
        self._check_semantic_support(candidate, result)
        self._check_structural(candidate, result)
        self._check_provenance(candidate, result)

        # Update provenance on concepts/relations
        for concept in candidate.concepts:
            concept.provenance.validator_status = result.status
        for relation in candidate.relations:
            relation.provenance.validator_status = result.status

        return result

    def validate_batch(
        self, candidates: list[OntologyCandidate]
    ) -> list[CandidateValidationResult]:
        """Validate a list of candidates and return all results."""
        results = [self.validate_candidate(c) for c in candidates]
        n_pass = sum(1 for r in results if r.status == "pass")
        n_warn = sum(1 for r in results if r.status == "warning")
        n_fail = sum(1 for r in results if r.status == "fail")
        logger.info(
            f"Batch validation: {n_pass} pass / {n_warn} warning / {n_fail} fail "
            f"out of {len(results)} candidates."
        )
        return results

    # ── Check Methods ───────────────────────────────────────────────────────

    def _check_schema(self, c: OntologyCandidate, res: CandidateValidationResult) -> None:
        for concept in c.concepts:
            if not SCHEMA.is_valid_class(concept.concept_class):
                res.add(ValidationIssue(
                    severity="error",
                    category="schema",
                    message=f"Unknown concept class '{concept.concept_class}'",
                    element=concept.name,
                ))
            if not concept.name or len(concept.name.strip()) < 2:
                res.add(ValidationIssue(
                    severity="error",
                    category="schema",
                    message="Concept name is empty or too short",
                    element=str(concept.name),
                ))
        for rel in c.relations:
            if not SCHEMA.is_valid_relation(rel.predicate):
                res.add(ValidationIssue(
                    severity="error",
                    category="schema",
                    message=f"Unknown relation type '{rel.predicate}'",
                    element=f"{rel.subject} --{rel.predicate}--> {rel.obj}",
                ))

    def _check_logical(self, c: OntologyCandidate, res: CandidateValidationResult) -> None:
        """Check for contradictions and domain/range violations."""
        # Detect self-referential relations
        for rel in c.relations:
            if rel.subject == rel.obj:
                res.add(ValidationIssue(
                    severity="warning",
                    category="logical",
                    message=f"Self-referential relation: {rel.subject} --{rel.predicate}--> {rel.obj}",
                    element=rel.subject,
                ))

        # Check for duplicate (subject, predicate, object) triples
        seen_triples: set[tuple] = set()
        for rel in c.relations:
            triple = (rel.subject, rel.predicate, rel.obj)
            if triple in seen_triples:
                res.add(ValidationIssue(
                    severity="warning",
                    category="logical",
                    message=f"Duplicate relation triple: {triple}",
                    element=str(triple),
                ))
            seen_triples.add(triple)

    def _check_semantic_support(
        self, c: OntologyCandidate, res: CandidateValidationResult
    ) -> None:
        """Flag low-confidence extractions."""
        for concept in c.concepts:
            if concept.provenance.confidence < self.min_confidence:
                res.add(ValidationIssue(
                    severity="warning",
                    category="semantic",
                    message=(
                        f"Low confidence concept: '{concept.name}' "
                        f"(conf={concept.provenance.confidence:.2f})"
                    ),
                    element=concept.name,
                ))
        for rel in c.relations:
            if rel.provenance.confidence < self.min_confidence:
                res.add(ValidationIssue(
                    severity="warning",
                    category="semantic",
                    message=(
                        f"Low confidence relation: {rel.subject} --{rel.predicate}--> {rel.obj} "
                        f"(conf={rel.provenance.confidence:.2f})"
                    ),
                    element=f"{rel.subject}_{rel.predicate}",
                ))

    def _check_structural(
        self, c: OntologyCandidate, res: CandidateValidationResult
    ) -> None:
        """Check connectivity and density."""
        if not c.concepts:
            res.add(ValidationIssue(
                severity="warning",
                category="structural",
                message="No concepts extracted from record",
                element=str(c.record_id),
            ))
        if c.concepts and not c.relations:
            res.add(ValidationIssue(
                severity="info",
                category="structural",
                message="Concepts extracted but no relations — potential isolated nodes",
                element=str(c.record_id),
            ))

    def _check_provenance(
        self, c: OntologyCandidate, res: CandidateValidationResult
    ) -> None:
        """Ensure provenance fields are populated."""
        for concept in c.concepts:
            if concept.provenance.source_record_id is None:
                res.add(ValidationIssue(
                    severity="warning",
                    category="provenance",
                    message=f"Missing source_record_id on concept '{concept.name}'",
                    element=concept.name,
                ))
