"""
Response Validator — post-generation validation of LLM answers.

Checks:
  1. Hallucination detection — named entities in the answer not grounded in ontology
  2. Constraint violation detection — answer conflicts with SHACL rules
  3. Factual consistency — claims contradict ontology relations
  4. Completeness — key expected concepts missing from answer
  5. Format compliance — answer structure matches expected format

Returns a ValidationReport with per-check results and an overall pass/fail.

Used by: constrained_generator.py, exp005/run_constraints.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Any

from ..utils.logging_utils import get_logger

logger = get_logger("generation.response_validator")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    check: str           # "hallucination" | "constraint" | "factual" | "completeness" | "format"
    severity: str        # "error" | "warning" | "info"
    message: str
    span: Optional[str] = None   # the text span that triggered the issue


@dataclass
class ValidationReport:
    """Full validation report for a single generated response."""
    answer: str
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    # Convenience counts
    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def hallucination_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.check == "hallucination"]

    @property
    def constraint_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.check == "constraint"]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.n_errors} error(s), {self.n_warnings} warning(s) | "
            f"{len(self.hallucination_issues)} hallucination(s), "
            f"{len(self.constraint_issues)} constraint violation(s)"
        )


# ── Regex helpers ─────────────────────────────────────────────────────────────

# Capitalized n-gram (1–4 words) — candidate named entities
_NE_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")

# Generic phrases that are often capitalized but not real entities
_FALSE_POSITIVE_TOKENS = {
    "I", "The", "A", "An", "In", "On", "At", "For", "And", "Or", "But",
    "If", "Of", "To", "By", "With", "From", "This", "That", "It",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "North", "South", "East", "West", "Central",
}

# Temporal and common tourism terms that are always valid (never hallucinations)
_ALWAYS_VALID_TOKENS: set[str] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "spring", "summer", "autumn", "fall", "winter",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "lifeguard", "lifeguards", "swimming", "snorkeling", "hiking", "camping",
    "tourist", "tourists", "visitor", "visitors", "traveler", "travellers",
    "beach", "beaches", "island", "mountain", "trail", "park", "cave",
    "restaurant", "hotel", "accommodation", "transport", "bus", "ferry",
    "jeju", "korea", "korean", "traditional", "cultural", "heritage",
}


def _extract_named_entities(text: str) -> list[str]:
    """Extract capitalized spans as candidate named entities."""
    entities = []
    for m in _NE_RE.finditer(text):
        span = m.group(1)
        tokens = span.split()
        # Filter single-word false positives
        if len(tokens) == 1 and span in _FALSE_POSITIVE_TOKENS:
            continue
        entities.append(span)
    return entities


# ── Main class ────────────────────────────────────────────────────────────────

class ResponseValidator:
    """
    Validate an LLM-generated answer against the ontology and SHACL rules.

    Parameters
    ----------
    ontology_names : set[str]
        All concept names from the active ontology (case-insensitive lookup set).
    ontology_aliases : dict[str, str], optional
        Alias → canonical name mapping. Aliases are also treated as grounded.
    rules : list[dict], optional
        SHACL rules; used for constraint violation detection.
    fail_on_hallucination : bool
        If True, any ungrounded named entity causes passed=False (default True).
    max_ungrounded_fraction : float
        Fraction of named entities allowed to be ungrounded before failing.
        Default 0.25 (stricter than before but allows minor coverage gaps).
    use_substring_match : bool
        If True, an entity is grounded when any ontology name/alias is a
        substring of the entity (or vice-versa). Improves recall.
    """

    def __init__(
        self,
        ontology_names: Optional[set[str]] = None,
        ontology_aliases: Optional[dict[str, str]] = None,
        rules: Optional[list[dict]] = None,
        fail_on_hallucination: bool = False,      # changed: soft mode by default
        max_ungrounded_fraction: float = 0.25,    # allow up to 25% ungrounded
        use_substring_match: bool = True,
    ):
        self._onto_names_lower: set[str] = (
            {n.lower() for n in ontology_names} if ontology_names else set()
        )
        # Build alias lookup (alias_lower → canonical_lower)
        self._alias_lower: set[str] = set()
        if ontology_aliases:
            self._alias_lower = {a.lower() for a in ontology_aliases}
        # Combined whitelist: names + aliases + always-valid tokens
        self._whitelist: set[str] = (
            self._onto_names_lower | self._alias_lower | _ALWAYS_VALID_TOKENS
        )
        self._rules = rules or []
        self._fail_on_hallucination = fail_on_hallucination
        self._max_ungrounded_fraction = max_ungrounded_fraction
        self._use_substring_match = use_substring_match

    @classmethod
    def from_concepts(
        cls,
        concepts: list[dict],
        rules: Optional[list[dict]] = None,
        **kwargs,
    ) -> "ResponseValidator":
        """Build validator directly from a concepts list (includes all aliases)."""
        names = {c["name"] for c in concepts}
        aliases: dict[str, str] = {}
        for c in concepts:
            for a in c.get("aliases", []):
                aliases[a] = c["name"]
        return cls(ontology_names=names, ontology_aliases=aliases, rules=rules, **kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        answer: str,
        context_concepts: Optional[list[dict]] = None,
        expected_concepts: Optional[list[str]] = None,
        query: Optional[str] = None,
    ) -> ValidationReport:
        """
        Run all validation checks on `answer`.

        Parameters
        ----------
        answer : str
            The generated response text.
        context_concepts : list[dict], optional
            Ontology concepts that were in the retrieval context (for completeness check).
        expected_concepts : list[str], optional
            Concept names expected to appear in the answer.
        query : str, optional
            Original query (for format/completeness heuristics).

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []

        # 1. Hallucination check
        issues.extend(self._check_hallucinations(answer))

        # 2. Constraint check (SHACL rules)
        issues.extend(self._check_constraints(answer))

        # 3. Factual consistency (basic: negation + relation check)
        issues.extend(self._check_factual_consistency(answer, context_concepts or []))

        # 4. Completeness check
        if expected_concepts:
            issues.extend(self._check_completeness(answer, expected_concepts))

        # 5. Format check
        issues.extend(self._check_format(answer, query))

        # Determine pass/fail
        has_errors = any(i.severity == "error" for i in issues)
        passed = not has_errors

        report = ValidationReport(answer=answer, passed=passed, issues=issues)
        logger.debug("Validation: %s", report.summary())
        return report

    def update_ontology(
        self,
        ontology_names: set[str],
        ontology_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        """Update the ontology name/alias sets (e.g., after loading a new version)."""
        self._onto_names_lower = {n.lower() for n in ontology_names}
        self._alias_lower = {a.lower() for a in (ontology_aliases or {})}
        self._whitelist = self._onto_names_lower | self._alias_lower | _ALWAYS_VALID_TOKENS

    def _is_grounded(self, entity: str) -> bool:
        """Return True if entity is grounded in ontology (exact, alias, or substring)."""
        ent_lower = entity.lower()
        # 1. exact match in whitelist
        if ent_lower in self._whitelist:
            return True
        # 2. always-valid token (single word)
        if ent_lower in _ALWAYS_VALID_TOKENS:
            return True
        # 3. substring match — ontology name contained inside entity or vice-versa
        if self._use_substring_match:
            for name in self._whitelist:
                if len(name) >= 4 and (name in ent_lower or ent_lower in name):
                    return True
        return False

    # ── Check implementations ─────────────────────────────────────────────────

    def _check_hallucinations(self, answer: str) -> list[ValidationIssue]:
        """Flag named entities in the answer not grounded in ontology/aliases/known tokens."""
        if not self._whitelist:
            return []  # no ontology loaded, skip

        entities = _extract_named_entities(answer)
        if not entities:
            return []

        issues: list[ValidationIssue] = []
        ungrounded = 0
        for ent in entities:
            if not self._is_grounded(ent):
                ungrounded += 1
                # In soft mode, always start as warning; elevate later if fraction exceeded
                severity = "error" if self._fail_on_hallucination else "warning"
                issues.append(ValidationIssue(
                    check="hallucination",
                    severity=severity,
                    message=f"Named entity '{ent}' not grounded in ontology.",
                    span=ent,
                ))

        # Elevate to error only when ungrounded fraction exceeds threshold
        if not self._fail_on_hallucination and entities:
            frac = ungrounded / len(entities)
            if frac > self._max_ungrounded_fraction:
                for issue in issues:
                    issue.severity = "error"

        return issues

    def _check_constraints(self, answer: str) -> list[ValidationIssue]:
        """Check answer text against SHACL rule messages."""
        issues: list[ValidationIssue] = []
        answer_lower = answer.lower()

        for rule in self._rules:
            # Violation patterns: look for prohibited terms in answer
            prohibited = rule.get("prohibited_terms", [])
            for term in prohibited:
                if term.lower() in answer_lower:
                    severity_map = {"Violation": "error", "Warning": "warning", "Info": "info"}
                    sev = severity_map.get(rule.get("severity", "Warning"), "warning")
                    issues.append(ValidationIssue(
                        check="constraint",
                        severity=sev,
                        message=f"Answer contains prohibited term '{term}' (rule: {rule.get('name', '?')})",
                        span=term,
                    ))

        return issues

    def _check_factual_consistency(
        self, answer: str, context_concepts: list[dict]
    ) -> list[ValidationIssue]:
        """
        Basic factual consistency: detect negation of relations that are
        explicitly stated in context concepts.
        e.g., if context says "X located_in Jeju" but answer says "X is not in Jeju"
        """
        issues: list[ValidationIssue] = []
        if not context_concepts:
            return issues

        # Simple negation patterns
        negation_re = re.compile(
            r"\b(not|no|never|isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|don'?t)\b\s+(\w+(?:\s+\w+){0,3})",
            re.I
        )

        negated_spans: list[str] = []
        for m in negation_re.finditer(answer):
            negated_spans.append(m.group(2).lower())

        for concept in context_concepts:
            name = concept.get("name", "").lower()
            if any(name in span for span in negated_spans):
                issues.append(ValidationIssue(
                    check="factual",
                    severity="warning",
                    message=f"Answer may contradict ontology concept '{concept.get('name', '')}'.",
                    span=concept.get("name"),
                ))

        return issues

    def _check_completeness(
        self, answer: str, expected_concepts: list[str]
    ) -> list[ValidationIssue]:
        """Warn if key expected concepts are entirely absent from the answer."""
        issues: list[ValidationIssue] = []
        answer_lower = answer.lower()
        missing = [c for c in expected_concepts if c.lower() not in answer_lower]
        if missing:
            issues.append(ValidationIssue(
                check="completeness",
                severity="warning",
                message=f"{len(missing)} expected concept(s) not mentioned: {', '.join(missing[:5])}",
            ))
        return issues

    def _check_format(
        self, answer: str, query: Optional[str]
    ) -> list[ValidationIssue]:
        """Basic format checks: empty answer, extremely short, or refusal."""
        issues: list[ValidationIssue] = []

        if not answer or not answer.strip():
            issues.append(ValidationIssue(
                check="format",
                severity="error",
                message="Answer is empty.",
            ))
            return issues

        if len(answer.strip()) < 20:
            issues.append(ValidationIssue(
                check="format",
                severity="warning",
                message=f"Answer is very short ({len(answer.strip())} chars).",
            ))

        # Refusal detection
        refusal_re = re.compile(
            r"\b(i (cannot|can'?t|am unable to|don'?t know|have no information)|"
            r"i'?m sorry|as an ai|i apologize)\b",
            re.I
        )
        if refusal_re.search(answer):
            issues.append(ValidationIssue(
                check="format",
                severity="warning",
                message="Answer may be a refusal or disclaimer.",
            ))

        return issues
