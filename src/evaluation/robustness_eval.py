"""
Robustness Evaluation — tests system resilience to adversarial/edge-case inputs.

Tests (addresses RQ7 from the paper):
  1. paraphrase_consistency   — same answer for rephrased queries (token F1)
  2. incomplete_query_score   — graceful handling of underspecified input
  3. adversarial_resistance   — resist fabricated entity names in queries
  4. constraint_conflict_score — handle impossible/contradictory constraints
  5. unseen_destination_score  — handle destinations not in ontology

Paper table: Table 8 — Robustness Metrics
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.robustness")


# ── Stopwords (shared with response_metrics) ──────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "in", "of", "to", "for", "and", "or", "is", "are",
    "was", "were", "at", "on", "by", "with", "from", "that", "this", "it",
    "be", "as", "have", "has", "i", "my", "you", "your", "we", "our",
}

# Signals that a response is hedging / acknowledging uncertainty
_HEDGE_PATTERNS = re.compile(
    r"\b(i'?m not sure|i cannot confirm|i don't know|cannot verify|"
    r"please (check|verify|consult)|not found|not available|"
    r"could you (clarify|specify)|please (clarify|specify)|"
    r"i need more (information|details)|not in (my|the) knowledge|"
    r"unable to (find|confirm|verify)|i cannot (find|confirm)|"
    r"does not exist|not recognized|not a (known|recognized))\b",
    re.I,
)

# Fabricated entity markers (signals the model invented something)
_FABRICATION_RE = re.compile(
    r"\b(km|meters?|KRW|won|usd|\$|€|hours?|mins?|days?|years?|%|km²|acres?)\b"
    r"|\b\d{4,}\b",  # 4-digit numbers (years, distances, prices)
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RobustnessReport:
    """Robustness scores for one test item."""
    query_id:              int
    mode:                  str
    test_type:             str   # one of the 5 robustness test types
    score:                 float  # 0-1; higher = more robust
    passed:                bool   # True if score >= threshold
    details:               str    # human-readable description

    def to_dict(self) -> dict:
        return {
            "query_id":  self.query_id,
            "mode":      self.mode,
            "test_type": self.test_type,
            "score":     round(self.score, 3),
            "passed":    self.passed,
            "details":   self.details,
        }


@dataclass
class RobustnessAggregate:
    """Aggregated robustness metrics for one mode."""
    mode:                        str
    n_tests:                     int
    avg_paraphrase_consistency:  float
    avg_incomplete_score:        float
    avg_adversarial_resistance:  float
    avg_constraint_conflict:     float
    avg_unseen_destination:      float
    overall_pass_rate:           float

    def to_dict(self) -> dict:
        return {
            "mode":                       self.mode,
            "n_tests":                    self.n_tests,
            "avg_paraphrase_consistency": round(self.avg_paraphrase_consistency, 3),
            "avg_incomplete_score":       round(self.avg_incomplete_score, 3),
            "avg_adversarial_resistance": round(self.avg_adversarial_resistance, 3),
            "avg_constraint_conflict":    round(self.avg_constraint_conflict, 3),
            "avg_unseen_destination":     round(self.avg_unseen_destination, 3),
            "overall_pass_rate":          round(self.overall_pass_rate, 3),
        }


# ── Main evaluator ────────────────────────────────────────────────────────────

class RobustnessEvaluator:
    """
    Evaluate system robustness across 5 adversarial test categories.

    Parameters
    ----------
    ontology_concepts : list[dict]
        All concept dicts from the ontology.
    pass_threshold : float
        Minimum score to count a test as passed (default 0.5).
    """

    def __init__(
        self,
        ontology_concepts: Optional[list[dict]] = None,
        pass_threshold: float = 0.5,
    ):
        self._concepts = ontology_concepts or []
        self._threshold = pass_threshold
        self._concept_names: set[str] = {
            c.get("name", "").lower()
            for c in self._concepts if c.get("name")
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def paraphrase_consistency(
        self,
        query_id: int,
        mode: str,
        answer1: str,
        answer2: str,
    ) -> RobustnessReport:
        """
        Token F1 between two answers to paraphrased queries.
        High score = consistent answers to the same question.
        """
        score = self._token_f1(answer1, answer2)
        return RobustnessReport(
            query_id=query_id,
            mode=mode,
            test_type="paraphrase_consistency",
            score=score,
            passed=score >= self._threshold,
            details=f"Token F1={score:.3f} between paraphrase pair",
        )

    def incomplete_query_score(
        self,
        query_id: int,
        mode: str,
        query: str,
        answer: str,
    ) -> RobustnessReport:
        """
        Score how gracefully the system handles an underspecified query.
        Checks for hedging language, clarification requests, or appropriate
        generic guidance rather than hallucinated specifics.
        """
        answer_lower = answer.lower()
        score = 0.0

        # Hedging / clarification present → good
        if _HEDGE_PATTERNS.search(answer_lower):
            score += 0.5

        # Not too specific without basis (fabricated numbers for vague query)
        words = len(answer.split())
        if words < 200:
            score += 0.2  # short answers less likely to hallucinate details

        # Asking for clarification is ideal
        if "?" in answer and any(
            kw in answer_lower
            for kw in ["could you", "can you", "please specify", "which", "what type"]
        ):
            score += 0.3

        return RobustnessReport(
            query_id=query_id,
            mode=mode,
            test_type="incomplete_query",
            score=min(1.0, score),
            passed=score >= self._threshold,
            details=f"Hedge={bool(_HEDGE_PATTERNS.search(answer_lower))}, words={words}",
        )

    def adversarial_resistance(
        self,
        query_id: int,
        mode: str,
        fake_entity: str,
        answer: str,
    ) -> RobustnessReport:
        """
        Checks whether the system resists endorsing a fabricated entity name
        embedded in the query. Resistance = hedging OR denial OR not repeating
        the fabricated entity confidently.
        """
        answer_lower = answer.lower()
        fake_lower   = fake_entity.lower()
        score = 0.0

        # If the answer hedges or denies, that's good
        if _HEDGE_PATTERNS.search(answer_lower):
            score += 0.6

        # If it explicitly says the entity is not known/found
        if fake_lower in answer_lower:
            # Entity mentioned — check if accompanied by denial
            for m in re.finditer(re.escape(fake_lower), answer_lower):
                window = answer_lower[max(0, m.start() - 80): m.end() + 80]
                deny_re = re.compile(
                    r"\b(not found|does not exist|not recognized|cannot confirm|"
                    r"not in|not a known|unable to find|not available)\b", re.I
                )
                if deny_re.search(window):
                    score += 0.4
                    break
        else:
            # Entity not mentioned at all → the model didn't invent about it
            score += 0.4

        # Fabrication signal: specific numbers about unknown entity → bad
        if fake_lower in answer_lower and _FABRICATION_RE.search(answer):
            score -= 0.3

        return RobustnessReport(
            query_id=query_id,
            mode=mode,
            test_type="adversarial_resistance",
            score=max(0.0, min(1.0, score)),
            passed=score >= self._threshold,
            details=f"fake_entity='{fake_entity}', entity_in_answer={fake_lower in answer_lower}",
        )

    def constraint_conflict_score(
        self,
        query_id: int,
        mode: str,
        answer: str,
        conflicting_constraints: Optional[list[str]] = None,
    ) -> RobustnessReport:
        """
        Score how well the system handles a query with contradictory constraints
        (e.g., "cheapest 5-star luxury hotel"). Good responses acknowledge the
        conflict or offer a compromise, rather than ignoring one constraint.
        """
        answer_lower = answer.lower()
        score = 0.0

        _CONFLICT_SIGNALS = re.compile(
            r"\b(however|but|although|trade-off|compromise|balance|"
            r"depends on|may not be possible|difficult to|challenging|"
            r"not possible to|cannot (both|simultaneously)|"
            r"one or the other|either|or)\b", re.I
        )

        if _CONFLICT_SIGNALS.search(answer_lower):
            score += 0.5

        if _HEDGE_PATTERNS.search(answer_lower):
            score += 0.3

        # If conflicting constraints are provided, check they're both addressed
        if conflicting_constraints:
            mentioned = sum(
                1 for c in conflicting_constraints
                if c.lower() in answer_lower
            )
            score += 0.2 * (mentioned / len(conflicting_constraints))

        return RobustnessReport(
            query_id=query_id,
            mode=mode,
            test_type="constraint_conflict",
            score=min(1.0, score),
            passed=score >= self._threshold,
            details=f"conflict_signal={bool(_CONFLICT_SIGNALS.search(answer_lower))}",
        )

    def unseen_destination_score(
        self,
        query_id: int,
        mode: str,
        destination: str,
        answer: str,
    ) -> RobustnessReport:
        """
        Score how well the system handles a query about a destination not in
        the ontology. Good response: acknowledges limited knowledge or offers
        general information; bad response: makes up specific facts.
        """
        answer_lower    = answer.lower()
        dest_lower      = destination.lower()
        in_ontology     = dest_lower in self._concept_names or any(
            dest_lower in k or k in dest_lower for k in self._concept_names
        )

        score = 0.0

        if not in_ontology:
            # Destination truly not in ontology
            if _HEDGE_PATTERNS.search(answer_lower):
                score += 0.5
            if dest_lower in answer_lower and _FABRICATION_RE.search(answer):
                # Inventing specific facts about unknown place
                score -= 0.3
            if dest_lower not in answer_lower:
                # Didn't even engage with the unknown destination
                score += 0.2
        else:
            # Destination IS in ontology — answer should be confident
            if dest_lower in answer_lower:
                score += 0.5
            if not _HEDGE_PATTERNS.search(answer_lower):
                score += 0.3

        return RobustnessReport(
            query_id=query_id,
            mode=mode,
            test_type="unseen_destination",
            score=max(0.0, min(1.0, score)),
            passed=score >= self._threshold,
            details=f"destination='{destination}', in_ontology={in_ontology}",
        )

    def aggregate(
        self,
        reports: list[RobustnessReport],
        mode: str,
    ) -> RobustnessAggregate:
        """Aggregate per-test robustness reports into mode-level summary."""
        n = len(reports)
        if n == 0:
            return RobustnessAggregate(
                mode=mode, n_tests=0,
                avg_paraphrase_consistency=0, avg_incomplete_score=0,
                avg_adversarial_resistance=0, avg_constraint_conflict=0,
                avg_unseen_destination=0, overall_pass_rate=0,
            )

        def _avg(test_type: str) -> float:
            items = [r.score for r in reports if r.test_type == test_type]
            return mean(items) if items else 0.0

        return RobustnessAggregate(
            mode=mode,
            n_tests=n,
            avg_paraphrase_consistency=_avg("paraphrase_consistency"),
            avg_incomplete_score=_avg("incomplete_query"),
            avg_adversarial_resistance=_avg("adversarial_resistance"),
            avg_constraint_conflict=_avg("constraint_conflict"),
            avg_unseen_destination=_avg("unseen_destination"),
            overall_pass_rate=mean(1.0 if r.passed else 0.0 for r in reports),
        )

    def compare_modes(
        self,
        mode_results: dict[str, RobustnessAggregate],
    ) -> str:
        """Return a markdown table comparing all modes."""
        lines = [
            "| Mode | Paraphrase | Incomplete | Adversarial | Conflict | Unseen | Pass% |",
            "|------|:----------:|:----------:|:-----------:|:--------:|:------:|:-----:|",
        ]
        for mode, agg in mode_results.items():
            lines.append(
                f"| {mode} "
                f"| {agg.avg_paraphrase_consistency:.3f} "
                f"| {agg.avg_incomplete_score:.3f} "
                f"| {agg.avg_adversarial_resistance:.3f} "
                f"| {agg.avg_constraint_conflict:.3f} "
                f"| {agg.avg_unseen_destination:.3f} "
                f"| {agg.overall_pass_rate*100:.1f}% |"
            )
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    def _token_f1(self, a: str, b: str) -> float:
        """Token-level F1 between two texts."""
        t1 = Counter(self._tokenize(a))
        t2 = Counter(self._tokenize(b))
        if not t1 or not t2:
            return 0.0
        common    = sum((t1 & t2).values())
        precision = common / sum(t2.values())
        recall    = common / sum(t1.values())
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
