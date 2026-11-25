"""
Response Quality Metrics — evaluate the quality of generated answers.

Metrics (addresses RQ5 from the paper):
  1. factual_correctness_score  — entity grounding rate in answer vs ontology
  2. constraint_satisfaction_rate — fraction of answers passing ResponseValidator
  3. completeness_score          — fraction of expected concepts mentioned
  4. instruction_satisfaction    — does the answer address the query type?
  5. paraphrase_consistency      — token overlap for same query paraphrases

Paper table: Table 5 — Response Quality Metrics
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.response_metrics")

# ── Stopwords for token overlap ───────────────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "in", "of", "to", "for", "and", "or", "is", "are",
    "was", "were", "at", "on", "by", "with", "from", "that", "this", "it",
    "be", "as", "have", "has", "i", "my", "you", "your", "we", "our",
}

# ── Query intent patterns ─────────────────────────────────────────────────────
_INTENT_PATTERNS: dict[str, list[str]] = {
    "recommendation":  [r"\bbest\b", r"\brecommend\b", r"\btop\b", r"\bsuggest\b"],
    "comparison":      [r"\bvs\b", r"\bcompare\b", r"\bdifference\b", r"\bor\b"],
    "factual":         [r"\bwhat is\b", r"\bwhere is\b", r"\bwhen\b", r"\bhow many\b"],
    "how_to":          [r"\bhow (to|do|can)\b", r"\bsteps\b", r"\bprocess\b"],
    "list":            [r"\blist\b", r"\bexamples?\b", r"\btypes?\b", r"\bkinds?\b"],
    "definition":      [r"\bdefine\b", r"\bwhat is\b", r"\bmeaning\b", r"\bexplain\b"],
}

# Minimum answer keywords expected per intent type
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "recommendation": ["recommend", "suggest", "best", "top", "visit", "try", "option"],
    "comparison":     ["while", "whereas", "however", "on the other hand", "both", "differ"],
    "factual":        [],   # hard to predict — scored by entity presence
    "how_to":         ["step", "first", "then", "finally", "can", "should", "need"],
    "list":           ["\n", "•", "-", "1.", "2.", "include", "such as", "like"],
    "definition":     ["is a", "refers to", "known as", "defined as", "type of"],
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ResponseMetrics:
    """Quality metrics for a single generated response."""
    query_id:                   int
    mode:                       str
    factual_correctness:        float   # fraction of answer entities grounded in ontology
    completeness:               float   # fraction of expected concepts mentioned
    instruction_satisfaction:   float   # 0-1 score for query-intent alignment
    constraint_satisfied:       bool    # did ResponseValidator pass?
    answer_length:              int     # word count
    query_intent:               str     # detected query type

    def to_dict(self) -> dict:
        return {
            "query_id":                 self.query_id,
            "mode":                     self.mode,
            "factual_correctness":      round(self.factual_correctness, 3),
            "completeness":             round(self.completeness, 3),
            "instruction_satisfaction": round(self.instruction_satisfaction, 3),
            "constraint_satisfied":     self.constraint_satisfied,
            "answer_length":            self.answer_length,
            "query_intent":             self.query_intent,
        }


@dataclass
class ResponseMetricsAggregate:
    """Aggregate response metrics across all queries for one mode."""
    mode:                           str
    n_responses:                    int
    avg_factual_correctness:        float
    avg_completeness:               float
    avg_instruction_satisfaction:   float
    constraint_satisfaction_rate:   float
    avg_answer_length:              float

    def to_dict(self) -> dict:
        return {
            "mode":                           self.mode,
            "n_responses":                    self.n_responses,
            "avg_factual_correctness":        round(self.avg_factual_correctness, 3),
            "avg_completeness":               round(self.avg_completeness, 3),
            "avg_instruction_satisfaction":   round(self.avg_instruction_satisfaction, 3),
            "constraint_satisfaction_rate":   round(self.constraint_satisfaction_rate, 3),
            "avg_answer_length":              round(self.avg_answer_length, 1),
        }


# ── Main class ────────────────────────────────────────────────────────────────

class ResponseMetricsCalculator:
    """
    Compute response quality metrics without requiring an LLM.

    Parameters
    ----------
    ontology_names : set[str]
        All concept names from the active ontology.
    """

    def __init__(self, ontology_names: Optional[set[str]] = None):
        self._onto_lower: set[str] = (
            {n.lower() for n in ontology_names} if ontology_names else set()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        query_id: int,
        query: str,
        answer: str,
        mode: str,
        expected_concepts: Optional[list[str]] = None,
        constraint_passed: Optional[bool] = None,
    ) -> ResponseMetrics:
        """
        Compute all response quality metrics for one answer.

        Parameters
        ----------
        query_id : int          Numeric identifier for the query.
        query : str             The original query string.
        answer : str            The generated answer.
        mode : str              Generation mode label.
        expected_concepts : list[str], optional
            Concept names expected to appear in the answer.
        constraint_passed : bool, optional
            Whether ResponseValidator passed (True/False). If None,
            estimated from the answer text.
        """
        intent = self._detect_intent(query)
        words  = self._tokenize(answer)

        factual  = self.factual_correctness_score(answer)
        complete = self.completeness_score(answer, expected_concepts or [])
        instruct = self.instruction_satisfaction(answer, query, intent)
        passed   = constraint_passed if constraint_passed is not None else (factual > 0.5)

        return ResponseMetrics(
            query_id=query_id,
            mode=mode,
            factual_correctness=factual,
            completeness=complete,
            instruction_satisfaction=instruct,
            constraint_satisfied=passed,
            answer_length=len(words),
            query_intent=intent,
        )

    def score_batch(
        self,
        items: list[dict],
        mode: str,
    ) -> list[ResponseMetrics]:
        """
        Score a batch of {query_id, query, answer, expected_concepts?,
        constraint_passed?} dicts.
        """
        return [
            self.score(
                query_id=item.get("query_id", i),
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                mode=mode,
                expected_concepts=item.get("expected_concepts"),
                constraint_passed=item.get("constraint_passed"),
            )
            for i, item in enumerate(items)
        ]

    def aggregate(
        self,
        metrics: list[ResponseMetrics],
        mode: str,
    ) -> ResponseMetricsAggregate:
        """Aggregate per-query metrics into mode-level summary."""
        n = len(metrics)
        if n == 0:
            return ResponseMetricsAggregate(
                mode=mode, n_responses=0,
                avg_factual_correctness=0, avg_completeness=0,
                avg_instruction_satisfaction=0, constraint_satisfaction_rate=0,
                avg_answer_length=0,
            )
        return ResponseMetricsAggregate(
            mode=mode,
            n_responses=n,
            avg_factual_correctness=mean(m.factual_correctness for m in metrics),
            avg_completeness=mean(m.completeness for m in metrics),
            avg_instruction_satisfaction=mean(m.instruction_satisfaction for m in metrics),
            constraint_satisfaction_rate=mean(1.0 if m.constraint_satisfied else 0.0 for m in metrics),
            avg_answer_length=mean(m.answer_length for m in metrics),
        )

    def compare_modes(
        self,
        mode_results: dict[str, ResponseMetricsAggregate],
    ) -> str:
        """Return a markdown table comparing all modes."""
        lines = [
            "| Mode | Factual | Complete | Instruct | Constraint% | Avg Len |",
            "|------|:-------:|:--------:|:--------:|:-----------:|:-------:|",
        ]
        for mode, agg in mode_results.items():
            lines.append(
                f"| {mode} | {agg.avg_factual_correctness:.3f} "
                f"| {agg.avg_completeness:.3f} "
                f"| {agg.avg_instruction_satisfaction:.3f} "
                f"| {agg.constraint_satisfaction_rate*100:.1f}% "
                f"| {agg.avg_answer_length:.0f} |"
            )
        return "\n".join(lines)

    # ── Individual metric functions ────────────────────────────────────────────

    def factual_correctness_score(self, answer: str) -> float:
        """
        Fraction of capitalised named entities in the answer that appear
        in the ontology. Range [0, 1]. Higher = more grounded.
        """
        if not self._onto_lower:
            return 1.0  # can't check without ontology

        # Extract capitalised multi-word spans
        spans = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b", answer)
        _FALSE_POS = {
            "The", "A", "An", "In", "On", "Of", "For", "And", "Or",
            "Is", "Are", "It", "This", "That", "We", "I",
        }
        entities = [s for s in spans if s not in _FALSE_POS and len(s) >= 3]
        if not entities:
            return 1.0  # no named entities → nothing to check

        grounded = sum(
            1 for e in entities
            if e.lower() in self._onto_lower
            or any(e.lower() in k or k in e.lower() for k in self._onto_lower)
        )
        return grounded / len(entities)

    def completeness_score(
        self,
        answer: str,
        expected_concepts: list[str],
    ) -> float:
        """
        Fraction of expected_concepts that appear (case-insensitive) in the answer.
        Returns 1.0 if no expected concepts provided.
        """
        if not expected_concepts:
            return 1.0
        answer_lower = answer.lower()
        found = sum(
            1 for c in expected_concepts
            if c.lower() in answer_lower
        )
        return found / len(expected_concepts)

    def instruction_satisfaction(
        self,
        answer: str,
        query: str,
        intent: Optional[str] = None,
    ) -> float:
        """
        Heuristic score [0,1] for how well the answer satisfies the query intent.
        - Recommendation queries: checks for hedging/suggestion language
        - List queries: checks for enumeration markers
        - How-to queries: checks for step-by-step structure
        - Others: fallback to non-empty + minimum length
        """
        if intent is None:
            intent = self._detect_intent(query)

        answer_lower = answer.lower()
        keywords = _INTENT_KEYWORDS.get(intent, [])

        if not keywords:
            # Factual / unknown: score by length and entity presence
            words = self._tokenize(answer)
            return min(1.0, len(words) / 50.0)  # saturates at 50 words

        matches = sum(1 for kw in keywords if kw in answer_lower)
        return min(1.0, matches / max(1, len(keywords) // 2))

    def paraphrase_consistency(
        self,
        answer1: str,
        answer2: str,
    ) -> float:
        """
        Token-level F1 between two answers to paraphrased queries.
        High score = consistent answers to the same question.
        """
        t1 = Counter(self._tokenize(answer1))
        t2 = Counter(self._tokenize(answer2))

        if not t1 or not t2:
            return 0.0

        common = sum((t1 & t2).values())
        precision = common / sum(t2.values())
        recall    = common / sum(t1.values())

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        for intent, patterns in _INTENT_PATTERNS.items():
            if any(re.search(p, query_lower) for p in patterns):
                return intent
        return "factual"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]
