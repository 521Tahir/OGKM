"""
Evaluation metrics for the tourism ontology-RAG system.

Metrics computed per condition (no_rag, doc_rag, onto_rag, constrained):
  - entity_recall    : fraction of expected ontology entities in the answer
  - entity_precision : fraction of answer entities that are in the ontology
  - hallucination_rate : fraction of answers with ≥1 hallucinated entity
  - grounding_score  : average ExplanationBuilder grounding score
  - avg_constraint_violations : mean violations per answer (constrained mode)
  - answer_length    : mean answer length in tokens (proxy for verbosity)

All metrics return a MetricsReport with per-item and aggregate values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..generation.constrained_generator import GenerationResult
from ..generation.explanation_builder import ExplanationTrace
from ..evaluation.benchmark import BenchmarkItem
from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class ItemMetrics:
    """Per-item metrics."""
    item_id:          int
    mode:             str
    entity_recall:    float
    entity_precision: float
    hallucinated:     bool
    n_violations:     int
    grounding_score:  float
    answer_length:    int   # word count

    def to_dict(self) -> dict:
        return {
            "item_id":          self.item_id,
            "mode":             self.mode,
            "entity_recall":    round(self.entity_recall, 3),
            "entity_precision": round(self.entity_precision, 3),
            "hallucinated":     self.hallucinated,
            "n_violations":     self.n_violations,
            "grounding_score":  round(self.grounding_score, 3),
            "answer_length":    self.answer_length,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics over all benchmark items for one mode."""
    mode:              str
    n_items:           int
    entity_recall:     float
    entity_precision:  float
    hallucination_rate: float
    avg_violations:    float
    avg_grounding:     float
    avg_answer_length: float

    def to_dict(self) -> dict:
        return {
            "mode":               self.mode,
            "n_items":            self.n_items,
            "entity_recall":      round(self.entity_recall, 3),
            "entity_precision":   round(self.entity_precision, 3),
            "hallucination_rate": round(self.hallucination_rate, 3),
            "avg_violations":     round(self.avg_violations, 3),
            "avg_grounding":      round(self.avg_grounding, 3),
            "avg_answer_length":  round(self.avg_answer_length, 1),
        }


class EvaluationMetrics:
    """
    Computes evaluation metrics for a set of generation results.

    Usage:
        metrics = EvaluationMetrics(ontology_concept_names)
        item_mets = metrics.score_item(result, trace, benchmark_item)
        agg = metrics.aggregate(item_mets_list, mode="onto_rag")
    """

    def __init__(self, ontology_concept_names: Optional[set[str]] = None) -> None:
        self.ontology_names_lower = (
            {n.lower() for n in ontology_concept_names}
            if ontology_concept_names else set()
        )

    def score_item(
        self,
        result: GenerationResult,
        trace: ExplanationTrace,
        benchmark_item: BenchmarkItem,
    ) -> ItemMetrics:
        answer = result.answer
        answer_lower = answer.lower()

        # Entity recall: expected ontology concepts present in answer
        expected = benchmark_item.expected_concepts
        if expected:
            recalled = sum(1 for e in expected if e.lower() in answer_lower)
            entity_recall = recalled / len(expected)
        else:
            entity_recall = 1.0  # nothing expected → trivially recalled

        # Entity precision: capitalized entities in answer that are in ontology
        answer_entities = set(re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}\b", answer))
        if answer_entities and self.ontology_names_lower:
            grounded_count = sum(
                1 for e in answer_entities if e.lower() in self.ontology_names_lower
            )
            entity_precision = grounded_count / len(answer_entities)
        else:
            entity_precision = 1.0

        n_violations  = len(result.violations)
        hallucinated  = n_violations > 0
        answer_length = len(answer.split())

        return ItemMetrics(
            item_id=benchmark_item.item_id,
            mode=result.mode,
            entity_recall=entity_recall,
            entity_precision=entity_precision,
            hallucinated=hallucinated,
            n_violations=n_violations,
            grounding_score=trace.grounding_score,
            answer_length=answer_length,
        )

    def aggregate(
        self,
        item_metrics: list[ItemMetrics],
        mode: str,
    ) -> AggregateMetrics:
        n = len(item_metrics)
        if n == 0:
            return AggregateMetrics(
                mode=mode, n_items=0,
                entity_recall=0, entity_precision=0,
                hallucination_rate=0, avg_violations=0,
                avg_grounding=0, avg_answer_length=0,
            )

        def _mean(vals) -> float:
            return sum(vals) / n

        return AggregateMetrics(
            mode=mode,
            n_items=n,
            entity_recall=_mean([m.entity_recall for m in item_metrics]),
            entity_precision=_mean([m.entity_precision for m in item_metrics]),
            hallucination_rate=_mean([1.0 if m.hallucinated else 0.0 for m in item_metrics]),
            avg_violations=_mean([m.n_violations for m in item_metrics]),
            avg_grounding=_mean([m.grounding_score for m in item_metrics]),
            avg_answer_length=_mean([m.answer_length for m in item_metrics]),
        )
