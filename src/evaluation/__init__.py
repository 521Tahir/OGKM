"""Evaluation module: benchmark creation and metrics computation."""
from .benchmark import TourismBenchmark
from .metrics import EvaluationMetrics
from .hallucination_eval import HallucinationEvaluator, HallucinationReport, HallucinationSummary
from .ontology_metrics import OntologyMetricsCalculator, OntologySnapshot, OntologyEvolutionReport
from .retrieval_metrics import RetrievalMetricsCalculator, RetrievalAggregateMetrics

__all__ = [
    "TourismBenchmark", "EvaluationMetrics",
    "HallucinationEvaluator", "HallucinationReport", "HallucinationSummary",
    "OntologyMetricsCalculator", "OntologySnapshot", "OntologyEvolutionReport",
    "RetrievalMetricsCalculator", "RetrievalAggregateMetrics",
]
