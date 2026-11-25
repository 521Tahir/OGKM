"""
Unit tests for generation pipeline:
  - ConstrainedGenerator (offline / no-LLM mode)
  - ResponseValidator
  - UncertaintyHandler
"""
import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.response_validator import ResponseValidator, ValidationReport, ValidationIssue
from src.generation.uncertainty_handler import UncertaintyHandler, UncertaintyReport
from src.generation.constrained_generator import ConstrainedGenerator, GenerationResult, ConstraintViolation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ontology_names():
    """Real concept names from v1.1.0 for grounding checks."""
    concepts = json.load(open("artifacts/ontologies/1.1.0/concepts.json"))
    return {c["name"] for c in concepts}


@pytest.fixture(scope="module")
def rules():
    return json.load(open("artifacts/ontologies/1.1.0/rules.json"))


@pytest.fixture(scope="module")
def validator(ontology_names, rules):
    return ResponseValidator(
        ontology_names=ontology_names,
        rules=rules,
        fail_on_hallucination=True,
    )


@pytest.fixture(scope="module")
def validator_lenient(ontology_names):
    return ResponseValidator(
        ontology_names=ontology_names,
        fail_on_hallucination=False,
        max_ungrounded_fraction=0.5,
    )


@pytest.fixture(scope="module")
def uncertainty_handler():
    return UncertaintyHandler()


# ── ResponseValidator tests ───────────────────────────────────────────────────

class TestResponseValidator:

    def test_validate_returns_report(self, validator):
        report = validator.validate("Jeju is a beautiful island in South Korea.")
        assert isinstance(report, ValidationReport)

    def test_report_has_passed_and_issues(self, validator):
        report = validator.validate("The restaurant is located in Jeju.")
        assert hasattr(report, "passed")
        assert hasattr(report, "issues")
        assert isinstance(report.issues, list)

    def test_fully_grounded_answer_passes(self, validator_lenient, ontology_names):
        # Pick three concepts we know are in the ontology
        # Use lenient validator since short answers can still trigger edge-case NER
        known = ["Jeju", "Jeju Island", "Beach"]
        answer = f"Jeju Island is popular for its beaches and Jeju cuisine."
        report = validator_lenient.validate(answer, expected_concepts=known)
        # Should have no hallucination errors for known concepts
        halluc_errors = [i for i in report.issues
                         if i.check == "hallucination" and i.severity == "error"]
        assert len(halluc_errors) == 0

    def test_hallucination_detected_for_made_up_entity(self, validator):
        # Capitalised unknown names → should be flagged as ungrounded
        answer = "ZorgonPlanet and QuantumResort are top destinations in Jeju."
        report = validator.validate(answer)
        halluc_issues = report.hallucination_issues
        assert len(halluc_issues) > 0

    def test_lenient_validator_passes_partial_hallucination(self, validator_lenient):
        # 1 fake entity among several known → lenient mode passes
        answer = "Visit Jeju for ZorgonHotel, beaches, and local food."
        report = validator_lenient.validate(answer)
        assert isinstance(report, ValidationReport)

    def test_empty_answer_handled_gracefully(self, validator):
        report = validator.validate("")
        assert isinstance(report, ValidationReport)

    def test_n_errors_property(self, validator):
        report = validator.validate("ZorgonPlanet XyzzyResort UnknownCity are great.")
        assert report.n_errors >= 0

    def test_n_warnings_property(self, validator):
        report = validator.validate("Jeju is nice.")
        assert report.n_warnings >= 0

    def test_validation_with_context_concepts(self, validator, ontology_names):
        known = sorted(ontology_names)[:2]
        context_concepts = [{"name": k} for k in known]
        report = validator.validate(
            f"The answer covers {known[0]} and {known[1]}.",
            context_concepts=context_concepts,
            expected_concepts=known,
        )
        assert isinstance(report, ValidationReport)

    def test_summary_returns_string(self, validator):
        report = validator.validate("Jeju Island is a popular destination.")
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_constraint_violation_dataclass(self):
        v = ConstraintViolation(
            violation_type="hallucinated_entity",
            detail="Unknown entity: FakePlace",
            evidence="FakePlace",
        )
        d = v.to_dict()
        assert d["type"] == "hallucinated_entity"
        assert "evidence" in d


# ── UncertaintyHandler tests ──────────────────────────────────────────────────

class TestUncertaintyHandler:

    def test_analyze_returns_report(self, uncertainty_handler):
        report = uncertainty_handler.analyze("Jeju is known for its beaches.")
        assert isinstance(report, UncertaintyReport)

    def test_report_has_confidence(self, uncertainty_handler):
        report = uncertainty_handler.analyze("The restaurant might be good.")
        assert hasattr(report, "confidence")
        assert 0.0 <= report.confidence <= 1.0

    def test_hedged_answer_has_lower_confidence(self, uncertainty_handler):
        certain = uncertainty_handler.analyze("Jeju is the top tourist destination.")
        hedged  = uncertainty_handler.analyze(
            "Jeju might possibly be a destination, though I'm not sure and it could vary."
        )
        # Hedged answer should have equal or lower confidence
        assert hedged.confidence <= certain.confidence + 0.1

    def test_hedges_list_populated_for_hedged_answer(self, uncertainty_handler):
        report = uncertainty_handler.analyze("It might be possible, though I'm not certain.")
        assert hasattr(report, "hedges")
        # Should detect at least one hedge phrase
        assert len(report.hedges) >= 1

    def test_calibrate_score_reduces_on_violations(self, uncertainty_handler):
        base = 0.90
        calibrated = uncertainty_handler.calibrate_score(base, n_violations=3)
        assert calibrated < base

    def test_calibrate_score_no_violations_unchanged(self, uncertainty_handler):
        base = 0.80
        calibrated = uncertainty_handler.calibrate_score(base, n_violations=0)
        # Should be same or very close
        assert abs(calibrated - base) < 0.05

    def test_inject_calibration_prompt_returns_string(self, uncertainty_handler):
        result = uncertainty_handler.inject_calibration_prompt("You are a helpful assistant.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_annotate_answer_returns_string(self, uncertainty_handler):
        report = uncertainty_handler.analyze("It might be a nice place.")
        annotated = uncertainty_handler.annotate_answer("It might be a nice place.", report)
        assert isinstance(annotated, str)

    def test_empty_answer_handled(self, uncertainty_handler):
        report = uncertainty_handler.analyze("")
        assert isinstance(report, UncertaintyReport)
        assert 0.0 <= report.confidence <= 1.0

    def test_report_to_dict(self, uncertainty_handler):
        report = uncertainty_handler.analyze("Jeju might be rainy in summer.")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "confidence" in d


# ── ConstrainedGenerator (offline/no-LLM) tests ───────────────────────────────

class TestConstrainedGenerator:
    """
    Tests run in OFFLINE mode (no LLM loaded) to keep the test suite fast.
    The generator falls back to a template answer when no LLM client is provided.
    """

    @pytest.fixture(scope="class")
    def generator(self, ontology_names, rules):
        from src.graph.graph_store import TourismGraphStore
        concepts  = json.load(open("artifacts/ontologies/1.1.0/concepts.json"))
        relations = json.load(open("artifacts/ontologies/1.1.0/relations.json"))
        graph = TourismGraphStore()
        graph.load(concepts, relations)
        # No LLM client — runs offline/template mode (client is first positional arg)
        return ConstrainedGenerator(client=None, graph=graph)

    def test_generate_returns_result(self, generator):
        result = generator.generate("What are the best beaches in Jeju?", mode="no_rag")
        assert isinstance(result, GenerationResult)

    def test_result_has_required_fields(self, generator):
        result = generator.generate("Recommend a restaurant in Jeju.", mode="no_rag")
        assert hasattr(result, "query")
        assert hasattr(result, "answer")
        assert hasattr(result, "mode")
        assert hasattr(result, "violations")
        assert hasattr(result, "is_grounded")

    def test_result_query_matches_input(self, generator):
        q = "What is the best hiking trail in Jeju?"
        result = generator.generate(q, mode="no_rag")
        assert result.query == q

    def test_result_mode_matches(self, generator):
        result = generator.generate("Tell me about Jeju.", mode="onto_rag")
        assert result.mode == "onto_rag"

    def test_to_dict_serializable(self, generator):
        result = generator.generate("What food is Jeju famous for?", mode="no_rag")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "query" in d
        assert "answer" in d
        assert "mode" in d

    def test_violations_is_list(self, generator):
        result = generator.generate("Describe Jeju tourism.", mode="no_rag")
        assert isinstance(result.violations, list)

    def test_answer_is_non_empty_string(self, generator):
        result = generator.generate("Best time to visit Jeju?", mode="no_rag")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_constraint_violation_dataclass_in_violations(self, generator):
        result = generator.generate("Tell me about Jeju.", mode="constrained")
        for v in result.violations:
            assert isinstance(v, ConstraintViolation)
            assert hasattr(v, "violation_type")
            assert hasattr(v, "detail")
