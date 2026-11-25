"""Unit tests for the extraction P/R/F1 scorer (Phase 10·A)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.extraction_prf import (
    ExtractionPRFScorer,
    f1_score,
    normalize_term,
)


@pytest.fixture
def concepts():
    return [
        {"name": "Hallasan", "concept_class": "Mountain", "aliases": ["Mt Hallasan"]},
        {"name": "Jeju", "concept_class": "City", "aliases": ["Jeju Island"]},
        {"name": "beach resort", "concept_class": "Accommodation", "aliases": []},
    ]


def test_normalize_term():
    assert normalize_term("  Jeju-Island!  ") == "jeju island"
    assert normalize_term("") == ""


def test_f1_score():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.5, 0.5) == 0.5
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(None, 0.5) is None
    assert f1_score(0.6, 0.4) == pytest.approx(0.48)


def test_concept_precision_from_human_labels(concepts):
    # 3 correct, 1 incorrect, 1 unlabelled → precision = 3/4
    concept_anns = [
        {"name": "a", "correct": True},
        {"name": "b", "correct": True},
        {"name": "c", "correct": True},
        {"name": "d", "correct": False},
        {"name": "e", "correct": None},
    ]
    scorer = ExtractionPRFScorer(concepts, [], concept_annotations=concept_anns)
    res = scorer.compute(use_silver=False)
    c = res["concept"]
    assert c["precision"] == pytest.approx(0.75)
    assert c["n_labeled"] == 4
    assert c["n_total"] == 5
    assert c["true_positives"] == 3
    assert c["false_positives"] == 1
    assert c["precision_source"] == "gold"


def test_precision_pending_when_all_null(concepts):
    concept_anns = [{"name": "x", "correct": None} for _ in range(5)]
    scorer = ExtractionPRFScorer(concepts, [], concept_annotations=concept_anns)
    res = scorer.compute(use_silver=False)
    c = res["concept"]
    assert c["precision"] is None
    assert c["precision_source"] == "none"
    assert any("no human or silver" in n for n in c["notes"])


def test_silver_precision_labels(concepts):
    # silver heuristic: article-prefixed / long phrases → incorrect
    concept_anns = [
        {"name": "beach resort", "concept_class": "Accommodation", "correct": None},
        {"name": "A budget-friendly stay", "concept_class": "Accommodation", "correct": None},
        {"name": "Hallasan", "concept_class": "Mountain", "correct": None},
        {"name": "an unusually long winded multi clause phrase here", "concept_class": "Activity", "correct": None},
    ]
    scorer = ExtractionPRFScorer(concepts, [], concept_annotations=concept_anns)
    res = scorer.compute(use_silver=True)
    c = res["concept"]
    # "beach resort" + "Hallasan" correct; article + long-phrase incorrect → 2/4
    assert c["precision"] == pytest.approx(0.5)
    assert c["precision_source"] == "silver"
    assert c["n_labeled"] == 4


def test_human_labels_override_silver(concepts):
    concept_anns = [
        {"name": "beach resort", "concept_class": "Accommodation", "correct": False},  # human says wrong
        {"name": "Hallasan", "concept_class": "Mountain", "correct": None},            # silver → correct
    ]
    scorer = ExtractionPRFScorer(concepts, [], concept_annotations=concept_anns)
    res = scorer.compute(use_silver=True)
    c = res["concept"]
    # human False + silver True → precision 1/2, source mixed
    assert c["precision"] == pytest.approx(0.5)
    assert c["precision_source"] == "mixed"


def test_concept_recall_strict_and_soft(concepts):
    gold = [
        {"expected_concepts": ["Hallasan", "Jeju Island", "submarine tour"]},
    ]
    # strict: Hallasan + Jeju Island (alias) match; submarine tour absent → 2/3
    strict = ExtractionPRFScorer(concepts, [], gold_annotations=gold, soft_match=False)
    rc, matched, expected, _ = strict.concept_recall()
    assert expected == 3
    assert matched == 2
    assert rc == pytest.approx(2 / 3)


def test_concept_recall_soft_match(concepts):
    # "luxury beach resort" should soft-match "beach resort"
    gold = [{"expected_concepts": ["luxury beach resort"]}]
    soft = ExtractionPRFScorer(concepts, [], gold_annotations=gold, soft_match=True)
    rc, matched, expected, _ = soft.concept_recall()
    assert matched == 1
    assert rc == pytest.approx(1.0)


def test_relation_precision_and_recall_status(concepts):
    rel_anns = [
        {"subject": "Hallasan", "predicate": "located_in", "object": "Jeju", "correct": True},
        {"subject": "x", "predicate": "located_in", "object": "y", "correct": False},
    ]
    scorer = ExtractionPRFScorer(concepts, [], relation_annotations=rel_anns)
    res = scorer.compute(use_silver=False)
    r = res["relation"]
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] is None  # no gold relation set
    assert r["f1"] is None
    assert any("relation recall unavailable" in n for n in r["notes"])
