"""Unit tests for the temporal ontology layer and retriever."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.temporal import (
    TemporalAnnotation, TemporalFact, season_to_months, months_to_phrase, parse_months,
)
from src.retrieval.temporal_retriever import TemporalRetriever


def test_season_to_months():
    assert season_to_months("spring") == [3, 4, 5]
    assert season_to_months("WINTER") == [12, 1, 2]
    assert season_to_months("nonsense") == []


def test_parse_months():
    assert parse_months([3, 4]) == [3, 4]
    assert parse_months("April") == [4]
    assert parse_months("summer") == [6, 7, 8]
    assert parse_months(["March", "apr"]) == [3, 4]


def test_months_to_phrase():
    assert months_to_phrase([3, 4]) == "March–April"
    assert months_to_phrase([11]) == "November"
    assert months_to_phrase([12, 1, 2]) == "December–February"   # wrap


def test_annotation_active():
    a = TemporalAnnotation(months=[3, 4], recurrence="annual")
    assert a.is_active(month=3) is True
    assert a.is_active(month=7) is False
    assert a.is_active() is True            # no time given → active

    always = TemporalAnnotation(recurrence="always")
    assert always.is_active(month=1) is True


def test_annotation_season_backfill():
    a = TemporalAnnotation(season="summer")
    assert a.months == [6, 7, 8]


def test_annotation_date_validity():
    a = TemporalAnnotation(recurrence="once", valid_from="2026-07-01", valid_to="2026-07-31")
    assert a.is_active(on=date(2026, 7, 15)) is True
    assert a.is_active(on=date(2026, 8, 1)) is False


def test_fact_sentence_and_roundtrip():
    f = TemporalFact(
        subject="Cherry blossoms", predicate="bloom_in", object="Jeju",
        temporal=TemporalAnnotation(months=[3, 4], note="late March to early April"),
        category="bloom",
    )
    assert "Cherry blossoms" in f.to_sentence()
    assert "late March to early April" in f.to_sentence()
    f2 = TemporalFact.from_dict(f.to_dict())
    assert f2.subject == f.subject
    assert f2.temporal.months == [3, 4]


def _facts():
    return [
        TemporalFact("Cherry blossoms", "bloom_in", "Jeju",
                     TemporalAnnotation(months=[3, 4], note="late March to early April"),
                     category="bloom", source="curated"),
        TemporalFact("Typhoon season", "affects", "Jeju",
                     TemporalAnnotation(months=[7, 8, 9], note="July to September"),
                     category="weather", source="curated"),
        TemporalFact("Hydrangeas", "bloom_in", "Jeju",
                     TemporalAnnotation(months=[6, 7], note="June to July"),
                     category="bloom", source="curated"),
    ]


def test_retriever_links_relevant_fact():
    r = TemporalRetriever(_facts())
    res = r.retrieve("When do cherry blossoms bloom in Jeju?", top_k=3)
    assert not res.is_empty()
    assert res.facts[0].subject == "Cherry blossoms"
    assert "March" in res.context


def test_retriever_no_match():
    r = TemporalRetriever(_facts())
    res = r.retrieve("What is the capital of France?", top_k=3)
    assert res.is_empty()
    assert "No temporal" in res.context


def test_retriever_loads_v120():
    """Integration: the produced v1.2.0 temporal layer loads and links."""
    path = ROOT / "artifacts" / "ontologies" / "1.2.0" / "temporal_facts.json"
    if not path.exists():
        pytest.skip("v1.2.0 temporal layer not built yet")
    r = TemporalRetriever.from_version("1.2.0")
    assert len(r.facts) > 0
    res = r.retrieve("When is typhoon season in Jeju?", top_k=5)
    assert not res.is_empty()
