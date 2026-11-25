"""Tests for ontology schema, validator, versioning, and exporters."""
import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ontology.schema import (
    TourismOntologySchema,
    OntologyConcept,
    OntologyRelation,
    OntologyCandidate,
    ProvenanceMetadata,
)
from src.ontology.validator import OntologyValidator, ValidationIssue
from src.ontology.versioning import OntologyVersionManager, OntologyChange
from src.ontology.exporters import OntologyExporter

SCHEMA = TourismOntologySchema


# ── Schema Tests ──────────────────────────────────────────────────────────────

def test_valid_class():
    assert SCHEMA.is_valid_class("City")
    assert SCHEMA.is_valid_class("Beach")
    assert SCHEMA.is_valid_class("Museum")
    assert not SCHEMA.is_valid_class("Hotel")  # use "Accommodation"
    assert not SCHEMA.is_valid_class("UnknownClass")


def test_valid_relation():
    assert SCHEMA.is_valid_relation("located_in")
    assert SCHEMA.is_valid_relation("reachable_by")
    assert not SCHEMA.is_valid_relation("is_adjacent_to")


def test_class_hierarchy():
    assert SCHEMA.get_parent_class("City") == "Destination"
    assert SCHEMA.get_parent_class("Beach") == "NaturalAttraction"
    assert SCHEMA.get_parent_class("Museum") == "CulturalAttraction"
    assert SCHEMA.get_parent_class("Destination") is None


def test_schema_to_dict():
    d = SCHEMA.to_dict()
    assert "concept_classes" in d
    assert "relation_types" in d
    assert "class_hierarchy" in d
    assert len(d["concept_classes"]) > 10


# ── Concept / Relation Tests ──────────────────────────────────────────────────

def test_concept_roundtrip():
    prov = ProvenanceMetadata(source_record_id=1, confidence=0.9)
    concept = OntologyConcept("Ubud", "City", ["Ubood"], {}, prov)
    d = concept.to_dict()
    assert d["name"] == "Ubud"
    assert d["concept_class"] == "City"
    assert "Ubood" in d["aliases"]
    restored = OntologyConcept.from_dict(d)
    assert restored.name == "Ubud"
    assert restored.provenance.confidence == 0.9


def test_relation_roundtrip():
    rel = OntologyRelation("Ubud", "located_in", "Bali", ProvenanceMetadata(confidence=0.8))
    d = rel.to_dict()
    assert d["subject"] == "Ubud"
    assert d["predicate"] == "located_in"
    assert d["object"] == "Bali"
    restored = OntologyRelation.from_dict(d)
    assert restored.obj == "Bali"


# ── Validator Tests ───────────────────────────────────────────────────────────

def _make_candidate(concepts=None, relations=None, record_id=0):
    c = OntologyCandidate(record_id=record_id)
    c.concepts = concepts or []
    c.relations = relations or []
    return c


def test_validator_clean_candidate():
    concept = OntologyConcept("Bali", "Island", [], {}, ProvenanceMetadata(source_record_id=1, confidence=0.9))
    rel = OntologyRelation("Bali", "located_in", "Indonesia", ProvenanceMetadata(confidence=0.9))
    candidate = _make_candidate([concept], [rel])
    v = OntologyValidator()
    result = v.validate_candidate(candidate)
    # Only warnings allowed (e.g., structural info), no errors
    errors = [i for i in result.issues if i.severity == "error"]
    assert len(errors) == 0


def test_validator_catches_bad_class():
    concept = OntologyConcept("Eiffel Tower", "Monument", [], {}, ProvenanceMetadata(source_record_id=1, confidence=0.9))
    candidate = _make_candidate([concept])
    v = OntologyValidator()
    result = v.validate_candidate(candidate)
    errors = [i for i in result.issues if i.severity == "error" and i.category == "schema"]
    assert len(errors) == 1


def test_validator_catches_bad_relation():
    concept = OntologyConcept("Bali", "Island", [], {}, ProvenanceMetadata(source_record_id=1, confidence=0.9))
    rel = OntologyRelation("Bali", "is_adjacent_to", "Java", ProvenanceMetadata(confidence=0.9))
    candidate = _make_candidate([concept], [rel])
    v = OntologyValidator()
    result = v.validate_candidate(candidate)
    errors = [i for i in result.issues if i.severity == "error" and i.category == "schema"]
    assert len(errors) == 1


def test_validator_self_loop():
    rel = OntologyRelation("Bali", "located_in", "Bali", ProvenanceMetadata(confidence=0.9))
    candidate = _make_candidate(relations=[rel])
    v = OntologyValidator()
    result = v.validate_candidate(candidate)
    warnings = [i for i in result.issues if i.severity == "warning" and i.category == "logical"]
    assert len(warnings) > 0


def test_validator_low_confidence():
    concept = OntologyConcept("SomeThing", "City", [], {}, ProvenanceMetadata(source_record_id=1, confidence=0.2))
    candidate = _make_candidate([concept])
    v = OntologyValidator(min_confidence=0.5)
    result = v.validate_candidate(candidate)
    warnings = [i for i in result.issues if i.category == "semantic"]
    assert len(warnings) > 0


# ── Versioning Tests ──────────────────────────────────────────────────────────

def test_version_save_load(tmp_path):
    vm = OntologyVersionManager(base_dir=tmp_path)
    concepts = [{"name": "Bali", "concept_class": "Island"}]
    relations = [{"subject": "Bali", "predicate": "located_in", "object": "Indonesia"}]
    vm.save_version("0.1.0", concepts, relations, [], {})

    loaded = vm.load_version("0.1.0")
    assert loaded["concepts"][0]["name"] == "Bali"
    assert loaded["relations"][0]["predicate"] == "located_in"


def test_version_list(tmp_path):
    vm = OntologyVersionManager(base_dir=tmp_path)
    vm.save_version("0.1.0", [], [], [], {})
    vm.save_version("0.2.0", [], [], [], {})
    versions = vm.list_versions()
    assert "0.1.0" in versions
    assert "0.2.0" in versions


def test_version_compare(tmp_path):
    vm = OntologyVersionManager(base_dir=tmp_path)
    vm.save_version("0.1.0", [{"name": "Bali", "concept_class": "Island"}], [], [], {})
    vm.save_version("0.2.0", [
        {"name": "Bali", "concept_class": "Island"},
        {"name": "Ubud", "concept_class": "City"},
    ], [{"subject": "Ubud", "predicate": "located_in", "object": "Bali"}], [], {})
    diff = vm.compare_versions("0.1.0", "0.2.0")
    assert "Ubud" in diff["concepts"]["added"]
    assert diff["relations"]["added"] == 1


# ── Exporter Tests ────────────────────────────────────────────────────────────

def test_export_jsonld(tmp_path):
    exporter = OntologyExporter(output_dir=tmp_path)
    concepts = [{"name": "Bali", "concept_class": "Island", "aliases": ["Ball island"], "provenance": {"confidence": 0.9}}]
    relations = [{"subject": "Bali", "predicate": "located_in", "object": "Indonesia"}]
    path = exporter.export_jsonld(concepts, relations, version="0.1.0")
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert "@context" in data
    assert "@graph" in data
    assert len(data["@graph"]) >= 1


def test_export_turtle_minimal(tmp_path):
    exporter = OntologyExporter(output_dir=tmp_path)
    concepts = [{"name": "Jeju", "concept_class": "Island", "aliases": [], "provenance": {"confidence": 0.95}}]
    relations = [{"subject": "Jeju", "predicate": "located_in", "object": "South Korea"}]
    path = exporter._export_turtle_minimal(concepts, relations, "0.1.0", None)
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "tourism:Jeju" in content
    assert "tourism:located_in" in content
