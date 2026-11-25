"""
Tests for all 9 FastAPI endpoints using httpx AsyncClient + TestClient.

The LLM and ontology are mocked so these tests run without GPU/disk.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# ── Minimal mock state injected via dependency override ───────────────────────

_MOCK_CONCEPT = {"name": "Hallasan", "concept_class": "Attraction", "confidence": 0.9}
_MOCK_RELATION = {"subject": "Hallasan", "predicate": "locatedIn", "object": "Jeju"}

_MOCK_STATE: dict = {
    "concepts":         [_MOCK_CONCEPT],
    "relations":        [_MOCK_RELATION],
    "ontology_version": "0.7.0",
    "model_name":       "mock-llm",
    "graph":            None,
    "linker":           None,
    "llm_client":       None,
}


def _make_app(extra_state: dict | None = None):
    """Build the FastAPI app with lifespan replaced by a mock state dict."""
    from src.api.routes import router, get_state
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    state = {**_MOCK_STATE, **(extra_state or {})}

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.app_state = state
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture()
def client():
    return TestClient(_make_app())


# ── 1. GET /health ────────────────────────────────────────────────────────────

def test_health_ok(client):
    with patch("torch.cuda.is_available", return_value=False):
        r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ontology_version"] == "0.7.0"
    assert body["n_concepts"] == 1
    assert body["n_relations"] == 1
    assert body["model"] == "mock-llm"
    assert isinstance(body["gpu_available"], bool)


# ── 2. POST /profile-data ─────────────────────────────────────────────────────

def test_profile_data_missing_split(client):
    r = client.post("/api/v1/profile-data", json={"split": "nonexistent_xyz"})
    assert r.status_code == 404


def test_profile_data_ok(client, tmp_path, monkeypatch):
    split_file = tmp_path / "train.jsonl"
    records = [
        {"Input": "What to do in Jeju", "Output": "Visit Hallasan mountain"},
        {"Input": "Where to eat sushi in Jeju", "Output": "Try the local haenyeo restaurants"},
    ]
    split_file.write_text("\n".join(json.dumps(r) for r in records))

    import src.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "ROOT", tmp_path.parent)

    data_dir = tmp_path.parent / "data" / "splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(split_file.read_text())

    r = client.post("/api/v1/profile-data", json={"split": "train"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_records"] == 2
    assert body["split"] == "train"
    assert isinstance(body["avg_input_len"], float)
    assert isinstance(body["top_tokens"], list)


# ── 3. POST /build-ontology ───────────────────────────────────────────────────

def test_build_ontology_returns_task_id(client):
    r = client.post("/api/v1/build-ontology")
    assert r.status_code == 200
    body = r.json()
    assert "task_id" in body
    assert body["status"] == "queued"
    assert len(body["task_id"]) > 0


# ── 4. POST /validate-ontology ────────────────────────────────────────────────

def test_validate_ontology_missing_version(client):
    r = client.post("/api/v1/validate-ontology", json={"ontology_version": "99.99.99"})
    assert r.status_code == 404


def test_validate_ontology_ok(client, tmp_path, monkeypatch):
    import src.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "ROOT", tmp_path)

    ont_dir = tmp_path / "artifacts" / "ontologies" / "0.7.0"
    ont_dir.mkdir(parents=True)
    (ont_dir / "concepts.json").write_text(json.dumps([_MOCK_CONCEPT]))
    (ont_dir / "relations.json").write_text(json.dumps([_MOCK_RELATION]))
    (ont_dir / "rules.json").write_text(json.dumps([]))

    mock_validator = MagicMock()
    mock_validator.return_value.validate_batch.return_value = []

    with patch("src.ontology.shacl_validator.SHACLValidator", mock_validator), \
         patch("src.utils.io.load_json", side_effect=[
             [_MOCK_CONCEPT], [_MOCK_RELATION], []
         ]):
        r = client.post("/api/v1/validate-ontology", json={"ontology_version": "0.7.0"})

    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is True
    assert body["n_violations"] == 0


# ── 5. POST /update-ontology ──────────────────────────────────────────────────

def test_update_ontology_ok(client):
    mock_vm = MagicMock()
    mock_vm.return_value.load_version.return_value = {
        "concepts":  [_MOCK_CONCEPT],
        "relations": [_MOCK_RELATION],
        "rules":     [],
        "schema":    {},
    }
    mock_vm.return_value.detect_conflicts.return_value = []
    mock_vm.return_value.add_concepts_incremental.return_value = (
        [_MOCK_CONCEPT],
        {"added": 1, "merged": 0, "flagged": 0},
    )

    mock_extractor = MagicMock()
    mock_extractor.return_value.extract.return_value = {"concepts": [_MOCK_CONCEPT]}

    payload = {
        "records": [{"Output": "Visit Hallasan"}],
        "policy": "priority",
        "ontology_version": "0.7.0",
    }

    with patch("src.ontology.versioning.OntologyVersionManager", mock_vm), \
         patch("src.ontology.offline_extractor.OfflineExtractor", mock_extractor):
        r = client.post("/api/v1/update-ontology", json=payload)

    assert r.status_code == 200
    body = r.json()
    assert body["new_version"] == "0.7.1"
    assert body["concepts_added"] == 1
    assert body["conflicts_detected"] == 0


# ── 6. POST /query ────────────────────────────────────────────────────────────

def test_query_no_llm_offline(client):
    r = client.post("/api/v1/query", json={
        "query": "What are the best beaches in Jeju?",
        "mode":  "no_rag",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "no_rag"
    assert "[offline]" in body["answer"]
    assert isinstance(body["grounding_score"], float)
    assert body["trace_id"] != ""


def test_query_with_mock_llm(client):
    mock_llm    = MagicMock()
    mock_llm.call.return_value = "Hallasan is a great attraction in Jeju."
    mock_linker = MagicMock()
    mock_linker.link.return_value = [{"concept": "Hallasan"}]

    app = _make_app(extra_state={"llm_client": mock_llm, "linker": mock_linker})
    tc  = TestClient(app)

    r = tc.post("/api/v1/query", json={
        "query": "Tell me about Hallasan",
        "mode":  "onto_rag",
    })
    assert r.status_code == 200
    body = r.json()
    assert "Hallasan" in body["answer"]
    assert body["grounding_score"] > 0  # "Hallasan" is in used_concepts and answer


def test_query_invalid_mode(client):
    r = client.post("/api/v1/query", json={
        "query": "What is Jeju?",
        "mode":  "bad_mode",
    })
    assert r.status_code == 422


def test_query_too_short(client):
    r = client.post("/api/v1/query", json={"query": "Hi", "mode": "no_rag"})
    assert r.status_code == 422


# ── 6b. Jeju guard ────────────────────────────────────────────────────────────

def test_query_non_jeju_blocked_seoul(client):
    """Queries about non-Jeju Korean cities must be rejected with 400."""
    r = client.post("/api/v1/query", json={
        "query": "What are the best restaurants in Seoul?",
        "mode":  "no_rag",
    })
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["error"] == "non_jeju_query"
    assert "Seoul" in body["detail"]["message"]


def test_query_non_jeju_blocked_bali(client):
    r = client.post("/api/v1/query", json={
        "query": "Best beaches in Bali?",
        "mode":  "no_rag",
    })
    assert r.status_code == 400
    assert "Bali" in r.json()["detail"]["message"]


def test_query_non_jeju_blocked_tokyo(client):
    r = client.post("/api/v1/query", json={
        "query": "Things to do in Tokyo Japan",
        "mode":  "no_rag",
    })
    assert r.status_code == 400


def test_query_jeju_keyword_allowed(client):
    """Queries containing a Jeju landmark must pass the guard and get a response."""
    r = client.post("/api/v1/query", json={
        "query": "What can I do near Hallasan?",
        "mode":  "no_rag",
    })
    assert r.status_code == 200


def test_query_no_location_allowed(client):
    """Queries with no location should be allowed (assumed Jeju context)."""
    r = client.post("/api/v1/query", json={
        "query": "What are some good hiking trails?",
        "mode":  "no_rag",
    })
    assert r.status_code == 200


def test_query_non_jeju_country_blocked(client):
    r = client.post("/api/v1/query", json={
        "query": "What are the best places to visit in Pakistan?",
        "mode":  "no_rag",
    })
    assert r.status_code == 400
    assert "Pakistan" in r.json()["detail"]["message"]


# ── 7. POST /explain ──────────────────────────────────────────────────────────

def test_explain_ok(client):
    mock_report = MagicMock()
    mock_report.grounding_precision     = 0.75
    mock_report.evidence_trace_coverage = 0.60

    with patch("src.evaluation.explainability_eval.ExplainabilityEvaluator") as MockEval:
        MockEval.return_value.evaluate.return_value = mock_report
        r = client.post("/api/v1/explain", json={
            "query":  "What can I do near Hallasan?",
            "answer": "You can hike Hallasan and visit nearby cafes.",
            "mode":   "onto_rag",
        })

    assert r.status_code == 200
    body = r.json()
    assert body["grounding_score"] == pytest.approx(0.75)
    assert body["evidence_coverage"] == pytest.approx(0.60)
    assert len(body["trace"]) == 3


# ── 8. POST /benchmark ────────────────────────────────────────────────────────

def test_benchmark_no_test_split(client):
    r = client.post("/api/v1/benchmark", json={"n_queries": 5, "mode": "no_rag"})
    # test.jsonl won't exist in CI — should 404
    assert r.status_code in (200, 404)


def test_benchmark_ok(client, tmp_path, monkeypatch):
    import src.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "ROOT", tmp_path)

    split_dir = tmp_path / "data" / "splits"
    split_dir.mkdir(parents=True)
    records = [{"Input": "Best attraction in Jeju?", "Output": "Hallasan mountain"}]
    (split_dir / "test.jsonl").write_text(json.dumps(records[0]))

    from src.evaluation.benchmark import BenchmarkItem as BI
    mock_bench = MagicMock()
    mock_item  = MagicMock()
    mock_item.item_id           = 0
    mock_item.query             = "Best attraction in Jeju?"
    mock_item.expected_answer   = "Hallasan mountain"
    mock_item.expected_concepts = ["Hallasan"]
    mock_bench.items = [mock_item]

    mock_report = MagicMock()
    mock_report.hallucination_score = 0.1

    with patch("src.evaluation.benchmark.TourismBenchmark") as MockBench, \
         patch("src.evaluation.hallucination_eval.HallucinationEvaluator") as MockHalluc:
        MockBench.build.return_value  = mock_bench
        MockHalluc.return_value.evaluate.return_value = mock_report

        r = client.post("/api/v1/benchmark", json={"n_queries": 1, "mode": "no_rag"})

    assert r.status_code == 200
    body = r.json()
    assert body["n_queries"] == 1
    assert "hallucination_rate" in body["aggregate_metrics"]
    assert len(body["per_item"]) == 1


# ── 9. GET /export-ontology ───────────────────────────────────────────────────

def test_export_unsupported_format(client):
    r = client.get("/api/v1/export-ontology?version=0.7.0&format=csv")
    assert r.status_code == 400


def test_export_missing_file(client):
    r = client.get("/api/v1/export-ontology?version=0.7.0&format=jsonld")
    assert r.status_code == 404


def test_export_ok(client, tmp_path, monkeypatch):
    import src.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "ROOT", tmp_path)

    ont_dir = tmp_path / "artifacts" / "ontologies"
    ont_dir.mkdir(parents=True)
    export_file = ont_dir / "tourism_v0.7.0.jsonld"
    export_file.write_text('{"@context": {}}')

    r = client.get("/api/v1/export-ontology?version=0.7.0&format=jsonld")
    assert r.status_code == 200
    assert "ld+json" in r.headers["content-type"]
