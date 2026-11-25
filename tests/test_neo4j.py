"""
Unit tests for Neo4jClient and cypher_queries.

All tests run in offline mode (uri=None) or with a fully mocked driver —
no real Neo4j instance required.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.graph.neo4j_client import Neo4jClient, _node_to_concept, _rel_to_dict
from src.graph import cypher_queries as CQ


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def offline_client():
    """Neo4jClient in offline mode — no driver, all queries return []."""
    return Neo4jClient(uri=None)


def _make_mock_node(data: dict):
    """Return a mock neo4j Node-like object."""
    node = MagicMock()
    node.items.return_value = data.items()
    node.__getitem__ = lambda self, k: data[k]
    node.get = lambda k, default=None: data.get(k, default)
    return node


def _make_mock_rel(data: dict, start: dict, end: dict):
    """Return a mock neo4j Relationship-like object."""
    rel = MagicMock()
    rel.items.return_value = data.items()
    rel.get = lambda k, default=None: data.get(k, default)
    rel.start_node = _make_mock_node(start)
    rel.end_node   = _make_mock_node(end)
    return rel


def _make_connected_client(session_records: list | None = None):
    """
    Return a Neo4jClient whose driver is fully mocked.

    session_records: list of dicts returned by session.run(...).__iter__
    """
    client = Neo4jClient(uri=None)   # start offline
    client._offline = False

    mock_driver  = MagicMock()
    mock_session = MagicMock()
    mock_result  = session_records or []

    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__  = MagicMock(return_value=False)
    mock_session.run.return_value = iter(mock_result)
    mock_driver.session.return_value = mock_session

    client._driver = mock_driver
    return client, mock_session


# ── Offline mode ──────────────────────────────────────────────────────────────

class TestOfflineMode:
    def test_offline_flag(self, offline_client):
        assert offline_client.is_offline is True
        assert offline_client.is_connected is False

    def test_ping_returns_false(self, offline_client):
        assert offline_client.ping() is False

    def test_get_concept_returns_none(self, offline_client):
        assert offline_client.get_concept("Hallasan") is None

    def test_get_concepts_by_class_returns_empty(self, offline_client):
        assert offline_client.get_concepts_by_class("Attraction") == []

    def test_neighbours_returns_empty(self, offline_client):
        assert offline_client.neighbours("Hallasan") == []

    def test_extract_subgraph_returns_empty(self, offline_client):
        result = offline_client.extract_subgraph(["Hallasan", "Jeju"])
        assert result["concepts"] == []
        assert result["relations"] == []

    def test_search_concepts_returns_empty(self, offline_client):
        assert offline_client.search_concepts("beach") == []

    def test_stats_returns_zeros(self, offline_client):
        stats = offline_client.stats()
        assert stats["n_concepts"] == 0
        assert stats["n_relations"] == 0

    def test_bulk_load_concepts_noop(self, offline_client):
        result = offline_client.bulk_load_concepts(
            [{"name": "Hallasan", "concept_class": "Attraction", "confidence": 0.9}]
        )
        assert result == 0

    def test_bulk_load_relations_noop(self, offline_client):
        result = offline_client.bulk_load_relations(
            [{"subject": "Hallasan", "predicate": "locatedIn", "object": "Jeju"}]
        )
        assert result == 0

    def test_context_manager(self):
        with Neo4jClient(uri=None) as client:
            assert client.is_offline

    def test_repr_offline(self, offline_client):
        assert "offline" in repr(offline_client)


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_node_to_concept_basic(self):
        node = _make_mock_node({
            "name": "Hallasan", "concept_class": "Attraction",
            "confidence": 0.9, "aliases": ["Halla Mountain"], "source": "llm"
        })
        result = _node_to_concept(node)
        assert result["name"] == "Hallasan"
        assert result["concept_class"] == "Attraction"
        assert isinstance(result["aliases"], list)

    def test_node_to_concept_string_aliases(self):
        node = _make_mock_node({
            "name": "Jeju", "concept_class": "Destination",
            "confidence": 1.0, "aliases": "Cheju, Jeju Island", "source": "manual"
        })
        result = _node_to_concept(node)
        assert "Cheju" in result["aliases"]
        assert "Jeju Island" in result["aliases"]

    def test_node_to_concept_null_aliases(self):
        node = _make_mock_node({
            "name": "Beach", "concept_class": "Attraction",
            "confidence": 0.8, "aliases": None, "source": ""
        })
        result = _node_to_concept(node)
        assert result["aliases"] == []

    def test_rel_to_dict(self):
        a = _make_mock_node({"name": "Hallasan"})
        b = _make_mock_node({"name": "Jeju"})
        r = MagicMock()
        r.get = lambda k, d=None: {"predicate": "locatedIn", "confidence": 0.9, "source": "llm"}.get(k, d)
        result = _rel_to_dict(a, r, b)
        assert result["subject"]   == "Hallasan"
        assert result["object"]    == "Jeju"
        assert result["predicate"] == "locatedIn"
        assert result["confidence"] == pytest.approx(0.9)


# ── Cypher query strings ──────────────────────────────────────────────────────

class TestCypherQueries:
    def test_merge_concept_is_parameterised(self):
        assert "$name" in CQ.MERGE_CONCEPT
        assert "$concept_class" in CQ.MERGE_CONCEPT
        assert "$confidence" in CQ.MERGE_CONCEPT

    def test_merge_relation_is_parameterised(self):
        assert "$subject" in CQ.MERGE_RELATION
        assert "$object"  in CQ.MERGE_RELATION
        assert "$predicate" in CQ.MERGE_RELATION

    def test_shortest_path_uses_max_hops_param(self):
        assert "$max_hops" in CQ.SHORTEST_PATH

    def test_node_label_consistent(self):
        assert CQ.NODE_LABEL in CQ.MERGE_CONCEPT
        assert CQ.NODE_LABEL in CQ.GET_CONCEPT_BY_NAME
        assert CQ.NODE_LABEL in CQ.GET_CONCEPTS_BY_CLASS

    def test_rel_type_consistent(self):
        assert CQ.REL_TYPE in CQ.OUTGOING_NEIGHBOURS
        assert CQ.REL_TYPE in CQ.INCOMING_NEIGHBOURS
        assert CQ.REL_TYPE in CQ.SHORTEST_PATH

    def test_create_indexes_list_not_empty(self):
        assert len(CQ.CREATE_INDEXES) >= 4
        for stmt in CQ.CREATE_INDEXES:
            assert "CREATE" in stmt.upper()


# ── Connected client queries (mock driver) ────────────────────────────────────

class TestConnectedQueries:

    def test_get_concept_found(self):
        node = _make_mock_node({
            "name": "Hallasan", "concept_class": "Attraction",
            "confidence": 0.9, "aliases": [], "source": "llm"
        })
        record = {"c": node}
        client, session = _make_connected_client([record])
        session.run.return_value = iter([record])

        result = client.get_concept("Hallasan")
        assert result is not None
        assert result["name"] == "Hallasan"

    def test_get_concept_not_found(self):
        client, session = _make_connected_client([])
        session.run.return_value = iter([])
        result = client.get_concept("NonexistentConcept")
        assert result is None

    def test_get_concepts_by_class(self):
        nodes = [
            {"c": _make_mock_node({"name": f"Beach{i}", "concept_class": "Beach",
                                    "confidence": 0.8, "aliases": [], "source": ""})}
            for i in range(3)
        ]
        client, session = _make_connected_client(nodes)
        session.run.return_value = iter(nodes)

        results = client.get_concepts_by_class("Beach")
        assert len(results) == 3
        assert all(r["concept_class"] == "Beach" for r in results)

    def test_neighbours_out(self):
        a_node = _make_mock_node({"name": "Hallasan", "concept_class": "Attraction",
                                   "confidence": 0.9, "aliases": [], "source": ""})
        b_node = _make_mock_node({"name": "Jeju", "concept_class": "Destination",
                                   "confidence": 1.0, "aliases": [], "source": ""})
        rel    = MagicMock()
        rel.get = lambda k, d=None: {"predicate": "locatedIn", "confidence": 0.9, "source": "llm"}.get(k, d)

        record = {"a": a_node, "r": rel, "b": b_node}
        client, session = _make_connected_client([record])
        session.run.return_value = iter([record])

        results = client.neighbours("Hallasan", direction="out")
        assert len(results) == 1
        assert results[0]["predicate"] == "locatedIn"
        assert results[0]["subject"]   == "Hallasan"
        assert results[0]["object"]    == "Jeju"

    def test_neighbours_with_predicate_filter(self):
        a_node = _make_mock_node({"name": "Hallasan", "concept_class": "Attraction",
                                   "confidence": 0.9, "aliases": [], "source": ""})
        b_node = _make_mock_node({"name": "Jeju", "concept_class": "Destination",
                                   "confidence": 1.0, "aliases": [], "source": ""})
        rel = MagicMock()
        rel.get = lambda k, d=None: {"predicate": "locatedIn", "confidence": 0.9, "source": "llm"}.get(k, d)
        record = {"a": a_node, "r": rel, "b": b_node}

        client, session = _make_connected_client([record])
        session.run.return_value = iter([record])

        results = client.neighbours("Hallasan", direction="out", predicates=["locatedIn"])
        assert len(results) == 1

    def test_search_concepts_fulltext(self):
        node   = _make_mock_node({"name": "Hallasan", "concept_class": "Attraction",
                                   "confidence": 0.9, "aliases": [], "source": ""})
        record = {"c": node, "score": 1.5}
        client, session = _make_connected_client([record])
        session.run.return_value = iter([record])

        results = client.search_concepts("hallasan")
        assert len(results) == 1
        assert results[0]["name"] == "Hallasan"

    def test_shortest_path_found(self):
        path_mock = MagicMock()
        src_node  = _make_mock_node({"name": "Hallasan"})
        tgt_node  = _make_mock_node({"name": "Jeju"})
        rel_mock  = MagicMock()
        rel_mock.items.return_value = {"predicate": "locatedIn"}.items()
        rel_mock.get = lambda k, d=None: {"predicate": "locatedIn"}.get(k, d)

        path_mock.nodes         = [src_node, tgt_node]
        path_mock.relationships = [rel_mock]

        record = {"path": path_mock}
        client, session = _make_connected_client([record])
        session.run.return_value = iter([record])

        result = client.shortest_path("Hallasan", "Jeju")
        assert result["found"]  is True
        assert result["length"] == 1
        assert "Hallasan" in result["nodes"]
        assert "Jeju"     in result["nodes"]

    def test_shortest_path_not_found(self):
        client, session = _make_connected_client([])
        session.run.return_value = iter([])
        result = client.shortest_path("Hallasan", "Eiffel Tower")
        assert result["found"]  is False
        assert result["length"] == 0

    def test_stats_returns_counts(self):
        stats_record = {
            "n_concepts": 8348, "n_relations": 38868,
            "n_classes": 12, "n_predicates": 25,
        }
        class_record1 = {"cls": "Attraction", "cnt": 3000}
        class_record2 = {"cls": "Restaurant", "cnt": 1500}

        client, session = _make_connected_client()
        # First call → total_counts; second call → concept_stats
        session.run.side_effect = [
            iter([stats_record]),
            iter([class_record1, class_record2]),
        ]

        result = client.stats()
        assert result["n_concepts"]  == 8348
        assert result["n_relations"] == 38868
        assert result["class_counts"]["Attraction"] == 3000


# ── Bulk loading ──────────────────────────────────────────────────────────────

class TestBulkLoading:

    def test_load_ontology_version_missing_path(self, offline_client):
        with pytest.raises(FileNotFoundError):
            offline_client.load_ontology_version("99.99.99")

    def test_load_ontology_version_ok(self, tmp_path):
        ont_dir = tmp_path / "0.7.0"
        ont_dir.mkdir()
        concepts  = [{"name": "Hallasan", "concept_class": "Attraction",
                      "confidence": 0.9, "aliases": []}]
        relations = [{"subject": "Hallasan", "predicate": "locatedIn",
                      "object": "Jeju", "confidence": 0.8}]
        (ont_dir / "concepts.json").write_text(json.dumps(concepts))
        (ont_dir / "relations.json").write_text(json.dumps(relations))

        client = Neo4jClient(uri=None)   # offline → all writes are no-ops
        result = client.load_ontology_version("0.7.0", ontology_dir=tmp_path)
        # offline: loaded counts are 0, but method should not raise
        assert "version" in result
        assert result["version"] == "0.7.0"
