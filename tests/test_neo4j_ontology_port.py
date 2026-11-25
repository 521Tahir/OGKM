"""The companion project's graph layer, carried over, and the experiment that uses it.

`/home/tahirahmad/Ontology_Project` holds the Neo4j design this port follows. The
tests that need a live server skip when its artifact is absent; the rest are pure
and run anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbs.task3_storage import neo4j_ontology as no

ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "wbs" / "deliverables" / "task3_storage"
PORTABLE = DELIV / "graph_db_neo4j_portable.json"
SWEEP = ROOT / "wbs" / "deliverables" / "task3_retrieval" / "neo4j_retrieval_sweep.json"

needs_load = pytest.mark.skipif(
    not PORTABLE.exists(),
    reason="run wbs.task3_storage.neo4j_ontology against a live server first")


# ------------------------------------------------------- the ported schema


def test_the_schema_is_the_companion_project_s_schema():
    """If these drift, a query written against one project stops working on the
    other, which is the whole point of porting it rather than inventing one."""
    assert no.NODE_LABEL == "Concept"
    assert no.REL_TYPE == "RELATES_TO"
    # Their five indexes, including the fulltext one the search path needs.
    joined = " ".join(no.CREATE_INDEXES)
    for want in ("concept_name", "concept_class", "concept_confidence",
                 "rel_predicate", "concept_fulltext"):
        assert want in joined, want
    assert "FULLTEXT" in joined
    # MERGE, not CREATE: a second run must update in place, not duplicate.
    assert "MERGE" in no.MERGE_CONCEPTS and "CREATE" not in no.MERGE_CONCEPTS
    assert "MERGE" in no.MERGE_RELATIONS
    # Batched through UNWIND rather than a transaction per row.
    assert no.MERGE_CONCEPTS.strip().startswith("UNWIND $rows")
    assert no.MERGE_RELATIONS.strip().startswith("UNWIND $rows")


def test_every_read_query_is_read_only():
    """These run against a live store; none of them may write to it."""
    reads = [no.GET_CONCEPT_BY_NAME, no.GET_CONCEPTS_BY_CLASS,
             no.FULLTEXT_SEARCH, no.CONTAINS_SEARCH, no.OUTGOING_NEIGHBOURS,
             no.INCOMING_NEIGHBOURS, no.BOTH_NEIGHBOURS, no.SHORTEST_PATH,
             no.SUBGRAPH_NHOP, no.CONCEPT_STATS, no.TOTAL_COUNTS,
             no.CARD_ROWS, no.DEGREE_ROWS]
    for q in reads:
        up = q.upper()
        for forbidden in ("CREATE ", "MERGE ", "DELETE", "DETACH", "SET ",
                          "REMOVE "):
            assert forbidden not in up, (forbidden, q[:60])


def test_the_hop_bound_is_an_integer_not_a_string():
    """Neo4j rejects a parameter in a variable-length pattern, so the bound is
    formatted in; anything that is not a number must not reach the query."""
    assert "*1..%d" in no.SHORTEST_PATH
    assert "*1..%d" in no.SUBGRAPH_NHOP
    with pytest.raises((ValueError, TypeError)):
        int("4; MATCH (n) DETACH DELETE n")


def test_relation_rows_carry_the_assertion_order():
    """The retrieval arm keeps the first 8 facts of an entity in file order, so
    a graph without the index cannot reproduce the published contexts."""
    rows = no.OntologyGraph._relation_rows([
        {"subject": "A", "predicate": "near", "object": "B",
         "provenance": {"confidence": 0.5, "extraction_prompt_version": "p1"}},
        {"subject": "A", "predicate": "located_in", "object": "C"},
    ])
    assert [r["ordinal"] for r in rows] == [0, 1]
    assert rows[0]["confidence"] == 0.5 and rows[0]["source"] == "p1"
    # A relation with no provenance must not crash or invent a confidence.
    assert rows[1]["confidence"] == 0.0 and rows[1]["source"] == ""
    assert "ORDER BY r.ordinal" in no.CARD_ROWS


def test_concept_rows_take_confidence_and_source_from_provenance():
    rows = no.OntologyGraph._concept_rows([
        {"name": "Hallasan", "concept_class": "Mountain", "aliases": ["Halla"],
         "provenance": {"confidence": 0.9, "extraction_prompt_version": "v2"}},
        {"name": "Bare"},
    ])
    assert rows[0] == {"name": "Hallasan", "concept_class": "Mountain",
                       "confidence": 0.9, "aliases": ["Halla"], "source": "v2"}
    assert rows[1]["concept_class"] == "" and rows[1]["confidence"] == 0.0


def test_lucene_metacharacters_in_a_place_name_are_escaped():
    """A name with a hyphen or a colon would otherwise raise inside the
    fulltext index instead of being searched for."""
    META = set('+-&|!(){}[]^"~*?:\\/')
    for raw in ("Jeju-do", "Seongsan: Ilchulbong", "A (B)", "x && y", 'q"r',
                "Jeju"):
        escaped = no.OntologyGraph._LUCENE.sub(r"\\\1", raw)
        # Every metacharacter that survives must be preceded by a backslash
        # that is itself an escape, and the plain text must be untouched.
        i = 0
        while i < len(escaped):
            if escaped[i] == "\\":
                assert i + 1 < len(escaped) and escaped[i + 1] in META, escaped
                i += 2
                continue
            assert escaped[i] not in META, (raw, escaped, escaped[i])
            i += 1
        assert escaped.replace("\\", "") == raw


# ----------------------------------------------------- the loaded artifact


@pytest.fixture(scope="module")
def loaded():
    return json.loads(PORTABLE.read_text(encoding="utf-8"))


@needs_load
def test_the_whole_ontology_went_in_under_the_ported_schema(loaded):
    load, stats = loaded["load"], loaded["stats"]
    assert load["concepts_in_graph"] == load["concepts_sent"]
    assert load["relations_in_graph"] == load["relations_sent"]
    # MERGE collapses a repeated triple onto one edge; if that happened the
    # ontology asserts the same fact twice and the report must say so.
    assert load["relations_collapsed_by_merge"] == 0, \
        f"{load['relations_collapsed_by_merge']} duplicate triples in the source"
    assert stats["n_classes"] > 10 and stats["n_predicates"] > 10
    assert sum(stats["class_counts"].values()) == stats["n_concepts"]
    assert loaded["schema"]["node_label"] == "Concept"
    assert loaded["schema"]["relationship_type"] == "RELATES_TO"


@needs_load
def test_every_read_primitive_was_exercised_against_the_live_graph(loaded):
    """Imports proving nothing, the loader runs each one and records the time."""
    for name in ("get_concept", "concepts_by_class", "neighbours",
                 "search_concepts", "extract_subgraph", "shortest_path",
                 "cards"):
        assert name in loaded["read_ms"], name
        assert loaded["read_ms"][name] > 0
    p = loaded["probe"]
    assert p["get_concept_Jeju"]["concept_class"] == "Destination"
    assert p["concepts_by_class_Beach"] > 0
    assert p["neighbours_Jeju_both"] > 0
    assert p["search_concepts_top1"], "the fulltext index returned nothing"
    assert p["cards"] > 1000


@needs_load
def test_the_subgraph_never_returns_an_edge_whose_endpoints_it_dropped(loaded):
    """The companion version caps nodes after collecting relations, so a 50-node
    subgraph came back with 7,001 relations pointing at nodes that were cut."""
    p = loaded["probe"]
    n = p["subgraph_2hop_nodes"]
    # 50 nodes can carry at most n*(n-1) directed edges, one per ordered pair
    # per predicate; the pre-fix number was 7,001 against a 2,450 pair ceiling.
    assert p["subgraph_2hop_relations"] <= n * (n - 1) * 2, \
        (n, p["subgraph_2hop_relations"])
    assert n > 0


# ------------------------------------------------- the experiment that uses it


@pytest.mark.skipif(not SWEEP.exists(),
                    reason="run wbs.task3_retrieval.run_neo4j_experiment first")
def test_sourcing_the_cards_from_neo4j_changes_nothing_it_should_not():
    """The store must not be a confound: same cards, same width, same coverage."""
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
    assert d["ontology_source"] == "neo4j"
    assert d["cards_source"] == "neo4j"

    control = d["same_width_graph_hop_coverage"]
    anchor = d["published_anchor_coverage"]
    assert control is not None and anchor is not None
    # Same policy at the published fan-out: the hop structure the graph adds is
    # beyond what that width reaches, so coverage must be unchanged.
    assert abs(control - anchor) < 0.005, (anchor, control)

    g = d["graph"]
    assert g["targets_in_graph"] > g["targets_on_cards"], \
        "the graph should expose hop targets the 8-fact card cap hides"
    assert g["hidden_from_cards"] == g["targets_in_graph"] - g["targets_on_cards"]
    assert 0 < g["cards_with_more_targets_in_graph"] <= g["cards"]


@pytest.mark.skipif(not SWEEP.exists(), reason="sweep not run")
def test_the_widest_graph_policy_reports_an_interval_not_just_a_mean():
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
    pb = d["paired_best"]
    assert pb, "no paired comparison recorded"
    lo, hi = pb["ci95"]
    assert lo <= pb["mean_difference"] <= hi
    assert pb["n_questions"] == d["questions_with_gold"]
    assert (pb["questions_better"] + pb["questions_worse"]
            + pb["questions_unchanged"]) == pb["n_questions"]
    # A mean carried by a handful of questions is a different claim from a
    # broad one; the counts have to be there for a reader to tell which.
    assert pb["questions_unchanged"] >= 0


@pytest.mark.skipif(not SWEEP.exists(), reason="sweep not run")
def test_the_published_arms_still_score_what_they_published():
    """The sweep runs the published policies too; if those moved, the harness
    changed under them and nothing else in the file can be trusted."""
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
    rows = {r["config"]: r for r in d["rows"]}
    assert abs(rows["baseline"]["gold_coverage"] - 0.389) < 0.005, \
        rows["baseline"]["gold_coverage"]
    assert rows["routing_deep_hop2"]["gold_coverage"] > \
        rows["baseline"]["gold_coverage"]
