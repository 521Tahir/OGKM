"""The Cypher path has to ask the same questions as the SPARQL path, or it proves nothing.

These tests need no server: the mapping functions and the question set are pure, and
the measured deliverable is checked only when it exists, so a clone without a JVM
skips the measurement assertions instead of failing them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbs.task3_storage import neo4j_store as ns
from wbs.task3_storage.sparql_store import QUERIES as SPARQL_QUERIES

ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "wbs" / "deliverables" / "task3_storage"
NEO_JSON = DELIV / "graph_db_neo4j.json"

needs_run = pytest.mark.skipif(
    not NEO_JSON.exists(),
    reason="graph_db_neo4j.json absent; run wbs.task3_storage.neo4j_store "
           "against a live server first")


# --------------------------------------------------------------- pure mapping


def test_labels_and_relationship_types_are_valid_and_recognisable():
    assert ns.label_of("NaturalAttraction") == "NaturalAttraction"
    assert ns.label_of("Travel Constraint") == "Travel_Constraint"
    assert ns.label_of("") == "Concept"
    assert ns.label_of(None) == "Concept"
    assert ns.reltype_of("located_in") == "LOCATED_IN"
    assert ns.reltype_of("has-risk") == "HAS_RISK"
    assert ns.reltype_of(None) == "RELATED_TO"
    # Nothing that would need escaping should survive the sanitiser.
    for raw in ("a b", "a-b", "a.b", "a/b", "a`b"):
        assert ns.LABEL_OK.search(ns.label_of(raw)) is None
        assert ns.LABEL_OK.search(ns.reltype_of(raw)) is None


def test_both_mappings_ask_the_same_nine_questions_as_the_sparql_path():
    sparql_names = [name for name, _, _ in SPARQL_QUERIES]
    for mapping in ("typed", "generic"):
        qs = ns.queries(mapping)
        assert [n for n, _, _ in qs] == sparql_names, mapping
        assert [q for _, q, _ in qs] == [q for _, q, _ in SPARQL_QUERIES], mapping
        assert all(cypher.strip() for _, _, cypher in qs)


def test_the_generic_mapping_filters_predicates_as_properties():
    """The point of the second mapping is the property filter; assert it is there."""
    generic = {n: c for n, _, c in ns.queries("generic")}
    typed = {n: c for n, _, c in ns.queries("typed")}
    assert "RELATES_TO" in generic["Q3 dining"]
    assert "predicate:" in generic["Q3 dining"]
    assert ":SERVED_AT" in typed["Q3 dining"]
    assert "RELATES_TO" not in typed["Q3 dining"]


def test_store_parity_reads_the_version_the_rdf_path_actually_loads():
    p = ns.store_parity("1.4.0", 1565, 7909)
    assert p["file_store_version"] == "1.4.0"
    if ns.TTL.exists():
        # Whatever the answer is, it must come from the file rather than a constant.
        assert p["rdf_store_declared_version"], "no owl:versionInfo found in the Turtle"
        assert isinstance(p["rdf_store_named_individuals"], int)
        assert p["same_version"] == (p["rdf_store_declared_version"] == "1.4.0")
        assert (p["note"] is None) == p["same_version"]


def test_node_and_edge_rows_carry_provenance_not_just_structure():
    concepts = [{"name": "A", "concept_class": "Beach", "aliases": ["a"],
                 "attributes": {"lat": 1.0, "lon": 2.0},
                 "provenance": {"confidence": 0.9, "extraction_prompt_version": "x",
                                "validator_status": "pass"}}]
    relations = [{"subject": "A", "predicate": "near", "object": "B",
                  "provenance": {"confidence": 0.5,
                                 "extraction_prompt_version": "y"}}]
    n = ns._node_rows(concepts)[0]
    assert (n["label"], n["confidence"], n["source"], n["lat"]) == \
        ("Beach", 0.9, "x", 1.0)
    e = ns._edge_rows(relations)[0]
    assert (e["reltype"], e["confidence"], e["source"]) == ("NEAR", 0.5, "y")
    # A concept with no provenance block must not crash or invent a confidence.
    bare = ns._node_rows([{"name": "B"}])[0]
    assert bare["confidence"] is None and bare["label"] == "Concept"


# -------------------------------------------------------- the measured artifact


@pytest.fixture(scope="module")
def measured():
    return json.loads(NEO_JSON.read_text(encoding="utf-8"))


@needs_run
def test_the_whole_ontology_was_loaded_under_every_mapping(measured):
    for mapping, r in measured["mappings"].items():
        load = r["load"]
        assert load["nodes"] == measured["concepts"], mapping
        assert load["relationships"] == measured["relations"], mapping
        assert load["edges_not_created"] == 0, mapping
        assert len(r["queries"]) == len(SPARQL_QUERIES), mapping
        assert all(q["error"] is None for q in r["queries"]), mapping


@needs_run
def test_the_typed_mapping_really_is_typed(measured):
    typed = measured["mappings"]["typed"]
    generic = measured["mappings"]["generic"]
    # Idiomatic: a label per ontology class and a relationship type per predicate.
    assert len(typed["labels"]) > 10, typed["labels"]
    assert len(typed["relationship_types"]) > 10, typed["relationship_types"]
    assert "Concept" in typed["labels"]
    # The generic mapping is the triplestore emulation: exactly one of each.
    assert generic["labels"] == ["GConcept"]
    assert generic["relationship_types"] == ["RELATES_TO"]


@needs_run
def test_row_counts_agree_with_the_rdf_paths_except_where_the_versions_differ(measured):
    agree = measured["vs_rdflib"]
    assert agree, "no comparison against the SPARQL path was recorded"
    disagreements = measured["row_count_disagreements"]
    assert disagreements == [c["query"] for c in agree if not c["same"]]
    parity = measured["store_parity"]
    if parity["same_version"]:
        # Same ontology on both paths: every question must return the same rows.
        assert disagreements == [], disagreements
    else:
        # Different versions: a disagreement is expected, and must be explained.
        assert parity["note"], "a version mismatch with no explanation"
        assert parity["file_store_concepts"] != \
            parity["rdf_store_named_individuals"]


@needs_run
def test_the_mapping_verdict_does_not_claim_more_than_the_numbers(measured):
    verdict = ns.mapping_verdict(measured)
    t = {q["query"]: q["p50_ms"] for q in measured["mappings"]["typed"]["queries"]}
    g = {q["query"]: q["p50_ms"] for q in measured["mappings"]["generic"]["queries"]}
    wide = [n for n in t
            if t[n] and g[n] and (g[n] / t[n] >= 2.0 or g[n] / t[n] <= 0.5)]
    if not wide:
        assert "inconclusive" in verdict
        assert "faster" not in verdict
    else:
        assert "inconclusive" not in verdict
        for name in wide:
            assert name in verdict, name


@needs_run
def test_the_markdown_reports_the_version_defect_rather_than_hiding_it(measured):
    md = (DELIV / "graph_db_neo4j.md").read_text(encoding="utf-8")
    parity = measured["store_parity"]
    if not parity["same_version"]:
        assert parity["rdf_store_declared_version"] in md
        assert "P0001" in md
        assert "has_designation" in md
    for mapping in measured["mappings"]:
        assert mapping in md
    assert str(len(measured["mappings"]["typed"]["relationship_types"])) in md


# ------------------------------------------------- the self-contained explorer

EXPLORER = DELIV / "ontology_explorer.html"


@pytest.mark.skipif(not EXPLORER.exists(),
                    reason="run wbs.task3_storage.ontology_explorer first")
def test_the_explorer_is_self_contained_and_carries_the_whole_ontology():
    """It has to work from file:// with no network, or it is not a fallback."""
    import re
    html = EXPLORER.read_text(encoding="utf-8")
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                     html, re.S)
    assert blob, "no embedded data block"
    d = json.loads(blob.group(1))

    src = json.loads((ROOT / "artifacts" / "ontologies" / d["version"] /
                      "concepts.json").read_text(encoding="utf-8"))
    rel = json.loads((ROOT / "artifacts" / "ontologies" / d["version"] /
                      "relations.json").read_text(encoding="utf-8"))
    assert len(d["nodes"]) == len(src)
    assert len(d["edges"]) + d["dropped_edges"] == len(rel)
    assert d["classes"] == sorted({c.get("concept_class") or "Concept" for c in src})
    assert d["predicates"] == sorted({r["predicate"] for r in rel})
    assert sum(d["class_counts"]) == len(src)

    # No external fetch of any kind: no CDN, no remote script, no stylesheet link.
    assert "<link" not in html.lower()
    assert "cdn" not in html.lower()
    for marker in ("src=\"http", "src='http", "@import"):
        assert marker not in html, marker
    assert html.count("<script") == html.count("</script>") == 2


@pytest.mark.skipif(not EXPLORER.exists(),
                    reason="run wbs.task3_storage.ontology_explorer first")
def test_the_explorer_shows_what_the_rdf_paths_are_missing():
    import re
    html = EXPLORER.read_text(encoding="utf-8")
    d = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        html, re.S).group(1))
    gov = [s for s, _ in d["sources"] if s.startswith("governance/")]
    if gov:
        # The page must say so rather than leaving the reader to notice.
        assert "governance" in html.lower()
        assert "graph_db_neo4j.md" in html


POSTER = ROOT / "wbs" / "deliverables" / "figures" / "fig17_ontology_whole.png"
LAYOUT = DELIV / "ontology_layout.json"


@pytest.mark.skipif(not LAYOUT.exists(),
                    reason="run wbs.task3_storage.ontology_poster first")
def test_the_poster_layout_covers_every_concept_and_is_reproducible():
    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    src = json.loads((ROOT / "artifacts" / "ontologies" / lay["version"] /
                      "concepts.json").read_text(encoding="utf-8"))
    # Row order depends on the source: Cypher returns concepts sorted by name,
    # the file store returns them in insertion order. Coverage is what the figure
    # needs, and the explorer matches coordinates by name rather than by index.
    assert sorted(lay["order"]) == sorted(c["name"] for c in src)
    assert len(lay["order"]) == len(set(lay["order"])), "a name appears twice"
    assert lay["seed"] == 7, "an unseeded layout is a different picture each run"
    for kind in ("force", "clustered"):
        if kind not in lay:
            continue
        pts = lay[kind]
        assert len(pts) == len(src), kind
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Normalised into the unit box, and actually spread rather than collapsed.
        assert max(abs(min(xs)), abs(max(xs)),
                   abs(min(ys)), abs(max(ys))) <= 1.0001, kind
        assert (max(xs) - min(xs)) > 0.5 and (max(ys) - min(ys)) > 0.5, kind
        assert len({(round(x, 3), round(y, 3)) for x, y in pts}) > 0.9 * len(pts), \
            f"{kind}: more than a tenth of the nodes share a position"


@pytest.mark.skipif(not (POSTER.exists() and EXPLORER.exists()),
                    reason="poster or explorer not built")
def test_the_explorer_draws_the_same_picture_as_the_poster():
    """Two views of one ontology must not disagree about where anything is."""
    import re
    html = EXPLORER.read_text(encoding="utf-8")
    d = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        html, re.S).group(1))
    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    assert d["layouts"], "the explorer embedded no layout"
    for kind, coords in d["layouts"].items():
        for i, name in enumerate(n[0] for n in d["nodes"]):
            j = lay["order"].index(name)
            assert coords[i] == lay[kind][j], (kind, name)
            break   # one spot check per layout; full equality is O(n^2) here
        assert len(coords) == len(d["nodes"])


# ------------------------------------------------- the ontology read out of Neo4j

SNAPSHOT = DELIV / "ontology_from_neo4j.json"

needs_export = pytest.mark.skipif(
    not SNAPSHOT.exists(),
    reason="run wbs.task3_storage.neo4j_export against a live server first")


def test_the_export_queries_read_the_unsanitised_vocabulary():
    """Reconstructing a predicate from a relationship type would be lossy."""
    from wbs.task3_storage import neo4j_export as nx
    # `has-risk` and `HAS_RISK` are not the same string, and only one of them is
    # the ontology's predicate; the export must read the stored property.
    assert "r.predicate" in nx.RELATION_CYPHER
    assert "n.concept_class" in nx.CONCEPT_CYPHER
    assert "type(r)" in nx.RELATION_CYPHER      # kept, but alongside, not instead
    for q in (nx.CONCEPT_CYPHER, nx.RELATION_CYPHER, nx.COUNT_CYPHER):
        assert q.lstrip().upper().startswith(("MATCH", "CALL")), q
        # Read-only: a view builder must not be able to change the store.
        for forbidden in ("CREATE", "MERGE", "DELETE", "SET ", "DETACH", "DROP"):
            assert forbidden not in q.upper(), (forbidden, q)


def test_verify_names_what_is_missing_rather_than_only_counting():
    """A count that happens to match hides a swap, so the diff must be by name."""
    from wbs.task3_storage import neo4j_export as nx
    concepts = [{"name": "A", "concept_class": "Beach"},
                {"name": "Ghost", "concept_class": "Beach"}]
    relations = [{"subject": "A", "predicate": "near", "object": "Ghost"}]

    class Fake:
        @staticmethod
        def read_ontology(_v):
            return ([{"name": "A", "concept_class": "Mountain"},
                     {"name": "B", "concept_class": "Beach"}],
                    [{"subject": "A", "predicate": "near", "object": "B"}])

    import wbs.task3_storage.neo4j_store as store
    real = store.read_ontology
    store.read_ontology = Fake.read_ontology
    try:
        v = nx.verify(concepts, relations, "1.4.0")
    finally:
        store.read_ontology = real

    assert v["exact"] is False
    assert v["missing_concepts"] == ["B"]
    assert v["extra_concepts"] == ["Ghost"]
    assert v["missing_relations"] == [["A", "near", "B"]]
    assert v["concepts_with_a_different_class"] == ["A"]


@needs_export
def test_the_whole_graph_came_back_out_of_neo4j_and_the_round_trip_is_exact():
    d = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert d["source"] == "neo4j"
    m, v = d["meta"], d["verification"]

    # Every node and every relationship the server holds was returned.
    assert d["nodes_not_returned_by_the_concept_query"] == 0
    assert d["relationships_not_returned_by_the_relation_query"] == 0
    assert len(d["concepts"]) == m["nodes"]
    assert len(d["relations"]) == m["relationships"]

    assert v["exact"] is True, v
    assert v["neo4j_concepts"] == v["file_store_concepts"]
    assert v["neo4j_relations"] == v["file_store_relations"]
    assert not v["missing_concepts"] and not v["extra_concepts"]
    assert not v["missing_relations"] and not v["extra_relations"]
    assert not v["concepts_with_a_different_class"]

    # Provenance survives the round trip, not just structure.
    byname = {c["name"]: c for c in d["concepts"]}
    jeju = byname["Jeju"]
    assert jeju["concept_class"] == "Destination"
    assert jeju["provenance"]["extraction_prompt_version"]
    assert isinstance(jeju["provenance"]["confidence"], (int, float))
    assert all(r["provenance"]["extraction_prompt_version"] is not None
               for r in d["relations"][:50])


@needs_export
@pytest.mark.skipif(not EXPLORER.exists(), reason="explorer not built")
def test_the_explorer_says_which_neo4j_it_was_built_from():
    """"Sourced from Neo4j" has to be visible on the page, and true."""
    import re
    html = EXPLORER.read_text(encoding="utf-8")
    d = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        html, re.S).group(1))
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert d["source"] == "neo4j"
    assert d["neo4j"]["meta"]["bolt"] == snap["meta"]["bolt"]
    assert d["neo4j"]["meta"]["engine"] == snap["meta"]["engine"]
    assert d["neo4j"]["verification"]["exact"] is True
    assert d["nodes"] and len(d["nodes"]) == snap["meta"]["nodes"]

    # The Neo4j vocabulary is carried per class and per predicate, and is the
    # sanitiser's output rather than the raw name.
    assert len(d["labels"]) == len(d["classes"])
    assert len(d["reltypes"]) == len(d["predicates"])
    assert d["labels"] == [ns.label_of(c) for c in d["classes"]]
    assert d["reltypes"] == [ns.reltype_of(p) for p in d["predicates"]]

    # And the page has a Neo4j view that raises the Browser's render caps, since
    # its defaults would show a quarter of this graph without saying so.
    assert 'data-view="neo4j"' in html
    assert 'id="v-neo4j"' in html
    assert "initialNodeDisplay" in html and "maxNeighbours" in html
    assert "db.schema.visualization" in html


@needs_export
@pytest.mark.skipif(not LAYOUT.exists(), reason="poster not built")
def test_the_printed_figures_record_the_server_they_were_drawn_from():
    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert lay["source"] == "neo4j"
    assert lay["neo4j"]["bolt"] == snap["meta"]["bolt"]
    assert lay["neo4j"]["verified_exact"] is True
    assert sorted(lay["order"]) == sorted(c["name"] for c in snap["concepts"])


SCRIPT = DELIV / "whole_ontology.cypher"


@needs_export
def test_the_browser_script_is_the_same_queries_the_page_shows():
    """One source of truth: the page, the .cypher file and the module must agree."""
    import re
    from wbs.task3_storage import neo4j_export as nx
    text = SCRIPT.read_text(encoding="utf-8")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert [b["title"] for b in snap["browser_script"]] == \
        [t for t, _, _ in nx.BROWSER_SCRIPT]
    for _, _, cypher in nx.BROWSER_SCRIPT:
        assert cypher in text, cypher[:40]

    if EXPLORER.exists():
        html = EXPLORER.read_text(encoding="utf-8")
        d = json.loads(re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            html, re.S).group(1))
        assert d["browser_script"] == snap["browser_script"]

    # The caps come first, because the Browser's defaults would draw under a
    # fifth of this graph and report nothing about the remainder.
    assert nx.BROWSER_SCRIPT[0][2].startswith(":config")
    assert "initialNodeDisplay" in nx.BROWSER_SCRIPT[0][2]
    caps = re.search(r"initialNodeDisplay:\s*(\d+).*?maxNeighbours:\s*(\d+)",
                     nx.BROWSER_SCRIPT[0][2])
    assert caps, nx.BROWSER_SCRIPT[0][2]
    assert int(caps.group(1)) > snap["meta"]["nodes"]
    assert int(caps.group(2)) > snap["meta"]["relationships"]

    # The tunnel line has to name both ports, or the Browser cannot reach Bolt.
    assert "ssh -L 7474:" in text and "7687:" in text


def test_every_browser_query_is_read_only():
    """A reader pastes these into a live store; none of them may write to it."""
    from wbs.task3_storage import neo4j_export as nx
    for title, _, cypher in nx.BROWSER_SCRIPT:
        if cypher.startswith(":"):
            continue
        up = cypher.upper()
        for forbidden in ("CREATE", "MERGE", "DELETE", "DETACH", "DROP",
                          "SET ", "REMOVE ", "LOAD CSV", "CALL APOC"):
            assert forbidden not in up, (title, forbidden)


@pytest.mark.skipif(not EXPLORER.exists(), reason="explorer not built")
def test_the_explorer_javascript_parses():
    """The page is shipped as a file, so a syntax error would only show in a
    browser. Parse it here instead."""
    import re
    esprima = pytest.importorskip("esprima")
    html = EXPLORER.read_text(encoding="utf-8")
    code = re.findall(
        r'<script(?![^>]*type="application/json")[^>]*>(.*?)</script>',
        html, re.S)
    assert len(code) == 1
    esprima.parseScript(code[0], {"tolerant": False})

    # Every element the script reaches for must exist in the markup, and every
    # tab must have a view behind it.
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    wanted = set(re.findall(r"getElementById\('([^']+)'\)", code[0]))
    missing = sorted(w for w in wanted - ids if not re.search(r"\d$", w))
    assert missing == [], missing
    views = set(re.findall(r'id="v-([a-z0-9]+)"', html))
    tabs = set(re.findall(r'data-view="([a-z0-9]+)"', html))
    assert views == tabs, views ^ tabs
