"""The two paper figures, and the honesty of the case selection behind one of them.

A qualitative figure is a selected slice by construction. These tests check that
the selection is reproducible, that the cases are real records rather than
anything assembled, and that the figure states the base rate instead of letting
four hand-picked rows imply one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "wbs" / "deliverables" / "figures"
CASES = ROOT / "wbs" / "deliverables" / "task4_evaluation" / "qualitative_cases.json"
EXP = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"

needs_build = pytest.mark.skipif(
    not CASES.exists(),
    reason="run wbs.task6_report.make_paper_figures first")


def test_key_entities_drop_leading_articles_and_stopwords():
    from wbs.task6_report.make_paper_figures import key_entities
    got = key_entities("The UNESCO tuff cone known as Sunrise Peak is "
                       "Seongsan Ilchulbong. Near Seongsan you can visit it.")
    assert "Sunrise Peak" in got
    assert "Seongsan Ilchulbong" in got
    # The article is part of the regex match but not part of the name.
    assert not any(g.startswith("The ") for g in got), got
    assert not any(g.startswith("Near ") for g in got), got
    assert "Jeju" not in got and "UNESCO" not in got


@pytest.fixture(scope="module")
def sel():
    return json.loads(CASES.read_text(encoding="utf-8"))


@needs_build
def test_every_case_is_a_real_scored_record(sel):
    """No case may be edited, trimmed or assembled: each must match the file."""
    for c in sel["cases"]:
        gen_r = EXP / "generations" / f"bench__{c['model']}__rag.jsonl"
        gen_o = EXP / "generations" / f"bench__{c['model']}__{sel['arm']}.jsonl"
        sc_r = EXP / "scores" / f"bench__{c['model']}__rag.jsonl"
        sc_o = EXP / "scores" / f"bench__{c['model']}__{sel['arm']}.jsonl"
        assert all(p.exists() for p in (gen_r, gen_o, sc_r, sc_o)), c["model"]

        def find(path, rid):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    if d["record_id"] == rid:
                        return d
            return None

        gr, go = find(gen_r, c["record_id"]), find(gen_o, c["record_id"])
        sr, so = find(sc_r, c["record_id"]), find(sc_o, c["record_id"])
        assert gr and go and sr and so, c["record_id"]
        assert c["question"] == go["Input"] == gr["Input"]
        assert c["reference"] == go["reference"]
        assert c["rag"] == gr["answer"].strip()
        assert c["ontology"] == go["answer"].strip()
        assert c["judge_rag"] == sr["judge"]
        assert c["judge_ontology"] == so["judge"]


@needs_build
def test_the_cases_meet_the_stated_selection_rule(sel):
    for c in sel["cases"]:
        assert c["judge_ontology"] >= sel["threshold_ontology"], c["record_id"]
        assert c["judge_rag"] <= sel["threshold_rag"], c["record_id"]
        # The ontology answer must actually carry more of the reference's
        # entities, or the figure's green marking says nothing.
        assert c["entities_ontology"] > c["entities_rag"], c["record_id"]
    # One case per model, so the figure is not four rows from one weak model.
    models = [c["model"] for c in sel["cases"]]
    assert len(models) == len(set(models)), models


@needs_build
def test_the_base_rate_is_recorded_and_the_slice_is_not_presented_as_typical(sel):
    """The reverse direction exists and must be counted, not omitted."""
    assert sel["scored_pairs"] > 0
    assert sel["ontology_right_rag_wrong"] >= len(sel["cases"])
    assert sel["rag_right_ontology_wrong"] > 0, (
        "if RAG never wins, that is a claim worth checking before printing it")
    # The selected cases are a small fraction of all pairs: the figure is an
    # illustration, and the numbers have to make that visible.
    assert sel["candidates"] / sel["scored_pairs"] < 0.2


@needs_build
def test_the_selection_is_reproducible(sel):
    from wbs.task6_report.make_paper_figures import select_cases
    again = select_cases(arm=sel["arm"], hi=sel["threshold_ontology"],
                         lo=sel["threshold_rag"], want=len(sel["cases"]))
    assert [c["record_id"] for c in again["cases"]] == \
        [c["record_id"] for c in sel["cases"]]
    assert again["ontology_right_rag_wrong"] == sel["ontology_right_rag_wrong"]
    assert again["rag_right_ontology_wrong"] == sel["rag_right_ontology_wrong"]


@pytest.mark.parametrize("stem", ["fig19_ontology_structure",
                                  "fig20_qualitative_rag_vs_ontology"])
def test_both_paper_figures_exist_as_raster_and_vector(stem):
    png, pdf = FIG / f"{stem}.png", FIG / f"{stem}.pdf"
    if not png.exists():
        pytest.skip("run wbs.task6_report.make_paper_figures first")
    assert pdf.exists(), "a paper figure needs a vector copy"
    assert png.stat().st_size > 50_000
    assert pdf.stat().st_size > 5_000
    # Type 42 fonts, so the PDF's text stays selectable and embeds properly.
    head = pdf.read_bytes()[:4]
    assert head == b"%PDF"


@pytest.mark.skipif(not (FIG / "fig19_ontology_structure.png").exists(),
                    reason="figures not built")
def test_the_structure_figure_covers_every_class_that_has_instances():
    from wbs.task6_report.make_paper_figures import (_taxonomy_groups,
                                                     _taxonomy_rows,
                                                     ontology_structure)
    d = ontology_structure()
    rows = _taxonomy_rows(d["hierarchy"], d["counts"])
    labelled = {label for _, label, _ in rows}
    missing = [c for c in d["counts"] if c not in labelled]
    assert missing == [], missing

    # The ring groups must partition the classes that carry instances; a class
    # dropped from the ring is a relation silently missing from panel (b).
    groups = _taxonomy_groups(d["hierarchy"], d["counts"])
    members = [c for g in groups for c in g["members"]]
    assert len(members) == len(set(members)), "a class appears in two groups"
    expected = {c for c, n in d["counts"].items() if n and c != "OntologyClass"}
    assert set(members) == expected, expected ^ set(members)


@pytest.mark.skipif(not (FIG / "FIGURE_INDEX.md").exists(),
                    reason="figure index not built")
def test_the_figure_index_lists_the_paper_figures():
    idx = (FIG / "FIGURE_INDEX.md").read_text(encoding="utf-8")
    for stem in ("fig19_ontology_structure", "fig20_qualitative_rag_vs_ontology"):
        if (FIG / f"{stem}.png").exists():
            assert stem in idx, stem
