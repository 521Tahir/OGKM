"""The English annual report must say what the deliverables say, and nothing else.

The report is generated, so the thing worth testing is not its prose but the
join between its prose and the JSON it quotes: a number that drifts, a list
printed where a count was meant, a figure path that stopped existing, or a
headline that survives a measurement moving under it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from docx import Document

from wbs.task6_report import make_annual_report as mar

ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "wbs" / "deliverables"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build the report into a temp directory, so a test never rewrites a deliverable."""
    out = tmp_path_factory.mktemp("annual_report")
    docx_path, md_path = mar.OUT_DOCX, mar.OUT_MD
    mar.OUT_DOCX, mar.OUT_MD = out / "report.docx", out / "report.md"
    try:
        mar.main()
        doc = Document(mar.OUT_DOCX)
        md = mar.OUT_MD.read_text(encoding="utf-8")
    finally:
        mar.OUT_DOCX, mar.OUT_MD = docx_path, md_path
    texts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                texts.append(cell.text)
    return {"doc": doc, "md": md, "texts": texts, "all": "\n".join(texts)}


def test_no_python_repr_leaks_into_the_report(built):
    """A list quoted where a count was meant is the failure this catches."""
    leaks = [t for t in built["texts"]
             if re.search(r"\[\{['\"]|\{['\"]\w+['\"]:\s", t)]
    assert leaks == [], f"raw Python/JSON repr in the report: {leaks[:2]}"


def test_every_figure_the_report_embeds_exists_and_is_cited(built):
    cited = re.findall(r"figures/(fig\d+_[a-z_]+\.png)", built["all"])
    assert cited, "the report embeds no figures"
    for name in set(cited):
        assert (DELIV / "figures" / name).exists(), name
    images = built["doc"].inline_shapes
    assert len(images) == len(cited), (
        f"{len(images)} images embedded but {len(cited)} source lines")


def test_the_achievement_table_is_the_audit_not_a_self_assessment(built):
    audit = json.loads((DELIV / "WBS_WORKBOOK_AUDIT.json").read_text("utf-8"))
    table = next(t for t in built["doc"].tables
                 if t.rows[0].cells[0].text.strip() == "WBS"
                 and "Degree" in t.rows[0].cells[-1].text)
    rows = {r.cells[0].text.strip(): r for r in table.rows[1:]}
    assert len(rows) == len(audit["items"]) == audit["total"]
    for item in audit["items"]:
        row = rows[item["code"]]
        degree = row.cells[-1].text.strip()
        assert degree == ("100" if item["complete"] else "-"), item["code"]
        # A line item may not be reported complete with an empty achievement.
        assert len(row.cells[2].text.strip()) > 40, item["code"]
    assert f"{audit['fully_satisfied']} of {audit['total']} line items" in \
        built["all"]


def test_the_proposal_counts_match_the_compliance_audit(built):
    pc = json.loads((DELIV / "PROPOSAL_COMPLIANCE.json").read_text("utf-8"))
    assert f"{pc['met']} of {pc['total']}" in built["all"]
    assert str(pc["out_of_scope"]) in built["all"]
    assert pc["missing"] == 0, (
        "a missing proposal requirement must be reported, not hidden; "
        "update the report's wording before changing this test")


def test_the_ontology_figures_are_the_delivered_ontology(built):
    qm = json.loads((DELIV / "task2_formalization" /
                     "ontology_quality_metrics.json").read_text("utf-8"))
    assert f"{qm['concepts']:,} concepts" in built["all"]
    assert f"{qm['relations']:,} relations" in built["all"]
    assert f"{qm['classes']} classes" in built["all"]


def test_the_variant_table_is_read_from_the_scores_file(built):
    vs = json.loads((DELIV / "task4_evaluation" /
                     "variant_scores__llama3.1-8b.json").read_text("utf-8"))
    table = next(t for t in built["doc"].tables
                 if t.rows[0].cells[0].text.strip() == "Variant")
    by_label = {r.cells[0].text.strip(): r for r in table.rows[1:]}
    for key, label in [("V1_llm", "V1 LLM only"), ("V2_rag", "V2 Simple RAG"),
                       ("V3_ontology_only", "V3 Ontology only"),
                       ("V4_ontology_rag", "V4 Ontology + RAG"),
                       ("V5_ontology_rag_validated",
                        "V5 Ontology + RAG + validation")]:
        cells = [c.text.strip() for c in by_label[label].cells[1:]]
        for metric, got in zip(("token_f1", "entity_f1", "entity_p",
                                "entity_hit", "ungrounded_rate", "clean"),
                               cells):
            assert got == f"{100 * vs['overall'][key][metric]:.1f}", \
                (label, metric)


def test_the_training_budget_tiers_are_the_scorer_s_own_counts(built):
    cm = json.loads((DELIV / "task4_evaluation" /
                     "cross_model_results.json").read_text("utf-8"))
    table = next(t for t in built["doc"].tables
                 if t.rows[0].cells[0].text.strip() == "Training budget")
    printed = {}
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        tier = cells[0].split()[0].lower()
        printed[tier] = (cells[1], cells[2], cells[3])
    for tier, key in (("equal", "equal"), ("extended", "extended"),
                      ("matched", "matched"), ("untuned", "untuned_rag")):
        s = cm["summary"][key]
        pairs, never_worse, sweeps = printed[tier]
        assert pairs == str(s["pairs"]), tier
        assert never_worse == f"{s['never_worse']} of {s['pairs']}", tier
        assert sweeps == str(s["clean_sweeps"]), tier
    # The one surviving win is named by model, not asserted in general.
    for model in cm["metric_rollup"]["matched"]["clean"]["ontology_better_at"]:
        assert model in built["all"], model


def test_the_report_does_not_claim_the_clause_the_project_cannot_support(built):
    """The standing goal asked for a clean sweep; the measurement refused it."""
    assert "What is not claimed" in built["all"]
    assert "No result in this report shows an ontology arm leading" in \
        built["all"]
    cm = json.loads((DELIV / "task4_evaluation" /
                     "cross_model_results.json").read_text("utf-8"))
    if cm["summary"]["matched"]["clean_sweeps"] == 0:
        assert "leads on every metric" not in built["all"].lower()


def test_every_part_of_the_korean_form_is_present(built):
    for heading in [
        "Annual Report", "Security grade", "< Summary >", "Contents",
        "Overview of the R&D project",
        "Process and content of the R&D project",
        "Results of the R&D project and degree of goal achievement",
        "Record of R&D expenditure", "R&D plan for the next year",
        "Degree of achievement of this year's objectives",
        "Qualitative R&D outcomes", "Quantitative R&D outcomes",
        "Detailed quantitative R&D outcomes",
        "Unplanned outcomes and contributions to the field",
        "Statement of R&D expenditure by research institution",
        "Statement of R&D expenditure by R&D project",
        "R&D objectives and content",
        "Changes in the domestic and international environment",
        "R&D implementation strategy",
        "R&D schedule and expected outcomes",
        "Plan for the use of R&D funding", "Commercialisation plan",
        "Utilisation of R&D outcomes and expected effects",
        "< Attachment >", "NOTICE",
    ]:
        assert heading in built["all"], heading


def test_the_blanks_are_declared_rather_than_filled(built):
    """Every administrative blank must be listed with who holds it."""
    assert "Fields left for the institution to complete" in built["all"]
    table = next(t for t in built["doc"].tables
                 if t.rows[0].cells[0].text.strip() == "Where"
                 and t.rows[0].cells[-1].text.strip() == "Who holds it")
    owners = [r.cells[-1].text.strip() for r in table.rows[1:]]
    assert len(owners) >= 8 and all(owners), "a blank with no owner"
    # The report must not have invented a budget figure anywhere.
    assert not re.search(r"\b\d{1,3}(,\d{3})+\s*KRW thousand", built["all"])


def test_the_markdown_mirror_carries_the_same_headline(built):
    audit = json.loads((DELIV / "WBS_WORKBOOK_AUDIT.json").read_text("utf-8"))
    assert f"{audit['fully_satisfied']} of " in built["md"]
    assert "Fields left for the institution" in built["md"]
