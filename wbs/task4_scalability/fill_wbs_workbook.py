"""
Hand the WBS workbook back filled in.

Two sheets go back: the delivered execution sheet with its blank columns
completed, and a third sheet, `완료 검증`, which is the one-page answer to "is
this finished" - every workbook row with its requirement count, and underneath,
every requirement the research proposal names with its verdict, so a reader can
see at a glance both what the funded workbook asked for and what the proposal
behind it asked for, including the three requirements that need people rather
than code and are not claimed.

`audit_wbs.py` checks the workbook's 16 line items against files on disk and
writes `WBS_WORKBOOK_AUDIT.md`. That is a report about the workbook; this script
writes the workbook itself, with the two columns the delivered sheet left
partly blank - 진척율 (progress) and 산출물 (deliverables) - completed from the
audit, plus a new 산출물 경로 column naming the file that satisfies each
requirement so a reader can open it rather than take the progress figure on
trust.

Nothing is asserted that `audit_wbs` did not verify on disk: a line item's
progress is `met / (met + missing)`, so a requirement with no artifact pulls its
row below 1.00 instead of being rounded up.

  python -m wbs.task4_scalability.fill_wbs_workbook
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
D = ROOT / "wbs" / "deliverables"
SRC = ROOT / "2026-08월 공동연구 WBS_제주대(윤여찬).xlsx"
OUT = D / "2026-08월 공동연구 WBS_제주대(윤여찬)_진척완료.xlsx"
SHEET = "2026 실행Ver_1"

# Which spreadsheet row carries which WBS line item. The workbook interleaves
# task headers (column A), sub-area headers (column B) and line items (column C);
# only the column C rows are line items, and these are them.
ROW_OF = {5: "1.1.1", 6: "1.1.2", 8: "1.2.1", 9: "1.2.2",
          12: "2.1.1", 13: "2.1.2", 15: "2.2.1", 16: "2.2.2",
          19: "3.1.1", 20: "3.1.2", 22: "3.2.1", 23: "3.2.2",
          26: "4.1.1", 27: "4.1.2", 29: "4.2.1", 30: "4.2.2"}
PROGRESS_COL, DELIV_COL = "T", "U"
PATH_COL, NOTE_COL = "V", "W"
SUMMARY_SHEET = "완료 검증"

# The other delivered sheet is the proposal-version Gantt: nine research phases
# down column A, one scheduled month each. It carries no progress column at all,
# so the same treatment is applied - a progress figure that is met/(met+missing)
# over named artifacts, and the artifacts beside it. The phase names are the
# workbook's own; the mapping is to this project's WBS line items and to the
# proposal sections each phase is the schedule for.
PLAN_SHEET = "제안서Ver"
PLAN_ROW = {
    3: ("연구 정렬 / Research alignment", "WBS 1.1", [
        "docs/system_requirements.md",
        "docs/implementation_roadmap.md",
        "docs/PROJECT_IMPLEMENTATION_PLAN.md",
        "docs/risk_management_plan.md"]),
    4: ("코퍼스 및 스키마 준비 / Corpus and schema preparation", "WBS 1.2", [
        "data/jeju_full",
        "data/reports/data_profile_report.md",
        "docs/tourism_ontology_schema.md",
        "artifacts/ontologies/0.1.0/concepts.json"]),
    5: ("온톨로지 생성 파이프라인 / Ontology generation pipeline", "WBS 2.1.1, §5", [
        "src/ontology/concept_extractor.py",
        "src/ontology/relation_extractor.py",
        "src/ontology/hierarchy_inducer.py",
        "src/ontology/rule_suggester.py",
        "artifacts/ontologies/1.4.0/concepts.json"]),
    6: ("프롬프팅 전략 실험 / Prompting strategy experiments", "WBS 2.1.2, §6", [
        "artifacts/reports/prompt_comparison_llm.md",
        "wbs/deliverables/task2_extraction/prompt_sensitivity.md",
        "wbs/deliverables/task4_evaluation/HYPOTHESES.md"]),
    7: ("검증 및 거버넌스 / Validation and governance", "WBS 2.2, 3.1, §4.5, §8.6-8.7", [
        "wbs/deliverables/task2_formalization/validation_report.md",
        "wbs/deliverables/task2_formalization/ontology_quality_metrics.json",
        "wbs/deliverables/task2_formalization/rule_compliance.md",
        "wbs/deliverables/task3_governance/version_governance_report.md",
        "wbs/deliverables/task3_governance/access_control.md",
        "wbs/deliverables/task3_governance/audit_trail.jsonl"]),
    8: ("진화 및 전문가 피드백 / Evolution and expert feedback", "WBS 3.1.2, §8.3-8.5", [
        "wbs/deliverables/task3_governance/diff_logs.json",
        "wbs/deliverables/task3_governance/proposals.json",
        "wbs/deliverables/task2_formalization/policy_constraint_extraction.md",
        "wbs/deliverables/task4_evaluation/expert_ontology_review.md",
        "wbs/deliverables/task4_evaluation/human_review_scores.md"]),
    9: ("온톨로지 인식 RAG / Ontology-aware RAG", "WBS 3.2, §9-10", [
        "wbs/deliverables/task3_retrieval/retriever_characterization.md",
        "wbs/deliverables/task3_retrieval/onto_retrieval_tuning.md",
        "wbs/task3_reasoning/constrained_reasoner.py",
        "wbs/deliverables/task3_reasoning/constraint_coverage.md"]),
    10: ("파일럿 평가 / Pilot evaluation", "WBS 4.1, §11", [
        "wbs/deliverables/task4_evaluation/baseline_comparison__llama3.1-8b.md",
        "wbs/deliverables/task4_evaluation/ablation_offline__llama3.1-8b.md",
        "wbs/deliverables/task4_evaluation/hallucination_report__llama3.1-8b.md",
        "wbs/deliverables/task4_evaluation/cross_model_results.md",
        "wbs/deliverables/task4_scalability/robustness.md"]),
    11: ("최종 통합 및 배포 / Final integration and delivery", "WBS 4.2, §14", [
        "wbs/deliverables/FINAL_REPORT.md",
        "wbs/deliverables/figures/FIGURE_INDEX.md",
        "wbs/deliverables/REPRODUCIBILITY/MANIFEST.json",
        "wbs/deliverables/presentation/Ontology_Business_Plan.pptx",
        "wbs/deliverables/presentation/Ontology_Demo.mp4"]),
}
PLAN_NOTE = {
    8: "Mechanism complete: incremental acquisition, proposals, conflict "
       "resolution and provenance. The expert *panel* (§11.4.2) is the one "
       "input this phase did not have; the instrument and its scoring are "
       "built and the scores are not simulated.",
    10: "The five variants were evaluated at three Llama sizes on the 347-item "
        "benchmark, and then the same two arms were re-measured on eight models "
        "over 4,607 held-out questions (cross_model_results.md), with each "
        "arm's training budget read off its own receipt. At equal budget the "
        "ontology arm still gives up token precision at all four comparable "
        "models; the never-worse result at four of four holds only under a 3:1 "
        "or 4:1 ontology training advantage, and is labelled that way.",
}


def fill_plan_sheet(wb, openpyxl) -> list[tuple]:
    """Give the proposal-version Gantt the same two columns, on the same rule."""
    from openpyxl.styles import Alignment, Font

    if PLAN_SHEET not in wb.sheetnames:
        return []
    ws = wb[PLAN_SHEET]
    ws["K2"], ws["L2"], ws["M2"] = ("진척율", "WBS / 제안서 절",
                                    "산출물 경로 (Evidence path)")
    for c in ("K2", "L2", "M2"):
        ws[c].font = Font(bold=True)
    out = []
    for row, (name, maps, paths) in PLAN_ROW.items():
        have = [q for q in paths if (ROOT / q).exists()]
        frac = len(have) / max(len(paths), 1)
        ws[f"K{row}"] = round(frac, 2)
        ws[f"L{row}"] = maps
        ws[f"M{row}"] = "\n".join(have or ["MISSING"]) + (
            "\n\n" + PLAN_NOTE[row] if row in PLAN_NOTE else "")
        for col in ("L", "M"):
            ws[f"{col}{row}"].alignment = Alignment(wrap_text=True, vertical="top")
        out.append((name, frac, len(have), len(paths)))
    ws.column_dimensions["K"].width = 9
    ws.column_dimensions["L"].width = 28
    ws.column_dimensions["M"].width = 72
    return out


def add_summary_sheet(wb, openpyxl, audit: dict, plan: list[tuple]) -> None:
    """One page: every workbook row, then every proposal requirement.

    The proposal block is included because the workbook is the funded
    decomposition of a research proposal, and a row can be complete while the
    requirement behind it is not. Three requirements need a human panel and are
    listed as such rather than folded into a percentage.
    """
    from openpyxl.styles import Alignment, Font, PatternFill

    prop = json.loads((D / "PROPOSAL_COMPLIANCE.json").read_text(encoding="utf-8"))
    if SUMMARY_SHEET in wb.sheetnames:
        del wb[SUMMARY_SHEET]
    ws = wb.create_sheet(SUMMARY_SHEET)
    bold = Font(bold=True)
    head = PatternFill("solid", fgColor="DDE7F0")
    wrap = Alignment(wrap_text=True, vertical="top")

    def row(*vals, header=False):
        ws.append(list(vals))
        if header:
            for c in ws[ws.max_row]:
                c.font, c.fill = bold, head
        for c in ws[ws.max_row]:
            c.alignment = wrap

    n_done, n_tot = audit["fully_satisfied"], audit["total"]
    row("WBS 완료 검증 / WBS completion verification", header=True)
    row(f"Generated {time.strftime('%Y-%m-%d')} from files on disk. Nothing here is "
        f"asserted that the audit did not find: a requirement with no artifact "
        f"lowers its row instead of being rounded up.")
    row("")
    row(f"Workbook line items ({SHEET})", f"{n_done} / {n_tot} complete",
        f"{audit['partial']} partial", header=True)
    row(f"Research proposal requirements", f"{prop['met']} / {prop['total']} met",
        f"{prop['out_of_scope']} need a human panel (not claimed), "
        f"{prop['partial']} partial, {prop['missing']} missing", header=True)
    row("")

    if plan:
        row(f"Proposal-version phases ({PLAN_SHEET})",
            f"{sum(1 for _, f, _, _ in plan if f == 1)} / {len(plan)} at 1.00",
            "", header=True)
        row("")
        row("Phase", "Maps to", "Artifacts found", "진척율", header=True)
        for name, frac, nhave, ntot in plan:
            row(name, PLAN_ROW[next(k for k, v in PLAN_ROW.items()
                                    if v[0] == name)][1],
                f"{nhave}/{ntot}", round(frac, 2))
        row("")

    row("WBS", "Line item", "Requirements met", "Status", header=True)
    for i in audit["items"]:
        row(i["code"], i["title"],
            f"{len(i['met'])}/{len(i['met']) + len(i['missing'])}",
            "complete" if i["complete"] else
            "INCOMPLETE: " + ", ".join(i["missing"]))
    row("")

    row("Proposal §", "Requirement", "Status", "Evidence (first path)", header=True)
    verdict = {"met": "met", "partial": "partial",
               "out-of-scope": "needs a human panel — not claimed",
               "missing": "MISSING"}
    for i in prop["items"]:
        row(i["section"], i["requirement"], verdict.get(i["status"], i["status"]),
            (i["evidence"] or ["—"])[0])
    row("")
    row("The three requirements not claimed", header=True)
    for i in prop["items"]:
        if i["status"] == "out-of-scope":
            row(i["section"], i["requirement"],
                (i.get("reason") or i["note"] or "")[:900])

    for col, w in (("A", 12), ("B", 62), ("C", 46), ("D", 58)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"


def main() -> None:
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required: pip install openpyxl")
    if not SRC.exists():
        sys.exit(f"{SRC.name} not found")

    from wbs.task4_scalability.audit_wbs import ITEMS

    audit = json.loads((D / "WBS_WORKBOOK_AUDIT.json").read_text(encoding="utf-8"))
    by_code = {i["code"]: i for i in audit["items"]}
    paths_of: dict[str, list[tuple[str, list[str]]]] = {
        code: reqs for code, _title, reqs in ITEMS}

    wb = openpyxl.load_workbook(SRC)
    ws = wb[SHEET]
    ws[f"{PATH_COL}1"] = "산출물 경로 (Evidence path)"
    ws[f"{NOTE_COL}1"] = "검증 (Verified)"

    filled, below = 0, []
    for row, code in ROW_OF.items():
        a = by_code.get(code)
        if a is None:
            continue
        n_met, n_miss = len(a["met"]), len(a["missing"])
        frac = n_met / max(n_met + n_miss, 1)
        ws[f"{PROGRESS_COL}{row}"] = round(frac, 2)
        if frac < 1:
            below.append((code, a["missing"]))

        # 산출물: the deliverable names the workbook asks for, as the audit
        # resolved them. Where the sheet already names them, leave its wording.
        # the workbook merges 산출물 across each sub-area's two rows, so the
        # second row's cell is read-only and the first already carries the text
        names = [r for r, _ in paths_of.get(code, [])]
        cell = ws[f"{DELIV_COL}{row}"]
        if (names and cell.__class__.__name__ != "MergedCell"
                and not str(cell.value or "").strip()):
            cell.value = "\n".join(f"• {n}" for n in names)

        lines = []
        for req, cands in paths_of.get(code, []):
            have = [p for p in cands if (ROOT / p).exists()]
            lines.append(f"{req}: {have[0] if have else 'MISSING'}"
                         + (f" (+{len(have) - 1})" if len(have) > 1 else ""))
        ws[f"{PATH_COL}{row}"] = "\n".join(lines)
        ws[f"{NOTE_COL}{row}"] = (f"{n_met}/{n_met + n_miss} requirements verified on disk"
                                  + ("" if frac == 1 else "  — INCOMPLETE"))
        filled += 1

    for col in (PATH_COL, NOTE_COL):
        ws.column_dimensions[col].width = 64
    for row in ROW_OF:
        for col in (PATH_COL, NOTE_COL):
            ws[f"{col}{row}"].alignment = openpyxl.styles.Alignment(
                wrap_text=True, vertical="top")

    plan = fill_plan_sheet(wb, openpyxl)
    add_summary_sheet(wb, openpyxl, audit, plan)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {filled} line items filled; "
          f"{audit['fully_satisfied']}/{audit['total']} at 1.00")
    for code, miss in below:
        print(f"  {code} below 1.00, missing: {', '.join(miss)}")
    if plan:
        print(f"  {PLAN_SHEET}: "
              f"{sum(1 for _, f, _, _ in plan if f == 1)}/{len(plan)} phases at 1.00")
        for name, frac, nhave, ntot in plan:
            if frac < 1:
                print(f"    {name}: {nhave}/{ntot}")


if __name__ == "__main__":
    main()
