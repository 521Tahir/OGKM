"""
Phase 10 — Evaluation Framework: consolidated runner & on-screen report.

Aggregates every Phase 10 sub-area (A–G) from the experiment reports into a
single, comprehensive on-screen summary and writes a consolidated report to
``artifacts/reports/phase10_summary_report.{json,md}``.

Sub-areas (mapped to CLAUDE_PROMPT Phase 10 and the proposal §11):
  A. Ontology quality      — concept/relation P/R/F1, taxonomy, structural
                             coherence, redundancy, disconnected rate, logical
                             consistency, validation failure rate
  B. Retrieval             — Recall@K, Precision@K, MRR, Hit@K (4 modes)
  C. Response quality      — factual correctness, constraint satisfaction,
                             fluency, overall (LLM-as-judge, 4 modes)
  D. Hallucination         — unsupported entity/relation, unsafe rec,
                             contradiction, fabricated logistics (V1–V5)
  E. Explainability        — evidence trace coverage, faithfulness, path
                             interpretability, user-trust proxy (V1–V5)
  F. Robustness            — paraphrase, incomplete, adversarial, constraint
                             conflict, unseen destination (V1–V5)
  G. Efficiency/scalability — extraction, validation, retrieval, generation
                             latency; storage; cost/1K; 10K→100K profile

Baseline comparison V1–V5:
  V1 Baseline LLM · V2 Doc-RAG · V3 Onto-RAG · V4 Onto-RAG hybrid ·
  V5 Constrained + Validation (full system)

Usage
-----
  python scripts/run_phase10_evaluation.py            # aggregate + print
  python scripts/run_phase10_evaluation.py --refresh  # re-run offline metrics first
  python scripts/run_phase10_evaluation.py --no-save   # print only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
REPORTS = PROJECT_ROOT / "artifacts" / "reports"
FIGURES = PROJECT_ROOT / "artifacts" / "figures"

# ── ANSI ───────────────────────────────────────────────────────────────────────
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"; BLU = "\033[94m"
CYN = "\033[96m"; BLD = "\033[1m"; RST = "\033[0m"; MGN = "\033[95m"; DIM = "\033[2m"


def hdr(title: str) -> None:
    print(f"\n{BLD}{'═'*78}{RST}")
    print(f"{BLD}{title:^78}{RST}")
    print(f"{BLD}{'═'*78}{RST}")


def sub(letter: str, title: str) -> None:
    print(f"\n{MGN}{BLD}── {letter}.  {title} {'─'*(70-len(title))}{RST}")


def load(name: str) -> dict:
    p = REPORTS / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_json_path(path: Path):
    """Load any JSON file; return [] on failure (used for ontology version files)."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return []


def g(d: dict, *keys, default=None):
    """Safe nested getter."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fnum(x, fmt="{:.3f}", na="—") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else na


def pct(x, na="—") -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) and not isinstance(x, bool) else na


def colorize(v: float, good_high: bool = True, hi: float = 0.5, lo: float = 0.3) -> str:
    if not isinstance(v, (int, float)):
        return DIM
    if good_high:
        return GRN if v >= hi else (YLW if v >= lo else RED)
    return GRN if v <= (1 - hi) else (YLW if v <= (1 - lo) else RED)


def section_header(text: str) -> None:
    """Markdown-style section header, e.g. '## A. Ontology quality metrics'."""
    print(f"\n{MGN}{BLD}## {text}{RST}")


def table(headers: list[str], rows: list[list], aligns: list[str] | None = None,
          footnote: str | None = None) -> None:
    """
    Render a clean box-drawing table.

    headers : column titles
    rows    : list of row cells (values are str()'d)
    aligns  : per-column 'l' or 'r' (default: first col left, rest right)
    """
    cells = [[str(c) for c in row] for row in rows]
    ncol = len(headers)
    aligns = aligns or (["l"] + ["r"] * (ncol - 1))
    widths = [len(h) for h in headers]
    for row in cells:
        for i in range(ncol):
            widths[i] = max(widths[i], len(row[i]) if i < len(row) else 0)

    def fmt(cell: str, i: int) -> str:
        return cell.ljust(widths[i]) if aligns[i] == "l" else cell.rjust(widths[i])

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print("  " + top)
    print("  │ " + " │ ".join(f"{BLD}{fmt(h, i)}{RST}" for i, h in enumerate(headers)) + " │")
    print("  " + mid)
    for row in cells:
        padded = [row[i] if i < len(row) else "" for i in range(ncol)]
        print("  │ " + " │ ".join(fmt(padded[i], i) for i in range(ncol)) + " │")
    print("  " + bot)
    if footnote:
        print(f"  {DIM}{footnote}{RST}")


# ══════════════════════════════════════════════════════════════════════════════
# Refresh offline-computable metrics
# ══════════════════════════════════════════════════════════════════════════════

def refresh() -> None:
    """Re-run the lightweight, offline metric generators before aggregating."""
    steps = [
        ("Extraction P/R/F1 (exp020)",
         [sys.executable, "experiments/exp020_extraction_prf/run_extraction_prf.py",
          "--version", "1.0.0", "--silver"]),
        ("Ontology quality (exp008)",
         [sys.executable, "-m", "experiments.exp008_ontology_quality.run_quality"]),
        ("PDF-aligned metrics (§11.4–11.9)",
         [sys.executable, "scripts/run_pdf_metrics_evaluation.py"]),
    ]
    for label, cmd in steps:
        print(f"{DIM}  refreshing: {label} ...{RST}")
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # pragma: no cover
            print(f"{YLW}    skipped ({e}){RST}")


# ══════════════════════════════════════════════════════════════════════════════
# Section builders — return dicts for the consolidated report, print as side-effect
# ══════════════════════════════════════════════════════════════════════════════

def section_baselines(main: dict) -> dict:
    section_header("Baseline comparison (V1–V5) — proposal §11.2")
    res = main.get("results", {})
    labels = {v["id"]: v["label"] for v in main.get("variants", [])}
    order = ["V1", "V2", "V3", "V4", "V5"]
    rows, out = [], {}
    for v in order:
        r = res.get(v, {})
        if not r:
            continue
        name = f"{v} {labels.get(v, '')}" + (" ★" if v == "V5" else "")
        rows.append([name, fnum(r.get("entity_recall")), fnum(r.get("entity_precision")),
                     pct(r.get("hallucination_rate")), fnum(r.get("avg_grounding"))])
        out[v] = {
            "label": labels.get(v, ""),
            "entity_recall": r.get("entity_recall"),
            "entity_precision": r.get("entity_precision"),
            "hallucination_rate": r.get("hallucination_rate"),
            "avg_grounding": r.get("avg_grounding"),
        }
    table(["Variant", "Entity Recall", "Entity Prec", "Hallucination", "Grounding"], rows,
          footnote=f"n_queries = {main.get('n_queries', '?')}, ontology v{main.get('ontology_version','?')}  ·  ★ = full system")
    return out


def section_A(version: str, concepts: list, relations: list, hierarchy: list, prf: dict) -> dict:
    """
    Version-pinned ontology quality. Structural metrics are computed directly
    from the loaded ontology version (not stale snapshots); P/R/F1 come from the
    extraction_prf report. SHACL is run live so logical-consistency / validation
    failure reflect exactly this version.
    """
    from src.evaluation.ontology_metrics import OntologyMetricsCalculator
    from src.ontology.shacl_validator import SHACLValidator

    section_header(f"A. Ontology quality metrics — §11.4   (ontology v{version})")
    calc = OntologyMetricsCalculator(concepts, relations, [], version=version)
    n_concepts = len(concepts)
    orphan = calc.orphan_rate()
    disc = calc.disconnected_subgraph_count()
    disc_rate = (disc / n_concepts) if n_concepts else None
    depth = calc.hierarchy_depth(hierarchy)
    breadth = calc.hierarchy_breadth(hierarchy)
    triples = [(r.get("subject", ""), r.get("predicate", ""), r.get("object", "")) for r in relations]
    redundancy = 1 - (len(set(triples)) / len(triples)) if triples else 0.0

    rep = SHACLValidator().validate_ontology(concepts, relations)
    n_elems = n_concepts + len(relations)
    logical_consistency = 1.0 if rep.conforms else None
    val_failure = (len(rep.violations) / n_elems) if n_elems else 0.0

    # P/R/F1 — warn if the prf report was computed on a different version
    prf_ver = prf.get("version")
    ver_note = "" if prf_ver in (None, version) else f" [prf@{prf_ver}!]"
    c, r = prf.get("concept", {}), prf.get("relation", {})

    rows = [
        ["Concept precision",      fnum(c.get("precision")), f"src={c.get('precision_source','?')} ({c.get('n_labeled',0)}/{c.get('n_total',0)} labeled){ver_note}"],
        ["Concept recall",         fnum(c.get("recall")),    f"src={c.get('recall_source','?')} ({c.get('n_matched',0)}/{c.get('n_expected',0)} matched)"],
        ["Concept F1",             fnum(c.get("f1")),        "harmonic mean"],
        ["Relation precision",     fnum(r.get("precision")), f"src={r.get('precision_source','?')} ({r.get('n_labeled',0)}/{r.get('n_total',0)} labeled)"],
        ["Relation recall",        fnum(r.get("recall")),    "no gold relation set" if r.get("recall") is None else ""],
        ["Relation F1",            fnum(r.get("f1")),        "pending recall"],
        ["Taxonomy depth (avg)",   fnum(depth), "is-a chain depth"],
        ["Taxonomy breadth (avg)", fnum(breadth), "children per parent"],
        ["Orphan rate",            pct(orphan), "concepts with 0 relations"],
        ["Redundancy rate",        pct(redundancy), "duplicate triples"],
        ["Disconnected subgraphs", str(disc), f"rate = {pct(disc_rate)}"],
        ["Logical consistency rate", fnum(logical_consistency), f"SHACL violations = {len(rep.violations)}"],
        ["Validation failure rate", pct(val_failure), f"{len(rep.violations)}/{n_elems} elements"],
    ]
    table(["Metric", "Value", "Source / Note"], rows, aligns=["l", "r", "l"],
          footnote=f"{n_concepts} concepts, {len(relations)} relations, {len(set(triples))} unique triples; "
                   f"SHACL conforms={rep.conforms}, warnings={len(rep.warnings)}.")
    last = {"hierarchy_depth": depth, "hierarchy_breadth": breadth,
            "orphan_rate": orphan, "disconnected_subgraph_count": disc, "n_concepts": n_concepts}
    return {
        "concept": c, "relation": r,
        "taxonomy_depth": last.get("hierarchy_depth"),
        "taxonomy_breadth": last.get("hierarchy_breadth"),
        "orphan_rate": last.get("orphan_rate"),
        "redundancy_rate": redundancy,
        "disconnected_subgraph_count": disc,
        "disconnected_subgraph_rate": disc_rate,
        "logical_consistency_rate": logical_consistency,
        "validation_failure_rate": val_failure,
    }


def section_B(retr: dict) -> dict:
    section_header("B. Retrieval evaluation — Phase 10B")
    agg = retr.get("aggregate", {})
    modes = [("no_retrieval", "No retrieval"), ("doc_only", "Doc-only"),
             ("onto_only", "Onto-only"), ("hybrid", "Hybrid")]
    rows, out = [], {}
    for key, label in modes:
        m = agg.get(key, {})
        if not m:
            continue
        rows.append([label, fnum(m.get("avg_context_recall")), fnum(m.get("mrr")),
                     fnum(m.get("ndcg_at_10")), fnum(m.get("hit_rate_at_10")),
                     fnum(m.get("subgraph_relevance")), fnum(m.get("evidence_coverage"))])
        out[key] = m
    table(["Mode", "Recall", "MRR", "nDCG@10", "Hit@10", "SubgraphRel", "EvidenceCov"], rows,
          footnote=f"n_queries = {retr.get('n_queries','?')}, ontology v{retr.get('ontology_version','?')}. "
                   f"Recall = context recall over gold concepts; SubgraphRel = retrieved that are relevant; "
                   f"EvidenceCov = gold concepts backed by a retrieved relation.")
    return out


def section_C(judge: dict) -> dict:
    section_header("C. Response quality evaluation (LLM-as-judge) — §11.5")
    agg = judge.get("aggregates", {})
    modes = [("no_rag", "V1 Baseline"), ("doc_rag", "V2 Doc-RAG"),
             ("onto_rag", "V3 Onto-RAG"), ("constrained", "V5 Constrained")]
    rows, out = [], {}
    for key, label in modes:
        m = agg.get(key, {})
        if not m:
            continue
        rows.append([label,
                     fnum(m.get("avg_factual_correctness"), "{:.2f}"),
                     fnum(m.get("avg_constraint_satisfaction"), "{:.2f}"),
                     fnum(m.get("avg_fluency_coherence"), "{:.2f}"),
                     fnum(m.get("avg_overall"), "{:.2f}")])
        out[key] = m
    table(["Mode", "Factual", "Constraint", "Fluency", "Overall"], rows,
          footnote=f"1–5 scale.  n_scored = {judge.get('n_total','?')} "
                   f"(judge={'LLM' if judge.get('use_llm') else 'offline'}).  "
                   f"Instruction-satisfaction/completeness/consistency: pdf §11.5.")
    return out


def section_D(pdf: dict, halluc: dict) -> dict:
    section_header("D. Hallucination evaluation — §11.6")
    h = pdf.get("section_11_6_hallucination", {})
    variants = [("V1_baseline", "V1 Baseline"), ("V2_doc_rag", "V2 Doc-RAG"),
                ("V3_onto_rag", "V3 Onto-RAG"), ("V4_hybrid", "V4 Hybrid"),
                ("V5_constrained", "V5 Constrained")]
    rows, out = [], {}
    for key, label in variants:
        m = h.get(key, {})
        if not m:
            continue
        rows.append([label, pct(m.get("overall_hallucination_rate")),
                     pct(m.get("unsupported_entity_rate")), pct(m.get("unsupported_relation_rate")),
                     pct(m.get("unsafe_recommendation_rate")), pct(m.get("contradiction_rate")),
                     pct(m.get("fabricated_logistics_rate"))])
        out[key] = m
    adv = g(halluc, "mode_results", "constrained", "overall_hallucination_rate")
    table(["Variant", "Overall", "Unsup. Entity", "Unsup. Relation", "Unsafe Rec", "Contradiction", "Fabricated"], rows,
          footnote=f"Adversarial probe (exp010, 50 hard queries): constrained overall = {pct(adv)}.")
    return out


def section_E(pdf: dict) -> dict:
    section_header("E. Explainability evaluation — §11.7")
    e = pdf.get("section_11_7_explainability", {})
    variants = [("V1_baseline", "V1 Baseline"), ("V2_doc_rag", "V2 Doc-RAG"),
                ("V3_onto_rag", "V3 Onto-RAG"), ("V4_hybrid", "V4 Hybrid"),
                ("V5_constrained", "V5 Constrained")]
    rows, out = [], {}
    for key, label in variants:
        m = e.get(key, {})
        if not m:
            continue
        rows.append([label, fnum(m.get("evidence_trace_coverage")),
                     fnum(m.get("explanation_faithfulness")),
                     fnum(m.get("ontology_alignment")),  # path interpretability proxy
                     fnum(m.get("user_trust_proxy"))])
        out[key] = m
    table(["Variant", "Evidence Cov", "Faithfulness", "Path Interp", "User Trust"], rows,
          footnote="Path interpretability proxied by ontology_alignment; user trust is a sampled proxy.")
    return out


def section_F(rob: dict) -> dict:
    section_header("F. Robustness evaluation — Phase 10F")
    vs = rob.get("variant_summaries", {})
    order = ["V1", "V2", "V3", "V4", "V5"]
    cats = ["paraphrase_consistency", "incomplete_query", "adversarial_resistance",
            "constraint_conflict", "unseen_destination"]
    rows, out = [], {}
    for v in order:
        m = vs.get(v, {})
        if not m:
            continue
        cells = [fnum(g(m, "categories", c, "pass_rate")) for c in cats]
        rows.append([f"{v} {m.get('variant_label','')}", pct(m.get("overall_pass_rate"))] + cells)
        out[v] = m
    table(["Variant", "Overall", "Paraphrase", "Incomplete", "Adversarial", "Conflict", "Unseen Dest"], rows,
          footnote=f"cells = per-category pass rate; n_items = {rob.get('n_items','?')}.")
    return out


def section_G(eff: dict, scaling: dict) -> dict:
    section_header("G. Efficiency / scalability evaluation — §11.8")
    st = eff.get("stages", {})
    storage = eff.get("storage_kb", {})
    total_storage_mb = sum(storage.values()) / 1024 if storage else None
    gen_ms = g(st, "generation_constrained", "avg_ms")
    cost_per_1k_gpu_s = (gen_ms / 1000 * 1000) if isinstance(gen_ms, (int, float)) else None
    rph = scaling.get("records_per_hour")

    rows = [
        ["Extraction latency",      fnum(g(st, "extraction", "avg_ms"), "{:.1f}"), "ms / record"],
        ["Validation throughput",   fnum(g(st, "validation", "throughput"), "{:.0f}"), "concepts / s"],
        ["Graph retrieval latency", fnum(g(st, "retrieval", "avg_ms"), "{:.1f}"), "ms / query"],
        ["Generation latency (V1)", fnum(g(st, "generation_no_rag", "avg_ms"), "{:.1f}"), "ms (no_rag baseline)"],
        ["Generation latency (V5)", fnum(gen_ms, "{:.1f}"), "ms (constrained, full system)"],
        ["Ontology storage size",   fnum(total_storage_mb, "{:.1f}"), "MB on disk"],
        ["Cost / 1K queries",       fnum(cost_per_1k_gpu_s, "{:.0f}"), "GPU-s (local model, $0 API)"],
        ["Scaling 10K→100K",
         f"{rph:,}" if isinstance(rph, (int, float)) and rph else "—",
         (f"rec/hr → 100K in ~{(100000/rph*3600):.0f}s" if isinstance(rph, (int, float)) and rph else "")],
    ]
    table(["Metric", "Value", "Unit / Note"], rows, aligns=["l", "r", "l"])
    return {
        "stages": st, "total_storage_mb": total_storage_mb,
        "cost_per_1k_gpu_seconds": cost_per_1k_gpu_s,
        "records_per_hour": rph,
    }


def section_deliverables(failure: dict) -> dict:
    section_header("Deliverables")
    checks = [
        ("evaluation scripts", PROJECT_ROOT / "experiments" / "exp020_extraction_prf" / "run_extraction_prf.py"),
        ("pdf-aligned §11 report", REPORTS / "pdf_aligned_metrics_report.json"),
        ("failure analysis report", REPORTS / "failure_analysis_report.json"),
        ("main results table (Table 3)", REPORTS / "main_results_table.md"),
        ("retrieval report", REPORTS / "exp009_retrieval_report.json"),
        ("hallucination report", REPORTS / "hallucination_report.json"),
        ("robustness report", REPORTS / "robustness_report.json"),
        ("efficiency report", REPORTS / "efficiency_report.json"),
        ("extraction P/R/F1 report", REPORTS / "extraction_prf_report.json"),
        ("5-variant comparison figure", FIGURES / "5variant_comparison.png"),
        ("hallucination reduction figure", FIGURES / "hallucination_reduction.png"),
        ("failure heatmap figure", FIGURES / "failure_category_heatmap.png"),
    ]
    rows = []
    for label, path in checks:
        rows.append(["✓" if path.exists() else "✗", label, str(path.relative_to(PROJECT_ROOT))])
    table(["", "Deliverable", "Path"], rows, aligns=["l", "l", "l"])

    # Failure analysis highlights
    weaknesses = failure.get("top_3_weaknesses", [])
    if weaknesses:
        frows = []
        for w in weaknesses[:3]:
            if isinstance(w, dict):
                frows.append([str(w.get("rank", "")), w.get("category", ""),
                              str(w.get("total_occurrences", "")), w.get("remedy", "")])
        if frows:
            print()
            table(["Rank", "Failure mode", "Occurrences", "Remedy"], frows, aligns=["r", "l", "r", "l"])
    return {"deliverables_present": all(p.exists() for _, p in checks)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 10 — consolidated evaluation")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-run offline metric generators (exp020, exp008, pdf metrics) first")
    ap.add_argument("--no-save", action="store_true", help="Print only; do not write report")
    ap.add_argument("--only", type=str, default=None,
                    help="Print only selected section(s), comma-separated: "
                         "A,B,C,D,E,F,G (e.g. --only A). Suppresses save/headline.")
    ap.add_argument("--ontology-version", default=None,
                    help="Ontology version for Section A (default: latest under artifacts/ontologies)")
    args = ap.parse_args()

    only = None
    if args.only:
        only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    hdr("PHASE 10 — EVALUATION FRAMEWORK  (Jeju Tourism Ontology + LLM)")
    print(f"  {DIM}Run: {datetime.now().isoformat(timespec='seconds')}  ·  "
          f"baselines V1–V5  ·  proposal §11.1–§11.9{RST}")

    if args.refresh:
        refresh()

    # Load all reports
    main_r = load("main_results_table.json")
    prf = load("extraction_prf_report.json")
    q8 = load("exp008_ontology_quality.json")
    val = load("exp003_validation_report.json")
    pdf = load("pdf_aligned_metrics_report.json")
    eff = load("efficiency_report.json")
    retr = load("exp009_retrieval_report.json")
    judge = load("llm_judge_scores.json")
    halluc = load("hallucination_report.json")
    rob = load("robustness_report.json")
    scaling = load("exp007_run_summary.json")
    failure = load("failure_analysis_report.json")

    # Resolve ontology version for Section A (default: latest available).
    def _latest_version() -> str:
        ont = PROJECT_ROOT / "artifacts" / "ontologies"
        cands = [p.name for p in ont.iterdir()
                 if p.is_dir() and (p / "concepts.json").exists()
                 and p.name[0].isdigit()]
        return max(cands, key=lambda v: [int(x) for x in v.split(".") if x.isdigit()]) if cands else "1.0.0"

    ov = args.ontology_version or _latest_version()
    ov_dir = PROJECT_ROOT / "artifacts" / "ontologies" / ov
    A_concepts = load_json_path(ov_dir / "concepts.json")
    A_relations = load_json_path(ov_dir / "relations.json")
    A_hierarchy = load_json_path(ov_dir / "hierarchy.json")

    # Section dispatch — allows running one sub-area at a time for review.
    builders = [
        ("A", "A_ontology_quality", lambda: section_A(ov, A_concepts, A_relations, A_hierarchy, prf)),
        ("B", "B_retrieval",        lambda: section_B(retr)),
        ("C", "C_response_quality", lambda: section_C(judge)),
        ("D", "D_hallucination",    lambda: section_D(pdf, halluc)),
        ("E", "E_explainability",   lambda: section_E(pdf)),
        ("F", "F_robustness",       lambda: section_F(rob)),
        ("G", "G_efficiency",       lambda: section_G(eff, scaling)),
    ]

    consolidated = {
        "phase": "Phase 10 — Evaluation Framework",
        "generated": datetime.now().isoformat(),
    }
    # Baselines print only on a full run (they contextualise all sections).
    if not only:
        consolidated["baselines"] = section_baselines(main_r)
    for letter, key, fn in builders:
        if only is None or letter in only:
            consolidated[key] = fn()
    if not only:
        consolidated["deliverables"] = section_deliverables(failure)

    # When reviewing a single section, stop here — no headline, no save.
    if only:
        print(f"\n  {CYN}Reviewing section(s): {', '.join(sorted(only))}.{RST} "
              f"{DIM}Re-run without --only for the full Phase 10 report.{RST}")
        print(f"{BLD}{'═'*78}{RST}\n")
        return

    # Headline — report BOTH hallucination definitions present in the artifacts,
    # since they measure different things and disagree (be transparent).
    # (1) categorized §11.6 benchmark; (2) strict per-entity grounding (exp018).
    h1_v1 = g(pdf, "section_11_6_hallucination", "V1_baseline", "overall_hallucination_rate")
    h1_v5 = g(pdf, "section_11_6_hallucination", "V5_constrained", "overall_hallucination_rate")
    h2_v1 = g(main_r, "results", "V1", "hallucination_rate")
    h2_v5 = g(main_r, "results", "V5", "hallucination_rate")
    v1g = g(main_r, "results", "V1", "avg_grounding")
    v5g = g(main_r, "results", "V5", "avg_grounding")
    hdr("PHASE 10 — HEADLINE")
    if isinstance(h1_v1, (int, float)) and isinstance(h1_v5, (int, float)):
        print(f"  {GRN}Hallucination (categorized §11.6 benchmark) V1→V5: "
              f"{pct(h1_v1)} → {pct(h1_v5)} ({(h1_v5-h1_v1)*100:+.1f}pp){RST}")
    if isinstance(h2_v1, (int, float)) and isinstance(h2_v5, (int, float)):
        print(f"  {YLW}Hallucination (strict per-entity grounding, exp018) V1→V5: "
              f"{pct(h2_v1)} → {pct(h2_v5)} ({(h2_v5-h2_v1)*100:+.1f}pp){RST}")
        print(f"  {DIM}  → the two definitions differ: §11.6 scores whole-answer category checks; "
              f"exp018 flags any ungrounded entity, so it stays high.{RST}")
    if isinstance(v1g, (int, float)) and isinstance(v5g, (int, float)):
        print(f"  {GRN}Grounding V1→V5: {fnum(v1g)} → {fnum(v5g)} "
              f"({(v5g-v1g)*100:+.1f}pp){RST}")
    print(f"  All 7 sub-areas (A–G) reported above across V1–V5 baselines.")

    if not args.no_save:
        jp = REPORTS / "phase10_summary_report.json"
        mp = REPORTS / "phase10_summary_report.md"
        jp.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
        mp.write_text(_markdown(consolidated), encoding="utf-8")
        print(f"\n  {CYN}Saved:{RST} {jp.relative_to(PROJECT_ROOT)}")
        print(f"  {CYN}Saved:{RST} {mp.relative_to(PROJECT_ROOT)}")
    print(f"{BLD}{'═'*78}{RST}\n")


def _markdown(c: dict) -> str:
    """Compact markdown rendering of the consolidated report."""
    L = [
        "# Phase 10 — Evaluation Framework — Summary Report",
        "",
        f"**Generated:** {c['generated']}  ",
        "**Baselines:** V1 Baseline LLM · V2 Doc-RAG · V3 Onto-RAG · V4 Hybrid · V5 Constrained+Validation",
        "",
        "## Baseline comparison (V1–V5)",
        "",
        "| Variant | Entity Recall | Entity Prec | Hallucination | Grounding |",
        "|---------|:---:|:---:|:---:|:---:|",
    ]
    for v, r in c["baselines"].items():
        L.append(f"| {v} {r['label']} | {fnum(r['entity_recall'])} | {fnum(r['entity_precision'])} "
                 f"| {pct(r['hallucination_rate'])} | {fnum(r['avg_grounding'])} |")

    a = c["A_ontology_quality"]
    cc, rr = a["concept"], a["relation"]
    L += ["", "## A. Ontology quality", "",
          f"- Concept P/R/F1: **{fnum(cc.get('precision'))} / {fnum(cc.get('recall'))} / {fnum(cc.get('f1'))}** "
          f"(precision={cc.get('precision_source')}, recall={cc.get('recall_source')})",
          f"- Relation P/R/F1: **{fnum(rr.get('precision'))} / {fnum(rr.get('recall'))} / {fnum(rr.get('f1'))}** "
          f"(precision={rr.get('precision_source')})",
          f"- Taxonomy depth/breadth: {fnum(a['taxonomy_depth'])} / {fnum(a['taxonomy_breadth'])}",
          f"- Orphan rate: {pct(a['orphan_rate'])}; redundancy: {pct(a['redundancy_rate'])}; "
          f"disconnected subgraphs: {a['disconnected_subgraph_count']} ({pct(a['disconnected_subgraph_rate'])})",
          f"- Logical consistency: {fnum(a['logical_consistency_rate'])}; "
          f"validation failure rate: {pct(a['validation_failure_rate'])}"]

    L += ["", "## D. Hallucination (V1–V5)", "",
          "| Variant | Overall | Entity | Relation | Unsafe | Contradiction | Fabricated |",
          "|---------|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for k, m in c["D_hallucination"].items():
        L.append(f"| {k} | {pct(m.get('overall_hallucination_rate'))} | {pct(m.get('unsupported_entity_rate'))} "
                 f"| {pct(m.get('unsupported_relation_rate'))} | {pct(m.get('unsafe_recommendation_rate'))} "
                 f"| {pct(m.get('contradiction_rate'))} | {pct(m.get('fabricated_logistics_rate'))} |")

    L += ["", "## G. Efficiency / scalability", "",
          f"- Retrieval latency: {fnum(g(c,'G_efficiency','stages','retrieval','avg_ms'),'{:.1f}')} ms/query",
          f"- Gen latency (V5): {fnum(g(c,'G_efficiency','stages','generation_constrained','avg_ms'),'{:.1f}')} ms",
          f"- Storage: {fnum(c['G_efficiency']['total_storage_mb'],'{:.1f}')} MB",
          f"- Scaling: {c['G_efficiency']['records_per_hour']:,} rec/hr" if isinstance(c['G_efficiency']['records_per_hour'],(int,float)) else "- Scaling: —",
          "",
          "> Sections B (retrieval), C (response quality), E (explainability), F (robustness) "
          "are rendered in full in the on-screen run and in the JSON sibling of this file.",
          ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
