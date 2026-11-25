"""
Experiment 020 — Real Concept/Relation Precision/Recall/F1 (Phase 10·A)

Computes ground-truth ontology extraction quality metrics from data, replacing
the hardcoded placeholder values that previously appeared in
``pdf_aligned_metrics_report.json``.

Inputs
------
  artifacts/ontologies/<version>/concepts.json
  artifacts/ontologies/<version>/relations.json
  artifacts/reports/manual_annotation_sample.json   (precision labels)
  artifacts/reports/gold_annotations_100.json        (recall: expected_concepts)

Outputs
-------
  artifacts/reports/extraction_prf_report.json
  artifacts/reports/extraction_prf_report.md

Precision is computed from human ``correct`` labels where present. If the
annotation template is still unlabelled, pass ``--silver`` to apply a documented
heuristic labeler so the pipeline produces end-to-end numbers; every silver
number is flagged as such in the report.

Usage
-----
  # Real precision once humans have filled manual_annotation_sample.json:
  python experiments/exp020_extraction_prf/run_extraction_prf.py --version 1.0.0

  # Interim run using heuristic silver precision labels:
  python experiments/exp020_extraction_prf/run_extraction_prf.py --version 1.0.0 --silver
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.extraction_prf import ExtractionPRFScorer
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("exp020")

ONTOLOGIES_DIR = ROOT / "artifacts" / "ontologies"
REPORTS_DIR = ROOT / "artifacts" / "reports"
ANNOTATIONS = REPORTS_DIR / "manual_annotation_sample.json"
GOLD = REPORTS_DIR / "gold_annotations_100.json"


def build_markdown(report: dict) -> str:
    c = report["concept"]
    r = report["relation"]

    def cell(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    lines = [
        "# Experiment 020 — Concept/Relation Precision/Recall/F1",
        "",
        f"**Date:** {report['date']}  ",
        f"**Ontology version:** {report['version']}  ",
        f"**Precision mode:** {'silver (heuristic) + human' if report['use_silver'] else 'human labels only'}  ",
        f"**Recall matching:** {'soft (token-subset)' if report['soft_match'] else 'strict (exact/alias)'}",
        "",
        "## Section 11.4.1 — Ontology Quality (P/R/F1)",
        "",
        "| Element | Precision | Recall | F1 | Prec. source | Labeled | Recall matched |",
        "|---------|:---------:|:------:|:--:|:------------:|:-------:|:--------------:|",
        f"| Concept | {cell(c['precision'])} | {cell(c['recall'])} | {cell(c['f1'])} "
        f"| {c['precision_source']} | {c['n_labeled']}/{c['n_total']} | {c['n_matched']}/{c['n_expected']} |",
        f"| Relation | {cell(r['precision'])} | {cell(r['recall'])} | {cell(r['f1'])} "
        f"| {r['precision_source']} | {r['n_labeled']}/{r['n_total']} | — |",
        "",
        "## Notes",
        "",
    ]
    for label, score in (("Concept", c), ("Relation", r)):
        for note in score.get("notes", []):
            lines.append(f"- **{label}:** {note}")
    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **Concept precision** = correct sampled concepts / labelled sampled concepts "
        "(`manual_annotation_sample.json`).",
        "- **Concept recall** = distinct gold `expected_concepts` present in the ontology / "
        "all distinct expected concepts (`gold_annotations_100.json`).",
        "- **Relation precision** = correct sampled triples / labelled sampled triples.",
        "- **Relation recall** = requires a gold relation set (not yet available).",
        "- **F1** = harmonic mean of precision and recall.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 020: Extraction P/R/F1")
    parser.add_argument("--version", default="1.0.0", help="Ontology version (default: 1.0.0)")
    parser.add_argument(
        "--silver", action="store_true",
        help="Apply heuristic silver precision labels to unlabelled samples",
    )
    parser.add_argument(
        "--strict-match", action="store_true",
        help="Use strict (exact/alias) recall matching instead of soft token-subset",
    )
    args = parser.parse_args()

    ensure_dir(REPORTS_DIR)
    vdir = ONTOLOGIES_DIR / args.version
    if not vdir.exists():
        logger.error("Ontology version %s not found at %s", args.version, vdir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 020 — Extraction P/R/F1")
    logger.info("  Version : %s", args.version)
    logger.info("  Silver  : %s", args.silver)
    logger.info("  Matching: %s", "strict" if args.strict_match else "soft")
    logger.info("=" * 60)

    concepts = load_json(vdir / "concepts.json") if (vdir / "concepts.json").exists() else []
    relations = load_json(vdir / "relations.json") if (vdir / "relations.json").exists() else []
    logger.info("Loaded %d concepts, %d relations", len(concepts), len(relations))

    if not ANNOTATIONS.exists():
        logger.error("Annotation file not found: %s", ANNOTATIONS)
        sys.exit(1)
    ann = load_json(ANNOTATIONS)
    concept_anns = ann.get("concept_annotations", [])
    relation_anns = ann.get("relation_annotations", [])

    gold = load_json(GOLD).get("annotations", []) if GOLD.exists() else []
    logger.info(
        "Loaded %d concept annotations, %d relation annotations, %d gold records",
        len(concept_anns), len(relation_anns), len(gold),
    )

    scorer = ExtractionPRFScorer(
        concepts=concepts,
        relations=relations,
        concept_annotations=concept_anns,
        relation_annotations=relation_anns,
        gold_annotations=gold,
        soft_match=not args.strict_match,
    )
    result = scorer.compute(use_silver=args.silver)

    report = {
        "experiment": "exp020_extraction_prf",
        "date": datetime.now().isoformat(),
        "version": args.version,
        **result,
        "inputs": {
            "annotations": str(ANNOTATIONS.relative_to(ROOT)),
            "gold": str(GOLD.relative_to(ROOT)) if GOLD.exists() else None,
        },
    }

    json_path = REPORTS_DIR / "extraction_prf_report.json"
    md_path = REPORTS_DIR / "extraction_prf_report.md"
    save_json(report, json_path)
    md_path.write_text(build_markdown(report), encoding="utf-8")
    logger.info("JSON report → %s", json_path)
    logger.info("Markdown    → %s", md_path)

    c, r = result["concept"], result["relation"]
    print("\n" + "=" * 70)
    print("EXPERIMENT 020 — EXTRACTION P/R/F1 COMPLETE")
    print("=" * 70)
    print(f"  Concept  : P={_p(c['precision'])} R={_p(c['recall'])} F1={_p(c['f1'])} "
          f"(prec={c['precision_source']}, {c['n_labeled']}/{c['n_total']} labeled, "
          f"recall {c['n_matched']}/{c['n_expected']})")
    print(f"  Relation : P={_p(r['precision'])} R={_p(r['recall'])} F1={_p(r['f1'])} "
          f"(prec={r['precision_source']}, {r['n_labeled']}/{r['n_total']} labeled)")
    if c["precision_source"] == "none":
        print("\n  ⚠ Concept precision pending: fill `correct` in manual_annotation_sample.json")
        print("    or re-run with --silver for an interim heuristic estimate.")
    print(f"\n  JSON     : {json_path}")
    print(f"  Markdown : {md_path}")
    print("=" * 70)


def _p(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
