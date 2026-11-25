"""
Experiment 008 — Ontology Quality Across Versions

Runs OntologyMetricsCalculator on every available version (v0.1.0 → v0.7.0),
computes all quality metrics including the 4 new taxonomy metrics, and
produces:

  artifacts/reports/exp008_ontology_quality.json    — full per-version data
  artifacts/reports/exp008_ontology_quality.md      — markdown table for paper

Usage:
  python experiments/exp008_ontology_quality/run_quality.py \
      --versions 0.1.0 0.2.0 0.3.0 0.5.0 0.7.0 \
      2>&1 | tee artifacts/runs/exp008.log

  or run all available versions (default):
  python experiments/exp008_ontology_quality/run_quality.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.ontology_metrics import OntologyMetricsCalculator, OntologySnapshot
from src.utils.logging_utils import get_logger
from src.utils.io import save_json, ensure_dir

logger = get_logger("exp008")

# Default versions to evaluate (all that exist)
DEFAULT_VERSIONS = ["0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.7.0"]

ONTOLOGIES_DIR = ROOT / "artifacts" / "ontologies"
REPORTS_DIR    = ROOT / "artifacts" / "reports"
RUNS_DIR       = ROOT / "artifacts" / "runs"
TEST_SPLIT     = ROOT / "data" / "splits" / "test.jsonl"


def load_test_records(n: int = 500) -> list[dict]:
    """Load first n records from test split for concept_coverage computation."""
    if not TEST_SPLIT.exists():
        logger.warning("Test split not found — concept_coverage will be 0.0")
        return []
    records = []
    with open(TEST_SPLIT) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} test records for coverage computation")
    return records


def run_version(version: str, test_records: list[dict]) -> OntologySnapshot | None:
    """Compute metrics for one ontology version."""
    vdir = ONTOLOGIES_DIR / version
    if not vdir.exists():
        logger.warning(f"Version {version} not found at {vdir} — skipping")
        return None

    logger.info(f"=== Processing v{version} ===")
    try:
        concepts, relations, rules, hierarchy = OntologyMetricsCalculator.load_ontology_version(
            ONTOLOGIES_DIR, version
        )
        logger.info(f"  Loaded: {len(concepts)} concepts, {len(relations)} relations, "
                    f"{len(rules)} rules, {len(hierarchy)} hierarchy nodes")

        calc = OntologyMetricsCalculator(concepts, relations, rules, version=version)
        snap = calc.compute(test_records=test_records, hierarchy=hierarchy)
        return snap

    except Exception as e:
        logger.error(f"  Failed to process v{version}: {e}")
        return None


def build_markdown_table(snapshots: list[OntologySnapshot]) -> str:
    """Build the paper-ready markdown comparison table."""
    lines = [
        "# Experiment 008 — Ontology Quality Across Versions",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
        f"**Versions evaluated:** {len(snapshots)}",
        "",
        "## Core Metrics",
        "",
        "| Version | Concepts | Relations | Rules | Classes | Density | Schema% | Coverage% |",
        "|---------|:--------:|:---------:|:-----:|:-------:|:-------:|:-------:|:---------:|",
    ]
    for s in snapshots:
        lines.append(
            f"| v{s.version} | {s.n_concepts:,} | {s.n_relations:,} | {s.n_rules} "
            f"| {s.n_classes} | {s.relation_density:.2f} "
            f"| {s.schema_completeness*100:.0f}% | {s.concept_coverage*100:.1f}% |"
        )

    lines += [
        "",
        "## Taxonomy Metrics (New — Week 5)",
        "",
        "| Version | Hier. Depth | Hier. Breadth | Orphan Rate | Components |",
        "|---------|:-----------:|:-------------:|:-----------:|:----------:|",
    ]
    for s in snapshots:
        lines.append(
            f"| v{s.version} | {s.hierarchy_depth:.2f} | {s.hierarchy_breadth:.2f} "
            f"| {s.orphan_rate*100:.1f}% | {s.disconnected_subgraph_count} |"
        )

    lines += [
        "",
        "## Growth Analysis",
        "",
        "| Version → | Δ Concepts | Δ Relations | Growth% Concepts | Growth% Relations |",
        "|-----------|:----------:|:-----------:|:----------------:|:-----------------:|",
    ]
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i-1], snapshots[i]
        dc = curr.n_concepts  - prev.n_concepts
        dr = curr.n_relations - prev.n_relations
        gc = dc / max(prev.n_concepts, 1) * 100
        gr = dr / max(prev.n_relations, 1) * 100
        lines.append(
            f"| v{prev.version}→v{curr.version} | {dc:+,} | {dr:+,} "
            f"| {gc:+.1f}% | {gr:+.1f}% |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **Density**: relations / concepts (connectivity measure)",
        "- **Schema%**: fraction of 26 target classes populated",
        "- **Coverage%**: fraction of 500 test records containing ≥1 ontology concept",
        "- **Hier. Depth**: average depth of is-a chains from leaf to root",
        "- **Hier. Breadth**: average number of children per parent node",
        "- **Orphan Rate**: fraction of concepts with zero relations",
        "- **Components**: isolated connected components in the relation graph",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Experiment 008: Ontology Quality")
    parser.add_argument(
        "--versions", nargs="+", default=DEFAULT_VERSIONS,
        help="Ontology versions to evaluate"
    )
    parser.add_argument(
        "--test-records", type=int, default=500,
        help="Number of test records to use for concept coverage (default: 500)"
    )
    args = parser.parse_args()

    ensure_dir(REPORTS_DIR)
    ensure_dir(RUNS_DIR)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 008 — Ontology Quality Evaluation")
    logger.info(f"  Versions : {args.versions}")
    logger.info(f"  Test recs: {args.test_records}")
    logger.info("=" * 60)

    test_records = load_test_records(args.test_records)

    snapshots: list[OntologySnapshot] = []
    for version in args.versions:
        snap = run_version(version, test_records)
        if snap is not None:
            snapshots.append(snap)

    if not snapshots:
        logger.error("No versions processed — exiting")
        sys.exit(1)

    # ── Build evolution report ────────────────────────────────────────────────
    evolution = OntologyMetricsCalculator.compare_versions(snapshots)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "experiment":    "exp008_ontology_quality",
        "date":          datetime.now().isoformat(),
        "versions":      args.versions,
        "snapshots":     [s.to_dict() for s in snapshots],
        "concept_growth":   evolution.concept_growth,
        "relation_growth":  evolution.relation_growth,
    }
    json_path = REPORTS_DIR / "exp008_ontology_quality.json"
    save_json(report, json_path)
    logger.info(f"JSON report → {json_path}")

    # ── Save Markdown table ───────────────────────────────────────────────────
    md_path = REPORTS_DIR / "exp008_ontology_quality.md"
    md_path.write_text(build_markdown_table(snapshots), encoding="utf-8")
    logger.info(f"Markdown report → {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 008 — ONTOLOGY QUALITY COMPLETE")
    print("=" * 70)
    print(f"  Versions evaluated : {len(snapshots)}")
    print()
    print(f"  {'Version':<10} {'Concepts':>9} {'Relations':>10} {'Density':>8} "
          f"{'Schema%':>8} {'Depth':>6} {'Orphan%':>8} {'Comp':>5}")
    print(f"  {'-'*10} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*5}")
    for s in snapshots:
        print(
            f"  v{s.version:<9} {s.n_concepts:>9,} {s.n_relations:>10,} "
            f"{s.relation_density:>8.2f} {s.schema_completeness*100:>7.0f}% "
            f"{s.hierarchy_depth:>6.2f} {s.orphan_rate*100:>7.1f}% "
            f"{s.disconnected_subgraph_count:>5}"
        )
    print()
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
