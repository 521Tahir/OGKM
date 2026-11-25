"""
Experiment 003 — Ontology Validation, Repair & Governance

Steps:
  1. Load ontology v0.2.0 (concepts, relations, rules, hierarchy)
  2. Run SHACLValidator on the full ontology snapshot
  3. Auto-repair simple violations (rule-based, no LLM):
       - Remove concepts with invalid/unknown class
       - Remove concepts with empty/too-short names
       - Remove relations with invalid predicate
       - Remove relations with empty subject or object
  4. Build expert review queue for domain/range warnings and low-confidence items
  5. Produce ontology diff vs v0.1.0
  6. Save repaired ontology as v0.3.0 (Turtle + JSON-LD)
  7. Write validation report to artifacts/reports/
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.shacl_validator import SHACLValidator, SHACLViolation
from src.ontology.exporters import OntologyExporter
from src.ontology.versioning import OntologyVersionManager
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp003")
ONTOLOGIES_DIR = PROJECT_ROOT / "artifacts" / "ontologies"
REPORTS_DIR    = PROJECT_ROOT / "artifacts" / "reports"


# ── Auto-repair helpers ───────────────────────────────────────────────────────

def _repair_concepts(
    concepts: list[dict],
    violations: list[SHACLViolation],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (kept, removed, expert_queue).
    Auto-removes concepts with hard Violations; flags warnings for expert queue.
    """
    removed_names: set[str] = set()
    expert_names: set[str] = set()

    for v in violations:
        node = v.focus_node
        if v.severity == "Violation" and v.shape_name == "ConceptShape":
            removed_names.add(node)
        elif v.severity == "Warning" and v.shape_name == "ConceptShape":
            expert_names.add(node)

    kept, removed, expert = [], [], []
    for c in concepts:
        name = c.get("name", "")
        if name in removed_names:
            removed.append({**c, "_removal_reason": "SHACL ConceptShape Violation"})
        elif name in expert_names:
            kept.append(c)          # keep but also add to queue
            expert.append({**c, "_review_reason": "SHACL ConceptShape Warning"})
        else:
            kept.append(c)
    return kept, removed, expert


def _repair_relations(
    relations: list[dict],
    violations: list[SHACLViolation],
    kept_concept_names: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (kept, removed, expert_queue).
    Auto-removes relations with invalid predicates or empty subject/object.
    Adds dangling-reference warnings to expert queue (but keeps the relation).
    """
    removed_nodes: set[str] = set()
    expert_nodes: set[str] = set()

    for v in violations:
        node = v.focus_node
        if v.severity == "Violation" and v.shape_name == "RelationShape":
            removed_nodes.add(node)
        elif v.severity == "Warning" and v.shape_name == "RelationShape":
            expert_nodes.add(node)

    kept, removed, expert = [], [], []
    for r in relations:
        node = f"{r.get('subject','')}--{r.get('predicate','')}-->{r.get('object','')}"
        if node in removed_nodes:
            removed.append({**r, "_removal_reason": "SHACL RelationShape Violation"})
        else:
            kept.append(r)
            if node in expert_nodes:
                expert.append({**r, "_review_reason": "SHACL RelationShape Warning"})
    return kept, removed, expert


# ── Ontology diff ─────────────────────────────────────────────────────────────

def _ontology_diff(
    old_concepts: list[dict],
    new_concepts: list[dict],
    old_relations: list[dict],
    new_relations: list[dict],
) -> dict:
    old_concept_names = {c.get("name", "") for c in old_concepts}
    new_concept_names = {c.get("name", "") for c in new_concepts}
    old_rel_keys = {
        (r.get("subject",""), r.get("predicate",""), r.get("object",""))
        for r in old_relations
    }
    new_rel_keys = {
        (r.get("subject",""), r.get("predicate",""), r.get("object",""))
        for r in new_relations
    }
    return {
        "concepts_added":   sorted(new_concept_names - old_concept_names),
        "concepts_removed": sorted(old_concept_names - new_concept_names),
        "relations_added":   len(new_rel_keys - old_rel_keys),
        "relations_removed": len(old_rel_keys - new_rel_keys),
        "total_concepts_before": len(old_concepts),
        "total_concepts_after":  len(new_concepts),
        "total_relations_before": len(old_relations),
        "total_relations_after":  len(new_relations),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp003: Ontology validation + repair")
    parser.add_argument("--source-version", default="0.2.0",
                        help="Ontology version to validate (default: 0.2.0)")
    parser.add_argument("--target-version", default="0.3.0",
                        help="Version label for repaired ontology (default: 0.3.0)")
    parser.add_argument("--diff-against", default="0.1.0",
                        help="Version to diff against (default: 0.1.0)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum concept/relation confidence (default: 0.5)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level,
                  log_file=str(PROJECT_ROOT / "artifacts" / "runs" / "exp003_validation.log"))

    logger.info("=== Experiment 003: Ontology Validation + Repair ===")
    logger.info(f"  Source version : {args.source_version}")
    logger.info(f"  Target version : {args.target_version}")
    logger.info(f"  Diff against   : {args.diff_against}")

    # ── 1. Load source ontology ───────────────────────────────────────────────
    src_dir = ONTOLOGIES_DIR / args.source_version
    concepts  = load_json(src_dir / "concepts.json")
    relations = load_json(src_dir / "relations.json")
    rules     = load_json(src_dir / "rules.json")
    hierarchy = load_json(src_dir / "hierarchy.json")
    logger.info(
        f"Loaded v{args.source_version}: "
        f"{len(concepts):,} concepts, {len(relations):,} relations, "
        f"{len(rules)} rules, {len(hierarchy):,} hierarchy nodes"
    )

    # ── 2. SHACL validation ───────────────────────────────────────────────────
    validator = SHACLValidator(
        min_confidence=args.min_confidence,
        check_domain_range=True,
        check_high_confidence=True,
    )
    logger.info("Running SHACL validation on full ontology …")
    shacl_report = validator.validate_ontology(concepts, relations)
    logger.info(
        f"SHACL report: conforms={shacl_report.conforms} | "
        f"{len(shacl_report.violations)} violations, "
        f"{len(shacl_report.warnings)} warnings"
    )

    # ── 3. Auto-repair ────────────────────────────────────────────────────────
    logger.info("Auto-repairing violations …")
    all_violations = shacl_report.violations + shacl_report.warnings

    repaired_concepts, removed_concepts, expert_concepts = _repair_concepts(
        concepts, all_violations
    )
    kept_names = {c.get("name", "") for c in repaired_concepts}
    repaired_relations, removed_relations, expert_relations = _repair_relations(
        relations, all_violations, kept_names
    )

    logger.info(
        f"Repair complete: "
        f"concepts {len(concepts):,} → {len(repaired_concepts):,} "
        f"(removed {len(removed_concepts)}), "
        f"relations {len(relations):,} → {len(repaired_relations):,} "
        f"(removed {len(removed_relations)})"
    )

    # ── 4. Expert review queue ────────────────────────────────────────────────
    expert_queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_version": args.source_version,
        "concepts_for_review": expert_concepts,
        "relations_for_review": expert_relations,
        "total_items": len(expert_concepts) + len(expert_relations),
    }
    queue_path = REPORTS_DIR / "expert_review_queue.json"
    save_json(expert_queue, queue_path)
    logger.info(
        f"Expert review queue: "
        f"{len(expert_concepts)} concepts + {len(expert_relations)} relations "
        f"→ {queue_path}"
    )

    # ── 5. Ontology diff vs baseline ──────────────────────────────────────────
    diff = {}
    diff_dir = ONTOLOGIES_DIR / args.diff_against
    if diff_dir.exists():
        old_concepts  = load_json(diff_dir / "concepts.json")
        old_relations = load_json(diff_dir / "relations.json")
        diff = _ontology_diff(old_concepts, repaired_concepts,
                              old_relations, repaired_relations)
        logger.info(
            f"Diff vs v{args.diff_against}: "
            f"+{len(diff['concepts_added'])} / -{len(diff['concepts_removed'])} concepts, "
            f"+{diff['relations_added']} / -{diff['relations_removed']} relations"
        )

    # ── 6. Save v0.3.0 ───────────────────────────────────────────────────────
    target_dir = ONTOLOGIES_DIR / args.target_version
    ensure_dir(target_dir)

    save_json(repaired_concepts,  target_dir / "concepts.json")
    save_json(repaired_relations, target_dir / "relations.json")
    save_json(rules,              target_dir / "rules.json")
    save_json(hierarchy,          target_dir / "hierarchy.json")

    # Copy schema.json from source
    src_schema = src_dir / "schema.json"
    if src_schema.exists():
        import shutil
        shutil.copy(src_schema, target_dir / "schema.json")

    # Changelog entry
    changelog = [{
        "version": args.target_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "change_type": "validation_repair",
        "reason": f"SHACL validation + auto-repair from v{args.source_version}",
        "stats": {
            "concepts_removed": len(removed_concepts),
            "relations_removed": len(removed_relations),
            "violations_fixed": len(shacl_report.violations),
            "warnings_flagged": len(shacl_report.warnings),
        },
    }]
    save_json(changelog, target_dir / "changelog.json")
    logger.info(f"Saved ontology v{args.target_version} → {target_dir}")

    # Export Turtle + JSON-LD
    exporter = OntologyExporter(output_dir=ONTOLOGIES_DIR)
    ttl_path  = exporter.export_turtle(
        repaired_concepts, repaired_relations, rules,
        version=args.target_version,
        output_path=ONTOLOGIES_DIR / f"tourism_v{args.target_version}.ttl",
    )
    jld_path = exporter.export_jsonld(
        repaired_concepts, repaired_relations,
        version=args.target_version,
        output_path=ONTOLOGIES_DIR / f"tourism_v{args.target_version}.jsonld",
    )
    logger.info(f"  Turtle  : {ttl_path}")
    logger.info(f"  JSON-LD : {jld_path}")

    # Update current.json pointer
    save_json(
        {"current_version": args.target_version,
         "updated_at": datetime.now(timezone.utc).isoformat()},
        ONTOLOGIES_DIR / "current.json",
    )

    # ── 7. Validation report ──────────────────────────────────────────────────
    # Violation breakdown by shape
    violation_by_shape: dict[str, int] = defaultdict(int)
    warning_by_shape:   dict[str, int] = defaultdict(int)
    for v in shacl_report.violations:
        violation_by_shape[v.shape_name] += 1
    for w in shacl_report.warnings:
        warning_by_shape[w.shape_name] += 1

    report = {
        "experiment": "exp003_validation",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "source_version": args.source_version,
        "target_version": args.target_version,
        "shacl": {
            "conforms":        shacl_report.conforms,
            "violations":      len(shacl_report.violations),
            "warnings":        len(shacl_report.warnings),
            "by_shape_violations": dict(violation_by_shape),
            "by_shape_warnings":   dict(warning_by_shape),
        },
        "repair": {
            "concepts_before":   len(concepts),
            "concepts_after":    len(repaired_concepts),
            "concepts_removed":  len(removed_concepts),
            "relations_before":  len(relations),
            "relations_after":   len(repaired_relations),
            "relations_removed": len(removed_relations),
        },
        "expert_queue": {
            "concepts": len(expert_concepts),
            "relations": len(expert_relations),
            "total": len(expert_concepts) + len(expert_relations),
            "path": str(queue_path),
        },
        "diff_vs_v010": diff,
        "output": {
            "ontology_dir": str(target_dir),
            "turtle":  str(ttl_path),
            "jsonld":  str(jld_path),
        },
    }
    report_path = REPORTS_DIR / "exp003_validation_report.json"
    save_json(report, report_path)
    logger.info(f"Validation report → {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 003 — ONTOLOGY VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Source : v{args.source_version}  →  Repaired : v{args.target_version}")
    print(f"\n  SHACL:")
    print(f"    Violations : {len(shacl_report.violations)}")
    print(f"    Warnings   : {len(shacl_report.warnings)}")
    print(f"\n  Auto-repair:")
    print(f"    Concepts   : {len(concepts):,} → {len(repaired_concepts):,} "
          f"(removed {len(removed_concepts)})")
    print(f"    Relations  : {len(relations):,} → {len(repaired_relations):,} "
          f"(removed {len(removed_relations)})")
    print(f"\n  Expert review queue : {len(expert_concepts) + len(expert_relations)} items")
    if diff:
        print(f"\n  Diff vs v{args.diff_against}:")
        print(f"    Concepts +{len(diff['concepts_added'])} / -{len(diff['concepts_removed'])}")
        print(f"    Relations +{diff['relations_added']} / -{diff['relations_removed']}")
    print(f"\n  Report   : {report_path}")
    print(f"  Ontology : {target_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
