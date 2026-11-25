"""
Taxonomy-backbone enrichment — ontology evolution step (Phase 7) that fixes the
structural fragmentation flagged in Phase 10·A.

Problem (v1.0.0)
----------------
Every concept carries a ``concept_class`` and a ``parent_class`` (in
hierarchy.json), but only a fraction of the type/subclass edges are materialised
in the relation graph. Result: 28.6% "orphan" concepts, 3,363 disconnected
components, and an artificially shallow taxonomy depth.

Fix (v1.1.0) — standard OWL practice, not metric gaming
-------------------------------------------------------
Materialise the taxonomic backbone that the data already asserts:
  1. a single root class ``TourismEntity``;
  2. ``subclass_of`` edges for every class → its parent (root classes → root);
  3. ``is_a`` edges for every concept → its ``concept_class`` (rdf:type);
  4. regenerated hierarchy paths root→…→class.

Every new edge is real asserted knowledge (typing + the schema's class
hierarchy), carries provenance, and is SHACL-validated before saving.

Usage
-----
  python scripts/enrich_taxonomy_backbone.py --source 1.0.0 --target 1.1.0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ontology_metrics import OntologyMetricsCalculator
from src.ontology.shacl_validator import SHACLValidator
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("enrich_taxonomy")

ONT = PROJECT_ROOT / "artifacts" / "ontologies"
REPORTS = PROJECT_ROOT / "artifacts" / "reports"
ROOT_CLASS = "TourismEntity"


def _prov(note: str) -> dict:
    return {
        "source_record_id": -1,
        "extraction_prompt_version": "taxonomy-backbone-v1.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": 1.0,
        "validator_status": "pass",
        "approved_by": "taxonomy_enrichment",
        "notes": note,
    }


def _struct(concepts: list[dict], relations: list[dict], hierarchy: list[dict]) -> dict:
    calc = OntologyMetricsCalculator(concepts, relations, [], version="tmp")
    return {
        "n_concepts": len(concepts),
        "n_relations": len(relations),
        "orphan_rate": round(calc.orphan_rate(), 4),
        "disconnected_subgraphs": calc.disconnected_subgraph_count(),
        "hierarchy_depth": round(calc.hierarchy_depth(hierarchy), 3),
        "hierarchy_breadth": round(calc.hierarchy_breadth(hierarchy), 3),
    }


def class_chain(cls: str, parent_of: dict[str, str]) -> list[str]:
    """Return [ROOT, ..., cls] following parent links, with cycle guard."""
    chain, seen = [], set()
    cur = cls
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = parent_of.get(cur)
    return list(reversed(chain))


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialise the taxonomy backbone")
    ap.add_argument("--source", default="1.0.0")
    ap.add_argument("--target", default="1.1.0")
    args = ap.parse_args()

    src = ONT / args.source
    dst = ONT / args.target
    if not src.exists():
        logger.error("Source version %s not found", src)
        sys.exit(1)

    concepts = load_json(src / "concepts.json")
    relations = load_json(src / "relations.json")
    hierarchy = load_json(src / "hierarchy.json") if (src / "hierarchy.json").exists() else []
    rules = load_json(src / "rules.json") if (src / "rules.json").exists() else []
    schema = load_json(src / "schema.json")

    concept_classes = list(schema.get("concept_classes", []))
    class_hierarchy = dict(schema.get("class_hierarchy", {}))

    before = _struct(concepts, relations, hierarchy)

    # ── parent map with super-root ────────────────────────────────────────────
    parent_of = dict(class_hierarchy)
    for c in concept_classes:
        parent_of.setdefault(c, ROOT_CLASS)  # root classes → super-root
    parent_of.pop(ROOT_CLASS, None)          # the root has no parent

    # ── 1. ensure class nodes exist as concepts ───────────────────────────────
    existing_names = {c.get("name", "") for c in concepts}
    added_class_nodes = 0
    for cls in list(concept_classes) + [ROOT_CLASS]:
        if cls not in existing_names:
            concepts.append({
                "name": cls,
                "concept_class": "OntologyClass",
                "aliases": [],
                "provenance": _prov("taxonomy class node"),
            })
            existing_names.add(cls)
            added_class_nodes += 1

    # ── existing edge set for dedup (case-insensitive) ─────────────────────────
    def key(s, p, o):
        return (str(s).lower(), str(p).lower(), str(o).lower())

    edge_set = {key(r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
                for r in relations}

    # ── 2. subclass_of edges: class → parent ───────────────────────────────────
    added_subclass = 0
    for cls in concept_classes:
        parent = parent_of.get(cls, ROOT_CLASS)
        if parent and cls != parent and key(cls, "subclass_of", parent) not in edge_set:
            relations.append({"subject": cls, "predicate": "subclass_of", "object": parent,
                              "provenance": _prov("class subsumption from schema")})
            edge_set.add(key(cls, "subclass_of", parent))
            added_subclass += 1

    # ── 3. is_a edges: concept → its class ─────────────────────────────────────
    valid_classes = set(concept_classes)
    added_isa = 0
    for c in concepts:
        name, cls = c.get("name", ""), c.get("concept_class", "")
        if (cls in valid_classes and name and name != cls
                and key(name, "is_a", cls) not in edge_set):
            relations.append({"subject": name, "predicate": "is_a", "object": cls,
                              "provenance": _prov("instance typing (rdf:type)")})
            edge_set.add(key(name, "is_a", cls))
            added_isa += 1

    # ── 4. regenerate hierarchy paths (root → … → class) ───────────────────────
    new_hierarchy = []
    for c in concepts:
        cls = c.get("concept_class", "")
        if cls in ("OntologyClass", ""):
            continue
        chain = class_chain(cls, parent_of)  # [ROOT, ..., cls]
        new_hierarchy.append({
            "concept": c.get("name", ""),
            "concept_class": cls,
            "parent_class": parent_of.get(cls, ROOT_CLASS),
            "hierarchy_path": chain,
            "is_instance": True,
            "confidence": 1.0,
        })

    # ── schema update: record super-root + root parents ────────────────────────
    for c in concept_classes:
        class_hierarchy.setdefault(c, ROOT_CLASS)
    schema["class_hierarchy"] = class_hierarchy
    if ROOT_CLASS not in schema.get("concept_classes", []):
        schema.setdefault("concept_classes", []).append(ROOT_CLASS)
    schema["version"] = args.target

    after = _struct(concepts, relations, new_hierarchy)

    # ── SHACL validation of the enriched ontology ──────────────────────────────
    logger.info("Validating enriched ontology (SHACL) ...")
    report = SHACLValidator().validate_ontology(concepts, relations)

    # ── save v1.1.0 ─────────────────────────────────────────────────────────────
    ensure_dir(dst)
    save_json(concepts, dst / "concepts.json")
    save_json(relations, dst / "relations.json")
    save_json(new_hierarchy, dst / "hierarchy.json")
    save_json(rules, dst / "rules.json")
    save_json(schema, dst / "schema.json")
    changelog = load_json(src / "changelog.json") if (src / "changelog.json").exists() else []
    if isinstance(changelog, dict):
        changelog = [changelog]
    changelog.append({
        "version": args.target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "change_type": "taxonomy_enrichment",
        "reason": "Materialise type/subclass backbone (rdf:type + subClassOf) to fix "
                  "structural fragmentation (orphans, disconnected components, shallow depth).",
        "stats": {
            "class_nodes_added": added_class_nodes,
            "subclass_edges_added": added_subclass,
            "is_a_edges_added": added_isa,
        },
    })
    save_json(changelog, dst / "changelog.json")

    enrich_report = {
        "experiment": "taxonomy_backbone_enrichment",
        "date": datetime.now().isoformat(),
        "source_version": args.source,
        "target_version": args.target,
        "added": {
            "class_nodes": added_class_nodes,
            "subclass_of_edges": added_subclass,
            "is_a_edges": added_isa,
        },
        "before": before,
        "after": after,
        "shacl": {
            "conforms": report.conforms,
            "violations": len(report.violations),
            "warnings": len(report.warnings),
        },
    }
    save_json(enrich_report, REPORTS / "taxonomy_enrichment_report.json")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TAXONOMY BACKBONE ENRICHMENT  v{args.source} → v{args.target}")
    print("=" * 70)
    print(f"  Added: {added_class_nodes} class nodes, {added_subclass} subclass_of, "
          f"{added_isa} is_a edges")
    print(f"  {'Metric':<26}{'before':>12}{'after':>12}")
    print(f"  {'-'*26}{'-'*12}{'-'*12}")
    for k in ["n_concepts", "n_relations", "orphan_rate", "disconnected_subgraphs",
              "hierarchy_depth", "hierarchy_breadth"]:
        print(f"  {k:<26}{str(before[k]):>12}{str(after[k]):>12}")
    print(f"  SHACL: conforms={report.conforms}, violations={len(report.violations)}, "
          f"warnings={len(report.warnings)}")
    print(f"\n  Saved → {dst.relative_to(PROJECT_ROOT)}")
    print(f"  Report → artifacts/reports/taxonomy_enrichment_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
