"""
Ontology diff tool — compare any two stored versions.

Usage:
  python scripts/diff_ontologies.py --v1 0.5.0 --v2 0.6.0
  python scripts/diff_ontologies.py --v1 0.6.0 --v2 0.7.0 --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.versioning import OntologyVersionManager
from src.utils.io import save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("scripts.diff_ontologies")


def print_diff(diff: dict) -> None:
    c = diff["concepts"]
    r = diff["relations"]
    print(f"\n{'='*60}")
    print(f"  Ontology Diff: v{diff['from_version']} → v{diff['to_version']}")
    print(f"{'='*60}")
    print(f"\nConcepts:")
    print(f"  Total:    {c['total_old']:>6} → {c['total_new']:>6}  "
          f"(Δ {c['total_new'] - c['total_old']:+d})")
    print(f"  Added:    {c['added_count']:>6}")
    print(f"  Removed:  {c['removed_count']:>6}")
    print(f"  Modified: {c['modified_count']:>6}")
    print(f"\nRelations:")
    print(f"  Total:    {r['total_old']:>6} → {r['total_new']:>6}  "
          f"(Δ {r['total_new'] - r['total_old']:+d})")
    print(f"  Added:    {r['added_count']:>6}")
    print(f"  Removed:  {r['removed_count']:>6}")

    if c.get("added_by_class"):
        print(f"\nAdded concepts by class:")
        for cls, n in sorted(c["added_by_class"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<30} {n:>4}")

    if c.get("added") and len(c["added"]) <= 20:
        print(f"\nNew concepts: {', '.join(c['added'][:20])}")

    if c.get("modified"):
        print(f"\nModified concepts (class changes):")
        for m in c["modified"][:10]:
            print(f"  {m['name']}: {m['old']} → {m['new']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two ontology versions")
    parser.add_argument("--v1", required=True, help="Older version (e.g. 0.5.0)")
    parser.add_argument("--v2", required=True, help="Newer version (e.g. 0.6.0)")
    parser.add_argument("--base-dir", default="artifacts/ontologies",
                        help="Ontology base directory")
    parser.add_argument("--save", action="store_true",
                        help="Save diff to artifacts/reports/diff_<v1>_<v2>.json")
    args = parser.parse_args()

    vm = OntologyVersionManager(ROOT / args.base_dir)
    available = vm.list_versions()
    logger.info(f"Available versions: {available}")

    for v in [args.v1, args.v2]:
        if v not in available:
            print(f"ERROR: version '{v}' not found. Available: {available}")
            sys.exit(1)

    diff = vm.generate_diff_report(args.v1, args.v2)
    print_diff(diff)

    if args.save:
        out = ROOT / "artifacts" / "reports" / f"diff_{args.v1}_{args.v2}.json"
        ensure_dir(out.parent)
        save_json(diff, out)
        print(f"Diff saved → {out}")


if __name__ == "__main__":
    main()
