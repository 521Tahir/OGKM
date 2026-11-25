"""
ingest_to_neo4j.py — Load a Jeju tourism ontology version into Neo4j.

Usage
-----
  # Load v0.7.0 (default) — reads NEO4J_URI/USER/PASSWORD from .env
  python scripts/ingest_to_neo4j.py

  # Load a specific version
  python scripts/ingest_to_neo4j.py --ontology-version 0.7.0

  # Clear existing graph before loading
  python scripts/ingest_to_neo4j.py --clear

  # Use a custom Neo4j URI
  NEO4J_URI=bolt://myserver:7687 python scripts/ingest_to_neo4j.py

  # Dry-run: print stats without writing
  python scripts/ingest_to_neo4j.py --dry-run

Environment variables (all optional — also read from .env):
  NEO4J_URI       bolt://localhost:7687
  NEO4J_USER      neo4j
  NEO4J_PASSWORD  <your password>
  ONTOLOGY_VERSION  0.7.0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Project root → add to sys.path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _load_env() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest tourism ontology into Neo4j")
    p.add_argument(
        "--ontology-version", "-v",
        default=os.environ.get("ONTOLOGY_VERSION", "0.7.0"),
        help="Ontology version to load (default: 0.7.0)",
    )
    p.add_argument(
        "--ontology-dir",
        default=str(PROJECT_ROOT / "artifacts" / "ontologies"),
        help="Root directory of versioned ontologies",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Delete all existing Concept nodes before loading",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print concept/relation counts without writing to Neo4j",
    )
    p.add_argument(
        "--batch-size", type=int, default=500,
        help="Concepts/relations per batch (default: 500)",
    )
    return p.parse_args()


def _dry_run(version: str, ontology_dir: str) -> None:
    from src.utils.io import load_json

    ont_path = Path(ontology_dir) / version
    if not ont_path.exists():
        print(f"[ERROR] Ontology path not found: {ont_path}")
        sys.exit(1)

    concepts  = load_json(ont_path / "concepts.json")
    relations = load_json(ont_path / "relations.json")

    print(f"\n{'='*50}")
    print(f"  Dry-run — Ontology v{version}")
    print(f"{'='*50}")
    print(f"  Concepts  : {len(concepts):,}")
    print(f"  Relations : {len(relations):,}")

    # Class distribution
    from collections import Counter
    cls_counts = Counter(c.get("concept_class", "?") for c in concepts)
    print(f"\n  Class distribution (top 10):")
    for cls, cnt in cls_counts.most_common(10):
        print(f"    {cls:<30} {cnt:>6,}")

    pred_counts = Counter(r.get("predicate", "?") for r in relations)
    print(f"\n  Predicate distribution (top 10):")
    for pred, cnt in pred_counts.most_common(10):
        print(f"    {pred:<30} {cnt:>6,}")

    print(f"\n  No data written (--dry-run mode).")
    print(f"{'='*50}\n")


def main() -> None:
    _load_env()
    args = _parse_args()

    if args.dry_run:
        _dry_run(args.ontology_version, args.ontology_dir)
        return

    from src.graph.neo4j_client import Neo4jClient

    uri      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.environ.get("NEO4J_USER",     "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    print(f"\n{'='*50}")
    print(f"  Neo4j Ontology Ingestion")
    print(f"{'='*50}")
    print(f"  URI      : {uri}")
    print(f"  Version  : {args.ontology_version}")
    print(f"  Dir      : {args.ontology_dir}")
    print(f"  Clear    : {args.clear}")
    print(f"  Batch sz : {args.batch_size}")
    print(f"{'='*50}\n")

    client = Neo4jClient(uri=uri, user=user, password=password)

    if client.is_offline:
        print("[ERROR] Could not connect to Neo4j. Check URI/credentials.")
        print("        Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env")
        sys.exit(1)

    print(f"[OK] Connected to Neo4j at {uri}")

    t_start = time.perf_counter()

    result = client.load_ontology_version(
        version=args.ontology_version,
        ontology_dir=args.ontology_dir,
        clear_existing=args.clear,
    )

    elapsed = round(time.perf_counter() - t_start, 1)

    print(f"\n{'='*50}")
    print(f"  Ingestion complete in {elapsed}s")
    print(f"  Concepts loaded  : {result['n_concepts']:,}")
    print(f"  Relations loaded : {result['n_relations']:,}")
    print(f"{'='*50}")

    # Verify counts
    stats = client.stats()
    print(f"\n  Graph stats after load:")
    print(f"    Total concepts  : {stats['n_concepts']:,}")
    print(f"    Total relations : {stats['n_relations']:,}")
    print(f"    Classes         : {stats['n_classes']}")
    print(f"    Predicates      : {stats['n_predicates']}")

    client.close()
    print("\n[DONE] Neo4j connection closed.")


if __name__ == "__main__":
    main()
