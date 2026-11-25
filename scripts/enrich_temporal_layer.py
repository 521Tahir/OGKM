"""
Temporal layer enrichment — ontology evolution v1.1.0 → v1.2.0.

Adds an explicit temporal dimension (comments.txt / Temporal Ontology Research):
the static v1.1.0 core is preserved unchanged (so all Phase-10·A structural
metrics still hold and SHACL still conforms), and a modular **temporal layer**
(``temporal_facts.json``) is attached, carrying time-scoped facts (blooms,
festivals, weather windows, harvests) with month/season validity.

The layer is built from:
  1. curated, high-accuracy Jeju facts (``data/temporal/jeju_temporal_facts.json``);
  2. facts *derived* from existing seasonal relations in the core ontology
     (``has_seasonality`` / ``avoid_during`` whose object names a known season),
     demonstrating that the temporal layer also lifts pre-existing knowledge.

Usage
-----
  python scripts/enrich_temporal_layer.py --source 1.1.0 --target 1.2.0
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.temporal import TemporalAnnotation, TemporalFact, season_to_months
from src.ontology.shacl_validator import SHACLValidator
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("enrich_temporal")

ONT = PROJECT_ROOT / "artifacts" / "ontologies"
CURATED = PROJECT_ROOT / "data" / "temporal" / "jeju_temporal_facts.json"
CORE_FILES = ["concepts.json", "relations.json", "hierarchy.json", "rules.json", "schema.json"]

# Map free-text season phrases found in the core ontology → month numbers
_SEASON_PHRASE_MONTHS = {
    "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11], "fall": [9, 10, 11],
    "winter": [12, 1, 2], "rainy season": [6, 7], "monsoon": [6, 7], "wet season": [6, 7],
    "dry season": [11, 12, 1, 2], "typhoon season": [7, 8, 9], "peak season": [7, 8],
}


def load_curated() -> list[TemporalFact]:
    data = load_json(CURATED)
    facts = []
    for d in data.get("facts", []):
        f = TemporalFact.from_dict(d)
        # backfill months from season if needed
        if not f.temporal.months and f.temporal.season:
            f.temporal.months = season_to_months(f.temporal.season)
        facts.append(f)
    return facts


def derive_from_core(relations: list[dict]) -> list[TemporalFact]:
    """Lift existing seasonal relations into temporal facts where a season is named."""
    derived: list[TemporalFact] = []
    seen: set[tuple] = set()
    for r in relations:
        pred = r.get("predicate", "")
        if pred not in ("has_seasonality", "avoid_during"):
            continue
        obj = str(r.get("object", "")).strip().lower()
        months = None
        for phrase, ms in _SEASON_PHRASE_MONTHS.items():
            if phrase in obj:
                months = ms
                break
        if not months:
            continue
        subj = r.get("subject", "")
        key = (subj.lower(), pred, obj)
        if key in seen:
            continue
        seen.add(key)
        note = f"{r.get('object')} (≈ {months})"
        derived.append(TemporalFact(
            subject=subj, predicate=pred, object=r.get("object", ""),
            temporal=TemporalAnnotation(months=months, recurrence="annual", note=r.get("object", "")),
            category="weather" if pred == "avoid_during" else "season",
            confidence=0.6, source="derived",
        ))
    return derived


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal layer enrichment")
    ap.add_argument("--source", default="1.1.0")
    ap.add_argument("--target", default="1.2.0")
    ap.add_argument("--max-derived", type=int, default=300,
                    help="cap on derived facts to keep the layer focused")
    args = ap.parse_args()

    src, dst = ONT / args.source, ONT / args.target
    if not src.exists():
        logger.error("Source version %s not found at %s", args.source, src); sys.exit(1)

    # ── Load core + build temporal layer ───────────────────────────────────────
    relations = load_json(src / "relations.json")
    curated = load_curated()
    derived = derive_from_core(relations)[: args.max_derived]
    all_facts = curated + derived
    logger.info("Temporal facts: %d curated + %d derived = %d total",
                len(curated), len(derived), len(all_facts))

    # ── Save v1.2.0: copy core unchanged + add temporal_facts.json ─────────────
    ensure_dir(dst)
    for fname in CORE_FILES:
        if (src / fname).exists():
            save_json(load_json(src / fname), dst / fname)

    # mark schema with the temporal layer + predicates (additive, documentation)
    schema = load_json(dst / "schema.json")
    schema["version"] = args.target
    schema["temporal_layer"] = {
        "enabled": True,
        "file": "temporal_facts.json",
        "predicates": sorted({f.predicate for f in all_facts}),
        "categories": sorted({f.category for f in all_facts}),
        "month_encoding": "1-12 (Northern hemisphere)",
    }
    save_json(schema, dst / "schema.json")

    temporal_payload = {
        "meta": {
            "version": args.target,
            "source_version": args.source,
            "date": datetime.now(timezone.utc).isoformat(),
            "n_curated": len(curated),
            "n_derived": len(derived),
            "n_total": len(all_facts),
        },
        "facts": [f.to_dict() for f in all_facts],
    }
    save_json(temporal_payload, dst / "temporal_facts.json")

    # changelog
    changelog = load_json(src / "changelog.json") if (src / "changelog.json").exists() else []
    if isinstance(changelog, dict):
        changelog = [changelog]
    changelog.append({
        "version": args.target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "change_type": "temporal_layer",
        "reason": "Add explicit temporal dimension (time-scoped facts: blooms, festivals, "
                  "weather windows, harvests) so the ontology represents time-varying knowledge.",
        "stats": {"curated_facts": len(curated), "derived_facts": len(derived)},
    })
    save_json(changelog, dst / "changelog.json")

    # ── Validate the (unchanged) core still conforms ───────────────────────────
    concepts = load_json(dst / "concepts.json")
    rep = SHACLValidator().validate_ontology(concepts, load_json(dst / "relations.json"))

    # ── Coverage stats by category ─────────────────────────────────────────────
    from collections import Counter
    by_cat = Counter(f.category for f in all_facts)

    print("\n" + "=" * 70)
    print(f"TEMPORAL LAYER ENRICHMENT  v{args.source} → v{args.target}")
    print("=" * 70)
    print(f"  Temporal facts: {len(all_facts)}  ({len(curated)} curated + {len(derived)} derived)")
    print(f"  By category   : {dict(by_cat)}")
    print(f"  Core SHACL    : conforms={rep.conforms}, violations={len(rep.violations)}")
    print(f"  Saved         → {dst.relative_to(PROJECT_ROOT)}/  (+ temporal_facts.json)")
    print("  Sample facts:")
    for f in curated[:5]:
        print(f"    • {f.to_sentence()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
