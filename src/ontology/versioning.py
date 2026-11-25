"""
Ontology version control and change management.

Supports incremental updates, version history, conflict detection,
and governance with role-based approval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..utils.logging_utils import get_logger
from ..utils.io import save_json, load_json, ensure_dir

logger = get_logger("ontology.versioning")


@dataclass
class OntologyChange:
    """Records a single change to the ontology."""
    change_id: str
    change_type: str        # "add_concept" | "remove_concept" | "add_relation" | ...
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    reason: str = ""
    originating_source: str = "automated_extractor"  # agent role
    approved_by: Optional[str] = None
    affected_entities: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "originating_source": self.originating_source,
            "approved_by": self.approved_by,
            "affected_entities": self.affected_entities,
            "payload": self.payload,
        }


class OntologyVersionManager:
    """
    Manages ontology versions on disk.

    Directory layout:
        artifacts/ontologies/
            v0.1.0/
                concepts.json
                relations.json
                rules.json
                schema.json
                changelog.json
            v0.2.0/ ...
            current -> v0.1.0   (symlink or file)
    """

    def __init__(self, base_dir: str | Path = "artifacts/ontologies"):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def _version_dir(self, version: str) -> Path:
        return self.base_dir / version

    def save_version(
        self,
        version: str,
        concepts: list[dict],
        relations: list[dict],
        rules: list[dict],
        schema: dict,
        changelog: list[dict] | None = None,
    ) -> Path:
        """Persist an ontology version to disk."""
        vdir = ensure_dir(self._version_dir(version))
        save_json(concepts, vdir / "concepts.json")
        save_json(relations, vdir / "relations.json")
        save_json(rules, vdir / "rules.json")
        save_json(schema, vdir / "schema.json")
        if changelog is not None:
            save_json(changelog, vdir / "changelog.json")

        # Update 'current' pointer
        current_ptr = self.base_dir / "current.json"
        save_json({"current_version": version, "updated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()}, current_ptr)

        logger.info(f"Ontology version {version} saved to {vdir}")
        return vdir

    def load_version(self, version: str) -> dict[str, Any]:
        """Load a stored ontology version. Returns dict with concepts, relations, rules."""
        vdir = self._version_dir(version)
        if not vdir.exists():
            raise FileNotFoundError(f"Ontology version '{version}' not found at {vdir}")
        return {
            "version": version,
            "concepts": load_json(vdir / "concepts.json"),
            "relations": load_json(vdir / "relations.json"),
            "rules": load_json(vdir / "rules.json"),
            "schema": load_json(vdir / "schema.json"),
        }

    def get_current_version(self) -> Optional[str]:
        current_ptr = self.base_dir / "current.json"
        if not current_ptr.exists():
            return None
        data = load_json(current_ptr)
        return data.get("current_version")

    def list_versions(self) -> list[str]:
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def compare_versions(self, v1: str, v2: str) -> dict[str, Any]:
        """Compute a diff summary between two versions."""
        ont1 = self.load_version(v1)
        ont2 = self.load_version(v2)

        concepts1 = {c["name"] for c in ont1["concepts"]}
        concepts2 = {c["name"] for c in ont2["concepts"]}

        rels1 = {(r["subject"], r["predicate"], r["object"]) for r in ont1["relations"]}
        rels2 = {(r["subject"], r["predicate"], r["object"]) for r in ont2["relations"]}

        return {
            "from_version": v1,
            "to_version": v2,
            "concepts": {
                "added": sorted(concepts2 - concepts1),
                "removed": sorted(concepts1 - concepts2),
                "unchanged": len(concepts1 & concepts2),
            },
            "relations": {
                "added": len(rels2 - rels1),
                "removed": len(rels1 - rels2),
                "unchanged": len(rels1 & rels2),
            },
        }

    def add_concepts_incremental(
        self,
        new_concepts: list[dict],
        existing_concepts: list[dict],
        policy: str = "priority",
    ) -> tuple[list[dict], dict]:
        """
        Merge new_concepts into existing_concepts without duplication.

        Deduplication key: concept name (case-insensitive).
        Conflicts (same name, different attributes) are resolved via resolve_conflict().

        Returns
        -------
        merged : list[dict]
            Updated concept list (existing + accepted new).
        stats : dict
            {"added": int, "merged": int, "skipped_duplicate": int, "flagged": int}
        """
        existing_by_name: dict[str, dict] = {
            c["name"].lower(): c for c in existing_concepts if c.get("name")
        }
        stats = {"added": 0, "merged": 0, "skipped_duplicate": 0, "flagged": 0}

        for nc in new_concepts:
            name_lower = nc.get("name", "").lower()
            if not name_lower:
                continue
            if name_lower not in existing_by_name:
                existing_by_name[name_lower] = nc
                stats["added"] += 1
            else:
                existing_c = existing_by_name[name_lower]
                resolved, action = resolve_conflict(
                    existing_c, nc, "duplicate_concept", policy
                )
                existing_by_name[name_lower] = resolved
                if action == "merged":
                    stats["merged"] += 1
                elif action == "flagged":
                    stats["flagged"] += 1
                else:
                    stats["skipped_duplicate"] += 1

        logger.info(
            f"add_concepts_incremental: added={stats['added']} merged={stats['merged']} "
            f"flagged={stats['flagged']} skipped={stats['skipped_duplicate']}"
        )
        return list(existing_by_name.values()), stats

    def detect_conflicts(
        self,
        existing_concepts: list[dict],
        incoming_concepts: list[dict],
    ) -> list[dict]:
        """
        Detect conflicts between existing and incoming concept lists.

        Conflict types detected:
          - duplicate_concept   : same name, different class
          - class_disagreement  : same name, class mismatch
          - confidence_collision: same name, confidence differs > 0.1

        Returns list of conflict dicts with keys:
          name, conflict_type, existing_value, incoming_value
        """
        existing_by_name: dict[str, dict] = {
            c["name"].lower(): c for c in existing_concepts if c.get("name")
        }
        conflicts: list[dict] = []

        for nc in incoming_concepts:
            name_lower = nc.get("name", "").lower()
            if not name_lower or name_lower not in existing_by_name:
                continue
            ec = existing_by_name[name_lower]

            # Class disagreement
            ec_class = ec.get("concept_class", ec.get("class", ""))
            nc_class = nc.get("concept_class", nc.get("class", ""))
            if ec_class and nc_class and ec_class.lower() != nc_class.lower():
                conflicts.append({
                    "name":           nc.get("name"),
                    "conflict_type":  "class_disagreement",
                    "existing_value": ec_class,
                    "incoming_value": nc_class,
                })

            # Confidence collision (> 0.1 gap)
            ec_conf = float(ec.get("confidence", ec.get("provenance", {}).get("confidence", 0.0)))
            nc_conf = float(nc.get("confidence", nc.get("provenance", {}).get("confidence", 0.0)))
            if abs(ec_conf - nc_conf) > 0.1:
                conflicts.append({
                    "name":           nc.get("name"),
                    "conflict_type":  "confidence_collision",
                    "existing_value": ec_conf,
                    "incoming_value": nc_conf,
                })

        logger.info(f"detect_conflicts: found {len(conflicts)} conflicts")
        return conflicts

    def generate_diff_report(
        self,
        v_old: str,
        v_new: str,
    ) -> dict:
        """
        Generate a structured diff JSON between two ontology versions.

        Extends compare_versions() with full lists for added/removed concepts
        and per-class breakdown. Saves report to artifacts/reports/.

        Returns the diff dict.
        """
        from ..utils.io import ensure_dir as _ensure_dir

        ont_old = self.load_version(v_old)
        ont_new = self.load_version(v_new)

        old_concepts = {c["name"]: c for c in ont_old["concepts"] if c.get("name")}
        new_concepts = {c["name"]: c for c in ont_new["concepts"] if c.get("name")}

        added_names   = sorted(set(new_concepts) - set(old_concepts))
        removed_names = sorted(set(old_concepts) - set(new_concepts))
        common_names  = set(old_concepts) & set(new_concepts)

        # Detect modified concepts (same name, changed class or attributes)
        modified = []
        for name in common_names:
            oc = old_concepts[name]
            nc = new_concepts[name]
            oc_class = oc.get("concept_class", oc.get("class", ""))
            nc_class = nc.get("concept_class", nc.get("class", ""))
            if oc_class != nc_class:
                modified.append({
                    "name": name,
                    "field": "concept_class",
                    "old": oc_class,
                    "new": nc_class,
                })

        # Relations diff
        old_rels = {
            (r.get("subject",""), r.get("predicate",""), r.get("object",""))
            for r in ont_old["relations"]
        }
        new_rels = {
            (r.get("subject",""), r.get("predicate",""), r.get("object",""))
            for r in ont_new["relations"]
        }

        # Per-class breakdown of added concepts
        class_breakdown: dict[str, int] = {}
        for name in added_names:
            cls = new_concepts[name].get("concept_class", new_concepts[name].get("class", "Unknown"))
            class_breakdown[cls] = class_breakdown.get(cls, 0) + 1

        diff = {
            "from_version":    v_old,
            "to_version":      v_new,
            "generated_at":    datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "concepts": {
                "total_old":    len(old_concepts),
                "total_new":    len(new_concepts),
                "added_count":  len(added_names),
                "removed_count":len(removed_names),
                "modified_count":len(modified),
                "added":        added_names[:200],   # cap for readability
                "removed":      removed_names[:200],
                "modified":     modified[:50],
                "added_by_class": class_breakdown,
            },
            "relations": {
                "total_old":    len(old_rels),
                "total_new":    len(new_rels),
                "added_count":  len(new_rels - old_rels),
                "removed_count":len(old_rels - new_rels),
            },
        }
        logger.info(
            f"Diff {v_old}→{v_new}: "
            f"+{len(added_names)} concepts, -{len(removed_names)} concepts, "
            f"~{len(modified)} modified"
        )
        return diff

    def append_changelog(self, version: str, change: OntologyChange) -> None:
        """Append a change record to the version's changelog."""
        vdir = self._version_dir(version)
        changelog_path = vdir / "changelog.json"
        if changelog_path.exists():
            existing: list[dict] = load_json(changelog_path)
        else:
            existing = []
        existing.append(change.to_dict())
        save_json(existing, changelog_path)


# ── Conflict resolution ───────────────────────────────────────────────────────

_FUNCTIONAL_PREDICATES = {
    "located_in", "has_price_range", "has_difficulty", "has_duration",
    "has_cuisine_type", "managed_by",
}

_CONFIDENCE_TIE_THRESHOLD = 0.05  # differences smaller than this = "tied"


def resolve_conflict(
    existing: dict,
    incoming: dict,
    conflict_type: str,
    policy: str = "priority",
) -> tuple[dict, str]:
    """
    Resolve a conflict between two ontology elements.

    Implements the priority rules from docs/conflict_resolution_policy.md:
      1. Expert correction > automated extraction
      2. Higher confidence wins (if gap > threshold)
      3. LLM > offline for same triple
      4. Recency as last tie-breaker

    Args:
        existing:      current element in the ontology (concept or relation dict)
        incoming:      new element being merged in
        conflict_type: "duplicate_concept" | "contradictory_relation" |
                       "class_disagreement" | "confidence_collision"
        policy:        "priority"  — rule-based priority (default)
                       "merge"     — combine fields where possible
                       "flag"      — keep existing, add to expert review queue

    Returns:
        (resolved_element, action_taken)
        action_taken: "kept_existing" | "replaced" | "merged" | "flagged"
    """
    if policy == "flag":
        return existing, "flagged"

    # ── Duplicate concept ─────────────────────────────────────────────────────
    if conflict_type == "duplicate_concept":
        exist_conf = float(existing.get("confidence", existing.get("provenance", {}).get("confidence", 0.0)))
        new_conf   = float(incoming.get("confidence", incoming.get("provenance", {}).get("confidence", 0.0)))

        primary, secondary = (incoming, existing) if new_conf > exist_conf else (existing, incoming)

        # Absorb secondary name + aliases into primary's alias list
        all_aliases = list({
            *primary.get("aliases", []),
            *secondary.get("aliases", []),
            secondary.get("name", ""),
        } - {primary.get("name", "")})

        merged = {**primary, "aliases": all_aliases}
        return merged, "merged"

    # ── Contradictory relation ────────────────────────────────────────────────
    if conflict_type == "contradictory_relation":
        predicate = existing.get("predicate", "")

        if predicate not in _FUNCTIONAL_PREDICATES:
            # Non-functional: both values are valid — caller should retain both
            return existing, "kept_existing"

        exist_conf = float(existing.get("confidence", 0.0))
        new_conf   = float(incoming.get("confidence", 0.0))

        # Prefer expert-reviewed element
        if existing.get("source") == "expert_review":
            return existing, "kept_existing"
        if incoming.get("source") == "expert_review":
            return incoming, "replaced"

        # Prefer LLM over offline when confidence is similar
        exist_src = existing.get("provenance", {}).get("extraction_method", "offline")
        new_src   = incoming.get("provenance", {}).get("extraction_method", "offline")

        gap = new_conf - exist_conf
        if abs(gap) > _CONFIDENCE_TIE_THRESHOLD:
            return (incoming, "replaced") if gap > 0 else (existing, "kept_existing")

        # Tied — prefer LLM extraction
        if new_src == "llm" and exist_src != "llm":
            return incoming, "replaced"
        if exist_src == "llm" and new_src != "llm":
            return existing, "kept_existing"

        # True tie — flag for human review
        return existing, "flagged"

    # ── Class disagreement ────────────────────────────────────────────────────
    if conflict_type == "class_disagreement":
        # Conservative: flag all class conflicts for human review.
        # The caller may attempt sub-class resolution before calling this.
        return existing, "flagged"

    # ── Confidence collision (same triple, different confidence value) ─────────
    if conflict_type == "confidence_collision":
        exist_conf = float(existing.get("confidence", 0.0))
        new_conf   = float(incoming.get("confidence", 0.0))
        merged = {**existing, "confidence": round((exist_conf + new_conf) / 2.0, 4)}
        return merged, "merged"

    # ── Unknown type ──────────────────────────────────────────────────────────
    logger.warning(f"resolve_conflict: unknown conflict_type='{conflict_type}' — flagging")
    return existing, "flagged"
