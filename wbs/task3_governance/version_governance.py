"""
WBS Task 3.1.2 - Ontology versioning, change detection, diff, rollback,
governance workflow and provenance-tracked incremental evolution.

Implemented against the real version series already on disk
(artifacts/ontologies/{0.1.0, 1.0.0, 1.1.0, 1.2.0, 1.3.0, 1.3.0-norm, 3.0.0}).

What this provides
  * content-addressed snapshots       sha256 over the canonicalised triple set, so
                                      two versions are identical iff their hashes are
  * semantic diff                     added / removed / modified concepts and
                                      relations between any two versions
  * change classification             additive | corrective | breaking, derived from
                                      what the diff actually contains rather than
                                      from the version number
  * rollback                          restore any version as the current pointer,
                                      recording the rollback itself in the audit log
  * governance workflow               proposals carry a state machine
                                      (proposed -> validated -> approved -> merged
                                       | rejected), and a proposal cannot reach
                                      `approved` while its validation has errors
  * append-only audit trail           every operation is logged with actor, time,
                                      before/after hash and a reason string
  * role-based access control         every entry point below asks
                                      `access_control.require()` first, so a
                                      principal without the permission gets an
                                      AccessDenied rather than a changed
                                      ontology (proposal §8.7; see
                                      `access_control.py`)

  python -m wbs.task3_governance.version_governance --report
  python -m wbs.task3_governance.version_governance --diff 1.2.0 1.3.0
  python -m wbs.task3_governance.version_governance --all
  python -m wbs.task3_governance.version_governance --demo-evolution \
      --target-version 1.7.0      # three principals; refuses to overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO_DIR = ROOT / "artifacts" / "ontologies"
GOV_DIR = ROOT / "wbs" / "deliverables" / "task3_governance"
AUDIT = GOV_DIR / "audit_trail.jsonl"
PROPOSALS = GOV_DIR / "proposals.json"
POINTER = GOV_DIR / "current_version.json"

from wbs.task3_governance import access_control as ac

STATES = ("proposed", "validated", "approved", "merged", "rejected")
TRANSITIONS = {
    "proposed": {"validated", "rejected"},
    "validated": {"approved", "rejected"},
    "approved": {"merged", "rejected"},
    "merged": set(),
    "rejected": set(),
}

# The permission a principal needs to move a proposal INTO each state. `merged`
# is reached only through merge_proposal(), which checks the namespace-specific
# merge permission before it writes anything, so the transition itself adds no
# second check.
NEEDS = {
    "validated": "proposal.validate",
    "approved": "proposal.approve",
    "rejected": "proposal.reject",
    "merged": None,
}


def key(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def actor() -> str:
    return os.environ.get("WBS_ACTOR") or os.environ.get("USER") or "unknown"


def audit(op: str, **fields) -> dict:
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "actor": actor(), "op": op, **fields}
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ── snapshots ─────────────────────────────────────────────────────────────
def versions() -> list[str]:
    if not ONTO_DIR.exists():
        return []
    return sorted([d.name for d in ONTO_DIR.iterdir()
                   if d.is_dir() and (d / "concepts.json").exists()],
                  key=lambda v: [int(x) if x.isdigit() else x
                                 for x in re.split(r"[.\-]", v)])


def load(version: str):
    d = ONTO_DIR / version
    rd = lambda f: json.loads((d / f).read_text(encoding="utf-8")) if (d / f).exists() else []
    return rd("concepts.json"), rd("relations.json"), rd("hierarchy.json")


def triple_set(relations) -> set[tuple]:
    return {(key(r.get("subject")), str(r.get("predicate")), key(r.get("object")))
            for r in relations if isinstance(r, dict)
            and r.get("subject") and r.get("predicate") and r.get("object")}


def concept_map(concepts) -> dict[str, dict]:
    out = {}
    for c in concepts:
        if isinstance(c, dict) and c.get("name"):
            out[key(c["name"])] = c
    return out


def content_hash(concepts, relations) -> str:
    h = hashlib.sha256()
    for k in sorted(concept_map(concepts)):
        h.update(("C|" + k + "|" + str(concept_map(concepts)[k].get("concept_class")) + "\n").encode())
    for t in sorted(triple_set(relations)):
        h.update(("R|" + "|".join(t) + "\n").encode())
    return h.hexdigest()


# ── diff ──────────────────────────────────────────────────────────────────
def diff(v1: str, v2: str) -> dict:
    c1, r1, _ = load(v1)
    c2, r2, _ = load(v2)
    m1, m2 = concept_map(c1), concept_map(c2)
    t1, t2 = triple_set(r1), triple_set(r2)

    added_c = sorted(set(m2) - set(m1))
    removed_c = sorted(set(m1) - set(m2))
    modified_c = []
    for k in sorted(set(m1) & set(m2)):
        a, b = m1[k], m2[k]
        changes = {}
        if a.get("concept_class") != b.get("concept_class"):
            changes["concept_class"] = [a.get("concept_class"), b.get("concept_class")]
        if sorted(a.get("aliases") or []) != sorted(b.get("aliases") or []):
            changes["aliases"] = [a.get("aliases"), b.get("aliases")]
        if (a.get("attributes") or {}) != (b.get("attributes") or {}):
            changes["attributes"] = [a.get("attributes"), b.get("attributes")]
        if changes:
            modified_c.append({"concept": b.get("name"), "changes": changes})

    added_r, removed_r = sorted(t2 - t1), sorted(t1 - t2)
    # classification: removals of concepts or relations can break downstream queries
    if removed_c or len(removed_r) > 0.05 * max(len(t1), 1):
        kind = "breaking"
    elif modified_c or removed_r:
        kind = "corrective"
    else:
        kind = "additive"

    return {
        "from": v1, "to": v2, "change_type": kind,
        "hash_from": content_hash(c1, r1), "hash_to": content_hash(c2, r2),
        "concepts": {"before": len(m1), "after": len(m2),
                     "added": len(added_c), "removed": len(removed_c),
                     "modified": len(modified_c)},
        "relations": {"before": len(t1), "after": len(t2),
                      "added": len(added_r), "removed": len(removed_r)},
        "predicate_delta": dict(
            (Counter(p for _, p, _ in t2) - Counter(p for _, p, _ in t1)).most_common(10)),
        "samples": {
            "added_concepts": [m2[k].get("name") for k in added_c[:15]],
            "removed_concepts": [m1[k].get("name") for k in removed_c[:15]],
            "modified_concepts": modified_c[:10],
            "added_relations": [" ".join(t) for t in added_r[:15]],
            "removed_relations": [" ".join(t) for t in removed_r[:15]],
        },
    }


# ── governance workflow ───────────────────────────────────────────────────
def load_proposals() -> list[dict]:
    if PROPOSALS.exists():
        return json.loads(PROPOSALS.read_text(encoding="utf-8"))
    return []


def save_proposals(ps: list[dict]) -> None:
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS.write_text(json.dumps(ps, indent=2, ensure_ascii=False), encoding="utf-8")


def propose(title: str, additions: list[dict], rationale: str) -> dict:
    ac.require("proposal.create", title=title, n_additions=len(additions or []))
    ps = load_proposals()
    pid = f"P{len(ps)+1:04d}"
    p = {"id": pid, "title": title, "state": "proposed", "rationale": rationale,
         "additions": additions, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "actor": actor(), "history": [], "validation": None}
    ps.append(p); save_proposals(ps)
    audit("proposal.create", proposal=pid, title=title, n_additions=len(additions))
    return p


def transition(pid: str, to_state: str, reason: str = "") -> dict:
    ps = load_proposals()
    p = next((x for x in ps if x["id"] == pid), None)
    if not p:
        raise SystemExit(f"no such proposal: {pid}")
    need = NEEDS.get(to_state)
    if need:
        ac.require(need, proposal=pid, **{"from": p["state"], "to": to_state})
    if to_state == "approved":
        ac.require_distinct_approver(p)
    if to_state not in TRANSITIONS[p["state"]]:
        raise SystemExit(f"illegal transition {p['state']} -> {to_state} for {pid}")
    if to_state == "approved":
        v = p.get("validation") or {}
        if v.get("errors", 1) != 0:
            raise SystemExit(
                f"{pid} cannot be approved: validation reports {v.get('errors')} error(s). "
                "Run --validate-proposal first and clear the errors.")
    prev = p["state"]; p["state"] = to_state
    p["history"].append({"from": prev, "to": to_state, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "actor": actor(), "reason": reason})
    save_proposals(ps)
    audit("proposal.transition", proposal=pid, **{"from": prev, "to": to_state, "reason": reason})
    return p


def validate_proposal(pid: str, base_version: str) -> dict:
    """Run the Task 2.2 structural+logical layers over base ∪ proposal."""
    from wbs.task2_formalization.validate_ontology import l1_structural, l2_logical
    ac.require("proposal.validate", proposal=pid, base=base_version)
    ps = load_proposals()
    p = next((x for x in ps if x["id"] == pid), None)
    if not p:
        raise SystemExit(f"no such proposal: {pid}")
    c, r, h = load(base_version)
    new_c = list(c) + [a for a in p["additions"] if a.get("kind") == "concept"]
    new_r = list(r) + [a for a in p["additions"] if a.get("kind") == "relation"]
    findings = l1_structural(new_c, new_r) + l2_logical(new_c, new_r, h)
    sev = Counter(f["severity"] for f in findings)
    p["validation"] = {"base": base_version, "errors": sev.get("error", 0),
                       "warnings": sev.get("warning", 0),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_proposals(ps)
    audit("proposal.validate", proposal=pid, errors=sev.get("error", 0),
          warnings=sev.get("warning", 0))
    return p["validation"]


def merge_proposal(pid: str, base_version: str, new_version: str) -> dict:
    """Apply an approved proposal, writing a NEW version with provenance stamped."""
    ps = load_proposals()
    p = next((x for x in ps if x["id"] == pid), None)
    if not p or p["state"] != "approved":
        raise SystemExit(f"{pid} is not approved (state={p['state'] if p else 'missing'})")
    # writing a production version is a different permission from writing a
    # staging one, and overriding a protected element is a third
    ac.require(ac.merge_permission(new_version), proposal=pid,
               target=new_version, namespace=ac.namespace_of(new_version))
    ac.guard_protected(p["additions"], base_version)
    c, r, h = load(base_version)
    stamp = {"source_record_id": -1, "extraction_prompt_version": f"governance/{pid}",
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "confidence": 1.0,
             "validator_status": "pass", "approved_by": actor(),
             "notes": p["rationale"][:200]}
    nc = list(c); nr = list(r)
    for a in p["additions"]:
        rec = {k: v for k, v in a.items() if k != "kind"}
        rec["provenance"] = stamp
        (nc if a.get("kind") == "concept" else nr).append(rec)
    d = ONTO_DIR / new_version
    d.mkdir(parents=True, exist_ok=True)
    (d / "concepts.json").write_text(json.dumps(nc, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "relations.json").write_text(json.dumps(nr, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "hierarchy.json").write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")
    before, after = content_hash(c, r), content_hash(nc, nr)
    transition(pid, "merged", f"merged into {new_version}")
    audit("version.create", version=new_version, base=base_version,
          proposal=pid, hash_before=before, hash_after=after)
    set_current(new_version, f"merge of {pid}")
    return {"version": new_version, "concepts": len(nc), "relations": len(nr),
            "hash_before": before, "hash_after": after}


def set_current(version: str, reason: str) -> None:
    ac.require("pointer.set", version=version, reason=reason)
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    prev = None
    if POINTER.exists():
        prev = json.loads(POINTER.read_text(encoding="utf-8")).get("version")
    POINTER.write_text(json.dumps(
        {"version": version, "set_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "actor": actor(), "reason": reason, "previous": prev}, indent=2), encoding="utf-8")
    audit("pointer.set", version=version, previous=prev, reason=reason)


def rollback(to_version: str, reason: str) -> dict:
    ac.require("version.rollback", to=to_version, reason=reason)
    if to_version not in versions():
        raise SystemExit(f"unknown version {to_version}")
    cur = json.loads(POINTER.read_text(encoding="utf-8"))["version"] if POINTER.exists() else None
    d = diff(to_version, cur) if cur and cur != to_version else None
    set_current(to_version, f"ROLLBACK: {reason}")
    audit("version.rollback", to=to_version, **{"from": cur}, reason=reason,
          reverted_relations=(d or {}).get("relations", {}).get("added"))
    return {"rolled_back_to": to_version, "from": cur, "reverted": d}


# ── reporting ─────────────────────────────────────────────────────────────
def _report_version() -> str:
    """The version the access-control summary is computed over: the published
    pointer when there is one, else the newest on disk."""
    if POINTER.exists():
        v = json.loads(POINTER.read_text(encoding="utf-8")).get("version")
        if v and (ONTO_DIR / v).exists():
            return v
    vs = versions()
    return vs[-1] if vs else "1.4.0"


def build_report() -> str:
    vs = versions()
    rows = []
    for v in vs:
        c, r, h = load(v)
        rows.append((v, len(c), len(triple_set(r)), len(h), content_hash(c, r)[:12]))
    md = ["# Ontology Version Control, Diff and Governance Report", "",
          "## Version series", "",
          "| Version | Concepts | Relations | Hierarchy | Content hash |", "|---|---|---|---|---|"]
    for v, nc, nr, nh, hsh in rows:
        md.append(f"| {v} | {nc:,} | {nr:,} | {nh:,} | `{hsh}` |")
    md += ["", "Content hash is sha256 over the canonicalised concept and triple sets, so two "
               "versions are byte-independent but content-identical iff the hashes match.", ""]
    md += ["## Consecutive diffs", ""]
    diffs = []
    for a, b in zip(vs, vs[1:]):
        d = diff(a, b); diffs.append(d)
        md += [f"### {a} → {b}  ({d['change_type']})", "",
               f"- concepts: {d['concepts']['before']:,} → {d['concepts']['after']:,} "
               f"(+{d['concepts']['added']:,} / −{d['concepts']['removed']:,} / "
               f"~{d['concepts']['modified']:,})",
               f"- relations: {d['relations']['before']:,} → {d['relations']['after']:,} "
               f"(+{d['relations']['added']:,} / −{d['relations']['removed']:,})"]
        if d["samples"]["added_concepts"]:
            md.append(f"- sample additions: {', '.join(d['samples']['added_concepts'][:6])}")
        if d["samples"]["removed_concepts"]:
            md.append(f"- sample removals: {', '.join(d['samples']['removed_concepts'][:6])}")
        md.append("")
    (GOV_DIR / "diff_logs.json").write_text(json.dumps(diffs, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    md += ["## Governance workflow", "",
           "States: `" + "` → `".join(STATES[:4]) + "`, with `rejected` reachable from any "
           "pre-merge state. A proposal **cannot** reach `approved` while its validation "
           "reports errors — the transition raises instead.", ""]
    md += ["Who may make each of those moves is role-based and enforced, not "
           "advisory: `access_control.require()` runs before every operation in "
           "this module and raises `AccessDenied` instead of writing. "
           f"{len(ac.load_principals())} principals over "
           f"{len(ac.ROLES)} roles, "
           f"{len(ac.derive_protected(_report_version())['items']):,} protected "
           "elements, and the two permissions that publish — "
           "`proposal.merge.production` and `pointer.set` — held by `admin` "
           "alone. See `access_control.md`.", ""]
    ps = load_proposals()
    if ps:
        md += ["| Proposal | Title | State | Errors | Warnings |", "|---|---|---|---|---|"]
        for p in ps:
            v = p.get("validation") or {}
            md.append(f"| {p['id']} | {p['title']} | **{p['state']}** | "
                      f"{v.get('errors','—')} | {v.get('warnings','—')} |")
        md.append("")
    if AUDIT.exists():
        lines = [json.loads(l) for l in open(AUDIT, encoding="utf-8") if l.strip()]
        md += ["## Audit trail", "",
               f"{len(lines):,} append-only records in `audit_trail.jsonl`.", "",
               "| Time | Actor | Operation | Detail |", "|---|---|---|---|"]
        for r in lines[-25:]:
            det = ", ".join(f"{k}={v}" for k, v in r.items()
                            if k not in ("ts", "actor", "op"))[:110]
            md.append(f"| {r['ts']} | {r['actor']} | `{r['op']}` | {det} |")
        md.append("")
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    (GOV_DIR / "version_governance_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--diff", nargs=2, metavar=("V1", "V2"))
    ap.add_argument("--demo-evolution", action="store_true",
                    help="exercise the full governance path on a real proposal")
    ap.add_argument("--target-version", default="1.4.0",
                    help="version --demo-evolution would write (it refuses to "
                         "overwrite one that exists)")
    ap.add_argument("--rollback", metavar="VERSION")
    ap.add_argument("--all", action="store_true",
                    help="regenerate every Task 3.1.2 deliverable: version "
                         "series, diffs, governance report, and the §8.7 "
                         "access-control policy, registry and enforcement demo")
    a = ap.parse_args()

    if a.all:
        d = ac.build_report(_report_version())
        print(f"access control: {len(d['roles'])} roles, "
              f"{len(d['principals'])} principals, "
              f"{d['protected']['total']:,} protected elements, "
              f"self-check {'passed' if d['self_check']['passed'] else 'FAILED'}")
        dm = ac.demo()
        print(f"enforcement demo: {dm['allowed']} allowed, {dm['denied']} denied")

    if a.diff:
        d = diff(*a.diff)
        print(json.dumps(d, indent=2, ensure_ascii=False)[:3000])
        return
    if a.rollback:
        print(json.dumps(rollback(a.rollback, "manual rollback"), indent=2)[:1200]); return

    if a.demo_evolution:
        if (ONTO_DIR / a.target_version).exists():
            raise SystemExit(
                f"v{a.target_version} already exists; this path writes a new "
                f"version and will not overwrite one. The run that produced "
                f"v1.4.0 is recorded in proposals.json (P0001) and "
                f"audit_trail.jsonl. Pass --target-version with an unused "
                f"version to exercise the path again.")
        # Separation of duties is enforced as of 2026-10-23, so the demo cannot
        # be one actor start to finish: the pipeline proposes, a reviewer
        # approves what it did not author, and an operator merges and
        # publishes. P0001 on record predates the rule and does not satisfy it.
        def as_(who):
            os.environ["WBS_ACTOR"] = who

        # A real, defensible addition: three UNESCO designations that the gazetteer
        # records but the ontology does not assert as relations.
        gaz = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                         .read_text(encoding="utf-8"))
        unesco = [x["name"] for x in gaz["locations"]
                  if isinstance(x, dict) and "unesco" in (x.get("note", "") + x["name"]).lower()]
        adds = [{"kind": "concept", "name": "UNESCO World Heritage",
                 "concept_class": "Designation", "aliases": ["UNESCO"], "attributes": {}}]
        adds += [{"kind": "relation", "subject": n, "predicate": "has_designation",
                  "object": "UNESCO World Heritage"} for n in unesco]
        as_("nlp-lab-pipeline")
        p = propose("Assert UNESCO designations recorded in the gazetteer", adds,
                    "The gazetteer notes UNESCO status for three sites; the ontology has no "
                    "has_designation relation for them. Adding them makes the designation "
                    "queryable and traceable to the gazetteer.")
        print(f"created {p['id']} with {len(adds)} additions "
              f"({len(unesco)} UNESCO sites: {', '.join(unesco)})")
        v = validate_proposal(p["id"], "1.3.0-norm")
        print(f"validation: {v['errors']} errors, {v['warnings']} warnings")
        transition(p["id"], "validated", "passed structural + logical validation")
        as_("domain-expert-1")
        transition(p["id"], "approved", "reviewed against the gazetteer source")
        as_("ontology-admin")
        res = merge_proposal(p["id"], "1.3.0-norm", a.target_version)
        print(f"merged -> v{a.target_version}: {res['concepts']:,} concepts, "
              f"{res['relations']:,} relations")
        print(f"  hash {res['hash_before'][:12]} -> {res['hash_after'][:12]}")
        d = diff("1.3.0-norm", a.target_version)
        print(f"  diff classified as: {d['change_type']}")

    print(build_report()[:1800])
    print(f"\nwrote {(GOV_DIR / 'version_governance_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
