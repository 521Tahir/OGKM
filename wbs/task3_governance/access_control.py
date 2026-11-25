"""
Proposal §8.7 - knowledge governance with role-based access control.

§8.7 asks for governance that "regulates who can modify knowledge and under what
conditions", distinguishing automated processes, domain experts and system
administrators, and names three properties it has to deliver:

  * critical knowledge elements are protected from unauthorized modification
  * experimental updates are isolated from production ontologies
  * accountability is maintained for all changes

This module implements all three as *enforcement*, not documentation. Every
governance entry point in `version_governance.py` calls `require()` first, and a
principal without the permission gets an `AccessDenied` instead of a changed
ontology. Both outcomes - allow and deny - are appended to the same append-only
audit trail the rest of Task 3.1.2 writes to, so the record of who was refused
is as durable as the record of who succeeded.

Three design decisions worth stating, because each one is a place where an RBAC
layer is usually vacuous:

  1. Unregistered principals get `reader`, not admin. Least privilege means an
     actor the registry has never heard of can read the ontology and nothing
     else. `principals.json` is the registry; a deployment maps it onto real
     accounts.
  2. The protected set is *derived from the ontology*, not listed by hand. An
     element is protected if a human approved it (§8.4: expert interventions
     "are protected against accidental removal or degradation in subsequent
     update cycles") or if it carries a safety constraint - the predicates a
     safety answer in this project actually rests on.
  3. Separation of duties is enforced: the principal who created a proposal
     cannot approve it. That rule is not free, and §"what this found" in the
     generated report says what it costs on this host.

  python -m wbs.task3_governance.access_control --report
  python -m wbs.task3_governance.access_control --demo
  python -m wbs.task3_governance.access_control --whoami
"""
from __future__ import annotations

import argparse
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
PRINCIPALS = GOV_DIR / "principals.json"
PROTECTED = GOV_DIR / "protected_elements.json"
MANUAL = GOV_DIR / "protected_manual.json"
REPORT_MD = GOV_DIR / "access_control.md"
REPORT_JSON = GOV_DIR / "access_control.json"


class AccessDenied(PermissionError):
    """Raised when a principal lacks the permission a governance call needs."""


# ── the permission vocabulary ─────────────────────────────────────────────
# One permission per governance operation that can change knowledge or change
# who can change knowledge. Reading is a permission too, so that a deployment
# can withhold it; on this host every role has it.
PERMISSIONS = (
    "ontology.read",
    "proposal.create",
    "proposal.validate",
    "proposal.approve",
    "proposal.reject",
    "proposal.merge.staging",
    "proposal.merge.production",
    "pointer.set",
    "version.rollback",
    "element.protect",
    "element.unprotect",
    "element.modify.protected",
    "principal.manage",
)

# §8.7's three actor kinds, plus `reader` as the floor for anything
# unregistered. `inherits` is transitive and checked for cycles below.
ROLES: dict[str, dict] = {
    "reader": {
        "describes": "read-only consumer of the published ontology",
        "inherits": [],
        "grants": ["ontology.read"],
    },
    "automation": {
        "describes": "automated process (extraction pipeline, scheduled job)",
        "inherits": ["reader"],
        # may propose and validate, and may write a staging version, but can
        # neither approve its own work nor touch what production serves
        "grants": ["proposal.create", "proposal.validate",
                   "proposal.merge.staging"],
    },
    "expert": {
        "describes": "domain expert reviewing and approving knowledge changes",
        "inherits": ["automation"],
        "grants": ["proposal.approve", "proposal.reject", "element.protect"],
    },
    "admin": {
        "describes": "system administrator operating the knowledge base",
        "inherits": ["expert"],
        "grants": ["proposal.merge.production", "pointer.set",
                   "version.rollback", "element.unprotect",
                   "element.modify.protected", "principal.manage"],
    },
}

DEFAULT_ROLE = "reader"

# The principals this host knows about. Written on first use so the registry is
# a file a reviewer can read and a deployment can replace.
SEED_PRINCIPALS = {
    "tahirahmad": {"role": "admin",
                   "note": "project owner; the account every published run was made under"},
    "ontology-admin": {"role": "admin",
                       "note": "operator identity used for production merges and rollbacks"},
    "domain-expert-1": {"role": "expert",
                        "note": "reviewer identity for the §11.4.2 expert instrument"},
    "nlp-lab-pipeline": {"role": "automation",
                         "note": "the extraction pipeline; proposes, never approves"},
    "guest": {"role": "reader", "note": "read-only"},
}

# Predicates whose triples carry a safety constraint. These are the facts an
# answer's safety claim rests on in this project: a risk, a month to avoid, a
# requirement, a restriction. Removing or overriding one silently degrades an
# answer that looked fine.
SAFETY_PREDICATES = ("has_risk", "avoid_during", "requires", "restricted_in")

# Separation of duties: a proposal's creator may not approve it.
REQUIRE_DISTINCT_APPROVER = True


# ── principals ────────────────────────────────────────────────────────────
def actor() -> str:
    return os.environ.get("WBS_ACTOR") or os.environ.get("USER") or "unknown"


def load_principals() -> dict:
    if PRINCIPALS.exists():
        return json.loads(PRINCIPALS.read_text(encoding="utf-8"))
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    PRINCIPALS.write_text(json.dumps(SEED_PRINCIPALS, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return dict(SEED_PRINCIPALS)


def role_of(principal: str | None = None) -> str:
    p = principal or actor()
    r = (load_principals().get(p) or {}).get("role")
    return r if r in ROLES else DEFAULT_ROLE


def permissions(role: str) -> set[str]:
    """Resolve a role's grants transitively through `inherits`."""
    if role not in ROLES:
        return set()
    out: set[str] = set()
    seen: set[str] = set()
    stack = [role]
    while stack:
        r = stack.pop()
        if r in seen:
            continue
        seen.add(r)
        out |= set(ROLES[r]["grants"])
        stack += [x for x in ROLES[r]["inherits"] if x in ROLES]
    return out


def allows(principal: str, permission: str) -> bool:
    return permission in permissions(role_of(principal))


# ── accountability ────────────────────────────────────────────────────────
def _audit(op: str, **fields) -> dict:
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "actor": actor(), "op": op, **fields}
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def require(permission: str, principal: str | None = None, **ctx) -> dict:
    """Allow or refuse `permission`, record the decision either way."""
    if permission not in PERMISSIONS:
        raise ValueError(f"unknown permission: {permission}")
    p = principal or actor()
    role = role_of(p)
    ok = permission in permissions(role)
    _audit("access.allow" if ok else "access.deny",
           principal=p, role=role, permission=permission, **ctx)
    if not ok:
        raise AccessDenied(
            f"{p} (role={role}) lacks `{permission}`. "
            f"Roles holding it: {', '.join(sorted(holders(permission))) or 'none'}.")
    return {"principal": p, "role": role, "permission": permission, "allowed": True}


def holders(permission: str) -> set[str]:
    return {r for r in ROLES if permission in permissions(r)}


# ── separation of duties ──────────────────────────────────────────────────
def require_distinct_approver(proposal: dict, principal: str | None = None) -> None:
    if not REQUIRE_DISTINCT_APPROVER:
        return
    p = principal or actor()
    creator = proposal.get("actor")
    if creator and creator == p:
        _audit("access.deny", principal=p, role=role_of(p),
               permission="proposal.approve", proposal=proposal.get("id"),
               reason="separation_of_duties")
        raise AccessDenied(
            f"{p} created {proposal.get('id')} and may not also approve it "
            "(separation of duties). A second principal holding `proposal.approve` "
            "must review it.")


# ── isolation: staging vs production ──────────────────────────────────────
PRODUCTION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def namespace_of(version: str) -> str:
    """A bare semantic version is production; anything else is experimental.

    `1.7.0` is production. `1.7.0-exp1`, `1.3.0-norm`, `staging-policy` are
    staging, and writing one needs only the staging permission - which is how
    an automated process can try an edit without being able to change what the
    deployed system serves.
    """
    return "production" if PRODUCTION_RE.match(str(version or "")) else "staging"


def merge_permission(version: str) -> str:
    return ("proposal.merge.production" if namespace_of(version) == "production"
            else "proposal.merge.staging")


# ── protected elements, derived from the ontology ──────────────────────────
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def _human_approved(prov) -> str | None:
    """Return the reason this element counts as human-approved, or None."""
    if not isinstance(prov, dict):
        return None
    if prov.get("approved_by"):
        return f"approved by {prov['approved_by']}"
    v = str(prov.get("extraction_prompt_version") or "")
    if v.startswith("governance/"):
        return f"merged through {v}"
    return None


def load_manual() -> dict:
    if MANUAL.exists():
        try:
            return json.loads(MANUAL.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def element_key(kind: str, spec: str) -> tuple[str, str]:
    """Canonical registry key and display label for one element.

    A relation is given as `subject|predicate|object`, not as a sentence: the
    predicate of "Jeju Island near Udo Island" cannot be recovered from the
    spaces, and guessing it is how a protection ends up stored under a key no
    lookup will ever match.
    """
    if kind == "concept":
        return _norm(spec), spec.strip()
    parts = [x.strip() for x in str(spec).split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "a relation is specified as 'subject|predicate|object', e.g. "
            "'Jeju Island|near|Udo Island'")
    sub, pred, obj = parts
    return f"{_norm(sub)}|{pred}|{_norm(obj)}", f"{sub} {pred} {obj}"


def protect(kind: str, spec: str, reason: str, principal: str | None = None) -> dict:
    """Mark one element critical by hand, on top of the derived rules.

    The derived rules cover whole predicate families; this covers the single
    element a reviewer decides matters. Experts may protect, only admins may
    unprotect - the asymmetry is deliberate, since the failure to avoid is
    quiet removal of protection. `spec` is a concept name, or a relation as
    `subject|predicate|object`.
    """
    if kind not in ("concept", "relation"):
        raise ValueError("kind must be concept or relation")
    key, label = element_key(kind, spec)
    require("element.protect", principal, kind=kind, label=label)
    m = load_manual()
    m[key] = {"kind": kind, "key": key, "label": label, "reason": reason,
              "protected_by": principal or actor(),
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    _audit("protected.add", key=key, kind=kind, label=label, reason=reason)
    return m[key]


def unprotect(key: str, reason: str, principal: str | None = None) -> bool:
    require("element.unprotect", principal, key=key, reason=reason)
    m = load_manual()
    if key not in m:
        return False
    m.pop(key)
    MANUAL.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    _audit("protected.remove", key=key, reason=reason)
    return True


def derive_protected(version: str) -> dict:
    """The protected registry for one ontology version.

    Nothing here is a hand-written list: an element is protected because of what
    its own provenance or predicate says, so a later version that adds safety
    facts protects them without this file being edited.
    """
    d = ONTO_DIR / version
    rd = lambda f: (json.loads((d / f).read_text(encoding="utf-8"))
                    if (d / f).exists() else [])
    concepts, relations = rd("concepts.json"), rd("relations.json")

    items: list[dict] = []
    for c in concepts:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        why = _human_approved(c.get("provenance"))
        if why:
            items.append({"kind": "concept", "key": _norm(c["name"]),
                          "label": c["name"], "rule": "human_approved",
                          "why": why})
    for r in relations:
        if not isinstance(r, dict) or not (r.get("subject") and r.get("predicate")
                                           and r.get("object")):
            continue
        key = f"{_norm(r['subject'])}|{r['predicate']}|{_norm(r['object'])}"
        label = f"{r['subject']} {r['predicate']} {r['object']}"
        why = _human_approved(r.get("provenance"))
        if why:
            items.append({"kind": "relation", "key": key, "label": label,
                          "rule": "human_approved", "why": why})
        elif r["predicate"] in SAFETY_PREDICATES:
            items.append({"kind": "relation", "key": key, "label": label,
                          "rule": "safety_constraint",
                          "why": f"`{r['predicate']}` carries a safety constraint"})
    seen = {i["key"] for i in items}
    for key, mi in load_manual().items():
        if key in seen:
            continue
        items.append({"kind": mi["kind"], "key": key, "label": mi["label"],
                      "rule": "manual",
                      "why": f"marked critical by {mi['protected_by']}: "
                             f"{mi['reason']}"})
    return {
        "version": version,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": {
            "human_approved": "provenance names an approver, or the element was "
                              "merged through a governance proposal (§8.4)",
            "safety_constraint": "the predicate is one of "
                                 + ", ".join(f"`{p}`" for p in SAFETY_PREDICATES),
            "manual": "a reviewer holding `element.protect` marked this one "
                      "element critical; only `admin` can lift it",
        },
        "counts": {
            "total": len(items),
            "concepts": sum(1 for i in items if i["kind"] == "concept"),
            "relations": sum(1 for i in items if i["kind"] == "relation"),
            "by_rule": dict(Counter(i["rule"] for i in items)),
            "by_predicate": dict(Counter(
                i["key"].split("|")[1] for i in items if i["kind"] == "relation")),
        },
        "items": items,
    }


def write_protected_registry(version: str) -> dict:
    reg = derive_protected(version)
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    PROTECTED.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    _audit("protected.derive", version=version, n=reg["counts"]["total"])
    return reg


def protected_keys(version: str) -> dict[str, dict]:
    """Protected elements of `version`, indexed by key, read from the registry
    when it was derived from that same version and recomputed otherwise."""
    reg = None
    if PROTECTED.exists():
        try:
            cand = json.loads(PROTECTED.read_text(encoding="utf-8"))
            if cand.get("version") == version:
                reg = cand
        except json.JSONDecodeError:
            reg = None
    reg = reg or derive_protected(version)
    return {i["key"]: i for i in reg["items"]}


def conflicts_with_protected(additions: list[dict], base_version: str) -> list[dict]:
    """Which protected elements a proposal would override or redefine.

    Two shapes count, and they are the two this system can actually produce:
    a triple that keeps a protected triple's subject and predicate but changes
    its object (an override), and a concept that reuses a protected concept's
    name under a different class (a redefinition). A plain addition that
    touches nothing protected is not a conflict.
    """
    prot = protected_keys(base_version)
    d = ONTO_DIR / base_version
    rd = lambda f: (json.loads((d / f).read_text(encoding="utf-8"))
                    if (d / f).exists() else [])
    by_sp: dict[tuple, list[dict]] = {}
    for r in rd("relations.json"):
        if isinstance(r, dict) and r.get("subject") and r.get("predicate"):
            k = f"{_norm(r['subject'])}|{r['predicate']}|{_norm(r.get('object'))}"
            if k in prot:
                by_sp.setdefault((_norm(r["subject"]), r["predicate"]), []).append(
                    {"key": k, "object": r.get("object")})
    cls_of = {_norm(c.get("name")): c.get("concept_class")
              for c in rd("concepts.json") if isinstance(c, dict) and c.get("name")}

    out = []
    for a in additions or []:
        if not isinstance(a, dict):
            continue
        if a.get("kind") == "relation":
            sp = (_norm(a.get("subject")), a.get("predicate"))
            for existing in by_sp.get(sp, []):
                if _norm(existing["object"]) != _norm(a.get("object")):
                    out.append({"protected": prot[existing["key"]]["label"],
                                "rule": prot[existing["key"]]["rule"],
                                "shape": "override",
                                "proposed": f"{a.get('subject')} {a.get('predicate')} "
                                            f"{a.get('object')}"})
        elif a.get("kind") == "concept":
            k = _norm(a.get("name"))
            if k in prot and cls_of.get(k) not in (None, a.get("concept_class")):
                out.append({"protected": prot[k]["label"],
                            "rule": prot[k]["rule"],
                            "shape": "redefinition",
                            "proposed": f"{a.get('name')} as "
                                        f"{a.get('concept_class')} "
                                        f"(was {cls_of.get(k)})"})
    return out


def guard_protected(additions: list[dict], base_version: str,
                    principal: str | None = None) -> list[dict]:
    """Let a protected-element change through only for a principal that may."""
    hits = conflicts_with_protected(additions, base_version)
    if hits:
        require("element.modify.protected", principal,
                base=base_version, n_conflicts=len(hits),
                first=hits[0]["protected"])
    return hits


# ── report ────────────────────────────────────────────────────────────────
def _self_check() -> list[str]:
    """Properties the policy has to hold. Reported, not assumed."""
    bad = []
    for r, spec in ROLES.items():
        for g in spec["grants"]:
            if g not in PERMISSIONS:
                bad.append(f"role {r} grants unknown permission {g}")
        for i in spec["inherits"]:
            if i not in ROLES:
                bad.append(f"role {r} inherits unknown role {i}")
    for r in ROLES:                                  # no inheritance cycles
        seen, stack = set(), [r]
        while stack:
            x = stack.pop()
            if x == r and seen:
                bad.append(f"inheritance cycle through {r}")
                break
            if x in seen:
                continue
            seen.add(x)
            stack += ROLES.get(x, {}).get("inherits", [])
    for p in PERMISSIONS:                            # every permission reachable
        if not holders(p):
            bad.append(f"permission {p} is held by no role")
    if "ontology.read" not in permissions(DEFAULT_ROLE):
        bad.append("the default role cannot read")
    for p in ("pointer.set", "proposal.merge.production", "element.modify.protected",
              "version.rollback"):
        if holders(p) != {"admin"}:                  # the four that must stay admin-only
            bad.append(f"{p} is not admin-only: {sorted(holders(p))}")
    return bad


def build_report(version: str | None = None) -> dict:
    from wbs.task3_governance.version_governance import load_proposals, versions

    version = version or _published_version() or (versions() or ["1.4.0"])[-1]
    reg = write_protected_registry(version)
    princ = load_principals()
    checks = _self_check()

    # what the audit trail already records about access decisions
    dec = Counter()
    if AUDIT.exists():
        for line in open(AUDIT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("op", "")).startswith("access."):
                dec[r["op"]] += 1

    # separation of duties against the proposals actually on record
    sod = []
    for p in load_proposals():
        appr = [h for h in p.get("history", []) if h.get("to") == "approved"]
        for h in appr:
            if h.get("actor") and h["actor"] == p.get("actor"):
                sod.append({"proposal": p["id"], "actor": h["actor"],
                            "created": p.get("created"), "approved": h.get("ts")})

    data = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version_checked": version,
        "roles": {r: {"describes": s["describes"], "inherits": s["inherits"],
                      "permissions": sorted(permissions(r))}
                  for r, s in ROLES.items()},
        "permissions": list(PERMISSIONS),
        "default_role": DEFAULT_ROLE,
        "principals": princ,
        "protected": reg["counts"],
        "protected_rules": reg["rules"],
        "separation_of_duties": {"enforced": REQUIRE_DISTINCT_APPROVER,
                                 "violations_on_record": sod},
        "isolation": {"production_pattern": PRODUCTION_RE.pattern,
                      "examples": {v: namespace_of(v)
                                   for v in ("1.4.0", "1.6.0", "1.3.0-norm",
                                             "1.7.0-exp1", "staging-policy")}},
        "audit_decisions": dict(dec),
        "self_check": {"problems": checks, "passed": not checks},
    }
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    pc = reg["counts"]
    L = ["# Knowledge Governance: Role-Based Access Control", "",
         "Proposal §8.7. Four properties, each enforced in code rather than "
         "described: who may change knowledge, which knowledge is protected from "
         "being changed, which versions an unprivileged principal may write at "
         "all, and a durable record of every decision including the refusals.", "",
         "Every governance entry point in `version_governance.py` calls "
         "`access_control.require()` before it touches a file. A principal "
         "without the permission receives `AccessDenied`; the ontology is not "
         "modified, and the refusal is appended to the same audit trail as the "
         "successes.", "",
         "## Roles and permissions", "",
         "`inherits` is transitive. Each row shows only what the role adds.", "",
         "| Role | Who it is | Inherits | Adds |", "|---|---|---|---|"]
    for r, s in ROLES.items():
        L.append(f"| `{r}` | {s['describes']} | "
                 f"{', '.join(f'`{i}`' for i in s['inherits']) or '—'} | "
                 f"{', '.join(f'`{g}`' for g in s['grants'])} |")
    L += ["",
          f"An actor the registry does not list gets `{DEFAULT_ROLE}` — least "
          "privilege, so an unknown principal can read the ontology and change "
          "nothing. The registry is "
          f"`{PRINCIPALS.relative_to(ROOT)}`, {len(princ)} principals: "
          + ", ".join(f"`{k}` ({v['role']})" for k, v in princ.items()) + ".", "",
          "| Permission | Held by |", "|---|---|"]
    for p in PERMISSIONS:
        L.append(f"| `{p}` | " + ", ".join(f"`{r}`" for r in sorted(holders(p))) + " |")
    L += ["",
          "## Protected knowledge elements", "",
          f"Derived from v{version} rather than listed by hand, so a later "
          "version that adds safety facts protects them without this file being "
          f"edited. **{pc['total']:,} elements** are protected: "
          f"{pc['concepts']:,} concept{'' if pc['concepts'] == 1 else 's'} and "
          f"{pc['relations']:,} relation{'' if pc['relations'] == 1 else 's'}.", "",
          "| Rule | What it selects | Count |", "|---|---|--:|"]
    for rule, n in sorted(reg["counts"]["by_rule"].items(), key=lambda x: -x[1]):
        L.append(f"| `{rule}` | {reg['rules'].get(rule, rule)} | {n:,} |")
    L += ["",
          "By predicate: "
          + ", ".join(f"`{k}` {v:,}" for k, v in
                      sorted(pc["by_predicate"].items(), key=lambda x: -x[1]))
          + ".", "",
          "A proposal is refused unless its author holds "
          "`element.modify.protected` when it would **override** a protected "
          "triple (same subject and predicate, different object) or "
          "**redefine** a protected concept (same name, different class). "
          "Adding knowledge that collides with nothing protected needs no "
          "elevated permission — the rule is about degradation, not about "
          "growth.", "",
          "A third rule, `manual`, exists for the element a reviewer decides "
          "matters where no family rule covers it: a principal holding "
          "`element.protect` adds one with "
          "`--protect relation 'subject|predicate|object' '<reason>'`, and only "
          "`admin` can lift it with `--unprotect`. The asymmetry is deliberate "
          "— the failure worth preventing is the quiet *removal* of a "
          "protection. "
          + (f"{len(load_manual())} manual protection(s) are in force "
             f"(`{MANUAL.name}`)." if load_manual()
             else f"None are in force at the moment; `{MANUAL.name}` is empty, "
                  "and the enforcement demo adds one and lifts it again in the "
                  "same run so the round trip is exercised rather than "
                  "asserted."), "",
          "## Isolation of experimental updates", "",
          "A bare semantic version is production; anything else is staging.", "",
          "| Version | Namespace | Permission to merge into it |", "|---|---|---|"]
    for v, ns in data["isolation"]["examples"].items():
        L.append(f"| `{v}` | {ns} | `{merge_permission(v)}` |")
    L += ["",
          "`automation` holds `proposal.merge.staging` and not "
          "`proposal.merge.production`, so the extraction pipeline can write a "
          "candidate version and measure it without being able to change what "
          "the deployed system serves. Moving the current-version pointer is a "
          "separate permission again (`pointer.set`, admin only), so even a "
          "production merge does not publish by itself.", "",
          "## Accountability", "",
          f"`{AUDIT.relative_to(ROOT)}` is append-only and now carries the "
          "access decisions alongside the version operations: "
          + (", ".join(f"`{k}` {v:,}" for k, v in sorted(dec.items())) or "none yet")
          + ". A denial is as durable as a change.", "",
          "## Self-check", ""]
    if checks:
        L += [f"**{len(checks)} problem(s) in the policy itself:**", ""]
        L += [f"- {c}" for c in checks] + [""]
    else:
        L += ["The policy is checked against itself on every run: every grant "
              "names a real permission, inheritance has no cycles, every "
              "permission is reachable from some role, the default role can "
              "read, and the four permissions that must never leave `admin` "
              "— `pointer.set`, `proposal.merge.production`, "
              "`element.modify.protected`, `version.rollback` — are held by "
              "`admin` alone. All checks pass.", ""]

    L += ["## What this found", ""]
    if sod:
        L += ["Separation of duties is enforced for new approvals, and the "
              f"proposals already on record do not satisfy it: "
              + ", ".join(f"**{s['proposal']}** was created and approved by "
                          f"`{s['actor']}`" for s in sod)
              + ". That merge predates this policy and is left in the record "
                "rather than rewritten; under today's rule it would be refused, "
                "and a second principal holding `proposal.approve` would have "
                "to review it. On a single-operator host that is a real cost, "
                "and it is the reason the registry carries distinct operator, "
                "reviewer and pipeline identities instead of one account.", ""]
    else:
        L += ["No proposal on record was approved by its own author.", ""]
    L += ["What this layer does not do: it authenticates nothing. The principal "
          "is whatever `WBS_ACTOR` or `USER` says, which is sufficient for "
          "attribution and policy enforcement inside one trust boundary and is "
          "not a substitute for an identity provider. R14 in "
          "`docs/risk_management_plan.md` carries that gap.", ""]

    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    return data


def _published_version() -> str | None:
    p = GOV_DIR / "current_version.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("version")
        except json.JSONDecodeError:
            return None
    return None


# ── demo ──────────────────────────────────────────────────────────────────
def demo() -> dict:
    """Exercise every rule on the real ontology, as four principals.

    Nothing here is mocked: each step sets `WBS_ACTOR` and calls the same
    `require()` the governance module calls, against the protected set derived
    from the published version.
    """
    base = _published_version() or "1.4.0"
    prot = derive_protected(base)
    a_protected = next((i for i in prot["items"] if i["kind"] == "relation"), None)
    subj, pred, _obj = (a_protected["key"].split("|") if a_protected
                        else ("Seongsan Ilchulbong", "has_risk", "x"))

    steps = []

    def attempt(who: str, what: str, fn):
        prev = os.environ.get("WBS_ACTOR")
        os.environ["WBS_ACTOR"] = who
        try:
            fn()
            steps.append({"principal": who, "role": role_of(who), "attempt": what,
                          "outcome": "allowed", "detail": ""})
        except AccessDenied as e:
            steps.append({"principal": who, "role": role_of(who), "attempt": what,
                          "outcome": "denied", "detail": str(e)})
        finally:
            if prev is None:
                os.environ.pop("WBS_ACTOR", None)
            else:
                os.environ["WBS_ACTOR"] = prev

    attempt("nlp-lab-pipeline", "create a change proposal",
            lambda: require("proposal.create"))
    attempt("nlp-lab-pipeline", "approve its own proposal",
            lambda: require("proposal.approve"))
    attempt("nlp-lab-pipeline", f"merge into a staging version ({base}-exp1)",
            lambda: require(merge_permission(f"{base}-exp1")))
    attempt("nlp-lab-pipeline", f"merge into production (1.7.0)",
            lambda: require(merge_permission("1.7.0")))
    attempt("domain-expert-1", "approve a proposal it did not create",
            lambda: require("proposal.approve"))
    attempt("domain-expert-1", "move the published-version pointer",
            lambda: require("pointer.set"))
    attempt("domain-expert-1", "override a protected safety triple",
            lambda: guard_protected(
                [{"kind": "relation", "subject": subj, "predicate": pred,
                  "object": "something else"}], base))
    attempt("ontology-admin", "override a protected safety triple",
            lambda: guard_protected(
                [{"kind": "relation", "subject": subj, "predicate": pred,
                  "object": "something else"}], base))
    attempt("ontology-admin", "move the published-version pointer",
            lambda: require("pointer.set"))
    attempt("guest", "create a change proposal", lambda: require("proposal.create"))
    attempt("nobody-in-the-registry", "create a change proposal",
            lambda: require("proposal.create"))
    attempt("nlp-lab-pipeline", "add a fact that collides with nothing protected",
            lambda: guard_protected(
                [{"kind": "relation", "subject": "Jeju Island",
                  "predicate": "near", "object": "Udo Island"}], base))
    # protect / unprotect, run for real and then undone, so the registry ends
    # where it started and the audit trail keeps all four decisions
    before_manual = load_manual()
    probe = "Jeju Island|near|Udo Island"
    probe_key, _ = element_key("relation", probe)
    attempt("guest", "mark an element critical",
            lambda: protect("relation", probe, "demo"))
    attempt("domain-expert-1", "mark an element critical",
            lambda: protect("relation", probe,
                            "enforcement demo; removed in the same run"))
    assert probe_key in load_manual(), "protect did not store under its own key"
    attempt("domain-expert-1", "lift that protection again",
            lambda: unprotect(probe_key, "demo"))
    attempt("ontology-admin", "lift that protection again",
            lambda: unprotect(probe_key, "end of the enforcement demo"))
    if load_manual() != before_manual:
        raise SystemExit(
            "the demo left the manual protection registry changed: "
            f"{sorted(set(load_manual()) - set(before_manual))}. "
            "That is a bug in protect/unprotect, not a finding.")

    allowed = sum(1 for s in steps if s["outcome"] == "allowed")
    out = {"base_version": base, "protected_total": prot["counts"]["total"],
           "steps": steps, "allowed": allowed, "denied": len(steps) - allowed}
    (GOV_DIR / "access_control_demo.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Access-control enforcement, exercised", "",
         f"{len(steps)} attempts against the published ontology (v{base}) and its "
         f"{prot['counts']['total']:,} protected elements, each made as a real "
         "principal from the registry through the same `require()` the "
         "governance module calls. **"
         f"{allowed} allowed, {len(steps) - allowed} denied.**", "",
         "| Principal | Role | Attempt | Outcome |", "|---|---|---|---|"]
    for s in steps:
        mark = "✅ allowed" if s["outcome"] == "allowed" else "⛔ denied"
        L.append(f"| `{s['principal']}` | `{s['role']}` | {s['attempt']} | {mark} |")
    L += ["", "Every row above is also a line in "
              f"`{AUDIT.relative_to(ROOT)}`, denials included.", ""]
    (GOV_DIR / "access_control_demo.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--whoami", action="store_true")
    ap.add_argument("--version", default=None)
    ap.add_argument("--protect", nargs=3, metavar=("KIND", "SPEC", "REASON"),
                    help="mark one element critical (needs element.protect). "
                         "SPEC is a concept name, or 'subject|predicate|object'")
    ap.add_argument("--unprotect", nargs=2, metavar=("KEY", "REASON"),
                    help="lift a manual protection (needs element.unprotect)")
    a = ap.parse_args()

    if a.protect:
        print(json.dumps(protect(*a.protect), indent=2, ensure_ascii=False))
        return
    if a.unprotect:
        print("removed" if unprotect(*a.unprotect) else "no such manual protection")
        return

    if a.whoami:
        p = actor()
        print(f"{p}: role={role_of(p)}")
        print("permissions: " + ", ".join(sorted(permissions(role_of(p)))))
        return
    if a.demo:
        d = demo()
        print(f"{d['allowed']} allowed, {d['denied']} denied over "
              f"{len(d['steps'])} attempts (v{d['base_version']}, "
              f"{d['protected_total']:,} protected elements)")
        for s in d["steps"]:
            print(f"  {'OK ' if s['outcome'] == 'allowed' else 'DENY'} "
                  f"{s['principal']:<22} {s['attempt']}")
        print(f"wrote {(GOV_DIR / 'access_control_demo.md').relative_to(ROOT)}")
        return

    d = build_report(a.version)
    print(f"v{d['version_checked']}: {d['protected']['total']:,} protected elements "
          f"({d['protected']['concepts']} concepts, {d['protected']['relations']} relations)")
    print(f"roles: {', '.join(d['roles'])}; principals: {len(d['principals'])}")
    print(f"self-check: {'passed' if d['self_check']['passed'] else d['self_check']['problems']}")
    sod = d["separation_of_duties"]["violations_on_record"]
    if sod:
        print(f"separation of duties: {len(sod)} proposal(s) on record approved by "
              f"their own author ({', '.join(s['proposal'] for s in sod)})")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
