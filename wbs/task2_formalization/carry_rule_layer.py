"""WBS Task 2.2 / proposal 5.6, R12 - carry the rule layer into the production
ontology version.

Fifteen rules (8 seed, 7 mined) were serialized into ontology versions 0.1.0-1.3.0
and 3.0.0. Version 1.4.0 - the one `KnowledgeBase`, the retriever and every
evaluation arm actually load - carries concepts, relations and hierarchy only, so
the production ontology has had no rule layer and no temporal layer. That is why
proposal 5.6 audited as partial and why R12 stayed open.

This carries both forward, and does not do it blindly. Every rule is re-checked
against what 1.4.0 actually contains: a rule that quantifies over an attribute no
concept carries, or a predicate no relation uses, is carried with
`enforceable_in` recording why it cannot fire. A rule layer whose rules cannot be
evaluated is worse than none, because it reads as governance.

Concepts, relations and hierarchy are not touched; this only adds files.

  python -m wbs.task2_formalization.carry_rule_layer --from 1.3.0 --to 1.4.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
OUT = ROOT / "wbs" / "deliverables" / "task2_formalization"

# Which rules can fire is a question `rule_compliance.py` already answers, from a
# curated per-rule `needs` table checked against what the ontology carries. This
# module used to re-derive the answer with a regex over the rule text, and that
# was worse in every case the two disagreed: it split the predicate
# `supports_dietary_need` into the words "dietary" and "traveler" and called
# R002/R008 unenforceable, and it could not see R006's "weather state" at all,
# because a multi-word need is not a symbol in the text. Two enforceability
# verdicts in one project is one too many, so this defers to that one and the
# regex survives only for rules the table does not cover (the mined rules).
DOTTED = re.compile(r"\b([a-z_]+(?:\.[a-z_]+)+)")
COMPARED = re.compile(r"\b([a-z_]{3,})\s*(?:=|<|>|IN\b|IS\b|NOT\b|EXISTS\b|"
                      r"MATCHES\b|CONTAINS\b)")
# `type` and `confidence` are structural: every concept carries a class and a
# provenance confidence, so a rule quantifying over them is enforceable.
STRUCTURAL = {"type", "confidence", "entity", "concept"}


def required_symbols(rule: dict) -> set[str]:
    text = f"{rule.get('condition', '')} {rule.get('consequence', '')}".lower()
    need = set()
    for path in DOTTED.findall(text):
        parts = path.split(".")
        need |= set(parts[1:])
        need.add(parts[0])
    need |= set(COMPARED.findall(text))
    return {w for w in need if w not in STRUCTURAL}


def attribute_support() -> dict:
    """`rule_compliance.py`'s measurement of what each attribute is backed by.

    Needed so the carry can say *why* an attribute is unavailable. "No project
    source supplies it" was the blanket phrase here, and for two of the four
    rules it was wrong: traveler type is carried by 270 of 283 `suitable_for`
    triples, and weather state exists as a class with 8 instances but no
    predicate links to it on more than 11% of its own triples. Those are
    different findings and a reader acting on them would do different things.
    """
    f = OUT / "rule_compliance.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8")).get("attribute_support") or {}
    return d


def _why_missing(attr: str, sup: dict) -> str:
    """One clause saying what is actually wrong with `attr`."""
    for scope in ("production", "seed"):
        e = (sup.get(scope) or {}).get(attr)
        if not e:
            continue
        if e.get("concepts") and e.get("linked_share") is not None:
            return (f"{attr} (the class has {e['concepts']} instances, but the "
                    f"predicate that links to it best, `{e['linked_by']}`, puts "
                    f"only {e['linked_triples']} of its {e['of_predicate_triples']} "
                    f"objects there - present in the vocabulary, not queryable)")
        if "storm_like_concepts" in e:
            return (f"{attr} (no WeatherCondition instance denotes a storm, "
                    f"typhoon or high-sea state)")
        return f"{attr} (no concept attribute, relation or class carries it)"
    return f"{attr} (not carried by this ontology version)"


def curated_verdicts() -> dict:
    """`rule_compliance.py`'s verdict per seed rule, if it has been run."""
    f = OUT / "rule_compliance.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    return d.get("enforceability", {}) or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="1.3.0")
    ap.add_argument("--to", dest="dst", default="1.4.0")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sd, dd = ONTO / a.src, ONTO / a.dst
    for d in (sd, dd):
        if not d.exists():
            sys.exit(f"{d} does not exist")

    concepts = json.loads((dd / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((dd / "relations.json").read_text(encoding="utf-8"))
    classes = {str(c.get("concept_class", "")).lower()
               for c in concepts if isinstance(c, dict)}
    attrs = set()
    for c in concepts:
        if isinstance(c, dict):
            attrs |= {str(k).lower() for k in (c.get("attributes") or {})}
    preds = {str(r.get("predicate", "")).lower()
             for r in relations if isinstance(r, dict)}

    rules = json.loads((sd / "rules.json").read_text(encoding="utf-8"))
    curated = curated_verdicts()
    carried, enforceable = [], 0
    SUPPORT = attribute_support()
    for r in rules:
        rid = r.get("rule_id", "")
        cur = curated.get(rid)
        if cur is not None:
            ok = bool(cur.get("enforceable"))
            need = sorted(cur.get("needs") or [])
            missing = sorted(cur.get("missing_from_ontology") or [])
            src_of = "rule_compliance.py (curated needs)"
            reason = ("" if ok else
                      "needs " + ", ".join(_why_missing(m, SUPPORT)
                                           for m in (missing or need)))
        else:
            # mined rules: no curated entry, so fall back to the text
            need = sorted(required_symbols(r))
            missing = sorted(w for w in need
                             if w not in attrs and w not in preds
                             and w not in classes and w.rstrip("s") not in classes)
            ok = not missing
            src_of = "symbols read from the rule text"
            reason = ("" if ok else
                      "quantifies over " + ", ".join(missing)
                      + f", which v{a.dst} does not carry as an attribute, "
                        f"predicate or class and no project source supplies at "
                        f"query time")
        enforceable += ok
        r2 = dict(r)
        r2["enforceable_in"] = {
            "version": a.dst, "enforceable": ok, "requires": need,
            "missing": missing, "verdict_from": src_of, "reason": reason}
        r2["carried_from"] = a.src
        carried.append(r2)

    tsrc = sd / "temporal_facts.json"
    tfacts = (json.loads(tsrc.read_text(encoding="utf-8"))
              if tsrc.exists() else None)
    ssrc = sd / "schema.json"
    schema = json.loads(ssrc.read_text(encoding="utf-8")) if ssrc.exists() else None

    print(f"{len(carried)} rules, {enforceable} enforceable against v{a.dst} "
          f"({len(classes)} classes, {len(attrs)} attribute names, "
          f"{len(preds)} predicates)", flush=True)
    for r in carried:
        if not r["enforceable_in"]["enforceable"]:
            print(f"  {r['rule_id']:<12} not enforceable: "
                  f"{r['enforceable_in']['reason']}", flush=True)
    if a.dry_run:
        print("dry run - nothing written")
        return

    (dd / "rules.json").write_text(json.dumps(carried, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    if tfacts is not None:
        meta = tfacts.setdefault("meta", {})
        meta["carried_from"] = a.src
        meta["carried_at"] = datetime.now(timezone.utc).isoformat()
        (dd / "temporal_facts.json").write_text(
            json.dumps(tfacts, indent=2, ensure_ascii=False), encoding="utf-8")
    if schema is not None:
        (dd / "schema.json").write_text(json.dumps(schema, indent=2,
                                                   ensure_ascii=False),
                                        encoding="utf-8")

    ch = dd / "changelog.json"
    log = json.loads(ch.read_text(encoding="utf-8")) if ch.exists() else []
    if isinstance(log, dict):
        log = [log]
    log.append({"date": datetime.now(timezone.utc).isoformat(),
                "change": "rule layer carried forward",
                "detail": f"rules.json ({len(carried)} rules, {enforceable} "
                          f"enforceable), temporal_facts.json and schema.json "
                          f"copied from v{a.src}; concepts, relations and "
                          f"hierarchy unchanged",
                "tool": "wbs/task2_formalization/carry_rule_layer.py"})
    ch.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rule_layer_carry.json").write_text(json.dumps(
        {"from": a.src, "to": a.dst, "n_rules": len(carried),
         "n_enforceable": enforceable,
         "files_written": [p for p in ("rules.json", "temporal_facts.json",
                                       "schema.json")
                           if (dd / p).exists()],
         "per_rule": {r["rule_id"]: r["enforceable_in"] for r in carried}},
        indent=2), encoding="utf-8")
    print(f"wrote {dd}/rules.json and {OUT / 'rule_layer_carry.json'}")


if __name__ == "__main__":
    main()
