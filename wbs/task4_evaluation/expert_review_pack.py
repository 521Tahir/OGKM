"""
Proposal sections 11.4.2 and 11.6 - the two evaluation requirements that need a
human annotator.

  11.4.2  EXPERT EVALUATION of the ontology: concept relevance and correctness,
          relation validity and completeness, usability for reasoning.
  11.6    MANUAL ANNOTATION of hallucinated statements, alongside the automated
          ontology-based detection the project already runs.

Neither can be satisfied by writing code, and neither may be simulated: an LLM
scoring its own ontology is not an expert evaluation, and a detector grading its
own output is not manual annotation. What code CAN do is build the instrument so
the human work is well-posed, unbiased and auditable. This script does that and
nothing more. It emits:

  expert_ontology_review.csv      stratified concept and relation sample, one row
                                  per item, with the provenance and the source
                                  evidence the rater needs, and empty score columns
  expert_ontology_review.md       the rubric, the sampling frame and the stopping
                                  rule, so a second rater can reproduce the sample
  manual_hallucination_review.csv blinded span-level sample across all variants:
                                  the variant column is a hash, so a rater cannot
                                  grade the system they expect to win
  manual_hallucination_review.md  the label definitions and the adjudication rule
  expert_review_pack.json         the sampling manifest: seed, strata, counts, and
                                  the blinding key, kept separate from the CSV

The CSVs ship unfilled. A deliverable that claims expert scores without an expert
would be the worst kind of result in a trust-and-verifiability project.

  python -m wbs.task4_evaluation.expert_review_pack
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
            "V5_ontology_rag_validated"]
SENT = re.compile(r"(?<=[.!?])\s+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--n-concepts", type=int, default=60)
    ap.add_argument("--n-relations", type=int, default=60)
    ap.add_argument("--n-spans", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans
    rng = random.Random(a.seed)

    cs = json.loads((ONTO / a.version / "concepts.json").read_text(encoding="utf-8"))
    rels = json.loads((ONTO / a.version / "relations.json").read_text(encoding="utf-8"))
    recs = {}
    src = ROOT / "data" / "processed" / "cleaned_records.jsonl"
    if src.exists():
        for l in open(src, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                recs[r.get("record_id")] = r

    # ── stratified concept sample: proportional over concept_class ────────
    by_class = defaultdict(list)
    for c in cs:
        if c.get("name"):
            by_class[c.get("concept_class") or "Untyped"].append(c)
    classes = sorted(by_class, key=lambda k: -len(by_class[k]))
    conc_rows, strata_c = [], {}
    # at least one per class, remainder proportional to class size
    quota = {k: 1 for k in classes}
    left = max(a.n_concepts - len(classes), 0)
    tot = sum(len(v) for v in by_class.values())
    for k in classes:
        quota[k] += int(round(left * len(by_class[k]) / tot))
    for k in classes:
        pool = by_class[k][:]
        rng.shuffle(pool)
        take = pool[: quota[k]]
        strata_c[k] = {"population": len(by_class[k]), "sampled": len(take)}
        for c in take:
            prov = c.get("provenance") or {}
            rid = prov.get("source_record_id")
            ev = ""
            if rid in recs:
                ev = f"{recs[rid].get('Input','')} {recs[rid].get('Output','')}"[:400]
            conc_rows.append({
                "item_id": f"C{len(conc_rows)+1:04d}", "kind": "concept",
                "name": c["name"], "concept_class": k,
                "aliases": "; ".join(c.get("aliases") or [])
                if isinstance(c.get("aliases"), list) else str(c.get("aliases") or ""),
                "confidence": prov.get("confidence"),
                "source_record_id": rid, "source_evidence": ev,
                "relevant_1to5": "", "correct_1to5": "",
                "class_is_right_yes_no": "", "usable_for_reasoning_1to5": "",
                "rater_comment": ""})

    # ── stratified relation sample: proportional over predicate ───────────
    by_pred = defaultdict(list)
    for r in rels:
        if r.get("subject") and r.get("predicate") and r.get("object"):
            by_pred[r["predicate"]].append(r)
    preds = sorted(by_pred, key=lambda k: -len(by_pred[k]))
    rel_rows, strata_r = [], {}
    quota = {k: 1 for k in preds}
    left = max(a.n_relations - len(preds), 0)
    tot = sum(len(v) for v in by_pred.values())
    for k in preds:
        quota[k] += int(round(left * len(by_pred[k]) / tot))
    for k in preds:
        pool = by_pred[k][:]
        rng.shuffle(pool)
        take = pool[: quota[k]]
        strata_r[k] = {"population": len(by_pred[k]), "sampled": len(take)}
        for r in take:
            prov = r.get("provenance") or {}
            rid = prov.get("source_record_id")
            ev = ""
            if rid in recs:
                ev = f"{recs[rid].get('Input','')} {recs[rid].get('Output','')}"[:400]
            rel_rows.append({
                "item_id": f"R{len(rel_rows)+1:04d}", "kind": "relation",
                "name": f"{r['subject']} --{r['predicate']}--> {r['object']}",
                "concept_class": k, "aliases": "",
                "confidence": prov.get("confidence"),
                "source_record_id": rid, "source_evidence": ev,
                "relevant_1to5": "", "correct_1to5": "",
                "class_is_right_yes_no": "", "usable_for_reasoning_1to5": "",
                "rater_comment": ""})

    pack = conc_rows + rel_rows
    cp = EVAL / "expert_ontology_review.csv"
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pack[0]))
        w.writeheader(); w.writerows(pack)
    print(f"wrote {cp.name}: {len(conc_rows)} concepts over {len(classes)} classes, "
          f"{len(rel_rows)} relations over {len(preds)} predicates", flush=True)

    # ── blinded span sample for manual hallucination annotation ───────────
    kb = KnowledgeBase()
    pool = []
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            continue
        for rec in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()):
            ans = rec.get("answer") or ""
            spans, ungrounded, generic = audit_spans(ans, kb)
            ung = set(ungrounded)
            for sp in spans:
                sent = next((s for s in SENT.split(ans) if sp in s), ans[:200])
                pool.append({"variant": v, "bench_id": rec["bench_id"],
                             "category": rec["category"],
                             "question": (rec.get("Input") or "")[:300],
                             "span": sp, "sentence": sent.strip()[:300],
                             "auto_label": "ungrounded" if sp in ung else "grounded"})
    # sample so that flagged and unflagged spans are both represented, and so that
    # no variant dominates: equal quota per variant, half from each auto label
    rng.shuffle(pool)
    per_v = max(a.n_spans // max(len(VARIANTS), 1), 2)
    chosen = []
    for v in VARIANTS:
        for lab in ("ungrounded", "grounded"):
            sub = [x for x in pool if x["variant"] == v and x["auto_label"] == lab]
            chosen += sub[: per_v // 2]
    rng.shuffle(chosen)
    # blinding: the rater sees a stable hash, the key lives in the manifest only
    key = {v: hashlib.sha1(f"{v}|{a.seed}".encode()).hexdigest()[:8] for v in VARIANTS}
    span_rows = []
    for i, x in enumerate(chosen, 1):
        span_rows.append({"item_id": f"H{i:04d}", "system": key[x["variant"]],
                          "bench_id": x["bench_id"], "category": x["category"],
                          "question": x["question"], "span": x["span"],
                          "sentence": x["sentence"],
                          "label_supported_unsupported_unverifiable_generic": "",
                          "is_a_place_yes_no": "", "rater_comment": ""})
    hp = EVAL / "manual_hallucination_review.csv"
    with open(hp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(span_rows[0]))
        w.writeheader(); w.writerows(span_rows)
    auto = Counter(x["auto_label"] for x in chosen)
    print(f"wrote {hp.name}: {len(span_rows)} blinded spans "
          f"({auto['ungrounded']} auto-flagged, {auto['grounded']} not)", flush=True)

    man = {"seed": a.seed, "ontology_version": a.version, "model": a.model,
           "concept_strata": strata_c, "relation_strata": strata_r,
           "n_concepts": len(conc_rows), "n_relations": len(rel_rows),
           "n_spans": len(span_rows), "auto_label_counts": dict(auto),
           "blinding_key": key,
           "status": "instruments emitted unfilled; no domain-expert panel is "
                     "available to this project, so the filled results are out of "
                     "scope for this phase and are not simulated"}
    (EVAL / "expert_review_pack.json").write_text(json.dumps(man, indent=2),
                                                  encoding="utf-8")

    # ── protocols ─────────────────────────────────────────────────────────
    (EVAL / "expert_ontology_review.md").write_text(f"""# Expert ontology evaluation protocol (proposal 11.4.2)

`expert_ontology_review.csv` holds {len(pack)} items drawn from ontology
v{a.version}: {len(conc_rows)} concepts stratified over {len(classes)} concept
classes and {len(rel_rows)} relations stratified over {len(preds)} predicates.
Sampling is proportional to stratum size with one guaranteed item per stratum, so
a rare class is represented without a common one swamping the sheet. Seed
{a.seed}; `expert_review_pack.json` records every stratum's population and draw,
which is what makes the sample reproducible rather than merely random.

## What each column asks

| Column | Scale | Question put to the rater |
|---|---|---|
| `relevant_1to5` | 1-5 | Does this belong in a Jeju tourism ontology at all? |
| `correct_1to5` | 1-5 | Is the assertion factually right about Jeju? |
| `class_is_right_yes_no` | yes/no | For a concept, is `concept_class` the right class? For a relation, is the predicate the right predicate? |
| `usable_for_reasoning_1to5` | 1-5 | Could a system answer a visitor's question better for having this item? |
| `rater_comment` | free text | Required whenever any score is 1 or 2. |

`source_evidence` is the cleaned source record the item was extracted from, where
provenance records one. Judge correctness against Jeju, not against the evidence:
an item faithfully extracted from a wrong sentence is still wrong.

## Procedure

1. Two raters score independently. No discussion before both sheets are complete.
2. Report Krippendorff's alpha for the ordinal columns and Cohen's kappa for the
   yes/no column before reporting any mean score. An agreement figure that is not
   reported is an agreement figure that was not computed.
3. Adjudicate every item where the two raters differ by 2 or more points, with a
   third rater or by discussion, and keep both the pre- and post-adjudication
   numbers.
4. Items scoring 1-2 on `correct_1to5` go to `version_governance` as a removal or
   repair proposal, through the normal proposal state machine, so an expert
   correction enters the ontology the same auditable way any other change does.

## Status

Unfilled. This project has no domain-expert panel, so the instrument is the
deliverable and the scores are out of scope for this phase. They are not
simulated and no placeholder numbers appear anywhere in the reports.
""", encoding="utf-8")

    (EVAL / "manual_hallucination_review.md").write_text(f"""# Manual hallucination annotation protocol (proposal 11.6)

`manual_hallucination_review.csv` holds {len(span_rows)} entity spans taken from
the answers of all five variants on the {a.model} run: {auto['ungrounded']} that
the automated audit flagged as ungrounded and {auto['grounded']} that it did not.
Including both is the point - a sheet of only flagged spans measures precision
and can say nothing about what the detector missed.

The `system` column is an 8-character hash, not a variant name. A rater who knows
which row came from the ontology arm is no longer blind, and in a project whose
stated expectation is that the ontology arm wins, that is the bias most likely to
appear. The key is in `expert_review_pack.json`, which the raters do not open.

## Labels

| Label | Meaning |
|---|---|
| `supported` | The span names a real Jeju entity AND the sentence's claim about it is true. |
| `unsupported` | The entity does not exist, or the claim about it is false. This is the hallucination count. |
| `unverifiable` | A real entity, but the claim cannot be checked from any public source. |
| `generic` | Not an entity at all: a heading, a month, an imperative verb, a common noun the detector mistook for a name. |

`is_a_place_yes_no` separates place names from other proper nouns, because the
project's entity metrics are computed over a 69-place gazetteer and a span that
is not a place is outside that metric's scope.

## Procedure

1. Two raters, independent, no discussion until both sheets are done.
2. Report Cohen's kappa over the four labels before any rate.
3. Adjudicate disagreements with a third rater.
4. Then, and only then, unblind and compute per-variant unsupported rates, and
   compare them against the automated `ungrounded_rate` in
   `hallucination_report__{a.model}.md`. The comparison yields the detector's
   precision (flagged spans that raters call `unsupported`) and its recall
   (unsupported spans the detector flagged), which is the number the project
   currently reports as a known unknown.

## Status

Unfilled, for the same reason as the expert ontology review: no annotator panel
is available to this phase. The automated detector's own false-positive
behaviour is documented in `hallucination_report__{a.model}.md`; this sheet is
what would replace that caveat with a measurement.
""", encoding="utf-8")
    print("wrote expert_ontology_review.md, manual_hallucination_review.md, "
          "expert_review_pack.json")


if __name__ == "__main__":
    main()
