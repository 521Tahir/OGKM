"""
WBS Task 4.1.2 - Score the five system variants: baseline comparison + hallucination audit.

Every number here comes from the real generations written by run_variants.py. Three
families of measurement are reported side by side because they answer different
questions and, on this corpus, they disagree:

  REFERENCE OVERLAP   token-level P/R/F1 against the gold answer (SQuAD-style
                      multiset overlap of content tokens). Rewards reproducing the
                      reference's wording, so a retrieval arm whose passages are drawn
                      from the same corpus as the reference has a structural advantage.

  ENTITY CORRECTNESS  the same P/R/F1 computed over gazetteer place names only. This
                      asks whether the answer names the right PLACES, which is what a
                      tourism answer is judged on and what an ontology is meant to supply.

  GROUNDEDNESS        the Task 3.2.2 knowledge base is used as an auditor, not a
                      generator: every entity-like span in the answer is checked
                      against ontology v1.4.0 concepts, aliases and the gazetteer.
                      ungrounded_rate is the hallucination measure - the fraction of
                      named entities the knowledge base cannot account for. It is
                      reference-free, so unlike the first two families it does not
                      reward copying the corpus.

Significance: per-item paired Wilcoxon signed-rank against the designated reference
variant, two-sided, over items where both variants produced a countable value. No
bootstrap - n=347 and the per-item deltas are not normal.

  python -m wbs.task4_evaluation.score_variants --model llama3.1-8b
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp042_prf.compute_prf import ents, prf, toks  # noqa: E402
from wbs.task3_reasoning.constrained_reasoner import (UNSUPPORTABLE,  # noqa: E402
                                                      KnowledgeBase)
from wbs.task4_evaluation.hallucination_report import audit_spans  # noqa: E402

VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only",
            "V4_ontology_rag", "V5_ontology_rag_validated"]
LABEL = {"V1_llm": "V1 LLM only", "V2_rag": "V2 Simple RAG",
         "V3_ontology_only": "V3 Ontology only", "V4_ontology_rag": "V4 Ontology+RAG",
         "V5_ontology_rag_validated": "V5 Ontology+RAG+validation"}
CATEGORIES = ["recommendation", "itinerary", "safety", "dining",
              "accessibility", "transport"]


def wilcoxon(a: list[float], b: list[float]):
    """Two-sided Wilcoxon signed-rank on paired samples. Returns (statistic, p, n)."""
    d = [x - y for x, y in zip(a, b) if x != y]
    if len(d) < 6:
        return None
    try:
        from scipy.stats import wilcoxon as w
        r = w(d, alternative="two-sided", zero_method="wilcox")
        return (float(r.statistic), float(r.pvalue), len(d))
    except Exception:
        return None


def score_item(rec: dict, kb: KnowledgeBase) -> dict:
    gold, pred = rec.get("reference") or "", rec.get("answer") or ""
    t = prf(toks(pred), toks(gold))
    ge, pe = ents(gold), ents(pred)
    e = prf(pe, ge)
    # audit-grade spans: the validator's own detector is recall-oriented and counts
    # sentence-initial imperatives, month names and markdown headings as entities,
    # which is right for a safety check and wrong for a measurement. Both are kept.
    rspans = kb.spans(pred)
    rbad = [s for s in rspans if not kb.known(s)]
    spans, ungrounded, _generic = audit_spans(pred, kb)
    unsupportable = sum(len(pat.findall(pred)) for pat, _ in UNSUPPORTABLE)
    return {
        "bench_id": rec["bench_id"], "category": rec["category"],
        "token_p": t[0] if t else None, "token_r": t[1] if t else None,
        "token_f1": t[2] if t else None,
        "entity_p": e[0] if e else None, "entity_r": e[1] if e else None,
        "entity_f1": e[2] if e else None,
        "entity_hit": (1.0 if sum((pe & ge).values()) > 0 else 0.0) if e else None,
        "n_spans": len(spans), "n_ungrounded": len(ungrounded),
        "ungrounded_rate": (len(ungrounded) / len(spans)) if spans else None,
        "ungrounded_rate_raw": (len(rbad) / len(rspans)) if rspans else None,
        "clean": 1.0 if spans and not ungrounded else (0.0 if spans else None),
        "unsupportable": float(unsupportable),
        "answer_chars": len(pred),
        "generation_s": rec.get("generation_s"),
        "prompt_tokens": rec.get("prompt_tokens"),
        "validation_ms": rec.get("validation_ms"),
        "repair_rounds": rec.get("repair_rounds"),
        "fallback": (1.0 if rec.get("uncertainty_fallback") else 0.0)
        if "uncertainty_fallback" in rec else None,
    }


FIELDS = ["token_p", "token_r", "token_f1", "entity_p", "entity_r", "entity_f1",
          "entity_hit", "ungrounded_rate", "ungrounded_rate_raw", "clean",
          "unsupportable",
          "answer_chars", "generation_s", "prompt_tokens"]


def agg(items: list[dict]) -> dict:
    o = {"n": len(items)}
    for f in FIELDS:
        v = [i[f] for i in items if i.get(f) is not None]
        o[f] = st.mean(v) if v else None
    tf = [i["token_f1"] for i in items if i["token_f1"] is not None]
    o["acc_token_f1_50"] = st.mean(1.0 if x >= .5 else 0.0 for x in tf) if tf else None
    ef = [i["entity_f1"] for i in items if i["entity_f1"] is not None]
    o["acc_entity_f1_50"] = st.mean(1.0 if x >= .5 else 0.0 for x in ef) if ef else None
    for f in ("validation_ms", "repair_rounds", "fallback"):
        v = [i[f] for i in items if i.get(f) is not None]
        o[f] = st.mean(v) if v else None
    return o


def fmt(v, pct=True, nd=1):
    if v is None:
        return "—"
    return f"{v*100:.{nd}f}" if pct else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--reference", default="V2_rag",
                    help="variant that significance tests compare against")
    a = ap.parse_args()

    D = ROOT / "wbs" / "deliverables" / "task4_evaluation"
    kb = KnowledgeBase()
    print(f"knowledge base: ontology v{kb.version}, "
          f"{len(kb.surface):,} surface forms, {len(kb.triples):,} triples", flush=True)

    per_item: dict[str, list[dict]] = {}
    for v in VARIANTS:
        p = D / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            print(f"  MISSING {p.name} - skipped")
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        per_item[v] = [score_item(r, kb) for r in recs]
        print(f"  {v:28s} {len(recs):4,} items scored", flush=True)
    if a.reference not in per_item:
        sys.exit(f"reference variant {a.reference} has no generations")

    overall = {v: agg(it) for v, it in per_item.items()}
    bycat = {v: {c: agg([i for i in it if i["category"] == c]) for c in CATEGORIES}
             for v, it in per_item.items()}

    # ── paired significance vs the reference variant ──────────────────────
    sig: dict[str, dict] = {}
    ref_by_id = {i["bench_id"]: i for i in per_item[a.reference]}
    for v, it in per_item.items():
        if v == a.reference:
            continue
        sig[v] = {}
        for f in ("token_f1", "entity_f1", "ungrounded_rate", "clean"):
            x, y = [], []
            for i in it:
                r = ref_by_id.get(i["bench_id"])
                if r and i.get(f) is not None and r.get(f) is not None:
                    x.append(i[f]); y.append(r[f])
            res = wilcoxon(x, y)
            sig[v][f] = {"delta": (st.mean(x) - st.mean(y)) if x else None,
                         "n_paired": len(x),
                         "p": res[1] if res else None,
                         "n_nonzero": res[2] if res else None}

    # ── V4 -> V5: what the validation layer actually changed ──────────────
    v45 = None
    if "V4_ontology_rag" in per_item and "V5_ontology_rag_validated" in per_item:
        v4 = {i["bench_id"]: i for i in per_item["V4_ontology_rag"]}
        rows5 = [json.loads(l) for l in
                 open(D / f"variant__{a.model}__V5_ontology_rag_validated.jsonl",
                      encoding="utf-8") if l.strip()]
        changed = sum(1 for r in rows5 if (r.get("draft") or "") != (r.get("answer") or ""))
        vt: dict[str, int] = defaultdict(int)
        for r in rows5:
            for x in r.get("violations") or []:
                vt[x["type"]] += 1
        i5 = {i["bench_id"]: i for i in per_item["V5_ontology_rag_validated"]}
        both = [b for b in i5 if b in v4]
        v45 = {
            "answers_rewritten": changed, "n": len(rows5),
            "violation_types": dict(vt),
            "violations_per_item": rate(sum(vt.values()), rows5),
            "fallback_rate": overall["V5_ontology_rag_validated"]["fallback"],
            "mean_repair_rounds": overall["V5_ontology_rag_validated"]["repair_rounds"],
            "mean_validation_ms": overall["V5_ontology_rag_validated"]["validation_ms"],
            "delta_ungrounded_rate": st.mean(
                [i5[b]["ungrounded_rate"] - v4[b]["ungrounded_rate"] for b in both
                 if i5[b]["ungrounded_rate"] is not None
                 and v4[b]["ungrounded_rate"] is not None]),
            "delta_token_f1": st.mean([i5[b]["token_f1"] - v4[b]["token_f1"] for b in both
                                       if i5[b]["token_f1"] is not None
                                       and v4[b]["token_f1"] is not None]),
            "delta_entity_f1": st.mean([i5[b]["entity_f1"] - v4[b]["entity_f1"] for b in both
                                        if i5[b]["entity_f1"] is not None
                                        and v4[b]["entity_f1"] is not None]),
        }

    out = {"model": a.model, "reference_variant": a.reference,
           "ontology_version": kb.version, "overall": overall,
           "by_category": bycat, "significance_vs_reference": sig,
           "validation_layer": v45}
    (D / f"variant_scores__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    # ── report ────────────────────────────────────────────────────────────
    L = []
    L.append(f"# Task 4.1 - Baseline Comparison of Five System Variants\n")
    L.append(f"Model **{a.model}**, greedy decoding, 347-item six-category tourism "
             f"benchmark. Groundedness audited against ontology v{kb.version} "
             f"({len(kb.surface):,} surface forms, {len(kb.triples):,} triples).\n")
    L.append("## Overall\n")
    L.append("| Variant | n | Token P | Token R | Token F1 | Acc@F1≥.5 | Entity P | "
             "Entity R | Entity F1 | Entity hit | Ungrounded ↓ | (raw) | "
             "Clean answers ↑ |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for v in VARIANTS:
        if v not in overall:
            continue
        o = overall[v]
        L.append(f"| {LABEL[v]} | {o['n']} | {fmt(o['token_p'])} | {fmt(o['token_r'])} | "
                 f"**{fmt(o['token_f1'])}** | {fmt(o['acc_token_f1_50'])} | "
                 f"{fmt(o['entity_p'])} | {fmt(o['entity_r'])} | **{fmt(o['entity_f1'])}** | "
                 f"{fmt(o['entity_hit'])} | **{fmt(o['ungrounded_rate'])}** | "
                 f"{fmt(o['ungrounded_rate_raw'])} | {fmt(o['clean'])} |")
    L.append("\nAll values are percentages, macro-averaged over items. *Ungrounded* is "
             "the share of named entities in the answer that ontology v"
             f"{kb.version} cannot account for (lower is better), after filtering out "
             "sentence-initial imperatives, month names and markdown section labels, "
             "which English capitalises but which are not claims about the world; the "
             "*(raw)* column is the same quantity as the Task 3.2.2 validator's own "
             "recall-oriented detector sees it, and the gap between the two is "
             "discussed in the hallucination report. *Clean answers* is the share of "
             "answers with at least one entity and no ungrounded one.\n")

    L.append("## Cost\n")
    L.append("| Variant | Prompt tokens | Answer chars | Generation s/item | "
             "Validation ms/item | Repair rounds | Fallback rate |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for v in VARIANTS:
        if v not in overall:
            continue
        o = overall[v]
        L.append(f"| {LABEL[v]} | {fmt(o['prompt_tokens'], False, 0)} | "
                 f"{fmt(o['answer_chars'], False, 0)} | {fmt(o['generation_s'], False, 2)} | "
                 f"{fmt(o['validation_ms'], False, 0)} | "
                 f"{fmt(o['repair_rounds'], False, 2)} | {fmt(o['fallback'])} |")

    L.append(f"\n## Significance vs {LABEL[a.reference]}\n")
    L.append("Two-sided Wilcoxon signed-rank on per-item values; Δ is the mean "
             "difference in percentage points.\n")
    L.append("| Variant | Δ Token F1 | p | Δ Entity F1 | p | Δ Ungrounded | p | "
             "Δ Clean | p |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for v in VARIANTS:
        if v not in sig:
            continue
        s = sig[v]
        cells = []
        for f in ("token_f1", "entity_f1", "ungrounded_rate", "clean"):
            d, p = s[f]["delta"], s[f]["p"]
            dd = "—" if d is None else f"{d*100:+.1f}"
            pp = "—" if p is None else (f"**{p:.2e}**" if p < .05 else f"{p:.3f}")
            cells += [dd, pp]
        L.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")

    for metric, title in (("token_f1", "Token F1"), ("entity_f1", "Entity F1"),
                          ("ungrounded_rate", "Ungrounded entity rate (lower better)")):
        L.append(f"\n## Per category - {title}\n")
        L.append("| Category | n | " + " | ".join(LABEL[v] for v in VARIANTS
                                                  if v in overall) + " |")
        L.append("|---|--:|" + "--:|" * len([v for v in VARIANTS if v in overall]))
        for c in CATEGORIES:
            n = bycat[a.reference][c]["n"]
            row = [fmt(bycat[v][c][metric]) for v in VARIANTS if v in overall]
            best = min if metric == "ungrounded_rate" else max
            vals = [bycat[v][c][metric] for v in VARIANTS if v in overall]
            ok = [x for x in vals if x is not None]
            if ok:
                bi = vals.index(best(ok))
                row[bi] = f"**{row[bi]}**"
            L.append(f"| {c} | {n} | " + " | ".join(row) + " |")

    if v45:
        L.append("\n## What the validation layer (V4 → V5) changed\n")
        L.append(f"- answers rewritten: **{v45['answers_rewritten']} / {v45['n']}**")
        L.append(f"- violations detected per item: {v45['violations_per_item']:.2f} "
                 f"({', '.join(f'{k} {n}' for k, n in sorted(v45['violation_types'].items(), key=lambda x: -x[1]))})")
        L.append(f"- uncertainty fallback fired on {v45['fallback_rate']*100:.1f}% of items; "
                 f"mean {v45['mean_repair_rounds']:.2f} repair rounds; "
                 f"{v45['mean_validation_ms']:.0f} ms validation overhead per item")
        L.append(f"- Δ ungrounded entity rate {v45['delta_ungrounded_rate']*100:+.2f} pp, "
                 f"Δ token F1 {v45['delta_token_f1']*100:+.2f} pp, "
                 f"Δ entity F1 {v45['delta_entity_f1']*100:+.2f} pp")

    p = D / f"baseline_comparison__{a.model}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")

    with open(D / f"variant_scores__{a.model}.csv", "w", encoding="utf-8") as f:
        cols = ["variant", "scope", "n"] + FIELDS + ["acc_token_f1_50", "acc_entity_f1_50"]
        f.write(",".join(cols) + "\n")
        for v in VARIANTS:
            if v not in overall:
                continue
            for scope, o in [("overall", overall[v])] + [(c, bycat[v][c]) for c in CATEGORIES]:
                f.write(",".join([v, scope, str(o["n"])] +
                                 ["" if o.get(c) is None else f"{o[c]:.6f}"
                                  for c in cols[3:]]) + "\n")
    print("\n" + "\n".join(L[:14]))
    print(f"\nwrote {p.name}, variant_scores__{a.model}.{{json,csv}}")


if __name__ == "__main__":
    main()
