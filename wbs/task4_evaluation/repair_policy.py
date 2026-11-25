"""
WBS Task 3.2.2 / 4.1.2 - R19: what the validation layer costs is the rewrite,
not the check.

The deployed validation layer repairs a draft by re-prompting the model with the
offending spans and taking the rewrite. `selective_axis.md` measured where its
reference-overlap cost sits: at llama3.1-8b the gated cell is 2.79 token F1
below its own draft, and 2.02 of that is rewrites against 0.77 abstentions. A
rewrite re-words 400 tokens of text that nothing was wrong with in order to
remove one unsupported place name, so the cost is structural rather than a
tuning problem.

`constrained_reasoner.surgical_repair` is the alternative: cut out exactly the
clause that carries the violation, copy every other byte of the draft through
unchanged, and keep the shortened sentence only while it is itself clean and
still names something the knowledge base knows. It asserts strictly less than
the draft did - nothing is added, reworded or hedged - and it needs no second
generation, so the layer stops costing a forward pass as well.

This script measures the policy against Simple RAG and against the layer as
published, on all 347 benchmark items, at every model size, scored with
`score_variants.score_item` - the scorer every headline number in this project
uses. It needs no GPU: every cell is a transformation of generations already on
disk.

  python -m wbs.task4_evaluation.repair_policy
  python -m wbs.task4_evaluation.repair_policy --models llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
MET = ["token_f1", "entity_f1", "entity_hit", "ungrounded_rate", "clean"]
REF = "V2 Simple RAG"
GATED_REGEN = "gated >=1 span, unsupportable only"

# (label, draft arm file, repair mode, granularity, min_violations, min_coverage)
CELLS = [
    ("V4 draft, layer off",              "V4_ontology_rag__hybded", None, None, 0, 0.0),
    ("V5 surgical segment, gated",       "V4_ontology_rag__hybded", "surgical", "segment", 1, 0.0),
    ("V5 surgical clause, gated",        "V4_ontology_rag__hybded", "surgical", "clause", 1, 0.0),
    ("V5 surgical clause, every answer", "V4_ontology_rag__hybded", "surgical", "clause", 0, 0.0),
    ("V5 surgical clause, gated, floor on", "V4_ontology_rag__hybded", "surgical", "clause", 1, 0.6),
    ("V6 surgical clause on the RAG draft, gated", "V2_rag", "surgical", "clause", 1, 0.0),
]


def holm(pairs: list[tuple[str, float | None]]) -> dict[str, float | None]:
    """Holm-Bonferroni over the metrics compared in one cell."""
    live = sorted([(k, p) for k, p in pairs if p is not None], key=lambda x: x[1])
    out = {k: None for k, _ in pairs}
    n, run = len(live), 0.0
    for i, (k, p) in enumerate(live):
        run = max(run, min(1.0, p * (n - i)))
        out[k] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase, constrained_answer
    from wbs.task4_evaluation.hallucination_report import audit_spans, common_words
    from wbs.task4_evaluation.score_variants import score_item, wilcoxon

    kb = KnowledgeBase()
    items = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]

    def variant(model: str, name: str) -> list[str] | None:
        p = EVAL / f"variant__{model}__{name}.jsonl"
        if not p.exists():
            return None
        by = {json.loads(l)["bench_id"]: json.loads(l)
              for l in open(p, encoding="utf-8") if l.strip()}
        if any(b["bench_id"] not in by for b in items):
            return None
        return [by[b["bench_id"]]["answer"] for b in items]

    def per_item(answers: list[str]) -> list[dict]:
        return [score_item({"bench_id": b["bench_id"], "category": b["category"],
                            "reference": b["Output"], "answer": ans}, kb)
                for ans, b in zip(answers, items)]

    for model in a.models:
        rag = variant(model, "V2_rag")
        if rag is None:
            print(f"{model}: no V2_rag generations, skipped"); continue
        cells: dict[str, dict] = {REF: {"answers": rag, "extra": {}}}

        # the layer as published, and as `selective_axis.md` reports it at its best
        for label, name in [("V5 regen, every answer, floor on (published)",
                             "V5_ontology_rag_validated__hybded")]:
            v = variant(model, name)
            if v is not None:
                cells[label] = {"answers": v, "extra": {}}
        sp = EVAL / f"ablation_selective__{model}__auditdet__hybrid20d__answers.jsonl"
        if sp.exists():
            rows = {json.loads(l)["bench_id"]: json.loads(l)
                    for l in open(sp, encoding="utf-8") if l.strip()}
            if all(b["bench_id"] in rows for b in items):
                cells["V5 regen, gated (selective_axis best)"] = {
                    "answers": [rows[b["bench_id"]]["answers"][GATED_REGEN] for b in items],
                    "extra": {}}

        for label, arm, mode, gran, mv, mc in CELLS:
            drafts = variant(model, arm)
            if drafts is None:
                print(f"{model}: no {arm} generations, {label} skipped"); continue
            if mode is None:
                cells[label] = {"answers": list(drafts), "extra": {"draft_arm": arm}}
                continue
            ans, fb, cutseg, cutcl, spent, lat = [], 0, 0, 0, 0, []
            t0 = time.perf_counter()
            for b, d in zip(items, drafts):
                cr = constrained_answer(b["Input"], d, kb, regenerate=None,
                                        min_coverage=mc, min_violations=mv,
                                        audit_grade=True, repair_mode=mode,
                                        repair_granularity=gran)
                ans.append(cr["answer"])
                fb += 1 if cr["uncertainty_fallback"] else 0
                cutcl += sum(1 for c in cr["segments_removed"]
                             if c.get("action") == "clause removed")
                cutseg += sum(1 for c in cr["segments_removed"]
                              if c.get("action") == "segment removed")
                spent += 0 if cr.get("selective_skip") else 1
                lat.append(cr["latency_ms"])
            cells[label] = {"answers": ans, "extra": {
                "draft_arm": arm, "granularity": gran, "min_violations": mv,
                "min_coverage": mc, "validated": spent, "fallbacks": fb,
                "clauses_removed": cutcl, "segments_removed": cutseg,
                "layer_ms_mean": round(st.mean(lat), 2),
                "layer_ms_total": round(sum(lat), 1),
                "wall_clock_s": round(time.perf_counter() - t0, 1),
                "chars_retained_pct": round(st.mean(
                    len(x) / max(len(d), 1) for x, d in zip(ans, drafts)) * 100, 2)}}

        def strict(answers: list[str]) -> dict:
            """The independent stricter detector, which the layer does not optimise.

            The surgical pass removes exactly the spans the published detector
            counts, so its published ungrounded rate is zero by construction and
            says nothing on its own. `audit_spans(strict=True)` is the other end
            of the detector band from `detector_sensitivity.md`: a different rule,
            not a tuning of this one, so a gain that survives it is a gain in the
            answers rather than in the detector.
            """
            rates, cl = [], []
            for x in answers:
                sp, bad, _ = audit_spans(x, kb, strict=True)
                if sp:
                    rates.append(len(bad) / len(sp))
                    cl.append(1.0 if not bad else 0.0)
            return {"ungrounded_rate_strict": round(st.mean(rates) * 100, 4) if rates else None,
                    "clean_strict": round(st.mean(cl) * 100, 4) if cl else None,
                    "n_strict": len(rates)}

        per = {k: per_item(v["answers"]) for k, v in cells.items()}
        out = {"model": model, "n_items": len(items),
               "ontology_version": kb.version, "cells": {}}
        for label, v in cells.items():
            row = {"mean": {}, "n": {}, **v["extra"],
                   "mean_chars": round(st.mean(len(x) for x in v["answers"])),
                   **strict(v["answers"])}
            for f in MET:
                xs = [r[f] for r in per[label] if r[f] is not None]
                row["mean"][f] = round(st.mean(xs) * 100, 4) if xs else None
                row["n"][f] = len(xs)
            if label != REF:
                tests = {}
                for f in MET:
                    d = [(x[f], y[f]) for x, y in zip(per[REF], per[label])
                         if x[f] is not None and y[f] is not None]
                    if not d:
                        tests[f] = None; continue
                    w = wilcoxon([q for _, q in d], [p for p, _ in d])
                    tests[f] = {
                        "delta": round((st.mean([q for _, q in d])
                                        - st.mean([p for p, _ in d])) * 100, 4),
                        "p": (w[1] if w else None), "n": len(d),
                        "n_differ": sum(1 for p_, q in d if p_ != q),
                        "n_only_one": sum(1 for x, y in zip(per[REF], per[label])
                                          if (x[f] is None) != (y[f] is None))}
                hp = holm([(f, (tests[f] or {}).get("p")) for f in MET])
                for f in MET:
                    if tests[f] is not None:
                        tests[f]["holm_p"] = hp[f]
                row["vs_rag"] = tests
            out["cells"][label] = row

        # What the layer removes, by kind. A single lower-case English word that
        # reached the detector capitalised after a dash or a colon is a known
        # false positive of the published span rule (R2), and removing the clause
        # around one is where the residual reference-overlap cost sits, so it is
        # counted rather than described.
        cw = common_words()
        from collections import Counter
        diag = {"n_cut": 0, "single_token": 0, "common_word": 0, "multi_token": 0,
                "unsupportable": 0, "examples": [], "unsupportable_examples": [],
                "top_entity_spans": [], "top_unsupportable_spans": []}
        ent_count: Counter = Counter()
        uns_count: Counter = Counter()
        drafts = variant(model, "V4_ontology_rag__hybded")
        if drafts is not None:
            from wbs.task3_reasoning.constrained_reasoner import surgical_repair
            for b, d in zip(items, drafts):
                _, cut = surgical_repair(d, kb, audit_grade=True)
                for c in cut:
                    diag["n_cut"] += 1
                    for h in c.get("violations", []):
                        sp_ = h["span"]
                        if h["type"] == "UNSUPPORTABLE_CLAIM":
                            diag["unsupportable"] += 1
                            uns_count[sp_] += 1
                            if len(diag["unsupportable_examples"]) < 8:
                                diag["unsupportable_examples"].append(
                                    {"span": sp_, "segment": c["segment"][:160]})
                        elif " " not in sp_:
                            ent_count[sp_] += 1
                            diag["single_token"] += 1
                            if sp_.lower() in cw:
                                diag["common_word"] += 1
                                if len(diag["examples"]) < 12:
                                    diag["examples"].append(
                                        {"span": sp_, "segment": c["segment"][:160]})
                        else:
                            ent_count[sp_] += 1
                            diag["multi_token"] += 1
        diag["top_entity_spans"] = [{"span": s, "n": n, "tokens": len(s.split())}
                                    for s, n in ent_count.most_common(12)]
        diag["top_unsupportable_spans"] = [{"span": s, "n": n}
                                           for s, n in uns_count.most_common(8)]
        out["cut_diagnostics"] = diag

        # the claim the standing goal asks about, checked rather than asserted
        MIN_DIFFER = 20   # a signed-rank test over a handful of pairs is not a decision

        def verdict(label: str) -> dict:
            t = out["cells"][label].get("vs_rag") or {}
            better, worse, tie = [], [], []
            for f in MET:
                x = t.get(f)
                if not x:
                    continue
                good = (x["delta"] < 0) if f == "ungrounded_rate" else (x["delta"] > 0)
                sig = (x.get("holm_p") is not None and x["holm_p"] < 0.05
                       and x["n_differ"] >= MIN_DIFFER)
                (better if (good and sig) else worse if ((not good) and sig)
                 else tie).append(f)
            return {"better_than_rag": better, "worse_than_rag": worse,
                    "indistinguishable": tie, "min_n_differ": MIN_DIFFER,
                    "never_worse_than_rag": not worse}
        out["claim_check"] = {k: verdict(k) for k in out["cells"] if k != REF}

        p = EVAL / f"repair_policy__{model}.json"
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
        for k, v in out["cells"].items():
            mm = v["mean"]
            print(f"  {k:46s} {mm['token_f1']:6.2f} {mm['entity_f1']:6.2f} "
                  f"{mm['ungrounded_rate']:5.2f} {mm['clean']:6.2f}")

        # the answers themselves, so a reader can check any cell by hand
        ap_ = EVAL / f"repair_policy__{model}__answers.jsonl"
        with open(ap_, "w", encoding="utf-8") as fh:
            for i, b in enumerate(items):
                fh.write(json.dumps({
                    "bench_id": b["bench_id"], "category": b["category"],
                    "question": b["Input"], "reference": b["Output"],
                    "answers": {k: v["answers"][i] for k, v in cells.items()}},
                    ensure_ascii=False) + "\n")
        print(f"wrote {ap_.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
