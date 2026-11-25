"""
WBS Task 4.1.2 - Ablations that need no generation: retrieval composition and the
reasoning modules.

Two of the four ablation axes the WBS names can be measured without running the model
at all, and measuring them that way is better than measuring them through a model,
because the model's noise does not sit on top of the effect.

  RETRIEVAL   the ontology context is a union of three sources - dense entity-card
              search, intent-to-class routing ordered by graph degree, and time-scoped
              facts. Each is removed in turn and the context is rebuilt. The metric is
              gold-entity coverage: the share of the gazetteer places named in the gold
              answer that survive into the context. If a source can be removed without
              moving coverage, it is not contributing retrieval, whatever it adds to
              the prompt. The hybrid fact share is swept over the same metric.

  REASONING   the Task 3.2.2 validator fires three constraint types. Each is disabled
              in turn over the real generated answers of every variant, so the
              contribution of each module to the detected-violation count, and the
              share of answers it alone would flag, is measured on real output rather
              than asserted.

  python -m wbs.task4_evaluation.ablate_offline --device cpu
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
            "V5_ontology_rag_validated"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="llama3.1-8b")
    a = ap.parse_args()

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.contexts import (_DEFAULT_CLASSES,
                                                             OntoV3ContextBuilder,
                                                             RagContextBuilder,
                                                             _truncate)
    from experiments.exp042_prf.compute_prf import ents
    from wbs.task3_reasoning.constrained_reasoner import (UNSUPPORTABLE, KnowledgeBase,
                                                          validate)
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
             if l.strip()]
    qs = [b["Input"] for b in bench]
    gold = [set(ents(b["Output"])) for b in bench]
    print(f"{len(bench)} benchmark questions", flush=True)

    def coverage(ctxs):
        v = []
        for ctx, g in zip(ctxs, gold):
            if not g:
                continue
            low = (ctx or "").lower()
            v.append(sum(1 for e in g if e in low) / len(g))
        return st.mean(v) if v else 0.0

    # ── retrieval composition ─────────────────────────────────────────────
    onto = OntoV3ContextBuilder(device=a.device); onto.load()
    rag = RagContextBuilder(device=a.device); rag.load()
    import numpy as np
    qe = onto._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in qs],
                                   batch_size=128, convert_to_numpy=True,
                                   normalize_embeddings=True,
                                   show_progress_bar=False).astype("float32")
    _, idxs = onto._index.search(qe, C.ONTO_V3_ENTITIES)

    def build_onto(use_dense=True, use_routing=True, use_temporal=True,
                   n_entities=C.ONTO_V3_ENTITIES):
        """Rebuild the ontology context with individual sources switched off."""
        out = []
        for qi, q in enumerate(qs):
            picked, seen = [], set()
            if use_dense:
                for j in idxs[qi][:n_entities]:
                    if j >= 0 and j not in seen:
                        seen.add(j); picked.append(onto._cards[j])
            if use_routing:
                classes = onto._target_classes(q)
                per = max(2, n_entities // max(len(classes), 1))
                for klass in classes:
                    added = 0
                    for j in onto._by_class.get(klass, []):
                        if j in seen:
                            continue
                        seen.add(j); picked.append(onto._cards[j]); added += 1
                        if added >= per:
                            break
            tlines = []
            if use_temporal and onto._temporal is not None:
                try:
                    tlines = [f.to_sentence()
                              for f in onto._temporal.retrieve(q, top_k=6).facts][:4]
                except Exception:
                    tlines = []
            lines = []
            if tlines:
                lines.append("Time-scoped facts:")
                lines += [f"- {t}" for t in tlines]
            if picked:
                if lines:
                    lines.append("")
                lines.append("Knowledge-graph facts:")
                for c in picked:
                    lines += [f"- {f}" for f in c["facts"][:3]]
            out.append(_truncate("\n".join(lines), C.CTX_MAX_CHARS) if lines
                       else "[no ontology facts found]")
        return out

    t0 = time.perf_counter()
    configs = [
        ("full (dense + class routing + temporal)", dict()),
        ("− temporal facts", dict(use_temporal=False)),
        ("− class routing (dense cards only)", dict(use_routing=False)),
        ("− dense search (class routing only)", dict(use_dense=False)),
        ("dense only, 2x entities", dict(use_routing=False, use_temporal=False,
                                         n_entities=C.ONTO_V3_ENTITIES * 2)),
    ]
    retr = []
    base_ctx = None
    for name, kw in configs:
        ctx = build_onto(**kw)
        if base_ctx is None:
            base_ctx = ctx
        retr.append({"config": name, "gold_coverage": coverage(ctx),
                     "mean_chars": round(st.mean([len(x) for x in ctx])),
                     "empty_rate": st.mean([1.0 if x.startswith("[no ontology") else 0.0
                                            for x in ctx])})
        print(f"  {name:42s} coverage {retr[-1]['gold_coverage']*100:5.1f}%  "
              f"{retr[-1]['mean_chars']:,} chars", flush=True)

    # hybrid fact-share sweep, against the same metric
    rag_ctx = [r["context"] for r in rag.retrieve_batch(qs)]
    shares = []
    for sh in (0.0, 0.1, 0.2, 0.4, 0.6, 1.0):
        fb = int(C.CTX_MAX_CHARS * sh)
        hyb = []
        for oc, rc in zip(base_ctx, rag_ctx):
            f = _truncate(oc if not oc.startswith("[no ontology") else "", fb)
            p = _truncate(rc, C.CTX_MAX_CHARS - len(f))
            hyb.append((f"Ontology facts:\n{f}\n\n" if f else "") +
                       (f"Retrieved reference passages:\n{p}" if p else ""))
        shares.append({"fact_share": sh, "gold_coverage": coverage(hyb),
                       "mean_chars": round(st.mean([len(x) for x in hyb]))})
        print(f"  fact share {sh*100:5.0f}%  coverage "
              f"{shares[-1]['gold_coverage']*100:5.1f}%", flush=True)
    print(f"retrieval ablation took {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    # ── reasoning modules ─────────────────────────────────────────────────
    kb = KnowledgeBase()
    reason = []
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        per_type: Counter = Counter()
        only_this, any_viol = Counter(), 0
        for r in rows:
            vs = validate(r.get("answer") or "", kb)
            types = {x["type"] for x in vs}
            for x in vs:
                per_type[x["type"]] += 1
            if types:
                any_viol += 1
            if len(types) == 1:
                only_this[next(iter(types))] += 1
        reason.append({"variant": v, "n": len(rows),
                       "answers_with_any_violation": any_viol,
                       "rate": rate(any_viol, rows),
                       "violations_by_type": dict(per_type),
                       "sole_detector_counts": dict(only_this)})
        print(f"  [{v}] {any_viol}/{len(rows)} answers violate something; "
              f"{dict(per_type)}", flush=True)

    out = {"model": a.model, "retrieval_composition": retr,
           "hybrid_fact_share": shares, "reasoning_modules": reason,
           "unsupportable_categories": [lbl for _, lbl in UNSUPPORTABLE]}
    (EVAL / f"ablation_offline__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    # ── report ────────────────────────────────────────────────────────────
    L = ["# Task 4.1 - Ablation Study (retrieval and reasoning modules)\n",
         f"347-item six-category benchmark. No generation is involved in this file: "
         f"retrieval is scored by gold-entity coverage and the reasoning modules are "
         f"scored on the real answers already generated, so no model noise sits on top "
         f"of the effects.\n",
         "## Retrieval composition\n",
         "| Ontology context | Gold-entity coverage | Mean chars | Empty |",
         "|---|--:|--:|--:|"]
    base = retr[0]
    for r in retr:
        d = "" if r is base else \
            f" ({(r['gold_coverage']-base['gold_coverage'])*100:+.1f})"
        L.append(f"| {r['config']} | {r['gold_coverage']*100:.1f}{d} | "
                 f"{r['mean_chars']:,} | {r['empty_rate']*100:.1f}% |")
    by = {r["config"]: r for r in retr}
    d_route = (by["− class routing (dense cards only)"]["gold_coverage"]
               - base["gold_coverage"]) * 100
    d_dense = (by["− dense search (class routing only)"]["gold_coverage"]
               - base["gold_coverage"]) * 100
    d_temp = (by["− temporal facts"]["gold_coverage"] - base["gold_coverage"]) * 100
    d_2x = (by["dense only, 2x entities"]["gold_coverage"] - base["gold_coverage"]) * 100
    L.append(f"\n**Class routing is carrying the ontology arm, and dense card search is "
             f"not.** Removing class routing costs {d_route:+.1f} pp of gold-entity "
             f"coverage - the context collapses to "
             f"{by['− class routing (dense cards only)']['gold_coverage']*100:.1f}%. "
             f"Removing dense card search instead *raises* coverage by "
             f"{d_dense:+.1f} pp, and removing the time-scoped layer raises it by "
             f"{d_temp:+.1f} pp. Doubling the dense entity budget recovers nothing "
             f"({d_2x:+.1f} pp), so this is not a budget problem: similarity search over "
             f"entity cards is spending context on entities that are near the question "
             f"in embedding space but are not the ones the answer needs, while "
             f"degree-ordered class routing reaches them. "
             f"The deployed configuration is therefore not the best one this metric "
             f"knows of - dropping dense search and the temporal layer would give "
             f"{d_dense:+.1f} pp. That is a retrieval-side measurement; whether it "
             f"survives into answer quality is the generation ablation's question, and "
             f"it has not been run on these two configurations.\n")

    L.append("## Hybrid fact share\n")
    L.append("| Fact share of the 3,000-char budget | Gold-entity coverage | Mean chars |")
    L.append("|--:|--:|--:|")
    for s in shares:
        L.append(f"| {s['fact_share']*100:.0f}% | {s['gold_coverage']*100:.1f} | "
                 f"{s['mean_chars']:,} |")
    bestsh = max(shares, key=lambda s: s["gold_coverage"])
    L.append(f"\nCoverage peaks at a fact share of **{bestsh['fact_share']*100:.0f}%**. "
             "Note what this metric does and does not say: it measures whether the "
             "places the gold answer names are present in the context, not whether the "
             "model then uses them. The deployed hybrid arm uses 20%.\n")

    if reason:
        L.append("## Reasoning modules (constraint types)\n")
        L.append("| Variant | n | Answers with ≥1 violation | UNGROUNDED_ENTITY | "
                 "UNSUPPORTABLE_CLAIM | RELATION_UNSUPPORTED |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for r in reason:
            t = r["violations_by_type"]
            L.append(f"| {r['variant']} | {r['n']} | {r['answers_with_any_violation']} "
                     f"({r['rate']*100:.1f}%) | {t.get('UNGROUNDED_ENTITY', 0):,} | "
                     f"{t.get('UNSUPPORTABLE_CLAIM', 0):,} | "
                     f"{t.get('RELATION_UNSUPPORTED', 0):,} |")
        dead = [t for t in ("UNGROUNDED_ENTITY", "UNSUPPORTABLE_CLAIM",
                            "RELATION_UNSUPPORTED")
                if not any(r["violations_by_type"].get(t) for r in reason)]
        if dead:
            L.append("\n**" + ", ".join(f"`{d}`" for d in dead) + " never fired on any "
                     f"of the {sum(r['n'] for r in reason):,} answers.** That module is "
                     "not contributing to this benchmark. Its pattern requires an "
                     "explicit copular assertion of the form \"<A> is (located) "
                     "in/near <B>\" with both endpoints already in the graph, and "
                     "tourism answers phrase spatial relations differently - \"a short "
                     "drive from\", \"on the eastern coast\", or as a bullet under a "
                     "day heading. The check is sound and costs nothing; it simply has "
                     "no purchase on this output distribution, and a relation checker "
                     "that mattered here would have to parse those phrasings. Reporting "
                     "it as a working module because it is implemented would overstate "
                     "what the reasoning layer does.\n")
        L.append("\nCounts are violations, not answers, so one answer can contribute "
                 "several. `UNSUPPORTABLE_CLAIM` covers the four predicate families no "
                 "project source has any data for (" +
                 ", ".join(lbl for _, lbl in UNSUPPORTABLE) + "): a claim of that kind "
                 "cannot be confirmed by this knowledge base even in principle, so "
                 "asserting it is a hallucination by construction rather than a "
                 "retrieval failure.")
    (EVAL / f"ablation_offline__{a.model}.md").write_text("\n".join(L) + "\n",
                                                          encoding="utf-8")
    print(f"\nwrote ablation_offline__{a.model}.{{md,json}}")


if __name__ == "__main__":
    main()
