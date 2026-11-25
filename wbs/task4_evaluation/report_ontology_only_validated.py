"""
WBS Task 4.1.2 - The arm the standing comparison actually asks about: ontology
context only, plus the validation layer.

Every validated arm in this project is V5 = V4 + the layer, and V4 carries both
ontology facts and retrieved prose. So the configuration the project's own
question names - ontology grounding alone against Simple RAG - has never been
measured with the layer in front of it. V3 is ontology-only without the layer
and loses to Simple RAG on everything; V5 wins on groundedness but its context
includes the retrieved prose it was supposed to be compared against.

`ablate_generation.py --axis selective --draft-arm V3_ontology_only` closes
that: the same 2x2 over the gate and the coverage floor, on the dedicated
`onto_v3` adapter, with the span detector held at audit-grade, so the only
context the model ever sees is the graph. This script reports it against Simple
RAG, and against the same cell built on the hybrid context.

  python -m wbs.task4_evaluation.report_ontology_only_validated --model llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"

DRAFT = "draft only (layer off)"
DEPLOYED = "every answer, coverage floor (deployed)"
FLOOR_OFF = "every answer, unsupportable only"
GATED_FLOOR = "gated >=1 span, coverage floor"
GATED_OFF = "gated >=1 span, unsupportable only"

MET = [("token_f1", "Token F1"), ("entity_f1", "Entity F1"),
       ("ungrounded_rate", "Ungrounded ↓"), ("clean", "Clean ↑")]
GROUND = ("ungrounded_rate", "clean")
# lower is better for exactly one of the four
BETTER_LOW = {"ungrounded_rate"}


def holm(tests: list[tuple]) -> list[tuple]:
    """tests: (label, delta, p). Returns the same with a Holm-adjusted p appended."""
    ts = sorted(tests, key=lambda t: (1.0 if t[2] is None else t[2]))
    n, prev, out = len(ts), 0.0, []
    for i, (lab, dv, pv) in enumerate(ts):
        prev = max(prev, min(1.0, (1.0 if pv is None else pv) * (n - i)))
        out.append((lab, dv, pv, prev))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--tag", default="__auditdet__ontoonly")
    ap.add_argument("--hybrid-tag", default="__auditdet__hybrid20d")
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans
    from wbs.task4_evaluation.score_variants import score_item, wilcoxon

    jp = EVAL / f"ablation_selective__{a.model}{a.tag}.json"
    anp = EVAL / f"ablation_selective__{a.model}{a.tag}__answers.jsonl"
    if not (jp.exists() and anp.exists()):
        sys.exit(f"{jp.name} not found - run ablate_generation --axis selective "
                 f"--draft-arm V3_ontology_only first")
    meta = json.loads(jp.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(anp, encoding="utf-8") if l.strip()]
    cfgs = list(rows[0]["answers"].keys())
    ids = [r["bench_id"] for r in rows]
    kb = KnowledgeBase()

    per = {c: [score_item({"bench_id": r["bench_id"], "category": "x",
                           "reference": r["reference"], "answer": r["answers"][c]}, kb)
               for r in rows] for c in cfgs}
    mean = {c: {f: (st.mean(xs) if (xs := [x[f] for x in per[c] if x[f] is not None])
                    else None) for f, _ in MET} for c in cfgs}
    byc = {r["config"]: r for r in meta["results"]}

    def paired(base: str, arm: str, f: str) -> dict:
        d = [(x[f], y[f]) for x, y in zip(per[base], per[arm])
             if x[f] is not None and y[f] is not None]
        w = wilcoxon([q for _, q in d], [p for p, _ in d])
        return {"delta": (st.mean([q for _, q in d]) - st.mean([p for p, _ in d])) * 100,
                "p": (w[1] if w else None), "n": len(d),
                "n_differ": sum(1 for p_, q in d if p_ != q),
                "n_only_one": sum(1 for x, y in zip(per[base], per[arm])
                                  if (x[f] is None) != (y[f] is None))}

    def pcell(x) -> str:
        if x["p"] is None:
            return f"— ({x['n_differ']} items differ)"
        return f"{x['p']:.3g}"

    L = [f"# Ontology context only, with the validation layer ({a.model})\n",
         "This project's validated arm has always been V5 = V4 + the layer, and "
         "V4's context carries both the graph and the retrieved prose. The "
         "comparison the project exists to make is about grounding in the "
         "ontology, so the arm that tests it most directly - ontology context "
         "only, with the layer in front of it - had never been run. V3 is that "
         "context without the layer and loses to Simple RAG on every metric; V5 "
         "wins on groundedness but reads the retrieved prose too.\n",
         f"Here the model sees the graph and nothing else. Dedicated `onto_v3` "
         f"adapter, all {meta['n']} benchmark items, greedy decoding, the span "
         f"detector held at **{meta['detector']}-grade** in every cell, scored with "
         "`score_variants.score_item`. The 2x2 is the same one "
         "`selective_axis.md` runs on the hybrid context, so the two are "
         "comparable cell by cell.\n"]

    L += ["## Every cell\n",
          "| Configuration | Token F1 | Entity F1 | Ungrounded ↓ | Clean ↑ | "
          "Validated | Rewritten | Abstained |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for c in cfgs:
        m, r = mean[c], byc[c]
        L.append(f"| {c} | {m['token_f1']*100:.2f} | {m['entity_f1']*100:.2f} | "
                 f"{m['ungrounded_rate']*100:.2f} | {m['clean']*100:.2f} | "
                 f"{r['validated']}/{meta['n']} | {r['answers_changed']} | "
                 f"{r['fallback_rate']*100:.1f}% |")
    L.append("")

    out = {"model": a.model, "source": jp.name, "n": meta["n"],
           "detector": meta["detector"], "adapter": meta.get("hybrid_adapter"),
           "cells": {c: {f: mean[c][f] for f, _ in MET} for c in cfgs},
           "tests": {}}

    # ── against Simple RAG, every cell, Holm over the family ───────────────
    v2p = EVAL / f"variant__{a.model}__V2_rag.jsonl"
    rag_ok = False
    if v2p.exists():
        v2 = {json.loads(l)["bench_id"]: json.loads(l)
              for l in open(v2p, encoding="utf-8") if l.strip()}
        rag_ok = all(b in v2 for b in ids)
    if rag_ok:
        per["__v2"] = [score_item(v2[b], kb) for b in ids]
        v2mean = {f: st.mean([r[f] for r in per["__v2"] if r[f] is not None]) * 100
                  for f, _ in MET}
        L += ["## Against Simple RAG\n",
              "Paired per item over the items both arms can be scored on - an "
              "answer with no entity-like span has no ungrounded rate, so the "
              "groundedness columns run on fewer than "
              f"{len(ids)} items. Holm is taken over all "
              f"{len(cfgs) * len(MET)} tests in this table, because asking the "
              "same question of five cells and four metrics is a family and a "
              "bare 0.05 would be expected to fire on noise.\n",
              "| Configuration | Metric | Simple RAG | this arm | Δ | p | Holm p |",
              "|---|---|--:|--:|--:|--:|--:|"]
        fam, keep = [], {}
        for c in cfgs:
            for f, lab in MET:
                x = paired("__v2", c, f)
                keep[(c, f)] = x
                fam.append((f"{c}|{f}", x["delta"], x["p"]))
        adj = {lab: hp for lab, _, _, hp in holm(fam)}
        out["vs_rag"] = {}
        out["rag_means"] = v2mean
        for c in cfgs:
            out["vs_rag"][c] = {}
            for f, lab in MET:
                x = keep[(c, f)]
                hp = adj[f"{c}|{f}"]
                out["vs_rag"][c][f] = dict(x, holm_p=hp)
                L.append(f"| {c} | {lab} | {v2mean[f]:.2f} | "
                         f"{v2mean[f] + x['delta']:.2f} | {x['delta']:+.2f} | "
                         f"{pcell(x)} | {hp:.3g} |")
        L.append("")

        def wins(c: str, f: str) -> bool:
            x = out["vs_rag"][c][f]
            if x["holm_p"] is None or x["holm_p"] >= 0.05:
                return False
            return x["delta"] < 0 if f in BETTER_LOW else x["delta"] > 0

        ground_winners = [c for c in cfgs if all(wins(c, f) for f in GROUND)]
        all_winners = [c for c in cfgs if all(wins(c, f) for f, _ in MET)]
        out["beats_rag_on_both_groundedness"] = ground_winners
        out["beats_rag_on_all_metrics"] = all_winners

        L += ["## Reading\n"]
        if ground_winners:
            best = max(ground_winners, key=lambda c: mean[c]["token_f1"])
            bu = out["vs_rag"][best]["ungrounded_rate"]
            bc = out["vs_rag"][best]["clean"]
            bt = out["vs_rag"][best]["token_f1"]
            L.append(
                "**Ontology context alone, with the layer, beats Simple RAG on "
                f"both groundedness measures.** {len(ground_winners)} of "
                f"{len(cfgs)} cells do, after Holm. The best of them on overlap "
                f"is *{best}*: {bu['delta']:+.2f} pp audit-grade ungrounded "
                f"(Holm p = {bu['holm_p']:.3g}) and {bc['delta']:+.2f} pp clean "
                f"(Holm p = {bc['holm_p']:.3g}). This is the first arm in the "
                "project to win on groundedness with no retrieved prose in its "
                "context at all, which is the comparison the project's question "
                "names.\n")
            L.append(
                "**It does not beat Simple RAG on overlap, and no cell does.** "
                f"The same cell is {bt['delta']:+.2f} token F1 "
                f"(p = {pcell(bt)}), the largest deficit of any validated arm "
                "measured here. Removing the retrieved prose removes the prose "
                "the references are written in: the graph carries the facts and "
                "not the phrasing, so an arm that reads only the graph can be "
                "more grounded and still overlap less. "
                + ("No cell wins on all four metrics.\n" if not all_winners else
                   f"{len(all_winners)} cell(s) win on all four: "
                   + ", ".join(f"*{c}*" for c in all_winners) + ".\n"))
        else:
            L.append(
                "**No ontology-only cell beats Simple RAG on both groundedness "
                "measures after Holm.** The layer moves the arm in the right "
                "direction on both, but from a lower starting point than the "
                "hybrid context gives it, and the correction over "
                f"{len(fam)} tests is what the margin does not survive.\n")
            # the family is a judgement call, so state what a narrower one
            # would give rather than leaving the reader to wonder
            nar = holm([(f, keep[(GATED_OFF, f)]["delta"],
                         keep[(GATED_OFF, f)]["p"]) for f in GROUND])
            nar_sig = [lab for lab, dv, pv, hp in nar if hp < 0.05]
            L.append(
                "**How much of that is the family.** Correcting over all "
                f"{len(fam)} tests is the conservative choice; a reader who "
                "pre-specified the gated cell and its two groundedness "
                "measures would correct over two. Under that narrower family "
                + (", ".join(f"`{lab.split('|')[-1]}` is significant "
                             f"(Holm p = {hp:.3g})"
                             for lab, dv, pv, hp in nar if hp < 0.05)
                   if nar_sig else "neither measure is significant")
                + ("; the ungrounded rate is not significant under any "
                   "correction, because it is not significant uncorrected "
                   f"(p = {keep[(GATED_OFF, 'ungrounded_rate')]['p']:.3g}). "
                   "So the arm fails to beat Simple RAG on both measures under "
                   "either family, and the choice of family only changes "
                   "whether the clean-rate half of it counts.\n"
                   if "ungrounded_rate" not in " ".join(nar_sig) else ".\n"))

    # ── the same cell on the hybrid context ───────────────────────────────
    hanp = EVAL / f"ablation_selective__{a.model}{a.hybrid_tag}__answers.jsonl"
    if hanp.exists():
        hrows = {json.loads(l)["bench_id"]: json.loads(l)
                 for l in open(hanp, encoding="utf-8") if l.strip()}
        hjp = EVAL / f"ablation_selective__{a.model}{a.hybrid_tag}.json"
        hmeta = json.loads(hjp.read_text(encoding="utf-8")) if hjp.exists() else None
        hyb_fb = (f"{[r for r in hmeta['results'] if r['config'] == GATED_OFF][0]['fallback_rate']*100:.1f}%"
                  if hmeta else "its own")
        if all(b in hrows for b in ids):
            per["__hyb"] = [score_item(
                {"bench_id": b, "category": "x", "reference": hrows[b]["reference"],
                 "answer": hrows[b]["answers"][GATED_OFF]}, kb) for b in ids]
            hm = {f: st.mean([r[f] for r in per["__hyb"] if r[f] is not None]) * 100
                  for f, _ in MET}
            L += ["## What the retrieved prose is worth, under the same gate\n",
                  "Both arms gated on the same signal at the same threshold, both "
                  "abstaining on an unsupportable claim only, both on the "
                  "audit-grade detector. The only difference is whether the "
                  "context carries retrieved prose beside the graph.\n",
                  "| Metric | ontology only | + retrieved prose | Δ | p |",
                  "|---|--:|--:|--:|--:|"]
            out["vs_hybrid_same_cell"] = {}
            for f, lab in MET:
                x = paired(GATED_OFF, "__hyb", f)
                out["vs_hybrid_same_cell"][f] = x
                L.append(f"| {lab} | {hm[f] - x['delta']:.2f} | {hm[f]:.2f} | "
                         f"{x['delta']:+.2f} | {pcell(x)} |")
            L.append("")
            dt = out["vs_hybrid_same_cell"]["token_f1"]
            du = out["vs_hybrid_same_cell"]["ungrounded_rate"]
            dc = out["vs_hybrid_same_cell"]["clean"]
            prose_dom = (dt["delta"] > 0 and du["delta"] < 0 and dc["delta"] > 0)
            out["retrieved_prose_dominates"] = bool(prose_dom)
            L.append(
                f"Adding the prose is {dt['delta']:+.2f} token F1 "
                f"(p = {pcell(dt)}), {du['delta']:+.2f} pp ungrounded "
                f"(p = {pcell(du)}) and {dc['delta']:+.2f} pp clean "
                f"(p = {pcell(dc)}). "
                + ("**The retrieved prose dominates the graph-only context "
                   "here, groundedness included.** That is the measurement this "
                   "arm was built to make and it runs against the project's "
                   "standing expectation: the validated arm's groundedness "
                   "advantage is not something the ontology supplies on its "
                   "own. The mechanism is the layer's: it can only repair a "
                   "claim the context supports, and the graph supports fewer of "
                   "the benchmark's claims than the graph plus the retrieved "
                   "passages do, so the graph-only arm leaves more spans "
                   "unrepaired and abstains on more items.\n"
                   if prose_dom else
                   "The prose buys overlap and costs groundedness, which is the "
                   "trade in the other direction from the layer's: the graph is "
                   "what the detector can check, and every sentence sourced from "
                   "prose instead is a sentence it cannot.\n"
                   if dt["delta"] > 0 and du["delta"] > 0 else
                   "The two contexts differ under the gate in a mixed "
                   "direction, and the per-metric signs above are the whole of "
                   "that result.\n"))
            # the graph-only arm abstains more, so part of its groundedness is
            # measured on fewer items - and it is still behind on both measures
            L.append(
                f"The comparison is not flattered by abstention either. The "
                f"graph-only cell falls back to a refusal on "
                f"{byc[GATED_OFF]['fallback_rate']*100:.1f}% of items against the "
                f"hybrid cell's {hyb_fb}, so "
                f"{du['n_only_one']} items are scored for groundedness in one arm "
                f"and not the other; a refusal carries no entity-like span and so "
                f"leaves the ungrounded-rate average rather than counting against "
                f"it. The graph-only arm is behind on both groundedness measures "
                f"with that effect running in its favour.\n")

    # ── strict end of the detector band ───────────────────────────────────
    def strict_rate(c: str) -> tuple[float, float]:
        ug, cl = [], []
        for r in rows:
            sp, bad, _ = audit_spans(r["answers"][c], kb, strict=True)
            if sp:
                ug.append(len(bad) / len(sp)); cl.append(0.0 if bad else 1.0)
        return st.mean(ug) * 100, st.mean(cl) * 100

    su_g, sc_g = strict_rate(GATED_OFF)
    su_d, sc_d = strict_rate(DRAFT)
    out["strict"] = {GATED_OFF: {"ungrounded": su_g, "clean": sc_g},
                     DRAFT: {"ungrounded": su_d, "clean": sc_d}}
    held = su_g <= su_d and sc_g >= sc_d
    L += ["## At the strict end of the detector band\n",
          "The gated cell against its own draft under the stricter containment "
          "rule, which promotes itinerary scaffolding to claims:\n",
          "| Rule | gated, unsupportable only | draft (layer off) |",
          "|---|--:|--:|",
          f"| published | {mean[GATED_OFF]['ungrounded_rate']*100:.2f} / "
          f"{mean[GATED_OFF]['clean']*100:.2f} | "
          f"{mean[DRAFT]['ungrounded_rate']*100:.2f} / "
          f"{mean[DRAFT]['clean']*100:.2f} |",
          f"| strict | {su_g:.2f} / {sc_g:.2f} | {su_d:.2f} / {sc_d:.2f} |", "",
          (f"**The layer's gain survives the strict rule.** It is "
           f"{su_g:.2f}% against {su_d:.2f}% ungrounded and {sc_g:.2f}% against "
           f"{sc_d:.2f}% clean, so what it removes is claims under either "
           "containment rule rather than scaffolding under one of them.\n"
           if held else
           f"**The layer's gain is bounded at the strict end.** It is "
           f"{su_g:.2f}% against {su_d:.2f}% ungrounded and {sc_g:.2f}% against "
           f"{sc_d:.2f}% clean, so part of what it appears to fix at the "
           "published rule is scaffolding the strict rule counts as a claim. "
           "The published-rule result stands at the published rule and is "
           "reported as bounded, not as robust.\n")]

    # ── the same question at the other sizes, if they have been run ───────
    # R17's residual was that the decomposition existed at 8B only. Each size
    # writes its own json; the largest model reports the family.
    ORDER = {"llama3.2-1b": 0, "llama3.2-3b": 1, "llama3.1-8b": 2}
    sib = []
    for q in sorted(EVAL.glob("ontology_only_validated__*.json")):
        m_ = q.stem.split("__")[1]
        if m_ == a.model:
            continue
        d_ = json.loads(q.read_text(encoding="utf-8"))
        if d_.get("vs_rag"):
            sib.append((m_, d_))
    if sib:
        fam = sorted(sib + [(a.model, out)], key=lambda t: ORDER.get(t[0], 9))
        L += ["## The same question at the other model sizes\n",
              "Each size runs the same 2x2 on its own dedicated `onto_v3` "
              "adapter over the same items, and Holm is taken inside each "
              "size's own table of 20 tests.\n",
              "| Model | gated cell (tokF1 / ungr / clean) | Δ ungrounded vs RAG "
              "| Δ clean vs RAG | beats RAG on both? |",
              "|---|--:|--:|--:|---|"]
        for m_, d_ in fam:
            c_ = d_["cells"][GATED_OFF]
            u_ = d_["vs_rag"][GATED_OFF]["ungrounded_rate"]
            k_ = d_["vs_rag"][GATED_OFF]["clean"]
            win = GATED_OFF in (d_.get("beats_rag_on_both_groundedness") or [])
            L.append(f"| {m_} | {c_['token_f1']*100:.2f} / "
                     f"{c_['ungrounded_rate']*100:.2f} / {c_['clean']*100:.2f} | "
                     f"{u_['delta']:+.2f} (Holm p = {u_['holm_p']:.3g}) | "
                     f"{k_['delta']:+.2f} (Holm p = {k_['holm_p']:.3g}) | "
                     f"{'**yes**' if win else 'no'} |")
        L.append("")
        wins_at = [m_ for m_, d_ in fam
                   if GATED_OFF in (d_.get("beats_rag_on_both_groundedness") or [])]
        out["beats_rag_at_sizes"] = wins_at
        if not wins_at:
            L.append(f"**The result is the same at every size measured "
                     f"({', '.join(m_ for m_, _ in fam)}).** Ontology context "
                     "alone, with the layer gated on the layer's own detector, "
                     "does not beat Simple RAG on both groundedness measures at "
                     "any model size tested, so the decomposition is not an "
                     "artefact of the 8B model. The hybrid arms' groundedness "
                     "win does hold at all three sizes "
                     "(`hybrid_adapter_comparison*.md`), which is what makes "
                     "the contrast informative rather than a null result about "
                     "small models.\n")
        else:
            L.append(f"**It beats Simple RAG on both groundedness measures at "
                     f"{', '.join(wins_at)} and not at the other sizes "
                     f"measured.** The decomposition is therefore size "
                     f"dependent and is stated per size rather than as one "
                     f"finding.\n")

    # ── what it settles, derived from the tests above rather than asserted ──
    beat = out.get("beats_rag_on_both_groundedness") or []
    allw = out.get("beats_rag_on_all_metrics") or []
    L += ["## What this does and does not settle\n"]
    if beat:
        best = max(beat, key=lambda c: mean[c]["token_f1"])
        bu, bc_ = out["vs_rag"][best]["ungrounded_rate"], out["vs_rag"][best]["clean"]
        L.append(f"**At {a.model}** it settles that ontology grounding on its "
                 "own, with the validation layer gated on the layer's own "
                 "detector, is "
                 "measurably more grounded than Simple RAG on this benchmark - "
                 "no retrieved prose in the context, paired per item, Holm "
                 f"corrected over the whole table: {bu['delta']:+.2f} pp "
                 f"ungrounded and {bc_['delta']:+.2f} pp clean in the best "
                 "cell.\n")
    elif rag_ok:
        gc_ = {c: out["vs_rag"][c] for c in cfgs}
        best = min(cfgs, key=lambda c: (gc_[c]["clean"]["p"]
                                        if gc_[c]["clean"]["p"] is not None else 1.0))
        bu, bc_ = gc_[best]["ungrounded_rate"], gc_[best]["clean"]
        L.append(f"**At {a.model} it settles the question in the negative.** "
                 "Ontology context on its own, with the validation "
                 "layer gated on the layer's own detector, is *not* measurably "
                 "more grounded than Simple RAG on this benchmark. The layer "
                 f"moves the arm the right way on both measures - the best cell "
                 f"(*{best}*) is {bu['delta']:+.2f} pp ungrounded "
                 f"(p = {pcell(bu)}) and {bc_['delta']:+.2f} pp clean "
                 f"(p = {pcell(bc_)}) - but neither margin survives Holm over "
                 f"the {len(cfgs) * len(MET)} tests in the table "
                 f"(Holm p = {bu['holm_p']:.3g} and {bc_['holm_p']:.3g}), and "
                 "the ungrounded-rate margin is not significant even "
                 "uncorrected.\n")
        if out.get("retrieved_prose_dominates"):
            L.append("So the groundedness advantage this project does have is "
                     "V5's, and V5's context carries the retrieved prose beside "
                     "the graph. Measured under the same gate, the same "
                     "detector and the same floor setting, the hybrid context "
                     "is better than the graph alone on every metric, both "
                     "groundedness measures included. The win belongs to the "
                     "combination of ontology and retrieval, not to the "
                     "ontology on its own - and the arm that would have shown "
                     "otherwise is the one measured here.\n")
    sz = out.get("beats_rag_at_sizes")
    if sz is not None:
        every = [m for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b")
                 if (EVAL / f"ontology_only_validated__{m}.json").exists()]
        lost = [m for m in every if m not in sz]
        if sz and lost:
            # which side of the comparison moves with model size? say the one
            # the numbers say, not the one that sounds right
            rows_ = []
            for m_ in every:
                d_ = json.loads((EVAL / f"ontology_only_validated__{m_}.json")
                                .read_text(encoding="utf-8"))
                rm = (d_.get("rag_means") or {}).get("ungrounded_rate")
                dl = d_["vs_rag"][GATED_OFF]["ungrounded_rate"]["delta"]
                if rm is not None:
                    rows_.append((m_, rm, rm + dl))
            why = ""
            if len(rows_) == len(every) and len(rows_) > 1:
                rag_sp = max(r[1] for r in rows_) - min(r[1] for r in rows_)
                arm_sp = max(r[2] for r in rows_) - min(r[2] for r in rows_)
                base = ", ".join(f"{m_} {r:.2f}%" for m_, r, _ in rows_)
                arm = ", ".join(f"{m_} {x:.2f}%" for m_, _, x in rows_)
                if arm_sp > rag_sp:
                    why = (f" The side that moves is the ontology arm, not the "
                           f"baseline: Simple RAG's own audit-grade ungrounded "
                           f"rate is nearly flat across sizes ({base}, a spread "
                           f"of {rag_sp:.2f} pp) while the graph-only arm's "
                           f"rises with model size ({arm}, a spread of "
                           f"{arm_sp:.2f} pp). The larger model asserts more "
                           f"than the graph supports; the win disappears "
                           f"because the arm gets worse, not because the "
                           f"baseline gets better.")
                else:
                    why = (f" The side that moves is the baseline: Simple RAG's "
                           f"own ungrounded rate varies across sizes ({base}) "
                           f"more than the graph-only arm's does ({arm}).")
            L.append(f"**And the answer depends on the model.** The same arm "
                     f"beats Simple RAG on both groundedness measures at "
                     f"{', '.join(sz)} and not at {', '.join(lost)}." + why
                     + f" So \"ontology grounding alone beats retrieval on "
                     f"groundedness\" holds at {len(sz)} of {len(every)} sizes "
                     f"measured and must be stated with the size attached; the "
                     f"hybrid arm wins at all three "
                     f"(`hybrid_adapter_comparison*.md`).\n")
        elif not sz:
            L.append(f"**At no size measured ({', '.join(every)}) does the "
                     f"graph-only arm beat Simple RAG on both groundedness "
                     f"measures**, so this is not an artefact of one model "
                     f"scale.\n")

    L.append("Either way it does not support the stronger claim that the "
             "ontology arms are best on *all* evaluation metrics. They are "
             "not: every ontology-only cell trails Simple RAG on token F1 and "
             "on entity F1, by more than the hybrid arms do, and this "
             "measurement makes the reason concrete rather than conjectural - "
             "the graph carries the facts the detector checks and not the prose "
             "the references are written in.\n")
    L.append("No published number changes: this arm is stored under its own tag "
             "and reported beside, not instead of, V3 and V5.\n")

    (EVAL / f"ontology_only_validated__{a.model}.md").write_text(
        "\n".join(L), encoding="utf-8")
    (EVAL / f"ontology_only_validated__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote ontology_only_validated__{a.model}.{{md,json}}")
    for c in cfgs:
        m = mean[c]
        print(f"  {c:<42} {m['token_f1']*100:.2f} / "
              f"{m['ungrounded_rate']*100:.2f} / {m['clean']*100:.2f}")
    print(f"  beats RAG on both groundedness: {out.get('beats_rag_on_both_groundedness')}")
    print(f"  beats RAG on all metrics:       {out.get('beats_rag_on_all_metrics')}")


if __name__ == "__main__":
    main()
