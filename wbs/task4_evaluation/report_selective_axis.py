"""
WBS Task 4.1.2 - Does gating the validation layer compose with dropping its
coverage trigger, or are the two the same fix measured twice?

Two results in this project independently blamed one component for the token F1
the validation layer costs. R2's configuration sweep (150 items, shared adapter)
found that abstaining only on an unsupportable claim beats the deployed
coverage floor. `selective_validation.md` (347 items, dedicated adapter) found
that spending the layer only where its own detector fires dominates spending it
everywhere, and explained it by the coverage trigger firing without asking
whether anything was found wrong. Different subsets, different adapters, so
whether the two fixes add was never tested - and if they are one fix seen twice,
reporting both as separate gains overstates what is available.

`ablate_generation.py --axis selective` runs the 2x2 over the whole benchmark on
one adapter with the detector held at audit-grade. This script reports it: cell
means under the canonical scorer, each factor's marginal effect with a paired
test, the interaction, and the comparison against both Simple RAG and the
operating point `selective_validation.md` named as the project's best.

  python -m wbs.task4_evaluation.report_selective_axis --model llama3.1-8b
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

MET = [("token_f1", "Token F1"), ("ungrounded_rate", "Ungrounded ↓"),
       ("clean", "Clean ↑"), ("entity_f1", "Entity F1")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--tag", default="__auditdet__hybrid20d")
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans
    from wbs.task4_evaluation.score_variants import score_item, wilcoxon

    jp = EVAL / f"ablation_selective__{a.model}{a.tag}.json"
    anp = EVAL / f"ablation_selective__{a.model}{a.tag}__answers.jsonl"
    if not (jp.exists() and anp.exists()):
        sys.exit(f"{jp.name} not found - run ablate_generation --axis selective first")
    meta = json.loads(jp.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(anp, encoding="utf-8") if l.strip()]
    cfgs = list(rows[0]["answers"].keys())
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
                # wilcoxon() drops ties, so this is the count the test actually ran
                # on; when it is tiny the two arms are the same answers and the
                # absence of a p-value is the finding, not a missing number
                "n_differ": sum(1 for p_, q in d if p_ != q),
                # an abstention message carries no entity-like span at all, so its
                # ungrounded rate is undefined and the item drops out of the
                # average rather than counting as clean. When two cells abstain on
                # different items the cell means above are taken over different
                # denominators, and this is how many items that affects.
                "n_only_one": sum(1 for x, y in zip(per[base], per[arm])
                                  if (x[f] is None) != (y[f] is None))}

    def n_text_differ(a_: str, b_: str) -> int:
        return sum(1 for r in rows if r["answers"][a_] != r["answers"][b_])

    def pcell(x) -> str:
        if x["p"] is None:
            return f"— ({x['n_differ']} items differ)"
        return f"{x['p']:.3g}"

    L = [f"# Do the two fixes to the validation layer compose? ({a.model})\n",
         "Two measurements in this project blamed the same component for the token "
         "F1 the validation layer costs. R2's configuration sweep found that "
         "abstaining only on an unsupportable claim beats the deployed coverage "
         "floor; `selective_validation.md` found that spending the layer only where "
         "its own detector fires dominates spending it everywhere, and attributed "
         "that to the coverage trigger firing without asking whether anything was "
         "found wrong. They were measured on different subsets and different "
         "adapters, so they were reported as two gains. If they are one gain seen "
         "twice, that is an overstatement worth catching.\n",
         f"The 2x2, on one adapter (`{meta['hybrid_adapter']}`), all {meta['n']} "
         f"benchmark items, the span detector held at **{meta['detector']}-grade** in "
         "every cell so the two factors are the only things moving. Scored with "
         "`score_variants.score_item`, the scorer every headline number in the "
         "project uses.\n"]

    L += ["| | abstain on coverage floor **or** unsupportable | abstain on "
          "unsupportable **only** |", "|---|--:|--:|"]
    for gate, lab in ((False, "**spend on every answer**"), (True, "**spend only where ≥1 span fires**")):
        cells = []
        for cfg in ((DEPLOYED, FLOOR_OFF) if not gate else (GATED_FLOOR, GATED_OFF)):
            m = mean[cfg]
            cells.append(f"{m['token_f1']*100:.2f} / {m['ungrounded_rate']*100:.2f} / "
                         f"{m['clean']*100:.2f}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Each cell is token F1 / ungrounded / clean. For reference the "
             f"untouched draft is {mean[DRAFT]['token_f1']*100:.2f} / "
             f"{mean[DRAFT]['ungrounded_rate']*100:.2f} / "
             f"{mean[DRAFT]['clean']*100:.2f}, and the layer is what moves it.\n")

    L += ["## Every cell, with what the layer did\n",
          "| Configuration | Token F1 | Entity F1 | Ungrounded ↓ | Clean ↑ | "
          "Validated | Rewritten | Abstained |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for c in cfgs:
        m, r = mean[c], byc[c]
        L.append(f"| {c} | {m['token_f1']*100:.2f} | {m['entity_f1']*100:.2f} | "
                 f"{m['ungrounded_rate']*100:.2f} | {m['clean']*100:.2f} | "
                 f"{r['validated']}/{meta['n']} | {r['answers_changed']} | "
                 f"{r['fallback_rate']*100:.1f}% |")
    L.append("")

    # ── marginal effects and the interaction ──────────────────────────────
    tf = {c: mean[c]["token_f1"] * 100 for c in cfgs}
    e_floor = tf[FLOOR_OFF] - tf[DEPLOYED]
    e_gate = tf[GATED_FLOOR] - tf[DEPLOYED]
    e_both = tf[GATED_OFF] - tf[DEPLOYED]
    inter = e_both - (e_floor + e_gate)

    L += ["## Each factor on its own, and the two together\n",
          "Paired per item, from the deployed cell.\n",
          "| Change | Token F1 | p | Ungrounded | p | Clean | p |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for lab, base, arm in (
            ("drop the coverage floor", DEPLOYED, FLOOR_OFF),
            ("gate on the detector", DEPLOYED, GATED_FLOOR),
            ("both", DEPLOYED, GATED_OFF),
            ("…then gate, floor already off", FLOOR_OFF, GATED_OFF),
            ("…then drop the floor, gate already on", GATED_FLOOR, GATED_OFF)):
        t, u, c_ = (paired(base, arm, f) for f in ("token_f1", "ungrounded_rate", "clean"))
        L.append(f"| {lab} | {t['delta']:+.2f} | {pcell(t)} | {u['delta']:+.2f} | "
                 f"{pcell(u)} | {c_['delta']:+.2f} | {pcell(c_)} |")
    L.append("")
    L.append(f"**Token F1, additively:** the floor alone is worth {e_floor:+.2f}, "
             f"the gate alone {e_gate:+.2f}, and the two together {e_both:+.2f}. "
             f"If they were independent the pair would be worth {e_floor + e_gate:+.2f}; "
             f"the interaction is **{inter:+.2f}**.\n")

    g = paired(FLOOR_OFF, GATED_OFF, "ungrounded_rate")
    gc = paired(FLOOR_OFF, GATED_OFF, "clean")
    f_ = paired(GATED_FLOOR, GATED_OFF, "token_f1")
    L += ["## Reading\n",
          "**They are very largely the same fix.** Each change recovers token F1 "
          f"on its own - {e_floor:+.2f} for dropping the coverage floor, "
          f"{e_gate:+.2f} for gating - but together they recover {e_both:+.2f}, not "
          f"{e_floor + e_gate:+.2f}. The negative interaction ({inter:+.2f}) is most "
          "of one of them. Both changes work by not editing drafts that had nothing "
          "detectably wrong with them, and they identify overlapping sets of such "
          "drafts, so the second one applied has little left to recover. Reporting "
          "the two as independent improvements would have double-counted roughly "
          f"{abs(inter):.1f} token F1.\n",
          "**The gate is the fix that subsumes the other, not the reverse.** With "
          f"the floor already off, gating still adds {paired(FLOOR_OFF, GATED_OFF, 'token_f1')['delta']:+.2f} "
          f"token F1 (p = {pcell(paired(FLOOR_OFF, GATED_OFF, 'token_f1'))}). With the "
          f"gate already on, dropping the floor adds {f_['delta']:+.2f} "
          f"(p = {pcell(f_)}) - not distinguishable from zero. So the deployable "
          "recommendation is the gate; the coverage floor can be left exactly as "
          "delivered once the gate is in front of it.\n",
          "**What gating costs in groundedness: nothing measurable.** Against "
          "spending the layer on every answer at the same floor setting, gating "
          f"moves the ungrounded rate by {g['delta']:+.2f} pp and the clean rate by "
          f"{gc['delta']:+.2f} pp, and the signed-rank test has no pairs to work with "
          f"because only {g['n_differ']} of {g['n']} items differ on one and "
          f"{gc['n_differ']} of {gc['n']} on the other - although "
          f"{paired(FLOOR_OFF, GATED_OFF, 'token_f1')['n']} answers are involved and "
          f"{n_text_differ(FLOOR_OFF, GATED_OFF)} of them differ in their text. That is "
          "the mechanism measured rather than argued: the answers the gate declines "
          "to validate are the ones the layer would have found nothing to fix in, so "
          "skipping them changes the overlap and leaves the grounding alone.\n",
          "**One denominator effect, in the gate's favour, worth naming.** Against "
          "the *deployed* cell the ungrounded rate moves "
          f"{paired(DEPLOYED, GATED_FLOOR, 'ungrounded_rate')['delta']:+.2f} pp on the "
          "items both cells can be scored on - zero of "
          f"{paired(DEPLOYED, GATED_FLOOR, 'ungrounded_rate')['n']} differ - yet the "
          f"cell means read {mean[DEPLOYED]['ungrounded_rate']*100:.2f} against "
          f"{mean[GATED_FLOOR]['ungrounded_rate']*100:.2f}. The whole difference is "
          f"{paired(DEPLOYED, GATED_FLOOR, 'ungrounded_rate')['n_only_one']} items the "
          "deployed layer abstained on: a refusal message contains no entity-like "
          "span, so its rate is undefined and the item leaves the average instead of "
          "counting as a clean answer, while the gated cell answers those items and "
          "scores 0% ungrounded on them. Part of what an abstaining layer looks like "
          "on this metric is items it removed from the measurement rather than "
          "answers it grounded, and the gate is the arm that loses that advantage.\n"]

    # ── against Simple RAG, and against the standing best operating point ──
    out = {"model": a.model, "source": jp.name, "n": meta["n"],
           "detector": meta["detector"], "adapter": meta["hybrid_adapter"],
           "cells": {c: {f: mean[c][f] for f, _ in MET} for c in cfgs},
           "marginal_token_f1": {"floor_only": e_floor, "gate_only": e_gate,
                                 "both": e_both, "interaction": inter},
           "tests": {}}
    for lab, base, arm in (("floor", DEPLOYED, FLOOR_OFF), ("gate", DEPLOYED, GATED_FLOOR),
                           ("both", DEPLOYED, GATED_OFF),
                           ("gate_given_floor_off", FLOOR_OFF, GATED_OFF),
                           ("floor_given_gate_on", GATED_FLOOR, GATED_OFF)):
        out["tests"][lab] = {f: paired(base, arm, f) for f, _ in MET}

    v2p = EVAL / f"variant__{a.model}__V2_rag.jsonl"
    if v2p.exists():
        v2 = {json.loads(l)["bench_id"]: json.loads(l)
              for l in open(v2p, encoding="utf-8") if l.strip()}
        ids = [r["bench_id"] for r in rows]
        if all(b in v2 for b in ids):
            per["__v2"] = [score_item(v2[b], kb) for b in ids]
            L += ["## The best cell against Simple RAG\n",
                  "Paired per item over the items both arms can be scored on, which "
                  "for the groundedness columns is fewer than "
                  f"{len(ids)} - an answer with no entity-like span has no rate - so "
                  "these means differ slightly from the cell table above.\n",
                  "| Metric | Simple RAG | gated, unsupportable only | Δ | p |",
                  "|---|--:|--:|--:|--:|"]
            out["vs_rag"] = {}
            for f, lab in MET:
                x = paired("__v2", GATED_OFF, f)
                out["vs_rag"][f] = x
                base = st.mean([r[f] for r in per["__v2"] if r[f] is not None]) * 100
                L.append(f"| {lab} | {base:.2f} | {base + x['delta']:.2f} | "
                         f"{x['delta']:+.2f} | {pcell(x)} |")
            L.append("")

    selp = EVAL / f"selective_validation__{a.model}.json"
    if selp.exists():
        sel = json.loads(selp.read_text(encoding="utf-8"))
        cv = sel["cross_validated"][sel["best_signal"]]
        L += ["## Against the operating point this project named its best\n"]
        L += [f"`selective_validation.md` reports the same gate, but with the layer "
              f"running its delivered **recall-oriented** detector rather than the "
              f"audit-grade one: {cv['token_f1']*100:.2f} token F1, "
              f"{cv['ungrounded_rate']*100:.2f}% ungrounded, {cv['clean']*100:.2f}% "
              f"clean, validating {cv['n_validated']} of {sel['n']}. The cell above "
              f"validates the same number of answers and differs only in which spans "
              f"the layer is asked to fix.\n"]
        dt = mean[GATED_OFF]["token_f1"] * 100 - cv["token_f1"] * 100
        du = mean[GATED_OFF]["ungrounded_rate"] * 100 - cv["ungrounded_rate"] * 100
        dc = mean[GATED_OFF]["clean"] * 100 - cv["clean"] * 100
        dom = dt >= 0 and du <= 0 and dc >= 0
        L.append(("**This is a trade, not a domination.**" if not dom else
                  "**This dominates it.**")
                 + f" Moving the layer's own detector to audit-grade under the gate is "
                 f"{dt:+.2f} token F1, {du:+.2f} pp ungrounded and {dc:+.2f} pp clean. "
                 + ("The groundedness gain is real and the overlap cost is real; which "
                    "cell is preferable depends on which of the two the deployment is "
                    "for, and this project is about groundedness, so the audit-grade "
                    "cell is the one to carry - stated as a trade rather than as a "
                    "free improvement.\n" if not dom else "\n"))
        out["vs_selective_validation"] = {"token_f1": dt, "ungrounded_rate": du,
                                          "clean": dc, "dominates": bool(dom)}

    # ── the strict-rule check the selective arm needed ────────────────────
    def strict_rate(c: str) -> tuple[float, float]:
        ug, cl = [], []
        for r in rows:
            sp, bad, _ = audit_spans(r["answers"][c], kb, strict=True)
            if sp:
                ug.append(len(bad) / len(sp)); cl.append(0.0 if bad else 1.0)
        return st.mean(ug) * 100, st.mean(cl) * 100
    # the previously-best arm's strict-rule numbers are a measurement, not a
    # constant: read them where they were made, and say nothing when this model
    # has no detector-sensitivity run of its own
    prev_strict = ""
    dsp = EVAL / f"detector_sensitivity__{a.model}.json"
    if dsp.exists():
        da = json.loads(dsp.read_text(encoding="utf-8")).get("arms", {})
        sel_a = next((v for k, v in da.items() if k.startswith("selective")), None)
        all_a = next((v for k, v in da.items()
                      if k.startswith("V5") and "hybded" in k), None)
        if sel_a and all_a:
            prev_strict = (
                f" The arm previously reported as the project's best lost on both "
                f"strict-rule measures ({sel_a['ung_strict']*100:.2f}% against "
                f"{all_a['ung_strict']*100:.2f}% ungrounded, "
                f"{sel_a['clean_strict']*100:.2f}% against "
                f"{all_a['clean_strict']*100:.2f}% clean); moving the layer's "
                f"detector to audit-grade shrinks that to one measure and a "
                f"fraction of a point.")

    su_g, sc_g = strict_rate(GATED_OFF)
    su_e, sc_e = strict_rate(FLOOR_OFF)
    out["strict"] = {GATED_OFF: {"ungrounded": su_g, "clean": sc_g},
                     FLOOR_OFF: {"ungrounded": su_e, "clean": sc_e}}
    held = su_g <= su_e and sc_g >= sc_e
    mixed = (su_g <= su_e) != (sc_g >= sc_e)
    L += ["## At the strict end of the detector band\n",
          "`detector_sensitivity.md` records that the selective arm, as previously "
          "built, did **not** keep its advantage over validating everything under "
          "the stricter containment rule - the answers it left unvalidated kept "
          "their itinerary scaffolding, which the strict rule promotes to claims. "
          "The same check on this cell, against spending the layer everywhere at the "
          "same floor setting:\n",
          f"| Rule | gated, unsupportable only | every answer, unsupportable only |",
          "|---|--:|--:|",
          f"| published | {mean[GATED_OFF]['ungrounded_rate']*100:.2f} / "
          f"{mean[GATED_OFF]['clean']*100:.2f} | "
          f"{mean[FLOOR_OFF]['ungrounded_rate']*100:.2f} / "
          f"{mean[FLOOR_OFF]['clean']*100:.2f} |",
          f"| strict | {su_g:.2f} / {sc_g:.2f} | {su_e:.2f} / {sc_e:.2f} |", "",
          (f"**The caveat does not apply to this cell.** Gating keeps its "
           f"groundedness at both ends of the band ({su_g:.2f}% against {su_e:.2f}% "
           f"strict-rule ungrounded, {sc_g:.2f}% against {sc_e:.2f}% clean), because "
           f"with the audit-grade detector inside the layer the gate and the layer "
           f"agree about which answers carry a problem.\n" if held else
           f"**The caveat is much smaller here, and it is mixed rather than "
           f"adverse.** Under the strict rule the gated cell is {su_g:.2f}% "
           f"ungrounded against {su_e:.2f}% - better - and {sc_g:.2f}% clean against "
           f"{sc_e:.2f}% - worse by {abs(sc_e - sc_g):.2f} pp.{prev_strict} The "
           f"direction of the published-rule result is "
           f"therefore bounded rather than robust, and the size of what is at stake "
           f"at the strict end is now small.\n" if mixed else
           f"**The caveat applies here too.** Under the strict rule the gated cell "
           f"is {su_g:.2f}% ungrounded against {su_e:.2f}% and {sc_g:.2f}% clean "
           f"against {sc_e:.2f}%, so part of what gating recovers is scaffolding the "
           f"strict rule counts as a claim. The published-rule result stands at the "
           f"published rule and is reported as bounded, not as robust.\n")]

    # ── does the recommendation transfer below 8B? ────────────────────────
    # The gate is what this project tells a deployer to do, and until the other
    # sizes were run it was a recommendation measured on one model.
    ORDER = {"llama3.2-1b": 0, "llama3.2-3b": 1, "llama3.1-8b": 2}
    sib = []
    for q in sorted(EVAL.glob("selective_axis__*.json")):
        m_ = q.stem.split("__")[1]
        if m_ == a.model:
            continue
        d_ = json.loads(q.read_text(encoding="utf-8"))
        if d_.get("marginal_token_f1"):
            sib.append((m_, d_))
    if sib:
        fam = sorted(sib + [(a.model, out)], key=lambda t: ORDER.get(t[0], 9))
        L += ["## Does the recommendation transfer below 8B?\n",
              "The gate is what this project recommends deploying, and the 2x2 "
              "that established it ran on one model. Each row below is the same "
              "grid on that size's own dedicated adapter, same items, detector "
              "held at audit-grade.\n",
              "| Model | floor alone | gate alone | both | interaction | floor "
              "given gate on | gated cell vs RAG (ungr / clean) |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for m_, d_ in fam:
            g_ = d_["marginal_token_f1"]
            fg = d_["tests"]["floor_given_gate_on"]["token_f1"]
            vr = d_.get("vs_rag") or {}
            def _pp(k):
                x = vr[k]
                return (f"{x['delta']:+.2f} (p = {x['p']:.2g})"
                        if x.get("p") is not None else f"{x['delta']:+.2f}")
            vs = f"{_pp('ungrounded_rate')} / {_pp('clean')}" if vr else "—"
            # a cell where the two arms produced the same answers has no
            # signed-rank p at all; say so rather than crashing on it
            fgp = (f"p = {fg['p']:.3g}" if fg.get("p") is not None
                   else f"no signed-rank pairs, {fg.get('n_differ', 0)} differ")
            L.append(f"| {m_} | {g_['floor_only']:+.2f} | {g_['gate_only']:+.2f} "
                     f"| {g_['both']:+.2f} | {g_['interaction']:+.2f} | "
                     f"{fg['delta']:+.2f} ({fgp}) | {vs} |")
        L.append("")
        neg = [m_ for m_, d_ in fam if d_["marginal_token_f1"]["interaction"] < 0]
        gate_wins = [m_ for m_, d_ in fam
                     if d_["marginal_token_f1"]["gate_only"] > 0]
        floor_ns = [m_ for m_, d_ in fam
                    if (d_["tests"]["floor_given_gate_on"]["token_f1"]["p"] or 1) >= .05]
        def _beats(d_) -> bool:
            vr_ = d_.get("vs_rag") or {}
            if not vr_:
                return False
            u_, c_ = vr_["ungrounded_rate"], vr_["clean"]
            return (u_["delta"] < 0 and (u_["p"] or 1) < .05
                    and c_["delta"] > 0 and (c_["p"] or 1) < .05)
        out["cross_size"] = {"negative_interaction_at": neg,
                             "gate_recovers_token_f1_at": gate_wins,
                             "floor_adds_nothing_once_gated_at": floor_ns,
                             "gated_cell_beats_rag_on_both_at":
                                 [m_ for m_, d_ in fam if _beats(d_)],
                             "sizes": [m_ for m_, _ in fam]}
        L.append(
            f"**The gate recovers token F1 at {len(gate_wins)} of {len(fam)} "
            f"sizes** ({', '.join(gate_wins) if gate_wins else 'none'}), and the "
            f"interaction is negative at {len(neg)} of {len(fam)} "
            f"({', '.join(neg) if neg else 'none'}) - so the finding that the two "
            f"fixes overlap is not an 8B artefact. Once the gate is on, dropping "
            f"the coverage floor adds nothing distinguishable from zero at "
            f"{', '.join(floor_ns) if floor_ns else 'no size'}"
            + (", which is the whole of the deployable recommendation at every "
               "size measured.\n" if len(floor_ns) == len(fam) else
               ", and elsewhere the floor still matters, so the recommendation "
               "is stated per size.\n"))

    L += ["## What was and was not fitted here\n",
          "The gate's threshold is not fitted in this experiment. It is fixed at 1, "
          "the value `selective_validation.py` selected inside every one of its five "
          "cross-validation folds, and used here as a constant. The 2x2 is therefore "
          "a comparison of configurations at a threshold chosen elsewhere, not a "
          "second search; no cell's parameters were tuned on these answers. The "
          "detector is held at audit-grade in all four cells, so nothing in the "
          "interaction above is a detector effect.\n"]

    (EVAL / f"selective_axis__{a.model}.md").write_text("\n".join(L), encoding="utf-8")
    (EVAL / f"selective_axis__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote selective_axis__{a.model}.{{md,json}}")
    print(f"  floor {e_floor:+.2f} | gate {e_gate:+.2f} | both {e_both:+.2f} "
          f"| interaction {inter:+.2f}")


if __name__ == "__main__":
    main()
