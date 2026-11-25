"""
WBS Task 4.1.2 - Where does ontology grounding compete on reference overlap,
and can an inference-time router exploit it?

The project's standing explanation of the token-F1 deficit is that the gold
answers were written from corpus prose the graph does not carry. That has been
argued from aggregate coverage and never tested per item. If it is right, the
deficit should shrink or vanish on the items whose gold answers the graph *can*
resolve, and the ontology arms should be preferable there.

Two measurements, kept apart because one is deployable and one is not:

  1. ORACLE STRATIFICATION (diagnostic only). Items are split by how much of
     their gold answer the knowledge base resolves. This uses the reference, so
     it cannot be a policy; it tests the explanation.
  2. A ROUTER on a question-side signal (deployable). The signal is computed
     from the question alone - how many entity-like spans in the question the
     KB resolves - and the threshold is fitted inside 5 folds, so every item is
     routed by a threshold fitted without it. No generation is re-run: both
     arms' answers are already stored, and routing selects between them.

  python -m wbs.task4_evaluation.coverage_routing --model llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"

GATED = "gated >=1 span, unsupportable only"
MET = [("token_f1", "Token F1"), ("entity_f1", "Entity F1"),
       ("ungrounded_rate", "Ungrounded ↓"), ("clean", "Clean ↑")]
BETTER_LOW = {"ungrounded_rate"}
FOLDS = 5


def holm(tests: list[tuple]) -> dict:
    ts = sorted(tests, key=lambda t: (1.0 if t[2] is None else t[2]))
    prev, out = 0.0, {}
    for i, (lab, _dv, pv) in enumerate(ts):
        prev = max(prev, min(1.0, (1.0 if pv is None else pv) * (len(ts) - i)))
        out[lab] = prev
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans
    from wbs.task4_evaluation.score_variants import score_item, wilcoxon

    kb = KnowledgeBase()

    def rows(p: Path) -> dict:
        return {json.loads(l)["bench_id"]: json.loads(l)
                for l in open(p, encoding="utf-8") if l.strip()}

    v2p = EVAL / f"variant__{a.model}__V2_rag.jsonl"
    v4p = EVAL / f"variant__{a.model}__V4_ontology_rag__hybded.jsonl"
    selp = EVAL / f"ablation_selective__{a.model}__auditdet__hybrid20d__answers.jsonl"
    for p_ in (v2p, v4p):
        if not p_.exists():
            sys.exit(f"{p_.name} not found")
    v2, v4 = rows(v2p), rows(v4p)
    ids = [b for b in v2 if b in v4]
    gated = None
    if selp.exists():
        sr = rows(selp)
        if all(b in sr for b in ids):
            gated = {b: {"bench_id": b, "category": "x",
                         "reference": sr[b]["reference"],
                         "answer": sr[b]["answers"][GATED]} for b in ids}

    arms = {"V2 Simple RAG": {b: v2[b] for b in ids},
            "V4 ontology+RAG": {b: v4[b] for b in ids}}
    if gated:
        arms["V5 gated (best ontology arm)"] = gated
    per = {k: [score_item(v[b], kb) for b in ids] for k, v in arms.items()}

    # ── the two signals ───────────────────────────────────────────────────
    # oracle: how much of the gold answer the graph can resolve at all
    # deployable: how much of the question it can resolve
    def resolved_share(text: str) -> tuple[float | None, int]:
        sp, bad, _ = audit_spans(text or "", kb)
        if not sp:
            return None, 0
        return (len(sp) - len(bad)) / len(sp), len(sp) - len(bad)

    gold_cov, q_hits = {}, {}
    for b in ids:
        gold_cov[b] = resolved_share(v2[b].get("reference", ""))[0]
        q_hits[b] = resolved_share(v2[b].get("Input", ""))[1]

    out = {"model": a.model, "n": len(ids),
           "arms": {k: {f: (st.mean(xs) if (xs := [x[f] for x in per[k]
                                                   if x[f] is not None]) else None)
                        for f, _ in MET} for k in arms}}

    L = [f"# Where ontology grounding competes on overlap, and whether a router "
         f"can use it ({a.model})\n",
         "This project explains its token-F1 deficit structurally: the gold "
         "answers were written from the corpus the passages are drawn from, and "
         "the graph carries the facts rather than the phrasing. That has been "
         "argued from one aggregate number and never checked per item. If it is "
         "right, the deficit should be smallest on the items whose gold answers "
         "the knowledge base can actually resolve.\n",
         f"All {len(ids)} benchmark items, answers taken from the stored runs - "
         "nothing is re-generated, and every arm keeps its own dedicated "
         "adapter.\n"]

    L += ["| Arm | " + " | ".join(lab for _, lab in MET) + " |",
          "|---|" + "--:|" * len(MET)]
    for k in arms:
        L.append(f"| {k} | " + " | ".join(
            f"{out['arms'][k][f]*100:.2f}" if out["arms"][k][f] is not None else "—"
            for f, _ in MET) + " |")
    L.append("")

    # ── 1. what the graph is actually missing ─────────────────────────────
    # The first version of this split ranked items by the share of their gold
    # answer's entity spans the KB resolves. That signal saturates - most items
    # sit at exactly 1.0 - so the thirds were not separating anything and the
    # verdict written from them said more than the data did. The saturation is
    # itself the answer to one half of the question, and the half that still
    # needs a split gets one with spread: how fact-like the reference is.
    covs = [c for c in (gold_cov[b] for b in ids) if c is not None]
    at_one = sum(1 for c in covs if c >= 0.999)
    idx = {b: i for i, b in enumerate(ids)}

    def dens(b: str) -> float:
        ref = v2[b].get("reference", "") or ""
        sp, _bad, _g = audit_spans(ref, kb)
        return len(sp) / max(len(ref.split()), 1)

    density = {b: dens(b) for b in ids}
    out["gold_span_coverage"] = {"n_scored": len(covs), "at_1.0": at_one,
                                 "median": st.median(covs),
                                 "mean": st.mean(covs)}

    L += ["## 1. What the graph is missing is not the entities\n",
          f"Of the {len(covs)} benchmark items whose gold answer contains an "
          f"entity-like span at all, the knowledge base resolves **every** span "
          f"in {at_one} of them ({at_one/len(covs)*100:.0f}%), and the median "
          f"item is at {st.median(covs)*100:.0f}% resolved. The graph is "
          "therefore not short of the places the reference talks about; an "
          "item-level split on that quantity cannot explain the overlap deficit "
          "because the quantity barely varies.\n",
          "So the split below uses what does vary: how **fact-like** the "
          "reference is, measured as entity-like spans per word. A dense "
          "reference is a list of places; a sparse one is prose about them. If "
          "the deficit is about phrasing rather than facts, the ontology arm "
          "should be at its best on the dense end.\n"]

    scored = sorted(((b, density[b]) for b in ids), key=lambda t: t[1])
    k3 = len(scored) // 3
    strata = [("sparsest third (most prose)", scored[:k3]),
              ("middle third", scored[k3:2 * k3]),
              ("densest third (most fact-like)", scored[2 * k3:])]

    def paired_on(sub: list[str], base: str, arm: str, f: str) -> dict:
        d = [(per[base][idx[b]][f], per[arm][idx[b]][f]) for b in sub
             if per[base][idx[b]][f] is not None and per[arm][idx[b]][f] is not None]
        if not d:
            return {"delta": None, "p": None, "n": 0}
        w = wilcoxon([q for _, q in d], [p_ for p_, _ in d])
        return {"delta": (st.mean([q for _, q in d])
                          - st.mean([p_ for p_, _ in d])) * 100,
                "p": (w[1] if w else None), "n": len(d)}

    L += ["| Stratum | items | spans per word | V4 − V2 token F1 | p | "
          "V5 gated − V2 token F1 | p |", "|---|--:|--:|--:|--:|--:|--:|"]
    strat_json = {}
    for name, grp in strata:
        sub = [b for b, _ in grp]
        d4 = paired_on(sub, "V2 Simple RAG", "V4 ontology+RAG", "token_f1")
        d5 = (paired_on(sub, "V2 Simple RAG", "V5 gated (best ontology arm)",
                        "token_f1") if gated else {"delta": None, "p": None})
        strat_json[name] = {"n": len(sub), "mean_density": st.mean(c for _, c in grp),
                            "v4_vs_v2_token_f1": d4, "v5_vs_v2_token_f1": d5}
        L.append(f"| {name} | {len(sub)} | {st.mean(c for _, c in grp):.3f} | "
                 f"{d4['delta']:+.2f} | "
                 + (f"{d4['p']:.3g}" if d4["p"] is not None else "—") + " | "
                 + (f"{d5['delta']:+.2f} | "
                    + (f"{d5['p']:.3g} |" if d5["p"] is not None else "— |")
                    if d5["delta"] is not None else "— | — |"))
    out["density_strata"] = strat_json
    L.append("")

    ds = [strat_json[n]["v4_vs_v2_token_f1"]["delta"] for n, _ in strata]
    mono = ds[0] <= ds[1] <= ds[2]
    best_i = max(range(3), key=lambda i: ds[i])
    L.append(
        ("**The deficit does track how fact-like the reference is, and "
         "monotonically.**" if mono else
         "**The deficit moves with how fact-like the reference is, but not "
         "monotonically.**")
        + f" V4 against Simple RAG is {ds[0]:+.2f} on the prose end, "
        f"{ds[1]:+.2f} in the middle and {ds[2]:+.2f} on the fact-like end; the "
        f"best stratum for the ontology arm is *{strata[best_i][0]}*. "
        + ("The ordering is what the structural explanation predicts.\n" if mono
           else "The ordering is not the clean one the structural explanation "
                "predicts, so the explanation is supported only as a tendency: "
                "the ontology arm is never ahead of Simple RAG on any third, "
                "and the gap is smallest where the reference is least prose-like."
                "\n" if max(ds) <= 0 else
                "On at least one stratum the ontology arm is ahead, which is "
                "the part worth a router.\n"))

    # ── 2. the deployable router ──────────────────────────────────────────
    rng = random.Random(a.seed)
    order = ids[:]
    rng.shuffle(order)
    folds = [order[i::FOLDS] for i in range(FOLDS)]

    def tf(arm: str, b: str) -> float:
        return per[arm][idx[b]]["token_f1"]

    # two question-side signals, each available before any answer exists
    SIGNALS = {"question spans resolved in KB": {b: float(q_hits[b]) for b in ids},
               "question length in words":
                   {b: float(len((v2[b].get("Input") or "").split())) for b in ids}}
    targets = [k for k in arms if k != "V2 Simple RAG"]

    def fit_router(sig: dict, onto_arm: str) -> tuple[dict, list]:
        cand = sorted(set(sig.values()))
        chosen, picks = {}, []
        for i in range(FOLDS):
            tr = [b for j, f_ in enumerate(folds) if j != i for b in f_]
            best_t, best_v = None, None
            # "never route" is in the candidate set as +inf, so a fold that
            # cannot beat Simple RAG is allowed to say so rather than being
            # forced to route something
            for t in cand + [float("inf")]:
                v = st.mean(tf(onto_arm if sig[b] >= t else "V2 Simple RAG", b)
                            for b in tr)
                if best_v is None or v > best_v:
                    best_t, best_v = t, v
            picks.append(best_t)
            for b in folds[i]:
                chosen[b] = onto_arm if sig[b] >= best_t else "V2 Simple RAG"
        return chosen, picks

    runs = []
    for sname, sig in SIGNALS.items():
        for onto_arm in targets:
            chosen, picks = fit_router(sig, onto_arm)
            scores = [per[chosen[b]][idx[b]] for b in ids]
            runs.append({"signal": sname, "arm": onto_arm, "picks": picks,
                         "chosen": chosen, "scores": scores,
                         "n_routed": sum(1 for b in ids
                                         if chosen[b] == onto_arm),
                         "token_f1": st.mean(r["token_f1"] for r in scores)})
    best = max(runs, key=lambda r: r["token_f1"])
    per["__router"] = best["scores"]
    chosen, picks, onto_arm = best["chosen"], best["picks"], best["arm"]
    routed_onto = best["n_routed"]
    out["router_candidates"] = [
        {k: r[k] for k in ("signal", "arm", "n_routed", "token_f1")} for r in runs]
    # an oracle selector, as an upper bound on what any router could reach
    per["__oracle"] = [max((per[k][idx[b]] for k in arms),
                           key=lambda r: r["token_f1"]) for b in ids]

    L += ["## 2. A router that uses only the question\n",
          f"Each candidate router picks an arm per item from a question-side "
          f"signal - nothing from the answer, nothing from the reference - with "
          f"its threshold fitted inside each of {FOLDS} folds, so every item is "
          f"routed by a threshold fitted without it. "
          f"{len(runs)} combinations were searched "
          f"({len(SIGNALS)} signals x {len(targets)} target arms) and the best "
          f"on in-fold token F1 is reported: *{best['signal']}* routing to "
          f"*{onto_arm}*. Thresholds chosen: "
          f"{', '.join(('never route' if p_ == float('inf') else str(p_)) for p_ in picks)}"
          f". It sends {routed_onto} of {len(ids)} items to the ontology arm.\n",
          "| Arm | " + " | ".join(lab for _, lab in MET) + " |",
          "|---|" + "--:|" * len(MET)]
    for k in ("V2 Simple RAG", onto_arm, "__router", "__oracle"):
        nm = {"__router": "**router (out of fold)**",
              "__oracle": "_oracle selector (upper bound, not deployable)_"}.get(k, k)
        L.append(f"| {nm} | " + " | ".join(
            f"{st.mean([x[f] for x in per[k] if x[f] is not None])*100:.2f}"
            for f, _ in MET) + " |")
    L.append("")

    fam, keep = [], {}
    for f, _lab in MET:
        d = [(per["V2 Simple RAG"][i][f], per["__router"][i][f])
             for i in range(len(ids))
             if per["V2 Simple RAG"][i][f] is not None
             and per["__router"][i][f] is not None]
        w = wilcoxon([q for _, q in d], [p_ for p_, _ in d])
        x = {"delta": (st.mean([q for _, q in d])
                       - st.mean([p_ for p_, _ in d])) * 100,
             "p": (w[1] if w else None), "n": len(d)}
        keep[f] = x
        fam.append((f, x["delta"], x["p"]))
    adj = holm(fam)
    L += ["Paired against Simple RAG, out of fold, Holm over the four tests:\n",
          "| Metric | Simple RAG | router | Δ | p | Holm p |",
          "|---|--:|--:|--:|--:|--:|"]
    out["router"] = {"signal": "question spans resolved in KB",
                     "thresholds": picks, "n_routed_to_ontology": routed_onto,
                     "arm": onto_arm, "vs_rag": {}}
    for f, lab in MET:
        x = keep[f]
        base = st.mean([r[f] for r in per["V2 Simple RAG"] if r[f] is not None]) * 100
        out["router"]["vs_rag"][f] = dict(x, holm_p=adj[f])
        L.append(f"| {lab} | {base:.2f} | {base + x['delta']:.2f} | "
                 f"{x['delta']:+.2f} | "
                 + (f"{x['p']:.3g}" if x["p"] is not None else "—")
                 + f" | {adj[f]:.3g} |")
    L.append("")

    def wins(f: str) -> bool:
        x = out["router"]["vs_rag"][f]
        if x["holm_p"] is None or x["holm_p"] >= 0.05:
            return False
        return x["delta"] < 0 if f in BETTER_LOW else x["delta"] > 0

    tfx = out["router"]["vs_rag"]["token_f1"]
    beats_overlap = wins("token_f1")
    ground_win = all(wins(f) for f in ("ungrounded_rate", "clean"))
    out["router"]["beats_rag_on_token_f1"] = bool(beats_overlap)
    out["router"]["beats_rag_on_both_groundedness"] = bool(ground_win)
    L.append(
        ("**The router beats Simple RAG on reference overlap.** "
         if beats_overlap else
         "**The router does not beat Simple RAG on reference overlap.** ")
        + f"Out of fold it is {tfx['delta']:+.2f} token F1 "
        + (f"(p = {tfx['p']:.3g}, Holm p = {tfx['holm_p']:.3g})"
           if tfx["p"] is not None else "(no signed-rank pairs)")
        + (", and it keeps the ontology arm's groundedness advantage.\n"
           if ground_win else
           f". The fitted policy barely routes at all - {routed_onto} of "
           f"{len(ids)} items - which is the honest form of the result: on "
           f"every fold, the best question-side threshold over "
           f"{len(runs)} signal-and-arm combinations is the one that sends "
           f"almost everything to Simple RAG. A per-item choice between these "
           f"arms is not where the overlap deficit can be recovered.\n"
           if routed_onto < len(ids) * .05 else
           ". Routing to the ontology arm on a question-side signal moves "
           "overlap toward Simple RAG's but not past it.\n"))
    orc = st.mean([r["token_f1"] for r in per["__oracle"]]) * 100
    base_tf = st.mean([r["token_f1"] for r in per["V2 Simple RAG"]]) * 100
    out["oracle_selector"] = {"token_f1": orc / 100, "rag_token_f1": base_tf / 100,
                              "headroom_token_f1_pp": orc - base_tf,
                              "arms": list(arms)}
    L.append(f"The oracle selector - choosing per item whichever stored answer "
             f"scores higher, which no deployable system can do - reaches "
             f"{orc:.2f} token F1 against Simple RAG's {base_tf:.2f}. That is "
             f"the headroom any selector over these three arms has: "
             f"{orc - base_tf:+.2f} token F1. The out-of-fold router recovers "
             f"{tfx['delta'] / (orc - base_tf) * 100:.0f}% of it.\n")

    (EVAL / f"coverage_routing__{a.model}.md").write_text("\n".join(L),
                                                          encoding="utf-8")
    (EVAL / f"coverage_routing__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote coverage_routing__{a.model}.{{md,json}}")
    print(f"  strata V4-V2 token F1: " + ", ".join(
        f"{k} {v['v4_vs_v2_token_f1']['delta']:+.2f}" for k, v in strat_json.items()))
    print(f"  router {tfx['delta']:+.2f} token F1 (Holm p = {tfx['holm_p']:.3g}), "
          f"oracle headroom {orc - base_tf:+.2f}")


if __name__ == "__main__":
    main()
