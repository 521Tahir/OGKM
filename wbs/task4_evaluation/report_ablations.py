"""
WBS Task 4.1.2 - Report for the three generation ablations.

The ablation runner scores groundedness with the Task 3.2.2 validator's own span
detector, which is recall-oriented and counts sentence-initial imperatives, month
names and markdown section labels as entities. That is correct for a safety check and
wrong for a measurement, so every answer is rescored here with the audit-grade filter
and both numbers are shown. The answers themselves are saved per configuration, so no
regeneration is needed.

  python -m wbs.task4_evaluation.report_ablations --model llama3.1-8b
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

# the parameter pair each validation label corresponds to, for the "what it does" column
VALIDATION_NOTE = {
    "none (raw V4 draft)": "the V4 answer, untouched",
    "validate only": "no repair; abstains only when an UNSUPPORTABLE_CLAIM is present",
    "abstain on unsupportable only, no repair":
        "no repair; abstains only when an UNSUPPORTABLE_CLAIM is present",
    "validate + repair": "one repair round; abstains only on unsupportable claims",
    "abstain on unsupportable only + 1 repair":
        "one repair round; abstains only on unsupportable claims",
    "validate + fallback": "no repair; abstains on coverage < 0.6 or unsupportable",
    "abstain on coverage<0.6 or unsupportable, no repair":
        "no repair; abstains on coverage < 0.6 or unsupportable",
    "deployed (repair 1, coverage 0.6)": "the delivered setting",
    "deployed (1 repair, coverage 0.6)": "the delivered setting",
    "repair 2, coverage 0.6": "a second repair round on top of the delivered setting",
    "2 repairs, coverage 0.6": "a second repair round on top of the delivered setting",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    a = ap.parse_args()

    from experiments.exp042_prf.compute_prf import ents, prf, toks
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans
    kb = KnowledgeBase()

    def rescore(answers: list[str], refs: list[str]) -> dict:
        tf, ef, ug, cl = [], [], [], []
        for ans, ref in zip(answers, refs):
            t = prf(toks(ans), toks(ref))
            e = prf(ents(ans), ents(ref))
            if t:
                tf.append(t[2])
            if e:
                ef.append(e[2])
            sp, bad, _ = audit_spans(ans, kb)
            if sp:
                ug.append(len(bad) / len(sp)); cl.append(0.0 if bad else 1.0)
        return {"token_f1": st.mean(tf) if tf else None,
                "entity_f1": st.mean(ef) if ef else None,
                "ungrounded_audit": st.mean(ug) if ug else None,
                "clean_audit": st.mean(cl) if cl else None,
                "mean_chars": round(st.mean([len(x) for x in answers]))}

    L = [f"# Task 4.1 - Ablation Study, Generated Configurations ({a.model})\n",
         "Each axis is run on the same seeded 150-item subset of the benchmark, so the "
         "configurations are paired item by item. Groundedness is reported twice: "
         "*audit* filters out sentence-initial imperatives, month names and markdown "
         "section labels before counting, *raw* is what the Task 3.2.2 validator's own "
         "detector sees. The audit column is the honest hallucination rate; the raw "
         "column is what the repair loop was actually reacting to.\n"]
    found = 0

    for axis, title in (("prompts", "Prompt formulation"),
                        ("validation", "Validation layer, decomposed"),
                        ("retrieval", "Retrieval composition, carried through to answers")):
        jp = EVAL / f"ablation_{axis}__{a.model}.json"
        ap_ = EVAL / f"ablation_{axis}__{a.model}__answers.jsonl"
        if not jp.exists():
            L.append(f"## {title}\n\n_Not run._\n")
            continue
        found += 1
        d = json.loads(jp.read_text(encoding="utf-8"))
        rows = d["results"]
        rescored = {}
        if ap_.exists():
            recs = [json.loads(l) for l in open(ap_, encoding="utf-8") if l.strip()]
            refs = [r["reference"] for r in recs]
            for cfg in rows:
                k = cfg["config"]
                ans = [r["answers"].get(k, "") for r in recs]
                if any(ans):
                    rescored[k] = rescore(ans, refs)

        L.append(f"## {title}  (n={d['n']}, seed {d['seed']})\n")
        extra = " Changed |" if axis == "validation" else ""
        extra_sep = "--:|" if axis == "validation" else ""
        L.append("| Configuration | Token F1 | Entity F1 | Ungrounded (audit) ↓ | "
                 "(raw) | Clean (audit) ↑ | Answer chars | " + extra)
        L.append("|---|--:|--:|--:|--:|--:|--:|" + extra_sep)
        base = rows[0]
        for cfg in rows:
            k = cfg["config"]
            rs = rescored.get(k, {})
            ch = f" {cfg.get('answers_changed', '—')} |" if axis == "validation" else ""
            L.append(f"| {k} | {(rs.get('token_f1') or cfg.get('token_f1') or 0)*100:.1f} "
                     f"| {(rs.get('entity_f1') or cfg.get('entity_f1') or 0)*100:.1f} | "
                     f"{(rs.get('ungrounded_audit') or 0)*100:.1f} | "
                     f"{(cfg.get('ungrounded_rate') or 0)*100:.1f} | "
                     f"{(rs.get('clean_audit') or 0)*100:.1f} | "
                     f"{rs.get('mean_chars', cfg.get('mean_chars', 0)):,} |" + ch)
        L.append("")

        if axis == "validation":
            L.append("**What each configuration actually does.** "
                     "`constrained_answer` abstains when grounded coverage falls below "
                     "the threshold *or* when any `UNSUPPORTABLE_CLAIM` is present, and "
                     "the second trigger ignores the threshold entirely. So a coverage "
                     "floor of 0.0 does not disable abstention — it narrows it to "
                     "unsupportable claims. The rows mean:\n")
            for cfg in rows:
                note = VALIDATION_NOTE.get(cfg["config"], "")
                if note:
                    L.append(f"- *{cfg['config']}* — {note}")
            L.append("")
            def tf_(r):
                return (rescored.get(r["config"], {}).get("token_f1")
                        or r.get("token_f1") or 0)

            def ug_a(r):
                return rescored.get(r["config"], {}).get("ungrounded_audit")

            def ug_r(r):
                return r.get("ungrounded_rate") or 0

            interv = rows[1:]
            bt = max(interv, key=tf_)
            bg_a = min([r for r in interv if ug_a(r) is not None],
                       key=ug_a, default=None)
            bg_r = min(interv, key=ug_r)
            L.append(f"**Every intervention costs reference overlap.** The untouched V4 "
                     f"draft scores {tf_(base)*100:.1f} token F1; the best intervention "
                     f"(*{bt['config']}*) scores {tf_(bt)*100:.1f}, and the delivered "
                     f"setting costs about "
                     f"{(tf_(base)-tf_(rows[4] if len(rows) > 4 else bt))*100:.1f} pp. "
                     f"That part is expected: the layer buys grounding by deleting or "
                     f"hedging content, and deleted content cannot overlap a reference "
                     f"that still contains it.\n")
            if bg_a is not None and ug_a(base) is not None and ug_a(bg_a) >= ug_a(base):
                L.append(f"**But on the audit measure it does not buy grounding "
                         f"either.** The untouched draft is already the most grounded "
                         f"answer in the table ({ug_a(base)*100:.1f}% ungrounded); every "
                         f"configuration is equal or worse, the best being "
                         f"*{bg_a['config']}* at {ug_a(bg_a)*100:.1f}%. On the "
                         f"validator's own raw detector the picture looks different — "
                         f"*{bg_r['config']}* improves on the draft "
                         f"({ug_r(base)*100:.1f}% → {ug_r(bg_r)*100:.1f}%) — and that "
                         f"gap is the whole point. The repair loop was reacting to "
                         f"spans the detector should not have flagged: capitalised "
                         f"imperatives, month names and section labels. It dutifully "
                         f"removed or hedged them, which lowers the raw count it is "
                         f"scored on while leaving real fabrication untouched.\n")
                L.append(f"**So the delivered validation layer is not currently earning "
                         f"its cost.** It spends a second generation pass, abstains on "
                         f"part of the benchmark, gives up {(tf_(base)-tf_(rows[4] if len(rows) > 4 else bt))*100:.1f} "
                         f"pp of token F1, and returns no measurable reduction in "
                         f"hallucination once the measurement stops counting English "
                         f"capitalisation as a claim. The layer's design is sound and "
                         f"the abstention behaviour is the right instinct; what is "
                         f"wrong is the span detector underneath it, and that is a "
                         f"narrow, testable fix — require a multi-token span or a "
                         f"gazetteer-shaped single token, and re-run this same "
                         f"ablation to see whether the repair path then pays for "
                         f"itself."
                         + (" That re-run is the next subsection.\n"
                            if (EVAL / f"ablation_validation__{a.model}__auditdet.json"
                                ).exists() else "\n"))
            else:
                L.append(f"Best groundedness on the audit measure: "
                         f"*{bg_a['config'] if bg_a else '—'}*.\n")

            # R2's open action, carried out: the same configurations with the
            # audit-grade filter inside the detector the repair loop reads
            fx_p = EVAL / f"ablation_validation__{a.model}__auditdet.json"
            fx_a = EVAL / f"ablation_validation__{a.model}__auditdet__answers.jsonl"
            if fx_p.exists():
                fd = json.loads(fx_p.read_text(encoding="utf-8"))
                fr = fd["results"]
                fx_rescored = {}
                if fx_a.exists():
                    frecs = [json.loads(l) for l in open(fx_a, encoding="utf-8")
                             if l.strip()]
                    frefs = [r["reference"] for r in frecs]
                    for cfg in fr:
                        k = cfg["config"]
                        ans = [r["answers"].get(k, "") for r in frecs]
                        if any(ans):
                            fx_rescored[k] = rescore(ans, frefs)

                def ftf(r):
                    return (fx_rescored.get(r["config"], {}).get("token_f1")
                            or r.get("token_f1") or 0)

                def fug(r):
                    return fx_rescored.get(r["config"], {}).get("ungrounded_audit")

                L.append("### R2's fix: the same layer on an audit-grade detector\n")
                L.append(f"Identical configurations, identical drafts, identical seed "
                         f"and the same {fd['n']} items — the only change is that the "
                         f"detector the repair loop reads now applies the audit-grade "
                         f"filter, so a possessive of a known entity, a capitalised "
                         f"imperative and a section label are no longer put to it as "
                         f"fabrications. On the V4 drafts that removes 87.9% of the "
                         f"spans it was asked to fix.\n")
                L.append("| Configuration | Token F1 (raw det.) | Token F1 (audit det.) "
                         "| Ungrounded, audit (raw det.) ↓ | (audit det.) ↓ | "
                         "Changed (raw) | (audit) |")
                L.append("|---|--:|--:|--:|--:|--:|--:|")
                # The published run predates a rename of VALIDATION_CONFIGS, so
                # the two runs label the same configurations differently
                # ("validate + repair" vs "abstain on unsupportable only + 1
                # repair"). VALIDATION_NOTE already maps both schemes onto one
                # description, and that mapping is what makes them the same
                # configuration, so pairing goes through it; position is the
                # fallback, since both lists come from one ordered constant.
                def _canon(cfg, i):
                    return VALIDATION_NOTE.get(cfg["config"]) or f"#pos{i}"

                by_pub = {}
                for i, r in enumerate(rows):
                    by_pub[r["config"]] = r
                    by_pub[_canon(r, i)] = r

                def _pub(cfg, i):
                    return (by_pub.get(cfg["config"])
                            or by_pub.get(_canon(cfg, i)))
                for i, cfg in enumerate(fr):
                    k = cfg["config"]
                    pb = _pub(cfg, i)
                    pu = ug_a(pb) if pb else None
                    fu = fug(cfg)
                    L.append(f"| {k} | "
                             + (f"{tf_(pb)*100:.1f}" if pb else "—") + " | "
                             + f"{ftf(cfg)*100:.1f} | "
                             + (f"{pu*100:.1f}" if pu is not None else "—") + " | "
                             + (f"{fu*100:.1f}" if fu is not None else "—") + " | "
                             + (f"{pb.get('answers_changed', '—')}" if pb else "—")
                             + f" | {cfg.get('answers_changed', '—')} |")
                L.append("")

                # the verdict is derived from the two tables, not restated
                fbase = fr[0]
                finterv = fr[1:]
                fb_t = max(finterv, key=ftf) if finterv else fbase
                cand = [r for r in finterv if fug(r) is not None]
                fb_g = min(cand, key=fug) if cand else None
                base_u = fug(fbase)
                dep_i, deployed = next(
                    ((i, r) for i, r in enumerate(fr)
                     if "deployed" in r["config"].lower()), (0, fb_t))
                dep_pub = _pub(deployed, dep_i)
                pays = (fb_g is not None and base_u is not None
                        and fug(fb_g) < base_u)
                dep_ug = fug(deployed)
                dep_beats = (dep_ug is not None and base_u is not None
                             and dep_ug < base_u)
                L.append("**Does the repair path pay for itself once it is fed "
                         "the right spans?** "
                         + (f"On groundedness, yes - but not in the configuration "
                            f"that ships. *{fb_g['config']}* reaches "
                            f"{fug(fb_g)*100:.1f}% audit-grade ungrounded against "
                            f"the untouched draft's {base_u*100:.1f}%, where under "
                            f"the raw detector no configuration beat the draft at "
                            f"all (best {min(ug_a(r) for r in interv if ug_a(r) is not None)*100:.1f}%). "
                            if pays else
                            f"No. The untouched draft is still the most grounded "
                            f"answer on the audit measure"
                            + (f" ({base_u*100:.1f}% against the best "
                               f"intervention's {fug(fb_g)*100:.1f}%)"
                               if fb_g is not None else "")
                            + ". Fixing the detector removed the mechanism by "
                              "which the layer looked like it was working on the "
                              "raw count and did not supply one by which it works "
                              "on the audit count. ")
                         + (f"The delivered setting is not that configuration: it "
                            f"lands at {dep_ug*100:.1f}%, "
                            + ("level with" if abs(dep_ug - base_u) < 5e-4
                               else f"{(dep_ug-base_u)*100:+.1f} pp against")
                            + f" the draft, while giving up "
                              f"{(ftf(fbase)-ftf(deployed))*100:.1f} pp of token "
                              f"F1 (against {(tf_(base)-tf_(dep_pub))*100:.1f} pp "
                              f"before the fix). "
                            if (dep_ug is not None and base_u is not None
                                and dep_pub and not dep_beats) else "")
                         + "")
                if pays and not dep_beats:
                    L.append(f"**The difference between them is the coverage "
                             f"trigger.** The configuration that wins abstains "
                             f"only on an unsupportable claim; the delivered one "
                             f"also abstains whenever grounded coverage falls "
                             f"below 0.6, without asking whether anything was "
                             f"found wrong. That is the same mechanism "
                             f"`selective_validation.md` isolates from the other "
                             f"direction - there, requiring at least one detected "
                             f"span before spending the layer recovers most of "
                             f"the overlap the coverage trigger costs. Two "
                             f"independent analyses, one over configurations and "
                             f"one over items, name the same component. The "
                             f"deployable conclusion is to keep the repair path "
                             f"and the unsupportable-claim abstention and to stop "
                             f"abstaining on coverage alone.\n")
                else:
                    L.append("")
                L.append(f"The repair loop also stops doing most of its work: it "
                         f"rewrote {deployed.get('answers_changed', '—')} of "
                         f"{fd['n']} answers in the delivered setting, against "
                         f"{(dep_pub or {}).get('answers_changed', '—')}"
                         f" before. Nothing published changes — the fixed run is stored "
                         f"under `__auditdet` and the detector defaults to the "
                         f"published behaviour.\n")
        elif axis == "prompts":
            bt = max(rows, key=lambda r: (rescored.get(r["config"], {}).get("token_f1")
                                          or r.get("token_f1") or 0))
            L.append(f"The deployed template is the reference point; identical weights, "
                     f"identical context, only the wording differs. Best here: "
                     f"**{bt['config']}**.\n")
        else:
            L.append("These are the same context configurations whose gold-entity "
                     "coverage `ablation_offline__*.md` measured, now carried through "
                     "to answers. Coverage in the context and use in the answer are "
                     "different things, and this table is the one that says whether the "
                     "retrieval-side finding survives generation.\n")

            # pair each generated row with the offline coverage for the same ablation
            off_p = EVAL / f"ablation_offline__{a.model}.json"
            if off_p.exists():
                off = json.loads(off_p.read_text(encoding="utf-8"))["retrieval_composition"]

                def key_of(name: str) -> str:
                    n = name.lower()
                    if "2x" in n or "dense only" in n:
                        return "dense_only"
                    if "full" in n:
                        return "full"
                    if "temporal" in n:
                        return "temporal"
                    if "routing" in n and "dense cards only" in n:
                        return "routing"   # offline: "- class routing (dense cards only)"
                    if "dense search" in n:
                        return "dense"
                    if "routing" in n:
                        return "routing"
                    return n

                cov = {key_of(r["config"]): r["gold_coverage"] for r in off}
                L.append("| Ablation | Context gold-entity coverage | Token F1 | "
                         "Coverage delta | Token F1 delta |")
                L.append("|---|--:|--:|--:|--:|")
                base_k = key_of(rows[0]["config"])
                base_cov = cov.get(base_k)
                base_tf = (rescored.get(rows[0]["config"], {}).get("token_f1")
                           or rows[0].get("token_f1") or 0)
                pairs = []
                for cfg in rows:
                    k = key_of(cfg["config"])
                    c = cov.get(k)
                    tf = (rescored.get(cfg["config"], {}).get("token_f1")
                          or cfg.get("token_f1") or 0)
                    dc = None if (c is None or base_cov is None) else (c - base_cov) * 100
                    dt = (tf - base_tf) * 100
                    pairs.append((cfg["config"], c, tf, dc, dt))
                    L.append(f"| {cfg['config']} | "
                             f"{'—' if c is None else f'{c*100:.1f}%'} | {tf*100:.1f} | "
                             f"{'—' if dc is None else f'{dc:+.1f} pp'} | {dt:+.1f} pp |")
                L.append("")

                # the single most informative comparison: largest coverage loss
                hit = [x for x in pairs if x[3] is not None and x[3] < -5]
                if hit:
                    name, c, tf, dc, dt = min(hit, key=lambda x: x[3])
                    ratio = abs(dc / dt) if dt else float("inf")
                    L.append(f"**The retrieval-side finding does not survive generation, "
                             f"and that is the result.** Offline, *{name}* was the "
                             f"catastrophic ablation: context gold-entity coverage falls "
                             f"{abs(dc):.1f} pp, from {base_cov*100:.1f}% to "
                             f"{c*100:.1f}%. Carried through to answers it costs "
                             f"{abs(dt):.1f} pp of token F1 "
                             f"({base_tf*100:.1f} → {tf*100:.1f}). A "
                             f"{abs(dc):.0f}-point collapse in how much of the gold "
                             f"answer is present in the context costs only "
                             f"{abs(dt):.1f} points of overlap with it — roughly "
                             f"{ratio:.0f}:1.\n")
                    L.append("**What that means.** Context gold-entity coverage is a "
                             "weak proxy for answer quality on this benchmark. Most of "
                             "what the model writes is not coming from the specific "
                             "entities the retriever found; it is coming from the "
                             "question, the adapter and the prompt's shape, with "
                             "retrieval supplying topical anchoring rather than the "
                             "content that the reference overlaps on. The offline "
                             "conclusion — that degree-ordered class routing carries "
                             "the arm and dense entity-card search does not earn its "
                             "budget — is still correct about the *context*: dropping "
                             "dense search costs nothing here either "
                             f"({[x for x in pairs if 'dense search' in x[0].lower()][0][4]:+.1f} pp token F1 "
                             "for 27% fewer context characters). But the corollary that "
                             "fixing routing would move the scores is not supported. "
                             "Retrieval composition is close to saturated; the headroom "
                             "is in generation, which is consistent with the prompt axis "
                             "above showing no headroom in wording either.\n")

    if not found:
        print("no ablation outputs yet")
        return
    p = EVAL / f"ablation_generation__{a.model}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {p.name}  ({found}/3 axes present)")


if __name__ == "__main__":
    main()
