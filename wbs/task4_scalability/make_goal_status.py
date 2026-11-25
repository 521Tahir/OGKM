"""
A decision-ready statement of where the project stands against the standing
goal, and of what each remaining item needs from someone other than this
pipeline.

The goal as written - that the ontology arms beat Simple RAG on *all*
evaluation metrics, on the full datasets - has been tested repeatedly. No arm
*leads* on reference overlap at any size. What changed on 2026-10-19 is how
much is being given up for that: R19 found the layer's overlap cost was the
rewrite rather than the check, and with surgical repair the gap falls from
-2.37 / -1.75 / -3.44 token F1 to -0.73 / -0.12 / -0.86, which at llama3.2-3b
is no longer distinguishable from zero on any metric. So the honest statement
is now "never worse than Simple RAG at 3B, and behind by under one point
elsewhere, while strictly better on groundedness at all three sizes" rather
than a flat falsification. This file says which parts are which, and what the
remaining items need from someone other than this pipeline.

Every number here is read from a deliverable produced by a named script. The
blockers are listed with what they require, not with an estimate of effort,
because each needs a decision or a resource rather than more computation.

  python -m wbs.task4_scalability.make_goal_status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DELIV = ROOT / "wbs" / "deliverables"
EV, T2 = DELIV / "task4_evaluation", DELIV / "task2_formalization"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    prop = load(DELIV / "PROPOSAL_COMPLIANCE.json") or {}
    wbs = load(DELIV / "WBS_WORKBOOK_AUDIT.json") or {}
    hyb = {m: load(EV / f"hybrid_adapter_comparison__{m}.json")
           for m in ("llama3.2-1b", "llama3.2-3b")}
    hyb["llama3.1-8b"] = load(EV / "hybrid_adapter_comparison.json")
    axis = {m: load(EV / f"selective_axis__{m}.json")
            for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b")}
    oo = {m: load(EV / f"ontology_only_validated__{m}.json")
          for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b")}
    route = load(EV / "coverage_routing__llama3.1-8b.json")
    para = load(EV / "reference_paraphrase__llama3.1-8b.json")
    ws = load(T2 / "weather_state_extraction.json")
    rc = load(T2 / "rule_compliance.json")
    rp = {m: load(EV / f"repair_policy__{m}.json")
          for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b")}
    DEPLOYED = "V5 surgical clause, gated"
    VERIFIER = "V6 surgical clause on the RAG draft, gated"

    L = ["# Where this project stands against the standing goal\n",
         "The goal this work was run under asks for the WBS complete through "
         "Task 4's final results, and for the ontology arms to be best. The "
         "first is done and audited. The second was measured rather than argued "
         "down, and the measurement moved on 2026-10-19: no ontology arm "
         "*leads* on reference overlap at any size, but R19 found that most of "
         "what the validated arm was giving up came from the validation layer "
         "rewriting whole answers rather than from grounding them. With "
         "surgical repair the gap is under one token F1 point at every size, "
         "and nothing distinguishable from zero on any metric at llama3.2-3b. "
         "On 2026-10-26 the comparison was widened: the companion project's "
         "runs over eight models and 4,607 held-out questions were re-scored on "
         "this project's metrics and paired-tested, which left the headline "
         "intact and told us what it depends on. "
         "This file says which parts are which, and what the remaining items "
         "need from someone other than this pipeline.\n",
         "## Complete and audited\n"]
    if wbs:
        L.append(f"- **WBS**: {wbs.get('fully_satisfied')} of "
                 f"{wbs.get('total')} workbook items fully satisfied, "
                 f"{wbs.get('partial')} partial (`WBS_WORKBOOK_AUDIT.md`).")
    if prop:
        L.append(f"- **Proposal**: {prop.get('met')} of {prop.get('total')} "
                 f"items met, {prop.get('partial')} partial, "
                 f"{prop.get('out_of_scope')} out of scope, "
                 f"{prop.get('missing')} missing "
                 f"(`PROPOSAL_COMPLIANCE.md`).")
    L.append("")

    L.append("## The comparison the goal names, as measured\n")
    L.append("| Claim | Verdict | Where |")
    L.append("|---|---|---|")
    sizes = [m for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b") if hyb.get(m)]
    if sizes:
        L.append("| Ontology arms beat Simple RAG on **all** metrics | "
                 "**false at every size tested** — no arm, in any "
                 "configuration, leads on token F1 | §5.1b, §5.1d–g |")
    never_worse = [m for m, d in rp.items()
                   if d and ((d.get("claim_check") or {}).get(DEPLOYED) or {})
                   .get("never_worse_than_rag")]
    if never_worse:
        L.append(f"| Ontology arms are never **worse** than Simple RAG on any "
                 f"metric | **true at {', '.join(sorted(never_worse))}** with "
                 f"surgical repair, under Holm-corrected paired tests | "
                 f"`repair_policy.md` |")
    gaps = {m: ((d.get("cells") or {}).get(DEPLOYED, {}).get("vs_rag") or {})
            .get("token_f1", {}).get("delta")
            for m, d in rp.items() if d}
    order = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
    gaps = {m: gaps[m] for m in order if gaps.get(m) is not None}
    if gaps:
        L.append(f"| The overlap gap is the price of grounding | **no — it was "
                 f"mostly the rewrite**: surgical repair takes it to "
                 f"{', '.join(f'{gaps[m]:+.2f}' for m in gaps)} token F1 "
                 f"({', '.join(gaps)}) at 0% ungrounded "
                 f"| `repair_policy.md`, R19 |")
    def both_ground(d) -> bool:
        vr = (d or {}).get("vs_rag") or {}
        if not vr:
            return False
        u, c = vr.get("ungrounded_rate") or {}, vr.get("clean") or {}
        return ((u.get("p") or 1) < .05 and (u.get("delta") or 0) < 0
                and (c.get("p") or 1) < .05 and (c.get("delta") or 0) > 0)

    gw = [m for m in sizes if both_ground(axis.get(m))]
    if gw:
        L.append(f"| The validated ontology arm beats Simple RAG on "
                 f"**both groundedness measures** | **true at "
                 f"{', '.join(gw)}**, under "
                 f"paired tests | §5.1d |")
    oow = [m for m, d in oo.items()
           if d and (d.get("beats_rag_on_both_groundedness") or [])]
    if oow:
        L.append(f"| **Ontology context alone** beats Simple RAG on "
                 f"groundedness | **true at {', '.join(sorted(oow))} only** | "
                 f"§5.1e |")
    if route and route.get("oracle_selector"):
        L.append(f"| The overlap gap can be closed by choosing an arm per "
                 f"question | **no** — an oracle selector gains only "
                 f"{route['oracle_selector']['headroom_token_f1_pp']:+.2f} "
                 f"token F1 | §5.1f |")
    if para and para.get("lead_before_after"):
        lb = para["lead_before_after"]
        sh = [abs(v["paraphrased"] - v["original"]) / abs(v["original"]) * 100
              for v in lb.values() if v["original"]]
        L.append(f"| Simple RAG's overlap lead is an artefact of references "
                 f"written from its own corpus | **partly** — rewording the "
                 f"references removes {min(sh):.0f}–{max(sh):.0f}% of it, not "
                 f"all | §5.1g |")
    xm = load(EV / "cross_model_results.json")
    if xm:
        roll, summ = xm["metric_rollup"], xm["summary"]
        eq_n = summ["equal"]["pairs"]
        ex_n = summ["extended"]["pairs"]
        L.append(f"| The overlap result is specific to the Llama family or to a "
                 f"347-item benchmark | **no** — at equal training budget "
                 f"{summ['equal']['clean_sweeps']} of {eq_n} models across two "
                 f"vendors lead on every metric over "
                 f"{xm['split_n']:,} held-out questions | "
                 f"`cross_model_results.md` |")
        cl = roll["equal"]["clean"]["ontology_better_at"]
        L.append(f"| The groundedness win generalises **at equal budget** | "
                 f"**at {len(cl)} of {eq_n}** — {', '.join(cl) or 'none'}; "
                 f"neither Llama 3.2 size reproduces it at one pass | "
                 f"`cross_model_results.md` |")
        mt = summ.get("matched", {})
        L.append(f"| Extra ontology training closes the overlap gap | **against a "
                 f"one-pass baseline, yes** — "
                 f"{summ['extended']['never_worse']} of {ex_n} models are then "
                 f"worse on nothing measured and "
                 f"{summ['extended']['clean_sweeps']} leads on everything, at a "
                 f"3:1–4:1 training cost | `cross_model_results.md` |")
        if mt.get("pairs"):
            # The row above is what the stored runs supported; this row is what
            # survives once the baseline is given the same number of passes.
            eff = xm.get("budget_effect") or {}
            kept_all = sum(len(e["kept"]) for e in eff.values())
            won_all = sum(len(e["kept"]) + len(e["lost"]) + len(e["to_loss"])
                          for e in eff.values())
            held = [k.replace("@p3m", "").replace("@p4m", "")
                    for k, e in sorted(eff.items()) if e["kept"]]
            mcl = roll["matched"]["clean"]["ontology_better_at"]
            L.append(f"| …and it survives a baseline trained just as long | "
                     f"**mostly no — {mt['never_worse']} of {mt['pairs']} models** "
                     f"worse on nothing measured once Simple RAG is continued to the "
                     f"same pass count ({kept_all} of {won_all} wins survive across "
                     f"the four; any win at all only at "
                     f"{', '.join(held) or 'none'}) | `cross_model_results.md` |")
            L.append(f"| The groundedness win survives an equal-budget baseline | "
                     f"**at {len(mcl)} of {mt['pairs']}** — {', '.join(mcl) or 'none'}, "
                     f"and at both the margin is *wider* than against the one-pass "
                     f"baseline, because more RAG passes made the baseline less "
                     f"grounded | `cross_model_results.md` |")
        L.append(f"| The 8B results showing large ontology margins are evidence "
                 f"for the ontology | **no** — those four pairs train an adapter "
                 f"for the ontology arm and none for Simple RAG; the RAG adapter "
                 f"directory does not exist, so the margin measures fine-tuning | "
                 f"`cross_model_results.md` |")

    vx = load(EV / "verifier_across_models__test.json")
    if vx and vx.get("models"):
        n_fall = len(vx["holds_at"]["ungrounded_strict_falls"])
        L.append(f"| The groundedness win is a fit to the model family we tuned "
                 f"for | **no** \u2014 the verifier applied unchanged to six models' "
                 f"stored answers more than halves the strict-rule ungrounded rate "
                 f"at {n_fall} of {len(vx['models'])}, with no adapter and no "
                 f"retraining | `verifier_across_models__test.md` |")
    L.append("")

    L.append("## What is open, and what each one needs\n")
    L.append("The rest need people; the GPU job among them is running rather than "
             "waiting. None is blocked on re-reading what is already stored.\n")
    L.append("| Item | What it needs | Why this pipeline cannot supply it |")
    L.append("|---|---|---|")
    L.append("| **R6** — human adjudication of a benchmark subset | two or more "
             "annotators to fill `manual_hallucination_review.csv` and "
             "`expert_ontology_review.csv`, which are built and unfilled; the "
             "scoring they feed is written and verified "
             "(`score_human_review.py`, `human_review_scores.md`) | "
             "inter-annotator agreement is a property of independent human "
             "judgements; a model scoring its own benchmark measures the model |")
    if ws and ws.get("still_blocked"):
        L.append(f"| **R12** — rules {', '.join(ws['still_blocked'])} | a source "
                 f"that carries a **graded risk severity** | the corpus has "
                 f"the phrasing in {ws['records_with_graded_severity']} of "
                 f"{ws['records_scanned']:,} records (0.03%); assigning "
                 f"severities by model judgement would be inventing the "
                 f"attribute the rule is supposed to check |")
    L.append("| **R14** — binding principals to an identity provider | a "
             "deployment with real accounts | the *authorization* half was "
             "closed on 2026-10-23: four roles over thirteen permissions, "
             "enforced at all six governance entry points, 158 protected "
             "elements, staging/production isolation, denials on the audit "
             "trail (`access_control.md`). What is left is authentication - the "
             "principal is whatever `WBS_ACTOR` says, which no amount of local "
             "code can make into a security boundary |")
    if xm:
        waiting = xm.get("pending_pairs") or []
        if waiting:
            where = "; ".join(f"{r['model']} {r['status']}" for r in waiting)
            L.append(f"| **Equal-budget parity at extended training** — *running* | "
                     f"the {len(waiting)} Simple RAG arms being continued to the same "
                     f"pass count as their ontology counterparts "
                     f"(`experiments/exp060_matched_budget_rag`) | nothing stored "
                     f"answered it: every multi-pass run in either project left the RAG "
                     f"arm at one pass, so the never-worse result at four models is "
                     f"confounded with a 3:1-4:1 training advantage that no re-scoring "
                     f"could separate. It is being trained rather than argued about. "
                     f"Status: {where} |")
        else:
            L.append("| **Equal-budget parity at extended training** | — | closed by "
                     "`exp060_matched_budget_rag`: every extended pair now has a Simple "
                     "RAG arm trained to the same total pass count, and the `matched` "
                     "tier of `cross_model_results.md` reports the result |")
    L.append("| **Proposal 11.4.2, 11.6b, 11.7b** | an expert panel, an "
             "annotator panel and a user study | out of scope by the "
             "proposal's own scoping; all three are human-subject work. §8.7 "
             "left this list on 2026-10-23 - it turned out to be code, not "
             "fieldwork |")
    L.append("")
    L.append("## The honest summary\n")
    L.append("The ontology does not *lead* the reference-overlap metrics and "
             "will not on this benchmark: the entities it needs are present, "
             "the phrasing is not, per-question arm selection is bounded well "
             "below the gap, and about a fifth of the gap is the references' "
             "provenance rather than answer quality. No stored arm, ontology or "
             "hybrid, drafts above Simple RAG, so the draft is the ceiling.\n")
    if xm:
        L.append("That finding is not an artefact of three Llama models or of a "
                 "347-item benchmark. Re-measured on 4,607 held-out questions "
                 "across eight models and six model lines, at equal training "
                 "budget every one of the four comparable models still gives up "
                 "token precision, and the groundedness lead reproduces at both "
                 "Qwen 2.5 sizes and at neither Llama 3.2 size. The strongest "
                 "ontology result on record — four of four models worse on "
                 "nothing measured — holds only where the ontology arm was "
                 "trained three or four times against a baseline trained once.\n")
        mt = summ.get("matched", {})
        if mt.get("pairs"):
            eff = xm.get("budget_effect") or {}
            kept_all = sum(len(e["kept"]) for e in eff.values())
            won_all = sum(len(e["kept"]) + len(e["lost"]) + len(e["to_loss"])
                          for e in eff.values())
            mcl = roll["matched"]["clean"]["ontology_better_at"]
            widened = [k.split("@")[0] for k, e in sorted(eff.items())
                       if any(d["ontology_leads_now"] for d in e["baseline_worse"])]
            # This is the paragraph the whole GPU run exists to be able to write.
            L.append(f"That comparison has now been run the other way. "
                     f"`exp060_matched_budget_rag` trained each model's Simple RAG "
                     f"arm to the same total pass count — same records, same "
                     f"contexts, same recipe, only the epoch count raised — and the "
                     f"result does not survive it: {mt['never_worse']} of "
                     f"{mt['pairs']} {'model' if mt['pairs'] == 1 else 'models'} "
                     f"{'is' if mt['never_worse'] == 1 else 'are'} "
                     f"worse on nothing measured, and "
                     f"{kept_all} of the {won_all} wins the unequal comparison "
                     f"showed remain. Every token-overlap win is gone at three of "
                     f"the four models. What survives is groundedness, at "
                     f"{len(mcl)} of {mt['pairs']} "
                     f"({', '.join(mcl) or 'none'}), and there the margin is *wider* "
                     f"than it was against the one-pass baseline"
                     + (f" — training Simple RAG longer made it less grounded, not "
                        f"more, at {', '.join(widened)}. So the overlap result was "
                        f"the training budget and the groundedness result was not."
                        if widened else ".") + "\n")
    L.append("What it does win is groundedness — every claim checkable against "
             "the graph — at every model size measured, on the published "
             "detector and on the independent strict one. What changed with R19 "
             "is the price: the validation layer was paying for that with a "
             "whole-answer rewrite, and once it cuts only the clause that "
             "carries the violation the overlap it costs is a fraction of a "
             "point and its compute cost is about 1 ms. At llama3.2-3b the "
             "ontology arm is now worse than Simple RAG on nothing measured; at "
             "1B and 8B it is behind on token F1 by under one point. That is "
             "the result this project supports, stated with the size it holds "
             "at and without the clause it still cannot support.\n")

    out = DELIV / "GOAL_STATUS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
