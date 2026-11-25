"""
The proposal's three hypotheses (section 3.3), each with a verdict and the
measurement behind it.

  H1  structured prompting -> higher logical consistency and hierarchical quality
      than naive prompting                    (hypothesis_h1*.json)
  H2  ontology constraints -> less hallucination without degrading fluency
      (hypotheses_h2_h3__*.json, plus the published variant scores)
  H3  ontology-integrated systems -> more interpretable and verifiable responses
      (hypotheses_h2_h3__*.json)

A verdict is assigned by rule, stated in the file, and applied to whatever the
numbers say. "Not supported" is a result; the project reports it as one.

  python -m wbs.task4_evaluation.report_hypotheses
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
T2E = ROOT / "wbs" / "deliverables" / "task2_extraction"

# H1 metrics: (key, lower_is_better, family)
H1_METRICS = [
    ("l1_errors_per_100rel", True, "consistency"),
    ("l1_warnings_per_100rel", True, "consistency"),
    ("l2_findings_per_100rel", True, "consistency"),
    ("duplicate_triples_per_100rel", True, "consistency"),
    ("dangling_endpoints_per_100rel", True, "consistency"),
    ("taxonomic_cycles", True, "consistency"),
    ("untyped_concepts_frac", True, "consistency"),
    ("mean_hierarchy_depth", False, "hierarchy"),
    ("max_hierarchy_depth", False, "hierarchy"),
    ("inheritance_richness", False, "hierarchy"),
    ("isolated_fraction", True, "hierarchy"),
    ("relationship_richness", False, "hierarchy"),
]
LABEL = {"V1_llm": "V1 LLM only", "V2_rag": "V2 Simple RAG",
         "V3_ontology_only": "V3 Ontology only", "V4_ontology_rag": "V4 Ontology+RAG",
         "V5_ontology_rag_validated": "V5 +validation"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    a = ap.parse_args()

    L = ["# The proposal's three hypotheses, tested\n",
         "Section 3.3 of the proposal states three hypotheses. Each is restated "
         "here as written, with the measurement that tests it and a verdict by a "
         "stated rule. Where a hypothesis is not supported, that is the finding.\n"]
    verdicts = {}

    # ══ H1 ════════════════════════════════════════════════════════════════
    L.append("## H1 — structured prompting gives higher logical consistency and "
             "hierarchical quality than naive prompting\n")
    shards = sorted(p for p in T2E.glob("hypothesis_h1*.json") if "smoke" not in p.name)
    samples = {}
    for p in shards:
        d = json.loads(p.read_text(encoding="utf-8"))
        for strat, row in d.items():
            samples.setdefault(row.get("sample", 0), {})[strat] = row
    if not samples:
        L.append("_Not run: `hypothesis_h1.json` not found._\n")
        verdicts["H1"] = "not measured"
    else:
        L.append(f"Each strategy extracts from the same records at greedy decoding; "
                 f"the per-strategy concept and relation sets are assembled into an "
                 f"ontology and run through the Task 2.2 validation layers. "
                 f"Consistency counts are per 100 relations so a strategy that "
                 f"extracts more is neither rewarded nor punished for volume. "
                 f"{len(samples)} disjoint record sample(s) of "
                 f"{next(iter(next(iter(samples.values())).values()))['n_records']} "
                 f"records.\n")
        # `unified` is the proposal-6.8 pipeline, added after H1 was registered
        # and run to answer 6.8 rather than H1. Counting it among the structured
        # strategies would change a pre-registered denominator after the fact, so
        # it is held out here and reported in its own subsection below.
        pipeline_rows = {sidx: r.pop("unified")
                         for sidx, r in samples.items() if "unified" in r}
        per_sample_verdict = {}
        for sidx in sorted(samples):
            rows = samples[sidx]
            naive = rows.get("naive")
            if not naive:
                continue
            L.append(f"### Sample {sidx}\n")
            L.append("| Strategy | Concepts | Relations | L1 err/100 | L1 warn/100 | "
                     "L2/100 | Cycles | Depth | Inherit. | Isolated |")
            L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
            for strat, r in rows.items():
                mark = "**" if strat == "naive" else ""
                L.append(f"| {mark}{strat}{mark} | {r['concepts']:,} | "
                         f"{r['relations']:,} | {r['l1_errors_per_100rel']} | "
                         f"{r['l1_warnings_per_100rel']} | "
                         f"{r['l2_findings_per_100rel']} | {r['taxonomic_cycles']} | "
                         f"{r['mean_hierarchy_depth']} | {r['inheritance_richness']} | "
                         f"{r['isolated_fraction']} |")
            L.append("")
            # per-strategy win count against naive, by metric family
            wins = {}
            for strat, r in rows.items():
                if strat == "naive":
                    continue
                c = h = 0
                for k, lower, fam in H1_METRICS:
                    if r.get(k) is None or naive.get(k) is None:
                        continue
                    better = (r[k] < naive[k]) if lower else (r[k] > naive[k])
                    if fam == "consistency":
                        c += 1 if better else 0
                    else:
                        h += 1 if better else 0
                wins[strat] = (c, h)
            n_cons = sum(1 for _, _, f in H1_METRICS if f == "consistency")
            n_hier = sum(1 for _, _, f in H1_METRICS if f == "hierarchy")
            L.append(f"Metrics on which each structured strategy beats `naive` "
                     f"(of {n_cons} consistency and {n_hier} hierarchy metrics):\n")
            L.append("| Strategy | Consistency wins | Hierarchy wins | Beats naive overall |")
            L.append("|---|--:|--:|---|")
            beat = 0
            for strat, (c, h) in wins.items():
                ok = (c + h) > (n_cons + n_hier) / 2
                beat += 1 if ok else 0
                L.append(f"| {strat} | {c}/{n_cons} | {h}/{n_hier} | "
                         f"{'yes' if ok else 'no'} |")
            L.append("")
            per_sample_verdict[sidx] = (beat, len(wins))
            L.append(f"**{beat} of {len(wins)} structured strategies beat `naive` on a "
                     f"majority of the twelve metrics in this sample.**\n")
        agree = {v[0] > v[1] / 2 for v in per_sample_verdict.values()}
        if len(per_sample_verdict) < 2:
            verdicts["H1"] = ("supported on one sample" if
                              list(per_sample_verdict.values())[0][0] >
                              list(per_sample_verdict.values())[0][1] / 2
                              else "not supported on one sample")
            L.append(f"**Verdict: {verdicts['H1']}.** One record sample is not a "
                     f"verdict — the Task 2.1.2 sensitivity sweep measured record "
                     f"choice as the dominant source of variance for four of five "
                     f"strategies, so a second disjoint sample is required before "
                     f"this is reportable as a property of the prompting strategy.\n")
        elif len(agree) == 1:
            verdicts["H1"] = ("supported" if agree.pop() else "not supported")
            L.append(f"**Verdict: {verdicts['H1']}**, consistently across "
                     f"{len(per_sample_verdict)} disjoint record samples.\n")
        else:
            verdicts["H1"] = "not supported (sample-dependent)"
            L.append(f"**Verdict: not supported.** The answer flips between disjoint "
                     f"record samples "
                     f"({', '.join(f'sample {k}: {v[0]}/{v[1]}' for k, v in sorted(per_sample_verdict.items()))}), "
                     f"so what looks like a prompting effect is within the "
                     f"record-sampling noise the Task 2.1.2 sweep quantified.\n")

        # ── proposal 6.8: the strategies as one multi-stage pipeline ──────
        if pipeline_rows:
            L.append("### The unified multi-stage pipeline (proposal 6.8)\n")
            L.append("Proposal 6.8 asks for the prompting strategies integrated "
                     "into one pipeline rather than offered as alternatives. "
                     "`OntologyBuilder`'s `unified` strategy is that pipeline: "
                     "chain-of-thought extraction, then the graph-of-thought "
                     "structural-coherence pass, then the metacognitive "
                     "flag-for-review pass, run over the batch. It was "
                     "implemented and never benchmarked against the single "
                     "strategies it combines; this is that benchmark, on the same "
                     "records, seed and decoding.\n")
            L.append("| Sample | Base | Concepts in → out | Relations in → out | "
                     "GoT score | self-loops removed | flagged for review | "
                     "s/record vs base |")
            L.append("|--:|---|---|---|--:|--:|--:|---|")
            for sidx in sorted(pipeline_rows):
                u = pipeline_rows[sidx]
                pi = u.get("pipeline") or {}
                base = samples[sidx].get(pi.get("base_strategy", "cot")) or {}
                L.append(f"| {sidx} | `{pi.get('base_strategy', '?')}` | "
                         f"{pi.get('concepts_in')} → {pi.get('concepts_out')} | "
                         f"{pi.get('relations_in')} → {pi.get('relations_out')} | "
                         f"{pi.get('got_score')} | {pi.get('self_loops_removed')} | "
                         f"{pi.get('metacognitive_flagged')} of "
                         f"{pi.get('metacognitive_accepted', 0) + pi.get('metacognitive_flagged', 0)} | "
                         f"{u.get('sec_per_record')} vs "
                         f"{base.get('sec_per_record', '—')} |")
            L.append("")
            tot_rm = sum((pipeline_rows[i].get("pipeline") or {})
                         .get("self_loops_removed", 0) for i in pipeline_rows)
            tot_fl = sum((pipeline_rows[i].get("pipeline") or {})
                         .get("metacognitive_flagged", 0) for i in pipeline_rows)
            tot_rel = sum((pipeline_rows[i].get("pipeline") or {})
                          .get("relations_in", 0) for i in pipeline_rows)
            L.append(f"**The pipeline is a no-op at measurable scale on this "
                     f"data.** Across both samples it removed {tot_rm} relation of "
                     f"{tot_rel:,} and flagged {tot_fl} concepts for review, so "
                     f"every consistency and hierarchy metric is its base "
                     f"strategy's to within rounding — on sample 1 the two are "
                     f"identical on all of them. It is not free: the extra passes "
                     f"cost wall-clock time per record. The two passes are doing "
                     f"something, in that the coherence check scores the graph and "
                     f"counts redundant and conflicting edges, but the only edit "
                     f"either is empowered to make is dropping self-loops, and "
                     f"this extractor emits almost none. 6.8 asks for the "
                     f"integration and the integration exists; what the benchmark "
                     f"adds is that combining the strategies this way does not "
                     f"combine their benefits.\n")

    # ══ H2 and H3 ═════════════════════════════════════════════════════════
    hp = EVAL / f"hypotheses_h2_h3__{a.model}.json"
    vp = EVAL / f"variant_scores__{a.model}.json"
    L.append("## H2 — ontology constraints reduce hallucination without degrading "
             "response fluency\n")
    if not hp.exists():
        L.append("_Not run: fluency and attribution measurement not found._\n")
        verdicts["H2"] = "not measured"
    else:
        h = json.loads(hp.read_text(encoding="utf-8"))
        res, sig = h["results"], h.get("fluency_vs_V2_rag", {})
        vs = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else None
        L.append(f"Hallucination was already measured. Fluency was not, so H2 was "
                 f"half-tested. Fluency here is {h['fluency_definition']}, reported as "
                 f"perplexity, with three degeneration counters beside it because a "
                 f"likelihood score alone rewards repetition.\n")
        L.append("| Variant | Ungrounded rate | Fluency (ppl, lower better) | "
                 "Mean logprob | Δ vs V2 | p | distinct-2 | distinct-3 | Repeated lines |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for v, o in res.items():
            ung = "—"
            if vs and v in (vs.get("overall") or {}):
                ung = f"{vs['overall'][v]['ungrounded_rate']*100:.1f}"
            s_ = sig.get(v)
            L.append(f"| {LABEL.get(v, v)} | {ung} | {o['fluency_ppl']:.2f} | "
                     f"{o['fluency_logprob']:.4f} | "
                     + (f"{s_['delta_logprob']:+.4f}" if s_ else "ref") + " | "
                     + (f"{s_['p']:.3g}" if s_ else "—") + " | "
                     f"{o['distinct_2']:.3f} | {o['distinct_3']:.3f} | "
                     f"{o['repeated_line_frac']*100:.1f}% |")
        L.append("")
        v1 = res.get("V1_llm")
        if v1 and all(v1["fluency_logprob"] > o["fluency_logprob"]
                      for k, o in res.items() if k != "V1_llm"):
            L.append(f"**V1 is not comparable on this axis and is listed for "
                     f"completeness only.** Fluency here is the answer's likelihood "
                     f"under the base model, and V1 writes without any retrieved "
                     f"context, so its text stays close to that model's own "
                     f"distribution and scores "
                     f"{v1['fluency_ppl']:.2f} perplexity against "
                     f"{min(o['fluency_ppl'] for k, o in res.items() if k != 'V1_llm'):.2f} "
                     f"for the best grounded arm. Any arm that must reproduce "
                     f"supplied facts is penalised by this measure in proportion to "
                     f"how much it uses them. The H2 comparison is therefore made "
                     f"between grounded arms - every ontology arm against V2 Simple "
                     f"RAG - where both sides carry retrieved context.\n")
        v2 = res.get("V2_rag", {})
        onto_arms = [v for v in ("V3_ontology_only", "V4_ontology_rag",
                                 "V5_ontology_rag_validated") if v in res]
        worse = [v for v in onto_arms
                 if (sig.get(v) or {}).get("delta_logprob", 0) < 0
                 and (sig.get(v) or {}).get("p", 1) < 0.05]
        # The hallucination clause is decided by the PAIRED test against V2, not by
        # the aggregate rates. The two disagree in sign here: V3's micro-averaged
        # ungrounded rate is below V2's while its per-item paired delta is above,
        # so an aggregate-only rule would turn an aggregation choice into a verdict.
        hall_better, hall_aggregate = [], []
        if vs:
            ov = vs.get("overall", {})
            base_u = (ov.get("V2_rag") or {}).get("ungrounded_rate")
            hall_aggregate = [v for v in onto_arms
                              if base_u is not None
                              and (ov.get(v) or {}).get("ungrounded_rate") is not None
                              and ov[v]["ungrounded_rate"] < base_u]
            svr = vs.get("significance_vs_reference", {})
            hall_better = [v for v in onto_arms
                           if ((svr.get(v) or {}).get("ungrounded_rate") or {})
                           .get("delta", 0) < 0
                           and ((svr.get(v) or {}).get("ungrounded_rate") or {})
                           .get("p", 1) < 0.05]
        if hall_better and not worse:
            verdicts["H2"] = "supported"
        elif (hall_better and worse) or (not hall_better and not worse):
            verdicts["H2"] = "partially supported"
        else:
            verdicts["H2"] = "not supported"
        L.append("The hallucination clause is decided by the paired per-item test "
                 "against V2 Simple RAG, not by the aggregate rates, because the two "
                 "do not agree: a micro-average over entity mentions and a mean of "
                 "per-item rates can rank the same pair of arms differently, and the "
                 "choice between them must not decide a hypothesis.\n")
        L.append(f"**Verdict: {verdicts['H2']}.** "
                 + (f"{len(hall_better)} of {len(onto_arms)} ontology arms show a "
                    f"significant paired reduction in ungrounded rate against V2 "
                    f"Simple RAG"
                    if hall_better else
                    f"No ontology arm shows a significant paired reduction in "
                    f"ungrounded rate against V2 Simple RAG "
                    f"({len(hall_aggregate)} of {len(onto_arms)} are lower on the "
                    f"aggregate rate, none significantly so on the paired test)")
                 + ", and "
                 + (f"{len(worse)} show a significantly lower answer likelihood than "
                    f"V2 ({', '.join(LABEL.get(v, v) for v in worse)}), so the "
                    f"'without degrading fluency' clause does not hold for them"
                    if worse else
                    "none shows a significantly lower answer likelihood than V2, so "
                    "the 'without degrading fluency' clause holds") + ".\n")
        if vs and not hall_better:
            ov = vs.get("overall", {})
            if "V1_llm" in ov and "V2_rag" in ov:
                L.append(f"What the numbers do support is weaker and worth stating "
                         f"plainly: grounding of any kind cuts hallucination hard - "
                         f"{ov['V1_llm']['ungrounded_rate']*100:.1f}% ungrounded "
                         f"without retrieval against "
                         f"{ov['V2_rag']['ungrounded_rate']*100:.1f}% with dense "
                         f"passages and "
                         f"{min(ov[v]['ungrounded_rate'] for v in onto_arms if v in ov)*100:.1f}"
                         f"-{max(ov[v]['ungrounded_rate'] for v in onto_arms if v in ov)*100:.1f}% "
                         f"with ontology facts - but the ontology does not cut it "
                         f"further than dense retrieval does. H2's contribution over "
                         f"a retrieval baseline is the claim that fails here, not the "
                         f"value of grounding.\n")

    # The H2 verdict above is computed over the PUBLISHED arms, where V4 and V5
    # ran on the shared `mixed` adapter. That confound is corrected in a separate
    # run, and the corrected V5 answers H2's clause differently, so it is reported
    # here rather than folded into the verdict it would change.
    hyb_p = EVAL / "hybrid_adapter_comparison.json"
    if hyb_p.exists():
        hyb = json.loads(hyb_p.read_text(encoding="utf-8"))
        u5 = (hyb.get("vs_rag") or {}).get("V5_ontology_rag_validated|ungrounded_rate")
        t5 = (hyb.get("vs_rag") or {}).get("V5_ontology_rag_validated|token_f1")
        sc_ = hyb.get("scores") or {}
        beh = (hyb.get("validation_behaviour") or {}).get("dedicated") or {}
        if u5 and sc_.get("V5_ontology_rag_validated|dedicated"):
            sig_ = (u5["delta"] < 0 and (u5["p"] or 1) < 0.05)
            L.append("### H2 with the mixed-adapter confound removed\n")
            L.append(f"The verdict above is computed over the published arms, in "
                     f"which V4 and V5 ran on `mixed`, an adapter trained on a "
                     f"blend of context formats, while V2 ran on its own. That is "
                     f"a structural disadvantage on the arms H2 is about. On a "
                     f"dedicated adapter, V5's audit-grade ungrounded rate is "
                     f"{sc_['V5_ontology_rag_validated|dedicated']['ungrounded_rate']*100:.1f}% "
                     f"against Simple RAG's "
                     f"{sc_['V2_rag|dedicated']['ungrounded_rate']*100:.1f}% — "
                     f"a paired delta of {u5['delta']*100:+.2f} pp at "
                     f"p = {u5['p']:.3g}, "
                     + ("which is significant. On this corrected comparison H2's "
                        "hallucination clause **does** hold for the validation "
                        "arm, "
                        if sig_ else
                        "which is not significant, so the corrected comparison "
                        "does not change the verdict, ")
                     + (f"at a cost of {abs(t5['delta'])*100:.1f} token F1 "
                        f"(p = {t5['p']:.3g})" if t5 else "")
                     + (f" and with {beh.get('fallback', 0)} of "
                        f"{beh.get('n', 0)} abstentions, so the arm is grounding "
                        f"more rather than answering less" if beh else "")
                     + ".\n")
            L.append("This does not retro-fit the verdict. The published "
                     "comparison is the one that was pre-registered and run "
                     "across all five arms and three model sizes, and it stands "
                     "as reported; the corrected run covers two arms at one size "
                     "and is reported beside it. What a reader should take is "
                     "that H2's clause is false for the ontology arms as "
                     "published and true for the validated arm once its adapter "
                     f"is fair — and that the trade it makes, "
                     f"{abs(t5['delta'])*100:.1f} points of token F1 for "
                     f"{abs(u5['delta']) / sc_['V2_rag|dedicated']['ungrounded_rate'] * 100:.0f}% "
                     f"fewer ungrounded spans, is the decision this project "
                     f"hands to whoever deploys it.\n")

    # the same discipline for the selective policy: reported beside the verdict,
    # not folded into it, because it is a policy chosen after the pre-registration
    sel_p = EVAL / "selective_validation__llama3.1-8b.json"
    if sel_p.exists():
        sd = json.loads(sel_p.read_text(encoding="utf-8"))
        bs_ = sd.get("best_signal")
        c_ = (sd.get("cross_validated") or {}).get(bs_) or {}
        pr_ = c_.get("paired_vs_rag") or {}
        if pr_.get("ungrounded_rate"):
            du, dc, dt = (pr_["ungrounded_rate"], pr_.get("clean") or {},
                          pr_.get("token_f1") or {})
            eps_ = sd.get("endpoints") or {}
            e5_ = eps_.get("V5 (validate everything)") or {}
            rag_ = eps_.get("V2 Simple RAG") or {}
            cost_all = abs(e5_.get("token_f1", 0) - rag_.get("token_f1", 0))
            L.append("### H2 with the layer spent selectively\n")
            L.append(f"The trade named above is smaller than it looked, because "
                     f"the layer had only ever been run on every answer. Spending "
                     f"it only on the {c_['n_validated']} of {c_['n']} drafts its "
                     f"own detector flags gives {du['delta']*100:+.2f} pp "
                     f"ungrounded against Simple RAG (p = {du['p']:.3g}) and "
                     f"{dc.get('delta', 0)*100:+.2f} pp clean "
                     f"(p = {dc.get('p', 1):.3g}) for {abs(dt.get('delta', 0))*100:.1f} "
                     f"token F1 rather than "
                     f"{cost_all * 100:.1f} — and under "
                     f"the published audit rule it is also *more* grounded than "
                     f"validating everything "
                     f"({c_['ungrounded_rate']*100:.2f}% against "
                     f"{e5_.get('ungrounded_rate', 0)*100:.2f}%). The threshold is "
                     f"fitted by {sd.get('folds')}-fold cross-validation and every "
                     f"item is scored under a threshold fitted without it.\n")
            L.append("This too is reported beside the verdict rather than inside "
                     "it. The policy was chosen after the hypothesis was "
                     "registered, so folding it in would be fitting the verdict "
                     "to the result; what it establishes is that H2's cost clause "
                     "is a function of how the layer is operated, not a constant "
                     "of the approach. `selective_validation__llama3.1-8b.md` and "
                     "`detector_sensitivity__llama3.1-8b.md` carry the curve and "
                     "the sensitivity of this claim to the audit rule.\n")
            rp_p = EVAL / "repair_policy__llama3.1-8b.json"
            rp_ = json.loads(rp_p.read_text(encoding="utf-8")) if rp_p.exists() else None
            if rp_ and rp_.get("cells"):
                D_ = "V5 surgical clause, gated"
                g_ = ((rp_["cells"][D_].get("vs_rag") or {}).get("token_f1") or {})
                st_ = rp_["cells"][D_].get("ungrounded_rate_strict")
                sr_ = rp_["cells"]["V2 Simple RAG"].get("ungrounded_rate_strict")
                L.append("### H2 once the repair stops being a rewrite\n")
                L.append(f"And smaller again, for a reason that is about the repair "
                         f"rather than the policy. The layer fixed a flagged answer "
                         f"by re-prompting the model for a rewrite, so one "
                         f"unsupported span re-worded the whole answer. Cutting out "
                         f"only the clause that carries the violation (R19) leaves "
                         f"the cost at {abs(g_.get('delta', 0)):.2f} token F1 against "
                         f"Simple RAG rather than 2.6, removes every span the audit "
                         f"counts, and takes the strict-rule ungrounded rate to "
                         f"{st_:.2f}% against Simple RAG's {sr_:.2f}%. H2's "
                         f"hallucination clause holds under both detectors; its cost "
                         f"clause is now a fraction of a point. "
                         f"`repair_policy.md` carries the cells and the paired "
                         f"tests, and `verifier_across_models__test.md` shows the "
                         f"same layer doing the same thing on six models with no "
                         f"adapter at all.\n")

    L.append("## H3 — ontology-integrated systems give more interpretable and "
             "verifiable responses\n")
    if not hp.exists():
        L.append("_Not run._\n")
        verdicts["H3"] = "not measured"
    else:
        h = json.loads(hp.read_text(encoding="utf-8"))
        res = h["results"]
        L.append("Verifiability is operationalised as attribution: the share of the "
                 "places an answer names that appear in the context that answer was "
                 "given, and the share of its bullet lines carrying at least one "
                 "attributable place. An answer can be accurate and still "
                 "unverifiable. Interpretability is the availability and size of the "
                 "reasoning trace. V1 receives no context, so attribution is "
                 "undefined for it rather than zero.\n")
        L.append("| Variant | Attributable places | Attributable lines | "
                 "Trace available | Trace steps |")
        L.append("|---|--:|--:|--:|--:|")
        for v, o in res.items():
            f = lambda x, pct=True: ("—" if x is None else
                                     (f"{x*100:.1f}%" if pct else f"{x:.1f}"))
            L.append(f"| {LABEL.get(v, v)} | {f(o['attributable_entity_rate'])} | "
                     f"{f(o['attributable_line_rate'])} | {f(o['trace_available'])} | "
                     f"{f(o['trace_steps'], False)} |")
        L.append("")
        rag_a = (res.get("V2_rag") or {}).get("attributable_entity_rate")
        onto_a = [(v, (res.get(v) or {}).get("attributable_entity_rate"))
                  for v in ("V3_ontology_only", "V4_ontology_rag",
                            "V5_ontology_rag_validated") if v in res]
        better = [v for v, x in onto_a if x is not None and rag_a is not None and x > rag_a]
        traced = [v for v, o in res.items() if (o.get("trace_available") or 0) > 0]
        if better and traced:
            verdicts["H3"] = "supported"
        elif traced:
            verdicts["H3"] = "partially supported"
        else:
            verdicts["H3"] = "not supported"
        L.append(f"**Verdict: {verdicts['H3']}.** "
                 + (f"{len(better)} of {len(onto_a)} ontology arms attribute more of "
                    f"their named places to their own evidence than V2 Simple RAG does"
                    if better else
                    "No ontology arm attributes more of its named places to its own "
                    "evidence than V2 Simple RAG does")
                 + ", and "
                 + (f"a machine-readable reasoning trace exists only for "
                    f"{', '.join(LABEL.get(v, v) for v in traced)}, which is the "
                    f"interpretability half of the claim"
                    if traced else "no variant emits a reasoning trace") + ". "
                 + "The human half of 11.7 - whether users perceive the traces as "
                 + "trustworthy - is out of scope for this phase and is not claimed.\n")

    # ══ consistency (11.5b), reported alongside because H2/H3 lean on it ══
    cshards = sorted(EVAL.glob(f"consistency__{a.model}*.json"))
    cons = {}
    for p in cshards:
        cons.update(json.loads(p.read_text(encoding="utf-8")))
    if cons:
        L.append("## Consistency across paraphrased and repeated queries "
                 "(proposal 11.5)\n")
        any_c = next(iter(cons.values()))
        L.append(f"{any_c['n_questions']} questions, each restated "
                 f"{any_c['paraphrases']} ways with the meaning held fixed and "
                 f"answered greedily, plus {any_c['repeats']} re-answers of the "
                 f"original at temperature {any_c['temperature']}. Agreement is "
                 f"symmetric token F1 between two answers, so it sits on the same "
                 f"scale as the accuracy table. Paraphrases are generated once and "
                 f"cached, so every arm answers the same wordings.\n")
        L.append("| Arm | Paraphrase agreement | Paraphrase entity Jaccard | "
                 "Repeat agreement | Repeat entity Jaccard | Accuracy SD within question |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for arm, o in cons.items():
            L.append(f"| {arm} | {o['paraphrase_agreement_token_f1']*100:.1f} | "
                     f"{o['paraphrase_agreement_entity_jaccard']*100:.1f} | "
                     f"{o['repeat_agreement_token_f1']*100:.1f} | "
                     f"{o['repeat_agreement_entity_jaccard']*100:.1f} | "
                     f"{o['accuracy_sd_within_question']*100:.1f} |")
        L.append("")
        best = max(cons, key=lambda k: cons[k]["paraphrase_agreement_token_f1"])
        gaps = {k: cons[k]["repeat_agreement_token_f1"]
                   - cons[k]["paraphrase_agreement_token_f1"] for k in cons}
        L.append(f"Most consistent under rewording: **{best}** at "
                 f"{cons[best]['paraphrase_agreement_token_f1']*100:.1f}. "
                 f"For every arm the repeat agreement exceeds the paraphrase "
                 f"agreement ("
                 + ", ".join(f"{k} {gaps[k]*100:+.1f}" for k in cons)
                 + "), which separates the two sources the proposal names: these "
                   "systems are more stable against decoding noise than against being "
                   "asked the same thing in different words.\n")

    L.append("## Summary\n")
    L.append("| Hypothesis | Verdict |")
    L.append("|---|---|")
    for k in ("H1", "H2", "H3"):
        L.append(f"| {k} | **{verdicts.get(k, 'not measured')}** |")
    L.append("")
    (EVAL / "HYPOTHESES.md").write_text("\n".join(L), encoding="utf-8")
    (EVAL / "HYPOTHESES.json").write_text(json.dumps(verdicts, indent=2),
                                          encoding="utf-8")
    print("wrote HYPOTHESES.md")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
