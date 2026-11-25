"""
WBS Task 2.1.2 - report for the prompt-strategy sensitivity and stability sweep.

Merges the per-GPU shards written by prompt_sensitivity.py and answers the two
questions the single-run comparison could not:

  1. Does a strategy give the same answer twice?  (stability: metric SD across
     identical re-runs, and concept-set Jaccard between those re-runs)
  2. Is the published ranking a property of the strategies or of the 40 records they
     happened to see?  (sensitivity: metric SD across disjoint samples, plus per-sample
     ranks and Kendall's tau between samples)

  python -m wbs.task2_extraction.report_sensitivity
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task2_extraction"
PRIOR = ROOT / "artifacts" / "reports" / "prompt_comparison_llm.json"

# the score the original report ranked on, so the ranking question is asked the same way
def score(m: dict) -> float:
    return (m["schema_validity_rate"] * m["parse_success_rate"]
            * (1 + m["concepts_per_record"] / 10))


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same items."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return float("nan")
    ra = {s: i for i, s in enumerate(a)}
    rb = {s: i for i, s in enumerate(b)}
    conc = disc = 0
    for x, y in itertools.combinations(common, 2):
        s = (ra[x] - ra[y]) * (rb[x] - rb[y])
        conc += s > 0
        disc += s < 0
    n = conc + disc
    return (conc - disc) / n if n else float("nan")


def main() -> None:
    shards = sorted(OUT.glob("prompt_sensitivity*.json"))
    shards = [p for p in shards if "smoke" not in p.name]
    if not shards:
        sys.exit("no sweep shards found; run wbs.task2_extraction.prompt_sensitivity")
    merged, meta = {}, None
    for p in shards:
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = meta or d
        merged.update(d["strategies"])
    S = list(merged)
    n_stab = meta["stability_runs"]
    n_sens = meta["sensitivity_samples"]
    # the original comparison's own record count, read from its JSON rather than hard-coded
    prior_n = "40"
    if PRIOR.exists():
        pr = json.loads(PRIOR.read_text(encoding="utf-8"))["results"]
        ns = {r["n_records"] for r in pr if "n_records" in r}
        if len(ns) == 1:
            prior_n = str(ns.pop())

    L = [f"# Task 2.1.2 - Prompt Strategy Sensitivity and Stability\n",
         f"`artifacts/reports/prompt_comparison_llm.md` ranks five extraction "
         f"strategies on one {prior_n}-record greedy run, "
         f"which gives a point estimate per strategy and no error bar. This sweep adds "
         f"the error bars. {len(S)} strategies, {meta['n_records']} records per run, "
         f"{n_stab} stability re-runs at temperature {meta['temperature']} plus "
         f"{n_sens} disjoint record samples at greedy, seed {meta['seed']}. "
         f"Decoding is {meta['decoding']} — absolute values are therefore not "
         f"comparable with the original 4-bit report, but the variances and the rank "
         f"agreement, which are the point, are internally consistent.\n"]

    # ── stability ────────────────────────────────────────────────────────────
    L.append("## Stability: does a strategy give the same answer twice?\n")
    L.append(f"Same {meta['n_records']} records, same prompt, {n_stab} runs at "
             f"temperature {meta['temperature']}. *Concept agreement* is the mean "
             f"pairwise Jaccard of the per-record extracted concept-name sets — 1.00 "
             f"means two runs named exactly the same concepts, 0.50 means half.\n")
    L.append("| Strategy | Concepts/rec | Parse rate | Schema validity | "
             "Validation pass | Concept agreement |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for s in S:
        g = merged[s]["stability"]["agg"]
        j = merged[s]["stability"]["concept_set_jaccard_mean"]
        jc = f"**{j:.3f}**" if j is not None else "—"
        L.append(f"| {s} | {g['concepts_per_record']['mean']:.2f} "
                 f"± {g['concepts_per_record']['sd']:.2f} | "
                 f"{g['parse_success_rate']['mean']:.2f} "
                 f"± {g['parse_success_rate']['sd']:.2f} | "
                 f"{g['schema_validity_rate']['mean']:.3f} "
                 f"± {g['schema_validity_rate']['sd']:.3f} | "
                 f"{g['validation_pass_rate']['mean']:.2f} "
                 f"± {g['validation_pass_rate']['sd']:.2f} | {jc} |")
    L.append("")
    js = {s: merged[s]["stability"]["concept_set_jaccard_mean"] for s in S
          if merged[s]["stability"]["concept_set_jaccard_mean"] is not None}
    if js:
        best, worst = max(js, key=js.get), min(js, key=js.get)
        L.append(f"**Most reproducible: `{best}` at {js[best]:.3f} concept agreement; "
                 f"least: `{worst}` at {js[worst]:.3f}.** Agreement well below 1.0 means "
                 f"re-running the same strategy on the same text yields a materially "
                 f"different concept set, so the ontology that gets built depends on "
                 f"the sampling draw as well as the strategy. For ontology construction "
                 f"that matters more than a fractional difference in mean concept count: "
                 f"a strategy is only usable if its output is stable enough to audit.\n")

    # ── sensitivity ──────────────────────────────────────────────────────────
    L.append("## Sensitivity: strategy or sample?\n")
    L.append(f"{n_sens} disjoint record samples of {meta['n_records']} records, greedy "
             f"decoding, so the only thing changing is which records the strategy "
             f"sees.\n")
    L.append("| Strategy | Concepts/rec | Parse rate | Schema validity | "
             "Score (orig. formula) |")
    L.append("|---|--:|--:|--:|--:|")
    per_sample_scores = {s: [] for s in S}
    for s in S:
        g = merged[s]["sensitivity"]["agg"]
        runs = merged[s]["sensitivity"]["runs"]
        sc = [score(r) for r in runs]
        per_sample_scores[s] = sc
        L.append(f"| {s} | {g['concepts_per_record']['mean']:.2f} "
                 f"± {g['concepts_per_record']['sd']:.2f} | "
                 f"{g['parse_success_rate']['mean']:.2f} "
                 f"± {g['parse_success_rate']['sd']:.2f} | "
                 f"{g['schema_validity_rate']['mean']:.3f} "
                 f"± {g['schema_validity_rate']['sd']:.3f} | "
                 f"{st.mean(sc):.3f} ± "
                 f"{st.stdev(sc) if len(sc) > 1 else 0:.3f} |")
    L.append("")

    # ── does the ranking survive resampling? ─────────────────────────────────
    k = len(next(iter(per_sample_scores.values())))
    rankings = []
    for i in range(k):
        order = sorted(S, key=lambda s: per_sample_scores[s][i], reverse=True)
        rankings.append(order)
    L.append("### Does the ranking survive resampling?\n")
    L.append("| Sample | Ranking, best first |")
    L.append("|---|---|")
    for i, r in enumerate(rankings, 1):
        L.append(f"| {i} | " + " > ".join(f"`{x}`" for x in r) + " |")
    L.append("")
    taus = [kendall_tau(a, b) for a, b in itertools.combinations(rankings, 2)]
    taus = [t for t in taus if t == t]
    winners = {r[0] for r in rankings}
    mean_sc = {s: st.mean(per_sample_scores[s]) for s in S}
    overall = sorted(S, key=lambda s: mean_sc[s], reverse=True)
    if taus:
        L.append(f"Mean pairwise Kendall's tau between sample rankings: "
                 f"**{st.mean(taus):+.2f}** (1.0 = identical order, 0 = unrelated, "
                 f"-1.0 = reversed), range {min(taus):+.2f} to {max(taus):+.2f}. ")
    if len(winners) == 1:
        L.append(f"The same strategy, `{list(winners)[0]}`, ranks first in every "
                 f"sample, so the top of the ranking is a property of the strategy and "
                 f"not of the sample.\n")
    else:
        L.append(f"**{len(winners)} different strategies take first place across "
                 f"{k} samples** ({', '.join('`%s`' % w for w in sorted(winners))}). "
                 f"The published single-run ranking therefore cannot be read as a "
                 f"strategy ordering: at this sample size the choice of records moves "
                 f"the winner. Aggregated over all samples the order is "
                 f"{' > '.join('`%s`' % x for x in overall)}, and that aggregate is the "
                 f"only version of the ranking the evidence supports.\n")

    # ── which axis dominates ─────────────────────────────────────────────────
    L.append("## Which source of variance dominates\n")
    L.append("| Strategy | SD across re-runs (sampling) | SD across samples (records) | "
             "Dominant source |")
    L.append("|---|--:|--:|---|")
    for s in S:
        a = merged[s]["stability"]["agg"]["concepts_per_record"]["sd"]
        b = merged[s]["sensitivity"]["agg"]["concepts_per_record"]["sd"]
        L.append(f"| {s} | {a:.2f} | {b:.2f} | "
                 f"{'records' if b > a else 'decoding'} |")
    L.append("")
    recs = sum(merged[s]["sensitivity"]["agg"]["concepts_per_record"]["sd"]
               > merged[s]["stability"]["agg"]["concepts_per_record"]["sd"] for s in S)
    L.append(f"For {recs} of {len(S)} strategies the between-sample spread exceeds the "
             f"between-run spread, i.e. *which records you evaluate on* moves the "
             f"concept count more than *the decoding draw* does. "
             + ("That is the expected ordering, and it is why a 40-record single run is "
                "too thin a basis for a ranking.\n"
                if recs > len(S) / 2 else
                "That is the reverse of the expected ordering: decoding noise exceeds "
                "sample noise, which points at the prompts admitting too much freedom "
                "rather than at the sample being small.\n"))

    L.append("## Cost\n")
    L.append("| Strategy | Sec/record |")
    L.append("|---|--:|")
    for s in S:
        spr = merged[s]['sec_per_record']
        # null when the strategy processed no records (R15); do not print 0.0
        L.append(f"| {s} | " + (f"{spr:.1f}" if spr is not None else "—") + " |")
    L.append("")

    p = OUT / "prompt_sensitivity.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {p.name}  ({len(S)} strategies from {len(shards)} shard(s))")


if __name__ == "__main__":
    main()
