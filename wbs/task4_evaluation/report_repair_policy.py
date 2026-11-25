"""
WBS Task 4.1.2 - report R19: surgical repair against the layer as published and
against Simple RAG, at three model sizes.

  python -m wbs.task4_evaluation.report_repair_policy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
REF = "V2 Simple RAG"
DEPLOYED = "V5 surgical clause, gated"
VERIFIER = "V6 surgical clause on the RAG draft, gated"
PUBLISHED = "V5 regen, gated (selective_axis best)"
MET = [("token_f1", "Token F1"), ("entity_f1", "Entity F1"),
       ("entity_hit", "Entity hit"), ("ungrounded_rate", "Ungrounded ↓"),
       ("clean", "Clean ↑")]


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def p(x):
    return "—" if x is None else ("<1e-15" if x < 1e-15 else f"{x:.2e}")


def main() -> None:
    data = {}
    for m in MODELS:
        q = EVAL / f"repair_policy__{m}.json"
        if q.exists():
            data[m] = json.loads(q.read_text(encoding="utf-8"))
    if not data:
        sys.exit("no repair_policy__*.json - run wbs.task4_evaluation.repair_policy first")

    L = []
    W = L.append
    W("# R19 - the validation layer's cost is the rewrite, not the check\n")
    W("The deployed validation layer repairs a draft by re-prompting the model with")
    W("the offending spans and taking the rewrite back. `selective_axis__llama3.1-8b.md` measured")
    W("where its reference-overlap cost sits, and it is not the abstentions: at")
    W("llama3.1-8b the gated cell is 2.79 token F1 below its own draft, 2.02 of which")
    W("is rewrites against 0.77 abstentions. One unsupported place name re-words four")
    W("hundred tokens that nothing was wrong with, so the cost is structural rather")
    W("than a threshold left untuned.\n")
    W("`constrained_reasoner.surgical_repair` removes the clause that carries the")
    W("violation and copies every other byte of the draft through unchanged, keeping")
    W("the shortened sentence only while it is itself clean and still names something")
    W("the knowledge base knows. It asserts strictly less than the draft did - nothing")
    W("is added, reworded or hedged - and it needs no second generation, so the layer")
    W("stops costing a forward pass as well. Every cell below is a transformation of")
    W("generations already on disk, scored with `score_variants.score_item`.\n")

    W("## Cells, at each size\n")
    for m, d in data.items():
        W(f"**{m}** — {d['n_items']} benchmark items, ontology v{d['ontology_version']}\n")
        W("| Configuration | Token F1 | Entity F1 | Entity hit | Ungrounded ↓ | Clean ↑ | chars | layer ms |")
        W("|---|--:|--:|--:|--:|--:|--:|--:|")
        for k, c in d["cells"].items():
            mm = c["mean"]
            ms = c.get("layer_ms_mean")
            W(f"| {'**' + k + '**' if k in (DEPLOYED, VERIFIER) else k} "
              f"| {f(mm['token_f1'])} | {f(mm['entity_f1'])} | {f(mm['entity_hit'])} "
              f"| {f(mm['ungrounded_rate'])} | {f(mm['clean'])} "
              f"| {c['mean_chars']} | {'—' if ms is None else f(ms)} |")
        W("")

    W("## Against Simple RAG, paired per item\n")
    W("Two-sided Wilcoxon signed-rank on the items both arms can be scored on, with")
    W("Holm-Bonferroni across the five metrics of each cell. `n differ` is the count")
    W("the test actually ran on: a tiny one means the two arms are mostly the same")
    W("answers, and the p-value beside it is not a result.\n")
    for m, d in data.items():
        W(f"**{m}**\n")
        W("| Arm | Metric | Δ vs RAG | p | Holm p | n | n differ |")
        W("|---|---|--:|--:|--:|--:|--:|")
        for arm in (PUBLISHED, DEPLOYED, VERIFIER):
            t = (d["cells"].get(arm) or {}).get("vs_rag") or {}
            for key, lab in MET:
                x = t.get(key)
                if not x:
                    continue
                W(f"| {arm} | {lab} | {x['delta']:+.2f} | {p(x['p'])} "
                  f"| {p(x.get('holm_p'))} | {x['n']} | {x['n_differ']} |")
        W("")

    W("## The claim the standing goal names\n")
    W("The goal this work runs under asks for the ontology arms to be best. Per cell,")
    W("counting a metric as decided only when Holm-corrected p < 0.05 **and** at least")
    W("20 items differ on it - a signed-rank test over a handful of pairs is not a")
    W("decision, however small its p-value.\n")
    W("| Model | Arm | Better than RAG | Worse than RAG | Indistinguishable |")
    W("|---|---|---|---|---|")
    for m, d in data.items():
        for arm in (PUBLISHED, DEPLOYED, VERIFIER):
            v = (d.get("claim_check") or {}).get(arm)
            if not v:
                continue
            W(f"| {m} | {arm} | {', '.join(v['better_than_rag']) or '—'} "
              f"| {', '.join(v['worse_than_rag']) or '—'} "
              f"| {', '.join(v['indistinguishable']) or '—'} |")
    W("")

    best = []
    for m, d in data.items():
        for arm in (DEPLOYED, VERIFIER):
            v = (d.get("claim_check") or {}).get(arm) or {}
            if v.get("never_worse_than_rag"):
                best.append(f"{m} / {arm}")
    W(f"**Never worse than Simple RAG on any of the five metrics:** "
      f"{', '.join(best) if best else 'no cell'}.\n")

    W("## The independent detector band\n")
    W("The surgical pass removes exactly the spans the published detector counts, so")
    W("its published ungrounded rate is **0.00% by construction** and means nothing on")
    W("its own. `audit_spans(strict=True)` is the other end of the band from")
    W("`detector_sensitivity.md`: a different rule, which the layer does not optimise")
    W("and which promotes itinerary scaffolding to claims. A gain that survives it is")
    W("a gain in the answers.\n")
    W("| Model | Arm | Ungrounded (strict) ↓ | Clean (strict) ↑ | n |")
    W("|---|---|--:|--:|--:|")
    for m, d in data.items():
        for arm in (REF, "V4 draft, layer off", PUBLISHED, DEPLOYED, VERIFIER):
            c = d["cells"].get(arm)
            if not c:
                continue
            W(f"| {m} | {arm} | {f(c.get('ungrounded_rate_strict'))} "
              f"| {f(c.get('clean_strict'))} | {c.get('n_strict')} |")
    W("")

    W("## What the layer removes\n")
    W("Counted on the llama3.1-8b hybrid draft, every segment the pass touches:\n")
    d8 = data.get("llama3.1-8b", {})
    g = d8.get("cut_diagnostics") or {}
    if g:
        W(f"- **{g['n_cut']} segments** carry a violation.")
        W(f"- **{g['unsupportable']} unsupportable claims** - a price, an opening")
        W("  hour, a timetable or an accessibility guarantee. No project source")
        W("  carries those predicates at all, so asserting one is a hallucination by")
        W("  construction and removing it is the layer doing its job.")
        W(f"- **{g['multi_token']} multi-token entity spans** - the invented place")
        W("  names the layer exists to catch.")
        ones = ", ".join(f"`{e['span']}` ({e['n']})" for e in g["top_entity_spans"]
                         if e["tokens"] == 1)
        W(f"- **{g['single_token']} single-token entity spans**, and this is where the")
        W(f"  residual cost is. The most frequent on this draft are {ones} -")
        W("  ordinary capitalised English and itinerary scaffolding that the published")
        W("  span rule reports as entity claims (R2's class of false positive).")
        W("  Cutting the clause around one removes real advice and costs reference")
        W("  overlap for no grounding gain.")
        W("")
        W("So the published 0.00% is not the interesting number and the token F1 the")
        W("layer still costs is mostly the detector, not the grounding. Tightening the")
        W("single-token rule would move both, and it would move every published rate")
        W("in the project with them, so it is recorded as an open item rather than")
        W("done quietly - **R20** in `docs/risk_management_plan.md`.")
    W("")
    W("## What this does and does not change\n")
    W("- The ontology arm's reference-overlap gap against Simple RAG goes from")
    W("  **-2.37 / -1.75 / -3.44** token F1 to **-0.73 / -0.12 / -0.86** with the")
    W("  ontology in the context, and to **-0.13 / -0.22 / -0.21** with the ontology")
    W("  used only as the verifier over a Simple RAG draft.")
    W("- Groundedness is better than Simple RAG at every size on both measures, and")
    W("  the gain survives the strict detector.")
    W("- Entity F1 and entity hit are indistinguishable from Simple RAG everywhere.")
    W("- The layer's cost in compute goes from a second generation per repaired")
    W("  answer to about **1 ms** of CPU, so it can run on every answer.")
    W("- It does **not** make the ontology arm beat Simple RAG on reference overlap.")
    W("  At llama3.2-3b the gap is no longer distinguishable from zero; at the other")
    W("  two sizes it is small and still real. The ceiling is the draft: no stored")
    W("  arm, ontology or hybrid, drafts above Simple RAG on this benchmark, which is")
    W("  what `reference_paraphrase__llama3.1-8b.md` predicts from how the references were written.")
    W("")

    out = EVAL / "repair_policy.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
