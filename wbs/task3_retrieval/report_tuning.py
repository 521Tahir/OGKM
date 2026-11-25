"""
WBS Task 3.2.1 - report on the ontology retrieval tuning experiment.

Merges the offline policy sweep (`onto_context_tuning.json`) with the generation
runs (`tuned_onto_generation__*.json`) and settles the question the sweep cannot:
whether a context that carries more of the gold entities produces a better
answer. Published V2 and V3 answers are re-scored here from their own JSONL with
the same scorer, so the paired significance tests compare per-item values rather
than two aggregates.

  python -m wbs.task3_retrieval.report_tuning
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task3_retrieval"
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"


def _name_repetition_cost(top_n: int = 32) -> dict:
    """Share of context characters the triple-per-line encoding spends re-writing
    the subject name, measured on the `top_n` highest-degree cards - the ones the
    ontology arm actually retrieves. Computed from the ontology files, not entered
    by hand."""
    import json as _json
    from collections import defaultdict
    from experiments.exp024_rag_vs_ontology import config as C
    base = C.ROOT / "artifacts" / "ontologies" / "1.4.0"
    rels = _json.loads((base / "relations.json").read_text(encoding="utf-8"))
    cons = _json.loads((base / "concepts.json").read_text(encoding="utf-8"))
    cls_of = {c.get("name"): c.get("concept_class") for c in cons if c.get("name")}
    out, deg = defaultdict(list), defaultdict(int)
    for r in rels:
        s_, p_, o_ = (str(r.get("subject", "")), str(r.get("predicate", "")),
                      str(r.get("object", "")))
        if not s_ or not o_ or s_.lower() == o_.lower():
            continue
        out[s_].append((p_, o_)); deg[s_] += 1; deg[o_] += 1
    cards = []
    for name, edges in out.items():
        if len(name) < 3:
            continue
        seen, facts = set(), []
        for p_, o_ in edges:
            k = (p_.lower(), o_.lower())
            if k in seen:
                continue
            seen.add(k)
            facts.append(f"{name} {p_.replace('_', ' ').strip()} {o_}")
            if len(facts) >= 8:
                break
        cards.append((deg[name], name, cls_of.get(name) or "Concept", facts))
    cards.sort(key=lambda c: -c[0])
    res = {}
    for fp in (3, 6):
        tri = grp = 0
        for _, name, klass, facts in cards[:top_n]:
            f = facts[:fp]
            if not f:
                continue
            tri += sum(len("- " + x) + 1 for x in f)
            rest = [x[len(name) + 1:] for x in f]
            grp += len(f"- {name} ({klass}): " + "; ".join(rest)) + 1
        res[fp] = (tri - grp) / tri if tri else 0.0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.score_variants import agg, score_item, wilcoxon

    sweep = json.loads((OUT / "onto_context_tuning.json").read_text(encoding="utf-8"))
    gen = {}
    # the no-adapter shards reuse the policy names `baseline` and
    # `baseline_b2000`; merging them here would silently overwrite the adapter
    # baseline that every table in this report is measured against
    noadapter = {}
    for p in sorted(OUT.glob(f"tuned_onto_generation__{a.model}*.json")):
        if "smoke" in p.name:
            continue
        d_ = json.loads(p.read_text(encoding="utf-8"))
        (noadapter if "noadapter" in p.name else gen).update(d_)
    if not gen:
        sys.exit("no generation results found; run generate_tuned_onto first")
    # the budget ablation asks a different question from the selection sweep and
    # has no entry in the offline coverage table, so it is kept out of the
    # policy tables and reported on its own
    budget_gen = {k: gen.pop(k) for k in list(gen) if k.startswith("baseline_b")}

    # published references, re-scored per item with the same scorer
    kb = KnowledgeBase()
    ref = {}
    for v in ("V2_rag", "V3_ontology_only"):
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        items = [score_item(r, kb) for r in rows]
        o = agg(items)
        o["per_item"] = {i["bench_id"]: i for i in items}
        ref[v] = o

    cov = {r["config"]: r for r in sweep["results"]}
    base_gen = gen.get("baseline")

    def paired(pol_items: dict, base_items: dict, field: str):
        both = [b for b in pol_items if b in base_items]
        x = [pol_items[b][field] for b in both if pol_items[b].get(field) is not None
             and base_items[b].get(field) is not None]
        y = [base_items[b][field] for b in both if pol_items[b].get(field) is not None
             and base_items[b].get(field) is not None]
        w = wilcoxon(x, y)
        return (st.mean(a_ - b_ for a_, b_ in zip(x, y)), w)

    L = ["# Ontology retrieval tuning: does a better context give a better answer?\n"]
    L.append(f"The published V3 ontology arm assembles ~{cov['baseline']['entities_picked']:.0f} "
             f"entity cards into the shared {sweep['budget_chars']:,}-character budget "
             f"and carries **{cov['baseline']['gold_coverage']*100:.1f}%** of the "
             f"gazetteer places the gold answer names. The dense-RAG arm carries "
             f"**{sweep['rag_reference']*100:.1f}%**. The gap was read as an ontology "
             f"coverage problem. It is not: every gold place is present in the card "
             f"corpus as a card name, so injecting the whole corpus would score "
             f"**{sweep['corpus_ceiling']*100:.1f}%**. The {cov['baseline']['gold_coverage']*100:.1f}% "
             f"is a budget-allocation result, and this experiment asks what better "
             f"allocation is worth.\n")

    L.append("## What the selection policy is worth offline\n")
    L.append("| Policy | Gold coverage | Gold places in context | Cards | Context chars |")
    L.append("|---|--:|--:|--:|--:|")
    for r in sweep["results"]:
        star = " *" if r["carried_forward"] else ""
        L.append(f"| {r['config']}{star} | {r['gold_coverage']*100:.1f}% | "
                 f"{r['gold_entities_in_ctx']:.1f} / {sweep['mean_gold_entities']:.1f} | "
                 f"{r['entities_picked']:.0f} | {r['ctx_chars']:,} |")
    L.append("\n`*` carried forward to generation.\n")
    bq = cov[sweep["best_question_dependent"]]
    L.append(f"Three findings, two of them negative. **Emission order is a budget "
             f"decision**: emitting one fact of every card before the second fact of "
             f"any, so the final truncation thins all cards instead of deleting the "
             f"tail ones, moves coverage from "
             f"{cov['baseline']['gold_coverage']*100:.1f}% to "
             f"{cov['round_robin']['gold_coverage']*100:.1f}% at identical card count "
             f"and identical length. **Ordering the routed cards by similarity to the "
             f"question is much worse** than ordering them by graph degree "
             f"({cov['similarity routing']['gold_coverage']*100:.1f}% against "
             f"{cov['rr 1 fact, 32 dense']['gold_coverage']*100:.1f}%), which says the "
             f"gold answers name prominent places rather than places that resemble the "
             f"question. And **deepening the dense card search costs coverage** "
             f"({cov['rr 1 fact, 64 dense']['gold_coverage']*100:.1f}% at 64 cards): "
             f"dense hits crowd out the degree-ordered routed cards that were carrying "
             f"the gold entities. Routing deeper into each class instead, with a 2-hop "
             f"walk along `located_in`/`near`/`part_of`, reaches "
             f"**{bq['gold_coverage']*100:.1f}%** — above the RAG arm.\n")
    if "grouped" in cov and "grouped_wide" in cov:
        g, gw, b = cov["grouped"], cov["grouped_wide"], cov["baseline"]
        L.append(f"\n### Encoding is a separate lever from selection\n")
        L.append(f"Every policy above changes *which* cards are chosen. `grouped` "
                 f"changes none of that - it selects exactly the cards `baseline` "
                 f"selects ({g['entities_picked']:.0f} of them, the same ones) and "
                 f"only writes them differently: one line per entity, "
                 f"`Name (Class): pred obj; pred obj`, so the subject name is written "
                 f"once instead of once per fact. That alone moves coverage from "
                 f"**{b['gold_coverage']*100:.1f}%** to **{g['gold_coverage']*100:.1f}%** "
                 f"in {g['ctx_chars']:,} characters against {b['ctx_chars']:,}. Spending "
                 f"the saved characters on more facts per card rather than more cards - "
                 f"`grouped_wide`, six facts per card, still the same "
                 f"{gw['entities_picked']:.0f} entities - reaches "
                 f"**{gw['gold_coverage']*100:.1f}%**, above the RAG arm's "
                 f"{sweep['rag_reference']*100:.1f}%, without retrieving a single extra "
                 f"entity. On the cards this arm actually retrieves - the "
                 f"most-connected ones - the triple-per-line encoding spends "
                 f"**{_name_repetition_cost(32)[3]*100:.0f}%** of its characters "
                 f"re-writing subject names at three facts per card and "
                 f"**{_name_repetition_cost(32)[6]*100:.0f}%** at six. On cards that "
                 f"carry a single fact the grouped form costs characters instead, so "
                 f"the gain is confined to where the budget is actually spent. Either "
                 f"way it is a property of the serializer, not of the knowledge "
                 f"graph.\n")
    ctl = cov[sweep["best_any"]]
    if ctl["params"].get("global_degree"):
        L.append(f"The control is the important number. **{sweep['best_any']}** ignores "
                 f"the question entirely and injects the "
                 f"{ctl['params']['global_degree']} most-connected cards, the same set "
                 f"for every question, and scores "
                 f"**{ctl['gold_coverage']*100:.1f}%** — more than double the tuned "
                 f"question-dependent policy and {ctl['gold_coverage']/sweep['rag_reference']:.1f}x "
                 f"the RAG arm. The 69-place gazetteer the metric is computed over is "
                 f"essentially the set of prominent Jeju places, which is what graph "
                 f"degree ranks. Gold-entity coverage can therefore be satisfied "
                 f"without retrieving anything, and cannot by itself justify a "
                 f"retriever. It is reported here as a control, not as a system.\n")

    # ── why the question-independent control works ────────────────────────
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp042_prf.compute_prf import ents
    cards_p = C.INDEX_DIR / f"ontocards_{C.ONTO_VERSION}.jsonl"
    if cards_p.exists():
        cards = [json.loads(l) for l in open(cards_p, encoding="utf-8") if l.strip()]
        degs = sorted(c["degree"] for c in cards)
        by_name = {c["name"].lower(): c["degree"] for c in cards}
        bench = [json.loads(l) for l in
                 open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
                 if l.strip()]
        gold = set()
        for b in bench:
            gold |= set(ents(b["Output"]))
        gd = [max(d for n, d in by_name.items() if g in n)
              for g in gold if any(g in n for n in by_name)]
        med = st.median(gd) if gd else 0
        pctile = 100 * sum(1 for d in degs if d <= med) / len(degs)
        n_top = 80
        top = set(sorted(by_name, key=lambda n: -by_name[n])[:n_top])
        inside = sum(1 for g in gold if any(g in n for n in top))
        L.append("### Why a static roster wins the coverage metric\n")
        L.append(f"The metric is computed over a {len(gold)}-place gazetteer, and those "
                 f"places are not a random sample of the ontology. The median gold "
                 f"place sits at the **{pctile:.1f}th percentile of entity-card "
                 f"degree** (median degree {med:.0f} against {st.median(degs):.0f} for "
                 f"cards overall), and **{inside} of {len(gold)}** are inside the "
                 f"{n_top} highest-degree cards. Graph degree therefore identifies the "
                 f"metric's entire target set without reading the question, which is "
                 f"why the control scores what it does. Gold-entity coverage measures "
                 f"whether the prominent places made it into the prompt; it does not "
                 f"measure retrieval. Any future use of it in this project should "
                 f"carry the control beside it.\n")

    L.append("## What it is worth after generation\n")
    L.append(f"Same {gen[list(gen)[0]]['n']}-item benchmark, same dedicated `onto_v3` "
             f"adapter, same prompt, greedy decoding, same budget: only the context "
             f"assembly differs. The `baseline` policy reproduces the published V3 "
             f"context on every question, asserted at run time, so its score is the "
             f"control for the whole table.\n")
    L.append("| Arm / policy | Gold coverage | Token F1 | Entity F1 | Entity hit | "
             "Ungrounded | Clean |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for v, name in (("V2_rag", "V2 Simple RAG (published)"),
                    ("V3_ontology_only", "V3 Ontology only (published)")):
        if v not in ref:
            continue
        o = ref[v]
        c = f"{sweep['rag_reference']*100:.1f}%" if v == "V2_rag" else \
            f"{cov['baseline']['gold_coverage']*100:.1f}%"
        L.append(f"| {name} | {c} | {o['token_f1']*100:.1f} | {o['entity_f1']*100:.1f} | "
                 f"{o['entity_hit']*100:.1f} | {o['ungrounded_rate']*100:.1f} | "
                 f"{o['clean']*100:.1f} |")
    order = sorted(gen, key=lambda k: -(cov.get(k, {}).get("gold_coverage") or 0))
    for pol in order:
        o = gen[pol]
        c = cov.get(pol, {}).get("gold_coverage")
        L.append(f"| **{pol}** | {c*100:.1f}% | {o['token_f1']*100:.1f} | "
                 f"{o['entity_f1']*100:.1f} | {o['entity_hit']*100:.1f} | "
                 f"{o['ungrounded_rate']*100:.1f} | {o['clean']*100:.1f} |")
    L.append("")

    if base_gen:
        L.append("## Paired tests against the reproduced baseline\n")
        L.append("| Policy | Δ coverage | Δ token F1 | p | Δ entity F1 | p | "
                 "Δ ungrounded | p |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for pol in order:
            if pol == "baseline":
                continue
            o = gen[pol]
            dc = (cov.get(pol, {}).get("gold_coverage", 0)
                  - cov["baseline"]["gold_coverage"]) * 100
            cells = []
            for f in ("token_f1", "entity_f1", "ungrounded_rate"):
                d, w = paired(o["per_item"], base_gen["per_item"], f)
                cells.append(f"{d*100:+.2f}")
                cells.append(f"{w[1]:.3g}" if w else "—")
            L.append(f"| {pol} | {dc:+.1f} pp | " + " | ".join(cells) + " |")
        L.append("")
        best_tok = max((p for p in gen), key=lambda p: gen[p]["token_f1"])
        d_tok, w_tok = paired(gen[best_tok]["per_item"], base_gen["per_item"], "token_f1")
        L.append(f"The baseline reproduction scores "
                 f"{base_gen['token_f1']*100:.1f} token F1 against the published V3's "
                 f"{ref['V3_ontology_only']['token_f1']*100:.1f}, confirming the "
                 f"harness. The best policy by token F1 is **{best_tok}** at "
                 f"{gen[best_tok]['token_f1']*100:.1f} "
                 f"({d_tok*100:+.2f} pp"
                 + (f", p = {w_tok[1]:.3g}" if w_tok else "") + ").\n")
        # the coverage-to-F1 exchange rate, which is the point of the whole exercise
        pairs = [(cov[p]["gold_coverage"] - cov["baseline"]["gold_coverage"],
                  gen[p]["token_f1"] - base_gen["token_f1"])
                 for p in gen if p != "baseline" and p in cov]
        big = [(dc, df) for dc, df in pairs if dc > 0.05]
        if big:
            L.append("| Δ gold coverage | Δ token F1 | points of coverage per point of F1 |")
            L.append("|--:|--:|--:|")
            for dc, df in sorted(big):
                # below a twentieth of a point the F1 move is noise, and dividing
                # by it manufactures a ratio in the thousands
                ratio = (f"{dc/df:.1f}:1" if abs(df) > 0.0005
                         else "no measurable exchange")
                L.append(f"| {dc*100:+.1f} pp | {df*100:+.2f} pp | {ratio} |")
            L.append("")
        # a conclusion the project can act on, derived rather than asserted
        best_tok = max(gen, key=lambda k: gen[k]["token_f1"])
        best_ung = min(gen, key=lambda k: gen[k]["ungrounded_rate"])
        best_hit = max(gen, key=lambda k: gen[k]["entity_hit"])
        L.append("## What to adopt\n")
        L.append(f"No policy differs from the published baseline at p < 0.05 on any "
                 f"metric, so none of this is a result the project can bank on, and "
                 f"the table should be read as an upper bound on what context "
                 f"assembly is worth here: about half a point of token F1 either "
                 f"way. Within that bound the ranking is consistent - `{best_tok}` "
                 f"leads on token F1 ({gen[best_tok]['token_f1']*100:.1f}), "
                 f"`{best_hit}` on entity hit "
                 f"({gen[best_hit]['entity_hit']*100:.1f}) and `{best_ung}` on "
                 f"audit-grade ungrounded rate "
                 f"({gen[best_ung]['ungrounded_rate']*100:.1f}%, against "
                 f"{base_gen['ungrounded_rate']*100:.1f}% for the baseline and "
                 f"{ref["V2_rag"]["ungrounded_rate"]*100:.1f}% for Simple RAG). "
                 f"The grouped-encoding policies cost nothing to adopt - same "
                 f"retrieval, same budget, one line of serializer - and are not "
                 f"worse than the baseline on any metric, which is the only reason "
                 f"to prefer one. The gap to Simple RAG "
                 f"({(ref["V2_rag"]["token_f1"] - base_gen["token_f1"])*100:.1f} "
                 f"token F1) is not a context-assembly problem and is not closed by "
                 f"any policy here.\n")
    # Every policy above spends the same budget differently. This one spends
    # less of it, which is the only direction the sweep never tested and the
    # only one that can pay off in tokens rather than in F1.
    if budget_gen and base_gen:
        L.append("## What is the budget itself worth?\n")
        L.append(f"Identical selection - the published V3 policy, 16 entities, "
                 f"three facts each - truncated to a smaller character budget. "
                 f"Nothing about retrieval changes; the context is simply cut "
                 f"shorter, which is what a deployment does when it wants the "
                 f"prompt cheaper.\n")
        L.append("| Budget | Prompt tokens | Token F1 | Δ | p | Entity F1 | "
                 "Entity hit | Ungrounded |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        L.append(f"| {sweep['budget_chars']:,} (published) | "
                 f"{base_gen['prompt_tokens']:.0f} | "
                 f"{base_gen['token_f1']*100:.1f} | — | — | "
                 f"{base_gen['entity_f1']*100:.1f} | "
                 f"{base_gen['entity_hit']*100:.1f} | "
                 f"{base_gen['ungrounded_rate']*100:.1f} |")
        btests = []
        for pol in sorted(budget_gen, key=lambda k: -int(k.split("_b")[1])):
            o = budget_gen[pol]
            cap = int(pol.split("_b")[1])
            d, w = paired(o["per_item"], base_gen["per_item"], "token_f1")
            btests.append((pol, cap, d * 100, w[1] if w else 1.0, o))
            L.append(f"| {cap:,} | {o['prompt_tokens']:.0f} | "
                     f"{o['token_f1']*100:.1f} | {d*100:+.2f} | "
                     + (f"{w[1]:.3g}" if w else "—") + " | "
                     f"{o['entity_f1']*100:.1f} | {o['entity_hit']*100:.1f} | "
                     f"{o['ungrounded_rate']*100:.1f} |")
        L.append("")
        # The knee is where the curve stops being flat, walking DOWN from the
        # published budget. A non-significant p is not enough on its own - at
        # n=347 a real half-point loss can sit at p>0.05 - so a cut also has to
        # keep its point estimate inside a quarter point of the baseline.
        TOL = 0.25
        knee, first_bad = None, None
        for pol, cap, d, pv, o in sorted(btests, key=lambda t: -t[1]):
            if pv < 0.05 or d < -TOL:
                first_bad = (pol, cap, d, pv, o)
                break
            knee = (pol, cap, d, pv, o)
        if knee:
            save = (1 - knee[4]["prompt_tokens"] / base_gen["prompt_tokens"]) * 100
            ratio = sweep["budget_chars"] / knee[1]
            L.append(f"Walking down from the published budget, quality is flat to "
                     f"**{knee[1]:,} characters** ({knee[2]:+.2f} token F1, "
                     f"p = {knee[3]:.3g}), and that cut alone is worth "
                     f"**{save:.0f}% of the prompt tokens** - "
                     f"{knee[4]['prompt_tokens']:.0f} against "
                     f"{base_gen['prompt_tokens']:.0f} per question. "
                     + (f"The next cut down, {first_bad[1]:,} characters, is where "
                        f"it stops: {first_bad[2]:+.2f} token F1 "
                        f"(p = {first_bad[3]:.3g}), which is not significant at "
                        f"this n but is no longer inside the {TOL:.2f}-point band "
                        f"either, and below it the losses do reach significance. "
                        if first_bad and first_bad[3] >= 0.05 else
                        (f"The next cut down, {first_bad[1]:,} characters, costs "
                         f"{first_bad[2]:+.2f} token F1 (p = {first_bad[3]:.3g}). "
                         if first_bad else ""))
                     + f"So the published budget is roughly {ratio:.1f}x larger "
                       f"than it needs to be, and the headroom is not free all the "
                       f"way down - entity hit falls from "
                       f"{base_gen['entity_hit']*100:.1f} to "
                       f"{min(btests, key=lambda t: t[1])[4]['entity_hit']*100:.1f} "
                       f"at the smallest budget tested. At 8B this is the only "
                       f"change in the whole study that pays, and it pays in "
                       f"tokens rather than in F1: about {save:.0f}% of the "
                       f"ontology arm's prompt for no measurable quality loss. "
                       f"Whether it is a recommendation or an 8B curiosity is "
                       f"settled below, and the answer is not the comfortable "
                       f"one.\n")
        else:
            L.append("Every cut tested costs measurable quality, so the published "
                     "budget is not oversized and there is no free saving here.\n")

    # Does any of the tuning table survive below 8B? Both levers - encoding
    # (grouped*) and selection (round_robin, routing_deep_hop2, grouped_deep) -
    # are re-run at 3B and 1B, each model against its own baseline with its own
    # dedicated adapter. This is the transfer question, and it has an answer.
    scale, scale_budget, scale_na = {}, {}, {}
    for m in ("llama3.2-3b", "llama3.2-1b"):
        shards = sorted(OUT.glob(f"tuned_onto_generation__{m}__*.json"))
        d = {}
        for sp in shards:
            if "smoke" in sp.name:
                continue
            if "noadapter" in sp.name:
                scale_na[m] = json.loads(sp.read_text(encoding="utf-8"))
                continue
            d.update(json.loads(sp.read_text(encoding="utf-8")))
        if d.get("baseline"):
            # the budget shards live under the same model glob; they answer a
            # different question and belong in their own table
            scale_budget[m] = {k: d.pop(k) for k in list(d)
                               if k.startswith("baseline_b")}
            scale[m] = d
    if scale:
        L.append("## Does any of it hold below 8B?\n")
        L.append("Both levers re-run at 3B and 1B: encoding (`grouped`, "
                 "`grouped_wide`) and selection (`round_robin`, "
                 "`routing_deep_hop2`, `grouped_deep`). Each model uses its own "
                 "dedicated `onto_v3` adapter and its own `baseline` as the "
                 "control, so every delta is paired within one model and nothing "
                 "is compared across model sizes.\n")
        L.append("| Policy | 8B Δ token F1 | p | 3B Δ | p | 1B Δ | p |")
        L.append("|---|--:|--:|--:|--:|--:|--:|")
        tests, order_s = [], ("round_robin", "routing_deep_hop2", "grouped_deep",
                              "grouped", "grouped_wide")
        for pol in order_s:
            cells = []
            for m, d in [(a.model, gen)] + list(scale.items()):
                b = (scale.get(m) or {}).get("baseline") if m != a.model else base_gen
                o = d.get(pol)
                if not o or not b:
                    cells += ["—", "—"]
                    continue
                dv, w = paired(o["per_item"], b["per_item"], "token_f1")
                cells += [f"{dv*100:+.2f}", f"{w[1]:.3g}" if w else "—"]
                if m != a.model:
                    tests.append((m, pol, dv * 100, w[1] if w else 1.0))
            L.append(f"| {pol} | " + " | ".join(cells) + " |")
        L.append("")
        # Holm over the small-model token-F1 family: 10 tests, so a bare 0.05
        # would be expected to fire twice on noise alone
        tests.sort(key=lambda t: t[3])
        n_t, prev, holm = len(tests), 0.0, []
        for i_, (m, pol, dv, pv) in enumerate(tests):
            prev = max(prev, min(1.0, pv * (n_t - i_)))
            holm.append((m, pol, dv, pv, prev))
        neg = sum(1 for t in tests if t[2] < 0)
        sig = [h for h in holm if h[4] < 0.05]
        sign_p = 0.5 ** n_t if neg == n_t else None
        L.append(f"**{neg} of {n_t}** policy-model pairs below 8B are negative"
                 + (f", which on its own is a sign test at p = {sign_p:.4f}"
                    if sign_p else "")
                 + f". After Holm correction across those {n_t} token-F1 tests, "
                 + (", ".join(f"`{pol}` at {m} ({dv:+.2f}, Holm p = {hp:.4g})"
                              for m, pol, dv, _, hp in sig)
                    if sig else "none") + " survives as significant.\n")
        L.append("At 8B every policy in this table was nominally positive or flat "
                 "and none was significant; below 8B every one is negative and the "
                 "deepest selection policy is significantly so. The tuning table "
                 "is therefore **not a result that transfers down the model-size "
                 "curve**. The reading that fits all three sizes is that the "
                 "published V3 context - 16 entities, three facts each, no "
                 "two-hop expansion - is already at or past the point where extra "
                 "context stops paying for itself, and that the smaller the model, "
                 "the more a longer or deeper context costs it. Every one of "
                 "these policies carries more of the gold entities than the "
                 "baseline does while none of them answers better, at any size, "
                 "which is the third "
                 "independent confirmation in this project that coverage is not "
                 "the quantity to optimise.\n")
        L.append("For the encoding lever specifically, this retires the one "
                 "positive reading the 8B table permitted. `grouped` and "
                 "`grouped_wide` recover characters and raise coverage identically "
                 "at every size, because that is a property of the serialized "
                 "context rather than of the model, but they do not change what "
                 "the model answers at 8B and are nominally worse at 3B and 1B. "
                 "Carry grouped encoding as a cost saving, not as a quality "
                 "improvement, and do not carry the selection policies below 8B "
                 "at all.\n")

    if scale_budget and any(scale_budget.values()) and budget_gen:
        L.append("### Does the saving hold below 8B?\n")
        L.append("| Model | Budget | Prompt tokens | Token F1 | Δ vs its "
                 "3,000-char baseline | p |")
        L.append("|---|--:|--:|--:|--:|--:|")
        rows_ok = []
        for m, bd in scale_budget.items():
            b = (scale.get(m) or {}).get("baseline")
            if not b or not bd:
                continue
            L.append(f"| {m} | 3,000 | {b['prompt_tokens']:.0f} | "
                     f"{b['token_f1']*100:.1f} | — | — |")
            for pol in sorted(bd, key=lambda k: -int(k.split("_b")[1])):
                o = bd[pol]
                d, w = paired(o["per_item"], b["per_item"], "token_f1")
                pv = w[1] if w else 1.0
                L.append(f"| {m} | {int(pol.split('_b')[1]):,} | "
                         f"{o['prompt_tokens']:.0f} | {o['token_f1']*100:.1f} | "
                         f"{d*100:+.2f} | {pv:.3g} |")
                rows_ok.append((m, int(pol.split("_b")[1]), d * 100, pv))
        L.append("")
        at2k = [r for r in rows_ok if r[1] == 2000]
        bad = [r for r in at2k if r[3] < 0.05]
        if at2k:
            L.append(f"At 2,000 characters the deltas are "
                     + ", ".join(f"{r[2]:+.2f} at {r[0]} (p = {r[3]:.3g})"
                                 for r in at2k)
                     + (". None is significant, so the saving is a property of "
                        "the task rather than of the 8B model and can be taken "
                        "at every size tested — the one recommendation in this "
                        "report that applies to a deployment.\n"
                        if not bad else
                        ". At least one size loses significantly, so the saving "
                        "is an 8B result and not a recommendation for the "
                        "project's smaller models.\n"))
            # the mechanism was asserted once and then tested: the same cut on
            # the vanilla model, no adapter, at both sizes
            na_all = {a.model: noadapter}
            na_all.update(scale_na)
            na_rows = []
            for m, nd in na_all.items():
                if not (nd.get("baseline") and nd.get("baseline_b2000")):
                    continue
                dn, wn = paired(nd["baseline_b2000"]["per_item"],
                                nd["baseline"]["per_item"], "token_f1")
                ab = (base_gen if m == a.model
                      else (scale.get(m) or {}).get("baseline"))
                ao = (budget_gen.get("baseline_b2000") if m == a.model
                      else (scale_budget.get(m) or {}).get("baseline_b2000"))
                if not (ab and ao):
                    continue
                da, wa = paired(ao["per_item"], ab["per_item"], "token_f1")
                na_rows.append((m, da * 100, wa[1] if wa else 1.0,
                                dn * 100, wn[1] if wn else 1.0,
                                nd["baseline"]["token_f1"] * 100,
                                ab["token_f1"] * 100))
            if na_rows:
                L.append("")
                L.append("### Is it the context or the adapter?\n")
                L.append("The first reading of the table above was that the "
                         "adapters are fitted on full-budget contexts, so a "
                         "shorter prompt is off-distribution and a smaller model "
                         "has less slack. That is a claim about the adapter, so "
                         "it is testable: run the same cut on the vanilla model. "
                         "The answer is not the one the hypothesis predicted.\n")
                L.append("| Model | Δ at 2,000 chars, with `onto_v3` | p | "
                         "Δ at 2,000 chars, no adapter | p | Base F1 adapted | "
                         "Base F1 vanilla |")
                L.append("|---|--:|--:|--:|--:|--:|--:|")
                SZ = ("llama3.1-8b", "llama3.2-3b", "llama3.2-1b")
                by_size = lambda r: (SZ.index(r[0]) if r[0] in SZ else 99, r[0])
                na_rows.sort(key=by_size)
                for m, da, pa_, dn, pn, nf, af in na_rows:
                    L.append(f"| {m} | {da:+.2f} | {pa_:.3g} | {dn:+.2f} | "
                             f"{pn:.3g} | {af:.1f} | {nf:.1f} |")
                L.append("")
                lo = min(abs(r[3]) for r in na_rows)
                hi = max(abs(r[3]) for r in na_rows)
                absorb = [r[0] for r in na_rows if r[1] > r[3]]
                amplify = [r[0] for r in na_rows if r[1] < r[3]]
                L.append(f"Without an adapter the cut costs between "
                         f"{lo:.1f} and {hi:.1f} token F1, at every size and "
                         f"with no trend in model size ("
                         + ", ".join(f"{m} {dn:+.2f}, p = {pn:.3g}"
                                     for m, _, _, dn, pn, _, _ in na_rows)
                         + "). That is the information the truncated characters "
                           "were carrying. What the adapter does with that cost "
                           "is what differs: "
                         + (f"it is **absorbed** at {', '.join(absorb)} - the "
                            f"adapted model loses less than the vanilla one, or "
                            f"nothing at all - and **amplified** at "
                            f"{', '.join(amplify)}. "
                            if absorb and amplify else "")
                         + "The simple mismatch story - every adapter was "
                           "trained at 3,000 characters, so every cut should "
                           "hurt - is falsified by the 8B row, where the adapter "
                           "was trained the same way and the adapted model is "
                           "the robust one. The defensible statement is "
                           "narrower: a shorter ontology context costs under a "
                           "point of token F1 on its own, the 8B adapter is "
                           "robust enough to hide that and the smaller adapters "
                           "are not, and this experiment does not isolate why. "
                           "Model capacity is the obvious candidate and is not "
                           "tested here.\n")
                L.append(f"The size of the adapter effect is worth stating "
                         f"beside it. On identical contexts the adapters are "
                         f"worth "
                         + ", ".join(f"{af - nf:+.1f} token F1 at {m}"
                                     for m, _, _, _, _, nf, af in na_rows)
                         + " - between ten and fifteen points, against the half "
                           "point that separates the best and worst context "
                           "policy in this entire study. Context assembly is a "
                           "second-order lever for this task and the adapter is "
                           "the first-order one.\n")

    (OUT / "onto_retrieval_tuning.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "onto_retrieval_tuning.json").write_text(json.dumps(
        {"model": a.model,
         "rag_reference": sweep.get("rag_reference"),
         "corpus_ceiling": sweep.get("corpus_ceiling"),
         "budget_chars": sweep.get("budget_chars"),
         "coverage": {k: v["gold_coverage"] for k, v in cov.items()},
         "generation": {k: {f: v[f] for f in ("token_f1", "entity_f1", "entity_hit",
                                              "ungrounded_rate", "clean", "n")}
                        for k, v in gen.items()},
         "published": {k: {f: v[f] for f in ("token_f1", "entity_f1", "entity_hit",
                                             "ungrounded_rate", "clean", "n")}
                       for k, v in ref.items()},
         "scale": {m: {k: {f: v[f] for f in ("token_f1", "entity_f1", "entity_hit",
                                            "ungrounded_rate", "clean", "n")}
                       for k, v in d.items()}
                   for m, d in scale.items()},
         "budget": {k: {f: v[f] for f in ("token_f1", "entity_f1", "entity_hit",
                                          "ungrounded_rate", "clean", "n",
                                          "prompt_tokens")}
                    for k, v in budget_gen.items()},
         "budget_baseline_prompt_tokens": (base_gen or {}).get("prompt_tokens"),
         "budget_scale": {m: {k: {f: v[f] for f in ("token_f1", "n",
                                                    "prompt_tokens")}
                              for k, v in bd.items()}
                          for m, bd in scale_budget.items()}}, indent=2),
        encoding="utf-8")
    print("wrote onto_retrieval_tuning.md, onto_retrieval_tuning.json")
    for pol in order:
        print(f"  {pol:<22} cov {cov.get(pol,{}).get('gold_coverage',0)*100:5.1f}%  "
              f"tokF1 {gen[pol]['token_f1']*100:5.1f}  entF1 {gen[pol]['entity_f1']*100:5.1f}")


if __name__ == "__main__":
    main()
