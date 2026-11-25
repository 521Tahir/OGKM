"""Collect every number and every quoted answer the slide deck and the demo
video are allowed to use, from the deliverables, into one JSON.

Nothing here is written by hand. The deck and the video read this file, so a
figure on a slide can always be traced to the run that produced it, and a
re-run of the pipeline moves the slide rather than leaving it stale.

    python -m wbs.task5_demo.build_demo_evidence
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DELIV = ROOT / "wbs" / "deliverables"
T4 = DELIV / "task4_evaluation"
T2 = DELIV / "task2_formalization"
OUT = DELIV / "presentation"
SIZES = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
PRETTY = {"llama3.2-1b": "Llama-3.2-1B", "llama3.2-3b": "Llama-3.2-3B",
          "llama3.1-8b": "Llama-3.1-8B"}
# the two stored items the demo tells as "before / after". Both are real runs of
# the 1B model: the small model is the one the business case is about.
DEMO_ITEMS = ["B0190", "B0276"]
# phrases the second case's draft asserts about how you get around the island.
# What each one is worth is computed below, not asserted here.
CLAIMED_MODES = ["Jeju Subway", "Subway", "Bus Rapid Transit", "Metro", "Train"]
DEMO_MODEL = "llama3.2-1b"
# The policy before/after demo is told at 3B. The probe set was run at both
# sizes and the other size is carried alongside; 3B is the size where the
# deployed ontology configuration is worse than plain retrieval on nothing we
# measure, so a demo told there is not leaning on the weakest baseline.
POLICY_MODEL = "llama3.2-3b"


def _overlap(a: str, b: str) -> float:
    """Token F1 between two answers, as a percentage. Used to say how far a
    control answer moved: an additive graph edit adds cards to the retrieval
    index, so a question the edit has nothing to say about can still come back
    worded differently. That is worth measuring rather than waving away."""
    from experiments.exp042_prf.compute_prf import prf, toks
    r = prf(toks(a), toks(b))
    return round((r[2] if r else 0.0) * 100, 1)


def jl(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p: Path) -> dict:
    return {r["bench_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def excerpt(answer: str, anchor: str, before: int = 230, after: int = 430) -> str:
    """The window of the answer that actually contains the claim being shown.

    A slide can hold a few hundred characters. Quoting the first few hundred
    and calling it the example would be showing the reader somewhere the
    hallucination is not, so the window is cut around the claim itself and
    marked with ellipses where text was dropped.
    """
    text = " ".join(answer.split())
    i = text.lower().find(anchor.lower())
    if i < 0:
        return (text[: before + after] + (" …" if len(text) > before + after else ""))
    lo, hi = max(0, i - before), min(len(text), i + after)
    if lo:
        lo = text.find(" ", lo) + 1
    out = text[lo:hi]
    return ("… " if lo else "") + out + (" …" if hi < len(text) else "")


def pct(x) -> float | None:
    return None if x is None else round(x * 100, 2)


def train_hours() -> dict:
    """Wall-clock of the LoRA runs, from the logs the adapters came out of.

    Both campaigns: the three hybrid adapters whose logs sit in the
    deliverables, and the twelve dedicated per-arm adapters from exp037. The
    cost-of-a-change slide compares an ontology edit against retraining, so it
    should compare against every retrain this project actually paid for.
    """
    out = {}
    logs = (sorted(T4.glob("train_*.log"))
            + sorted((ROOT / "experiments" / "exp037_dedicated_adapter")
                     .glob("train__*.log")))
    for log in logs:
        for m in re.finditer(r"\[([\w.\-]+)/([\w\d]+)\] done in ([\d.]+)h", log.read_text(
                encoding="utf-8", errors="ignore")):
            out[f"{m.group(1)}__{m.group(2)}"] = float(m.group(3))
    return dict(sorted(out.items()))


STATE_WORDS = {"Typhoon": ("typhoon",), "Storm": ("storm", "gale", "squall"),
               "Rough Seas": ("rough sea", "rough wave", "high wave", "strong current",
                              "rip tide", "strong wind")}


def claim_check(live: dict) -> dict:
    """Is the answer's place-and-weather pairing an asserted fact, or just fluent?

    The validation layer can only confirm a claim the graph asserts, so the
    question an ontology edit settles is not whether the model sounds careful -
    it already does - but whether what it says is backed. For every (place,
    state) pair the answer puts in one sentence, this asks the two ontology
    versions the same question: is that pair a triple you hold?
    """
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase, key as kkey

    vers = [live["ontology"]["before"], live["ontology"]["after"]]
    kbs = {v: KnowledgeBase(v) for v in vers}
    asserted = {v: {(s, o) for s, p, o in kbs[v].triples if p == "avoid_during"}
                for v in vers}
    # only named places, and the longest match wins: "woljeongri" and
    # "woljeongri beach" are the same claim counted twice otherwise, and
    # "swimming" is a subject of the predicate without being a place at all.
    gaz = kbs[vers[-1]].gazetteer
    places = sorted({s for v in vers for s, _ in asserted[v] if s in gaz},
                    key=len, reverse=True)

    out = {"versions": vers, "items": []}
    for it in live["items"]:
        row = {"question": it["question"], "runs": {}}
        for v in vers:
            ans = it["runs"][v]["answer"]
            pairs = []
            for sent in re.split(r"(?<=[.!?])\s+", ans):
                low = sent.lower()
                matched: list[str] = []
                for place in places:
                    if place not in low or any(place in m for m in matched):
                        continue
                    matched.append(place)
                    if True:
                        for state, cues in STATE_WORDS.items():
                            if any(c in low for c in cues):
                                pair = (place, kkey(state))
                                pairs.append({"sentence": sent.strip(),
                                              "place": place, "state": state,
                                              "asserted_before": pair in asserted[vers[0]],
                                              "asserted_after": pair in asserted[vers[1]]})
            seen, uniq = set(), []
            for p in pairs:
                k = (p["place"], p["state"])
                if k not in seen:
                    seen.add(k)
                    uniq.append(p)
            row["runs"][v] = {
                "pairs_claimed": len(uniq),
                "pairs_asserted_at_this_version": sum(
                    1 for p in uniq if p["asserted_after" if v == vers[1] else "asserted_before"]),
                "pairs": uniq}
        out["items"].append(row)
    tot = {v: {"claimed": sum(i["runs"][v]["pairs_claimed"] for i in out["items"]),
               "asserted": sum(i["runs"][v]["pairs_asserted_at_this_version"]
                               for i in out["items"])} for v in vers}
    out["totals"] = tot
    return out


def main() -> None:
    from wbs.task4_evaluation.hallucination_report import audit_spans
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase

    OUT.mkdir(parents=True, exist_ok=True)
    ev: dict = {"generated_from": "wbs/deliverables", "sizes": SIZES, "pretty": PRETTY}

    # ── 1. the five variants at every size ────────────────────────────────
    variants, gated, repair = {}, {}, {}
    for m in SIZES:
        v = jl(T4 / f"variant_scores__{m}.json")["overall"]
        variants[m] = {k: {"token_f1": pct(d["token_f1"]), "entity_f1": pct(d["entity_f1"]),
                           "entity_hit": pct(d["entity_hit"]),
                           "ungrounded": pct(d["ungrounded_rate"]), "clean": pct(d["clean"]),
                           "generation_s": round(d["generation_s"], 2),
                           "prompt_tokens": round(d["prompt_tokens"]),
                           "validation_ms": (None if d["validation_ms"] is None
                                             else round(d["validation_ms"]))}
                       for k, d in v.items()}
        sa = jl(T4 / f"selective_axis__{m}.json")
        cell = sa["cells"]["gated >=1 span, unsupportable only"]
        # R19 replaced the repair step: the deployed configuration is the same
        # gate with the rewrite swapped for clause removal. Where that run
        # exists it is the arm the deck describes, because it is the one the
        # project now recommends; the re-prompting cell stays in `repair` so the
        # deck can show what changed instead of quietly restating a better
        # number.
        rpj = T4 / f"repair_policy__{m}.json"
        rp_ = jl(rpj) if rpj.exists() else None
        SURG = "V5 surgical clause, gated"
        if rp_ and SURG in rp_.get("cells", {}):
            c2 = rp_["cells"][SURG]
            gated[m] = {
                "token_f1": round(c2["mean"]["token_f1"], 2),
                "entity_f1": round(c2["mean"]["entity_f1"], 2),
                "ungrounded": round(c2["mean"]["ungrounded_rate"], 2),
                "clean": round(c2["mean"]["clean"], 2),
                "ungrounded_strict": round(c2["ungrounded_rate_strict"], 2),
                "clean_strict": round(c2["clean_strict"], 2),
                "vs_rag": {k: {"delta": round(d["delta"], 2), "p": d["p"],
                               "n": d["n"], "holm_p": d.get("holm_p")}
                           for k, d in c2["vs_rag"].items() if d},
                "repair": "surgical clause removal (R19)",
                "layer_ms": c2.get("layer_ms_mean"),
                "draft": {"token_f1": round(
                    rp_["cells"]["V4 draft, layer off"]["mean"]["token_f1"], 2)},
                "rag": {"token_f1": round(
                    rp_["cells"]["V2 Simple RAG"]["mean"]["token_f1"], 2),
                    "ungrounded_strict": round(
                        rp_["cells"]["V2 Simple RAG"]["ungrounded_rate_strict"], 2)},
                "never_worse_than_rag": (rp_["claim_check"][SURG]
                                         ["never_worse_than_rag"]),
                "gate_token_f1_gain": round(sa["marginal_token_f1"]["gate_only"], 2),
                "interaction": round(sa["marginal_token_f1"]["interaction"], 2),
                "beats_rag_on_both": m in sa["cross_size"]["gated_cell_beats_rag_on_both_at"]}
        else:
            gated[m] = {"token_f1": pct(cell["token_f1"]), "entity_f1": pct(cell["entity_f1"]),
                        "ungrounded": pct(cell["ungrounded_rate"]), "clean": pct(cell["clean"]),
                        "vs_rag": {k: {"delta": round(d["delta"], 2), "p": d["p"], "n": d["n"]}
                                   for k, d in sa["vs_rag"].items()},
                        "repair": "re-prompt the model",
                        "draft": {"token_f1": pct(sa["cells"]["draft only (layer off)"]["token_f1"])},
                        "rag": {"token_f1": pct(variants[m]["V2_rag"]["token_f1"] / 100)},
                        "gate_token_f1_gain": round(sa["marginal_token_f1"]["gate_only"], 2),
                        "interaction": round(sa["marginal_token_f1"]["interaction"], 2),
                        "beats_rag_on_both": m in sa["cross_size"]["gated_cell_beats_rag_on_both_at"]}
        repair[m] = {
            "before": {"token_f1": pct(cell["token_f1"]),
                       "ungrounded": pct(cell["ungrounded_rate"]),
                       "clean": pct(cell["clean"]),
                       "vs_rag_token_f1": round(sa["vs_rag"]["token_f1"]["delta"], 2),
                       "cost": "a second generation per repaired answer"},
            "after": ({"token_f1": gated[m]["token_f1"],
                       "ungrounded": gated[m]["ungrounded"],
                       "clean": gated[m]["clean"],
                       "vs_rag_token_f1": gated[m]["vs_rag"]["token_f1"]["delta"],
                       "cost": f"{gated[m].get('layer_ms')} ms of CPU"}
                      if rp_ else None)}
    ev["variants"] = variants
    ev["gated"] = gated
    ev["repair"] = repair
    ev["rag_reference"] = {m: variants[m]["V2_rag"] for m in SIZES}

    # the headline of the SLM argument: a 1B model with the ontology against the
    # 8B model without it. Same benchmark, same scorer, both stored runs.
    small, big = variants[SIZES[0]], variants[SIZES[-1]]
    ev["small_beats_big"] = {
        "small_model": PRETTY[SIZES[0]], "big_model": PRETTY[SIZES[-1]],
        "small_with_ontology": small["V4_ontology_rag"],
        "big_without_ontology": big["V1_llm"],
        "token_f1_gain": round(small["V4_ontology_rag"]["token_f1"] - big["V1_llm"]["token_f1"], 2),
        "ungrounded_cut": round(big["V1_llm"]["ungrounded"] - small["V4_ontology_rag"]["ungrounded"], 2),
        "clean_gain": round(small["V4_ontology_rag"]["clean"] - big["V1_llm"]["clean"], 2),
        "holds": (small["V4_ontology_rag"]["token_f1"] > big["V1_llm"]["token_f1"]
                  and small["V4_ontology_rag"]["ungrounded"] < big["V1_llm"]["ungrounded"]),
    }
    # the same comparison inside one model: what the ontology adds to the 1B
    ev["lift_within_size"] = {
        m: {"token_f1": round(variants[m]["V4_ontology_rag"]["token_f1"]
                              - variants[m]["V1_llm"]["token_f1"], 2),
            "ungrounded": round(variants[m]["V4_ontology_rag"]["ungrounded"]
                                - variants[m]["V1_llm"]["ungrounded"], 2),
            "clean": round(variants[m]["V4_ontology_rag"]["clean"]
                           - variants[m]["V1_llm"]["clean"], 2)}
        for m in SIZES}

    # ── 2. cost of an edit against cost of retraining ─────────────────────
    hours = train_hours()
    wx = jl(T2 / "weather_state_extraction.json")
    rc_before, rc_after = jl(T2 / "rule_compliance.json"), jl(T2 / "rule_compliance__1.5.0.json")

    def enforceable(d) -> list[str]:
        return sorted(k for k, v in d["enforceability"].items() if v.get("enforceable"))

    ev["edit_vs_retrain"] = {
        "train_hours": hours,
        "train_hours_min": round(min(hours.values()), 2) if hours else None,
        "train_hours_max": round(max(hours.values()), 2) if hours else None,
        "adapters_trained": len(hours),
        "edit_gpu_hours": 0.0,
        "triples_added": wx.get("accepted_pairs"),
        "entities_touched": wx.get("entities"),
        "states": wx.get("states"),
        "by_state": wx.get("by_state"),
        "support_threshold": wx.get("min_support"),
        "candidate_pairs": wx.get("candidate_pairs"),
        "rejected_for_support": wx.get("rejected_for_support"),
        "sentences_with_weather_language": wx.get("sentences_with_weather_language"),
        "rules_enforceable_before": len(enforceable(rc_before)),
        "rules_enforceable_after": len(enforceable(rc_after)),
        "rules_total": len(rc_after["enforceability"]),
        "unblocked": sorted(set(enforceable(rc_after)) - set(enforceable(rc_before))),
        "still_blocked": sorted(set(rc_after["enforceability"]) - set(enforceable(rc_after))),
        "blocked_reason": {k: v.get("missing_from_ontology")
                           for k, v in rc_after["enforceability"].items()
                           if not v.get("enforceable")},
    }
    vr = jl(T2 / "validation_report__1.5.0.json")
    sev = vr.get("severity_counts") or {}
    ev["edit_vs_retrain"]["validation_after_edit"] = {
        "version": vr.get("version"),
        "errors": sev.get("error", 0),
        "warnings": sev.get("warning", 0),
        "shacl_conforms": (vr.get("shacl") or {}).get("conforms"),
        "owl_entailed": (vr.get("owl") or {}).get("entailed"),
    }
    ev["ontology"] = {"before": jl(T2 / "ontology_quality_metrics.json"),
                      "after": jl(T2 / "ontology_quality_metrics__1.5.0.json")}

    # ── 3. before / after on stored runs ──────────────────────────────────
    kb = KnowledgeBase("1.4.0")
    v1 = rows(T4 / f"variant__{DEMO_MODEL}__V1_llm.jsonl")
    v5 = rows(T4 / f"variant__{DEMO_MODEL}__V5_ontology_rag_validated__hybded.jsonl")
    v4 = rows(T4 / f"variant__{DEMO_MODEL}__V4_ontology_rag__hybded.jsonl")
    cases = []
    for bid in DEMO_ITEMS:
        before, after = v1[bid], v5[bid]
        _, unres_b, _ = audit_spans(before["answer"], kb)
        _, unres_a, _ = audit_spans(after["answer"], kb)
        # one row per distinct span, longest form first, and no contractions or
        # possessives: "I've" is a tokeniser artefact, not an invented place.
        checks, seen = [], set()
        for span in sorted(unres_b, key=lambda s: (-len(s), s)):
            k = span.lower()
            base = k.split("'")[0]
            if "'" in span or base in seen or any(base in t for t in seen):
                continue
            seen.add(base)
            checks.append({"span": span, "in_ontology": k in kb.surface,
                           "in_gazetteer": k in kb.gazetteer,
                           "triples": len(kb.triple_index.get(k, []))})
        # the slide's span table suppresses a shorter span contained in a longer
        # one; the film wants the bare name the audience recognises, so keep a
        # second list ordered by how often the answer leans on it
        ans_low = before["answer"].lower()
        headline = sorted(
            ({"span": sp, "hits": ans_low.count(sp.lower()),
              "triples": len(kb.triple_index.get(sp.lower(), []))}
             for sp in dict.fromkeys(unres_b) if "'" not in sp and len(sp) > 3),
            key=lambda r: (-r["hits"], -len(r["span"])))[:4]
        cases.append({
            "bench_id": bid, "category": before["category"], "question": before["Input"],
            "reference": before["reference"],
            "before": {"arm": "V1 base LLM, no ontology", "answer": before["answer"],
                       "unresolved_spans": sorted(unres_b)},
            "after": {"arm": "V5 ontology + retrieval + validation layer",
                      "answer": after["answer"], "unresolved_spans": sorted(unres_a)},
            "also_v4": v4[bid]["answer"],
            "span_checks": checks,
            "headline_spans": headline,
            "spans_removed": len(set(unres_b) - set(unres_a)),
            "anchor": "",            # filled below, once the claims are known
            "before_excerpt": "",
        })
    ev["cases"] = cases

    # the window each case is quoted through: the span the audit flagged for
    # the first, the invented transport mode for the second
    anchors = [cases[0]["headline_spans"][0]["span"],
               next((m for m in CLAIMED_MODES
                     if m.lower() in cases[1]["before"]["answer"].lower()),
                    cases[1]["headline_spans"][0]["span"]
                    if cases[1]["headline_spans"] else "")]
    for case, anchor in zip(cases, anchors):
        case["anchor"] = anchor
        case["before_excerpt"] = excerpt(case["before"]["answer"], anchor)

    # ── what the graph says about getting around, for case 2 ─────────────
    # A transport mode the island does not have is the clearest kind of
    # invented fact: the graph holds the generic concept and no edge at all
    # between it and Jeju, while the modes that do exist are asserted five
    # different ways.
    def support(name: str) -> dict:
        k = name.lower()
        ts = kb.triple_index.get(k, [])
        return {"phrase": name, "in_ontology": k in kb.surface, "triples": len(ts),
                "linked_to_jeju": sum(1 for t in ts if "jeju" in (t[0], t[2])),
                "sample": [" ".join(t) for t in ts[:2]]}

    # and the other half of the check: of the transport modes the grounded
    # answer does name, how many does the graph actually tie to the island?
    cons = json.loads((ROOT / "artifacts" / "ontologies" / "1.4.0" / "concepts.json")
                      .read_text(encoding="utf-8"))
    typed = sorted({(c.get("name") or "") for c in cons
                    if c.get("concept_class") == "TransportMode" and c.get("name")},
                   key=len, reverse=True)
    after_low = cases[1]["after"]["answer"].lower()
    named, taken = [], []
    for name in typed:
        k = name.lower()
        if len(k) < 4 or k not in after_low or any(k in t for t in taken):
            continue
        taken.append(k)
        named.append(support(name))
    ev["mode_support"] = {
        "claimed": [support(m) for m in CLAIMED_MODES],
        "named_in_grounded_answer": named,
        "named_and_linked": sum(1 for m in named if m["linked_to_jeju"]),
    }

    # one real trace: which sentence rested on which asserted triple
    HUBS = ("jeju ", "island ", "korea ")

    def readable(triples: list[str]) -> list[str]:
        out = []
        for t in triples:
            if t.lower().startswith(HUBS) or t in out:
                continue
            out.append(t)
        return out

    trace = None
    for r in v5.values():
        t = [s for s in (r.get("explanation_trace") or []) if s.get("grounded")
             and len(readable(s.get("supporting_triples") or [])) >= 2]
        if len(t) >= 3:
            trace = {"bench_id": r["bench_id"], "question": r["Input"],
                     "answer": r["answer"],
                     "sentences": [{"sentence": s["sentence"],
                                    "triples": readable(s["supporting_triples"])[:3]}
                                   for s in t[:4]],
                     "grounded_coverage": r.get("grounded_coverage")}
            break
    ev["trace"] = trace

    # ── 4. what the project will not claim ────────────────────────────────
    rp = jl(T4 / "reference_paraphrase__llama3.1-8b.json")
    cr = jl(T4 / "coverage_routing__llama3.1-8b.json")
    oo = {m: jl(T4 / f"ontology_only_validated__{m}.json") for m in SIZES}
    ev["limits"] = {
        "token_f1_trails_at": [m for m in SIZES
                               if gated[m]["vs_rag"]["token_f1"]["delta"] < 0],
        "paraphrase": {
            "accepted": rp["n_accepted"], "total": rp["n_paraphrased"],
            "share_of_lead_explained_pct": {
                arm: round(100 * (d["original"] - d["paraphrased"]) / d["original"])
                for arm, d in rp["lead_before_after"].items()},
            "lead_before_after": rp["lead_before_after"],
            "lead_change_tests": rp["lead_change_tests"]},
        "oracle_headroom": cr.get("oracle_selector", {}).get("headroom_token_f1_pp"),
        "router_gain": (cr.get("router") or {}),
        "ontology_only_beats_rag_at": [m for m in SIZES
                                       if oo[m].get("beats_rag_on_both_groundedness")],
        "goal_status_md": "wbs/deliverables/GOAL_STATUS.md",
    }

    # ── 5. scale of the thing ─────────────────────────────────────────────
    man = jl(T4 / "benchmark_manifest.json")
    ev["scale"] = {"benchmark_items": man["total"],
                   "categories": sorted(man["categories"]),
                   "corpus_records": wx.get("records_scanned") or wx.get("records"),
                   "concepts": ev["ontology"]["after"]["concepts"],
                   "relations": ev["ontology"]["after"]["relations"],
                   "classes": ev["ontology"]["after"]["classes"],
                   "predicates": ev["ontology"]["after"]["distinct_predicates"],
                   "isolated_concepts": ev["ontology"]["after"]["isolated_concepts"],
                   "provenance_coverage": ev["ontology"]["after"]["provenance_coverage"]}

    # ── 6. the live edit, if it has been run ──────────────────────────────
    live = OUT / "demo_live_edit.json"
    ev["live_edit"] = jl(live) if live.exists() else None
    if ev["live_edit"]:
        ev["live_edit"]["claim_check"] = claim_check(ev["live_edit"])

    # ── 7. the policy edit: the before/after the business brief asks for ──
    px = T2 / "policy_constraint_extraction.json"
    ev["policy_extraction"] = jl(px) if px.exists() else None
    pol = sorted(OUT.glob("demo_policy_before_after__*.json"))
    primary = [q for q in pol if POLICY_MODEL in q.name] or pol
    ev["policy_edit"] = None
    if pol:
        d = jl(primary[0])
        s16 = d["summary"]
        tgt = [i for i in d["items"] if i["kind"] != "control"]
        # The cases the deck tells, strongest first. Two kinds qualify, and the
        # deck says which kind each one is rather than blurring them:
        #   "corrected" - the rule was absent from the answer and is now there;
        #   "backed"    - the answer already named the rule, the graph could not
        #                 confirm it, and now it can. That is a smaller claim
        #                 and it is labelled as one.
        def kind_of(i):
            b, a = i[d["before"]], i[d["after"]]
            if a["states_the_rule"] and not b["states_the_rule"]:
                return "corrected"
            if (i["changed"] and a["states_the_rule"]
                    and a["rule_asserted_in_graph"]
                    and not b["rule_asserted_in_graph"]):
                return "backed"
            return None
        flipped = []
        for i in tgt:
            k = kind_of(i)
            if k:
                i = dict(i, _kind_of_change=k)
                flipped.append(i)
        flipped.sort(key=lambda i: (i["_kind_of_change"] != "corrected", i["id"]))
        vr6 = T2 / "validation_report__1.6.0.json"
        v6 = jl(vr6) if vr6.exists() else {}
        sev6 = v6.get("severity_counts") or {}
        ev["policy_edit"] = {
            "model": d["model"], "before": d["before"], "after": d["after"],
            "decoding": d["decoding"], "summary": s16,
            "retraining": d["retraining"], "edit": d["edit"],
            "cases": [{"id": i["id"], "kind": i["kind"], "question": i["q"],
                       "rule": i["rule"], "triple": i["triple"],
                       "kind_of_change": i["_kind_of_change"],
                       "before": i[d["before"]], "after": i[d["after"]]}
                      for i in flipped],
            "all_targeted": [{"id": i["id"], "rule": i["rule"],
                              "triple": i["triple"],
                              "stated_before": i[d["before"]]["states_the_rule"],
                              "stated_after": i[d["after"]]["states_the_rule"],
                              "opposite_before": i[d["before"]]["states_the_opposite"],
                              "opposite_after": i[d["after"]]["states_the_opposite"]}
                             for i in tgt],
            "controls": [{"id": i["id"], "question": i["q"],
                          "changed": i["changed"],
                          "overlap": _overlap(i[d["before"]]["answer"],
                                              i[d["after"]]["answer"])}
                         for i in d["items"] if i["kind"] == "control"],
            "cards": d["cards"],
            "also_run_at": [{"model": o["model"],
                             "rule_stated_before": o["summary"]["rule_stated_before"],
                             "rule_stated_after": o["summary"]["rule_stated_after"],
                             "targeted": o["summary"]["targeted"],
                             "control_answers_changed":
                                 o["summary"]["control_answers_changed"],
                             "controls": o["summary"]["controls"]}
                            for o in (jl(q) for q in pol if q != primary[0])],
            "control_overlap_mean": round(
                sum(_overlap(i[d["before"]]["answer"], i[d["after"]]["answer"])
                    for i in d["items"] if i["kind"] == "control")
                / max(1, sum(1 for i in d["items"] if i["kind"] == "control")), 1),
            "validation": {"version": v6.get("version"),
                           "errors": sev6.get("error", 0),
                           "warnings": sev6.get("warning"),
                           "shacl_conforms": (v6.get("shacl") or {}).get("conforms"),
                           "owl_entailed": (v6.get("owl") or {}).get("entailed")},
        }

    # ── 8. both edits together: what the business actually pays ───────────
    er = ev["edit_vs_retrain"]
    pe = ev.get("policy_edit")
    ev["edits"] = {
        "n": 2 if pe else 1,
        "facts": er["triples_added"] + (pe["edit"]["triples_added"] if pe else 0),
        "versions": ["1.5.0"] + (["1.6.0"] if pe else []),
        "gpu_hours": 0.0,
        "index_rebuild_s": round(
            (ev["live_edit"]["ontology"]["card_index_build_s"]["1.5.0"]
             if ev["live_edit"] else 0)
            + (pe["edit"]["card_index_build_s"][pe["after"]] if pe else 0), 1),
        "retrain_adapters": er["adapters_trained"],
        "retrain_gpu_hours_total": round(sum(er["train_hours"].values()), 1),
        "retrain_gpu_hours_per_model": [er["train_hours_min"], er["train_hours_max"]],
        "validation_errors": max(
            er["validation_after_edit"]["errors"],
            (pe["validation"]["errors"] or 0) if pe else 0),
    }

    p = OUT / "demo_evidence.json"
    p.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(ROOT)}")
    print(f"  variants at {len(variants)} sizes, {len(cases)} before/after cases, "
          f"live edit: {'yes' if ev['live_edit'] else 'not run yet'}")
    print(f"  small-beats-big holds: {ev['small_beats_big']['holds']}")
    pe = ev.get("policy_edit")
    if pe:
        s16 = pe["summary"]
        print(f"  policy edit: rule stated {s16['rule_stated_before']} → "
              f"{s16['rule_stated_after']} of {s16['targeted']}; "
              f"{len(pe['cases'])} cases for the deck; "
              f"controls changed {s16['control_answers_changed']}/{s16['controls']}")


if __name__ == "__main__":
    main()
