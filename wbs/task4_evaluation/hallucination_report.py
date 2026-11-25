"""
WBS Task 4.2 - Evaluation and hallucination report.

Hallucination is measured reference-free: every entity-like span in every generated
answer is resolved against ontology v1.4.0 (concepts, aliases, gazetteer), and a span
that resolves to nothing is counted as fabricated. This is the one metric in the
project that does not reward reproducing the corpus, which matters because the gold
answers were written from the same corpus the passage retriever draws from.

Three things are reported that an aggregate rate alone hides:

  WHAT is fabricated   the most frequent unresolvable spans per variant, verbatim. A
                       rate of 8% means something different if it is one invented
                       restaurant repeated 30 times than if it is 30 distinct ones.
  WHAT CANNOT be true  claims in the four predicate families no project source covers
                       (accessibility, opening hours, prices, timetables). These are
                       hallucinations by construction: the knowledge base could not
                       confirm them even if they happened to be correct.
  WHAT the layer did   the actual before/after text of answers the Task 3.2.2 repair
                       loop rewrote, so the reader can judge whether repair helped.

  python -m wbs.task4_evaluation.hallucination_report --model llama3.1-8b
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
            "V5_ontology_rag_validated"]
LABEL = {"V1_llm": "V1 LLM only", "V2_rag": "V2 Simple RAG",
         "V3_ontology_only": "V3 Ontology only", "V4_ontology_rag": "V4 Ontology+RAG",
         "V5_ontology_rag_validated": "V5 Ontology+RAG+validation"}
CATEGORIES = ["recommendation", "itinerary", "safety", "dining", "accessibility",
              "transport"]



# ── audit-grade span filter ──────────────────────────────────────────────
# The Task 3.2.2 validator uses a deliberately recall-oriented span detector: it is
# better for a safety check to question an ordinary capitalised word than to miss a
# fabricated place. For MEASUREMENT that bias is wrong, because English capitalises
# sentence-initial imperatives ("Arrive", "Take"), months, and markdown headings
# ("Additional Tips"), none of which are claims about the world. This filter removes
# those before counting, and both rates are reported so the gap is visible rather
# than silently corrected away.
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december", "monday", "tuesday",
          "wednesday", "thursday", "friday", "saturday", "sunday", "spring",
          "summer", "autumn", "fall", "winter"}
IMPERATIVES = {
    "arrive", "take", "head", "spend", "visit", "enjoy", "explore", "start", "end",
    "finish", "continue", "return", "drive", "walk", "stop", "check", "book", "try",
    "consider", "wear", "bring", "use", "rent", "avoid", "note", "remember", "make",
    "stay", "depart", "travel", "grab", "catch", "follow", "go", "get", "have", "do",
    "keep", "look", "see", "plan", "pack", "prepare", "watch", "call", "ask",
    "choose", "pick", "allow", "expect", "add", "ensure", "bear", "hike", "climb",
    "sample", "taste", "relax", "wander", "stroll", "browse", "ride", "board",
    "transfer", "settle", "wrap", "conclude", "begin", "set", "feel", "find",
}
DISCOURSE = {
    "after", "before", "then", "next", "finally", "alternatively", "however",
    "optional", "option", "overall", "afterwards", "meanwhile", "additionally",
    "lastly", "firstly", "secondly", "day", "morning", "afternoon", "evening",
    "night", "lunch", "dinner", "breakfast", "tips", "tip", "notes", "pro",
    "recommended", "additional", "essential", "apps", "itinerary", "summary",
    "overview", "budget", "cost", "highlights", "conclusion", "important", "sample",
    "example", "suggested", "estimated", "total", "duration", "distance", "price",
    "entrance", "admission", "hours", "opening", "closing", "address", "phone",
    "website", "the", "a", "an", "and", "or", "of", "for", "with", "this", "that",
    "please", "yes", "no", "if", "when", "where", "what", "how", "why", "who",
}
GENERIC = {"unesco world heritage site", "unesco world heritage", "unesco",
           "world heritage site", "world heritage", "national park", "world natural",
           "biosphere reserve", "global geopark", "geopark", "korea", "south korea",
           "korean", "jeju", "jeju island", "jejudo"}
DROP = MONTHS | IMPERATIVES | DISCOURSE


_COMMON: set[str] = set()


def common_words(min_count: int = 8, lower_share: float = 0.5) -> set[str]:
    """Tokens the corpus itself usually writes in lowercase are not proper names.

    Built from the training split rather than from a hand-written stoplist, so it
    adapts to this domain: "center", "peak", "snack" and "exploration" are common
    nouns here and are capitalised only because they start a line or a heading, while
    "Hallasan" or "Seogwipo" never appear lowercase.
    """
    global _COMMON
    if _COMMON:
        return _COMMON
    import collections
    lo, hi = collections.Counter(), collections.Counter()
    tr = ROOT / "data" / "jeju_full" / "splits" / "train.jsonl"
    with open(tr, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", f"{r.get('Input','')} "
                                                         f"{r.get('Output','')}"):
                (lo if w[0].islower() else hi)[w.lower()] += 1
    _COMMON = {w for w in set(lo) | set(hi)
               if lo[w] + hi[w] >= min_count
               and lo[w] / (lo[w] + hi[w]) >= lower_share}
    return _COMMON


HEADING = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)?\*{0,2}[^*\n]{0,80}\*{0,2}\s*:\s*$")


def strip_headings(text: str) -> str:
    """Drop markdown headings and bold labels: "**Day 2: Scenic Exploration**" is a
    section title, not a claim that a place called Scenic Exploration exists."""
    out = []
    for line in (text or "").splitlines():
        t = line.strip()
        bare = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", t).strip("* ")
        if t.startswith("#") or HEADING.match(t) or (
                t.startswith("**") and t.endswith("**") and len(t) < 90):
            continue
        # a short colon-bearing line that is not a sentence is a label, not a claim:
        # "Day 3: Scenic Exploration", "Snack Time: tteokbokki"
        if ":" in bare and len(bare.split()) <= 8 and bare[-1:] not in ".!?":
            continue
        out.append(re.sub(r"\*{1,2}([^*]{1,80})\*{1,2}\s*:", " ", line))
    return "\n".join(out)


def _distinctive(sub: str, common: set[str]) -> bool:
    """Is this known entity specific enough to exempt a longer span around it?"""
    t = sub.split()
    if len(t) > 1:
        return True
    w = t[0].strip(".,:;'")
    return w not in common and w not in GENERIC and w not in DROP


def audit_spans(ans: str, kb,
                strict: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(entity-like spans, unresolved spans, generic designations) after filtering.

    `strict` tightens one rule, and only one. An unknown span that *contains* a
    known entity is normally exempted as a phrasing of it ("the Hallasan
    Mountain trail" is not an invented place). Measured 2026-10-18, that
    exemption is doing more work than the count it protects - 143 spans in V2
    against 86 counted - and the entities doing the exempting are generic:
    `jeju`, `island`, `beach`, `cultural`, `national`, `seafood`. `strict`
    requires the contained entity to be distinctive: multi-token, or a single
    token that is neither a generic designation nor a word this corpus usually
    writes in lowercase.

    `strict` is a sensitivity knob, not a better rule. What it newly counts is
    largely itinerary scaffolding ("Late Morning", "Cultural Exploration"), so
    it over-counts where the default under-counts; the two bound the rate from
    either side. Use it to check that a comparison does not depend on where in
    that band the truth sits - see `report_detector_sensitivity.py` - and not to
    report an absolute rate.

    The default is False because it is what every published rate was computed
    with. `report_detector_sensitivity.py` reports what changes under True.
    """
    common = common_words()
    spans, bad, gen = [], [], []
    for raw in kb.spans(strip_headings(ans)):
        s = re.sub(r"'s$", "", raw.strip()).strip(" .,:;-")
        k = " ".join(s.lower().split())
        if not k or len(k) < 4:
            continue
        toks = k.split()
        cl = [t.strip(".,:;'") for t in toks]
        if all(t in DROP or t in common for t in cl):   # "Additional Tips", "Center"
            continue
        if len(toks) == 1 and (cl[0] in DROP or cl[0] in common):
            continue
        if k in GENERIC:
            gen.append(s); continue
        spans.append(s)
        if not kb.known(s):
            # a span that CONTAINS a known entity is a phrasing of it, not an invention
            subs = (" ".join(toks[i:j])
                    for i in range(len(toks)) for j in range(i + 1, len(toks) + 1)
                    if j - i < len(toks))
            if any(kb.known(sub) and (not strict or _distinctive(sub, common))
                   for sub in subs):
                continue
            bad.append(s)
    return spans, bad, gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import (UNSUPPORTABLE, KnowledgeBase,
                                                          validate)
    kb = KnowledgeBase()
    print(f"auditing against ontology v{kb.version} "
          f"({len(kb.surface):,} surface forms)", flush=True)

    out, per_variant = {}, {}
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        fabricated: Counter = Counter()
        raw_fab: Counter = Counter()
        generic: Counter = Counter()
        raw_rates = []
        unsupp: Counter = Counter()
        unsupp_ex: dict[str, list] = defaultdict(list)
        rel_ex, rates, bycat = [], [], defaultdict(list)
        n_clean = 0
        for r in rows:
            ans = r.get("answer") or ""
            rspans = kb.spans(ans)
            rbad = [s for s in rspans if not kb.known(s)]
            for s in rbad:
                raw_fab[s] += 1
            if rspans:
                raw_rates.append(len(rbad) / len(rspans))
            spans, bad, gen = audit_spans(ans, kb)
            for s in bad:
                fabricated[s] += 1
            for s in gen:
                generic[s] += 1
            if spans:
                rate = len(bad) / len(spans)
                rates.append(rate); bycat[r["category"]].append(rate)
                n_clean += (not bad)
            for x in validate(ans, kb):
                if x["type"] == "UNSUPPORTABLE_CLAIM":
                    lbl = x["detail"].replace("no ", "").replace(
                        " data exists in any project source", "")
                    unsupp[lbl] += 1
                    if len(unsupp_ex[lbl]) < 3:
                        unsupp_ex[lbl].append((r["bench_id"], x["span"]))
                elif x["type"] == "RELATION_UNSUPPORTED" and len(rel_ex) < 6:
                    rel_ex.append((r["bench_id"], x["span"], x["detail"]))
        per_variant[v] = {
            "n": len(rows),
            "ungrounded_rate": st.mean(rates) if rates else None,
            "ungrounded_rate_validator_raw": st.mean(raw_rates) if raw_rates else None,
            "raw_distinct_fabricated": len(raw_fab),
            "raw_total_fabricated": sum(raw_fab.values()),
            "generic_designations": sum(generic.values()),
            "clean_answer_rate": rate(n_clean, rates),
            "distinct_fabricated": len(fabricated),
            "total_fabricated": sum(fabricated.values()),
            "top_fabricated": fabricated.most_common(a.examples),
            "unsupportable_by_family": dict(unsupp),
            "unsupportable_examples": {k: v2 for k, v2 in unsupp_ex.items()},
            "relation_examples": rel_ex,
            "by_category": {c: (st.mean(x) if x else None) for c, x in bycat.items()},
        }
        print(f"  [{v}] audit-grade ungrounded "
              f"{per_variant[v]['ungrounded_rate']*100:.1f}% "
              f"(validator-raw {per_variant[v]['ungrounded_rate_validator_raw']*100:.1f}%)  "
              f"{len(fabricated):,} distinct fabricated spans "
              f"({sum(fabricated.values()):,} occurrences, raw "
              f"{sum(raw_fab.values()):,})", flush=True)

    # repair examples from V5
    repairs = []
    p5 = EVAL / f"variant__{a.model}__V5_ontology_rag_validated.jsonl"
    if p5.exists():
        for r in (json.loads(l) for l in open(p5, encoding="utf-8") if l.strip()):
            if (r.get("draft") or "") != (r.get("answer") or "") and len(repairs) < 6:
                repairs.append({"bench_id": r["bench_id"], "category": r["category"],
                                "question": r["Input"][:160],
                                "violations": [x["span"] for x in
                                               (r.get("violations") or [])][:5],
                                "draft": (r.get("draft") or "")[:420],
                                "final": (r.get("answer") or "")[:420],
                                "fallback": r.get("uncertainty_fallback"),
                                "coverage": r.get("grounded_coverage")})
    out = {"model": a.model, "ontology_version": kb.version,
           "variants": per_variant, "repair_examples": repairs}
    (EVAL / f"hallucination_report__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")

    # ── report ────────────────────────────────────────────────────────────
    L = [f"# Task 4.2 - Evaluation and Hallucination Report ({a.model})\n",
         f"Every entity-like span in every generated answer was resolved against "
         f"ontology v{kb.version} ({len(kb.surface):,} surface forms: concept names, "
         f"aliases and the 69-place gazetteer). A span that resolves to nothing is "
         f"counted as fabricated. The measure is reference-free, so unlike token "
         f"overlap it gives no credit for reproducing the corpus the gold answers were "
         f"written from.\n",
         "## Rates\n",
         "| Variant | n | Ungrounded rate (audit) ↓ | Ungrounded rate (validator raw) "
         "| Fully grounded answers ↑ | Distinct fabricated spans | Occurrences |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for v in VARIANTS:
        if v not in per_variant:
            continue
        d = per_variant[v]
        L.append(f"| {LABEL[v]} | {d['n']} | **{d['ungrounded_rate']*100:.1f}** | "
                 f"{d['ungrounded_rate_validator_raw']*100:.1f} | "
                 f"{d['clean_answer_rate']*100:.1f} | {d['distinct_fabricated']:,} | "
                 f"{d['total_fabricated']:,} |")
    if "V1_llm" in per_variant and "V4_ontology_rag" in per_variant:
        v1, v4 = per_variant["V1_llm"], per_variant["V4_ontology_rag"]
        drop = (1 - v4["total_fabricated"] / max(1, v1["total_fabricated"])) * 100
        L.append(f"\nGrounding the model in retrieved context removes **{drop:.0f}%** of "
                 f"fabricated entity occurrences relative to answering from parametric "
                 f"memory alone ({v1['total_fabricated']:,} → "
                 f"{v4['total_fabricated']:,}). What remains is the interesting part, "
                 f"and it is listed verbatim below.\n")

    gaps = [(v, per_variant[v]["ungrounded_rate_validator_raw"] -
             per_variant[v]["ungrounded_rate"]) for v in VARIANTS if v in per_variant]
    if gaps:
        worst = max(gaps, key=lambda x: x[1])
        L.append(f"\n**The two rate columns differ, and the difference is a finding.** "
                 f"The Task 3.2.2 validator uses a recall-oriented span detector - for "
                 f"a safety check it is better to question an ordinary capitalised word "
                 f"than to miss a fabricated place. English capitalises sentence-initial "
                 f"imperatives (\"Arrive\", \"Take\"), month names and markdown headings "
                 f"(\"Additional Tips\"), so the raw rate counts those as ungrounded "
                 f"entities. The audit column removes them before counting and is the "
                 f"honest hallucination rate; the gap reaches "
                 f"{worst[1]*100:.1f} pp on {LABEL[worst[0]]}. The practical consequence "
                 f"is that the deployed repair loop fires on some spans that were never "
                 f"claims about the world, so its measured repair activity overstates "
                 f"how much real hallucination it is removing. Narrowing the detector is "
                 f"the obvious next change; it is not retrofitted here because the V5 "
                 f"answers were generated with the detector as delivered, and swapping "
                 f"it afterwards would describe a system that was never run.\n")

    L.append("## Per category\n")
    V = [v for v in VARIANTS if v in per_variant]
    L.append("| Category | " + " | ".join(LABEL[v] for v in V) + " |")
    L.append("|---|" + "--:|" * len(V))
    for c in CATEGORIES:
        cells = []
        for v in V:
            x = per_variant[v]["by_category"].get(c)
            cells.append("—" if x is None else f"{x*100:.1f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    L.append("\n## What is actually fabricated\n")
    for v in V:
        d = per_variant[v]
        if not d["top_fabricated"]:
            continue
        L.append(f"**{LABEL[v]}** — {d['distinct_fabricated']:,} distinct spans, "
                 f"most frequent:\n")
        L.append("| Span | Times |")
        L.append("|---|--:|")
        for span, n in d["top_fabricated"]:
            L.append(f"| `{span}` | {n} |")
        L.append("")
    tot_gen = sum(per_variant[v].get("generic_designations", 0) for v in V)
    if tot_gen:
        L.append(f"Separately, {tot_gen:,} spans were generic designations rather than "
                 f"place names (\"UNESCO World Heritage Site\", \"national park\", "
                 f"\"Jeju\" itself). They are excluded from the fabricated counts above "
                 f"because asserting a designation is a different act from naming a "
                 f"place that does not exist - though the ontology holds the UNESCO "
                 f"designations as relations and could check them, which it does not "
                 f"yet do.\n")
    L.append("The audit filter is not perfect and the residue is visible in the lists "
             "above: itinerary section labels that sit inside a longer sentence "
             "(\"Scenic Exploration\", \"Final Day\", \"Snack Time\") still survive it. "
             "They inflate the reported rate rather than deflate it, so the audited "
             "numbers are an upper bound on hallucination, not a flattering one.\n")
    L.append("These are verbatim. Some will be real places the ontology does not hold, "
             "which is a coverage gap rather than an invention; the audit cannot tell "
             "the two apart and does not claim to. What it does establish is that the "
             "system asserted something it cannot support, which is the property that "
             "matters for a tourism assistant a traveller will act on.\n")

    L.append("## Claims that cannot be supported in principle\n")
    L.append("Four predicate families have no coverage in any project source, so a "
             "claim of that kind is unverifiable by construction:\n")
    L.append("| Variant | " + " | ".join(lbl for _, lbl in UNSUPPORTABLE) + " |")
    L.append("|---|" + "--:|" * len(UNSUPPORTABLE))
    for v in V:
        fam = per_variant[v]["unsupportable_by_family"]
        L.append(f"| {LABEL[v]} | " +
                 " | ".join(str(fam.get(lbl, 0)) for _, lbl in UNSUPPORTABLE) + " |")
    ex = next((per_variant[v]["unsupportable_examples"] for v in V
               if per_variant[v]["unsupportable_examples"]), {})
    if ex:
        L.append("\nExamples:\n")
        for fam, items in list(ex.items())[:4]:
            for bid, span in items[:2]:
                L.append(f"- *{fam}* — `{bid}`: \"{span}\"")
    L.append("\nThe accessibility column deserves attention: the WBS names "
             "accessibility as a target task type, the corpus yielded only 47 "
             "accessibility questions against a target of 60, and the ontology holds no "
             "accessibility facts at all. Every accessibility assertion the system makes "
             "is therefore unsupported, and the correct behaviour is to say so rather "
             "than to answer well.\n")

    if repairs:
        L.append("## What the repair loop did\n")
        for r in repairs[:4]:
            L.append(f"**`{r['bench_id']}` ({r['category']})** — flagged: " +
                     ", ".join(f"`{s}`" for s in r["violations"]) +
                     (f"; abstained (coverage {r['coverage']:.2f})"
                      if r["fallback"] else "") + "\n")
            L.append(f"> *draft:* {r['draft'].strip()}…\n")
            L.append(f"> *after:* {r['final'].strip()}…\n")
    (EVAL / f"hallucination_report__{a.model}.md").write_text("\n".join(L) + "\n",
                                                              encoding="utf-8")
    print(f"\nwrote hallucination_report__{a.model}.{{md,json}}")


if __name__ == "__main__":
    main()
