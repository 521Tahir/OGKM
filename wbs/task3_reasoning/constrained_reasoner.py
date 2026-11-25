"""
WBS Task 3.2.2 - Constrained response generation with ontology rules,
post-generation validation, repair loop, explanation tracing and uncertainty
fallback.

This is the `ontology-RAG + validation` system variant that Task 4.1 evaluates.
It wraps any of the project's existing arms (rag / onto_v3 / hybrid20) and adds a
governance layer on top of the raw generation:

  1. GENERATE       the wrapped arm produces a draft answer from its own context.
  2. VALIDATE       every capitalised span in the draft is checked against the
                    ontology surface forms and the 69-place gazetteer. A span that
                    matches neither is an UNGROUNDED_ENTITY. Asserted relations of
                    the shape "<A> ... in/near <B>" are checked against the actual
                    `located_in` / `near` assertions; a contradicted one is a
                    RELATION_UNSUPPORTED. Claims whose predicate has no coverage in
                    any project source at all (accessibility, opening hours, prices,
                    timetables) are UNSUPPORTABLE_CLAIM - the ontology cannot ever
                    confirm them, so asserting them is a hallucination by
                    construction.
  3. REPAIR         if violations exist, the model is re-prompted with the offending
                    spans quoted back and an instruction to remove or hedge exactly
                    those, keeping everything else. Up to --max-repairs rounds; the
                    loop stops early when a round introduces no improvement, so a
                    model that cannot fix itself does not burn turns.
  4. TRACE          each surviving sentence is linked to the ontology triples whose
                    subject and object both occur in it, producing a per-sentence
                    evidence trace rather than a single global citation.
  5. FALLBACK       if, after repair, grounded coverage is below --min-coverage, the
                    answer is replaced by an explicit abstention naming what could
                    not be verified. Abstaining is the correct behaviour when the
                    knowledge base does not hold the answer.

  python -m wbs.task3_reasoning.constrained_reasoner --selftest
  python -m wbs.task3_reasoning.constrained_reasoner --model llama3.1-8b --gpu 3 \
      --split bench --arm onto_v3 --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPAN = re.compile(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})\b")
SENT = re.compile(r"(?<=[.!?;])\s+")
STOPSPAN = {
    "the", "a", "an", "in", "on", "of", "for", "and", "or", "is", "are", "it",
    "this", "that", "we", "i", "you", "your", "my", "our", "day", "morning",
    "afternoon", "evening", "night", "visit", "enjoy", "explore", "start", "end",
    "note", "notes", "tip", "tips", "jeju", "jeju island", "korea", "south korea",
    "optional", "breakfast", "lunch", "dinner", "transportation", "transport",
}
# Predicates with zero coverage anywhere in this project's sources.
UNSUPPORTABLE = [
    (re.compile(r"\b(wheelchair|barrier[- ]free|step[- ]free|elevator|lift access|"
                r"accessible (path|route|facilit|entrance)|senior[- ]friendly|"
                r"disabled access)\b", re.I), "accessibility"),
    (re.compile(r"\b(\d{1,2}[:.]\d{2}\s*(am|pm)?|opens? at|closes? at|opening hours|"
                r"operating hours)\b", re.I), "opening hours"),
    (re.compile(r"(₩|KRW|\$\s?\d|\b\d+\s*won\b|admission fee|ticket price)", re.I), "price"),
    (re.compile(r"\b(bus (number |no\.? )?\d+|every \d+ minutes|departs? at|timetable)\b",
                re.I), "transport schedule"),
]


# Proposal 10.3 names four constraint families. Type and exclusion constraints are
# the two above; these two support the other two. A predicate is treated as
# FUNCTIONAL only if the ontology itself says so - at least MIN_SUBJ distinct
# subjects and at least FUNC_FRAC of them carrying exactly one object - so the
# cardinality check is derived from the data rather than asserted about it.
FUNC_FRAC, MIN_SUBJ = 0.95, 20
IS_A = re.compile(r"\b([A-Z][\w'\- ]{3,40}?)\s+is\s+(?:a|an)\s+([a-z][\w\- ]{2,30})")
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
SEASON_MONTHS = {"spring": {3, 4, 5}, "summer": {6, 7, 8},
                 "autumn": {9, 10, 11}, "fall": {9, 10, 11}, "winter": {12, 1, 2}}
# `avoid_during` and `affects` list the months to stay away from; every other
# temporal predicate lists the months the fact holds in.
NEGATIVE_TEMPORAL = {"avoid_during", "affects"}
# "peak season" is NOT here: a sentence warning that somewhere is crowded in
# peak season is not recommending a visit then, and counting it as one turns a
# correct warning into a violation.
RECOMMEND = re.compile(r"\b(best time|ideal time|perfect time|recommend\w*|"
                       r"great time|good time|popular time|should visit|"
                       r"worth visiting|is best in|are best in)\b", re.I)


def key(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


class KnowledgeBase:
    """Ontology surface forms + gazetteer + the asserted relations used for checking."""

    def __init__(self, version: str = "1.4.0"):
        d = ROOT / "artifacts" / "ontologies" / version
        if not d.exists():
            d = ROOT / "artifacts" / "ontologies" / "1.3.0"
        self.version = d.name
        concepts = json.loads((d / "concepts.json").read_text(encoding="utf-8"))
        relations = json.loads((d / "relations.json").read_text(encoding="utf-8"))
        gaz = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                         .read_text(encoding="utf-8"))
        self.surface: set[str] = set()
        for c in concepts:
            if isinstance(c, dict) and c.get("name"):
                self.surface.add(key(c["name"]))
                for a in (c.get("aliases") or []):
                    self.surface.add(key(a))
        self.gazetteer = {key(x["name"]): x for x in gaz["locations"] if isinstance(x, dict)}
        self.surface |= set(self.gazetteer)
        for bucket in ("cuisine", "activities", "traveler_types", "dietary", "seasons"):
            for x in gaz.get(bucket, []):
                self.surface.add(key(x if isinstance(x, str) else x.get("name", "")))
        self.surface.discard("")
        self.triples: list[tuple[str, str, str]] = []
        self.by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in relations:
            if isinstance(r, dict) and r.get("subject") and r.get("object"):
                s, p, o = key(r["subject"]), str(r["predicate"]), key(r["object"])
                self.triples.append((s, p, o))
                self.by_pair[(s, o)].add(p)
        self.triple_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for s, p, o in self.triples:
            self.triple_index[s].append((s, p, o))
            self.triple_index[o].append((s, p, o))
        # cardinality: which predicates does this ontology actually use functionally?
        objs: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for s, p, o in self.triples:
            objs[p][s].add(o)
        self.functional: dict[str, float] = {}
        for p, subs in objs.items():
            if len(subs) < MIN_SUBJ:
                continue
            frac = sum(1 for v in subs.values() if len(v) == 1) / len(subs)
            if frac >= FUNC_FRAC:
                self.functional[p] = round(frac, 4)
        self.class_of: dict[str, set[str]] = defaultdict(set)
        for s, p, o in self.triples:
            if p in ("is_a", "subclass_of", "instance_of"):
                self.class_of[s].add(o)
        # temporal: each recurring fact keeps its own months AND its polarity,
        # because `avoid_during` lists the months to stay away rather than the
        # months the fact holds - merging the two inverts half the layer.
        # v1.4.0 carries no temporal file, so fall back as the ontology does.
        self.temporal: list[dict] = []
        tmp: dict[tuple[str, str], dict] = {}
        tp = d / "temporal_facts.json"
        if not tp.exists():
            tp = ROOT / "artifacts" / "ontologies" / "1.2.0" / "temporal_facts.json"
        if tp.exists():
            for f in (json.loads(tp.read_text(encoding="utf-8")).get("facts") or []):
                t = f.get("temporal") or {}
                months = set(t.get("months") or [])
                if not months and t.get("season"):
                    months = set(SEASON_MONTHS.get(key(t["season"]), ()))
                subj = key(f.get("subject"))
                if not months or len(subj) < 5:
                    continue
                pred = key(f.get("predicate"))
                # facts sharing a subject AND predicate are alternatives, not
                # competing claims: `Jeju Island has_seasonality` is asserted once
                # per season, so each one alone would contradict the other three
                cur = tmp.setdefault((subj, pred), {
                    "subject": subj, "predicate": pred, "months": set(),
                    "negative": pred in NEGATIVE_TEMPORAL,
                    "cues": {w for w in re.split(r"[_\s]+", pred)
                             if len(w) > 3 and w not in ("during", "held")}
                            | ({key(f.get("category"))}
                               if f.get("category") not in (None, "general")
                               else set()),
                    "note": ""})
                cur["months"] |= months
                if t.get("note") and not cur["note"]:
                    cur["note"] = t["note"]
        # a subject asserted for every month constrains nothing
        self.temporal = [f for f in tmp.values() if len(f["months"]) < 12]

    # R8: the fallback below was a substring scan over every surface form, which
    # is O(|surface|) per span and made the validation pass 12.6x slower from
    # 7,909 to 100,000 triples while every other query-time stage stayed flat.
    # A query only reaches the scan when len(k) > 6, and if k occurs inside s
    # then so does k[:GRAM], so an inverted index from GRAM-grams to the surface
    # forms containing them narrows the scan to a handful of candidates. The
    # containment test still runs on those candidates, so the answer is exactly
    # the answer the scan gave - this is an index, not an approximation.
    GRAM = 7

    def _gram_index(self) -> dict[str, list[str]]:
        if getattr(self, "_grams", None) is None:
            idx: dict[str, list[str]] = {}
            g = self.GRAM
            for s in self.surface:
                for i in range(len(s) - g + 1):
                    idx.setdefault(s[i:i + g], []).append(s)
            self._grams = idx
        return self._grams

    def known(self, span: str) -> bool:
        k = key(span)
        if k in self.surface:
            return True
        if len(k) <= 6:
            return False
        g = self.GRAM
        if len(k) < g:                      # only reachable if GRAM is raised
            return any(k in s for s in self.surface)
        return any(k in s for s in self._gram_index().get(k[:g], ()))

    def spans(self, text: str) -> list[str]:
        """Entity candidates.

        A single capitalised word that merely STARTS a sentence ("Then", "Admission",
        "Visit") is ordinary English capitalisation, not a name, so it is skipped.
        The cost is that a fabricated one-word entity in sentence-initial position is
        missed; multi-word fabrications and anything mid-sentence are still caught,
        and a real one-word place is recognised by the knowledge base regardless.
        """
        text = text or ""
        starts = {0}
        for m in re.finditer(r"(?<=[.!?;:\n])\s+", text):
            starts.add(m.end())
        out, seen = [], set()
        for m in SPAN.finditer(text):
            s = m.group(1).strip()
            # the regex is greedy across adjacent capitalised words, so an
            # imperative lead-in is swallowed into the name ("Visit Yongduam Rock").
            # Strip leading scaffolding tokens until a real name head remains.
            toks = s.split()
            while len(toks) > 1 and key(toks[0]) in STOPSPAN:
                toks.pop(0)
            s = " ".join(toks)
            k = key(s)
            if len(k) < 4 or k in STOPSPAN or k in seen:
                continue
            if " " not in k and m.start() in starts and not self.known(s):
                continue
            seen.add(k); out.append(s)
        return out


def ungrounded_entities(answer: str, kb: KnowledgeBase,
                        audit_grade: bool = False) -> list[str]:
    """The spans to report as ungrounded entity claims.

    R2: `kb.spans` is recall-oriented by design, so a sentence-initial
    imperative, a month name and a markdown heading all reach it as entity
    claims. The repair loop then removes text that was never a fabrication,
    which is how the validation layer came to cost 4.9 pp of token F1 for no
    measurable grounding gain.

    Every *reported* rate in this project already passes through the audit-grade
    filter in `hallucination_report.audit_spans`, which requires a multi-token
    span or a single token the corpus does not usually write in lowercase, drops
    headings and labels, and treats a span containing a known entity as a
    phrasing of it. `audit_grade=True` applies that same filter inside the
    detector, so the loop repairs what the audit would have counted.

    The default is False because it is what every published V4/V5 number was
    produced with; the fix is evaluated under its own tag before anything moves.
    """
    if not audit_grade:
        return [s for s in kb.spans(answer) if not kb.known(s)]
    # imported lazily: hallucination_report imports this module inside main(),
    # and a module-level import here would make the direction load-order dependent
    from wbs.task4_evaluation.hallucination_report import audit_spans
    return audit_spans(answer, kb)[1]


def validate(answer: str, kb: KnowledgeBase, context: str = "",
             audit_grade: bool = False) -> list[dict]:
    """Return every constraint violation, each with the span that caused it."""
    v = []
    for span in ungrounded_entities(answer, kb, audit_grade):
        v.append({"type": "UNGROUNDED_ENTITY", "span": span,
                  "detail": "not a concept, alias or gazetteer entry in "
                            f"ontology v{kb.version}"})
    for pat, label in UNSUPPORTABLE:
        for m in pat.finditer(answer or ""):
            v.append({"type": "UNSUPPORTABLE_CLAIM", "span": m.group(0).strip(),
                      "detail": f"no {label} data exists in any project source"})
    # contradicted spatial relations
    for sent in SENT.split(answer or ""):
        rel = re.search(r"\b([A-Z][\w'\- ]{3,40}?)\s+is\s+(?:located\s+)?(in|near)\s+"
                        r"([A-Z][\w'\- ]{3,40})", sent)
        if not rel:
            continue
        a, word, b = key(rel.group(1)), rel.group(2), key(rel.group(3))
        want = "located_in" if word == "in" else "near"
        if a in kb.triple_index and b in kb.triple_index:
            preds = kb.by_pair.get((a, b), set()) | kb.by_pair.get((b, a), set())
            if preds and want not in preds:
                v.append({"type": "RELATION_UNSUPPORTED",
                          "span": rel.group(0).strip(),
                          "detail": f"ontology records {sorted(preds)} between these, "
                                    f"not {want}"})
    v += _cardinality(answer or "", kb)
    v += _temporal(answer or "", kb)
    return v


def _cardinality(answer: str, kb: KnowledgeBase) -> list[dict]:
    """Proposal 10.3, cardinality family.

    Only predicates the ontology uses functionally are checked, and `is_a` is the
    one that surfaces in prose: an answer that calls the same place two different
    things has asserted more objects than the predicate admits.

    The check is deliberately internal to the answer. Comparing the answer's noun
    against the ontology's own class ("is a volcanic crater" against
    `NaturalAttraction`) would fire on almost every sentence, because the class
    vocabulary is abstract and prose is concrete, and no prose-noun-to-class
    mapping exists in this project. A contradiction the answer makes with itself
    needs no such mapping.
    """
    if "is_a" not in kb.functional:
        return []
    seen: dict[str, tuple[str, str]] = {}
    out = []
    for m in IS_A.finditer(answer):
        subj, cls = key(m.group(1)), key(m.group(2))
        # keep the head noun: "is a volcanic crater formed by" -> "volcanic crater"
        # compare head nouns, not phrases: "a popular destination" and "a
        # beautiful destination" are the same class with different adjectives,
        # and treating them as two classes manufactures violations
        cls = (cls.split()[:3] or [""])[-1] if len(cls.split()) <= 3 \
            else cls.split()[2]
        cls = cls.rstrip("s")
        if not kb.known(subj) or len(cls) < 3:
            continue
        prev = seen.get(subj)
        if prev and prev[0] != cls and not (cls in prev[0] or prev[0] in cls):
            out.append({"type": "CARDINALITY_VIOLATION",
                        "span": m.group(0).strip(),
                        "detail": f"`is_a` is functional in this ontology "
                                  f"({kb.functional['is_a']*100:.1f}% of "
                                  f"subjects), but the answer also asserts "
                                  f"\"{prev[1]}\""})
        else:
            seen[subj] = (cls, m.group(0).strip())
    return out


def _temporal(answer: str, kb: KnowledgeBase) -> list[dict]:
    """Proposal 10.3, causal/temporal family.

    A fact fires only when the sentence names its subject, names a month or
    season, and carries a lexical cue from the fact itself - otherwise any
    mention of Hallasan in May would contradict a fact about Hallasan in winter,
    which is not a contradiction at all. Polarity is respected: a positive fact
    is violated by placing it outside its months, and an `avoid_during` fact is
    violated by RECOMMENDING its subject inside them.
    """
    if not kb.temporal:
        return []
    out, seen = [], set()
    for sent in SENT.split(answer):
        low = " " + sent.lower() + " "
        said = {MONTHS[w] for w in MONTHS if re.search(rf"\b{w}\b", low)}
        for seas, ms in SEASON_MONTHS.items():
            if re.search(rf"\b{seas}\b", low):
                said |= ms
        if not said:
            continue
        for f in kb.temporal:
            if not re.search(rf"\b{re.escape(f['subject'])}\b", low):
                continue
            # a positive fact needs a lexical cue of its own predicate; a
            # negative one cannot have one - a sentence that recommends what the
            # ontology says to avoid will never contain the word "avoid" - so
            # there the recommendation itself is the cue
            if not f["negative"] and not any(
                    re.search(rf"\b{re.escape(c)}", low) for c in f["cues"]):
                continue
            if f["negative"]:
                # violated by recommending the subject inside the avoid window
                if not (said & f["months"]) or not RECOMMEND.search(sent):
                    continue
                detail = (f"\"{f['subject']}\" is recorded as "
                          f"`{f['predicate']}` months {sorted(f['months'])}"
                          + (f" ({f['note']})" if f["note"] else "")
                          + f", and the sentence recommends it in "
                            f"{sorted(said & f['months'])}")
            else:
                if said & f["months"]:
                    continue
                detail = (f"\"{f['subject']}\" is recorded as "
                          f"`{f['predicate']}` months {sorted(f['months'])}"
                          + (f" ({f['note']})" if f["note"] else "")
                          + f", the sentence places it in {sorted(said)}")
            k = (f["subject"], f["predicate"], sent[:60])
            if k in seen:
                continue
            seen.add(k)
            out.append({"type": "TEMPORAL_CONFLICT",
                        "span": sent.strip()[:160], "detail": detail})
    return out


def trace(answer: str, kb: KnowledgeBase) -> list[dict]:
    """Per-sentence evidence: triples whose subject AND object occur in that sentence."""
    out = []
    for sent in [s for s in SENT.split(answer or "") if s.strip()]:
        low = key(sent)
        ev = []
        for span in kb.spans(sent):
            for (s, p, o) in kb.triple_index.get(key(span), [])[:40]:
                if s in low and o in low and len(ev) < 6:
                    t = f"{s} {p.replace('_', ' ')} {o}"
                    if t not in ev:
                        ev.append(t)
        out.append({"sentence": sent.strip()[:220], "supporting_triples": ev,
                    "grounded": bool(ev)})
    return out


def coverage(answer: str, kb: KnowledgeBase) -> float:
    sp = kb.spans(answer)
    if not sp:
        return 0.0
    return sum(1 for s in sp if kb.known(s)) / len(sp)


# ── surgical repair: remove the claim, keep the prose ────────────────────────
# R19. The repair loop above re-prompts the model with the offending spans and
# asks for a rewrite, so a single unsupported place name re-words the whole
# answer. Measured on the deployed gated cell, that is where the validation
# layer's reference-overlap cost comes from: 2.02 of the 2.79 token F1 it costs
# at llama3.1-8b is rewrites, not abstentions, and the model is being asked to
# reproduce 400 tokens of text that nothing was wrong with. A surgical pass
# instead deletes exactly the segments that carry a violation and leaves every
# other byte of the draft alone. It asserts strictly less than the draft did -
# nothing is added, reworded or hedged - needs no second generation, and the
# spans it removes are the ones the audit would have counted.
SEGSEP = re.compile(r"(?<=[.!?;])[ \t]+|\n+")
BULLET_ONLY = re.compile(r"^[\s>]*(?:[-*•]|\d+[.)])?\s*$")


def segments(text: str) -> list[tuple[str, str]]:
    """(segment, separator) pairs whose concatenation is `text` byte for byte.

    Sentence ends and line breaks are both boundaries: a tourism answer is
    mostly markdown bullets and day headings, and splitting on sentence
    punctuation alone would put a whole itinerary day in one segment.
    """
    text = text or ""
    out, i = [], 0
    for m in SEGSEP.finditer(text):
        out.append((text[i:m.start()], m.group(0)))
        i = m.end()
    if i < len(text):
        out.append((text[i:], ""))
    return out


def _tidy(text: str) -> str:
    """Close the holes left by removal without touching surviving text."""
    lines = [l for l in text.split("\n") if not BULLET_ONLY.match(l) or not l.strip()]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


CLAUSE = re.compile(r",|;|:|\s+(?:and|or|but|then|with|while|including|featuring|"
                    r"such as)\s+|\(|\)|\s+[-\u2013\u2014]\s+")
MIN_CLAUSE_WORDS = 4


def _clause_bounds(seg: str, i: int, j: int) -> tuple[int, int]:
    """Widen [i, j) to the clause that holds it."""
    lo, hi = 0, len(seg)
    for m in CLAUSE.finditer(seg):
        if m.end() <= i:
            lo = m.end()
        if m.start() >= j:
            hi = m.start()
            break
    return lo, hi


def _clause_excise(seg: str, spans: list[str]) -> str | None:
    """`seg` with the clauses carrying `spans` cut out, or None if nothing matched."""
    cuts = []
    for sp in spans:
        if not sp:
            continue
        for m in re.finditer(re.escape(sp), seg):
            cuts.append(list(_clause_bounds(seg, m.start(), m.end())))
    if not cuts:
        return None
    cuts.sort()
    merged = [cuts[0]]
    for a, b in cuts[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, prev = "", 0
    for a, b in merged:
        out += seg[prev:a]
        prev = b
    out += seg[prev:]
    out = re.sub(r"\s+", " ", out).strip()
    # removing a clause leaves the joinery behind: a conjunction with nothing to
    # join, a doubled comma, a comma against a full stop. None of that changes
    # what is asserted, and leaving it in makes the layer look like it mangled
    # the answer rather than trimmed it.
    for _ in range(2):
        out = re.sub(r"\s+(?:and|or|but|then|with|while|including|featuring|"
                     r"such as)\s*(?=[,.;:]|$)", "", out)
        out = re.sub(r"\s*,\s*(?=[,.;:])", "", out)
        out = re.sub(r",\s*(and|or|but)\s+", r" \1 ", out)
        out = re.sub(r"\s+,", ",", out)
        out = re.sub(r"\(\s*\)", "", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"^[,;:\-\u2013\u2014\s]+", "", out)
    out = re.sub(r"[,;:\s]+$", "", out)
    return out


def surgical_repair(answer: str, kb: KnowledgeBase, *,
                    audit_grade: bool = True,
                    granularity: str = "clause") -> tuple[str, list[dict]]:
    """Cut out the text that carries a violation; return (answer, what was cut).

    A violation's `span` is always a substring of the answer, so the segment
    holding it is found by containment. `granularity="clause"` first tries to
    remove only the clause around the span and keeps the shortened segment when
    what remains is itself clean and still a sentence; `"segment"` always drops
    the whole segment. Everything not cut is copied through byte for byte,
    separator included, so the layer only ever asserts less than the draft did.
    """
    def dirty(text: str) -> list[dict]:
        """What the audit would still count in `text`, with why."""
        return ([{"type": "UNGROUNDED_ENTITY", "span": s}
                 for s in ungrounded_entities(text, kb, audit_grade=audit_grade)]
                + [{"type": "UNSUPPORTABLE_CLAIM", "span": m.group(0).strip()}
                   for pat, _ in UNSUPPORTABLE for m in pat.finditer(text)])

    # Closing a hole can join two lines, and a span that was line-initial - and
    # so exempt from the span rule - stops being line-initial when it is no
    # longer at the start of a line. That is rare (1 answer in 852 on the exp024
    # test split) but it is the difference between "removes what the audit
    # counts" and "removes most of it", so the pass runs again on its own output
    # until nothing is left or it stops making progress.
    text, cut_all = (answer or ""), []
    for _round in range(3):
        text, cut = _surgical_once(text, kb, audit_grade, granularity, dirty)
        cut_all += cut
        if not cut:
            break
    return text, cut_all


def _surgical_once(answer, kb, audit_grade, granularity, dirty):
    viols = validate(answer or "", kb, audit_grade=audit_grade)
    if not viols:
        return (answer or ""), []
    kept, cut = [], []
    for seg, sep in segments(answer):
        if not seg.strip():
            kept.append((seg, sep))
            continue
        hits = [{"type": v["type"], "span": v["span"]} for v in viols
                if v["span"] and v["span"] in seg]
        if not hits:
            # a span the whole-answer pass attributed elsewhere can still be
            # entity-like here, so re-check the segment on its own
            hits = dirty(seg)
        if not hits:
            kept.append((seg, sep))
            continue
        spans = [h["span"] for h in hits]
        record = {"segment": seg.strip()[:240], "spans": sorted(set(spans))[:6],
                  "violations": hits[:6]}
        if granularity == "clause":
            short = _clause_excise(seg, spans)
            # keeping a trimmed segment is only worth it while it still names
            # something the knowledge base knows; "is worth the trip." asserts
            # nothing and reads like damage, so that segment goes whole
            if (short and len(short.split()) >= MIN_CLAUSE_WORDS
                    and not dirty(short)
                    and any(kb.known(x) for x in kb.spans(short))):
                if short[-1] not in ".!?;:":
                    short += "."
                kept.append((short, sep))
                record["action"] = "clause removed"
                cut.append(record)
                continue
        record["action"] = "segment removed"
        cut.append(record)
    return _tidy("".join(s + p for s, p in kept)), cut


REPAIR_TEMPLATE = (
    "Your previous answer contained statements this knowledge base cannot support.\n\n"
    "Unsupported statements:\n{issues}\n\n"
    "Rewrite the answer so that:\n"
    "- every place name you keep is one of the supported names listed below;\n"
    "- you REMOVE the unsupported statements entirely, or state plainly that the "
    "information is not available;\n"
    "- you do not introduce any new place name.\n\n"
    "Supported names: {allowed}\n\n"
    "Question: {q}\n\nRewritten answer:"
)
FALLBACK = (
    "I can only answer the part of this that the Jeju knowledge base actually "
    "covers.\n\n{supported}\n\nThe following could not be verified against the "
    "knowledge base and I will not assert it: {unverified}."
)


def constrained_answer(question: str, draft: str, kb: KnowledgeBase, *,
                       regenerate=None, max_repairs: int = 2,
                       min_coverage: float = 0.6,
                       audit_grade: bool = False,
                       min_violations: int = 0,
                       repair_mode: str = "regen",
                       repair_granularity: str = "clause") -> dict:
    """Run validate → repair → trace → fallback over a draft answer.

    `audit_grade` is R2's detector fix: see `ungrounded_entities`. It changes
    which spans the repair loop is asked to fix, so it is off by default and
    every published number is produced with it off.

    `min_violations` is the selective policy `selective_validation.md` measured.
    The layer abstains when grounded coverage falls below `min_coverage` *or*
    when an unsupportable claim is present, and the coverage trigger does not
    ask whether anything was actually found wrong - so a draft with nothing
    detectably ungrounded in it can still be rewritten or replaced, which can
    only cost reference overlap. Requiring at least `min_violations` audit-grade
    ungrounded spans before spending the pass dominated spending it on
    everything at 8B: 1.91% ungrounded against 2.20%, 90.4% clean against 88.2%,
    and +5.2 token F1, with the threshold at 1 in all five cross-validation
    folds. The default is 0, which is the published behaviour.

    `repair_mode` is R19's fix. `"regen"` is the published loop: re-prompt the
    model and take the rewrite. `"surgical"` deletes exactly the segments that
    carry a violation and copies every other byte of the draft through - no
    second generation, no rewording of text nothing was wrong with. Measured
    against the deployed gated cell, rewrites rather than abstentions are where
    the layer's reference-overlap cost sits, so the mode is the lever on it.
    The default is `"regen"`, which is what every published number used.
    """
    t0 = time.perf_counter()
    history = []
    answer = draft
    viols = validate(answer, kb, audit_grade=audit_grade)
    if min_violations:
        n_ung = len(ungrounded_entities(draft, kb, audit_grade=True))
        if n_ung < min_violations:
            return {"answer": draft, "draft": draft, "violations": viols,
                    "repair_rounds": 0, "repair_history": history,
                    "grounded_coverage": round(coverage(draft, kb), 4),
                    "explanation_trace": trace(draft, kb),
                    "uncertainty_fallback": False,
                    "audit_grade_detector": audit_grade,
                    "repair_mode": repair_mode, "segments_removed": [],
                    "selective_skip": True,
                    "n_ungrounded_draft": n_ung,
                    "ontology_version": kb.version,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    history.append({"round": 0, "violations": len(viols), "coverage": round(coverage(answer, kb), 4)})

    rounds = 0
    cut: list[dict] = []
    if repair_mode == "surgical":
        if viols:
            answer, cut = surgical_repair(answer, kb, audit_grade=audit_grade,
                                          granularity=repair_granularity)
            viols = validate(answer, kb, audit_grade=audit_grade)
            rounds = 1
            history.append({"round": 1, "violations": len(viols),
                            "coverage": round(coverage(answer, kb), 4),
                            "segments_removed": len(cut)})
    while repair_mode == "regen" and viols and regenerate is not None \
            and rounds < max_repairs:
        allowed = ", ".join(sorted({s for s in kb.spans(answer) if kb.known(s)})) or \
                  "(none of the names you used are supported)"
        issues = "\n".join(f"- “{v['span']}” — {v['detail']}" for v in viols[:12])
        prompt = REPAIR_TEMPLATE.format(issues=issues, allowed=allowed, q=question)
        new = (regenerate(prompt) or "").strip()
        rounds += 1
        if not new:
            break
        new_v = validate(new, kb, audit_grade=audit_grade)
        history.append({"round": rounds, "violations": len(new_v),
                        "coverage": round(coverage(new, kb), 4)})
        if len(new_v) >= len(viols):          # no improvement - stop burning turns
            history[-1]["note"] = "no improvement; repair loop halted"
            break
        answer, viols = new, new_v

    cov = coverage(answer, kb)
    tr = trace(answer, kb)
    fell_back = False
    if cov < min_coverage or not answer.strip() or \
            any(v["type"] == "UNSUPPORTABLE_CLAIM" for v in viols):
        supported = " ".join(s["sentence"] for s in tr if s["grounded"])[:900]
        unver = ", ".join(sorted({v["span"] for v in viols})[:8]) or "the unsupported portions"
        if not supported:
            supported = ("The knowledge base does not contain enough information about "
                         "this question to give a grounded answer.")
        answer = FALLBACK.format(supported=supported, unverified=unver)
        fell_back = True
        tr = trace(answer, kb)

    return {"answer": answer, "draft": draft, "violations": viols,
            "repair_rounds": rounds, "repair_history": history,
            "repair_mode": repair_mode, "segments_removed": cut,
            "grounded_coverage": round(cov, 4), "explanation_trace": tr,
            "uncertainty_fallback": fell_back,
            "audit_grade_detector": audit_grade,
            "selective_skip": False,
            "ontology_version": kb.version,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ── self-test on fabricated drafts with known-bad content ─────────────────
def selftest() -> None:
    kb = KnowledgeBase()
    print(f"knowledge base: ontology v{kb.version}, {len(kb.surface):,} surface forms, "
          f"{len(kb.triples):,} triples, {len(kb.gazetteer)} gazetteer places\n")
    cases = [
        ("Where should I go near the airport?",
         "Visit Yongduam Rock and Dongmun Market. Then try the Jeju Skylark Hotel "
         "and Dragon Peak Observatory for sunset."),
        ("Is the cave accessible?",
         "Manjanggul Cave has wheelchair access and opens at 09:00. Admission is 4000 won."),
        ("Tell me about Seongsan Ilchulbong.",
         "Seongsan Ilchulbong is a UNESCO tuff cone on Jeju Island. It is near Seopjikoji."),
    ]
    for q, draft in cases:
        r = constrained_answer(q, draft, kb, regenerate=None)
        print(f"Q: {q}")
        print(f"  draft      : {draft[:95]}")
        print(f"  violations : {len(r['violations'])}")
        for v in r["violations"][:4]:
            print(f"      [{v['type']}] “{v['span']}” — {v['detail'][:70]}")
        print(f"  coverage   : {r['grounded_coverage']:.2f}   fallback: {r['uncertainty_fallback']}")
        sg, cut = surgical_repair(draft, kb, audit_grade=True)
        print(f"  surgical   : {sg[:120]!r}")
        if cut:
            print(f"               cut {len(cut)}: "
                  + "; ".join(f"{c['action']} ({', '.join(c['spans'][:2])})" for c in cut[:3]))
        grounded = [t for t in r["explanation_trace"] if t["grounded"]]
        if grounded:
            print(f"  evidence   : {grounded[0]['supporting_triples'][:2]}")
        print(f"  final      : {r['answer'][:150]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--split", default="bench")
    ap.add_argument("--arm", default="onto_v3")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--repair-mode", choices=["regen", "surgical"], default="regen",
                    help="regen re-prompts the model for a rewrite (published "
                         "behaviour); surgical cuts out the clause that carries "
                         "the violation and needs no generation (R19)")
    ap.add_argument("--repair-granularity", choices=["clause", "segment"],
                    default="clause")
    ap.add_argument("--audit-grade", action="store_true",
                    help="repair what the audit would count, not what the "
                         "recall-oriented detector flags (R2)")
    ap.add_argument("--min-violations", type=int, default=0,
                    help="spend the layer only on drafts with at least this many "
                         "audit-grade ungrounded spans")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest or a.gpu is None:
        selftest()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ["EXP024_TAG"] = "ded"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    kb = KnowledgeBase()
    cfg = C.MODEL_BY_KEY[a.model]
    adapter = C.ADAPTER_DIR / f"{a.model}__{a.arm}"
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(
        str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]),
        for_training=False)
    sys_ok = supports_system_role(tok)
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()

    def gen(prompt_text: str, max_new=512) -> str:
        enc = tok(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    ctxf = C.CTX_DIR / f"{a.split}_{a.arm}.jsonl"
    rows = [json.loads(l) for l in open(ctxf, encoding="utf-8") if l.strip()][: a.limit]
    out = []
    for i, r in enumerate(rows, 1):
        draft = gen(render_prompt(tok, r["Input"], a.arm, r.get("context"), sys_ok)).strip()
        res = constrained_answer(r["Input"], draft, kb,
                                 regenerate=lambda p: gen(p, 400),
                                 max_repairs=a.max_repairs, min_coverage=a.min_coverage,
                                 audit_grade=a.audit_grade,
                                 min_violations=a.min_violations,
                                 repair_mode=a.repair_mode,
                                 repair_granularity=a.repair_granularity)
        res["record_id"] = r.get("record_id")
        res["Input"] = r["Input"]
        res["reference"] = r.get("Output")
        out.append(res)
        if i % 10 == 0 or i == len(rows):
            nv = sum(len(x["violations"]) for x in out)
            nf = sum(1 for x in out if x["uncertainty_fallback"])
            print(f"  {i}/{len(rows)}  violations={nv}  fallbacks={nf}", flush=True)

    p = Path(a.out) if a.out else (ROOT / "wbs" / "deliverables" / "task3_reasoning" /
                                   f"{a.split}__{a.model}__{a.arm}__constrained.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {p.relative_to(ROOT)}  ({len(out)} items)")


if __name__ == "__main__":
    main()
