"""
exp041 - deterministic, auditable scoring of the three demo queries.

Every component score is computed from an explicit check whose verdict is
recorded, so the dashboard can show WHY a number is what it is. Nothing is
asserted by a judge without the evidence being stored next to it.

Components (master-plan weights):
  F factual correctness   30%  named places that exist in the gazetteer AND whose
                               stated attributes agree with it
  C constraint satisfaction 25% per-query checklist, each item machine-checked
  G evidence grounding    20%  named places that actually appear in THAT arm's
                               own retrieved context (no credit for lucky
                               parametric recall)
  A completeness          15%  required answer elements present
  H hallucination avoid.  10%  1 - (unsupported claims / checkable claims)

  Composite = 100 x (0.30F + 0.25C + 0.20G + 0.15A + 0.10H)

Latency and answer length are reported separately, never folded into the score.

  python -m experiments.exp041_demo_traces.score_queries
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GAZ = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                 .read_text(encoding="utf-8"))
PLACES = {x["name"]: x for x in GAZ["locations"] if isinstance(x, dict)}
LOWER = {k.lower(): v for k, v in PLACES.items()}
UNESCO = {n for n, v in PLACES.items() if "unesco" in (v.get("note", "") + n).lower()}
INDOOR_CATS = {"Museum", "Market"}
COASTAL = {n for n, v in PLACES.items()
           if v["category"] == "Beach"
           or any(w in v.get("note", "").lower() for w in ("coast", "cape", "cliff", "shore"))}
CULTURAL = {n for n, v in PLACES.items()
            if v["category"] in ("CulturalAttraction", "Museum", "ReligiousSite")}

# Claims our sources cannot support, anywhere in the corpus.
UNSUPPORTABLE = [
    (re.compile(r"\b(wheelchair|barrier[- ]free|step[- ]free|elevator|lift access|"
                r"accessible (path|facilit|route|entrance)|senior[- ]friendly|"
                r"disabled access)\b", re.I), "accessibility claim (0/69 gazetteer entries carry accessibility data)"),
    (re.compile(r"\b(\d{1,2}[:.]\d{2}\s*(am|pm)?|\bopens? at\b|\bcloses? at\b|"
                r"opening hours|operating hours)\b", re.I), "opening-hours claim (2/69 entries mention hours)"),
    (re.compile(r"\b(₩|krw|\$\s?\d|\d+\s*won|admission fee|ticket price|costs? \d)", re.I),
     "price claim (no price field in sources)"),
    (re.compile(r"\bbus (number |no\.? )?\d+|\bevery \d+ minutes|\bdeparts? at\b|timetable",
                re.I), "transport-schedule claim (no timetable in sources)"),
]


def named_places(text: str) -> list[str]:
    """Gazetteer places explicitly named in the answer (longest match first)."""
    t = text.lower()
    found = []
    for name in sorted(PLACES, key=len, reverse=True):
        if name.lower() in t and not any(name.lower() in f.lower() for f in found):
            found.append(name)
    return found


def unsupported_claims(text: str) -> list[dict]:
    out = []
    for pat, why in UNSUPPORTABLE:
        for m in pat.finditer(text):
            out.append({"span": m.group(0).strip(), "reason": why})
    return out


def false_unesco(text: str, named: list[str]) -> list[dict]:
    """A UNESCO attribution attached to a place our data does not mark UNESCO."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+", text):
        if "unesco" not in sent.lower():
            continue
        in_sent = [n for n in named if n.lower() in sent.lower()]
        for n in in_sent:
            if n not in UNESCO:
                out.append({"span": f"UNESCO attributed to {n}",
                            "reason": f"{n} is categorised {PLACES[n]['category']}; "
                                      "gazetteer note carries no UNESCO designation"})
        if not in_sent and "unesco" in sent.lower():
            out.append({"span": sent.strip()[:90],
                        "reason": "UNESCO mentioned without naming a UNESCO site from our sources"})
    return out


def constraints(qid: str, text: str, named: list[str]) -> list[dict]:
    t = text.lower()
    has_order = bool(re.search(r"\b(morning|afternoon|evening|first|then|finally|day \d|1\.|2\.)\b", t))
    if qid == "Q1":
        return [
            {"id": "names a UNESCO site from our sources",
             "pass": any(n in UNESCO for n in named),
             "evidence": [n for n in named if n in UNESCO] or "none named"},
            {"id": "names a coastal viewpoint",
             "pass": any(n in COASTAL for n in named),
             "evidence": [n for n in named if n in COASTAL] or "none named"},
            {"id": "names a cultural destination",
             "pass": any(n in CULTURAL for n in named),
             "evidence": [n for n in named if n in CULTURAL] or "none named"},
            {"id": "gives a practical visiting order", "pass": has_order,
             "evidence": "ordering language present" if has_order else "absent"},
            {"id": "addresses transport", "pass": bool(re.search(r"\b(bus|taxi|car|rental|drive|transport)\b", t)),
             "evidence": "transport mentioned" if re.search(r"\b(bus|taxi|car|rental|drive|transport)\b", t) else "absent"},
        ]
    if qid == "Q2":
        return [
            {"id": "Seongsan Ilchulbong classified natural/volcanic",
             "pass": bool(re.search(r"seongsan ilchulbong[^.;]{0,120}?(natural|volcan|tuff|peak|crater)", t)),
             "evidence": "gazetteer: NaturalAttraction, UNESCO tuff cone"},
            {"id": "Manjanggul Cave classified natural/volcanic",
             "pass": bool(re.search(r"manjanggul[^.;]{0,120}?(natural|lava|cave|volcan|geolog)", t)),
             "evidence": "gazetteer: NaturalAttraction, UNESCO lava tube"},
            {"id": "Seongeup Folk Village classified cultural",
             "pass": bool(re.search(r"seongeup[^.;]{0,120}?(cultur|tradition|histor|folk|village)", t)),
             "evidence": "gazetteer: CulturalAttraction, preserved historic village"},
            {"id": "states what cannot be confirmed",
             "pass": bool(re.search(r"\b(cannot (be )?(confirm|verif)|not (confirmed|verified|available)|"
                                    r"unverified|no information|unable to confirm|unclear)\b", t)),
             "evidence": "query explicitly asked for this"},
        ]
    return [
        {"id": "names real places from our sources", "pass": len(named) > 0,
         "evidence": named or "no gazetteer place named"},
        {"id": "offers an indoor option for bad weather",
         "pass": any(PLACES[n]["category"] in INDOOR_CATS for n in named),
         "evidence": [n for n in named if PLACES[n]["category"] in INDOOR_CATS] or "none named"},
        {"id": "flags accessibility as unverified (the correct behaviour)",
         "pass": bool(re.search(r"\b(cannot (be )?(confirm|verif)|not (confirmed|verified|available)|"
                                r"unverified|no (accessibility )?information|need(s)? (additional )?verification|"
                                r"recommend (you )?(check|contact|confirm))\b", t)),
         "evidence": "0/69 gazetteer entries carry accessibility data"},
    ]


def completeness(qid: str, text: str) -> list[dict]:
    t = text.lower()
    reqs = {
        "Q1": [("three attractions named", len(named_places(text)) >= 3),
               ("per-location justification", bool(re.search(r"\b(because|since|as it|which|for a|to (learn|see|enjoy|discover))\b", t))),
               ("constraints discussed", bool(re.search(r"\b(transport|schedul|time|hour|allow|plan)\b", t)))],
        "Q2": [("all three entities addressed",
                all(k in t for k in ("seongsan", "manjanggul", "seongeup"))),
               ("category relation stated", bool(re.search(r"\b(natural|cultural)\b.*\b(categor|attraction|heritage|tourism)\b", t))),
               ("verification status addressed", bool(re.search(r"\b(verif|confirm)\w*", t)))],
        "Q3": [("destinations named", len(named_places(text)) >= 2),
               ("constraint coverage discussed", bool(re.search(r"\b(walk|accessib|indoor|weather|older|senior)\b", t))),
               ("verification gap addressed", bool(re.search(r"\b(verif|confirm|check|unknown|not available)\w*", t)))],
    }[qid]
    return [{"id": r, "pass": bool(p)} for r, p in reqs]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/exp041_demo_traces/traces.json")
    ap.add_argument("--out", default="experiments/exp041_demo_traces/scores.json")
    a_ = ap.parse_args()
    tp = ROOT / a_.traces
    T = json.loads(tp.read_text(encoding="utf-8"))
    W = {"F": 0.30, "C": 0.25, "G": 0.20, "A": 0.15, "H": 0.10}
    report = {"weights": W, "per_query": {}}

    for qid in ("Q1", "Q2", "Q3"):
        report["per_query"][qid] = {}
        for arm in ("rag", "onto_v3"):
            ans = T["answers"][qid][arm]["answer"]
            ctx = (T["rag"][qid]["stage_7_context"] if arm == "rag"
                   else T["onto"][qid]["stage_6_context"])
            named = named_places(ans)
            # F: named places that exist (by construction) and are not mis-attributed
            bad_attr = false_unesco(ans, named)
            checkable = max(len(named) + len(bad_attr), 1)
            F = max(0.0, (len(named) - len(bad_attr)) / checkable) if named else 0.0
            # C
            cons = constraints(qid, ans, named)
            Cv = sum(1 for c in cons if c["pass"]) / len(cons)
            # G: grounded in this arm's OWN context
            grounded = [n for n in named if n.lower() in ctx.lower()]
            G = len(grounded) / len(named) if named else 0.0
            # A
            comp = completeness(qid, ans)
            A = sum(1 for c in comp if c["pass"]) / len(comp)
            # H
            unsup = unsupported_claims(ans) + bad_attr
            total_claims = max(len(named) + len(unsup), 1)
            H = max(0.0, 1 - len(unsup) / total_claims)
            composite = 100 * (W["F"] * F + W["C"] * Cv + W["G"] * G
                               + W["A"] * A + W["H"] * H)
            report["per_query"][qid][arm] = {
                "named_places": named, "grounded_in_own_context": grounded,
                "constraints": cons, "completeness": comp,
                "unsupported_claims": unsup,
                "F_factual": round(F, 3), "C_constraints": round(Cv, 3),
                "G_grounding": round(G, 3), "A_completeness": round(A, 3),
                "H_halluc_avoidance": round(H, 3),
                "composite_0_100": round(composite, 1),
                "answer_tokens": T["answers"][qid][arm]["answer_tokens"],
                "retrieval_ms": T["answers"][qid][arm]["retrieval_ms"],
                "generation_s": T["answers"][qid][arm]["generation_s"],
            }

    op = ROOT / a_.out
    op.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    hdr = f"{'query':5s} {'arm':9s} {'F':>6s} {'C':>6s} {'G':>6s} {'A':>6s} {'H':>6s} {'COMP':>7s} {'unsup':>6s} {'tok':>5s} {'gen_s':>6s}"
    print(hdr); print("-" * len(hdr))
    for qid in ("Q1", "Q2", "Q3"):
        for arm in ("rag", "onto_v3"):
            r = report["per_query"][qid][arm]
            print(f"{qid:5s} {arm:9s} {r['F_factual']:6.3f} {r['C_constraints']:6.3f} "
                  f"{r['G_grounding']:6.3f} {r['A_completeness']:6.3f} {r['H_halluc_avoidance']:6.3f} "
                  f"{r['composite_0_100']:7.1f} {len(r['unsupported_claims']):6d} "
                  f"{r['answer_tokens']:5d} {r['generation_s']:6.1f}")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
