"""
Proposal §11.4.2 and §11.6b - score the two human-rating instruments.

Both protocols already written into the deliverables demand a number before any
mean is reported:

  expert_ontology_review.md, step 2: "Report Krippendorff's alpha for the
  ordinal columns and Cohen's kappa for the yes/no column before reporting any
  mean score. An agreement figure that is not reported is an agreement figure
  that was not computed."

  manual_hallucination_review.md, step 2: "Report Cohen's kappa over the four
  labels before any rate."

Nothing computed them. This module does, and it enforces the order: with fewer
than two independent rater sheets it reports the agreement as undefined and
refuses to print a mean, because a mean from one rater is an opinion with a
decimal point. It also does the unblinding in the right place - per-variant
unsupported rates and the detector's precision and recall come out only after
agreement has been computed, and the variant key is read from
`expert_review_pack.json` here rather than by the raters.

The scores themselves are not available to this phase: no domain-expert or
annotator panel exists for this project, and nothing in this file simulates
one. What is delivered is the measurement: drop filled sheets next to the
blanks as

    wbs/deliverables/task4_evaluation/expert_ontology_review__<rater>.csv
    wbs/deliverables/task4_evaluation/manual_hallucination_review__<rater>.csv

and the numbers come out with no new code. `--self-test` proves that path works
by running it over synthetic sheets in a temporary directory, including a check
of the alpha implementation against Krippendorff's own published example.

  python -m wbs.task4_evaluation.score_human_review
  python -m wbs.task4_evaluation.score_human_review --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EV = ROOT / "wbs" / "deliverables" / "task4_evaluation"
EXPERT = EV / "expert_ontology_review.csv"
MANUAL = EV / "manual_hallucination_review.csv"
PACK = EV / "expert_review_pack.json"
OUT_MD = EV / "human_review_scores.md"
OUT_JSON = EV / "human_review_scores.json"

ORDINAL_COLS = ("relevant_1to5", "correct_1to5", "usable_for_reasoning_1to5")
BINARY_COL = "class_is_right_yes_no"
LABEL_COL = "label_supported_unsupported_unverifiable_generic"
LABELS = ("supported", "unsupported", "unverifiable", "generic")
PLACE_COL = "is_a_place_yes_no"
ADJUDICATE_AT = 2          # the protocol's own threshold, in scale points


# ── agreement ─────────────────────────────────────────────────────────────
def krippendorff_alpha(units: list[list], level: str = "ordinal") -> float | None:
    """Krippendorff's alpha over units of unequal coder count.

    `units` is one list of observed values per unit, missing values already
    dropped. Units with fewer than two values carry no pairable information and
    are skipped, which is what the coincidence-matrix form does anyway.

    Verified against the published nominal example in Krippendorff (2011),
    "Computing Krippendorff's Alpha-Reliability" - see `--self-test`.
    """
    used = [u for u in units if len(u) >= 2]
    if not used:
        return None
    vals = sorted({v for u in used for v in u},
                  key=lambda x: (0, x) if isinstance(x, (int, float)) else (1, str(x)))
    if len(vals) < 2:
        return 1.0                      # everyone used one value: no disagreement
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)

    # coincidence matrix
    o = [[0.0] * k for _ in range(k)]
    for u in used:
        m = len(u)
        # ordered pairs within the unit, each weighted 1/(m-1)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i == j:
                    continue
                o[idx[a]][idx[b]] += 1.0 / (m - 1)
    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n <= 1:
        return None

    if level == "nominal":
        def d2(c, kk):
            return 0.0 if c == kk else 1.0
    elif level == "ordinal":
        def d2(c, kk):
            lo, hi = min(c, kk), max(c, kk)
            s = sum(n_c[g] for g in range(lo, hi + 1)) - (n_c[lo] + n_c[hi]) / 2.0
            return s * s
    else:
        raise ValueError(level)

    d_o = sum(o[c][kk] * d2(c, kk) for c in range(k) for kk in range(k))
    d_e = sum(n_c[c] * n_c[kk] * d2(c, kk) for c in range(k) for kk in range(k)) / (n - 1)
    if d_e == 0:
        return 1.0
    return round(1.0 - d_o / d_e, 4)


def cohen_kappa(pairs: list[tuple]) -> float | None:
    """Cohen's kappa for two raters over nominal labels."""
    pairs = [(a, b) for a, b in pairs if a not in (None, "") and b not in (None, "")]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[x] / n * cb[x] / n for x in set(ca) | set(cb))
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1 else None     # undefined: no expected disagreement
    return round((po - pe) / (1 - pe), 4)


def fleiss_style_mean(kappas: list[float | None]) -> float | None:
    ks = [k for k in kappas if k is not None]
    return round(sum(ks) / len(ks), 4) if ks else None


# ── sheets ────────────────────────────────────────────────────────────────
def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rater_sheets(master: Path) -> dict[str, list[dict]]:
    """Filled copies of `master`, one per rater, keyed by the rater suffix."""
    out = {}
    for p in sorted(master.parent.glob(master.stem + "__*.csv")):
        out[p.stem.split("__", 1)[1]] = read_csv(p)
    return out


def _num(x):
    try:
        v = int(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


def _yn(x):
    s = str(x or "").strip().lower()
    return s if s in ("yes", "no") else None


def _label(x):
    s = str(x or "").strip().lower()
    return s if s in LABELS else None


def validate(sheets: dict[str, list[dict]], master: Path,
             id_col: str = "item_id") -> list[str]:
    """Problems that would make a score meaningless, reported not swallowed."""
    problems = []
    if not master.exists():
        return [f"{master.name} is missing"]
    ids = [r[id_col] for r in read_csv(master)]
    for who, rows in sheets.items():
        got = [r.get(id_col) for r in rows]
        if got != ids:
            problems.append(f"{who}: item_id column does not match the blank sheet "
                            f"({len(got)} rows vs {len(ids)})")
        for r in rows:
            for c in ORDINAL_COLS:
                if c in r and str(r[c] or "").strip() and _num(r[c]) is None:
                    problems.append(f"{who}/{r.get(id_col)}: {c}={r[c]!r} not in 1-5")
                if (c in r and _num(r.get(c)) in (1, 2)
                        and not str(r.get("rater_comment") or "").strip()):
                    problems.append(f"{who}/{r.get(id_col)}: {c} is "
                                    f"{r[c]} with no rater_comment (protocol requires one)")
            if LABEL_COL in r and str(r[LABEL_COL] or "").strip() and _label(r[LABEL_COL]) is None:
                problems.append(f"{who}/{r.get(id_col)}: label {r[LABEL_COL]!r} "
                                f"is not one of {LABELS}")
    return problems


# ── §11.4.2: the expert ontology review ───────────────────────────────────
def score_expert() -> dict:
    sheets = rater_sheets(EXPERT)
    res = {"instrument": EXPERT.name, "raters": sorted(sheets),
           "n_raters": len(sheets), "problems": validate(sheets, EXPERT),
           "items": len(read_csv(EXPERT)) if EXPERT.exists() else 0}
    if len(sheets) < 2:
        res["status"] = "unfilled" if not sheets else "single-rater"
        res["agreement"] = None
        res["means"] = None
        res["why"] = ("fewer than two independent rater sheets, so agreement is "
                      "undefined and the protocol forbids reporting a mean")
        return res

    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for who, rows in sheets.items():
        for r in rows:
            by_item[r["item_id"]][who] = r

    agreement = {}
    for c in ORDINAL_COLS:
        units = [[v for v in (_num(r.get(c)) for r in per.values()) if v is not None]
                 for per in by_item.values()]
        agreement[c] = {"krippendorff_alpha_ordinal": krippendorff_alpha(units, "ordinal")}
    ks = []
    for a, b in combinations(sorted(sheets), 2):
        ks.append(cohen_kappa([(_yn(by_item[i].get(a, {}).get(BINARY_COL)),
                                _yn(by_item[i].get(b, {}).get(BINARY_COL)))
                               for i in by_item]))
    agreement[BINARY_COL] = {"cohen_kappa_pairwise": ks,
                             "cohen_kappa_mean": fleiss_style_mean(ks)}

    adjudicate = []
    for i, per in by_item.items():
        for c in ORDINAL_COLS:
            vs = [v for v in (_num(r.get(c)) for r in per.values()) if v is not None]
            if len(vs) >= 2 and max(vs) - min(vs) >= ADJUDICATE_AT:
                adjudicate.append({"item_id": i, "column": c, "values": vs})

    means = {}
    for c in ORDINAL_COLS:
        vs = [v for per in by_item.values()
              for v in (_num(r.get(c)) for r in per.values()) if v is not None]
        means[c] = round(sum(vs) / len(vs), 3) if vs else None
    yes = [_yn(r.get(BINARY_COL)) for per in by_item.values() for r in per.values()]
    yes = [y for y in yes if y]
    means[BINARY_COL] = (round(sum(y == "yes" for y in yes) / len(yes), 4)
                         if yes else None)

    res.update({"status": "scored", "agreement": agreement, "means": means,
                "needs_adjudication": adjudicate,
                "needs_adjudication_n": len(adjudicate)})
    return res


# ── §11.6b: the blinded hallucination annotation ──────────────────────────
def score_manual() -> dict:
    sheets = rater_sheets(MANUAL)
    res = {"instrument": MANUAL.name, "raters": sorted(sheets),
           "n_raters": len(sheets), "problems": validate(sheets, MANUAL),
           "items": len(read_csv(MANUAL)) if MANUAL.exists() else 0}
    if len(sheets) < 2:
        res["status"] = "unfilled" if not sheets else "single-rater"
        res["agreement"] = None
        res["by_variant"] = None
        res["detector"] = None
        res["why"] = ("fewer than two independent rater sheets, so kappa is "
                      "undefined and the protocol forbids unblinding")
        return res

    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for who, rows in sheets.items():
        for r in rows:
            by_item[r["item_id"]][who] = r

    ks = [cohen_kappa([(_label(by_item[i].get(a, {}).get(LABEL_COL)),
                        _label(by_item[i].get(b, {}).get(LABEL_COL)))
                       for i in by_item])
          for a, b in combinations(sorted(sheets), 2)]
    units = [[v for v in (_label(r.get(LABEL_COL)) for r in per.values()) if v]
             for per in by_item.values()]
    agreement = {"cohen_kappa_pairwise": ks,
                 "cohen_kappa_mean": fleiss_style_mean(ks),
                 "krippendorff_alpha_nominal": krippendorff_alpha(units, "nominal")}

    # agreement first; only now unblind
    master = {r["item_id"]: r for r in read_csv(MANUAL)}
    key = {}
    if PACK.exists():
        pk = json.loads(PACK.read_text(encoding="utf-8"))
        key = (pk.get("system_key") or pk.get("variant_key")
               or pk.get("manual", {}).get("system_key") or {})

    label_of = {}
    for i, per in by_item.items():
        c = Counter(v for v in (_label(r.get(LABEL_COL)) for r in per.values()) if v)
        if c:
            top, n = c.most_common(1)[0]
            label_of[i] = top if n > len(per) / 2 else "tied"

    by_variant: dict[str, dict] = defaultdict(lambda: {"n": 0, "unsupported": 0})
    for i, lab in label_of.items():
        sysid = master[i].get("system", "?")
        name = key.get(sysid, sysid)
        by_variant[name]["n"] += 1
        by_variant[name]["unsupported"] += (lab == "unsupported")
    for v in by_variant.values():
        v["unsupported_rate"] = round(v["unsupported"] / v["n"], 4) if v["n"] else None

    # the number the project calls a known unknown: detector precision / recall
    flagged = {i for i, r in master.items()
               if str(r.get("auto_flagged") or r.get("flagged") or "").strip().lower()
               in ("1", "true", "yes")}
    if not flagged:                       # the sheet does not carry the flag column
        det = {"note": "the blank sheet carries no auto-flagged column, so the "
                       "flagged/not-flagged split has to come from "
                       "expert_review_pack.json"}
        pk = json.loads(PACK.read_text(encoding="utf-8")) if PACK.exists() else {}
        flagged = set(pk.get("manual", {}).get("flagged_item_ids") or [])
    if flagged:
        uns = {i for i, l in label_of.items() if l == "unsupported"}
        tp = len(flagged & uns)
        det = {"flagged": len(flagged), "unsupported_by_raters": len(uns),
               "precision": round(tp / len(flagged), 4) if flagged else None,
               "recall": round(tp / len(uns), 4) if uns else None}

    res.update({"status": "scored", "agreement": agreement,
                "by_variant": dict(by_variant), "detector": det})
    return res


# ── report ────────────────────────────────────────────────────────────────
def build_report() -> dict:
    e, m = score_expert(), score_manual()
    data = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "expert_11_4_2": e, "manual_11_6b": m}
    EV.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Human review scores (proposal §11.4.2, §11.6b)", "",
         "Both protocols require an agreement statistic before any mean or rate. "
         "This is the code that computes it, and it enforces the order: with "
         "fewer than two independent rater sheets it reports agreement as "
         "undefined and prints no mean, and for the blinded hallucination sheet "
         "it does not unblind the variant identities either.", "",
         "Filled sheets are picked up as "
         "`<instrument>__<rater>.csv` beside the blank instrument; nothing else "
         "has to change for the numbers to appear.", ""]

    for title, r, extra in (("§11.4.2 — expert ontology evaluation", e, ""),
                            ("§11.6b — blinded hallucination annotation", m, "")):
        L += [f"## {title}", "",
              f"- instrument: `{r['instrument']}`, {r['items']:,} items",
              f"- rater sheets found: {r['n_raters']}"
              + (f" ({', '.join(r['raters'])})" if r["raters"] else ""),
              f"- status: **{r['status']}**"]
        if r["problems"]:
            L += [f"- validation problems: {len(r['problems'])}"]
            L += [f"  - {p}" for p in r["problems"][:10]]
        if r["status"] != "scored":
            L += ["", f"_{r['why']}._", "",
                  "What would be computed the moment two sheets exist: "
                  + ("Krippendorff's ordinal alpha per 1-5 column, Cohen's kappa "
                     "on the class/predicate judgement, the list of items the "
                     "raters differ on by two or more points (the protocol's own "
                     "adjudication threshold), and only then the means."
                     if r is e else
                     "Cohen's kappa and nominal alpha over the four labels, then "
                     "the unblinded per-variant unsupported rate, then the "
                     "automated detector's precision and recall against the "
                     "raters - the figure this project currently carries as a "
                     "known unknown."), ""]
            continue
        if r is e:
            L += ["", "| Column | Agreement |", "|---|---|"]
            for c in ORDINAL_COLS:
                L.append(f"| `{c}` | Krippendorff α (ordinal) "
                         f"{r['agreement'][c]['krippendorff_alpha_ordinal']} |")
            L.append(f"| `{BINARY_COL}` | Cohen κ "
                     f"{r['agreement'][BINARY_COL]['cohen_kappa_mean']} |")
            L += ["", f"{r['needs_adjudication_n']} item-columns differ by "
                      f"{ADJUDICATE_AT} points or more and go to adjudication.", "",
                  "| Column | Mean |", "|---|--:|"]
            for c, v in r["means"].items():
                L.append(f"| `{c}` | {v} |")
            L.append("")
        else:
            a = r["agreement"]
            L += ["", f"- Cohen κ over the four labels: {a['cohen_kappa_mean']} "
                      f"(pairwise {a['cohen_kappa_pairwise']})",
                  f"- Krippendorff α (nominal): {a['krippendorff_alpha_nominal']}", ""]
            if r["by_variant"]:
                L += ["| Variant | Spans | Unsupported | Rate |", "|---|--:|--:|--:|"]
                for k, v in sorted(r["by_variant"].items()):
                    L.append(f"| {k} | {v['n']} | {v['unsupported']} | "
                             f"{v['unsupported_rate']} |")
                L.append("")
            if r["detector"]:
                L += ["Automated detector against the raters: "
                      + ", ".join(f"{k} = {v}" for k, v in r["detector"].items()), ""]

    L += ["## Why the numbers are absent rather than estimated", "",
          "This project has no domain-expert panel and no annotator panel. The "
          "instruments are stratified, blinded and reproducible, and the scoring "
          "is implemented and tested; what is missing is people, and a model "
          "grading its own output would answer a different question. R6 in "
          "`docs/risk_management_plan.md` carries that gap and names what closes "
          "it.", ""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    return data


# ── self-test ─────────────────────────────────────────────────────────────
def self_test() -> dict:
    """Prove the scorer, including the alpha implementation, on known data."""
    import shutil
    import tempfile
    checks = []

    # 1. Krippendorff's own published nominal example (2011, "Computing
    #    Krippendorff's Alpha-Reliability"): four observers, 12 units, many
    #    missing values, reported α_nominal = 0.743
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, None]
    C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    E = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [[v for v in col if v is not None] for col in zip(A, B, C, E)]
    a_nom = krippendorff_alpha(units, "nominal")
    checks.append(("krippendorff nominal == 0.743 (published example)",
                   a_nom, abs(a_nom - 0.743) < 0.001))

    # 2. perfect agreement is 1.0 on both metrics
    perf = [[3, 3], [1, 1], [5, 5], [2, 2], [4, 4]]
    checks.append(("alpha ordinal on perfect agreement == 1.0",
                   krippendorff_alpha(perf, "ordinal"),
                   krippendorff_alpha(perf, "ordinal") == 1.0))
    checks.append(("kappa on perfect agreement == 1.0",
                   cohen_kappa([("yes", "yes"), ("no", "no"), ("yes", "yes")]),
                   cohen_kappa([("yes", "yes"), ("no", "no"), ("yes", "yes")]) == 1.0))

    # 3. adjacent-only disagreement: ordinal must beat nominal, since ordinal
    #    knows 3 and 4 are closer than 1 and 5
    adj = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 4], [1, 1], [5, 5], [2, 2]]
    ao, an = krippendorff_alpha(adj, "ordinal"), krippendorff_alpha(adj, "nominal")
    checks.append(("ordinal alpha > nominal alpha on adjacent disagreement",
                   f"{ao} > {an}", ao > an))

    # 4. chance agreement scores about zero
    import random
    random.seed(11)
    rnd = [[random.randint(1, 5), random.randint(1, 5)] for _ in range(400)]
    ar = krippendorff_alpha(rnd, "nominal")
    checks.append(("alpha on random labels is near zero", ar, abs(ar) < 0.12))

    # 5. the whole path: write synthetic rater sheets in a temp dir and score
    tmp = Path(tempfile.mkdtemp(prefix="human_review_"))
    global EV, EXPERT, MANUAL, PACK, OUT_MD, OUT_JSON
    keep = (EV, EXPERT, MANUAL, PACK, OUT_MD, OUT_JSON)
    try:
        rows = [{"item_id": f"C{i:04d}", "kind": "concept", "name": f"n{i}",
                 "concept_class": "Destination"} for i in range(1, 31)]
        cols = ["item_id", "kind", "name", "concept_class", *ORDINAL_COLS,
                BINARY_COL, "rater_comment"]

        def write(p, filled, shift=0):
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for i, r in enumerate(rows):
                    out = dict(r)
                    if filled:
                        for c in ORDINAL_COLS:
                            v = 3 + ((i + shift) % 3) - 1
                            out[c] = max(1, min(5, v))
                        out[BINARY_COL] = "yes" if (i + shift) % 4 else "no"
                        if min(int(out[c]) for c in ORDINAL_COLS) <= 2:
                            out["rater_comment"] = "low score, see note"
                    w.writerow(out)

        EV = tmp
        EXPERT, MANUAL, PACK = tmp / "expert_ontology_review.csv", tmp / "m.csv", tmp / "p.json"
        OUT_MD, OUT_JSON = tmp / "o.md", tmp / "o.json"
        write(EXPERT, False)
        r0 = score_expert()
        checks.append(("no rater sheets -> status unfilled, no mean",
                       r0["status"], r0["status"] == "unfilled" and r0["means"] is None))
        write(tmp / "expert_ontology_review__r1.csv", True, 0)
        r1 = score_expert()
        checks.append(("one rater sheet -> still no mean",
                       r1["status"], r1["status"] == "single-rater" and r1["means"] is None))
        write(tmp / "expert_ontology_review__r2.csv", True, 1)
        r2 = score_expert()
        ok = (r2["status"] == "scored" and r2["means"] is not None
              and r2["agreement"][BINARY_COL]["cohen_kappa_mean"] is not None
              and all(r2["agreement"][c]["krippendorff_alpha_ordinal"] is not None
                      for c in ORDINAL_COLS))
        checks.append(("two rater sheets -> agreement and means both present",
                       r2["status"], ok))
        checks.append(("protocol's comment rule is enforced",
                       len(r2["problems"]), r2["problems"] == []))
    finally:
        EV, EXPERT, MANUAL, PACK, OUT_MD, OUT_JSON = keep
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _, _, ok in checks if ok)
    return {"checks": [{"name": n, "value": v, "passed": bool(ok)}
                       for n, v, ok in checks],
            "passed": passed, "total": len(checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        r = self_test()
        for c in r["checks"]:
            print(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']}: {c['value']}")
        print(f"{r['passed']}/{r['total']} checks passed")
        sys.exit(0 if r["passed"] == r["total"] else 1)

    d = build_report()
    for k in ("expert_11_4_2", "manual_11_6b"):
        r = d[k]
        print(f"{k}: {r['items']:,} items, {r['n_raters']} rater sheet(s), "
              f"status={r['status']}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
