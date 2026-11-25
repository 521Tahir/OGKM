"""
exp039 - entity-level factual metrics, ported from the companion project.

Why these metrics and not the five we have
------------------------------------------
Four of the five headline metrics (BERT-F1, METEOR, COMET, Key-token) score
overlap with ONE writer's phrasing of ONE valid itinerary. Section 5.5 of the
results draft already records that 36.4% of the tokens Key-token counts are
scaffolding words ("Morning", "Afternoon", "Visit"). An ontology supplies facts,
not phrasing, so those metrics systematically undercount what it contributes -
which is why fifteen ontology variants all landed in the same narrow band.

The companion project at /home/tahirahmad/Ontology_Project evaluates the same
research question with entity-level metrics instead, and reports Onto-RAG BEATING
Doc-RAG on them (main_results_table.md, Table 3: precision 49.8% vs 48.0%,
grounding 58.0% vs 56.6%). Those definitions are ported here verbatim in spirit:

  entity_recall     gold entities that appear in the generated answer
  entity_precision  generated entities that appear in the gold answer
  entity_f1         harmonic mean of the two
  onto_grounding    generated entities that exist in the ontology
                    (= factual_correctness_score in src/evaluation/response_metrics.py)
  halluc_rate       generated entities in NEITHER the gold answer NOR the
                    ontology - i.e. invented, ungroundable

All are computed from generation files that already exist, on the full
4,607-item test split. No training, no GPU.

  python -m experiments.exp039_entity_metrics.eval_entity --models llama3.2-1b llama3.2-3b
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C

OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"

# Companion project's span pattern (src/evaluation/response_metrics.py) widened by
# one token to catch four-word Jeju place names ("Jeju Folk Village Museum").
SPAN = re.compile(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})\b")
FALSE_POS = {
    "the","a","an","in","on","of","for","and","or","is","are","it","this","that",
    "we","i","you","your","my","our","he","she","they","there","here","also",
    "morning","afternoon","evening","night","day","days","visit","enjoy","explore",
    "start","end","optional","note","notes","tip","tips","overview","summary",
    "breakfast","lunch","dinner","jeju","jeju island","korea","south korea",
}


def entities(text: str) -> set[str]:
    out = set()
    for s in SPAN.findall(text or ""):
        k = re.sub(r"\s+", " ", s).strip().lower()
        if len(k) >= 4 and k not in FALSE_POS:
            out.add(k)
    return out


def onto_names() -> set[str]:
    d = ROOT / "artifacts" / "ontologies" / C.ONTO_CLEAN_VERSION
    names = set()
    for fn, keys in [("concepts.json", ("label", "name", "preferred_label", "id"))]:
        for x in json.loads((d / fn).read_text(encoding="utf-8")):
            if isinstance(x, dict):
                for k in keys:
                    if x.get(k):
                        names.add(str(x[k]).strip().lower())
                        break
            elif isinstance(x, str):
                names.add(x.strip().lower())
    for r in json.loads((d / "relations.json").read_text(encoding="utf-8")):
        for k in ("subject", "object"):
            v = r.get(k)
            if v:
                names.add(str(v).strip().lower())
    return {n for n in names if len(n) >= 4}


def grounded(e: str, onto: set[str]) -> bool:
    """Companion project's matching rule: exact, or containment either way."""
    if e in onto:
        return True
    return any(e in k or k in e for k in onto)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    onto = onto_names()
    print(f"ontology surface forms: {len(onto):,}\n")
    # cache containment tests - the answer vocabulary repeats heavily across items
    gcache: dict[str, bool] = {}

    ARMS = [("rag", "generations", "Simple RAG"),
            ("rag", "generations_ded", "Simple RAG @ dedicated"),
            ("onto_v3", "generations", "Ontology V3"),
            ("onto_v3", "generations_ded", "Ontology V3 @ dedicated"),
            ("onto_v6", "generations_ded", "Ontology V6 @ dedicated"),
            ("hybrid20", "generations", "Onto-RAG hybrid 20%")]

    for model in a.models:
        rows = {}
        for arm, gdir, label in ARMS:
            p = OUT / gdir / f"{a.split}__{model}__{arm}.jsonl"
            if not p.exists():
                continue
            recs = []
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            per = {k: [] for k in ("rec", "prec", "f1", "ground", "halluc")}
            for r in recs:
                gold = entities(r.get("Output") or r.get("reference") or "")
                pred = entities(r.get("answer") or r.get("prediction") or "")
                if not gold and not pred:
                    continue
                tp = len(gold & pred)
                rec = tp / len(gold) if gold else 1.0
                prc = tp / len(pred) if pred else 1.0
                per["rec"].append(rec)
                per["prec"].append(prc)
                per["f1"].append(0.0 if rec + prc == 0 else 2 * rec * prc / (rec + prc))
                if pred:
                    g = 0
                    h = 0
                    for e in pred:
                        if e not in gcache:
                            gcache[e] = grounded(e, onto)
                        ing = gcache[e]
                        g += 1 if ing else 0
                        if not ing and e not in gold:
                            h += 1
                    per["ground"].append(g / len(pred))
                    per["halluc"].append(h / len(pred))
            rows[label] = {k: st.mean(v) for k, v in per.items() if v} | {"n": len(recs)}

        if not rows:
            print(f"{model}: no generations found")
            continue
        print(f"## {model} — {a.split} split")
        print("| Arm | n | Entity Recall | Entity Prec | Entity F1 | Onto Grounding | Halluc. |")
        print("|---|---|---|---|---|---|---|")
        for lab, m in rows.items():
            print(f"| {lab} | {m['n']:,} | {m['rec']*100:.1f}% | {m['prec']*100:.1f}% | "
                  f"{m['f1']*100:.1f}% | {m['ground']*100:.1f}% | {m['halluc']*100:.1f}% |")
        print()


if __name__ == "__main__":
    main()
