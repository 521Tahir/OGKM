"""
Rebuild the demo page from whatever results are currently on disk.

Selects the strongest `onto_retrieved_and_used` cases (ontology surfaced a gold
fact RAG lacked AND the model used it), embeds them with their full injected
contexts, and rewrites docs/jeju_ontology_demo.html.

  python -m experiments.exp028_demo.make_demo --onto-arm onto_c --n 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.utils.proctitle import set_proctitle

HERE = Path(__file__).resolve().parent
E26 = ROOT / "experiments" / "exp026_onto_improve" / "outputs"
E27 = ROOT / "experiments" / "exp027_qualitative" / "outputs"
BENCH = ROOT / "experiments" / "exp025_multihop" / "outputs" / "jeju_bench_200.jsonl"
PAGE = ROOT / "docs" / "jeju_ontology_demo.html"
TEMPLATE = HERE / "demo_template.html"


def trim(t: str, n: int) -> str:
    t = (t or "").strip()
    return t if len(t) <= n else t[:n].rsplit("\n", 1)[0] + "\n…"


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--onto-arm", default="onto_c")
    ap.add_argument("--split", default="bench")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    qual = json.loads((E27 / f"qualitative_{a.split}_{a.onto_arm}.json").read_text(encoding="utf-8"))
    rank = json.loads((E26 / "variant_ranking.json").read_text(encoding="utf-8"))
    bench = {json.loads(l)["record_id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}

    # strongest decisive wins, at most one per (item, model) and spread across models
    wins = [r for r in qual["rows"] if r["mode"] == "onto_retrieved_and_used"]
    wins.sort(key=lambda r: -r["gap"])
    picked, seen_models, seen_items = [], {}, set()
    for r in wins:
        if seen_models.get(r["model"], 0) >= 2:
            continue
        if (r["record_id"], r["model"]) in seen_items:
            continue
        seen_items.add((r["record_id"], r["model"]))
        seen_models[r["model"]] = seen_models.get(r["model"], 0) + 1
        picked.append(r)
        if len(picked) >= a.n:
            break

    cases = []
    for r in picked:
        b = bench[r["record_id"]]
        cases.append({
            "id": r["record_id"], "model": C.MODEL_BY_KEY[r["model"]]["name"],
            "kind": b["kind"], "template": b["template"], "gap": round(r["gap"], 2),
            "question": r["question"], "gold": r["reference"], "evidence": b["evidence"],
            "entities": r["gold_entities"],
            "ctxRag": trim(r["ctx_rag"], 1100), "ctxOnto": trim(r["ctx_onto"], 1100),
            "ansRag": trim(r["answer_rag"], 600), "ansOnto": trim(r["answer_onto"], 600),
            "entCtxRag": r["ent_ctx_rag"], "entCtxOnto": r["ent_ctx_onto"],
            "entAnsRag": r["ent_ans_rag"], "entAnsOnto": r["ent_ans_onto"]})

    mh = rank["macro"]["multihop"]
    payload = {"cases": cases, "modes": qual["modes"], "modeGap": qual["mode_mean_gap"],
               "nPaired": qual["n"],
               "results": {"rag": mh["rag"], "onto_c": mh[a.onto_arm]},
               "metrics": [["bertf1", "BERT-F1"], ["meteor", "METEOR"], ["comet", "COMET"],
                           ["keytok", "Key-token"], ["judge", "Resp. Acc."]]}

    tpl = TEMPLATE.read_text(encoding="utf-8")
    PAGE.write_text(tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
                    encoding="utf-8")
    (HERE / "demo_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"rebuilt {PAGE} ({PAGE.stat().st_size:,} bytes) with {len(cases)} cases")
    for c in cases:
        print(f"  {c['id']} {c['model']:16s} +{c['gap']:.2f}  {c['question'][:64]}")


if __name__ == "__main__":
    main()
