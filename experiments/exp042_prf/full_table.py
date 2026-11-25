"""
exp042 - one consolidated table: every evaluation metric, every model, every arm.

Joins the two metric families that were computed separately:
  scores/ , scores_ded/              BERT-F1, METEOR, COMET, Key-token, Resp.Acc (LLM judge)
  exp042 prf_results*.json           token & entity Precision / Recall / F1 / Accuracy

Emits Markdown and CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "deepseek-r1-8b",
          "qwen3-8b", "mistral-7b"]
ARMS = [("Simple RAG", "rag", "scores"),
        ("Simple RAG (dedicated)", "rag", "scores_ded"),
        ("Ontology V3", "onto_v3", "scores"),
        ("Ontology V3 (dedicated)", "onto_v3", "scores_ded"),
        ("Ontology (cleaned)", "onto_c", "scores"),
        ("Ontology V6 (dedicated)", "onto_v6", "scores_ded"),
        ("Onto-RAG hybrid 20%", "hybrid20", "scores"),
        ("Onto-RAG hybrid 30%", "hybrid", "scores"),
        ("Vanilla (no fine-tune)", "base", "scores")]
CLASSIC = [("bertf1", "BERT-F1"), ("meteor", "METEOR"), ("comet", "COMET"),
           ("keytok", "Key-tok"), ("judge", "Resp.Acc")]


def classic(sdir, split, model, arm):
    p = OUT / sdir / f"{split}__{model}__{arm}.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in open(p, encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None
    o = {"n": len({r["record_id"] for r in rows})}
    for k, _ in CLASSIC:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        o[k] = st.mean(v) if v else None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    prf_file = ("prf_results.json" if a.split == "test" else f"prf_results_{a.split}.json")
    P = json.loads((ROOT / "experiments" / "exp042_prf" / prf_file)
                   .read_text(encoding="utf-8"))["results"]

    hdr = (["Model", "Arm", "n"] + [l for _, l in CLASSIC] +
           ["Tok-P", "Tok-R", "Tok-F1", "Acc@tF1", "Ent-P", "Ent-R", "Ent-F1", "Acc@hit"])
    rows, md = [], []
    for model in MODELS:
        block = []
        for label, arm, sdir in ARMS:
            c = classic(sdir, a.split, model, arm)
            pr = (P.get(model) or {}).get(label)
            if not c and not pr:
                continue
            n = (c or pr).get("n")
            r = [model, label, n]
            r += [None if not c else c.get(k) for k, _ in CLASSIC]
            for k in ("token_precision", "token_recall", "token_f1", "acc_token_f1_50",
                      "entity_precision", "entity_recall", "entity_f1", "acc_entity_hit"):
                r.append(None if not pr else pr.get(k))
            block.append(r); rows.append(r)
        if not block:
            continue
        # best per numeric column within the model block
        best = {}
        for ci in range(3, len(hdr)):
            vals = [(b[ci], i) for i, b in enumerate(block) if isinstance(b[ci], (int, float))]
            if vals:
                best[ci] = max(vals)[1]
        md.append(f"\n### {model}\n")
        md.append("| " + " | ".join(hdr[1:]) + " |")
        md.append("|" + "---|" * (len(hdr) - 1))
        for i, b in enumerate(block):
            cells = [b[1], f"{b[2]:,}"]
            for ci in range(3, len(hdr)):
                v = b[ci]
                if v is None:
                    cells.append("—"); continue
                s = f"{v:.4f}" if ci < 3 + len(CLASSIC) else f"{v*100:.1f}"
                cells.append(f"**{s}**" if best.get(ci) == i else s)
            md.append("| " + " | ".join(cells) + " |")

    title = ("Full held-out test split — 4,607 items" if a.split == "test"
             else "Factual multi-hop benchmark — 200 questions")
    head = f"""# All evaluation metrics — every model, every arm

**{title}**

Columns 1–5 are the original metrics (means over scored items). Columns 6–13 are the
Precision / Recall / F1 / Accuracy family, shown as percentages.

- **Tok-*** — SQuAD-style content-token multiset overlap against the gold answer.
- **Ent-*** — the same over the 69 curated gazetteer place names, i.e. did the answer
  name the right places, independent of phrasing.
- **Acc@tF1** — fraction of items reaching token F1 ≥ 0.50.
- **Acc@hit** — fraction of items naming at least one gold place.
- **Resp.Acc** — the LLM-judge Response Accuracy (a separate measurement, not recomputed).

Bold = best value in that column for that model.
"""
    body = head + "\n".join(md) + "\n"
    mdp = ROOT / "experiments" / "exp042_prf" / f"ALL_METRICS_{a.split}.md"
    mdp.write_text(body, encoding="utf-8")
    csvp = ROOT / "experiments" / "exp042_prf" / f"ALL_METRICS_{a.split}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f); wtr.writerow(hdr); wtr.writerows(rows)
    print(body)
    print(f"\nwrote {mdp.relative_to(ROOT)} and {csvp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
