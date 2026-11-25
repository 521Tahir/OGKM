"""Render the qualitative analysis as a readable markdown report with full
side-by-side cases, so the mechanism claims can be checked against real output."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.utils.proctitle import set_proctitle

OUT = Path(__file__).resolve().parent / "outputs"

MODE_STORY = {
    "onto_retrieved_and_used": "Ontology surfaced a gold fact RAG missed, and the model used it. "
                               "This is the ontology's decisive win condition.",
    "rag_retrieved_and_used": "RAG surfaced a gold fact the ontology missed, and the model used it.",
    "onto_retrieved_not_used": "The ontology had the right fact but the model did not put it in the "
                               "answer. Pure headroom: better grounding, not better retrieval.",
    "rag_retrieved_not_used": "RAG had the fact and the model ignored it.",
    "both_retrieved_similar": "Both contexts carried comparable gold content; the gap is decoding "
                              "noise rather than retrieval.",
    "neither_retrieved": "Neither arm retrieved any gold entity — both answers are unsupported "
                         "guesses.",
}


def wrap(t: str, n: int = 100, indent: str = "") -> str:
    return "\n".join(textwrap.fill(line, n, initial_indent=indent, subsequent_indent=indent)
                     for line in (t or "").strip().splitlines() if line.strip())


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench")
    ap.add_argument("--onto-arm", default="onto_c")
    ap.add_argument("--cases", type=int, default=4)
    a = ap.parse_args()

    d = json.loads((OUT / f"qualitative_{a.split}_{a.onto_arm}.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    L = [f"# Qualitative analysis — Ontology ({a.onto_arm}) vs Simple RAG", "",
         f"Split: `{a.split}` · paired items: **{d['n']:,}** "
         f"(ontology better {d['onto_win']}, RAG better {d['rag_win']}, tie {d['tie']})", "",
         "Each item is classified by *mechanism*: did the arm's context actually contain the "
         "gold entities, and did the model then use them? This separates a retrieval win from "
         "a generation win.", "",
         "## Mechanism breakdown", "",
         "| Mechanism | Items | Share | Mean judge gap (onto − rag) | Reading |",
         "|---|---|---|---|---|"]
    for mode, cnt in sorted(d["modes"].items(), key=lambda x: -x[1]):
        L.append(f"| `{mode}` | {cnt:,} | {100*cnt/d['n']:.1f}% | "
                 f"{d['mode_mean_gap'][mode]:+.4f} | {MODE_STORY.get(mode, '')} |")

    dec_o = d["modes"].get("onto_retrieved_and_used", 0)
    dec_r = d["modes"].get("rag_retrieved_and_used", 0)
    L += ["", "### The headline", "",
          f"- **Decisive ontology wins are {dec_o/max(dec_r,1):.1f}× more frequent than decisive "
          f"RAG wins** ({dec_o} vs {dec_r} items). When the ontology surfaces a fact RAG lacks and "
          f"the model uses it, the judge score moves by "
          f"{d['mode_mean_gap'].get('onto_retrieved_and_used', 0):+.3f}.",
          f"- **The largest headroom is `onto_retrieved_not_used` "
          f"({d['modes'].get('onto_retrieved_not_used',0)} items, "
          f"{100*d['modes'].get('onto_retrieved_not_used',0)/d['n']:.1f}%)** — the ontology "
          f"already had the answer and the model failed to use it. That is a grounding/prompting "
          f"problem, not a retrieval problem.",
          f"- Roughly half of all items ({100*d['modes'].get('both_retrieved_similar',0)/d['n']:.0f}%) "
          f"are ties in content, where the measured gap is decoding noise.", ""]

    for mode in ("onto_retrieved_and_used", "onto_retrieved_not_used", "rag_retrieved_and_used"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        sel.sort(key=lambda r: -abs(r["gap"]))
        L += [f"## Cases: `{mode}`", "", MODE_STORY.get(mode, ""), ""]
        for r in sel[: a.cases]:
            L += [f"### {C.MODEL_BY_KEY[r['model']]['name']} · item {r['record_id']} "
                  f"· judge gap {r['gap']:+.2f}", "",
                  f"**Question** — {r['question']}", "",
                  f"**Gold** — {r['reference']}", "",
                  f"**Gold entities** — `{', '.join(r['gold_entities']) or '(none)'}`", "",
                  f"- in RAG context: `{', '.join(r['ent_ctx_rag']) or '—'}`",
                  f"- in ontology context: `{', '.join(r['ent_ctx_onto']) or '—'}`", "",
                  "**Simple RAG answer**", "", "```", wrap(r["answer_rag"][:700]), "```", "",
                  "**Ontology answer**", "", "```", wrap(r["answer_onto"][:700]), "```", "",
                  "**Ontology context (excerpt)**", "", "```",
                  wrap(r["ctx_onto"][:700]), "```", "", "---", ""]

    p = OUT / f"qualitative_report_{a.split}_{a.onto_arm}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
