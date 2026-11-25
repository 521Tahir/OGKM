"""
exp035 - onto_v8: the same ontology facts, rendered as prose instead of bullets.

The hypothesis this tests
-------------------------
Ten approaches varied the CONTENT of the ontology context - more facts, better
facts, cleaner facts, compositional facts, an explicit instruction to use them.
All four ontology-only arms landed in the same narrow band (BERT-F1
0.8829-0.8854) regardless. onto_v6 nearly doubled gold-entity coverage
(0.270 -> 0.512) and scored *lower* on four of five metrics.

That pattern says the binding constraint is not what the context contains. The
remaining difference between the arms is FORM: Simple RAG injects flowing
itinerary prose, every ontology arm injects a bulleted fact list, and the gold
answers are flowing prose. A model conditioned on bullets writes bullets.

v8 holds the content fixed (the onto_v6 fact set, best coverage of any ontology
arm) and changes only the rendering: triples are verbalised and joined into
connected sentences grouped by subject, so the block reads like description
rather than a database dump. If the band is caused by form, this moves it; if it
does not move, form is ruled out too and the ceiling is the task itself.

  python -m experiments.exp035_onto_prose.build_onto_v8 --splits test bench
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import _truncate
from src.utils.proctitle import set_proctitle

PRED_PHRASE = {
    "located_in": "is in", "near": "is close to", "has_activity": "is known for",
    "best_for": "suits", "available_in": "is available in", "served_at": "is served at",
    "reachable_by": "is reached by", "part_of": "is part of", "is_a": "is a",
    "suitable_for": "suits", "requires": "needs", "connected_to": "connects to",
    "offers": "offers", "has_seasonality": "is best in", "co_occurs_with": "pairs well with",
}


def verbalise(subject: str, facts: list[tuple[str, str]]) -> str:
    """One readable sentence per subject rather than one bullet per triple."""
    by_pred: dict[str, list[str]] = defaultdict(list)
    for p, o in facts:
        by_pred[p].append(o)
    clauses = []
    for p, objs in list(by_pred.items())[:4]:
        phrase = PRED_PHRASE.get(p, p.replace("_", " "))
        uniq = list(dict.fromkeys(objs))[:3]
        if len(uniq) == 1:
            clauses.append(f"{phrase} {uniq[0]}")
        else:
            clauses.append(f"{phrase} {', '.join(uniq[:-1])} and {uniq[-1]}")
    if not clauses:
        return ""
    return f"{subject} " + "; ".join(clauses) + "."


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "bench"])
    ap.add_argument("--source-arm", default="onto_v6",
                    help="arm whose fact CONTENT is reused; only rendering changes")
    a = ap.parse_args()

    bullet_re = re.compile(r"^-\s*(.+?)\s*$")
    # "Subject predicate object" as emitted by the ontology builders
    onto_dir = ROOT / "artifacts" / "ontologies" / C.ONTO_CLEAN_VERSION
    rels = json.loads((onto_dir / "relations.json").read_text(encoding="utf-8"))
    triple_of: dict[str, tuple[str, str, str]] = {}
    for r in rels:
        s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
        if s and o:
            triple_of[f"{s} {p.replace('_', ' ')} {o}"] = (s, p, o)

    for split in a.splits:
        src = C.CTX_DIR / f"{split}_{a.source_arm}.jsonl"
        if not src.exists():
            print(f"  skip {split}: missing {src.name}")
            continue
        out = C.CTX_DIR / f"{split}_onto_v8.jsonl"
        n, conv = 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
                loose: list[str] = []
                for ln in (r.get("context") or "").splitlines():
                    m = bullet_re.match(ln.strip())
                    if not m:
                        continue
                    body = m.group(1).rstrip(".")
                    t = triple_of.get(body)
                    if t:
                        grouped[t[0]].append((t[1], t[2]))
                    else:
                        loose.append(body)
                sentences = [verbalise(s, fs) for s, fs in grouped.items()]
                sentences = [s for s in sentences if s]
                # loose lines (temporal facts, prior entries) stay as short sentences
                sentences += [l if l.endswith(".") else l + "." for l in loose[:40]]
                prose = " ".join(sentences)
                if not prose.strip():
                    prose = (r.get("context") or "").strip()
                else:
                    conv += 1
                r["context"] = _truncate(prose, C.CTX_MAX_CHARS)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        import statistics as st
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        print(f"  {split}: {n:,} items -> {out.name}  "
              f"ctx {int(st.mean(len(x['context']) for x in rows)):,} chars, "
              f"{conv:,} converted to prose")
        if split == "test":
            print("  sample:", rows[3]["context"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
