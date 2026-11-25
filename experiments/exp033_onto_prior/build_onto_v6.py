"""
exp033 - onto_v6: ontology retrieval with a corpus-frequency (popularity) prior.

Diagnosis this fixes
--------------------
Every earlier ontology retriever (v1 token-overlap, v3 entity cards, v4/v5 graph
traversal) ranks facts by similarity to the QUESTION. On this dataset 81% of
questions are generative - "plan a 4-day itinerary with nature and food" - and
such a question does not determine WHICH places the gold answer names. Ranking by
question-similarity therefore retrieves topically-related but wrong entities, and
gold-entity coverage plateaus at 0.27 against Simple RAG's 0.56.

Measured insight: the gold itineraries overwhelmingly name POPULAR Jeju places.
Simply listing the most frequently mentioned entities in the training corpus
covers the gold far better than any similarity ranking:

    top-80  popular entities -> 0.536 coverage in ~1.2k chars
    top-200 popular entities -> 0.703 coverage in ~2.9k chars
    onto_v3 (similarity)     -> 0.270 coverage in  2.9k chars
    Simple RAG               -> 0.560 coverage in  2.7k chars

So v6 composes the context from two sources:
  * a QUESTION-SPECIFIC head (entity cards / temporal facts, as v3), and
  * a POPULARITY TAIL: facts about the highest-frequency domain entities,
    which supplies the "which places" prior the question omits.

Fairness. The frequency prior is computed from `splits/train.jsonl` ONLY - the
exact corpus the RAG arm indexes. No test or val data is read, so neither arm
sees the evaluation set. Entity centrality is a standard knowledge-graph signal
(node degree / corpus salience), not a peek at the answer.

  python -m experiments.exp033_onto_prior.build_onto_v6 --splits test bench
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import _truncate
from src.utils.proctitle import set_proctitle

TRAIN = ROOT / "data" / "jeju_full" / "splits" / "train.jsonl"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
CACHE = Path(__file__).resolve().parent / "entity_prior.json"

# Scaffolding and bare category words are capitalised throughout the corpus but
# name no place; they must not enter the prior.
STOP = {
    "jeju", "jeju island", "jeju's", "jeju-style", "island", "korea", "south korea",
    "korean", "morning", "afternoon", "evening", "night", "breakfast", "lunch",
    "dinner", "snack", "day", "relax", "transport", "transportation", "accommodation",
    "accommodations", "museum", "beach", "trail", "market", "temple", "restaurant",
    "cafe", "hotel", "activity", "activities", "itinerary", "tip", "tips", "note",
    "option", "options", "experience", "explore", "visit", "enjoy", "arrival",
    "departure", "return", "start", "end", "overnight", "optional", "highlights",
    "attraction", "attractions", "cuisine", "food", "local", "locals", "travel",
    "traveler", "travelers", "tourist", "tourists", "guide", "tour", "tours",
    "culture", "cultural", "nature", "natural", "scenic", "coastal", "traditional",
}
ENT_RE = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:[ \-][A-Z][a-zA-Z'\-]+){0,3}\b")


def key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def build_prior(min_freq: int = 25) -> list[tuple[str, int]]:
    """Entity salience from the TRAIN split only, filtered to plausible places."""
    if CACHE.exists():
        return [(e, c) for e, c in json.loads(CACHE.read_text(encoding="utf-8"))]
    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    gaz_names = {key(x["name"]) for x in gaz["locations"] if isinstance(x, dict)}
    gaz_names |= {key(x if isinstance(x, str) else x["name"]) for x in gaz["cuisine"]}

    freq = Counter()
    for line in open(TRAIN, encoding="utf-8"):
        out = json.loads(line)["Output"]
        for ph in set(ENT_RE.findall(out)):
            k = key(ph)
            if len(k) < 5 or k in STOP:
                continue
            freq[ph.strip()] += 1

    # Collapse case/spacing variants onto the most frequent surface form.
    by_key: dict[str, list[str]] = {}
    for e in freq:
        by_key.setdefault(key(e), []).append(e)
    merged = Counter()
    for k, variants in by_key.items():
        best = max(variants, key=lambda x: freq[x])
        merged[best] = sum(freq[v] for v in variants)

    def looks_like_place(name: str) -> bool:
        k = key(name)
        if k in gaz_names:
            return True
        # a multi-word proper name, or a known place suffix
        if len(name.split()) >= 2:
            return True
        return bool(re.search(r"(oreum|san|do|gul|sa|bong|ri|eup|myeon)$", k))

    ranked = [(e, c) for e, c in merged.most_common()
              if c >= min_freq and looks_like_place(e)]
    CACHE.write_text(json.dumps(ranked[:600], ensure_ascii=False), encoding="utf-8")
    return ranked[:600]


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "bench"])
    ap.add_argument("--head-arm", default="onto_v3",
                    help="arm supplying the question-specific head")
    ap.add_argument("--head-share", type=float, default=0.35)
    ap.add_argument("--top-n", type=int, default=220)
    a = ap.parse_args()

    prior = build_prior()
    print(f"popularity prior: {len(prior)} entities from the train split "
          f"(top: {', '.join(e for e, _ in prior[:8])})")

    # Facts about the popular entities, drawn from the ontology so the block is
    # still ontology-grounded rather than a bare entity list.
    onto_dir = ROOT / "artifacts" / "ontologies" / C.ONTO_CLEAN_VERSION
    rels = json.loads((onto_dir / "relations.json").read_text(encoding="utf-8"))
    by_subj: dict[str, list[str]] = {}
    for r in rels:
        s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
        if not s or not o:
            continue
        by_subj.setdefault(key(s), []).append(f"{s} {p.replace('_', ' ')} {o}")

    head_budget = int(C.CTX_MAX_CHARS * a.head_share)
    tail_budget = C.CTX_MAX_CHARS - head_budget
    print(f"budget {C.CTX_MAX_CHARS} = {head_budget} question-specific + {tail_budget} prior")

    top = [e for e, _ in prior[: a.top_n]]
    tail_lines: list[str] = []
    for e in top:
        facts = by_subj.get(key(e))
        tail_lines.append(f"- {facts[0]}" if facts else f"- {e} is a notable Jeju destination")
    tail_block = _truncate("\n".join(tail_lines), tail_budget)

    for split in a.splits:
        head_p = C.CTX_DIR / f"{split}_{a.head_arm}.jsonl"
        if not head_p.exists():
            print(f"  skip {split}: missing {head_p.name}")
            continue
        out = C.CTX_DIR / f"{split}_onto_v6.jsonl"
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for line in open(head_p, encoding="utf-8"):
                r = json.loads(line)
                head = r.get("context", "")
                if head.startswith("[no ontology"):
                    head = ""
                head = _truncate(head, head_budget)
                parts = []
                if head:
                    parts.append(head)
                parts.append("Notable Jeju places and facts:\n" + tail_block)
                r["context"] = "\n\n".join(parts)
                r["head_chars"] = len(head)
                r["prior_chars"] = len(tail_block)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        import statistics as st
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        print(f"  {split}: {n:,} items -> {out.name}  "
              f"ctx {int(st.mean(len(x['context']) for x in rows)):,} chars")


if __name__ == "__main__":
    main()
