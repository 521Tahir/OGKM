"""
exp040 - onto_v9: coverage AND composition, the last untried mechanism.

What every prior arm was missing
--------------------------------
Measured ceiling: the ontology's total token-weighted coverage of test gold
entities is 0.599, and onto_v6 already retrieves 0.512 - 85% of everything the
graph holds. So no further coverage work can close the gap to Simple RAG. exp038
confirmed it: pairing onto_v6's coverage with a dedicated adapter moved free-form
BERT-F1 by +0.0003.

What the graph holds and NO retrieval arm has used as primary content is
COMPOSITION: 10,124 PMI-filtered `co_occurs_with` edges in ontology v3.0.0 - pairs
of places appearing in the same training itinerary far more often than chance,
mined from `splits/train.jsonl`, the same corpus the RAG arm indexes. A gold
itinerary is a composition, and that is precisely the information an entity-fact
list cannot express.

onto_v7 tried this and lost 95% of the signal: its place filter required a
gazetteer hit or a place-type suffix, cutting 10,124 edges to 527. Whitelisting
against the corpus-salience prior instead (600 train-frequent entities + the
69-place gazetteer) keeps 1,608 edges over 370 anchors and reaches 0.416 of gold
entity mentions.

v9 therefore composes the budget as:
  * a popularity-prior fact block   (the onto_v6 coverage mechanism), and
  * a co-occurrence pairing block   (the composition mechanism),
kept to the same 3,000-character budget as every other arm.

Section headings survive the whitelist ("Exploring Jeju City", "Andeok Region",
"Jeju's Hidden Gems"), so they are removed explicitly - a cluster is only useful
if its members are places.

  python -m experiments.exp040_onto_compose2.build_onto_v9 --splits train test bench
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import _truncate
from src.utils.proctitle import set_proctitle

PRIOR = ROOT / "experiments" / "exp033_onto_prior" / "entity_prior.json"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"

# Place membership comes from the ontology's OWN concept typing, not a regex.
# The corpus-salience prior admits itinerary section headings ("Leisure
# Experiences", "Wildlife Adventures", "Final Experiences") that pass any
# shape-based test; concept_class does not. Region IS a place class here - Andeok,
# Seongsan, Hallim and Aewol are real Jeju regions - so it stays in.
PLACE_CLASSES = {
    "Destination", "CulturalAttraction", "NaturalAttraction", "Restaurant",
    "Accommodation", "Attraction", "Trail", "Market", "Landmark", "Museum",
    "Beach", "ReligiousSite", "Park", "Mountain", "Island", "City", "Region",
    "Route",
}


def key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test", "bench"])
    ap.add_argument("--head-arm", default="onto_v6",
                    help="arm supplying the coverage block (popularity prior)")
    ap.add_argument("--head-share", type=float, default=0.55)
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--per-cluster", type=int, default=5)
    a = ap.parse_args()

    prior_ranked = [e for e, _ in json.loads(PRIOR.read_text(encoding="utf-8"))]
    concepts = json.loads((ROOT / "artifacts" / "ontologies" / C.ONTO_CLEAN_VERSION
                           / "concepts.json").read_text(encoding="utf-8"))
    white = {key(x["name"]) for x in concepts
             if isinstance(x, dict) and x.get("name")
             and x.get("concept_class") in PLACE_CLASSES}
    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    white |= {key(x["name"]) for x in gaz["locations"] if isinstance(x, dict)}
    print(f"typed place whitelist: {len(white):,} names")

    def is_place(n: str) -> bool:
        return len(key(n)) >= 5 and key(n) in white

    rels = json.loads((ROOT / "artifacts" / "ontologies" / "3.0.0" / "relations.json")
                      .read_text(encoding="utf-8"))
    adj: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for r in rels:
        if r.get("predicate") != "co_occurs_with":
            continue
        s, o = str(r.get("subject", "")), str(r.get("object", ""))
        if is_place(s) and is_place(o) and key(s) != key(o):
            adj[s].append((float(r.get("provenance", {}).get("confidence", 0.5)), o))
    for k in adj:
        adj[k].sort(reverse=True)
    print(f"composition graph: {len(adj):,} anchors, "
          f"{sum(len(v) for v in adj.values()):,} edges")

    anchors = [e for e in prior_ranked if e in adj] + \
              [e for e in adj if e not in set(prior_ranked)]
    print(f"ranked anchors: {len(anchors):,} (top: {', '.join(anchors[:6])})")

    head_budget = int(C.CTX_MAX_CHARS * a.head_share)
    comp_budget = C.CTX_MAX_CHARS - head_budget
    print(f"budget {C.CTX_MAX_CHARS} = {head_budget} coverage + {comp_budget} composition")

    def composition(q: str) -> str:
        named = [e for e in anchors if key(e) in q]
        seeds = named + [e for e in anchors if e not in named]
        used, lines = set(), []
        for anc in seeds:
            if len(lines) >= a.clusters:
                break
            if key(anc) in used:
                continue
            partners = [o for _, o in adj.get(anc, []) if key(o) not in used][: a.per_cluster]
            if len(partners) < 2:
                continue
            used.add(key(anc))
            used.update(key(p) for p in partners)
            lines.append(f"- Visitors who go to {anc} usually also visit "
                         + ", ".join(partners[:-1]) + f" and {partners[-1]}.")
        return _truncate("\n".join(lines), comp_budget)

    for split in a.splits:
        src = C.CTX_DIR / f"{split}_{a.head_arm}.jsonl"
        if not src.exists():
            print(f"  skip {split}: missing {src.name}")
            continue
        out = C.CTX_DIR / f"{split}_onto_v9.jsonl"
        n, ncomp = 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                head = _truncate(r.get("context", ""), head_budget)
                comp = composition(r["Input"].lower())
                parts = [p for p in (head, ) if p]
                if comp:
                    parts.append("Places commonly visited together on Jeju:\n" + comp)
                    ncomp += 1
                r["context"] = "\n\n".join(parts)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        import statistics as stt
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        print(f"  {split}: {n:,} -> {out.name}  ctx "
              f"{int(stt.mean(len(x['context']) for x in rows)):,} chars, "
              f"{ncomp:,} with composition")
        if split == "test":
            print("  sample tail:", rows[5]["context"][-260:].replace("\n", " | "))


if __name__ == "__main__":
    main()
