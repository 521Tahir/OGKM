"""
exp034 - onto_v7: compositional ontology context from co-occurrence clusters.

Why this arm exists
-------------------
Nine earlier attempts localised the constraint. It is NOT retrieval coverage:
onto_v6 nearly doubled gold-entity coverage (0.270 -> 0.512) and moved one metric
by +0.01. It is NOT prompting: an explicit grounding instruction (onto_g) made
every metric worse. What Simple RAG supplies and a fact list does not is
COMPOSITION - knowing that Seongsan Ilchulbong belongs on day 1 at sunrise,
paired with Seopjikoji and an afternoon at Hamdeok Beach.

Ontology v3.0.0 (exp031) holds 10,124 PMI-filtered `co_occurs_with` edges: pairs
of places that appear in the same training itinerary far more often than chance.
That is exactly the compositional signal, and no arm has used it - v1/v3/v4/v5/v6
all read the 1.3.0 graph, which has no co-occurrence edges.

v7 therefore emits CLUSTERS rather than independent facts: an anchor place plus
the places most often visited alongside it, grouped so the block reads as a set of
combinations rather than a list of trivia.

Extraction noise: the v3.0.0 entity pass admitted verbs and section headings
("Uncover", "Grilling", "General Travel Advice"). Those are filtered here against
the gazetteer plus place-shape heuristics, so clusters contain places only.

Fairness: co-occurrence statistics come from `splits/train.jsonl` only - the same
corpus the RAG arm indexes. No test or val data is read.

  python -m experiments.exp034_onto_compose.build_onto_v7 --splits test bench
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

ONTO = ROOT / "artifacts" / "ontologies" / "3.0.0"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
PRIOR = ROOT / "experiments" / "exp033_onto_prior" / "entity_prior.json"

# Verbs, section headings and bare categories that the v3.0.0 entity pass let in.
NOT_A_PLACE = {
    "uncover", "appreciate", "explore", "discover", "experience", "enjoy", "visit",
    "relax", "grilling", "steak", "general travel advice", "nature highlights",
    "hands-on activities", "travel advice", "practical tips", "getting around",
    "final thoughts", "additional tips", "morning", "afternoon", "evening", "day",
    "transport", "transportation", "accommodation", "accommodations", "budget",
    "highlights", "overview", "itinerary", "optional", "note", "notes", "tip",
    "tips", "summary", "conclusion", "recommendation", "recommendations",
}
PLACE_SUFFIX = re.compile(
    r"(beach|cave|waterfall|falls|museum|park|market|temple|trail|oreum|peak|island|"
    r"village|garden|forest|rock|cliff|coast|harbor|harbour|port|airport|tower|"
    r"street|road|hall|center|centre|springs|resort|land|house|bridge|lake|"
    r"observatory|farm|arboretum|sa|san|do|gul|bong)\b", re.I)


def key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "bench"])
    ap.add_argument("--head-arm", default="onto_v3")
    ap.add_argument("--clusters", type=int, default=9)
    ap.add_argument("--per-cluster", type=int, default=5)
    ap.add_argument("--head-share", type=float, default=0.25)
    a = ap.parse_args()

    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    gaz_names = {key(x["name"]) for x in gaz["locations"] if isinstance(x, dict)}

    # A permissive filter (any multi-word capitalised phrase) let through
    # "Your Trip Cautiously", "Stay Calm", "Appropriate Authorities". Clusters are
    # only useful if every member is a real place, so require positive evidence:
    # a gazetteer entry, or a place-type suffix with no imperative-verb head.
    VERB_HEAD = re.compile(r"^(your|stay|seek|keep|check|make|take|be|do|get|use|"
                           r"plan|book|bring|avoid|respect|follow|consider|"
                           r"exploring|appropriate|general|additional|practical)\b", re.I)

    def is_place(name: str) -> bool:
        k = key(name)
        if k in NOT_A_PLACE or len(k) < 5 or VERB_HEAD.match(k):
            return False
        if k in gaz_names:
            return True
        # a place-type suffix, and the head word is not a verb/adjective heading
        return bool(PLACE_SUFFIX.search(k))

    rels = json.loads((ONTO / "relations.json").read_text(encoding="utf-8"))
    adj: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in rels:
        if r.get("predicate") != "co_occurs_with":
            continue
        s, o = str(r.get("subject", "")), str(r.get("object", ""))
        if not (is_place(s) and is_place(o)):
            continue
        adj[s].append((float(r.get("provenance", {}).get("confidence", 0.5)), o))
    for k in adj:
        adj[k].sort(reverse=True)
    print(f"co-occurrence graph after place filtering: {len(adj):,} anchors, "
          f"{sum(len(v) for v in adj.values()):,} edges")

    # Anchor ranking: corpus salience, restricted to anchors that have a cluster.
    prior = [e for e, _ in json.loads(PRIOR.read_text(encoding="utf-8"))] if PRIOR.exists() else []
    anchors = [e for e in prior if e in adj and is_place(e)]
    anchors += [e for e in adj if e not in set(anchors)]
    print(f"ranked anchors with clusters: {len(anchors):,} "
          f"(top: {', '.join(anchors[:6])})")

    def cluster_block(seed_names: list[str], budget: int) -> str:
        used, lines = set(), []
        for anc in seed_names:
            if len(lines) >= a.clusters * 2:
                break
            partners = [o for _, o in adj.get(anc, []) if key(o) not in used][: a.per_cluster]
            if len(partners) < 2:
                continue
            used.add(key(anc))
            used.update(key(p) for p in partners)
            lines.append(f"- {anc} is commonly combined with: " + ", ".join(partners) + ".")
        return _truncate("\n".join(lines), budget)

    head_budget = int(C.CTX_MAX_CHARS * a.head_share)
    body_budget = C.CTX_MAX_CHARS - head_budget
    print(f"budget {C.CTX_MAX_CHARS} = {head_budget} question-specific + {body_budget} clusters")

    for split in a.splits:
        head_p = C.CTX_DIR / f"{split}_{a.head_arm}.jsonl"
        if not head_p.exists():
            print(f"  skip {split}: missing {head_p.name}")
            continue
        out = C.CTX_DIR / f"{split}_onto_v7.jsonl"
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for line in open(head_p, encoding="utf-8"):
                r = json.loads(line)
                q = r["Input"].lower()
                # anchors the question names, else the corpus-salience ranking
                named = [e for e in anchors if key(e) in q][: a.clusters]
                seeds = named + [e for e in anchors if e not in named]
                head = r.get("context", "")
                if head.startswith("[no ontology"):
                    head = ""
                head = _truncate(head, head_budget)
                body = cluster_block(seeds, body_budget)
                parts = ([head] if head else []) + [
                    "Places commonly visited together on Jeju:\n" + body]
                r["context"] = "\n\n".join(parts)
                r["n_named_anchors"] = len(named)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        import statistics as st
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        print(f"  {split}: {n:,} items -> {out.name}  "
              f"ctx {int(st.mean(len(x['context']) for x in rows)):,} chars, "
              f"named anchors {st.mean(x['n_named_anchors'] for x in rows):.2f}")


if __name__ == "__main__":
    main()
