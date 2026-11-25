"""
Interface test: every data path the single-page app dereferences must exist in the
embedded bundle. There is no browser on this machine (chromium needs libatk and
nine other system libraries that require root), so the page cannot be
click-tested here; this asserts the data contract instead, which is what the
JavaScript would fail on at runtime.

  python Jeju_Tourism_Research_Demo/experiments/evaluation/test_bundle_contract.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "Jeju_Tourism_Research_Demo" / "demo" / "interactive_application" / "index.html"
QIDS = ["Q1", "Q2", "Q3"]
MET = ["bertf1", "meteor", "comet", "keytok", "judge"]

html = APP.read_text(encoding="utf-8")
m = re.search(r'<script id="bundle" type="application/json">(.*?)</script>', html, re.S)
B = json.loads(m.group(1))

fails: list[str] = []


def ck(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)


# ── top level ─────────────────────────────────────────────────────────────
for k in ("meta", "ontology", "gazetteer", "queries", "rag", "onto", "answers",
          "scores", "full_dataset", "distances_km"):
    ck(k in B, f"bundle.{k} missing")

for k in ("model", "hf_id", "adapters", "decoding", "ctx_budget_chars",
          "embed_model", "ontology_version", "captured"):
    ck(k in B["meta"], f"meta.{k} missing")

# ── ontology ──────────────────────────────────────────────────────────────
o = B["ontology"]
for k in ("version", "source_files", "stats", "class_counts", "predicate_counts",
          "nodes", "edges", "hierarchy", "class_roots", "instances"):
    ck(k in o, f"ontology.{k} missing")
for k in ("typed_concepts", "graph_nodes", "relations", "distinct_predicates",
          "concept_classes", "hierarchy_edges", "class_tree_roots",
          "typed_instances", "gazetteer_places"):
    ck(k in o["stats"], f"ontology.stats.{k} missing")
ck(len(o["nodes"]) > 1000, f"too few nodes: {len(o['nodes'])}")
ck(len(o["edges"]) > 5000, f"too few edges: {len(o['edges'])}")
ck(len(o["hierarchy"]) > 0, "class hierarchy empty — Mode A would render nothing")
for n in o["nodes"][:50]:
    for k in ("id", "label", "klass", "degree"):
        ck(k in n, f"node missing {k}: {n.get('id')}")
for e in o["edges"][:50]:
    for k in ("s", "p", "o"):
        ck(k in e, f"edge missing {k}")
ids = {n["id"] for n in o["nodes"]}
dangling = [e for e in o["edges"] if e["s"] not in ids or e["o"] not in ids]
ck(not dangling, f"{len(dangling)} edges reference nodes absent from nodes[]")
for h in o["hierarchy"][:20]:
    ck("parent" in h and "child" in h, "hierarchy edge missing parent/child")

# ── per-query traces ──────────────────────────────────────────────────────
for q in QIDS:
    ck(q in B["queries"], f"queries.{q} missing")
    for k in ("id", "title", "text", "tests", "verifiable", "unsupported_by_data"):
        ck(k in B["queries"][q], f"queries.{q}.{k} missing")
    r = B["rag"].get(q, {})
    for k in ("stage_1_query", "stage_2_embed_model", "stage_2_query_prefix",
              "stage_3_corpus_size", "stage_4_index", "stage_5_top_k",
              "stage_6_hits", "stage_7_context", "timing_ms"):
        ck(k in r, f"rag.{q}.{k} missing")
    for k in ("embed", "search"):
        ck(k in r.get("timing_ms", {}), f"rag.{q}.timing_ms.{k} missing")
    for h in r.get("stage_6_hits", []):
        for k in ("rank", "cosine", "record_id", "source_question", "passage"):
            ck(k in h, f"rag.{q} hit missing {k}")
    t = B["onto"].get(q, {})
    for k in ("stage_1_query", "stage_2_intent_classes", "stage_3_card_index_size",
              "stage_4_entity_cards", "stage_5_triples_in_context",
              "stage_6_context", "timing_ms"):
        ck(k in t, f"onto.{q}.{k} missing")
    ck("retrieve" in t.get("timing_ms", {}), f"onto.{q}.timing_ms.retrieve missing")
    for c in t.get("stage_4_entity_cards", []):
        for k in ("entity", "concept_class", "cosine", "facts"):
            ck(k in c, f"onto.{q} card missing {k}")
    # Mode C needs the retrieved entities to be real graph nodes
    ents = [c["entity"] for c in t.get("stage_4_entity_cards", []) if c.get("entity")]
    missing = [e for e in ents if e not in ids]
    ck(not missing, f"onto.{q}: {len(missing)} retrieved entities not in graph: {missing[:3]}")
    for arm in ("rag", "onto_v3"):
        a = B["answers"].get(q, {}).get(arm, {})
        for k in ("prompt", "prompt_tokens", "answer", "answer_tokens",
                  "generation_s", "retrieval_ms", "adapter"):
            ck(k in a, f"answers.{q}.{arm}.{k} missing")
        ck(len(a.get("answer", "")) > 0, f"answers.{q}.{arm}.answer is empty")
        s = B["scores"]["per_query"].get(q, {}).get(arm, {})
        for k in ("F_factual", "C_constraints", "G_grounding", "A_completeness",
                  "H_halluc_avoidance", "composite_0_100", "constraints",
                  "completeness", "unsupported_claims", "named_places",
                  "grounded_in_own_context"):
            ck(k in s, f"scores.{q}.{arm}.{k} missing")
        for c in s.get("constraints", []):
            ck("id" in c and "pass" in c and "evidence" in c,
               f"scores.{q}.{arm} constraint malformed")

# ── dashboard tables ──────────────────────────────────────────────────────
ck(len(B["full_dataset"]) >= 2, "full_dataset should carry several models")
for model, splits in B["full_dataset"].items():
    for split, arms in splits.items():
        ck(split in ("test", "bench"), f"unexpected split {split}")
        for lab, cellv in arms.items():
            ck("n" in cellv, f"full_dataset.{model}.{split}.{lab}.n missing")
            for mk in MET:
                ck(mk in cellv, f"full_dataset.{model}.{split}.{lab}.{mk} missing")

print(f"checked bundle: {len(B['ontology']['nodes']):,} nodes, "
      f"{len(B['ontology']['edges']):,} edges, {len(QIDS)} queries, "
      f"{sum(len(v) for m in B['full_dataset'].values() for v in m.values())} benchmark cells")
if fails:
    print(f"\nFAILED {len(fails)} check(s):")
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1)
print("ALL DATA-CONTRACT CHECKS PASSED")
