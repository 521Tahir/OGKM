"""
exp041 - assemble the demo data bundle from REAL project artefacts.

Nothing here is authored by hand. Sources:
  ontology v1.3.0   artifacts/ontologies/1.3.0/{concepts,relations,hierarchy}.json
  gazetteer         data/jeju_full/jeju_gazetteer.json
  traces            experiments/exp041_demo_traces/traces.json   (real inference)
  query scores      experiments/exp041_demo_traces/scores.json   (auditable rubric)
  full-split results experiments/exp024_rag_vs_ontology/outputs/scores{,_ded}/

The bundle is embedded directly in the single-page app so it opens from
file:// with no server and no CORS problem.
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C

OD = ROOT / "artifacts" / "ontologies" / C.ONTO_CLEAN_VERSION
OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
MET = ["bertf1", "meteor", "comet", "keytok", "judge"]


def cell(d: str, split: str, model: str, arm: str):
    p = OUT / d / f"{split}__{model}__{arm}.jsonl"
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
    for m in MET:
        v = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        o[m] = round(st.mean(v), 4) if v else None
    return o


def main() -> None:
    concepts = json.loads((OD / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((OD / "relations.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((OD / "hierarchy.json").read_text(encoding="utf-8"))
    gaz = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                     .read_text(encoding="utf-8"))

    nodes, byname = [], {}
    for c in concepts:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        nid = c["name"]
        byname[nid] = {
            "id": nid, "label": nid,
            "klass": c.get("concept_class", "Concept"),
            "aliases": (c.get("aliases") or [])[:4],
            "attrs": c.get("attributes") or {},
            "provenance": {
                "source_record_id": (c.get("provenance") or {}).get("source_record_id"),
                "confidence": (c.get("provenance") or {}).get("confidence"),
                "validator": (c.get("provenance") or {}).get("validator_status"),
            },
        }
    edges, pred_count, deg = [], collections.Counter(), collections.Counter()
    for r in relations:
        s, p, o = r.get("subject"), r.get("predicate"), r.get("object")
        if not (s and p and o):
            continue
        edges.append({"s": s, "p": p, "o": o,
                      "conf": round(float((r.get("provenance") or {}).get("confidence", 0)), 2)})
        pred_count[p] += 1
        deg[s] += 1
        deg[o] += 1
        for n in (s, o):
            if n not in byname:
                byname[n] = {"id": n, "label": n, "klass": "ObjectValue",
                             "aliases": [], "attrs": {}, "provenance": {}}
    for n in byname.values():
        n["degree"] = deg.get(n["id"], 0)
    nodes = list(byname.values())

    klass_count = collections.Counter(n["klass"] for n in nodes)
    # hierarchy.json records one entry per concept carrying its full
    # hierarchy_path, e.g. Hallasan -> [TourismEntity, Attraction,
    # NaturalAttraction, Mountain]. The class tree is the set of consecutive
    # pairs along every path; concept->leaf-class gives instance membership.
    class_edges, inst_edges, roots = set(), [], set()
    for h in hierarchy:
        if not isinstance(h, dict):
            continue
        path = h.get("hierarchy_path") or []
        if path:
            roots.add(path[0])
            for a, b in zip(path, path[1:]):
                class_edges.add((a, b))
        leaf = h.get("concept_class") or (path[-1] if path else None)
        if h.get("concept") and leaf:
            inst_edges.append({"concept": h["concept"], "klass": leaf,
                               "is_instance": bool(h.get("is_instance")),
                               "path": path,
                               "confidence": h.get("confidence")})
    hier_edges = [{"parent": a, "child": b} for a, b in sorted(class_edges)]

    traces = json.loads((ROOT / "experiments" / "exp041_demo_traces" / "traces.json")
                        .read_text(encoding="utf-8"))
    scores = json.loads((ROOT / "experiments" / "exp041_demo_traces" / "scores.json")
                        .read_text(encoding="utf-8"))

    full = {}
    for model in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b"):
        full[model] = {}
        for lab, d, arm in (("Simple RAG", "scores", "rag"),
                            ("Simple RAG (dedicated)", "scores_ded", "rag"),
                            ("Ontology V3", "scores", "onto_v3"),
                            ("Ontology V3 (dedicated)", "scores_ded", "onto_v3"),
                            ("Ontology V6 (dedicated)", "scores_ded", "onto_v6"),
                            ("Onto-RAG hybrid 20%", "scores", "hybrid20"),
                            ("Vanilla (no fine-tune)", "scores", "base")):
            for split in ("test", "bench"):
                r = cell(d, split, model, arm)
                if r:
                    full[model].setdefault(split, {})[lab] = r

    # the goal names 1B and 3B specifically, so the demo carries all three models'
    # three-query runs, not just the 8B one used for the walk-through
    multi = {}
    for key, tf, sf in (("llama3.2-1b", "traces_llama3.2-1b.json", "scores_llama3.2-1b.json"),
                        ("llama3.2-3b", "traces_llama3.2-3b.json", "scores_llama3.2-3b.json"),
                        ("llama3.1-8b", "traces.json", "scores.json")):
        tp = ROOT / "experiments" / "exp041_demo_traces" / tf
        sp = ROOT / "experiments" / "exp041_demo_traces" / sf
        if not (tp.exists() and sp.exists()):
            continue
        tt = json.loads(tp.read_text(encoding="utf-8"))
        ss = json.loads(sp.read_text(encoding="utf-8"))
        multi[key] = {
            "model": tt["meta"]["model"],
            "scores": ss["per_query"],
            "answers": {q: {a: {"answer": v["answer"], "answer_tokens": v["answer_tokens"],
                                "prompt_tokens": v["prompt_tokens"],
                                "generation_s": v["generation_s"], "adapter": v["adapter"]}
                            for a, v in arms.items()}
                        for q, arms in tt["answers"].items()},
        }
    print("multi-model three-query runs: " + ", ".join(multi))

    bundle = {
        "generated": traces["meta"]["captured"],
        "multi_model": multi,
        "meta": traces["meta"],
        "ontology": {
            "version": C.ONTO_CLEAN_VERSION,
            "source_files": [str(OD / f) for f in
                             ("concepts.json", "relations.json", "hierarchy.json")],
            "stats": {
                "typed_concepts": sum(1 for c in concepts if isinstance(c, dict)),
                "graph_nodes": len(nodes),
                "relations": len(edges),
                "distinct_predicates": len(pred_count),
                "concept_classes": len(klass_count),
                "hierarchy_edges": len(hier_edges),
                "class_tree_roots": len(roots),
                "typed_instances": len(inst_edges),
                "gazetteer_places": len([x for x in gaz["locations"] if isinstance(x, dict)]),
            },
            "class_counts": klass_count.most_common(),
            "predicate_counts": pred_count.most_common(),
            "nodes": nodes, "edges": edges, "hierarchy": hier_edges,
            "class_roots": sorted(roots), "instances": inst_edges,
        },
        "gazetteer": {x["name"]: x for x in gaz["locations"] if isinstance(x, dict)},
        "distances_km": traces["gazetteer_distances_km"],
        "queries": traces["queries"],
        "rag": traces["rag"], "onto": traces["onto"], "answers": traces["answers"],
        "scores": scores,
        "full_dataset": full,
    }

    p = ROOT / "Jeju_Tourism_Research_Demo" / "demo" / "recorded_traces" / "bundle.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
                 encoding="utf-8")
    s = bundle["ontology"]["stats"]
    print(f"ontology: {s['typed_concepts']:,} typed concepts, {s['graph_nodes']:,} graph nodes, "
          f"{s['relations']:,} relations, {s['distinct_predicates']} predicates, "
          f"{s['concept_classes']} classes, {s['hierarchy_edges']:,} hierarchy edges")
    print(f"full-dataset cells: " + ", ".join(
        f"{m}:{sum(len(v) for v in full[m].values())}" for m in full))
    print(f"wrote {p}  ({p.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
