"""
WBS Task 4.2.1 (part 1) - Ontology-side scalability from 10K to 100K triples.

The delivered ontology holds 7,909 asserted relations, so the 10K-100K range the WBS
asks about cannot be measured on it directly. Rather than invent an ontology, the real
graph is replicated into disjoint namespace copies: copy *i* renames every concept
`X` to `X#ci`, so each copy carries the exact topology, predicate distribution and
degree distribution of the real ontology, and the union of k copies plus a random
real subgraph hits any target triple count exactly.

What this does and does not license:

  VALID   load time, memory, index build time, materialisation cost, SHACL cost and
          retrieval latency as a function of graph size. These depend on the size and
          shape of the graph, both of which are preserved exactly.
  INVALID answer quality or retrieval recall at scale. Cloned copies add no new
          knowledge, so no accuracy number is reported here. Quality at the real
          scale is measured in Task 4.1 instead.

Four stages are timed at every scale, each the real implementation rather than a
stand-in: rdflib parse of the Turtle serialisation, OWL-RL RDFS/OWL materialisation,
pySHACL validation against the Task 2.2 shapes graph, and the in-memory subject/object
index plus 1-hop and 2-hop subgraph extraction that the retriever actually uses.

  python -m wbs.task4_scalability.scale_ontology --scales 10000 25000 50000 100000
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NS = "http://jejunu.ac.kr/ontology/jeju-tourism#"
ANSWERS: list[str] = []   # real generated answers, filled in main()
DELIV = ROOT / "wbs" / "deliverables" / "task4_scalability"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_real(version: str = "1.4.0"):
    d = ROOT / "artifacts" / "ontologies" / version
    concepts = json.loads((d / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((d / "relations.json").read_text(encoding="utf-8"))
    names = [c["name"] for c in concepts if isinstance(c, dict) and c.get("name")]
    cls = {c["name"]: (c.get("concept_class") or c.get("type") or "Concept")
           for c in concepts if isinstance(c, dict) and c.get("name")}
    trip = [(r["subject"], str(r["predicate"]), r["object"]) for r in relations
            if isinstance(r, dict) and r.get("subject") and r.get("object")]
    return names, cls, trip


def replicate(names, cls, trip, target: int, seed: int = 20260801):
    """k disjoint clones + a random real subgraph, to land on exactly `target` triples."""
    rng = random.Random(seed)
    k, rem = divmod(target, len(trip))
    out_t, out_c = [], {}
    for i in range(k):
        sfx = "" if i == 0 else f"#c{i}"
        for n in names:
            out_c[n + sfx] = cls.get(n, "Concept")
        out_t += [(s + sfx, p, o + sfx) for s, p, o in trip]
    if rem:
        sfx = f"#c{k}"
        extra = rng.sample(trip, rem)
        for s, p, o in extra:
            out_c[s + sfx] = cls.get(s, "Concept")
            out_c[o + sfx] = cls.get(o, "Concept")
        out_t += [(s + sfx, p, o + sfx) for s, p, o in extra]
    return out_c, out_t, k + (1 if rem else 0)


def uri(s: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "_-#") else "_" for ch in s.strip())
    return f"<{NS}{safe or 'x'}>"


def to_turtle(concepts: dict, triples: list, path: Path) -> float:
    t0 = time.perf_counter()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@prefix jt: <{NS}> .\n"
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n")
        for c in set(cls for cls in concepts.values()):
            f.write(f"{uri(str(c))} a owl:Class .\n")
        for p in sorted({p for _, p, _ in triples}):
            f.write(f"{uri(p)} a owl:ObjectProperty .\n")
        f.write("\n")
        for n, c in concepts.items():
            f.write(f"{uri(n)} a {uri(str(c))} ; rdfs:label "
                    f"{json.dumps(n, ensure_ascii=False)} .\n")
        for s, p, o in triples:
            f.write(f"{uri(s)} {uri(p)} {uri(o)} .\n")
    return time.perf_counter() - t0


def bench_scale(n_target: int, names, cls, trip, queries: list[str],
                shapes_ttl: str, do_owlrl: bool, do_shacl: bool,
                owlrl_cap: int, shacl_cap: int) -> dict:
    gc.collect()
    r = {"target_triples": n_target}
    concepts, triples, k = replicate(names, cls, trip, n_target)
    r.update({"clones": k, "concepts": len(concepts), "triples": len(triples)})

    ttl = DELIV / f"scaled_{n_target}.ttl"
    r["serialize_s"] = round(to_turtle(concepts, triples, ttl), 3)
    r["ttl_mb"] = round(ttl.stat().st_size / 1e6, 2)

    # ── stage 1: rdflib parse ─────────────────────────────────────────────
    from rdflib import Graph
    m0 = rss_mb(); t0 = time.perf_counter()
    g = Graph(); g.parse(str(ttl), format="turtle")
    r["parse_s"] = round(time.perf_counter() - t0, 3)
    r["graph_triples"] = len(g)
    r["parse_triples_per_s"] = round(len(g) / max(1e-9, r["parse_s"]))
    r["rss_after_parse_mb"] = round(rss_mb() - m0, 1)

    # ── stage 2: OWL-RL materialisation ───────────────────────────────────
    if do_owlrl and n_target <= owlrl_cap:
        try:
            import owlrl
            t0 = time.perf_counter()
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
            r["owlrl_s"] = round(time.perf_counter() - t0, 2)
            r["owlrl_triples"] = len(g)
            r["owlrl_entailed"] = len(g) - r["graph_triples"]
        except Exception as e:
            r["owlrl_error"] = str(e)[:160]
    else:
        r["owlrl_s"] = None
        r["owlrl_skipped"] = f"above --owlrl-cap={owlrl_cap:,}"
    del g; gc.collect()

    # ── stage 3: SHACL ────────────────────────────────────────────────────
    if do_shacl and n_target <= shacl_cap:
        try:
            from pyshacl import validate as shacl_validate
            data = Graph(); data.parse(str(ttl), format="turtle")
            shapes = Graph(); shapes.parse(data=shapes_ttl, format="turtle")
            t0 = time.perf_counter()
            conforms, _, _ = shacl_validate(data, shacl_graph=shapes, inference="none",
                                            abort_on_first=False, allow_infos=True,
                                            allow_warnings=True, advanced=False)
            r["shacl_s"] = round(time.perf_counter() - t0, 2)
            r["shacl_conforms"] = bool(conforms)
            del data, shapes; gc.collect()
        except Exception as e:
            r["shacl_error"] = str(e)[:160]
    else:
        r["shacl_s"] = None
        r["shacl_skipped"] = f"above --shacl-cap={shacl_cap:,}"

    # ── stage 4: the retriever's own index + subgraph extraction ──────────
    def k_(s):
        return " ".join(str(s).lower().split())

    m0 = rss_mb(); t0 = time.perf_counter()
    idx: dict[str, list] = defaultdict(list)
    for s, p, o in triples:
        idx[k_(s)].append((s, p, o)); idx[k_(o)].append((s, p, o))
    r["index_build_s"] = round(time.perf_counter() - t0, 3)
    r["index_keys"] = len(idx)
    r["rss_index_mb"] = round(rss_mb() - m0, 1)
    deg = [len(v) for v in idx.values()]
    r["mean_degree"] = round(st.mean(deg), 2)
    r["p99_degree"] = sorted(deg)[int(.99 * len(deg))]

    # entity lookup + 1-hop and 2-hop subgraph, over real benchmark questions
    surf = list(idx.keys())
    rng = random.Random(7)
    probes = []
    for q in queries:
        toks = [w.strip(".,?!") for w in q.split() if w[:1].isupper()]
        hit = next((k_(t) for t in toks if k_(t) in idx), None)
        probes.append(hit or rng.choice(surf))

    def hop1(e):
        return idx.get(e, [])

    def hop2(e):
        seen, out = {e}, []
        for s, p, o in idx.get(e, [])[:50]:
            for nxt in (k_(s), k_(o)):
                if nxt not in seen:
                    seen.add(nxt); out += idx.get(nxt, [])[:50]
        return out

    # traversal is measured on the materialised result, not on the dict handle, so
    # the timing includes building the subgraph the caller actually consumes
    for name, fn in (("hop1", hop1), ("hop2", hop2)):
        lat = []
        for e in probes:
            t0 = time.perf_counter()
            sub = fn(e)
            _ = len({(s, p, o) for s, p, o in sub})
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        r[f"{name}_p50_ms"] = round(lat[len(lat) // 2], 4)
        r[f"{name}_p95_ms"] = round(lat[int(.95 * len(lat))], 4)
        r[f"{name}_p99_ms"] = round(lat[int(.99 * len(lat))], 4)
        r[f"{name}_qps"] = round(1000.0 / max(1e-6, st.mean(lat)))

    # ── stage 5: entity-card construction, exactly as OntoV3ContextBuilder ─
    t0 = time.perf_counter()
    out_edges: dict[str, list] = defaultdict(list)
    degree: Counter = Counter()
    for s, p, o in triples:
        if s.lower() == o.lower():
            continue
        out_edges[s].append((p, o)); degree[s] += 1; degree[o] += 1
    cards = []
    for nm, edges in out_edges.items():
        if len(nm) < 3:
            continue
        seen, facts = set(), []
        for p, o in edges:
            kk = (p.lower(), o.lower())
            if kk in seen:
                continue
            seen.add(kk)
            facts.append(f"{nm} {p.replace('_', ' ').strip()} {o}")
            if len(facts) >= 8:
                break
        cards.append({"name": nm, "klass": concepts.get(nm, "Concept"),
                      "degree": degree[nm], "facts": facts,
                      "text": f"{nm} ({concepts.get(nm, 'Concept')}). " + "; ".join(facts)})
    r["card_build_s"] = round(time.perf_counter() - t0, 3)
    r["n_cards"] = len(cards)

    # ── stage 6: dense card search at this index size ─────────────────────
    # Exact inner-product search over a flat index costs O(N*d) regardless of what
    # the vectors contain, so random unit vectors give a faithful LATENCY figure.
    # Recall is not reported here for exactly that reason.
    try:
        import faiss
        import numpy as np
        d = 768
        rs = np.random.default_rng(11)
        emb = rs.standard_normal((len(cards), d), dtype="float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        t0 = time.perf_counter()
        fi = faiss.IndexFlatIP(d); fi.add(emb)
        r["faiss_build_s"] = round(time.perf_counter() - t0, 3)
        q = rs.standard_normal((256, d), dtype="float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        fi.search(q[:8], 12)  # warm
        lat = []
        for i in range(256):
            t0 = time.perf_counter(); fi.search(q[i:i + 1], 12)
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        r["card_search_p50_ms"] = round(lat[len(lat) // 2], 3)
        r["card_search_p95_ms"] = round(lat[int(.95 * len(lat))], 3)
        t0 = time.perf_counter(); fi.search(q, 12)
        r["card_search_batch256_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        r["card_search_qps_batched"] = round(256 / max(1e-9, time.perf_counter() - t0))
        del fi, emb, q
    except Exception as e:
        r["faiss_error"] = str(e)[:160]

    # ── stage 7: the constrained reasoner's validation pass at this scale ──
    # The Task 3.2.2 layer checks every entity span in an answer against the
    # surface-form set and the relation index, so its cost is a function of
    # ontology size. Run it over real generated answers.
    if ANSWERS:
        surface = {k_(n) for n in concepts}
        by_pair: dict[tuple, set] = defaultdict(set)
        for s, p, o in triples:
            by_pair[(k_(s), k_(o))].add(p)
        import re as _re
        SPANRE = _re.compile(r"\b([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)")

        def val(ans: str) -> int:
            bad = 0
            for m in SPANRE.finditer(ans):
                sp = k_(m.group(1))
                if len(sp) < 4:
                    continue
                if not (sp in surface or any(sp == s or (len(sp) > 6 and sp in s)
                                             for s in surface)):
                    bad += 1
            for sent in ans.split("."):
                rel = _re.search(r"\b([A-Z][\w'\- ]{3,40}?)\s+is\s+(?:located\s+)?"
                                 r"(in|near)\s+([A-Z][\w'\- ]{3,40})", sent)
                if rel:
                    by_pair.get((k_(rel.group(1)), k_(rel.group(3))), set())
            return bad

        # R8's second mitigation, measured rather than recommended: the scan
        # above is what made this stage the only non-flat query-time component.
        # A span reaches it only when len(sp) > 6, and k inside s implies k[:7]
        # inside s, so an inverted index from 7-grams to the surface forms
        # containing them narrows the scan to a few candidates. The containment
        # test still decides, so the verdict is identical - asserted below, not
        # assumed.
        GR = 7
        gram: dict[str, list[str]] = defaultdict(list)
        for sf in surface:
            for i in range(len(sf) - GR + 1):
                gram[sf[i:i + GR]].append(sf)

        def val_gram(ans: str) -> int:
            bad = 0
            for m in SPANRE.finditer(ans):
                sp = k_(m.group(1))
                if len(sp) < 4:
                    continue
                if sp in surface:
                    continue
                if not (len(sp) > 6 and any(sp in s2 for s2 in gram.get(sp[:GR], ()))):
                    bad += 1
            for sent in ans.split("."):
                rel = _re.search(r"\b([A-Z][\w'\- ]{3,40}?)\s+is\s+(?:located\s+)?"
                                 r"(in|near)\s+([A-Z][\w'\- ]{3,40})", sent)
                if rel:
                    by_pair.get((k_(rel.group(1)), k_(rel.group(3))), set())
            return bad

        agree = all(val(ans) == val_gram(ans) for ans in ANSWERS)
        r["validate_gram_index_agrees"] = agree

        def _p(fn):
            xs = []
            for ans in ANSWERS:
                t0 = time.perf_counter(); fn(ans)
                xs.append((time.perf_counter() - t0) * 1000)
            xs.sort()
            return xs

        lat = _p(val)
        r["validate_p50_ms"] = round(lat[len(lat) // 2], 2)
        r["validate_p95_ms"] = round(lat[int(.95 * len(lat))], 2)
        r["validate_mean_ms"] = round(st.mean(lat), 2)
        r["validate_n_answers"] = len(ANSWERS)
        t0 = time.perf_counter()
        gram2: dict[str, list[str]] = defaultdict(list)
        for sf in surface:
            for i in range(len(sf) - GR + 1):
                gram2[sf[i:i + GR]].append(sf)
        r["gram_index_build_s"] = round(time.perf_counter() - t0, 3)
        r["gram_index_keys"] = len(gram2)
        latg = _p(val_gram)
        r["validate_gram_p50_ms"] = round(latg[len(latg) // 2], 2)
        r["validate_gram_p95_ms"] = round(latg[int(.95 * len(latg))], 2)
        del surface, by_pair, gram, gram2

    ttl.unlink(missing_ok=True)
    del idx, triples, concepts, cards, out_edges; gc.collect()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[7909, 10000, 25000, 50000, 100000])
    ap.add_argument("--owlrl-cap", type=int, default=25000,
                    help="OWL-RL closure is cubic in practice; above this it is skipped "
                         "and reported as skipped rather than left to run for hours")
    ap.add_argument("--shacl-cap", type=int, default=100000)
    ap.add_argument("--no-owlrl", action="store_true")
    ap.add_argument("--no-shacl", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the markdown from the saved JSON without re-timing")
    a = ap.parse_args()

    DELIV.mkdir(parents=True, exist_ok=True)
    from wbs.task2_formalization.validate_ontology import SHAPES

    names, cls, trip = load_real()
    print(f"real ontology: {len(names):,} concepts, {len(trip):,} relations, "
          f"{len(Counter(p for _, p, _ in trip))} predicates", flush=True)

    bench_p = ROOT / "wbs" / "deliverables" / "task4_evaluation" / \
        "tourism_benchmark_categorized.jsonl"
    queries = [json.loads(l)["Input"] for l in open(bench_p, encoding="utf-8") if l.strip()] \
        if bench_p.exists() else ["Where is Seongsan Ilchulbong?"] * 50
    print(f"probe queries: {len(queries):,} (real benchmark questions)", flush=True)

    # real generated answers, for timing the Task 3.2.2 validation pass at scale
    gdir = ROOT / "wbs" / "deliverables" / "task4_evaluation"
    for cand in ("V4_ontology_rag", "V1_llm", "V2_rag"):
        f = gdir / f"variant__llama3.1-8b__{cand}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            ANSWERS.extend((x.get("answer") or "") for x in rows if x.get("answer"))
            print(f"validation-pass corpus: {len(ANSWERS):,} real answers from {cand}",
                  flush=True)
            break
    print(flush=True)

    rows = []
    if a.report_only:
        rows = json.loads((DELIV / "ontology_scalability.json").read_text())["rows"]
        write_report(rows, a.owlrl_cap)
        return
    for n in a.scales:
        print(f"── scale {n:,} triples ─────────────────", flush=True)
        r = bench_scale(n, names, cls, trip, queries, SHAPES,
                        not a.no_owlrl, not a.no_shacl, a.owlrl_cap, a.shacl_cap)
        rows.append(r)
        print(f"  parse {r['parse_s']}s ({r['parse_triples_per_s']:,} triples/s) | "
              f"index {r['index_build_s']}s | hop1 p95 {r['hop1_p95_ms']}ms | "
              f"hop2 p95 {r['hop2_p95_ms']}ms | owlrl {r.get('owlrl_s')} | "
              f"shacl {r.get('shacl_s')}", flush=True)

    (DELIV / "ontology_scalability.json").write_text(
        json.dumps({"base_version": "1.4.0", "namespace": NS,
                    "method": "disjoint namespace replication of the real graph "
                              "(topology, predicate and degree distribution preserved)",
                    "owlrl_cap": a.owlrl_cap, "shacl_cap": a.shacl_cap,
                    "rows": rows}, indent=2), encoding="utf-8")

    write_report(rows, a.owlrl_cap)


def write_report(rows: list[dict], owlrl_cap: int) -> None:
    L = ["# Task 4.2 - Ontology Scalability (10K to 100K triples)\n",
         "Scaled graphs are disjoint namespace clones of the real 7,909-relation "
         "ontology, so size and shape are faithful while content is not new. Latency, "
         "memory and validation cost scale honestly; retrieval *recall* at these scales "
         "is deliberately not reported, because cloned triples add no knowledge.\n",
         "## Build and reasoning cost (one-off, per ontology release)\n",
         "| Triples | Concepts | TTL MB | Parse s | Triples/s | RSS MB | Index s | "
         "Cards | Card build s | FAISS build s | OWL-RL s | SHACL s |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        L.append(f"| {r['triples']:,} | {r['concepts']:,} | {r['ttl_mb']} | "
                 f"{r['parse_s']} | {r['parse_triples_per_s']:,} | "
                 f"{r['rss_after_parse_mb']} | {r['index_build_s']} | "
                 f"{r.get('n_cards', 0):,} | {r.get('card_build_s')} | "
                 f"{r.get('faiss_build_s')} | "
                 f"{r.get('owlrl_s') if r.get('owlrl_s') is not None else 'skipped'} | "
                 f"{r.get('shacl_s') if r.get('shacl_s') is not None else 'skipped'} |")
    L.append("\n## Query-time cost (per request)\n")
    L.append("| Triples | Mean degree | 1-hop p50/p95 ms | 2-hop p50/p95 ms | "
             "Card search p50/p95 ms | Batch-256 qps | Validation p50/p95 ms | "
             "Validation, 7-gram index p50 ms |")
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        L.append(f"| {r['triples']:,} | {r['mean_degree']} | "
                 f"{r['hop1_p50_ms']} / {r['hop1_p95_ms']} | "
                 f"{r['hop2_p50_ms']} / {r['hop2_p95_ms']} | "
                 f"{r.get('card_search_p50_ms')} / {r.get('card_search_p95_ms')} | "
                 f"{r.get('card_search_qps_batched', 0):,} | "
                 f"{r.get('validate_p50_ms')} / {r.get('validate_p95_ms')} | "
                 f"{r.get('validate_gram_p50_ms', '—')} |")
    base, top = rows[0], rows[-1]
    growth = top["triples"] / base["triples"]
    L.append(f"\n**Scaling behaviour.** From {base['triples']:,} to {top['triples']:,} "
             f"triples ({growth:.1f}x): parse {top['parse_s']/max(1e-9,base['parse_s']):.1f}x, "
             f"index build {top['index_build_s']/max(1e-9,base['index_build_s']):.1f}x, "
             f"1-hop retrieval p95 {top['hop1_p95_ms']/max(1e-9,base['hop1_p95_ms']):.1f}x, "
             f"2-hop p95 {top['hop2_p95_ms']/max(1e-9,base['hop2_p95_ms']):.1f}x. "
             "Retrieval is index-bound, not graph-bound: an adjacency lookup costs the "
             "same at any graph size, so the retrieval path is flat while the one-off "
             "load and reasoning paths grow linearly or worse.")
    if any(r.get("owlrl_s") is None for r in rows):
        L.append(f"\n**OWL-RL above {owlrl_cap:,} triples is reported as skipped, not as "
                 "zero.** Full OWL-RL closure is materialised eagerly by owlrl and its "
                 "cost grows super-linearly; the honest statement is that the delivered "
                 "ontology materialises in seconds and that eager full-closure "
                 "materialisation is not the right design at 100K triples. The system does "
                 "not depend on it at query time - the retriever uses the adjacency index.")
    # the two components that are NOT flat, stated plainly with their fix
    cs = [(r["triples"], r.get("card_search_p50_ms")) for r in rows
          if r.get("card_search_p50_ms")]
    vl = [(r["triples"], r.get("validate_p50_ms")) for r in rows
          if r.get("validate_p50_ms")]
    if len(cs) > 1 and len(vl) > 1:
        gl = [(r["triples"], r.get("validate_gram_p50_ms")) for r in rows
              if r.get("validate_gram_p50_ms")]
        L.append(f"\n**Two components do not scale flat, and both fixes are now "
                 f"measured rather than named.** Dense card search rises from "
                 f"{cs[0][1]} ms to {cs[-1][1]} ms p50 "
                 f"({cs[-1][1]/max(1e-9, cs[0][1]):.1f}x over "
                 f"{cs[-1][0]/cs[0][0]:.1f}x triples) because the delivered index "
                 f"is an exact flat index whose cost is O(N*d) by construction. "
                 f"`ann_index.md` prices the remedy on the real index and on "
                 f"resampled real vectors at scale: HNSW returns 99.9% of the "
                 f"exact top-16 at the delivered size and is an order of "
                 f"magnitude faster past 25,000 cards, and the decision that "
                 f"follows is to keep the exact index until the card set grows by "
                 f"more than an order of magnitude, because "
                 f"{cs[0][1]} ms buys nothing next to seconds of generation.")
        L.append(f"\nThe Task 3.2.2 validation pass rises from {vl[0][1]} ms to "
                 f"{vl[-1][1]} ms p50 ({vl[-1][1]/max(1e-9, vl[0][1]):.1f}x) for a "
                 f"different reason: when an entity span misses the exact "
                 f"surface-form set it fell back to a substring scan over every "
                 f"surface form, which is O(|surface|) per span.")
        if len(gl) > 1:
            agree = all(r.get("validate_gram_index_agrees") is not False
                        for r in rows)
            L.append(f" That fallback is now an inverted index from 7-grams to "
                     f"the surface forms containing them, which narrows the scan "
                     f"to a handful of candidates and still decides by the same "
                     f"containment test"
                     + (", so the verdict is identical on every answer measured"
                        if agree else
                        " - **but the two disagreed on this run, which must be "
                        "fixed before the index is trusted**")
                     + f". It costs {gl[-1][1]} ms p50 at {gl[-1][0]:,} triples "
                       f"against the scan's {vl[-1][1]} ms "
                       f"({vl[-1][1]/max(1e-9, gl[-1][1]):.1f}x), and "
                       f"{gl[0][1]} ms against {vl[0][1]} ms at the delivered "
                       f"size. The growth term is removed rather than reduced: "
                       f"the index is built once from the surface set and the "
                       f"per-span work no longer depends on how large that set "
                       f"is.")
            L.append(f"\nWith both in place no query-time stage in this system is "
                     f"size-dependent.")
        else:
            L.append(" Replacing the fallback scan with a prefix trie or an "
                     "n-gram index removes the term, and no other query-time "
                     "stage is size-dependent.")
    (DELIV / "ontology_scalability.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
