"""WBS Task 4.2 / R8 - the approximate index the scalability report named.

`ontology_scalability.md` found dense card search to be the one query-time stage
whose cost is not flat: p50 0.119 ms at 7,909 triples and 2.4 ms at 100,000,
because the delivered index is `IndexFlatIP`, whose search is O(N*d) by
construction. The report named IVF and HNSW as the remedy and neither was
implemented, which left R8 open with a recommendation nobody had priced.

Pricing it needs two numbers, and they have to be measured on different data:

  * **Recall** has to come from the real index and real questions. An
    approximate index trades recall for speed, and how much it trades depends
    entirely on how the vectors are distributed. The 347 benchmark questions are
    encoded with the same encoder and prefix the deployed retriever uses, and
    the top-16 (`ONTO_V3_ENTITIES`) neighbours are compared with what the flat
    index returns for the same query. That is the recall the system would lose.

  * **Latency at scale** cannot come from the real index, because the real index
    has 1,585 cards. Rather than random unit vectors - which have no cluster
    structure and are close to the worst case for a graph index - the scaled
    corpora are built by resampling the REAL card vectors and perturbing them,
    so the neighbourhood structure the index exploits is the structure the data
    actually has. Flat search is measured on the same corpus, so the comparison
    is like for like whatever the perturbation does.

Nothing here changes the deployed retriever. It stays flat and exact; this
measures what switching would cost and buys R8 a decision instead of a
recommendation.

  python -m wbs.task4_scalability.ann_index
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task4_scalability"
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"


def _lat(fn, q, n=256):
    """p50/p95 ms for single-query search."""
    fn(q[:8])
    xs = []
    for i in range(min(n, len(q))):
        t0 = time.perf_counter()
        fn(q[i:i + 1])
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return round(xs[len(xs) // 2], 3), round(xs[int(.95 * len(xs))], 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    # the real index size is always the first row; these are the ones above it
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[10000, 25000, 50000, 100000])
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--nprobe", type=int, default=16)
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    import faiss
    import numpy as np
    from experiments.exp024_rag_vs_ontology import config as C

    K = C.ONTO_V3_ENTITIES
    flat_p = C.INDEX_DIR / f"ontocards_{C.ONTO_VERSION}_bge.faiss"
    if not flat_p.exists():
        cand = sorted(C.INDEX_DIR.glob("ontocards_*_bge.faiss"))
        if not cand:
            sys.exit("no card index on disk; run the retriever build first")
        flat_p = cand[-1]
    flat = faiss.read_index(str(flat_p))
    emb = flat.reconstruct_n(0, flat.ntotal).astype("float32")
    d = emb.shape[1]
    print(f"real card index: {flat.ntotal:,} cards, d={d}  ({flat_p.name})",
          flush=True)

    # ── recall on the real index with the real questions ──────────────────
    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
             if l.strip()]
    qs = [b["Input"] for b in bench]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    qe = st.encode([C.EMBED_QUERY_PREFIX + q for q in qs], batch_size=32,
                   convert_to_numpy=True, normalize_embeddings=True,
                   show_progress_bar=False).astype("float32")
    print(f"encoded {len(qs)} benchmark questions on CPU", flush=True)

    _, ref = flat.search(qe, K)

    def build(kind: str, vecs):
        if kind == "flat":
            ix = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            ix = faiss.IndexHNSWFlat(d, a.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            ix.hnsw.efConstruction = 200
            ix.hnsw.efSearch = a.ef_search
        else:
            nlist = max(4, int(4 * len(vecs) ** 0.5))
            ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist,
                                    faiss.METRIC_INNER_PRODUCT)
            ix.train(vecs)
            ix.nprobe = a.nprobe
        t0 = time.perf_counter()
        ix.add(vecs)
        return ix, round(time.perf_counter() - t0, 3)

    recall = {}
    for kind in ("hnsw", "ivf"):
        ix, bs = build(kind, emb)
        _, got = ix.search(qe, K)
        hits = sum(len(set(r) & set(g)) for r, g in zip(ref, got))
        recall[kind] = {"recall_at_k": round(hits / (len(qs) * K), 4),
                        "k": K, "build_s": bs, "n": int(flat.ntotal),
                        "queries": len(qs)}
        print(f"  {kind:<5} recall@{K} = {recall[kind]['recall_at_k']:.4f} "
              f"on the real index, build {bs}s", flush=True)

    # ── latency at scale, on resampled real vectors ───────────────────────
    rs = np.random.default_rng(7)
    rows = []
    for n in [int(flat.ntotal)] + [s for s in a.sizes if s > flat.ntotal]:
        if n <= len(emb):
            vecs = emb[:n].copy()
            note = "real vectors"
        else:
            pick = rs.integers(0, len(emb), size=n - len(emb))
            noise = rs.standard_normal((n - len(emb), d)).astype("float32") * 0.05
            vecs = np.vstack([emb, emb[pick] + noise])
            note = f"real + {n - len(emb):,} resampled (sigma 0.05)"
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        r = {"n": n, "corpus": note}
        base_ix, _ = build("flat", vecs)
        _, base = base_ix.search(qe, K)
        for kind in ("flat", "hnsw", "ivf"):
            ix, bs = build(kind, vecs)
            p50, p95 = _lat(lambda q, ix=ix: ix.search(q, K), qe)
            r[f"{kind}_p50_ms"], r[f"{kind}_p95_ms"] = p50, p95
            r[f"{kind}_build_s"] = bs
            if kind != "flat":
                _, got = ix.search(qe, K)
                hits = sum(len(set(x) & set(y)) for x, y in zip(base, got))
                r[f"{kind}_recall"] = round(hits / (len(qs) * K), 4)
            del ix
        rows.append(r)
        print(f"  n={n:>7,}  flat {r['flat_p50_ms']:7.3f} ms | hnsw "
              f"{r['hnsw_p50_ms']:7.3f} ms (r={r.get('hnsw_recall')}) | ivf "
              f"{r['ivf_p50_ms']:7.3f} ms (r={r.get('ivf_recall')})", flush=True)
        del base_ix, vecs

    # ── the deliverable ───────────────────────────────────────────────────
    L = ["# R8: what an approximate index would actually cost\n"]
    L.append(f"`ontology_scalability.md` named IVF and HNSW as the remedy for the "
             f"one query-time stage that is not flat, and priced neither. "
             f"Recall is measured on the real {flat.ntotal:,}-card index with the "
             f"{len(qs)} benchmark questions encoded by the deployed encoder and "
             f"prefix; latency is measured at larger sizes on corpora built by "
             f"resampling and perturbing those same vectors, because an "
             f"approximate index's speed depends on the neighbourhood structure "
             f"of the data and random vectors do not have any.\n")

    L.append(f"## Recall at the delivered size (top-{K}, against the exact index)\n")
    L.append("| Index | Recall@%d | Build |" % K)
    L.append("|---|--:|--:|")
    for kind, v in recall.items():
        L.append(f"| {kind.upper()} | {v['recall_at_k']*100:.2f}% | "
                 f"{v['build_s']}s |")
    L.append("")
    best = max(recall.items(), key=lambda kv: kv[1]["recall_at_k"])
    L.append(f"At this size the question answers itself: {best[0].upper()} "
             f"retrieves {best[1]['recall_at_k']*100:.2f}% of what the exact "
             f"index retrieves"
             + (", so the approximation is free in recall terms"
                if best[1]["recall_at_k"] > 0.99 else
                f", so {(1-best[1]['recall_at_k'])*100:.2f}% of the top-{K} "
                f"would be different cards")
             + ". 1,585 cards is far below the size at which either index is "
               "worth its complexity, which is the real finding for the "
               "delivered system.\n")

    L.append("## Latency as the card set grows\n")
    L.append(f"| Cards | Corpus | Flat p50 | HNSW p50 | HNSW recall | IVF p50 | "
             f"IVF recall |")
    L.append("|--:|---|--:|--:|--:|--:|--:|")
    for r in rows:
        L.append(f"| {r['n']:,} | {r['corpus']} | {r['flat_p50_ms']} ms | "
                 f"{r['hnsw_p50_ms']} ms | "
                 f"{(r.get('hnsw_recall') or 0)*100:.1f}% | "
                 f"{r['ivf_p50_ms']} ms | "
                 f"{(r.get('ivf_recall') or 0)*100:.1f}% |")
    L.append("")

    big = rows[-1]
    sp_h = big["flat_p50_ms"] / max(big["hnsw_p50_ms"], 1e-9)
    sp_i = big["flat_p50_ms"] / max(big["ivf_p50_ms"], 1e-9)
    L.append(f"**The remedy works, and at this project's scale it is not needed.** "
             f"At {big['n']:,} cards HNSW is {sp_h:.1f}x faster than exact search "
             f"and IVF {sp_i:.1f}x, for "
             f"{(1-(big.get('hnsw_recall') or 0))*100:.1f}% and "
             f"{(1-(big.get('ivf_recall') or 0))*100:.1f}% of the exact top-{K} "
             f"respectively. At the delivered {flat.ntotal:,} cards exact search "
             f"costs {rows[0]['flat_p50_ms']} ms, which is not a cost worth any "
             f"recall at all next to seconds of generation. R8's mitigation is "
             f"therefore implemented and measured rather than recommended, and "
             f"the decision it supports is to keep the exact index until the card "
             f"set grows by more than an order of magnitude.\n")
    L.append("The deployed retriever is unchanged: it still builds "
             "`IndexFlatIP`. This file is a measurement, not a migration.\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ann_index.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "ann_index.json").write_text(json.dumps(
        {"k": K, "index_file": flat_p.name, "n_real": int(flat.ntotal),
         "hnsw_m": a.hnsw_m, "ef_search": a.ef_search, "nprobe": a.nprobe,
         "recall_real_index": recall, "scaling": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'ann_index.md'}")


if __name__ == "__main__":
    main()
