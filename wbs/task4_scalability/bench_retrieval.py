"""
WBS Task 4.2.1 (part 2) - Retrieval scalability 10K-100K passages, end-to-end
latency breakdown and throughput.

Three measurements, all on the real system:

  SCALABILITY   the delivered dense index holds 36,847 real train-split passages.
                Latency is measured at 10K, 25K and 36,847 by taking real
                sub-indices, and at 50K and 100K by tiling the real vector matrix.
                For an exact flat inner-product index the search cost is O(N*d) and
                independent of vector content, so tiling measures latency faithfully;
                recall is reported only at the real sizes, where the passages are
                genuinely distinct, and is left blank for the tiled sizes.

  LATENCY       the per-stage breakdown of one request: query encoding, dense search,
                ontology card search, context assembly, generation, validation. The
                generation and validation columns are taken from the Task 4.1 run
                itself (347 real requests per variant), not re-estimated.

  THROUGHPUT    requests/second for the retrieval path at batch 1, 8, 32 and 128, and
                the end-to-end requests/minute actually achieved by each variant in
                Task 4.1, including model load.

  python -m wbs.task4_scalability.bench_retrieval --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DELIV = ROOT / "wbs" / "deliverables" / "task4_scalability"
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only",
            "V4_ontology_rag", "V5_ontology_rag_validated"]


def pct(lat: list[float], q: float) -> float:
    s = sorted(lat)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="device for query encoding")
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[10000, 25000, 36847, 50000, 100000])
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the markdown from the saved JSON")
    a = ap.parse_args()

    import faiss
    import numpy as np
    from experiments.exp024_rag_vs_ontology import config as C

    DELIV.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        saved = json.loads((DELIV / "retrieval_scalability.json").read_text())
        write_report(saved, C, a)
        return
    out: dict = {"device_for_encoding": a.device}

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
             if l.strip()]
    queries = [b["Input"] for b in bench][: a.n_queries]
    print(f"{len(queries)} real benchmark queries", flush=True)

    # ── query encoding latency (the first stage of every request) ─────────
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(C.EMBED_MODEL, device=a.device)
    enc.max_seq_length = 512
    enc.encode(queries[:4], convert_to_numpy=True, normalize_embeddings=True)
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        lat.append((time.perf_counter() - t0) * 1000)
    out["encode"] = {"device": a.device, "p50_ms": round(pct(lat, .5), 2),
                     "p95_ms": round(pct(lat, .95), 2),
                     "mean_ms": round(st.mean(lat), 2)}
    print(f"query encoding on {a.device}: p50 {out['encode']['p50_ms']} ms", flush=True)
    for bs in (1, 8, 32, 128):
        reps = max(1, len(queries) // bs)
        t0 = time.perf_counter()
        for i in range(reps):
            enc.encode(queries[(i * bs) % len(queries):][:bs] or queries[:bs],
                       convert_to_numpy=True, normalize_embeddings=True,
                       batch_size=bs)
        el = time.perf_counter() - t0
        out["encode"][f"qps_batch{bs}"] = round(reps * bs / el, 1)
    qemb = enc.encode(queries, convert_to_numpy=True,
                      normalize_embeddings=True).astype("float32")
    del enc

    # ── dense passage index at scale ──────────────────────────────────────
    idx_p = C.INDEX_DIR / "train_bge.faiss"
    base = faiss.read_index(str(idx_p))
    n_real, d = base.ntotal, base.d
    print(f"real index: {n_real:,} passages x {d}d", flush=True)
    V = base.reconstruct_n(0, n_real)
    gold_at = {}
    rows = []
    for n in a.scales:
        if n <= n_real:
            M = V[:n]
            synthetic = False
        else:
            reps = -(-n // n_real)
            M = np.tile(V, (reps, 1))[:n]
            synthetic = True
        fi = faiss.IndexFlatIP(d); fi.add(np.ascontiguousarray(M))
        fi.search(qemb[:4], 5)
        lat = []
        for i in range(len(qemb)):
            t0 = time.perf_counter(); fi.search(qemb[i:i + 1], C.RAG_TOP_K)
            lat.append((time.perf_counter() - t0) * 1000)
        r = {"passages": n, "synthetic_tiling": synthetic,
             "search_p50_ms": round(pct(lat, .5), 3),
             "search_p95_ms": round(pct(lat, .95), 3),
             "search_mean_ms": round(st.mean(lat), 3),
             "index_mb": round(M.nbytes / 1e6, 1)}
        for bs in (8, 32, 128):
            t0 = time.perf_counter(); fi.search(qemb[:bs], C.RAG_TOP_K)
            el = time.perf_counter() - t0
            r[f"qps_batch{bs}"] = round(bs / el, 1)
        if not synthetic:
            _, I = fi.search(qemb, C.RAG_TOP_K)
            gold_at[n] = I
        rows.append(r)
        print(f"  {n:>7,} passages  p50 {r['search_p50_ms']} ms  "
              f"p95 {r['search_p95_ms']} ms  batch128 {r['qps_batch128']:,} qps"
              f"{'  (tiled)' if synthetic else ''}", flush=True)
        del fi, M

    # retrieval stability as the real index grows: does adding real passages change
    # the top-5 a query gets? Measured only between genuinely distinct index sizes.
    if len(gold_at) > 1:
        ks = sorted(gold_at)
        full = gold_at[ks[-1]]
        for k in ks[:-1]:
            ov = [len(set(gold_at[k][i]) & set(full[i])) / full.shape[1]
                  for i in range(full.shape[0])]
            for r in rows:
                if r["passages"] == k:
                    r["top5_overlap_vs_full_index"] = round(st.mean(ov), 3)
    out["passage_search"] = rows
    out["real_index_size"] = n_real
    out["n_queries"] = len(queries)
    del V, base

    # ── ontology card index ───────────────────────────────────────────────
    card_idx = C.INDEX_DIR / f"ontocards_{C.ONTO_VERSION}_bge.faiss"
    if card_idx.exists():
        ci = faiss.read_index(str(card_idx))
        ci.search(qemb[:4], 5)
        lat = []
        for i in range(len(qemb)):
            t0 = time.perf_counter(); ci.search(qemb[i:i + 1], C.ONTO_V3_ENTITIES)
            lat.append((time.perf_counter() - t0) * 1000)
        out["card_search"] = {"cards": ci.ntotal,
                              "p50_ms": round(pct(lat, .5), 3),
                              "p95_ms": round(pct(lat, .95), 3)}
        print(f"ontology card search ({ci.ntotal:,} cards): "
              f"p50 {out['card_search']['p50_ms']} ms", flush=True)
        del ci

    # ── end-to-end, from the Task 4.1 run itself ──────────────────────────
    e2e = {}
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        g = [r["generation_s"] for r in rs if r.get("generation_s") is not None]
        vm = [r["validation_ms"] for r in rs if r.get("validation_ms") is not None]
        e2e[v] = {"n": len(rs),
                  "gen_p50_s": round(pct(g, .5), 2), "gen_p95_s": round(pct(g, .95), 2),
                  "gen_mean_s": round(st.mean(g), 2),
                  "prompt_tokens_mean": round(st.mean(
                      [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")])),
                  "validation_mean_ms": round(st.mean(vm), 1) if vm else None,
                  "requests_per_min": round(60.0 / st.mean(g), 2)}
    mani = EVAL / f"variant_run_manifest__{a.model}.json"
    if mani.exists():
        m = json.loads(mani.read_text())
        for v, d_ in (m.get("variants") or {}).items():
            if v in e2e:
                e2e[v]["wall_min_incl_load"] = d_.get("wall_min")
                e2e[v]["achieved_items_per_min"] = round(
                    d_["n"] / d_["wall_min"], 2) if d_.get("wall_min") else None
    out["end_to_end"] = e2e

    (DELIV / "retrieval_scalability.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    write_report(out, C, a)


def write_report(out: dict, C, a) -> None:
    rows = out["passage_search"]
    n_real = out["real_index_size"]
    e2e = out.get("end_to_end", {})
    queries = range(out.get("n_queries", a.n_queries))
    L = ["# Task 4.2 - Retrieval Scalability, Latency and Throughput\n",
         f"Query encoding measured on `{out['device_for_encoding']}` with {C.EMBED_MODEL}; "
         f"{len(queries)} real benchmark queries; dense index is the delivered "
         f"{n_real:,}-passage train-split index.\n",
         "## Dense passage search at scale\n",
         "| Passages | Index MB | p50 ms | p95 ms | qps @batch8 | qps @batch32 | "
         "qps @batch128 | Top-5 overlap vs full index | Real or tiled |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in rows:
        L.append(f"| {r['passages']:,} | {r['index_mb']} | {r['search_p50_ms']} | "
                 f"{r['search_p95_ms']} | {r['qps_batch8']:,} | {r['qps_batch32']:,} | "
                 f"{r['qps_batch128']:,} | "
                 f"{r.get('top5_overlap_vs_full_index', '—')} | "
                 f"{'tiled (latency only)' if r['synthetic_tiling'] else 'real passages'} |")
    ov = [(r["passages"], r.get("top5_overlap_vs_full_index")) for r in rows
          if r.get("top5_overlap_vs_full_index") is not None]
    if ov:
        lo = min(ov, key=lambda x: x[1])
        L.append(f"\n**Corpus size changes what is retrieved, not just how fast.** At "
                 f"{lo[0]:,} passages only {lo[1]*100:.0f}% of a query's top-5 survives "
                 f"into the top-5 of the full {n_real:,}-passage index. A smaller "
                 f"corpus is therefore not a cheap approximation of a larger one, and "
                 f"any result measured on a subsampled index is measuring a different "
                 f"retriever.")
    L.append("\nTiled rows repeat the real vectors to reach the requested size. Exact "
             "flat inner-product search costs O(N·d) irrespective of vector content, so "
             "the latency is faithful; no recall figure is given for those rows because "
             "duplicated passages would make recall meaningless.\n")

    L.append("## Per-request latency breakdown\n")
    L.append(f"Query encoding ({out['device_for_encoding']}): p50 "
             f"**{out['encode']['p50_ms']} ms**, "
             f"p95 {out['encode']['p95_ms']} ms. ")
    if "card_search" in out:
        L.append(f"Ontology card search over {out['card_search']['cards']:,} cards: "
                 f"p50 **{out['card_search']['p50_ms']} ms**. ")
    L.append(f"Dense passage search at the delivered {n_real:,} passages: p50 "
             f"**{[r for r in rows if r['passages'] == n_real][0]['search_p50_ms']} ms**.\n")
    if e2e:
        L.append("| Variant | n | Prompt tokens | Generation p50 s | p95 s | mean s | "
                 "Validation+repair ms | Requests/min (1 GPU, batch 1) | Wall min incl. load |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for v in VARIANTS:
            if v not in e2e:
                continue
            d_ = e2e[v]
            L.append(f"| {v} | {d_['n']} | {d_['prompt_tokens_mean']:,} | "
                     f"{d_['gen_p50_s']} | {d_['gen_p95_s']} | {d_['gen_mean_s']} | "
                     f"{d_['validation_mean_ms'] if d_['validation_mean_ms'] else '—'} | "
                     f"{d_['requests_per_min']} | "
                     f"{d_.get('wall_min_incl_load', '—')} |")
        rr0 = [r for r in rows if r["passages"] == n_real][0]
        retr_ms = (out["encode"]["p50_ms"] + rr0["search_p50_ms"]
                   + (out.get("card_search", {}).get("p50_ms") or 0))
        gen_v = next((v for v in ("V4_ontology_rag", "V2_rag", "V1_llm") if v in e2e),
                     None)
        if gen_v:
            gen_ms = e2e[gen_v]["gen_mean_s"] * 1000
            L.append(f"\n**Generation dominates.** The whole retrieval path - query "
                     f"encoding plus the ontology card search plus the dense passage "
                     f"search - costs about **{retr_ms:.1f} ms**, against "
                     f"{gen_ms/1000:.1f} s of generation for {gen_v}: "
                     f"**{retr_ms/gen_ms*100:.2f}%** of the request. Whichever arm is "
                     f"chosen, retrieval is not what the user waits for.\n")
        v5 = e2e.get("V5_ontology_rag_validated")
        v4 = e2e.get("V4_ontology_rag")
        if v5 and v5.get("validation_mean_ms"):
            L.append(f"**The validation layer is the one real overhead, and the "
                     f"'{v5['validation_mean_ms']:,.0f} ms' in the table is not a "
                     f"cheap check.** That figure is the whole Task 3.2.2 pass, and "
                     f"when a violation fires the pass includes a full re-generation. "
                     f"The check itself - resolving every entity span against the "
                     f"ontology - costs 1-2 ms at the delivered graph size (measured "
                     f"separately in `ontology_scalability.md`); the rest is the repair "
                     f"model call." +
                     (f" End to end that is {v5['gen_mean_s']/v4['gen_mean_s']:.1f}x "
                      f"V4's generation time per item."
                      if v4 and v4.get("gen_mean_s") else "") +
                     " Validation is cheap; *repair* costs a second generation, and the "
                     "ablation in `ablation_validation__*.json` is what says whether "
                     "that second pass is worth paying for.\n")

    L.append("## Retrieval throughput\n")
    L.append("| Stage | batch 1 | batch 8 | batch 32 | batch 128 |")
    L.append("|---|--:|--:|--:|--:|")
    L.append(f"| Query encoding ({out['device_for_encoding']}) | "
             f"{out['encode']['qps_batch1']:,} | "
             f"{out['encode']['qps_batch8']:,} | {out['encode']['qps_batch32']:,} | "
             f"{out['encode']['qps_batch128']:,} |")
    rr = [r for r in rows if r["passages"] == n_real][0]
    L.append(f"| Dense search ({n_real:,} passages) | "
             f"{round(1000/rr['search_mean_ms']):,} | {rr['qps_batch8']:,} | "
             f"{rr['qps_batch32']:,} | {rr['qps_batch128']:,} |")
    L.append("\nQuery encoding is the retrieval bottleneck, not the index. Both are "
             "orders of magnitude cheaper than generation.")
    (DELIV / "retrieval_scalability.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\nwrote retrieval_scalability.{md,json}")


if __name__ == "__main__":
    main()
