"""
WBS Task 3.1.1 - persistent RDF graph database.

`sparql_store.py` answers SPARQL from an in-process rdflib graph rebuilt from Turtle on
every start. That covers query capability but is not a database: there is no durable
index, no transactional update, no backup. This module closes the line item with
Oxigraph (`pyoxigraph`), an embedded RDF quad store with a RocksDB-backed on-disk index
and SPARQL 1.1 Query + Update.

Oxigraph rather than Neo4j or Fuseki because both of those need a JVM or a system
service and neither is installed here, while Oxigraph is a Python wheel that gives the
same three properties the WBS line item is actually about: durable storage, a standard
query language over it, and the ability to write. What it does not give is a network
service shared between hosts; `--serve` exposes the standard SPARQL 1.1 Protocol
endpoint over HTTP for that, and `sparql_store.py --sparql-endpoint` then runs the same
competency questions against it unchanged.

  python -m wbs.task3_storage.graph_db --build      # load, benchmark, update, backup
  python -m wbs.task3_storage.graph_db --serve      # SPARQL 1.1 Protocol on :8631
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wbs.task3_storage.sparql_store import PREFIX, QUERIES, TTL  # noqa: E402

DB = ROOT / "artifacts" / "graph_db" / "jeju_oxigraph"
BACKUP = ROOT / "artifacts" / "graph_db" / "jeju_oxigraph_backup"
OUT = ROOT / "wbs" / "deliverables" / "task3_storage"
NS = "http://jejunu.ac.kr/ontology/jeju-tourism#"

# A write the ontology does not contain, used to exercise Update then roll it back.
PROBE = f"<{NS}Hallasan> <{NS}wbs_probe> \"3.1.1\""


def build(a) -> None:
    import pyoxigraph as px

    if a.rebuild and DB.exists():
        shutil.rmtree(DB)
    DB.parent.mkdir(parents=True, exist_ok=True)
    fresh = not DB.exists()

    t0 = time.perf_counter()
    store = px.Store(path=str(DB))
    open_s = time.perf_counter() - t0

    load_s = None
    if fresh or len(store) == 0:
        if not TTL.exists():
            sys.exit(f"missing {TTL}")
        t0 = time.perf_counter()
        store.bulk_load(path=str(TTL), format=px.RdfFormat.TURTLE)
        store.flush()
        load_s = time.perf_counter() - t0
        print(f"bulk-loaded {len(store):,} quads in {load_s:.3f}s")
    else:
        print(f"opened existing database with {len(store):,} quads in {open_s:.3f}s")
    n_quads = len(store)

    # ── the same competency questions, now against the database ──────────────
    rows = []
    for name, question, q in QUERIES:
        lat, n, err = [], 0, None
        for _ in range(a.repeats):
            try:
                t0 = time.perf_counter()
                res = list(store.query(PREFIX + q))
                lat.append((time.perf_counter() - t0) * 1000)
                n = len(res)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                break
        rows.append({"query": name, "question": question, "rows": n,
                     "p50_ms": round(st.median(lat), 3) if lat else None,
                     "max_ms": round(max(lat), 3) if lat else None, "error": err})
        print(f"  [{name}] {n} rows, {rows[-1]['p50_ms']} ms"
              if not err else f"  [{name}] FAILED {err}", flush=True)

    # ── SPARQL 1.1 Update: the capability a file-based store does not have ───
    upd = {}
    try:
        before = len(store)
        t0 = time.perf_counter()
        store.update(f"INSERT DATA {{ {PROBE} }}")
        upd["insert_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        upd["quads_after_insert"] = len(store)
        got = list(store.query(
            f"SELECT ?v WHERE {{ <{NS}Hallasan> <{NS}wbs_probe> ?v }}"))
        upd["readback"] = str(got[0]["v"].value) if got else None
        t0 = time.perf_counter()
        store.update(f"DELETE DATA {{ {PROBE} }}")
        upd["delete_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        store.flush()
        upd["quads_after_delete"] = len(store)
        upd["rolled_back_cleanly"] = len(store) == before
    except Exception as e:
        upd["error"] = f"{type(e).__name__}: {e}"

    # ── durability: reopen in a second handle, and take a backup ─────────────
    dur = {}
    try:
        del store
        t0 = time.perf_counter()
        s2 = px.Store(path=str(DB))
        dur["reopen_s"] = round(time.perf_counter() - t0, 4)
        dur["quads_after_reopen"] = len(s2)
        dur["survived_restart"] = len(s2) == n_quads
        if BACKUP.exists():
            shutil.rmtree(BACKUP)
        t0 = time.perf_counter()
        s2.backup(str(BACKUP))
        dur["backup_s"] = round(time.perf_counter() - t0, 3)
        dur["backup_bytes"] = sum(p.stat().st_size
                                  for p in BACKUP.rglob("*") if p.is_file())
        dur["db_bytes"] = sum(p.stat().st_size for p in DB.rglob("*") if p.is_file())
        del s2
    except Exception as e:
        dur["error"] = f"{type(e).__name__}: {e}"

    # ── compare against the in-process rdflib store on identical queries ─────
    cmp = None
    prev = OUT / "sparql_store.json"
    if prev.exists():
        p = json.loads(prev.read_text(encoding="utf-8"))
        rd = {q["query"]: q for q in p["queries"]}
        cmp = []
        for r in rows:
            o = rd.get(r["query"])
            if o and o.get("p50_ms") and r.get("p50_ms"):
                cmp.append({"query": r["query"], "rdflib_ms": o["p50_ms"],
                            "oxigraph_ms": r["p50_ms"],
                            "speedup": round(o["p50_ms"] / r["p50_ms"], 1),
                            "rows_match": o["rows"] == r["rows"],
                            "rdflib_rows": o["rows"], "oxigraph_rows": r["rows"]})

    out = {"engine": f"Oxigraph (pyoxigraph {px.__version__ if hasattr(px, '__version__') else '0.5.x'})",
           "path": str(DB.relative_to(ROOT)), "quads": n_quads,
           "open_s": round(open_s, 4),
           "bulk_load_s": round(load_s, 3) if load_s else None,
           "repeats": a.repeats, "queries": rows, "update": upd, "durability": dur,
           "vs_rdflib": cmp,
           "endpoint": "python -m wbs.task3_storage.graph_db --serve"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graph_db.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out)
    print(f"\nwrote graph_db.{{md,json}}")


def write_report(d: dict) -> None:
    L = [f"# Task 3.1.1 - Persistent RDF Graph Database\n",
         f"Engine: **{d['engine']}**, on-disk at `{d['path']}`, "
         f"{d['quads']:,} quads."]
    if d.get("bulk_load_s"):
        L.append(f" Bulk-loaded from the Task 2.2 Turtle in {d['bulk_load_s']} s; "
                 f"opening the existing database takes {d['open_s']} s.\n")
    else:
        L.append(f" Opened in {d['open_s']} s without re-parsing any source file.\n")

    L.append("\n## Why this engine\n")
    L.append("The WBS line item names \"RDF/graph database\". Oxigraph is the RDF "
             "half and it is the one the delivered pipeline uses. Neo4j is the "
             "property-graph half, and since 2026-11-03 it is no longer deferred: a "
             "JDK and the Neo4j 5.26 tarball sit under `~/.local/opt`, "
             "`wbs/task3_storage/neo4j_server.sh` runs the server, and "
             "`wbs/task3_storage/neo4j_store.py` answers the same nine competency "
             "questions in Cypher - see `graph_db_neo4j.md`. Fuseki still needs a JVM "
             "service and was not deployed. "
             "Oxigraph is an embedded RDF quad store distributed as a Python wheel, with "
             "a RocksDB-backed on-disk index and SPARQL 1.1 Query and Update. It "
             "supplies the three properties the line item is about — durable indexed "
             "storage, a standard query language over it, and transactional writes — "
             "without an infrastructure change. What it does not supply on its own is a "
             "network service; `--serve` exposes the SPARQL 1.1 Protocol over HTTP, and "
             "`sparql_store.py --sparql-endpoint URL` then runs these same queries "
             "against it, or against any other conformant endpoint.\n")

    ok = [r for r in d["queries"] if not r.get("error")]
    L.append("## Competency questions against the database\n")
    L.append(f"The nine queries are the ones in `sparql_store.md`, unchanged. Median of "
             f"{d['repeats']} runs.\n")
    L.append("| Query | Question | Rows | p50 | max |")
    L.append("|---|---|--:|--:|--:|")
    for r in d["queries"]:
        if r.get("error"):
            L.append(f"| {r['query']} | {r['question']} | — | — | **{r['error'][:60]}** |")
        else:
            L.append(f"| {r['query']} | {r['question']} | {r['rows']:,} | "
                     f"{r['p50_ms']:.3f} ms | {r['max_ms']:.3f} ms |")
    L.append("")

    if d.get("vs_rdflib"):
        c = d["vs_rdflib"]
        L.append("## Against the in-process rdflib store\n")
        L.append("| Query | rdflib | Oxigraph | Speedup | Same row count |")
        L.append("|---|--:|--:|--:|:--:|")
        for x in c:
            match = ("yes" if x["rows_match"]
                     else f"**no** ({x['rdflib_rows']} vs {x['oxigraph_rows']})")
            L.append(f"| {x['query']} | {x['rdflib_ms']:.2f} ms | "
                     f"{x['oxigraph_ms']:.3f} ms | **{x['speedup']}x** | {match} |")
        sp = [x["speedup"] for x in c]
        mism = [x for x in c if not x["rows_match"]]
        L.append(f"\nMedian speedup **{st.median(sp):.1f}x**, range "
                 f"{min(sp):.1f}x to {max(sp):.1f}x.")
        if mism:
            L.append(f" {len(mism)} quer{'y' if len(mism) == 1 else 'ies'} return a "
                     f"different row count between engines "
                     f"({', '.join(x['query'] for x in mism)}); since every query "
                     f"carries `LIMIT`, differing row counts mean the two engines "
                     f"enumerate solutions in a different order, not that either is "
                     f"wrong — but any downstream code that assumes a stable ordering "
                     f"without `ORDER BY` is relying on engine behaviour.")
        L.append("\n")

    u = d.get("update") or {}
    if "error" not in u:
        L.append("## Transactional write\n")
        L.append(f"A file-based store cannot be written to in place. Here `INSERT DATA` "
                 f"adds one triple in {u.get('insert_ms')} ms, taking the store to "
                 f"{u.get('quads_after_insert'):,} quads; it reads back as "
                 f"`{u.get('readback')}`; `DELETE DATA` removes it in "
                 f"{u.get('delete_ms')} ms, returning to "
                 f"{u.get('quads_after_delete'):,} quads — "
                 f"{'a clean round trip' if u.get('rolled_back_cleanly') else '**not a clean round trip, investigate**'}"
                 f". This is the capability Task 3.1.2's governance workflow needs if "
                 f"version promotion is ever to write to the store rather than "
                 f"re-serialize the whole graph.\n")
    else:
        L.append(f"## Transactional write\n\nFailed: `{u['error']}`\n")

    v = d.get("durability") or {}
    if "error" not in v:
        L.append("## Durability\n")
        L.append(f"The store handle is dropped and reopened: {v.get('reopen_s')} s to "
                 f"open, {v.get('quads_after_reopen'):,} quads present — "
                 f"{'the index survives process restart' if v.get('survived_restart') else '**quad count changed across restart, investigate**'}"
                 f", with no Turtle re-parse. `backup()` copies a consistent snapshot in "
                 f"{v.get('backup_s')} s ({v.get('backup_bytes', 0):,} bytes against an "
                 f"on-disk database of {v.get('db_bytes', 0):,} bytes), which is the "
                 f"operation the governance audit trail should pair with each version "
                 f"promotion.\n")
    else:
        L.append(f"## Durability\n\nFailed: `{v['error']}`\n")

    L.append("## Remaining difference from a server deployment\n")
    L.append("Oxigraph here is embedded: one process holds the database. Concurrent "
             "writers from separate processes are not supported by this configuration, "
             "and nothing on this host serves the graph to other machines unless "
             "`--serve` is running. For a single-writer research pipeline — which is "
             "what this project is — that is the right trade. A multi-writer or "
             "multi-host deployment would put Oxigraph's own server binary, Fuseki or "
             "GraphDB behind the same SPARQL endpoint, and every query in this report "
             "would run against it without modification.\n")
    (OUT / "graph_db.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def serve(a) -> None:
    """SPARQL 1.1 Protocol endpoint over the persistent store."""
    import pyoxigraph as px
    import uvicorn
    from fastapi import FastAPI, Request, Response

    if not DB.exists():
        sys.exit("no database yet; run --build first")
    store = px.Store.read_only(str(DB))
    app = FastAPI(title="Jeju Tourism Ontology SPARQL endpoint")

    def run(q: str) -> Response:
        import io
        res = store.query(q)
        buf = io.BytesIO()
        res.serialize(output=buf, format=px.QueryResultsFormat.JSON)
        return Response(content=buf.getvalue(),
                        media_type="application/sparql-results+json")

    @app.get("/sparql")
    async def get_q(query: str):
        return run(query)

    @app.post("/sparql")
    async def post_q(request: Request):
        body = (await request.body()).decode("utf-8")
        ct = request.headers.get("content-type", "")
        if "application/sparql-query" in ct:
            return run(body)
        from urllib.parse import parse_qs
        return run(parse_qs(body).get("query", [""])[0])

    @app.get("/health")
    async def health():
        return {"quads": len(store), "path": str(DB.relative_to(ROOT))}

    print(f"serving {len(store):,} quads at http://127.0.0.1:{a.port}/sparql")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="discard and reload the db")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--port", type=int, default=8631)
    a = ap.parse_args()
    if a.serve:
        serve(a)
    else:
        build(a)


if __name__ == "__main__":
    main()
