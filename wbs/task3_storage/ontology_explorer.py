"""
WBS Task 3.1.1 / 4.2.2 - the whole ontology as one self-contained HTML page.

The Neo4j Browser needs a running server and a forwarded port. This does not: it
writes a single file with the entire v1.4.0 ontology embedded, no server, no
network, no CDN. Open it by double-clicking, or over `file://`, and it works
offline.

Views over the same data:
  schema   - the class-level graph: every ontology class, sized by instance count,
             joined by the predicates that actually connect them in the data
  classes  - every class, its instances, and any instance's neighbourhood
  whole    - all 1,565 concepts and 7,909 relations in one frame
  search   - by name or alias, filtered by class
  neo4j    - the server this came from, and the Cypher behind each view

The WBS names Neo4j as the ontology store, so the default source is the running
server: `neo4j_export` reads the whole graph back out over Cypher, verifies it
against the file store, and this page is built from those rows. Every class is
shown with the Neo4j label it carries and every predicate with its relationship
type, so what the page says and what `MATCH` returns are the same vocabulary.

Falling back to `--source files` reads `artifacts/ontologies/<version>/` instead,
for a clone with no server. Either way the ontology is the *complete* one,
including the governance-approved elements the RDF serialisations are missing
(see `graph_db_neo4j.md`).

  python -m wbs.task3_storage.ontology_explorer            # from Neo4j
  python -m wbs.task3_storage.ontology_explorer --source files
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
OUT = ROOT / "wbs" / "deliverables" / "task3_storage" / "ontology_explorer.html"

HIER = {"subclass_of", "is_a"}


def build_payload(version: str, source: str = "neo4j") -> dict:
    neo4j_meta = None
    if source == "neo4j":
        from wbs.task3_storage.neo4j_export import load_snapshot
        concepts, relations, snap = load_snapshot()
        version = snap["version"]
        neo4j_meta = {"meta": snap["meta"], "cypher": snap["cypher"],
                      "verification": snap.get("verification"),
                      "mapping": snap.get("mapping")}
        browser_script = snap.get("browser_script") or []
    else:
        browser_script = []
        cdir = ONTO / version
        concepts = json.loads((cdir / "concepts.json").read_text(encoding="utf-8"))
        relations = json.loads((cdir / "relations.json").read_text(encoding="utf-8"))

    names = [c["name"] for c in concepts]
    index = {n: i for i, n in enumerate(names)}
    classes = sorted({(c.get("concept_class") or "Concept") for c in concepts})
    cls_index = {c: i for i, c in enumerate(classes)}

    nodes = []
    for c in concepts:
        prov = c.get("provenance") or {}
        attrs = c.get("attributes") or {}
        nodes.append([
            c["name"],
            cls_index[c.get("concept_class") or "Concept"],
            round(float(prov.get("confidence") or 0), 3),
            prov.get("extraction_prompt_version") or "",
            [str(a) for a in (c.get("aliases") or [])][:3],
            attrs.get("lat"), attrs.get("lon"),
        ])

    predicates = sorted({r["predicate"] for r in relations})
    pred_index = {p: i for i, p in enumerate(predicates)}

    edges, dropped = [], 0
    for r in relations:
        s, o = index.get(r["subject"]), index.get(r["object"])
        if s is None or o is None:
            dropped += 1
            continue
        prov = r.get("provenance") or {}
        edges.append([s, pred_index[r["predicate"]], o,
                      round(float(prov.get("confidence") or 0), 3)])

    # Class-level aggregation: which classes a predicate really joins.
    schema_edges = Counter()
    for s, p, o, _ in edges:
        schema_edges[(nodes[s][1], p, nodes[o][1])] += 1

    degree = Counter()
    for s, _, o, _ in edges:
        degree[s] += 1
        degree[o] += 1

    # The class hierarchy, from subclass_of only.
    children = defaultdict(list)
    for s, p, o, _ in edges:
        if predicates[p] == "subclass_of":
            children[nodes[o][0]].append(nodes[s][0])

    sources = Counter(n[3] for n in nodes if n[3])

    # The poster computes a good layout over the whole graph, which is far too
    # slow to redo in a browser. Reuse its coordinates so the "Everything" view
    # and the printed figures are literally the same picture.
    layouts = {}
    lay_file = (ROOT / "wbs" / "deliverables" / "task3_storage" /
                "ontology_layout.json")
    if lay_file.exists():
        lay = json.loads(lay_file.read_text(encoding="utf-8"))
        order = lay.get("order") or []
        if len(order) == len(names):
            remap = [order.index(n) if n in order else None for n in names]
            for kind in ("force", "clustered"):
                if kind in lay and all(r is not None for r in remap):
                    coords = lay[kind]
                    layouts[kind] = [coords[r] for r in remap]

    # The names a MATCH needs depend on which mapping the server is holding, so
    # they are taken from the mapping the export detected rather than assumed.
    # Under the generic mapping every node is :Concept and every edge
    # [:RELATES_TO]; printing :Beach there would send a reader to a query that
    # returns nothing.
    from wbs.task3_storage.neo4j_store import label_of, reltype_of

    generic = bool(neo4j_meta
                   and (neo4j_meta.get("mapping") or {}).get("name") == "generic")
    if generic:
        labels = ["Concept"] * len(classes)
        reltypes = ["RELATES_TO"] * len(predicates)
    else:
        labels = [label_of(c) for c in classes]
        reltypes = [reltype_of(p) for p in predicates]

    return {
        "layouts": layouts,
        "version": version,
        "source": source,
        "neo4j": neo4j_meta,
        "browser_script": browser_script,
        "labels": labels,
        "reltypes": reltypes,
        "classes": classes,
        "predicates": predicates,
        "nodes": nodes,
        "edges": edges,
        "schema_edges": [[a, p, b, n] for (a, p, b), n in
                         sorted(schema_edges.items(), key=lambda kv: -kv[1])],
        "class_counts": [sum(1 for n in nodes if n[1] == i)
                         for i in range(len(classes))],
        "degree": [degree.get(i, 0) for i in range(len(nodes))],
        "hierarchy": {k: sorted(v) for k, v in children.items()},
        "sources": sources.most_common(),
        "isolated": sum(1 for i in range(len(nodes)) if degree.get(i, 0) == 0),
        "dropped_edges": dropped,
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jeju Tourism Ontology</title>
<style>
:root{
  --bg:#f7f8fa; --panel:#fff; --ink:#15181d; --muted:#5b6471; --line:#dfe3ea;
  --accent:#1f4e79; --accent2:#2e7d5b; --warn:#c98a2b; --chip:#eef2f8;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12151a; --panel:#1a1e25; --ink:#e8ecf2; --muted:#9aa4b2; --line:#2b313b;
  --accent:#6ea8dc; --accent2:#6cc39a; --warn:#e0ad5e; --chip:#222834;
}}
:root[data-theme="dark"]{
  --bg:#12151a; --panel:#1a1e25; --ink:#e8ecf2; --muted:#9aa4b2; --line:#2b313b;
  --accent:#6ea8dc; --accent2:#6cc39a; --warn:#e0ad5e; --chip:#222834;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:14px 16px;border-bottom:1px solid var(--line);background:var(--panel);
  display:flex;flex-wrap:wrap;gap:12px;align-items:baseline}
h1{margin:0;font-size:17px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12px}
nav{display:flex;gap:4px;margin-left:auto;flex-wrap:wrap}
button.tab{background:transparent;border:1px solid var(--line);color:var(--ink);
  padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px}
button.tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);
  color:#fff}
main{padding:16px;max-width:1500px;margin:0 auto}
.view{display:none}.view.on{display:block}
.grid{display:grid;grid-template-columns:300px 1fr;gap:16px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px}
.card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted)}
canvas{width:100%;height:620px;display:block;border-radius:8px;cursor:grab}
canvas:active{cursor:grabbing}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.stat{background:var(--chip);border-radius:8px;padding:10px}
.stat b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:11px}
ul.list{list-style:none;margin:0;padding:0;max-height:560px;overflow:auto}
ul.list li{padding:6px 8px;border-radius:6px;cursor:pointer;display:flex;
  justify-content:space-between;gap:8px}
ul.list li:hover{background:var(--chip)}
ul.list li[aria-current="true"]{background:var(--accent);color:#fff}
ul.list .n{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px}
ul.list li[aria-current="true"] .n{color:#fff}
input[type=search],select{width:100%;padding:8px 10px;border:1px solid var(--line);
  border-radius:7px;background:var(--bg);color:var(--ink);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);
  vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.05em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;background:var(--chip);border-radius:999px;
  padding:1px 8px;font-size:11px;color:var(--muted)}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--chip);
  border-radius:0 8px 8px 0;font-size:13px;margin-top:12px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;font-size:11px;
  color:var(--muted)}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block;
  margin-right:4px;vertical-align:-1px}
a{color:var(--accent)}
.rowflex{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.muted{color:var(--muted)}
pre.cy{background:var(--chip);border:1px solid var(--line);border-radius:8px;
  padding:10px 12px;overflow:auto;font:12px/1.45 ui-monospace,SFMono-Regular,
  Menlo,Consolas,monospace;margin:6px 0 0;white-space:pre}
.cyhead{display:flex;align-items:baseline;gap:10px;margin-top:14px}
.cyhead h3{margin:0;font-size:13px}
.cyhead .what{color:var(--muted);font-size:12px}
button.copy{background:transparent;border:1px solid var(--line);color:var(--muted);
  border-radius:6px;padding:3px 9px;font-size:11px;cursor:pointer;margin-left:auto}
button.copy:hover{color:var(--ink);border-color:var(--accent)}
.badge{display:inline-block;background:var(--accent);color:#fff;border-radius:5px;
  padding:1px 7px;font-size:11px;font-weight:600;letter-spacing:.02em}
.badge.ok{background:var(--accent2)}
.badge.bad{background:#b03a4a}
</style></head><body>
<header>
  <div><h1>Jeju Tourism Ontology</h1>
  <div class="sub" id="hdr"></div></div>
  <nav role="tablist">
    <button class="tab" role="tab" data-view="schema" aria-selected="true">Schema</button>
    <button class="tab" role="tab" data-view="classes" aria-selected="false">Classes</button>
    <button class="tab" role="tab" data-view="whole" aria-selected="false">Everything</button>
    <button class="tab" role="tab" data-view="search" aria-selected="false">Search</button>
    <button class="tab" role="tab" data-view="about" aria-selected="false">Facts</button>
    <button class="tab" role="tab" data-view="neo4j" aria-selected="false">Neo4j</button>
  </nav>
</header>
<main>
  <section class="view on" id="v-schema">
    <div class="card">
      <h2>Class-level graph &mdash; drag to pan, scroll to zoom, click a class to isolate it</h2>
      <canvas id="cv-schema"></canvas>
      <div class="legend" id="lg-schema"></div>
    </div>
  </section>

  <section class="view" id="v-classes">
    <div class="grid">
      <div class="card">
        <h2>Classes</h2>
        <ul class="list" id="cls-list"></ul>
      </div>
      <div>
        <div class="card" style="margin-bottom:16px">
          <h2 id="inst-head">Instances</h2>
          <div class="rowflex"><input type="search" id="inst-filter"
            placeholder="filter instances&hellip;"></div>
          <ul class="list" id="inst-list" style="max-height:220px"></ul>
        </div>
        <div class="card">
          <h2 id="ego-head">Neighbourhood</h2>
          <canvas id="cv-ego" style="height:420px"></canvas>
          <div id="ego-facts"></div>
        </div>
      </div>
    </div>
  </section>

  <section class="view" id="v-whole">
    <div class="card">
      <h2>Every concept and every relation at once &mdash; drag to pan, scroll to
        zoom, click a node to open it in Classes</h2>
      <div class="rowflex">
        <select id="whole-layout" style="flex:0 1 260px"></select>
        <label style="font-size:13px"><input type="checkbox" id="whole-labels"
          checked> label the hubs</label>
        <span class="muted" id="whole-note"></span>
      </div>
      <canvas id="cv-whole" style="height:720px"></canvas>
      <div class="legend" id="lg-whole"></div>
    </div>
  </section>

  <section class="view" id="v-search">
    <div class="card">
      <h2>Search by name or alias</h2>
      <div class="rowflex">
        <div style="flex:1 1 260px"><input type="search" id="q"
          placeholder="type at least two characters&hellip;" autocomplete="off"></div>
        <div style="flex:0 1 200px"><select id="q-class"></select></div>
      </div>
      <table><thead><tr><th>Concept</th><th>Class</th><th class="num">Degree</th>
        <th class="num">Confidence</th><th>Source</th><th>Aliases</th></tr></thead>
        <tbody id="q-rows"></tbody></table>
      <p class="muted" id="q-note"></p>
    </div>
  </section>

  <section class="view" id="v-about">
    <div class="card">
      <h2>What is in this file</h2>
      <div class="stats" id="facts"></div>
      <div class="note" id="gov"></div>
      <h2 style="margin-top:18px">Class hierarchy</h2>
      <table><thead><tr><th>Parent</th><th class="num">Children</th>
        <th>Subclasses</th></tr></thead><tbody id="hier"></tbody></table>
      <h2 style="margin-top:18px">Predicates</h2>
      <table><thead><tr><th>Predicate</th><th class="num">Edges</th>
        <th>Connects</th></tr></thead><tbody id="preds"></tbody></table>
      <h2 style="margin-top:18px">Provenance</h2>
      <table><thead><tr><th>Source</th><th class="num">Concepts</th></tr></thead>
        <tbody id="srcs"></tbody></table>
    </div>
  </section>

  <section class="view" id="v-neo4j">
    <div class="card">
      <h2>The server this page was built from</h2>
      <div class="stats" id="nmeta"></div>
      <div class="note" id="nverify"></div>

      <h2 style="margin-top:20px">Open the whole ontology in the Neo4j Browser</h2>
      <p class="muted" id="nhow"></p>
      <div id="ncy"></div>

      <h2 style="margin-top:20px">Neo4j vocabulary</h2>
      <p class="muted" id="nvocab"></p>
      <div class="grid">
        <div><table><thead><tr><th>Class</th><th>:Label</th>
          <th class="num">n</th></tr></thead><tbody id="nlabels"></tbody></table></div>
        <div><table><thead><tr><th>Predicate</th><th>[:TYPE]</th>
          <th class="num">Edges</th></tr></thead><tbody id="nreltypes"></tbody>
          </table></div>
      </div>
    </div>
  </section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const NAME=0, CLS=1, CONF=2, SRC=3, ALIAS=4;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const PALETTE = ['#1f4e79','#2e7d5b','#c98a2b','#8e44ad','#b03a4a','#2b7a8c',
  '#6b7b2a','#a0522d','#4a69bd','#7f8c8d'];
const colorOf = i => PALETTE[i % PALETTE.length];

document.getElementById('hdr').innerHTML =
  `v${D.version} · ${D.nodes.length.toLocaleString()} concepts · ` +
  `${D.edges.length.toLocaleString()} relations · ${D.classes.length} classes ` +
  `· ${D.predicates.length} predicates · ` +
  (D.neo4j
    ? `<span class="badge">${D.neo4j.meta.engine}</span> read over Cypher from ` +
      `<code>${D.neo4j.meta.bolt}</code>`
    : `built from the versioned file store`);

/* ---------------------------------------------------------------- tabs */
const tabs = [...document.querySelectorAll('.tab')];
tabs.forEach(t => t.onclick = () => {
  tabs.forEach(o => o.setAttribute('aria-selected', String(o === t)));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('on'));
  document.getElementById('v-' + t.dataset.view).classList.add('on');
  if (t.dataset.view === 'schema') schema.reset();
  if (t.dataset.view === 'classes') ego.reset();
  if (t.dataset.view === 'whole') whole.draw();
});

/* ------------------------------------------------- tiny force layout engine */
function layout(canvas, getGraph, onPick) {
  const ctx = canvas.getContext('2d');
  let nodes = [], links = [], tx = 0, ty = 0, scale = 1, raf = 0, ticks = 0;
  let hover = null, drag = null, isolate = null;

  function size() {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }
  function reset() {
    const g = getGraph(); nodes = g.nodes; links = g.links;
    const r = size();
    if (!r.width || !r.height) { return; }   // hidden view; re-seeded on tab entry
    nodes.forEach((n, i) => {
      const a = (i / nodes.length) * Math.PI * 2;
      n.x = r.width / 2 + Math.cos(a) * Math.min(r.width, r.height) * 0.33;
      n.y = r.height / 2 + Math.sin(a) * Math.min(r.width, r.height) * 0.33;
      n.vx = n.vy = 0;
    });
    tx = ty = 0; scale = 1; ticks = 0; isolate = null; kick();
  }
  function step() {
    const r = canvas.getBoundingClientRect();
    const cx = r.width / 2, cy = r.height / 2;
    for (const n of nodes) { n.vx *= 0.82; n.vy *= 0.82; }
    // repulsion, O(n^2) but n is small in every view this page draws
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy || 0.01;
        const d = Math.sqrt(d2);
        const rep = (900 + 60 * (a.r + b.r)) / d2;
        const fx = dx / d * rep, fy = dy / d * rep;
        a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
      }
    }
    for (const l of links) {
      const a = nodes[l.s], b = nodes[l.t];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const want = 90 + a.r + b.r;
      const f = (d - want) * 0.012;
      a.vx += dx / d * f; a.vy += dy / d * f;
      b.vx -= dx / d * f; b.vy -= dy / d * f;
    }
    for (const n of nodes) {
      n.vx += (cx - n.x) * 0.004; n.vy += (cy - n.y) * 0.004;
      if (n !== drag) { n.x += n.vx; n.y += n.vy; }
    }
    ticks++;
  }
  function draw() {
    const r = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    ctx.save(); ctx.translate(tx, ty); ctx.scale(scale, scale);
    const dim = isolate !== null;
    for (const l of links) {
      const a = nodes[l.s], b = nodes[l.t];
      if (!a || !b) continue;
      const lit = !dim || l.s === isolate || l.t === isolate;
      ctx.strokeStyle = lit ? (l.color || css('--line')) : css('--line');
      ctx.globalAlpha = lit ? (dim ? 0.95 : 0.5) : 0.08;
      ctx.lineWidth = Math.max(0.6, Math.min(5, (l.w || 1)));
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const lit = !dim || i === isolate ||
        links.some(l => (l.s === isolate && l.t === i) ||
                        (l.t === isolate && l.s === i));
      ctx.globalAlpha = lit ? 1 : 0.15;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.2832);
      ctx.fillStyle = n.color; ctx.fill();
      if (i === hover || i === isolate) {
        ctx.lineWidth = 2; ctx.strokeStyle = css('--ink'); ctx.stroke();
      }
      if (lit && (n.r > 7 || i === hover || nodes.length < 60)) {
        ctx.font = '600 11px -apple-system,sans-serif';
        ctx.fillStyle = css('--ink'); ctx.textAlign = 'center';
        ctx.fillText(n.label, n.x, n.y - n.r - 5);
      }
    }
    ctx.restore();
  }
  function frame() {
    if (ticks < 400 || drag) step();
    draw();
    raf = requestAnimationFrame(frame);
  }
  function kick() { ticks = 0; cancelAnimationFrame(raf); frame(); }
  function at(ev) {
    const r = canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left - tx) / scale, y = (ev.clientY - r.top - ty) / scale;
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      if ((n.x - x) ** 2 + (n.y - y) ** 2 <= (n.r + 4) ** 2) return i;
    }
    return null;
  }
  let pan = null;
  canvas.onmousedown = ev => {
    const i = at(ev);
    if (i !== null) { drag = nodes[i]; }
    else pan = { x: ev.clientX - tx, y: ev.clientY - ty };
  };
  canvas.onmousemove = ev => {
    const r = canvas.getBoundingClientRect();
    if (drag) {
      drag.x = (ev.clientX - r.left - tx) / scale;
      drag.y = (ev.clientY - r.top - ty) / scale;
      drag.vx = drag.vy = 0;
    } else if (pan) { tx = ev.clientX - pan.x; ty = ev.clientY - pan.y; }
    else {
      const i = at(ev); if (i !== hover) { hover = i; canvas.title =
        i === null ? '' : nodes[i].tip || nodes[i].label; }
    }
  };
  window.addEventListener('mouseup', () => { drag = null; pan = null; });
  canvas.onclick = ev => {
    const i = at(ev);
    if (i === null) { isolate = null; return; }
    isolate = (isolate === i) ? null : i;
    if (onPick) onPick(nodes[i], i);
  };
  canvas.onwheel = ev => {
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const k = Math.exp(-ev.deltaY * 0.0012), ns = Math.min(4, Math.max(0.25, scale * k));
    tx = mx - (mx - tx) * (ns / scale); ty = my - (my - ty) * (ns / scale);
    scale = ns;
  };
  new ResizeObserver(() => { size(); }).observe(canvas);
  return { reset, kick };
}

/* ------------------------------------------------------------ schema view */
const schema = (() => {
  const canvas = document.getElementById('cv-schema');
  const eng = layout(canvas, () => {
    const max = Math.max(...D.class_counts);
    const nodes = D.classes.map((c, i) => ({
      label: c, color: colorOf(i), r: 6 + 20 * Math.sqrt(D.class_counts[i] / max),
      tip: `${c} — ${D.class_counts[i].toLocaleString()} concepts`
    }));
    const byPair = new Map();
    for (const [a, p, b, n] of D.schema_edges) {
      const k = a + ':' + b;
      byPair.set(k, (byPair.get(k) || 0) + n);
    }
    const links = [];
    for (const [k, n] of byPair) {
      const [a, b] = k.split(':').map(Number);
      if (a === b || n < 5) continue;
      links.push({ s: a, t: b, w: 0.5 + Math.log10(n), color: css('--accent') });
    }
    return { nodes, links };
  });
  const lg = document.getElementById('lg-schema');
  lg.innerHTML = D.classes.map((c, i) =>
    `<span><i style="background:${colorOf(i)}"></i>${c} ` +
    `<span class="chip">${D.class_counts[i].toLocaleString()}</span></span>`).join('');
  return eng;
})();
schema.reset();

/* ----------------------------------------------------------- classes view */
const adj = (() => {
  const out = Array.from({ length: D.nodes.length }, () => []);
  for (const [s, p, o] of D.edges) { out[s].push([o, p, 1]); out[o].push([s, p, 0]); }
  return out;
})();

let curClass = 0, curNode = null;
const clsList = document.getElementById('cls-list');
const instList = document.getElementById('inst-list');
const instFilter = document.getElementById('inst-filter');

function renderClasses() {
  clsList.innerHTML = D.classes.map((c, i) =>
    `<li data-i="${i}" aria-current="${i === curClass}">${c}` +
    `<span class="n">${D.class_counts[i].toLocaleString()}</span></li>`).join('');
  [...clsList.children].forEach(li => li.onclick = () => {
    curClass = +li.dataset.i; renderClasses(); renderInstances();
  });
}
function renderInstances() {
  const q = instFilter.value.trim().toLowerCase();
  const items = [];
  for (let i = 0; i < D.nodes.length; i++) {
    if (D.nodes[i][CLS] !== curClass) continue;
    if (q && !D.nodes[i][NAME].toLowerCase().includes(q)) continue;
    items.push(i);
  }
  items.sort((a, b) => D.degree[b] - D.degree[a]);
  document.getElementById('inst-head').textContent =
    `Instances of ${D.classes[curClass]} (${items.length.toLocaleString()})`;
  instList.innerHTML = items.slice(0, 400).map(i =>
    `<li data-i="${i}" aria-current="${i === curNode}">${D.nodes[i][NAME]}` +
    `<span class="n">${D.degree[i]}</span></li>`).join('') ||
    '<li class="muted">nothing matches</li>';
  [...instList.children].forEach(li => li.onclick = () => {
    if (!li.dataset.i) return;
    curNode = +li.dataset.i; renderInstances(); ego.reset();
  });
  if (curNode === null && items.length) { curNode = items[0]; ego.reset(); }
}
instFilter.oninput = renderInstances;

const ego = layout(document.getElementById('cv-ego'), () => {
  if (curNode === null) return { nodes: [], links: [] };
  const seen = new Map([[curNode, 0]]);
  const nodes = [{ label: D.nodes[curNode][NAME], color: css('--accent'), r: 14,
                   tip: D.classes[D.nodes[curNode][CLS]] }];
  const links = [];
  for (const [other, p, outward] of adj[curNode].slice(0, 70)) {
    if (!seen.has(other)) {
      seen.set(other, nodes.length);
      nodes.push({ label: D.nodes[other][NAME],
        color: colorOf(D.nodes[other][CLS]), r: 8,
        tip: `${D.classes[D.nodes[other][CLS]]} · degree ${D.degree[other]}` });
    }
    const j = seen.get(other);
    links.push({ s: outward ? 0 : j, t: outward ? j : 0,
                 w: 1.2, color: css('--accent2') });
  }
  return { nodes, links };
}, n => {
  const i = D.nodes.findIndex(x => x[NAME] === n.label);
  if (i >= 0 && i !== curNode) {
    curNode = i; curClass = D.nodes[i][CLS];
    renderClasses(); renderInstances(); ego.reset();
  }
});

const origEgoReset = ego.reset;
ego.reset = function () {
  origEgoReset();
  if (curNode === null) return;
  const n = D.nodes[curNode];
  document.getElementById('ego-head').textContent =
    `${n[NAME]} — ${D.classes[n[CLS]]}, degree ${D.degree[curNode]}`;
  const byPred = new Map();
  for (const [other, p, outward] of adj[curNode]) {
    const k = (outward ? '' : '← ') + D.predicates[p];
    if (!byPred.has(k)) byPred.set(k, []);
    byPred.get(k).push(D.nodes[other][NAME]);
  }
  const rows = [...byPred.entries()].sort((a, b) => b[1].length - a[1].length)
    .map(([k, v]) => `<tr><td><code>${k}</code></td><td class="num">${v.length}</td>` +
      `<td>${v.slice(0, 8).join(', ')}${v.length > 8 ? ' …' : ''}</td></tr>`).join('');
  document.getElementById('ego-facts').innerHTML =
    `<table style="margin-top:12px"><thead><tr><th>Predicate</th>` +
    `<th class="num">n</th><th>Objects</th></tr></thead><tbody>${rows}</tbody></table>` +
    `<p class="muted">confidence ${n[CONF]} · source <code>${n[SRC]}</code>` +
    (n[ALIAS].length ? ` · aliases ${n[ALIAS].join(', ')}` : '') + `</p>`;
};
renderClasses(); renderInstances();

/* ----------------------------------------------------------- whole graph */
const whole = (() => {
  const canvas = document.getElementById('cv-whole');
  const sel = document.getElementById('whole-layout');
  const labelsOn = document.getElementById('whole-labels');
  const note = document.getElementById('whole-note');
  const kinds = Object.keys(D.layouts || {});
  const ctx = canvas.getContext('2d');
  let tx = 0, ty = 0, scale = 1, pan = null, hover = null, fitted = false;

  if (!kinds.length) {
    note.textContent = 'No precomputed layout in this build — run ' +
      'wbs.task3_storage.ontology_poster to add one.';
    return { draw() {} };
  }
  sel.innerHTML = kinds.map(k => `<option value="${k}">${
    k === 'force' ? 'force-directed — the shape the graph has'
                  : 'by class — every class in its own cluster'}</option>`).join('');

  const deg = D.degree;
  const maxDeg = Math.max(...deg);
  const hubs = [...deg.keys()].sort((a, b) => deg[b] - deg[a]).slice(0, 40);

  function pts() { return D.layouts[sel.value]; }
  function fit() {
    const P = pts(), r = canvas.getBoundingClientRect();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of P) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const m = 30;
    scale = Math.min((r.width - 2 * m) / (x1 - x0 || 1),
                     (r.height - 2 * m) / (y1 - y0 || 1));
    tx = r.width / 2 - ((x0 + x1) / 2) * scale;
    ty = r.height / 2 + ((y0 + y1) / 2) * scale;   // canvas y grows downward
    fitted = true;
  }
  const SX = (x) => x * scale + tx;
  const SY = (y) => -y * scale + ty;

  function draw() {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!fitted) fit();
    ctx.clearRect(0, 0, r.width, r.height);
    const P = pts();

    ctx.lineWidth = 0.4;
    ctx.strokeStyle = css('--line');
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    for (const [s, , o] of D.edges) {
      ctx.moveTo(SX(P[s][0]), SY(P[s][1]));
      ctx.lineTo(SX(P[o][0]), SY(P[o][1]));
    }
    ctx.stroke();

    ctx.globalAlpha = 1;
    for (let i = 0; i < P.length; i++) {
      const rad = 1.4 + 5.2 * Math.sqrt(deg[i] / maxDeg);
      ctx.beginPath();
      ctx.arc(SX(P[i][0]), SY(P[i][1]), rad, 0, 6.2832);
      ctx.fillStyle = colorOf(D.nodes[i][CLS]);
      ctx.fill();
      if (i === hover) {
        ctx.lineWidth = 2; ctx.strokeStyle = css('--ink'); ctx.stroke();
      }
    }
    if (labelsOn.checked) {
      ctx.font = '600 11px -apple-system,sans-serif';
      ctx.textAlign = 'center';
      for (const i of hubs) {
        const x = SX(P[i][0]), y = SY(P[i][1]);
        const t = D.nodes[i][NAME];
        const w = ctx.measureText(t).width;
        ctx.globalAlpha = 0.82; ctx.fillStyle = css('--panel');
        ctx.fillRect(x - w / 2 - 3, y - 20, w + 6, 13);
        ctx.globalAlpha = 1; ctx.fillStyle = css('--ink');
        ctx.fillText(t, x, y - 10);
      }
    }
    note.textContent = `${D.nodes.length.toLocaleString()} concepts and ` +
      `${D.edges.length.toLocaleString()} relations drawn`;
  }
  function at(ev) {
    const r = canvas.getBoundingClientRect(), P = pts();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 81;
    for (let i = 0; i < P.length; i++) {
      const dx = SX(P[i][0]) - mx, dy = SY(P[i][1]) - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  canvas.onmousedown = ev => { pan = { x: ev.clientX - tx, y: ev.clientY - ty }; };
  window.addEventListener('mouseup', () => { pan = null; });
  canvas.onmousemove = ev => {
    if (pan) { tx = ev.clientX - pan.x; ty = ev.clientY - pan.y; draw(); return; }
    const i = at(ev);
    if (i !== hover) {
      hover = i;
      canvas.title = i === null ? '' :
        `${D.nodes[i][NAME]} — ${D.classes[D.nodes[i][CLS]]}, degree ${deg[i]}`;
      draw();
    }
  };
  canvas.onwheel = ev => {
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const k = Math.exp(-ev.deltaY * 0.0012);
    const ns = Math.min(40, Math.max(0.2, scale * k));
    tx = mx - (mx - tx) * (ns / scale); ty = my - (my - ty) * (ns / scale);
    scale = ns; draw();
  };
  canvas.onclick = ev => {
    const i = at(ev);
    if (i === null) return;
    curNode = i; curClass = D.nodes[i][CLS];
    renderClasses(); renderInstances(); ego.reset();
    document.querySelector('.tab[data-view="classes"]').click();
  };
  sel.onchange = () => { fitted = false; draw(); };
  labelsOn.onchange = draw;
  new ResizeObserver(() => { fitted = false; draw(); }).observe(canvas);
  document.getElementById('lg-whole').innerHTML = D.classes.map((c, i) =>
    `<span><i style="background:${colorOf(i)}"></i>${c}</span>`).join('');
  return { draw };
})();

/* ------------------------------------------------------------ search view */
const qEl = document.getElementById('q'), qCls = document.getElementById('q-class');
qCls.innerHTML = '<option value="-1">every class</option>' +
  D.classes.map((c, i) => `<option value="${i}">${c}</option>`).join('');
function runSearch() {
  const q = qEl.value.trim().toLowerCase(), cf = +qCls.value;
  const rows = [];
  if (q.length >= 2 || cf >= 0) {
    for (let i = 0; i < D.nodes.length; i++) {
      const n = D.nodes[i];
      if (cf >= 0 && n[CLS] !== cf) continue;
      if (q.length >= 2) {
        const hay = (n[NAME] + ' ' + n[ALIAS].join(' ')).toLowerCase();
        if (!hay.includes(q)) continue;
      }
      rows.push(i);
    }
  }
  rows.sort((a, b) => D.degree[b] - D.degree[a]);
  document.getElementById('q-rows').innerHTML = rows.slice(0, 250).map(i => {
    const n = D.nodes[i];
    return `<tr><td><b>${n[NAME]}</b></td><td>${D.classes[n[CLS]]}</td>` +
      `<td class="num">${D.degree[i]}</td><td class="num">${n[CONF]}</td>` +
      `<td><code>${n[SRC]}</code></td><td class="muted">${n[ALIAS].join(', ')}</td></tr>`;
  }).join('');
  document.getElementById('q-note').textContent = rows.length
    ? `${rows.length.toLocaleString()} match${rows.length === 1 ? '' : 'es'}` +
      (rows.length > 250 ? ', showing the 250 most connected' : '')
    : (q.length >= 2 || cf >= 0 ? 'nothing matches' : '');
}
qEl.oninput = runSearch; qCls.onchange = runSearch;

/* ------------------------------------------------------------- facts view */
const predCount = new Map();
for (const [, p] of D.edges) predCount.set(p, (predCount.get(p) || 0) + 1);
const topDeg = [...D.degree.keys()].sort((a, b) => D.degree[b] - D.degree[a]).slice(0, 5);
document.getElementById('facts').innerHTML = [
  ['Concepts', D.nodes.length.toLocaleString()],
  ['Relations', D.edges.length.toLocaleString()],
  ['Classes', D.classes.length],
  ['Predicates', D.predicates.length],
  ['Isolated concepts', D.isolated],
  ['Most connected', D.nodes[topDeg[0]][NAME] + ' (' + D.degree[topDeg[0]] + ')'],
].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

const gov = D.sources.filter(([s]) => s.startsWith('governance/'));
document.getElementById('gov').innerHTML = gov.length
  ? `<b>Governance-approved elements:</b> ` + gov.map(([s, n]) =>
      `<code>${s}</code> (${n} concept${n === 1 ? '' : 's'})`).join(', ') +
    `. These are present in this file and in Neo4j, and absent from every RDF path &mdash;
     the serialisations declare an older version. See
     <code>graph_db_neo4j.md</code>.`
  : `No governance-approved concepts in this version.`;

document.getElementById('hier').innerHTML = Object.entries(D.hierarchy)
  .sort((a, b) => b[1].length - a[1].length).map(([p, kids]) =>
    `<tr><td><b>${p}</b></td><td class="num">${kids.length}</td>` +
    `<td class="muted">${kids.slice(0, 14).join(', ')}` +
    `${kids.length > 14 ? ' …' : ''}</td></tr>`).join('');

const pairsFor = p => D.schema_edges.filter(e => e[1] === p)
  .slice(0, 3).map(e => `${D.classes[e[0]]}→${D.classes[e[2]]}`).join(', ');
document.getElementById('preds').innerHTML = [...predCount.entries()]
  .sort((a, b) => b[1] - a[1]).map(([p, n]) =>
    `<tr><td><code>${D.predicates[p]}</code></td><td class="num">${n.toLocaleString()}</td>` +
    `<td class="muted">${pairsFor(p)}</td></tr>`).join('');

document.getElementById('srcs').innerHTML = D.sources.map(([s, n]) =>
  `<tr><td><code>${s}</code></td><td class="num">${n.toLocaleString()}</td></tr>`).join('');

/* ------------------------------------------------------------- neo4j view */
/* The query list comes from neo4j_export.BROWSER_SCRIPT, so this page, the
   whole_ontology.cypher script and the schema report cannot drift apart. */
const BROWSER_CYPHER = D.browser_script || [];

if (D.neo4j) {
  const m = D.neo4j.meta, v = D.neo4j.verification;
  document.getElementById('nmeta').innerHTML = [
    ['Engine', m.engine.replace('Neo4j Kernel ', '')],
    ['Edition', m.edition],
    ['Bolt', m.bolt.replace('bolt://', '')],
    ['Database', m.database],
    ['Nodes', m.nodes.toLocaleString()],
    ['Relationships', m.relationships.toLocaleString()],
    ['Read', m.exported_utc.replace('T', ' ').replace('Z', ' UTC')],
  ].map(([k, x]) => `<div class="stat"><b>${x}</b><span>${k}</span></div>`).join('');

  const mp = D.neo4j.mapping;
  if (mp) {
    document.getElementById('nmeta').innerHTML +=
      `<div class="stat"><b>${mp.name}</b><span>Schema in the database</span></div>`;
  }
  document.getElementById('nverify').innerHTML = v
    ? (v.exact
        ? `<span class="badge ok">verified</span> Every row on this page came out
           of Neo4j over Cypher, and the round trip is exact: the same
           ${v.file_store_concepts.toLocaleString()} concept names, the same
           ${v.file_store_relations.toLocaleString()} subject–predicate–object
           triples and the same class for every concept as the
           v${v.file_store_version} file store the server was loaded from.`
        : `<span class="badge bad">mismatch</span> What came out of Neo4j is not
           what went in. See <code>ontology_from_neo4j.json</code> →
           <code>verification</code>.`)
    : `Exported without verification against the file store.`;

  const http = m.bolt.replace('bolt://', '').replace(/:\d+$/, ':7474');
  const hostPort = m.bolt.replace('bolt://', '');
  document.getElementById('nhow').innerHTML =
    `The server listens on localhost only, so it is reached through an SSH tunnel
     rather than over the open internet:
     <pre class="cy">ssh -L 7474:${http} -L 7687:${hostPort} ${m.user_hint || 'you@host'}</pre>
     Then open <code>http://${http}</code>, connect to
     <code>${m.bolt}</code> as user <code>neo4j</code>, and paste these in order.`;

  document.getElementById('ncy').innerHTML = BROWSER_CYPHER.map((b, i) => {
    /* Each query was run against this server when the page was generated; the
       row count is reported so a query that comes back empty is visibly empty
       by design rather than looking like a broken ontology. */
    const ran = (b.rows === null || b.rows === undefined)
      ? (b.note ? `<span class="badge bad">${b.note}</span>` : '')
      : (b.rows === 0
          ? `<span class="badge bad">0 rows</span>`
          : `<span class="badge ok">${b.rows.toLocaleString()} rows · ${b.ms} ms</span>`);
    return `<div class="cyhead"><h3>${b.title}</h3><span class="what">${b.why}</span>
      ${ran}<button class="copy" data-i="${i}">copy</button></div>
     <pre class="cy" id="cy${i}">${b.cypher.replace(/</g, '&lt;')}</pre>`;
  }).join('');

  for (const b of document.querySelectorAll('button.copy')) {
    b.onclick = () => {
      const txt = document.getElementById('cy' + b.dataset.i).textContent;
      const done = () => { b.textContent = 'copied';
        setTimeout(() => { b.textContent = 'copy'; }, 1200); };
      const fail = () => { b.textContent = 'select it'; };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done, fail);
      } else { fail(); }
    };
  }
} else {
  document.getElementById('nmeta').innerHTML =
    `<div class="stat"><b>file store</b><span>Source</span></div>`;
  document.getElementById('nverify').textContent =
    'This page was built from the versioned file store, not from a running ' +
    'server, so there is nothing to report about Neo4j here. Rebuild with ' +
    '--source neo4j to populate this view.';
}

document.getElementById('nvocab').innerHTML =
  (D.neo4j && D.neo4j.mapping && D.neo4j.mapping.name === 'generic')
    ? `This database holds the <b>generic</b> mapping: every concept is
       <code>:Concept</code> and every relation <code>[:RELATES_TO]</code>, with
       the class and the predicate carried as properties. It is the schema the
       companion project at <code>Ontology_Project</code> uses. Match on the
       property, not the label: <code>{concept_class: 'Beach'}</code> and
       <code>{predicate: 'has_risk'}</code>.`
    : `This database holds the <b>typed</b> mapping: a label per ontology class
       and a relationship type per predicate. The loader sanitises them, so
       <code>Travel Constraint</code> is the label <code>Travel_Constraint</code>
       and <code>has-risk</code> is the type <code>HAS_RISK</code>.`;

const clsCount = (i) => D.class_counts[i];
document.getElementById('nlabels').innerHTML = D.classes
  .map((c, i) => [c, D.labels[i], clsCount(i)])
  .sort((a, b) => b[2] - a[2]).map(([c, l, n]) =>
    `<tr><td>${c}</td><td><code>:${l}</code></td>` +
    `<td class="num">${n.toLocaleString()}</td></tr>`).join('');

document.getElementById('nreltypes').innerHTML = [...predCount.entries()]
  .sort((a, b) => b[1] - a[1]).map(([p, n]) =>
    `<tr><td>${D.predicates[p]}</td><td><code>[:${D.reltypes[p]}]</code></td>` +
    `<td class="num">${n.toLocaleString()}</td></tr>`).join('');
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--source", default="neo4j", choices=("neo4j", "files"),
                    help="neo4j reads the snapshot written by neo4j_export, "
                         "which is the graph as Cypher returns it; files reads "
                         "artifacts/ontologies/<version>/ for a clone with no "
                         "server")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    payload = build_payload(a.version, a.source)
    html = PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":"),
                                               ensure_ascii=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    if payload["neo4j"]:
        m = payload["neo4j"]["meta"]
        v = payload["neo4j"]["verification"]
        print(f"source: {m['engine']} at {m['bolt']}, read {m['exported_utc']}"
              + ("" if not v else
                 f" ({'verified exact' if v['exact'] else 'MISMATCH'} "
                 f"against the file store)"))
    else:
        print("source: the versioned file store (no server)")
    print(f"{payload['version']}: {len(payload['nodes']):,} concepts, "
          f"{len(payload['edges']):,} relations, {len(payload['classes'])} classes, "
          f"{len(payload['predicates'])} predicates")
    if payload["dropped_edges"]:
        print(f"relations whose endpoints are not in the concept set: "
              f"{payload['dropped_edges']}")
    print(f"wrote {out} ({kb:,.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
