"""
exp041 - export the complete Jeju tourism ontology (v1.3.0, the version the demo
app embeds) as paper-ready figures, data files and tables.

    python -m experiments.exp041_demo_traces.export_paper_ontology

Reads   Jeju_Tourism_Research_Demo/ontology/original/{concepts,relations,hierarchy,schema}.json
Writes  Jeju_Tourism_Research_Demo/paper_ontology/
          figures/  fig1_class_hierarchy, fig2_full_ontology_network,
                    fig3_relation_schema, fig4_instances_per_class,
                    fig5_relations_per_predicate     (.pdf vector, .svg, .png 600 dpi)
          data/     jeju_tourism_ontology_v1.3.0.{ttl,owl,jsonld}, concepts.csv,
                    relations.csv, class_hierarchy.csv, relation_schema.csv, original_json/
          tables/   ontology_summary, class_statistics, predicate_statistics (.csv + .tex)
          README.md
Nothing is hand-entered: every number comes from the source JSON.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.transforms  # noqa: E402,F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "Jeju_Tourism_Research_Demo"
SRC = DEMO / "ontology" / "original"
OUT = DEMO / "paper_ontology"
FIG, DATA, TAB = OUT / "figures", OUT / "data", OUT / "tables"
VERSION = "1.3.0"  # the version the demo app embeds (bundle.json -> ontology.version)
NAMESPACE = "http://jejunu.ac.kr/ontology/jeju-tourism#"

# ---- colour: 8 categorical groups in the validated fixed order + neutral -----
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#ffffff"
GROUPS = [  # (name, colour, member classes)
    ("Destinations & places", "#2a78d6", ["Destination", "Region", "City", "Island", "Country"]),
    ("Food & dining", "#eb6834", ["Cuisine", "Restaurant", "Market"]),
    ("Natural attractions", "#1baf7a", ["NaturalAttraction", "Beach", "Mountain", "Trail", "Park"]),
    ("Transport", "#eda100", ["TransportMode", "Route"]),
    ("Cultural attractions", "#e87ba4", ["Attraction", "CulturalAttraction", "Museum", "ReligiousSite",
                                         "Landmark", "LocalCustom"]),
    ("Activities & itineraries", "#008300", ["Activity", "Itinerary"]),
    ("Traveller profile & lodging", "#6250d6", ["TravelerType", "TravelConstraint", "DietaryRequirement",
                                                "Accommodation"]),
    ("Safety", "#e34948", ["EmergencyService", "SafetyAdvice"]),
]
NEUTRAL = ("Season & weather / other", "#9a9992")
CLASS_GROUP = {c: (n, col) for n, col, cs in GROUPS for c in cs}


def group_of(cls: str):
    return CLASS_GROUP.get(cls, NEUTRAL)


def colour(cls: str) -> str:
    return group_of(cls)[1]


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def save(fig, name: str):
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print(f"  figures/{name}.pdf|svg|png")


# ------------------------------------------------------------------ load data
def load():
    concepts = json.loads((SRC / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((SRC / "relations.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((SRC / "hierarchy.json").read_text(encoding="utf-8"))
    schema = json.loads((SRC / "schema.json").read_text(encoding="utf-8"))
    return concepts, relations, hierarchy, schema


def class_tree(schema, hierarchy):
    """parent -> [children] over ontology classes.

    The instance paths in hierarchy.json carry the full v1.3.0 tree (31 edges under
    TourismEntity, identical to the app's class hierarchy); schema.json still holds the
    older 14-edge v1.2.0 tree, so it is merged in only as a subset.
    """
    edges = set()
    for h in hierarchy:
        p = h.get("hierarchy_path") or []
        edges.update(zip(p, p[1:]))
    ch = schema.get("class_hierarchy")
    if isinstance(ch, dict):
        for child, parent in ch.items():
            if isinstance(parent, str) and parent and parent != child:
                edges.add((parent, child))
    elif isinstance(ch, list):
        for e in ch:
            if isinstance(e, dict) and e.get("parent") and e.get("child"):
                edges.add((e["parent"], e["child"]))
    if not edges:
        for h in hierarchy:
            p = h.get("hierarchy_path") or []
            edges.update(zip(p, p[1:]))
    kids = defaultdict(list)
    for p, c in sorted(edges):
        kids[p].append(c)
    children = {c for _, c in edges}
    roots = sorted({p for p, _ in edges} - children)
    return kids, roots, sorted(edges)


# ---------------------------------------------------------- figure 1: classes
def fig_hierarchy(kids, roots, inst_direct):
    def total(c):
        return inst_direct.get(c, 0) + sum(total(k) for k in kids.get(c, []))

    pos, y = {}, [0]

    def lay(c, d):
        ks = sorted(kids.get(c, []), key=lambda k: (-total(k), k))
        if not ks:
            pos[c] = (d, y[0]); y[0] += 1
        else:
            for k in ks:
                lay(k, d + 1)
            pos[c] = (d, sum(pos[k][1] for k in ks) / len(ks))
    for r in roots:
        lay(r, 0)

    depth = max(d for d, _ in pos.values())
    BOXW, W = 2.35, 2.9
    fig, ax = plt.subplots(figsize=(2.2 + W * depth, 0.27 * y[0] + 0.5))
    for p, ks in kids.items():
        for k in ks:
            (d0, y0), (d1, y1) = pos[p], pos[k]
            xm = d0 * W + BOXW + (W - BOXW) / 2 - 0.05
            ax.plot([d0 * W + BOXW - 0.05, xm, xm, d1 * W - 0.05], [y0, y0, y1, y1], color=MUTED, lw=0.7,
                    solid_capstyle="round", zorder=1)
    for c, (d, yy) in pos.items():
        dirn, tot = inst_direct.get(c, 0), total(c)
        lab = f"{c}  {dirn}" + (f" / {tot}" if kids.get(c) else "")
        ax.add_patch(FancyBboxPatch((d * W - 0.05, yy - 0.34), BOXW, 0.68,
                                    boxstyle="round,pad=0,rounding_size=0.12", lw=0,
                                    fc=colour(c) if c in CLASS_GROUP else "#d9d8d2", zorder=2, alpha=0.18))
        ax.add_patch(FancyBboxPatch((d * W - 0.05, yy - 0.34), 0.07, 0.68, boxstyle="square,pad=0", lw=0,
                                    fc=colour(c) if c in CLASS_GROUP else INK2, zorder=3))
        ax.text(d * W + 0.08, yy, lab, va="center", ha="left", fontsize=7.4, color=INK, zorder=4,
                fontweight="bold" if kids.get(c) else "normal")
    ax.set_xlim(-0.15, depth * W + BOXW + 0.05)
    ax.set_ylim(y[0] - 0.4, -0.6)
    ax.axis("off")
    ax.set_title("Class hierarchy of the Jeju tourism ontology (v1.3.0)  -  number = direct instances "
                 "/ incl. subclasses", fontsize=8.5, color=INK, loc="left")
    save(fig, "fig1_class_hierarchy")
    return {c: total(c) for c in pos}


# ---------------------------------------------------- figure 2: full network
def clustered_layout(nodes_by_class, centre_key=None, sp=1.0):
    GA = math.pi * (3 - math.sqrt(5))
    rad = lambda k: sp * math.sqrt(k) + 2.2  # noqa: E731
    pos, titles = {}, {}
    ring = sorted([(k, v) for k, v in nodes_by_class.items() if k != centre_key], key=lambda kv: -len(kv[1]))
    circ = sum(2 * rad(len(v)) + 3.2 for _, v in ring)
    centre = nodes_by_class.get(centre_key, [])
    R = max(circ / (2 * math.pi), rad(len(centre)) + rad(len(ring[0][1])) + 6)

    def place(ns, cx, cy):
        for i, n in enumerate(ns):
            r, t = sp * math.sqrt(i + 0.5), i * GA
            pos[n] = (cx + r * math.cos(t), cy + r * math.sin(t))
    if centre:
        place(centre, 0, 0); titles[centre_key] = (0, rad(len(centre)) + 0.4, len(centre), None)
    a = math.pi / 2
    for k, ns in ring:
        r = rad(len(ns)); a -= (r + 1.6) / R
        cx, cy = R * math.cos(a), R * math.sin(a)
        place(ns, cx, cy)
        # title sits radially outside the cluster, so neighbouring titles can never overlap
        titles[k] = ((R + r + 0.8) * math.cos(a), (R + r + 0.8) * math.sin(a), len(ns), a)
        a -= (r + 1.6) / R
    return pos, titles


def fig_network(concepts, relations, class_nodes):
    deg = Counter()
    for r in relations:
        deg[r["subject"]] += 1; deg[r["object"]] += 1
    by_cls = defaultdict(list)
    for c in concepts:
        k = "OntologyClass" if c["name"] in class_nodes else c["concept_class"]
        by_cls[k].append(c["name"])
    for v in by_cls.values():
        v.sort(key=lambda n: (-deg[n], n))
    pos, titles = clustered_layout(by_cls, "OntologyClass")
    ccls = {c["name"]: c["concept_class"] for c in concepts}

    fig, ax = plt.subplots(figsize=(11, 11))
    segs = [(pos[r["subject"]], pos[r["object"]]) for r in relations
            if r["subject"] in pos and r["object"] in pos]
    ax.add_collection(LineCollection(segs, colors=INK2, linewidths=0.12, alpha=0.10, zorder=1))
    names = [n for n in pos]
    xs, ys = [pos[n][0] for n in names], [pos[n][1] for n in names]
    ss = [4 + 2.2 * math.sqrt(deg[n]) for n in names]
    cs = ["#ffffff" if n in class_nodes else colour(ccls.get(n, "")) for n in names]
    ec = [INK if n in class_nodes else SURFACE for n in names]
    ax.scatter(xs, ys, s=ss, c=cs, edgecolors=ec, linewidths=[0.6 if n in class_nodes else 0.25 for n in names],
               zorder=2)
    # fix the view first so rendered label extents can be measured and collisions culled
    ext = [p for p in pos.values()] + [(t[0], t[1]) for t in titles.values()]
    longest = max(len(k) for k in titles) + 5
    pad = 0.62 * longest + 2  # room for the longest radial title
    ax.set_xlim(min(x for x, _ in ext) - pad, max(x for x, _ in ext) + pad)
    ax.set_ylim(min(y for _, y in ext) - pad, max(y for _, y in ext) + pad)
    ax.set_aspect("equal")
    ax.apply_aspect()  # fix the final data->pixel transform before measuring labels
    rend = fig.canvas.get_renderer()
    tlist = []
    for k, (x, y, n, a) in titles.items():
        lab = "Ontology classes" if k == "OntologyClass" else k
        if a is None:
            t = ax.text(x, y, f"{lab} ({n})", ha="center", va="bottom", fontsize=6.8, color=INK,
                        fontweight="bold", zorder=4)
        else:
            deg_a = math.degrees(a) % 360
            left = 90 < deg_a < 270
            t = ax.text(x, y, f"{lab} ({n})", rotation=deg_a - 180 if left else deg_a,
                        rotation_mode="anchor", ha="right" if left else "left", va="center",
                        fontsize=6.8, color=INK, fontweight="bold", zorder=4)
        tlist.append(t)
    # grow the view until every radial title fits inside it, then re-measure
    inv = ax.transData.inverted()
    for _ in range(3):
        ax.apply_aspect()
        pts = [inv.transform(c) for t in tlist for c in t.get_window_extent(rend).corners()]
        x0 = min(min(p[0] for p in pts), min(x for x, _ in pos.values())) - 1.5
        x1 = max(max(p[0] for p in pts), max(x for x, _ in pos.values())) + 1.5
        y0 = min(min(p[1] for p in pts), min(y for _, y in pos.values())) - 1.5
        y1 = max(max(p[1] for p in pts), max(y for _, y in pos.values())) + 1.5
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        inv = ax.transData.inverted()
    ax.apply_aspect()
    boxes = [t.get_window_extent(rend).expanded(1.04, 1.1) for t in tlist]
    shown = 0
    for n in sorted((m for m in names if m not in class_nodes), key=lambda m: -deg[m]):
        if shown >= 12:
            break
        x, y = pos[n]
        t = ax.text(x, y - 0.55, n, ha="center", va="top", fontsize=5.4, color=INK2, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))
        bb = t.get_window_extent(rend).expanded(1.2, 2.0)  # include the label's background box
        if any(bb.overlaps(o) for o in boxes):
            t.remove(); continue
        boxes.append(bb); shown += 1
    handles = [Patch(fc=c, label=n) for n, c, _ in GROUPS] + [Patch(fc=NEUTRAL[1], label=NEUTRAL[0]),
               Line2D([], [], marker="o", ls="", mfc="#ffffff", mec=INK, label="Ontology class node")]
    ax.axis("off")
    # place the heading above and the legend below the measured extent of everything drawn
    # (radial titles may reach outside the axes box), so neither can overlap a cluster title
    allbb = matplotlib.transforms.Bbox.union([t.get_window_extent(rend) for t in tlist] + [ax.get_window_extent(rend)])
    ftr = fig.transFigure.inverted()
    (_, ytop), (_, ybot) = ftr.transform((0, allbb.y1)), ftr.transform((0, allbb.y0))
    xl, _ = ftr.transform((allbb.x0, 0))
    fig.text(xl, ytop + 0.012, f"Complete Jeju tourism ontology v1.3.0: all {len(concepts):,} concepts and "
             f"{len(relations):,} relations, clustered by class (node size = degree)",
             fontsize=9, color=INK, ha="left", va="bottom")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, ybot - 0.008), ncol=5, frameon=False,
               fontsize=6.8, handlelength=1.2, columnspacing=1.2)
    save(fig, "fig2_full_ontology_network")


# ------------------------------------------------------ figure 3: rel schema
def fig_schema(schema, pred_counts):
    cons = schema.get("relation_constraints") or {}
    preds = sorted(cons, key=lambda p: (-pred_counts.get(p, 0), p))
    dom = sorted({c for p in preds for c in cons[p].get("subject_classes", [])})
    rng = sorted({c for p in preds for c in cons[p].get("object_classes", [])})
    order = {c: i for i, (n, _, cs) in enumerate(GROUPS) for c in cs}
    dom.sort(key=lambda c: (order.get(c, 99), c)); rng.sort(key=lambda c: (order.get(c, 99), c))
    H = max(len(dom), len(rng), len(preds))
    yl = lambda items: {c: (H - 1) * (i + 0.5) / len(items) for i, c in enumerate(items)}  # noqa: E731
    yd, yp, yr = yl(dom), yl(preds), yl(rng)
    X = (0, 4.0, 8.7)
    fig, ax = plt.subplots(figsize=(8.6, 0.24 * H + 1.0))
    for p in preds:
        for c in cons[p].get("subject_classes", []):
            ax.plot([X[0] + 1.75, X[1] - 0.05], [yd[c], yp[p]], color=colour(c), lw=0.55, alpha=0.55, zorder=1)
        for c in cons[p].get("object_classes", []):
            ax.plot([X[1] + 2.45, X[2] - 0.05], [yp[p], yr[c]], color=colour(c), lw=0.55, alpha=0.55, zorder=1)

    def box(x, y, text, fc, w):
        ax.add_patch(FancyBboxPatch((x, y - 0.36), w, 0.72, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=fc, ec="none", alpha=0.20, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y - 0.36), 0.07, 0.72, boxstyle="square,pad=0", fc=fc, ec="none",
                                    zorder=3))
        ax.text(x + 0.14, y, text, va="center", ha="left", fontsize=6.8, color=INK, zorder=4)
    for c in dom:
        box(X[0], yd[c], c, colour(c), 1.75)
    for p in preds:
        box(X[1], yp[p], f"{p}  ({pred_counts.get(p, 0):,})", "#52514e", 2.45)
    for c in rng:
        box(X[2], yr[c], c, colour(c), 1.75)
    for x, t in zip(X, ("Domain (subject classes)", "Relation type  (asserted triples)", "Range (object classes)")):
        ax.text(x, -1.1, t, fontsize=7.6, fontweight="bold", color=INK, ha="left")
    ax.set_xlim(-0.1, X[2] + 1.85); ax.set_ylim(H, -1.6); ax.axis("off")
    ax.set_title(f"Relation schema of the Jeju tourism ontology: the {len(preds)} relation types with a declared "
                 f"domain and range ({sum(pred_counts.get(p, 0) for p in preds):,} of {sum(pred_counts.values()):,} "
                 "asserted triples)",
                 fontsize=8.5, color=INK, loc="left", pad=14)
    save(fig, "fig3_relation_schema")
    return cons


# ------------------------------------------------ figures 4-5: distributions
def fig_bars(items, name, title, xlabel, colours=None, legend=False):
    items = sorted(items, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 0.19 * len(items) + 0.9))
    ys = range(len(items))
    ax.barh(list(ys), [v for _, v in items], height=0.62, color=colours(items) if colours else "#2a78d6",
            edgecolor=SURFACE, linewidth=0.6)
    ax.set_yticks(list(ys), [k for k, _ in items], fontsize=7)
    vmax = max(v for _, v in items)
    for y, (_, v) in zip(ys, items):
        ax.text(v + vmax * 0.008, y, f"{v:,}", va="center", fontsize=6.4, color=INK2)
    ax.set_xlim(0, vmax * 1.09)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.xaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    if legend:
        present = {group_of(k)[0] for k, _ in items}
        hs = [Patch(fc=c, label=n) for n, c, _ in GROUPS if n in present]
        if NEUTRAL[0] in present:
            hs.append(Patch(fc=NEUTRAL[1], label=NEUTRAL[0]))
        ax.legend(handles=hs, loc="lower right", frameon=False, fontsize=6.4)
    save(fig, name)


# ------------------------------------------------------------- data exports
def export_rdf(concepts, relations, tree_edges, cons, schema):
    from rdflib import BNode, Graph, Literal, Namespace, URIRef
    from rdflib.collection import Collection
    from rdflib.namespace import DCTERMS, OWL, PROV, RDF, RDFS, SKOS, XSD

    # the project's official namespace (as in wbs/deliverables/task2_formalization/*.ttl);
    # schema.json still carries the generic namespace and schema version of v1.2.0
    ns = NAMESPACE
    J = Namespace(ns)
    safe = lambda s: URIRef(ns + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s.strip()))  # noqa: E731
    g = Graph()
    for p, n in (("jeju", J), ("owl", OWL), ("rdfs", RDFS), ("skos", SKOS), ("prov", PROV),
                 ("dcterms", DCTERMS), ("xsd", XSD)):
        g.bind(p, n)
    onto = URIRef(ns.rstrip("#/"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, DCTERMS.title, Literal("Jeju Tourism Ontology")))
    g.add((onto, OWL.versionInfo, Literal(VERSION)))
    g.add((onto, DCTERMS.creator, Literal("NLP-Lab, Jeju National University")))

    classes = {c for e in tree_edges for c in e} | {c["concept_class"] for c in concepts}
    for c in classes:
        g.add((safe(c), RDF.type, OWL.Class)); g.add((safe(c), RDFS.label, Literal(c)))
    for p, c in tree_edges:
        g.add((safe(c), RDFS.subClassOf, safe(p)))
    for k in ("confidence", "lat", "lon"):
        g.add((J[k], RDF.type, OWL.DatatypeProperty))

    def cls_expr(cs):
        if len(cs) == 1:
            return safe(cs[0])
        b, lst = BNode(), BNode()
        g.add((b, RDF.type, OWL.Class)); Collection(g, lst, [safe(c) for c in cs]); g.add((b, OWL.unionOf, lst))
        return b
    preds = {r["predicate"] for r in relations} | set(cons)
    for p in preds:
        u = safe(p)
        g.add((u, RDF.type, OWL.ObjectProperty)); g.add((u, RDFS.label, Literal(p.replace("_", " "))))
        if p in cons:
            if cons[p].get("subject_classes"):
                g.add((u, RDFS.domain, cls_expr(sorted(cons[p]["subject_classes"]))))
            if cons[p].get("object_classes"):
                g.add((u, RDFS.range, cls_expr(sorted(cons[p]["object_classes"]))))
    for p in ("near", "connected_to"):
        if p in preds:
            g.add((safe(p), RDF.type, OWL.SymmetricProperty))

    for c in concepts:
        u = safe(c["name"])
        g.add((u, RDF.type, OWL.NamedIndividual)); g.add((u, RDF.type, safe(c["concept_class"])))
        g.add((u, RDFS.label, Literal(c["name"]))); g.add((u, SKOS.prefLabel, Literal(c["name"])))
        for a in c.get("aliases") or []:
            g.add((u, SKOS.altLabel, Literal(a)))
        for k, v in (c.get("attributes") or {}).items():
            if isinstance(v, (int, float)):
                g.add((u, J[k], Literal(v, datatype=XSD.double)))
            elif v not in (None, ""):
                g.add((u, J[k], Literal(str(v))))
        pv = c.get("provenance") or {}
        if pv.get("confidence") is not None:
            g.add((u, J.confidence, Literal(float(pv["confidence"]), datatype=XSD.double)))
        if pv.get("extraction_prompt_version"):
            g.add((u, PROV.wasGeneratedBy, Literal(pv["extraction_prompt_version"])))
    for r in relations:
        g.add((safe(r["subject"]), safe(r["predicate"]), safe(r["object"])))

    base = DATA / f"jeju_tourism_ontology_v{VERSION}"  # not with_suffix: it would eat the ".0"
    g.serialize(f"{base}.ttl", format="turtle")
    g.serialize(f"{base}.owl", format="xml")  # pretty-xml drops owl:unionOf lists
    g.serialize(f"{base}.jsonld", format="json-ld", indent=1)
    print(f"  data/jeju_tourism_ontology_v1.3.0.ttl|owl|jsonld  ({len(g):,} triples)")
    return len(g)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def tex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_tex(path, caption, label, header, rows, align):
    lines = [r"% generated by experiments/exp041_demo_traces/export_paper_ontology.py",
             r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(tex_escape(f"{c:,}" if isinstance(c, int) else c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    for d in (FIG, DATA, TAB, DATA / "original_json"):
        d.mkdir(parents=True, exist_ok=True)
    concepts, relations, hierarchy, schema = load()
    kids, roots, tree_edges = class_tree(schema, hierarchy)
    class_names = {c for e in tree_edges for c in e}
    inst_direct = Counter(c["concept_class"] for c in concepts if c["name"] not in class_names)
    pred_counts = Counter(r["predicate"] for r in relations)
    deg = Counter()
    for r in relations:
        deg[r["subject"]] += 1; deg[r["object"]] += 1

    print("figures")
    totals = fig_hierarchy(kids, roots, inst_direct)
    fig_network(concepts, relations, class_names)
    cons = fig_schema(schema, pred_counts)
    fig_bars(list(inst_direct.items()), "fig4_instances_per_class",
             f"Instances per class ({sum(inst_direct.values()):,} instances, {len(inst_direct)} classes)",
             "Number of instances", colours=lambda it: [colour(k) for k, _ in it], legend=True)
    fig_bars(list(pred_counts.items()), "fig5_relations_per_predicate",
             f"Asserted relations per relation type ({len(relations):,} relations, {len(pred_counts)} types)",
             "Number of asserted triples")

    print("data")
    n_triples = export_rdf(concepts, relations, tree_edges, cons, schema)
    write_csv(DATA / "concepts.csv", ["name", "class", "is_class_node", "degree", "aliases", "lat", "lon",
                                      "confidence", "source", "validator_status"],
              [[c["name"], c["concept_class"], c["name"] in class_names, deg[c["name"]],
                "; ".join(c.get("aliases") or []), (c.get("attributes") or {}).get("lat", ""),
                (c.get("attributes") or {}).get("lon", ""), (c.get("provenance") or {}).get("confidence", ""),
                (c.get("provenance") or {}).get("extraction_prompt_version", ""),
                (c.get("provenance") or {}).get("validator_status", "")] for c in concepts])
    write_csv(DATA / "relations.csv", ["subject", "predicate", "object", "confidence", "source", "validator_status"],
              [[r["subject"], r["predicate"], r["object"], (r.get("provenance") or {}).get("confidence", ""),
                (r.get("provenance") or {}).get("extraction_prompt_version", ""),
                (r.get("provenance") or {}).get("validator_status", "")] for r in relations])
    write_csv(DATA / "class_hierarchy.csv", ["parent_class", "child_class"], tree_edges)
    write_csv(DATA / "relation_schema.csv", ["relation_type", "domain_classes", "range_classes", "asserted_triples"],
              [[p, "; ".join(v.get("subject_classes", [])), "; ".join(v.get("object_classes", [])),
                pred_counts.get(p, 0)] for p, v in sorted(cons.items())])
    for f in ("concepts.json", "relations.json", "hierarchy.json", "schema.json", "rules.json"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DATA / "original_json" / f)
    print("  data/concepts.csv, relations.csv, class_hierarchy.csv, relation_schema.csv, original_json/")

    print("tables")
    depth = {}
    def walk(c, d):
        depth[c] = d
        for k in kids.get(c, []):
            walk(k, d + 1)
    for r in roots:
        walk(r, 0)
    summary = [("Ontology version", VERSION),
               ("Concepts (graph nodes)", len(concepts)),
               ("Instances (excluding class nodes)", sum(inst_direct.values())),
               ("Ontology classes", len(class_names)),
               ("Subclass edges / max. depth", f"{len(tree_edges)} / {max(depth.values())}"),
               ("Relation types used", len(pred_counts)),
               ("Relation types declared in schema", len(schema.get("relation_types") or [])),
               ("Asserted relations", len(relations)),
               ("Concepts with geo-coordinates", sum(1 for c in concepts if (c.get("attributes") or {}).get("lat") is not None)),
               ("Concepts with aliases", sum(1 for c in concepts if c.get("aliases"))),
               ("Mean degree", f"{2 * len(relations) / len(concepts):.2f}"),
               ("RDF triples (Turtle export)", n_triples)]
    write_csv(TAB / "ontology_summary.csv", ["metric", "value"], summary)
    write_tex(TAB / "ontology_summary.tex", "Summary statistics of the Jeju tourism ontology (v1.3.0).",
              "tab:ontology-summary", ["Metric", "Value"], summary, "lr")
    parent = {c: p for p, c in tree_edges}
    crow = [(c, parent.get(c, "-"), depth.get(c, 0), inst_direct.get(c, 0), totals.get(c, 0))
            for c in sorted(class_names, key=lambda c: (-totals.get(c, 0), c))]
    write_csv(TAB / "class_statistics.csv", ["class", "superclass", "depth", "direct_instances",
                                             "instances_incl_subclasses"], crow)
    write_tex(TAB / "class_statistics.tex", "Classes of the Jeju tourism ontology with instance counts.",
              "tab:ontology-classes", ["Class", "Superclass", "Depth", "Direct", "Incl. sub."], crow, "llrrr")
    prow = [(p, pred_counts.get(p, 0), "; ".join(cons.get(p, {}).get("subject_classes", [])) or "-",
             "; ".join(cons.get(p, {}).get("object_classes", [])) or "-")
            for p in sorted(pred_counts, key=lambda p: -pred_counts[p])]
    write_csv(TAB / "predicate_statistics.csv", ["relation_type", "asserted_triples", "domain", "range"], prow)
    write_tex(TAB / "predicate_statistics.tex", "Relation types with asserted-triple counts and declared domain/range.",
              "tab:ontology-relations", ["Relation", "Triples", "Domain", "Range"],
              prow, r"lrp{4.2cm}p{4.2cm}")
    print("  tables/ontology_summary, class_statistics, predicate_statistics (.csv + .tex)")
    return summary


if __name__ == "__main__":
    main()
