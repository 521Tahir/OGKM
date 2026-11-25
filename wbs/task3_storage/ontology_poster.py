"""
WBS Task 4.2.2 - the entire ontology in one frame.

Every other view of this ontology is partial: the schema graph abstracts the
instances away, the class browser shows one neighbourhood at a time, and the
Neo4j Browser draws whatever a query returned. This draws all of it at once -
every concept, every relation - as a single image.

Two layouts, because they answer different questions and the honest thing is to
offer both rather than pick one and call it the ontology:

  force      - the organic shape the graph actually has. One node, `Jeju`, holds
               about a fifth of all edges, and a force layout shows that
               truthfully: a dense core with class-coloured fringes.
  clustered  - every class given its own position on a ring, instances placed
               around their class by degree. Nothing overlaps, every class is
               findable, and the cost is that it imposes a shape the data does
               not have.

The layout is computed once, seeded, and written to JSON so the interactive
explorer draws the identical picture instead of re-deriving a different one.

The WBS names Neo4j as the ontology store, so the default source is the graph as
Cypher returns it: `neo4j_export` reads every node and relationship back off the
running server and verifies the round trip, and these figures are drawn from
those rows. `--source files` reads the versioned file store instead, for a clone
with no server. The subtitle records which one was used, so no printed figure can
claim a provenance it does not have.

  python -m wbs.task3_storage.ontology_poster              # both, from Neo4j
  python -m wbs.task3_storage.ontology_poster --layout force --dpi 200
  python -m wbs.task3_storage.ontology_poster --source files
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# This host is shared and busy. The matmul below is tiny but BLAS will happily
# spawn a thread per core and then spin-wait on contention: measured 13 s of user
# time against 5 m 46 s of system time until these were pinned.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
FIGS = ROOT / "wbs" / "deliverables" / "figures"
STORE = ROOT / "wbs" / "deliverables" / "task3_storage"

HIER_PREDS = {"is_a", "subclass_of"}

# A 33-class legend cannot be read by hue alone. The twelve largest classes get a
# distinct colour; everything else is grey and named in the legend with its count.
PALETTE = [
    "#1f4e79", "#2e7d5b", "#c98a2b", "#8e44ad", "#b03a4a", "#2b7a8c",
    "#6b7b2a", "#a0522d", "#4a69bd", "#117a65", "#7d3c98", "#cb4335",
    "#e07b39", "#167d7f", "#9b59b6", "#52796f", "#d64550", "#3d6b9c",
    "#8a6d3b", "#1b998b", "#c2185b", "#4e6e58", "#7b5ea7", "#b5651d",
]
GREY = "#9aa4b2"


# --------------------------------------------------------------------- loading


def load(version: str, source: str = "neo4j"):
    """Concepts and edges, from the Neo4j snapshot or from the file store.

    Returns the Neo4j metadata as well, so the caption can name the server the
    numbers came from rather than leaving the reader to assume it.
    """
    meta = None
    if source == "neo4j":
        from wbs.task3_storage.neo4j_export import load_snapshot
        concepts, relations, snap = load_snapshot()
        meta = snap["meta"]
        meta["verified_exact"] = bool((snap.get("verification") or {}).get("exact"))
    else:
        cdir = ONTO / version
        concepts = json.loads((cdir / "concepts.json").read_text(encoding="utf-8"))
        relations = json.loads((cdir / "relations.json").read_text(encoding="utf-8"))
    idx = {c["name"]: i for i, c in enumerate(concepts)}
    edges, hier = [], []
    for r in relations:
        s, o = idx.get(r["subject"]), idx.get(r["object"])
        if s is None or o is None:
            continue
        edges.append((s, o))
        hier.append(r["predicate"] in HIER_PREDS)
    return (concepts, np.array(edges, dtype=np.int32),
            np.array(hier, dtype=bool), idx, meta)


# --------------------------------------------------------------------- layouts


def force_layout(n, edges, iters=300, seed=7):
    """ForceAtlas2-flavoured layout: repulsion as a matmul, into fixed buffers.

    Two things make this fast enough to run on a shared host.

    1. The all-pairs repulsion sum factorises, so it needs no (n, n, 2) tensor:

           F_i = sum_j m_i m_j (p_i - p_j) / d_ij^3
               = p_i * rowsum(K)_i  -  (K @ p)_i,    K_ij = m_i m_j / d_ij^3

       One (n, n) matrix and one BLAS call per step instead of six 39 MB
       temporaries.

    2. Every array is allocated once and written in place. The obvious version
       churns about 40 MB per iteration, which is free on an idle box and very
       much not free here: this machine runs at a load average near 30 with
       ~35 GB of 1 TB available, and the allocation traffic turned into page
       reclaim - 6 s of user time against 8 minutes of system time.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n, 2)).astype(np.float32)
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float32)
    mass = deg + 1.0
    src, dst = edges[:, 0], edges[:, 1]
    # An edge into a hub constrains position less than an edge to a specific
    # neighbour, so it pulls less; unweighted, `Jeju` drags everything onto itself.
    w = (1.0 / np.sqrt(mass[src] * mass[dst])).astype(np.float32)
    mm = np.outer(mass, mass).astype(np.float32)

    D = np.empty((n, n), dtype=np.float32)      # pairwise squared distance, then K
    T = np.empty((n, n), dtype=np.float32)      # scratch for d^3
    F = np.empty((n, 2), dtype=np.float32)
    KP = np.empty((n, 2), dtype=np.float32)
    sq = np.empty(n, dtype=np.float32)
    rowsum = np.empty(n, dtype=np.float32)
    gravity = (0.012 * mass).astype(np.float32)[:, None]

    for it in range(iters):
        cool = 1.0 - it / iters
        np.einsum("ij,ij->i", P, P, out=sq)
        np.dot(P, P.T, out=D)
        D *= -2.0
        D += sq[:, None]
        D += sq[None, :]
        np.maximum(D, 1e-4, out=D)
        np.sqrt(D, out=T)
        T *= D                                   # T = d^3
        np.divide(mm, T, out=D)                  # D is now K
        np.fill_diagonal(D, 0.0)
        D.sum(axis=1, out=rowsum)
        np.dot(D, P, out=KP)
        np.multiply(P, rowsum[:, None], out=F)
        F -= KP
        F *= 0.22

        delta = P[dst] - P[src]
        pull = delta * (w * 1.6)[:, None]
        np.add.at(F, src, pull)
        np.add.at(F, dst, -pull)

        F -= P * gravity                         # keeps it one drawing, not many
        step = np.linalg.norm(F, axis=1) + 1e-9
        F *= (np.minimum(10.0 * cool + 0.5, step) / step)[:, None]
        P += F

    # Normalise BEFORE de-overlapping: the separation radii below are in units of
    # the finished picture, not of the layout's arbitrary scale. The other order
    # asks a KD-tree for every pair within a radius that spans the whole cloud.
    P = P.astype(np.float64)
    P -= P.mean(0)
    P /= np.abs(P).max()

    from scipy.spatial import cKDTree
    r = 0.004 + 0.016 * np.sqrt(deg / max(float(deg.max()), 1.0))
    for _ in range(25):
        pairs = cKDTree(P).query_pairs(float(r.max()) * 2.0, output_type="ndarray")
        if len(pairs) == 0:
            break
        a_, b_ = pairs[:, 0], pairs[:, 1]
        d = P[b_] - P[a_]
        dist = np.sqrt((d ** 2).sum(1)) + 1e-12
        want = r[a_] + r[b_]
        hit = dist < want
        if not hit.any():
            break
        a_, b_, d, dist, want = a_[hit], b_[hit], d[hit], dist[hit], want[hit]
        push = (d / dist[:, None]) * ((want - dist) * 0.5)[:, None]
        np.add.at(P, a_, -push)
        np.add.at(P, b_, push)

    P -= P.mean(0)
    return P / np.abs(P).max()


def clustered_layout(concepts, classes, seed=7):
    """One ring position per class, instances packed inside it.

    Every coordinate - instance, class centre, cluster radius - is normalised by
    the same scale at the end. Normalising only the instances is what put the
    class captions in empty space on the first attempt, naming clusters that sat
    somewhere else entirely.

    A class's disc radius is tied to the angular slot it owns rather than to its
    size alone, so neighbouring classes cannot grow into each other.
    """
    rng = np.random.default_rng(seed)
    counts = Counter(c.get("concept_class") or "Concept" for c in concepts)
    order = sorted(classes, key=lambda c: -counts[c])
    ring = 1.0

    # Share the ring by sqrt(size), but with a floor: on size alone the nine
    # smallest classes land in one narrow arc and their captions stack.
    raw = {c: math.sqrt(counts[c]) for c in order}
    tot_raw = sum(raw.values())
    floor = 1.0 / (len(order) * 2.0)
    blended = {c: 0.55 * raw[c] / tot_raw + 0.45 * floor * 2.0 / len(order) * 10
               for c in order}
    total = sum(blended.values())
    centres, radii, acc = {}, {}, 0.0
    for c in order:
        share = blended[c] / total
        angle = 2 * math.pi * (acc + share / 2)
        acc += share
        centres[c] = np.array([math.cos(angle), math.sin(angle)]) * ring
        slot = 2 * math.pi * ring * share          # arc length this class owns
        radii[c] = min(0.42, max(0.045, slot * 0.40))

    by_class = defaultdict(list)
    for i, c in enumerate(concepts):
        by_class[c.get("concept_class") or "Concept"].append(i)

    pos = np.zeros((len(concepts), 2))
    for c, members in by_class.items():
        cx, cy = centres[c]
        k = max(1, len(members))
        rad = radii[c]
        for j, i in enumerate(members):
            t = (j + 0.5) / k                      # sunflower: even density
            r = rad * math.sqrt(t)
            ang = j * 2.399963 + rng.uniform(-0.04, 0.04)
            pos[i] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

    shift = pos.mean(0)
    pos -= shift
    scale = np.abs(pos).max()
    pos /= scale
    # Same shift, same scale - otherwise the captions describe a different picture.
    centres = {c: (v - shift) / scale for c, v in centres.items()}
    radii = {c: r / scale for c, r in radii.items()}
    return pos, {"centres": centres, "radii": radii}


# ---------------------------------------------------------------------- drawing


def draw(concepts, edges, hier, pos, classes, class_colour, title, subtitle,
         out_png, out_svg=None, dpi=200, label_top=55, centres=None,
         figsize=(26, 19), provenance=None):
    n = len(concepts)
    deg = np.bincount(edges.ravel(), minlength=n)
    cls = [c.get("concept_class") or "Concept" for c in concepts]
    counts = Counter(cls)

    pad = 0.05
    x0, x1 = pos[:, 0].min() - pad, pos[:, 0].max() + pad
    y0, y1 = pos[:, 1].min() - pad, pos[:, 1].max() + pad
    if centres:
        reach = max(float(np.linalg.norm(v)) + centres["radii"][c] + 0.17
                    for c, v in centres["centres"].items())
        x0 = y0 = -reach
        x1 = y1 = reach
    data_aspect = (x1 - x0) / (y1 - y0)

    fig = plt.figure(figsize=figsize, facecolor="white")
    # Reserve the right-hand strip for the legend, then give the plot exactly the
    # width its own aspect needs - no more, so the drawing fills its frame.
    # The provenance line needs its own band above the plot; without lowering the
    # top the text would be drawn over the first row of nodes.
    top, bottom = (0.932 if provenance else 0.945), 0.012
    box_h = top - bottom
    box_w = min(0.80, data_aspect * box_h * figsize[1] / figsize[0])
    ax = fig.add_axes([0.008, bottom, box_w, box_h])
    legend_x = 0.012 + box_w + 0.012
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Edges: hierarchy under lateral relations, both faint enough to read through.
    for mask, colour, lw, alpha, z in (
            (hier, "#b9c2cf", 0.35, 0.30, 1),
            (~hier, "#7f8fa6", 0.30, 0.16, 2)):
        if not mask.any():
            continue
        seg = np.stack([pos[edges[mask, 0]], pos[edges[mask, 1]]], axis=1)
        ax.add_collection(LineCollection(seg, colors=colour, linewidths=lw,
                                         alpha=alpha, zorder=z))

    sizes = 6 + 110 * np.sqrt(deg / max(deg.max(), 1))
    colours = [class_colour[c] for c in cls]
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colours, linewidths=0.25,
               edgecolors="white", zorder=3)

    # Labels for the hubs only, nudged apart so they do not stack. In the
    # clustered view the class captions carry the naming, so hub labels there
    # would be a second, competing set of text over the same ink.
    top = np.argsort(-deg)[:(0 if centres else label_top)]
    placed = []
    for i in top:
        x, y = pos[i]
        dy = 0.012 + sizes[i] / 9000
        for _ in range(14):
            if all(abs(x - px) > 0.055 or abs(y + dy - py) > 0.013
                   for px, py in placed):
                break
            dy += 0.014
        placed.append((x, y + dy))
        # The handful of hubs that organise the whole picture are set larger, so
        # the eye has somewhere to start before reading the fringe.
        rank = list(top).index(i)
        big = rank < 8
        ax.text(x, y + dy, concepts[i]["name"],
                fontsize=11.5 if big else 7.6, ha="center", va="bottom",
                zorder=6 if big else 4, color="#15181d",
                weight="bold" if big else "normal",
                bbox=dict(boxstyle="round,pad=0.18" if big
                          else "round,pad=0.12",
                          fc="white", ec="#c3cbd6" if big else "none",
                          lw=0.7, alpha=0.9 if big else 0.72))

    if centres:
        cs, rs = centres["centres"], centres["radii"]
        # Captions go radially outside their own disc. Adjacent small classes
        # share almost the same angle, so resolve collisions by pushing the
        # later one further out rather than letting the boxes stack.
        tips, order_by_angle = {}, sorted(
            cs, key=lambda c: math.atan2(cs[c][1], cs[c][0]))
        for c in order_by_angle:
            ctr = cs[c]
            u = ctr / (np.linalg.norm(ctr) or 1.0)
            out = rs[c] + 0.075
            for _ in range(24):
                tip = ctr + u * out
                if all(np.linalg.norm(tip - t) > 0.115 for t in tips.values()):
                    break
                out += 0.045
            tips[c] = ctr + u * out
        for c, ctr in cs.items():
            tip = tips[c]
            ax.text(tip[0], tip[1], f"{c}\n{counts[c]:,}",
                    fontsize=9.8, ha="center", va="center", zorder=6,
                    color=class_colour[c], weight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=class_colour[c], lw=0.9, alpha=0.96))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    fig.text(0.012, 0.975, title, fontsize=27, weight="bold", color="#15181d")
    fig.text(0.012, 0.953, subtitle, fontsize=13, color="#5b6471")
    if provenance:
        fig.text(0.012, 0.938, provenance, fontsize=11.5, color="#7a8494",
                 style="italic")

    # Legend: every class, with its instance count, so nothing is unaccounted for.
    lx = legend_x
    fig.text(lx, 0.935, f"{len(classes)} classes", fontsize=12.5, weight="bold",
             color="#15181d")
    handles = [Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=class_colour[c], markeredgecolor="white",
                      markersize=9, label=f"{c}  ({counts[c]:,})")
               for c in sorted(classes, key=lambda c: -counts[c])]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(lx, 0.925), frameon=False, fontsize=10.4,
                     labelspacing=0.52, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color("#15181d")

    top_pred = Counter()
    fig.text(lx, 0.085, "Edge colour", fontsize=11.5, weight="bold",
             color="#15181d")
    fig.legend(handles=[
        Line2D([], [], color="#b9c2cf", lw=2,
               label=f"is_a / subclass_of  ({int(hier.sum()):,})"),
        Line2D([], [], color="#7f8fa6", lw=2,
               label=f"other predicates  ({int((~hier).sum()):,})")],
        loc="upper left", bbox_to_anchor=(lx, 0.078), frameon=False,
        fontsize=10.4, labelspacing=0.45)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches=None, pad_inches=0)
    if out_svg:
        fig.savefig(out_svg, facecolor="white", pad_inches=0)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--layout", default="both",
                    choices=["both", "force", "clustered"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--svg", action="store_true",
                    help="also write a vector copy, for zooming without limit")
    ap.add_argument("--source", default="neo4j", choices=("neo4j", "files"),
                    help="neo4j draws the graph as Cypher returns it, from the "
                         "snapshot written by neo4j_export; files draws the "
                         "versioned file store")
    a = ap.parse_args()

    concepts, edges, hier, _, meta = load(a.version, a.source)
    classes = sorted({c.get("concept_class") or "Concept" for c in concepts})
    counts = Counter(c.get("concept_class") or "Concept" for c in concepts)
    ranked = [c for c, _ in counts.most_common()]
    class_colour = {c: (PALETTE[i] if i < len(PALETTE) else GREY)
                    for i, c in enumerate(ranked)}

    deg = np.bincount(edges.ravel(), minlength=len(concepts))
    hub = int(np.argmax(deg))
    # The provenance belongs in the picture: a figure that travels into a report
    # has to carry where its numbers came from without the surrounding text.
    if meta:
        engine = meta["engine"].replace("Neo4j Kernel ", "Neo4j ")
        origin = (f"read from {engine} at {meta['bolt']}"
                  + (", round trip verified" if meta.get("verified_exact") else ""))
    else:
        origin = "read from the versioned file store"
    sub = (f"v{a.version} · {len(concepts):,} concepts · "
           f"{len(edges):,} relations · {len(classes)} classes · "
           f"every node and every edge drawn · "
           f"largest hub: {concepts[hub]['name']} ({deg[hub]:,} edges)")

    wrote = []
    layouts = (["force", "clustered"] if a.layout == "both" else [a.layout])
    cache = {}
    for kind in layouts:
        if kind == "force":
            print(f"computing force layout over {len(concepts):,} nodes "
                  f"({a.iters} iterations)...", flush=True)
            pos = force_layout(len(concepts), edges, iters=a.iters)
            centres = None
            title = "Jeju Tourism Ontology — the whole graph"
            png = FIGS / "fig17_ontology_whole.png"
        else:
            pos, centres = clustered_layout(concepts, classes)   # dict payload
            title = "Jeju Tourism Ontology — every class, every instance"
            png = FIGS / "fig18_ontology_by_class.png"
        out = draw(concepts, edges, hier, pos, classes, class_colour, title, sub,
                   png, png.with_suffix(".svg") if a.svg else None,
                   dpi=a.dpi, centres=centres, provenance=origin)
        cache[kind] = [[round(float(x), 4), round(float(y), 4)] for x, y in pos]
        mb = out.stat().st_size / 1e6
        print(f"wrote {out.relative_to(ROOT)} ({mb:.1f} MB, "
              f"{int(out.stat().st_size and a.dpi)} dpi)")
        wrote.append(out)

    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "ontology_layout.json").write_text(json.dumps({
        "version": a.version, "seed": 7, "iterations": a.iters,
        "source": a.source, "neo4j": meta,
        "order": [c["name"] for c in concepts], **cache},
        separators=(",", ":")), encoding="utf-8")
    print(f"wrote {(STORE / 'ontology_layout.json').relative_to(ROOT)} "
          f"({', '.join(cache)} coordinates, for the interactive explorer)")


if __name__ == "__main__":
    main()
