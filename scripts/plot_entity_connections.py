"""
Plot 2 — Entity Connection Map (slide-ready).

Two sub-figures:
  LEFT  : Full schema-level connection graph (classes as nodes, all
          constrained + hierarchy relations as directed edges, labelled)
  RIGHT : Zoomed semantic cluster showing how a real concept
          (e.g., "Hallasan") connects to its neighbours
"""
from __future__ import annotations
import json, sys, math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

try:
    import networkx as nx
except ImportError:
    print("Installing networkx …")
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","networkx","-q"])
    import networkx as nx

ROOT    = Path(__file__).resolve().parents[1]
ONT_DIR = ROOT / "artifacts" / "ontologies" / "0.7.0"
OUTDIR  = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(ONT_DIR / "schema.json")    as f: schema   = json.load(f)
with open(ONT_DIR / "concepts.json")  as f: concepts = json.load(f)
with open(ONT_DIR / "relations.json") as f: rels     = json.load(f)

HIER   = schema["class_hierarchy"]
CONSTR = schema["relation_constraints"]
cls_counts = Counter(c["concept_class"] for c in concepts)

# ── Group colours ────────────────────────────────────────────────────────────
GROUPS = {
    "Geography":  (["Destination","Region","Country","City","Island"],
                   "#1565C0","#90CAF9","#E3F2FD"),
    "Attraction": (["Attraction","NaturalAttraction","CulturalAttraction",
                    "Beach","Mountain","Trail","Park","Landmark",
                    "ReligiousSite","Museum","Market"],
                   "#2E7D32","#A5D6A7","#E8F5E9"),
    "Services":   (["Accommodation","Restaurant","Cuisine",
                    "DietaryRequirement","TransportMode","Route"],
                   "#6A1B9A","#CE93D8","#F3E5F5"),
    "Experience": (["Activity","Itinerary","TravelerType","Season"],
                   "#E65100","#FFCC80","#FFF3E0"),
    "Safety":     (["SafetyAdvice","EmergencyService","TravelConstraint",
                    "WeatherCondition","LocalCustom"],
                   "#C62828","#EF9A9A","#FFEBEE"),
}
CLS2DARK = {}; CLS2MID = {}; CLS2LIGHT = {}
for grp,(classes,dark,mid,light) in GROUPS.items():
    for cls in classes:
        CLS2DARK[cls]=dark; CLS2MID[cls]=mid; CLS2LIGHT[cls]=light

# ══════════════════════════════════════════════════════════════════════════════
# Build schema-level directed graph
# ══════════════════════════════════════════════════════════════════════════════
G = nx.MultiDiGraph()

# 1) All concept classes as nodes
for cls in schema["concept_classes"]:
    G.add_node(cls, group=next((g for g,(cs,*_) in GROUPS.items() if cls in cs),"Other"),
               count=cls_counts.get(cls,0))

# 2) is_a hierarchy edges
for child, parent in HIER.items():
    G.add_edge(child, parent, rel="is_a", style="hierarchy")

# 3) Constrained relation edges (domain→range, one representative per pair)
added = set()
for rel_name, v in CONSTR.items():
    for subj in v["subject_classes"]:
        for obj in v["object_classes"]:
            key = (subj, obj, rel_name)
            if key not in added and subj in G and obj in G:
                G.add_edge(subj, obj, rel=rel_name, style="constrained")
                added.add(key)

# ══════════════════════════════════════════════════════════════════════════════
# Layout: group-based manual seed for spring layout
# ══════════════════════════════════════════════════════════════════════════════
group_centres = {
    "Geography":  ( 0.15,  0.70),
    "Attraction": ( 0.50,  0.85),
    "Services":   ( 0.85,  0.65),
    "Experience": ( 0.75,  0.20),
    "Safety":     ( 0.20,  0.20),
    "Other":      ( 0.50,  0.50),
}
init_pos = {}
for node, data in G.nodes(data=True):
    grp = data.get("group","Other")
    cx, cy = group_centres.get(grp,(0.5,0.5))
    np.random.seed(hash(node) % 9999)
    init_pos[node] = (cx + np.random.uniform(-0.08,0.08),
                      cy + np.random.uniform(-0.08,0.08))

pos = nx.spring_layout(G, pos=init_pos, seed=42, k=0.38, iterations=200,
                        weight=None)

# ══════════════════════════════════════════════════════════════════════════════
# Build concept-level ego graph around "Hallasan"
# ══════════════════════════════════════════════════════════════════════════════
SEED = "Hallasan"
# Build concept lookup
concept_map = {c["name"].lower(): c for c in concepts}
# Find all relations involving seed (2-hop)
seed_rels = [r for r in rels if
             r["subject"].lower()==SEED.lower() or
             r["object"].lower()==SEED.lower()]
# Collect neighbours
neighbours = set()
for r in seed_rels:
    neighbours.add(r["subject"]); neighbours.add(r["object"])

EGO = nx.DiGraph()
EGO.add_nodes_from(neighbours)
for r in seed_rels[:35]:   # cap for readability
    EGO.add_edge(r["subject"], r["object"], rel=r["predicate"])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(30, 16), facecolor="white")
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(1, 2, width_ratios=[0.60, 0.40],
                      left=0.01, right=0.99, top=0.93, bottom=0.04,
                      wspace=0.05)
ax_L = fig.add_subplot(gs[0])
ax_R = fig.add_subplot(gs[1])

fig.text(0.5, 0.97,
    "Jeju Tourism Ontology — Entity Connection Map  v0.7.0",
    ha="center", va="top", fontsize=20, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.945,
    "LEFT: Schema-level class connections (all constrained relations + OWL is_a hierarchy)  |  "
    "RIGHT: Concept-level ego-graph for  'Hallasan'  (2-hop neighbourhood)",
    ha="center", va="top", fontsize=10, color="#546E7A")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Schema-level network
# ══════════════════════════════════════════════════════════════════════════════
ax_L.set_facecolor("#FAFAFA")
ax_L.set_title("Class-Level Connection Schema", fontsize=14,
                fontweight="bold", color="#1A1A2E", pad=10)
ax_L.axis("off")

# Draw edges first (behind nodes)
EDGE_STYLES = {
    "hierarchy":   dict(color="#90A4AE", width=1.2, style="dashed", alpha=0.65),
    "constrained": dict(color="#1565C0", width=1.5, style="solid",  alpha=0.50),
}
edge_label_map = {}

for u, v, data in G.edges(data=True):
    style = data.get("style","constrained")
    rel   = data.get("rel","")
    cfg   = EDGE_STYLES.get(style, EDGE_STYLES["constrained"])
    x0,y0 = pos[u]; x1,y1 = pos[v]
    # Slight curve via rad
    rad = 0.15 if style=="constrained" else 0.08
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
        ax=ax_L,
        edge_color=cfg["color"],
        width=cfg["width"],
        style=cfg["style"],
        alpha=cfg["alpha"],
        connectionstyle=f"arc3,rad={rad}",
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        min_source_margin=18,
        min_target_margin=18)
    mx, my = (x0+x1)/2, (y0+y1)/2
    if style=="constrained":
        edge_label_map[(u,v)] = rel

# Node sizes proportional to instance count
counts = [G.nodes[n].get("count",0) for n in G.nodes()]
max_c  = max(counts) if counts else 1
node_sizes = [300 + 2000*(G.nodes[n].get("count",0)/max_c) for n in G.nodes()]
node_colors_dark  = [CLS2DARK.get(n,"#37474F")  for n in G.nodes()]
node_colors_light = [CLS2LIGHT.get(n,"#FAFAFA") for n in G.nodes()]
node_colors_mid   = [CLS2MID.get(n,"#BDBDBD")   for n in G.nodes()]

# Draw nodes: light fill + coloured border
nx.draw_networkx_nodes(G, pos, ax=ax_L,
    node_size=node_sizes,
    node_color=node_colors_light,
    edgecolors=node_colors_dark,
    linewidths=2.2)

# Labels (bold, class name + count)
label_dict = {n: f"{n}\n({G.nodes[n].get('count',0):,})" for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=label_dict, ax=ax_L,
    font_size=6.5, font_color="#1A1A2E", font_weight="bold")

# Edge labels for constrained edges (small, italic)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_map, ax=ax_L,
    font_size=5.8, font_color="#1565C0", bbox=dict(boxstyle="round,pad=0.15",
    facecolor="white", alpha=0.75, edgecolor="none"),
    label_pos=0.38)

# Legend
grp_handles = [mpatches.Patch(facecolor=light, edgecolor=dark, linewidth=1.5,
                               label=f"{grp}  ({len(classes)} classes)")
               for grp,(classes,dark,mid,light) in GROUPS.items()]
grp_handles += [
    mpatches.Patch(facecolor="white", edgecolor="#90A4AE",  linewidth=1.2,
                   label="is_a  (hierarchy)", linestyle="--"),
    mpatches.Patch(facecolor="white", edgecolor="#1565C0",  linewidth=1.5,
                   label="constrained relation"),
]
ax_L.legend(handles=grp_handles, loc="lower left", fontsize=8.5,
            framealpha=0.95, edgecolor="#BDBDBD", title="Legend",
            title_fontsize=9, handlelength=1.6)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Concept ego graph around Hallasan
# ══════════════════════════════════════════════════════════════════════════════
ax_R.set_facecolor("#FAFAFA")
ax_R.set_title(f"Concept Ego-Graph: '{SEED}' (sample neighbourhood)",
                fontsize=13, fontweight="bold", color="#1A1A2E", pad=10)
ax_R.axis("off")

if len(EGO.nodes()) > 1:
    ego_pos = nx.spring_layout(EGO, seed=7, k=0.7, iterations=120)

    # Colour nodes by concept class
    ego_colors_fill = []
    ego_colors_edge = []
    ego_sizes = []
    for node in EGO.nodes():
        info = concept_map.get(node.lower(), {})
        cls  = info.get("concept_class","?") if isinstance(info,dict) else "?"
        dark = CLS2DARK.get(cls,"#37474F")
        light= CLS2LIGHT.get(cls,"#FAFAFA")
        ego_colors_fill.append(light)
        ego_colors_edge.append(dark)
        ego_sizes.append(900 if node.lower()==SEED.lower() else 420)

    nx.draw_networkx_nodes(EGO, ego_pos, ax=ax_R,
        node_size=ego_sizes,
        node_color=ego_colors_fill,
        edgecolors=ego_colors_edge,
        linewidths=[3.5 if n.lower()==SEED.lower() else 1.8 for n in EGO.nodes()])

    nx.draw_networkx_labels(EGO, ego_pos, ax=ax_R,
        font_size=7, font_color="#1A1A2E", font_weight="bold",
        labels={n: n for n in EGO.nodes()})

    # Edge colours by relation
    rel_palette = {
        "located_in":"#1565C0","has_activity":"#2E7D32","near":"#7B1FA2",
        "is_a":"#546E7A","suitable_for":"#E65100","reachable_by":"#C62828",
        "part_of":"#00695C","best_for":"#F57F17","available_in":"#1565C0",
    }
    for u, v, data in EGO.edges(data=True):
        rel = data.get("rel","")
        col = rel_palette.get(rel,"#90A4AE")
        nx.draw_networkx_edges(EGO, ego_pos, edgelist=[(u,v)], ax=ax_R,
            edge_color=col, width=1.8, alpha=0.70,
            connectionstyle="arc3,rad=0.12",
            arrows=True, arrowsize=14, arrowstyle="-|>",
            min_source_margin=14, min_target_margin=14)

    # Edge labels
    ego_edge_labels = {(u,v): d["rel"] for u,v,d in EGO.edges(data=True)}
    nx.draw_networkx_edge_labels(EGO, ego_pos, edge_labels=ego_edge_labels,
        ax=ax_R, font_size=6, font_color="#37474F",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  alpha=0.80, edgecolor="none"),
        label_pos=0.45)

    # Annotate seed node
    sx, sy = ego_pos[SEED]
    ax_R.annotate(f"SEED NODE\n'{SEED}'", xy=(sx, sy), xytext=(sx+0.15, sy+0.15),
                  fontsize=8, fontweight="bold", color="#1565C0",
                  arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.2),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                             edgecolor="#1565C0", linewidth=1.2))

    # Relation colour legend
    rel_handles = [mpatches.Patch(color=col, label=rel, linewidth=0)
                   for rel, col in rel_palette.items() if rel in
                   {d["rel"] for _,_,d in EGO.edges(data=True)}]
    ax_R.legend(handles=rel_handles, loc="lower left", fontsize=7.5,
                framealpha=0.95, edgecolor="#BDBDBD",
                title="Relation Type", title_fontsize=8,
                handlelength=1.2)
else:
    ax_R.text(0.5, 0.5, f"No relations found\nfor '{SEED}'\nin loaded ontology.",
              ha="center", va="center", fontsize=12, color="#546E7A",
              transform=ax_R.transAxes)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
    "Node size ∝ instance count  |  Dashed edges = is_a hierarchy  |  "
    "Solid edges = constrained relations  |  "
    "Jeju Tourism Ontology v0.7.0  ©  2026",
    ha="center", va="bottom", fontsize=8, color="#9E9E9E")

out = OUTDIR / "ontology_entity_connections.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
