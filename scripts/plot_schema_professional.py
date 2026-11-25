"""
Plot 1 — Professional Ontology Schema Overview (slide-ready, light background).

Panels:
  TOP    : title + 8 headline stats
  LEFT   : OWL Class Hierarchy tree (colour-coded by group)
  CENTRE : Relation Domain–Range constraint table
  RIGHT  : Controlled Vocabularies + Rule examples
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
ONT_DIR = ROOT / "artifacts" / "ontologies" / "0.7.0"
OUTDIR  = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(ONT_DIR / "schema.json")    as f: schema   = json.load(f)
with open(ONT_DIR / "concepts.json")  as f: concepts = json.load(f)
with open(ONT_DIR / "relations.json") as f: rels     = json.load(f)
with open(ONT_DIR / "rules.json")     as f: rules    = json.load(f)

cls_counts = Counter(c["concept_class"] for c in concepts)
rel_counts = Counter(r["predicate"]     for r in rels)
HIER       = schema["class_hierarchy"]
CONSTR     = schema["relation_constraints"]
VOCAB      = schema["controlled_vocabularies"]

# ── Colour groups ────────────────────────────────────────────────────────────
GROUPS = {
    "Geography":    (["Destination","Region","Country","City","Island"],        "#1565C0", "#E3F2FD"),
    "Attraction":   (["Attraction","NaturalAttraction","CulturalAttraction",
                       "Beach","Mountain","Trail","Park","Landmark",
                       "ReligiousSite","Museum","Market"],                      "#2E7D32", "#E8F5E9"),
    "Services":     (["Accommodation","Restaurant","Cuisine",
                       "DietaryRequirement","TransportMode","Route"],            "#6A1B9A", "#F3E5F5"),
    "Experience":   (["Activity","Itinerary","TravelerType","Season"],           "#E65100", "#FFF3E0"),
    "Safety":       (["SafetyAdvice","EmergencyService","TravelConstraint",
                       "WeatherCondition","LocalCustom"],                        "#C62828", "#FFEBEE"),
}
CLS2COLOR  = {}
CLS2LIGHT  = {}
for grp, (classes, dark, light) in GROUPS.items():
    for cls in classes:
        CLS2COLOR[cls] = dark
        CLS2LIGHT[cls] = light

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 18), facecolor="white")
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(3, 3,
    height_ratios=[0.10, 0.52, 0.38],
    width_ratios =[0.28, 0.44, 0.28],
    hspace=0.32, wspace=0.22,
    left=0.02, right=0.98, top=0.96, bottom=0.03)

ax_hdr  = fig.add_subplot(gs[0, :])
ax_tree = fig.add_subplot(gs[1:, 0])
ax_mat  = fig.add_subplot(gs[1,  1])
ax_bar  = fig.add_subplot(gs[2,  1])
ax_voc  = fig.add_subplot(gs[1:, 2])

for ax in [ax_hdr, ax_tree, ax_mat, ax_bar, ax_voc]:
    ax.set_facecolor("white")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1); ax_hdr.axis("off")

# Title background
rect = FancyBboxPatch((0, 0.45), 1, 0.55, boxstyle="round,pad=0.01",
                       linewidth=0, facecolor="#1565C0", transform=ax_hdr.transAxes,
                       clip_on=False)
ax_hdr.add_patch(rect)
ax_hdr.text(0.5, 0.73, "Jeju Island Tourism Ontology — Schema Architecture  v0.7.0",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", transform=ax_hdr.transAxes)
ax_hdr.text(0.5, 0.52,
    "OWL2 · SHACL Validation · LLM-Powered Extraction (CoT / GoT / Metacognitive) · "
    "Namespace: http://tourism.ontology.org/",
    ha="center", va="center", fontsize=9, color="#BBDEFB",
    transform=ax_hdr.transAxes)

# Stats row
stats = [
    ("8,348",  "Concept\nInstances",   "#1565C0"),
    ("38,868", "Relations\n(Triples)", "#2E7D32"),
    ("31",     "OWL\nClasses",         "#6A1B9A"),
    ("24",     "Relation\nTypes",      "#E65100"),
    ("9",      "Constrained\nRelations","#C62828"),
    ("28",     "Inference\nRules",     "#00695C"),
    ("14",     "Hierarchy\nEdges",     "#37474F"),
    ("4",      "Ontology\nVersions",   "#4527A0"),
]
xs = np.linspace(0.04, 0.96, len(stats))
for x, (val, lbl, col) in zip(xs, stats):
    box = FancyBboxPatch((x-0.055, 0.01), 0.10, 0.40,
                          boxstyle="round,pad=0.01", linewidth=1.5,
                          edgecolor=col, facecolor=col+"11",
                          transform=ax_hdr.transAxes, clip_on=False)
    ax_hdr.add_patch(box)
    ax_hdr.text(x, 0.28, val, ha="center", va="center",
                fontsize=17, fontweight="bold", color=col,
                transform=ax_hdr.transAxes)
    ax_hdr.text(x, 0.08, lbl, ha="center", va="center",
                fontsize=7, color="#546E7A",
                transform=ax_hdr.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# A — CLASS HIERARCHY TREE
# ══════════════════════════════════════════════════════════════════════════════
ax_tree.set_xlim(0, 10); ax_tree.set_ylim(-1, 37); ax_tree.axis("off")

# Group legend at top
ax_tree.set_title("OWL Class Hierarchy", fontsize=13, fontweight="bold",
                   color="#1A1A2E", pad=8)
legend_handles = [mpatches.Patch(facecolor=light, edgecolor=dark, label=grp, linewidth=1.5)
                  for grp, (_, dark, light) in GROUPS.items()]
ax_tree.legend(handles=legend_handles, loc="upper right", fontsize=7.5,
               framealpha=0.95, edgecolor="#BDBDBD", ncol=1,
               title="Category", title_fontsize=8)

# Build ordered tree rows
children_map: dict[str, list] = defaultdict(list)
for child, parent in HIER.items():
    children_map[parent].append(child)

all_children = set(HIER.keys())
all_cls      = set(schema["concept_classes"])
roots        = sorted(all_cls - all_children)

rows: list[tuple[str, int]] = []
def collect(node, depth):
    rows.append((node, depth))
    for ch in sorted(children_map.get(node, [])):
        collect(ch, depth + 1)
for r_node in roots:
    collect(r_node, 0)

total = len(rows)

# Draw tree
parent_y: dict[str, float] = {}
for i, (node, depth) in enumerate(rows):
    y   = total - 1 - i
    x   = depth * 1.55 + 0.4
    dark  = CLS2COLOR.get(node, "#37474F")
    light = CLS2LIGHT.get(node, "#FAFAFA")
    cnt   = cls_counts.get(node, 0)
    parent_y[node] = y

    # Connector from parent
    if depth > 0:
        parent = HIER[node]
        py = parent_y[parent]
        conn_x = (depth - 1) * 1.55 + 0.4 + 0.1
        ax_tree.plot([conn_x + 5.2, x - 0.05], [py, py],
                     color=dark, lw=0.9, alpha=0.4, solid_capstyle="round")
        ax_tree.plot([conn_x + 5.2, conn_x + 5.2], [py, y],
                     color=dark, lw=0.9, alpha=0.3, solid_capstyle="round")

    # Node box
    bw = 5.5 - depth * 0.2
    box = FancyBboxPatch((x - 0.08, y - 0.37), bw, 0.74,
                          boxstyle="round,pad=0.06", linewidth=1.4,
                          edgecolor=dark, facecolor=light,
                          transform=ax_tree.transData)
    ax_tree.add_patch(box)

    # Class name
    prefix = "  " * depth
    ax_tree.text(x + 0.25, y + 0.05, f"{prefix}{node}",
                 va="center", ha="left", fontsize=8.5, fontweight="bold",
                 color=dark)
    # Count badge
    ax_tree.text(x + bw - 0.25, y + 0.05, f"{cnt:,}",
                 va="center", ha="right", fontsize=7.5, color="#546E7A")

ax_tree.text(0.98, 1.01, "instances →", ha="right", va="bottom",
             fontsize=7, color="#9E9E9E", transform=ax_tree.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# B — RELATION CONSTRAINT TABLE
# ══════════════════════════════════════════════════════════════════════════════
ax_mat.set_title("Relation Domain–Range Constraints", fontsize=12,
                  fontweight="bold", color="#1A1A2E", pad=8)
ax_mat.set_facecolor("white")

rel_names = sorted(CONSTR.keys())
all_cls_mentioned = set()
for v in CONSTR.values():
    all_cls_mentioned |= set(v["subject_classes"]) | set(v["object_classes"])
cls_cols = sorted(all_cls_mentioned)

nR, nC = len(rel_names), len(cls_cols)
mat = np.zeros((nR, nC))
for i, rel in enumerate(rel_names):
    subj = set(CONSTR[rel]["subject_classes"])
    obj  = set(CONSTR[rel]["object_classes"])
    for j, cls in enumerate(cls_cols):
        s, o = cls in subj, cls in obj
        mat[i, j] = 3 if (s and o) else (1 if s else (2 if o else 0))

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_mat = ListedColormap(["#F5F5F5", "#1565C0", "#2E7D32", "#E65100"])
norm_mat = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5], cmap_mat.N)

im = ax_mat.imshow(mat, cmap=cmap_mat, norm=norm_mat, aspect="auto",
                   interpolation="nearest")

ax_mat.set_xticks(range(nC))
ax_mat.set_xticklabels(cls_cols, rotation=45, ha="right", fontsize=7.5, color="#1A1A2E")
ax_mat.set_yticks(range(nR))
ax_mat.set_yticklabels(rel_names, fontsize=9, color="#1A1A2E")
ax_mat.tick_params(length=2, color="#BDBDBD")

for x in range(-1, nC):
    ax_mat.axvline(x + 0.5, color="#E0E0E0", lw=0.5)
for y in range(-1, nR):
    ax_mat.axhline(y + 0.5, color="#E0E0E0", lw=0.5)

ax_mat.spines["top"].set_visible(False)
ax_mat.spines["right"].set_visible(False)
for sp in ["bottom","left"]:
    ax_mat.spines[sp].set_edgecolor("#BDBDBD")

legend_items = [
    mpatches.Patch(color="#1565C0", label="Subject (domain)"),
    mpatches.Patch(color="#2E7D32", label="Object (range)"),
    mpatches.Patch(color="#E65100", label="Both"),
    mpatches.Patch(color="#F5F5F5", label="Not constrained", linewidth=0.8,
                   edgecolor="#BDBDBD"),
]
ax_mat.legend(handles=legend_items, loc="lower right", fontsize=7.5,
              framealpha=0.95, edgecolor="#BDBDBD", handlelength=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# C — CONCEPT DISTRIBUTION BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
ax_bar.set_title("Concept Instance Distribution (Top 15 Classes)", fontsize=11,
                  fontweight="bold", color="#1A1A2E", pad=8)
ax_bar.set_facecolor("white")

top15 = cls_counts.most_common(15)
labels_ = [l for l, _ in top15]
vals_   = [v for _, v in top15]
colors_ = [CLS2COLOR.get(l, "#37474F") for l in labels_]

y_pos = np.arange(len(labels_))
bars  = ax_bar.barh(y_pos, vals_, color=colors_, alpha=0.85,
                     edgecolor="white", height=0.65)

# Gradient-like effect: lighter cap
for bar, col in zip(bars, colors_):
    ax_bar.barh(bar.get_y(), bar.get_width(), color=col, alpha=0.25,
                 height=bar.get_height())

for bar, val in zip(bars, vals_):
    ax_bar.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", ha="left", fontsize=8, color="#546E7A")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(labels_, fontsize=9, color="#1A1A2E")
ax_bar.set_xlabel("Number of instances", fontsize=8.5, color="#546E7A")
ax_bar.invert_yaxis()
ax_bar.set_xlim(0, max(vals_) * 1.22)
ax_bar.set_facecolor("#FAFAFA")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
for sp in ["bottom","left"]:
    ax_bar.spines[sp].set_edgecolor("#BDBDBD")
ax_bar.tick_params(colors="#546E7A", labelsize=8)
ax_bar.xaxis.label.set_color("#546E7A")
ax_bar.axvline(0, color="#E0E0E0", lw=0.8)
ax_bar.grid(axis="x", linestyle="--", linewidth=0.5, color="#E0E0E0", alpha=0.7)

# ══════════════════════════════════════════════════════════════════════════════
# D — CONTROLLED VOCABULARIES + TOP RELATIONS
# ══════════════════════════════════════════════════════════════════════════════
ax_voc.axis("off")
ax_voc.set_title("Controlled Vocabularies & Inference Rules",
                  fontsize=12, fontweight="bold", color="#1A1A2E", pad=8)

def draw_section(ax, title, items, dark, light, y_top):
    # Section header box
    bx = FancyBboxPatch((0.01, y_top - 0.032), 0.98, 0.040,
                         boxstyle="round,pad=0.006", linewidth=1.2,
                         edgecolor=dark, facecolor=light,
                         transform=ax.transAxes)
    ax.add_patch(bx)
    ax.text(0.03, y_top - 0.010, title, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=dark, va="center")
    y = y_top - 0.048
    for item in items:
        ax.text(0.06, y, f"• {item}", transform=ax.transAxes,
                fontsize=7.5, color="#37474F", va="top")
        y -= 0.030
    return y - 0.010

y_cur = 0.97

y_cur = draw_section(ax_voc, "Budget Levels",
                     VOCAB["budget_levels"],
                     "#6A1B9A", "#F3E5F5", y_cur)
y_cur = draw_section(ax_voc, "Traveler Types",
                     VOCAB["traveler_types"],
                     "#E65100", "#FFF3E0", y_cur)
y_cur = draw_section(ax_voc, "Dietary Requirements",
                     VOCAB["dietary_requirements"],
                     "#2E7D32", "#E8F5E9", y_cur)
y_cur = draw_section(ax_voc, "Transport Modes",
                     VOCAB["transport_modes"],
                     "#1565C0", "#E3F2FD", y_cur)

# Sample rules
bx = FancyBboxPatch((0.01, y_cur - 0.032), 0.98, 0.040,
                     boxstyle="round,pad=0.006", linewidth=1.2,
                     edgecolor="#C62828", facecolor="#FFEBEE",
                     transform=ax_voc.transAxes)
ax_voc.add_patch(bx)
ax_voc.text(0.03, y_cur - 0.010, f"Inference Rules  ({len(rules)} total)",
            transform=ax_voc.transAxes, fontsize=8.5, fontweight="bold",
            color="#C62828", va="center")
y_cur -= 0.048
sample_rules = [
    ("R001 [hard]", "Beach + severe_storm → SUPPRESS swimming"),
    ("R002 [hard]", "Hiking trail + typhoon → CLOSE trail"),
    ("R003 [soft]", "Haenyeo dive site → RECOMMEND haenyeo tour"),
    ("R004 [soft]", "Peak season → RECOMMEND early reservation"),
]
for rid, desc in sample_rules:
    ax_voc.text(0.04, y_cur, rid, transform=ax_voc.transAxes,
                fontsize=7, fontweight="bold", color="#C62828", va="top")
    ax_voc.text(0.26, y_cur, desc, transform=ax_voc.transAxes,
                fontsize=7, color="#37474F", va="top")
    y_cur -= 0.030

# Top relations
y_cur -= 0.005
bx = FancyBboxPatch((0.01, y_cur - 0.032), 0.98, 0.040,
                     boxstyle="round,pad=0.006", linewidth=1.2,
                     edgecolor="#00695C", facecolor="#E0F2F1",
                     transform=ax_voc.transAxes)
ax_voc.add_patch(bx)
ax_voc.text(0.03, y_cur - 0.010, "Top Relation Types",
            transform=ax_voc.transAxes, fontsize=8.5, fontweight="bold",
            color="#00695C", va="center")
y_cur -= 0.048
for rel, cnt in rel_counts.most_common(10):
    bar_w = cnt / rel_counts.most_common(1)[0][1] * 0.58
    bx2 = FancyBboxPatch((0.38, y_cur + 0.003), bar_w, 0.019,
                          boxstyle="square,pad=0.00", linewidth=0,
                          facecolor="#00695C", alpha=0.3,
                          transform=ax_voc.transAxes)
    ax_voc.add_patch(bx2)
    ax_voc.text(0.04, y_cur, f"• {rel}", transform=ax_voc.transAxes,
                fontsize=7.5, color="#37474F", va="top")
    ax_voc.text(0.96, y_cur, f"{cnt:,}", transform=ax_voc.transAxes,
                fontsize=7.5, color="#00695C", va="top", ha="right")
    y_cur -= 0.028

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
    "Built with: LLM extraction (CoT/GoT/Metacognitive) · NetworkX · RDFLib · pyshacl · "
    "Exported as OWL2 Turtle + JSON-LD  |  Jeju Tourism Ontology © 2026",
    ha="center", va="bottom", fontsize=7.5, color="#9E9E9E")

out = OUTDIR / "ontology_schema_professional.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
