"""
Generate artifacts/figures/ontology_schema.png

Panels:
  A — Class Hierarchy tree (left)
  B — Relation Constraints matrix (centre-top)
  C — Concept class distribution bar chart (centre-bottom)
  D — Controlled Vocabularies (right)
  E — Header stats strip (top)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ONT_DIR = ROOT / "artifacts" / "ontologies" / "0.7.0"
OUT     = ROOT / "artifacts" / "figures" / "ontology_schema.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(ONT_DIR / "schema.json")   as f: schema    = json.load(f)
with open(ONT_DIR / "concepts.json") as f: concepts  = json.load(f)
with open(ONT_DIR / "relations.json")as f: relations = json.load(f)
with open(ONT_DIR / "rules.json")    as f: rules     = json.load(f)

cls_counts  = Counter(c.get("concept_class","?") for c in concepts)
rel_counts  = Counter(r.get("predicate","?")      for r in relations)
HIERARCHY   = schema["class_hierarchy"]          # child → parent
REL_TYPES   = schema["relation_types"]
REL_CONSTR  = schema["relation_constraints"]
VOCAB       = schema["controlled_vocabularies"]

# ── Colour palette (dark-research theme) ─────────────────────────────────────
BG   = "#0d1117"
GRID = "#21262d"
ACC1 = "#58a6ff"   # blue
ACC2 = "#3fb950"   # green
ACC3 = "#ff7b72"   # red/coral
ACC4 = "#d2a8ff"   # purple
ACC5 = "#ffa657"   # orange
WHITE= "#e6edf3"
MUTED= "#8b949e"

CLASS_COLORS = {
    "Destination": "#1f6feb", "Region": "#388bfd", "Country": "#58a6ff",
    "City": "#79c0ff",        "Island": "#a5d6ff",
    "Attraction": "#2ea043",  "NaturalAttraction": "#3fb950",
    "Beach": "#56d364",       "Mountain": "#7ee787", "Trail": "#a2f0ac",
    "Park": "#56d364",        "Landmark": "#7ee787",
    "CulturalAttraction": "#bf8700", "ReligiousSite": "#d29922",
    "Museum": "#e3b341",      "Market": "#f0c040",
    "Accommodation": "#8957e5","Restaurant": "#a371f7",
    "Cuisine": "#d2a8ff",     "Activity": "#ff7b72",
    "TransportMode": "#ffa657","Itinerary": "#ffb966",
    "TravelerType": "#20b2aa","Season": "#00ced1",
    "WeatherCondition": "#48d1cc","SafetyAdvice": "#ff6347",
    "EmergencyService": "#dc143c","LocalCustom": "#daa520",
    "TravelConstraint": "#808080","Route": "#708090",
    "DietaryRequirement": "#9acd32",
}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 18), facecolor=BG)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03,
                    hspace=0.38, wspace=0.28)

gs = fig.add_gridspec(3, 3,
    height_ratios=[0.08, 0.50, 0.42],
    width_ratios =[0.30, 0.40, 0.30])

ax_hdr  = fig.add_subplot(gs[0, :])     # E — header
ax_tree = fig.add_subplot(gs[1:, 0])    # A — hierarchy tree
ax_mat  = fig.add_subplot(gs[1, 1])     # B — constraint matrix
ax_bar  = fig.add_subplot(gs[2, 1])     # C — class distribution
ax_voc  = fig.add_subplot(gs[1:, 2])    # D — vocabularies

for ax in [ax_hdr, ax_tree, ax_mat, ax_bar, ax_voc]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

# ══════════════════════════════════════════════════════════════════════════════
# E — Header strip
# ══════════════════════════════════════════════════════════════════════════════
ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
ax_hdr.axis("off")

ax_hdr.text(0.5, 0.82,
    "Jeju Tourism Ontology — Schema v0.7.0",
    ha="center", va="center", fontsize=22, fontweight="bold",
    color=WHITE, transform=ax_hdr.transAxes)

stats = [
    (f"{len(concepts):,}", "Concepts"),
    (f"{len(relations):,}", "Relations"),
    (f"{len(schema['concept_classes'])}", "Classes"),
    (f"{len(REL_TYPES)}", "Relation\nTypes"),
    (f"{len(REL_CONSTR)}", "Constrained\nRelations"),
    (f"{len(rules)}", "Rules"),
    ("OWL2", "Serialisation"),
    ("SHACL", "Validation"),
]
xs = np.linspace(0.04, 0.96, len(stats))
for x, (val, lbl) in zip(xs, stats):
    ax_hdr.text(x, 0.40, val, ha="center", va="center",
                fontsize=15, fontweight="bold", color=ACC1,
                transform=ax_hdr.transAxes)
    ax_hdr.text(x, 0.06, lbl, ha="center", va="center",
                fontsize=7.5, color=MUTED,
                transform=ax_hdr.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# A — Class Hierarchy Tree
# ══════════════════════════════════════════════════════════════════════════════
ax_tree.set_xlim(0, 10); ax_tree.set_ylim(-1, 36)
ax_tree.axis("off")
ax_tree.set_title("Class Hierarchy", color=WHITE, fontsize=13,
                   fontweight="bold", pad=6)

# Build tree: parent → children
from collections import defaultdict
children_map: dict[str, list[str]] = defaultdict(list)
for child, parent in HIERARCHY.items():
    children_map[parent].append(child)

# Top-level classes (not a child in hierarchy)
all_children = set(HIERARCHY.keys())
all_classes  = set(schema["concept_classes"])
roots = sorted(all_classes - all_children)

def draw_node(ax, label, x, y, color, indent=0):
    count = cls_counts.get(label, 0)
    bg_rect = mpatches.FancyBboxPatch(
        (x - 0.1, y - 0.35), 6.8 - x*0.3, 0.72,
        boxstyle="round,pad=0.05", linewidth=0.8,
        edgecolor=color, facecolor=color + "22",
        transform=ax.transData, clip_on=False)
    ax.add_patch(bg_rect)
    ax.text(x + 0.15, y, label, va="center", ha="left",
            fontsize=8.5, color=color, fontweight="bold")
    ax.text(8.5, y, f"{count:,}", va="center", ha="right",
            fontsize=7.5, color=MUTED)

def draw_connector(ax, px, py, cx, cy, color):
    ax.plot([px+0.05, px+0.05, cx], [py-0.36, (py+cy)/2, (py+cy)/2],
            color=color+"66", lw=0.8)
    ax.plot([cx, cx], [(py+cy)/2, cy+0.36],
            color=color+"66", lw=0.8)

# Lay out tree rows
rows: list[tuple[str, int]] = []  # (class_name, depth)

def collect(node, depth):
    rows.append((node, depth))
    for child in sorted(children_map.get(node, [])):
        collect(child, depth + 1)

for root in roots:
    collect(root, 0)

# Draw connector lines first, then nodes
for i, (node, depth) in enumerate(rows):
    y = len(rows) - 1 - i
    x = depth * 1.2 + 0.3
    color = CLASS_COLORS.get(node, ACC1)
    if depth > 0:
        parent = HIERARCHY[node]
        pi = next(j for j, (n, _) in enumerate(rows) if n == parent)
        py = len(rows) - 1 - pi
        ax.plot([x - 0.9, x - 0.1], [y, y],
                 color=color+"55", lw=0.7) if True else None
        ax_tree.plot([x - 0.85, x - 0.1], [y, y],
                     color=color + "55", lw=0.8)
        ax_tree.plot([x - 0.85, x - 0.85], [py - 0.0, y],
                     color=color + "33", lw=0.8)
    draw_node(ax_tree, node, x, y, color)

ax_tree.text(8.6, len(rows), "#", va="center", ha="right",
             fontsize=7, color=MUTED)

# ══════════════════════════════════════════════════════════════════════════════
# B — Relation Constraint Matrix
# ══════════════════════════════════════════════════════════════════════════════
ax_mat.set_title("Relation Domain–Range Constraints", color=WHITE,
                  fontsize=11, fontweight="bold", pad=6)

relations_with_constraints = sorted(REL_CONSTR.keys())
# Collect all unique classes mentioned
all_mentioned = set()
for v in REL_CONSTR.values():
    all_mentioned |= set(v["subject_classes"]) | set(v["object_classes"])
sorted_classes = sorted(all_mentioned)

nR = len(relations_with_constraints)
nC = len(sorted_classes)

# Build matrix: 1=subject, 2=object, 3=both
mat = np.zeros((nR, nC))
for i, rel in enumerate(relations_with_constraints):
    subj = set(REL_CONSTR[rel]["subject_classes"])
    obj  = set(REL_CONSTR[rel]["object_classes"])
    for j, cls in enumerate(sorted_classes):
        s = cls in subj
        o = cls in obj
        mat[i, j] = 3 if (s and o) else (1 if s else (2 if o else 0))

# Custom colormap: 0=bg, 1=blue(subj), 2=green(obj), 3=orange(both)
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap([BG, ACC1, ACC2, ACC5])
norm = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5], cmap.N)

ax_mat.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")

ax_mat.set_xticks(range(nC))
ax_mat.set_xticklabels(sorted_classes, rotation=45, ha="right",
                        fontsize=7, color=WHITE)
ax_mat.set_yticks(range(nR))
ax_mat.set_yticklabels(relations_with_constraints, fontsize=8, color=WHITE)
ax_mat.tick_params(colors=MUTED, length=2)

# Grid lines
for x in range(-1, nC):
    ax_mat.axvline(x + 0.5, color=GRID, lw=0.4)
for y in range(-1, nR):
    ax_mat.axhline(y + 0.5, color=GRID, lw=0.4)

# Legend
legend_items = [
    mpatches.Patch(color=ACC1, label="Subject (domain)"),
    mpatches.Patch(color=ACC2, label="Object (range)"),
    mpatches.Patch(color=ACC5, label="Both"),
]
ax_mat.legend(handles=legend_items, loc="lower right", fontsize=7,
              facecolor=GRID, edgecolor=GRID, labelcolor=WHITE,
              handlelength=1.0, framealpha=0.9)

ax_mat.set_facecolor(BG)
for sp in ax_mat.spines.values():
    sp.set_edgecolor(GRID)

# ══════════════════════════════════════════════════════════════════════════════
# C — Concept Class Distribution
# ══════════════════════════════════════════════════════════════════════════════
ax_bar.set_title("Concept Instance Distribution", color=WHITE,
                  fontsize=11, fontweight="bold", pad=6)

top_classes = cls_counts.most_common(15)
labels = [c for c, _ in top_classes]
values = [v for _, v in top_classes]
colors_bar = [CLASS_COLORS.get(l, ACC1) for l in labels]

y_pos = range(len(labels))
bars = ax_bar.barh(y_pos, values, color=colors_bar, alpha=0.85,
                    edgecolor=GRID, height=0.65)

for bar, val in zip(bars, values):
    ax_bar.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", ha="left", fontsize=7.5, color=MUTED)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(labels, fontsize=8.5, color=WHITE)
ax_bar.set_xlabel("Number of instances", fontsize=8, color=MUTED)
ax_bar.tick_params(colors=MUTED, labelsize=7.5)
ax_bar.set_facecolor(BG)
ax_bar.xaxis.label.set_color(MUTED)
ax_bar.tick_params(axis="x", colors=MUTED)
ax_bar.tick_params(axis="y", colors=WHITE)
for sp in ax_bar.spines.values():
    sp.set_edgecolor(GRID)
ax_bar.set_xlim(0, max(values) * 1.18)
ax_bar.invert_yaxis()
ax_bar.axvline(0, color=GRID, lw=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# D — Controlled Vocabularies + Top Relations
# ══════════════════════════════════════════════════════════════════════════════
ax_voc.axis("off")
ax_voc.set_title("Controlled Vocabularies & Top Relations",
                  color=WHITE, fontsize=11, fontweight="bold", pad=6)

y = 0.97
def section(ax, title, items, color, y_start, item_color=None):
    ax.text(0.03, y_start, title, transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=color, va="top")
    y_cur = y_start - 0.045
    for item in items:
        ax.text(0.06, y_cur, f"• {item}", transform=ax.transAxes,
                fontsize=8, color=item_color or WHITE, va="top")
        y_cur -= 0.038
    return y_cur - 0.012

sections = [
    ("Budget Levels", VOCAB["budget_levels"],         ACC3,  None),
    ("Traveler Types", VOCAB["traveler_types"],        ACC4,  None),
    ("Dietary Needs",  VOCAB["dietary_requirements"],  ACC2,  None),
    ("Transport Modes",VOCAB["transport_modes"],       ACC5,  None),
]

y_cur = 0.97
for title, items, col, icol in sections:
    y_cur = section(ax_voc, title, items, col, y_cur, MUTED)

# Top relation types
y_cur -= 0.01
ax_voc.text(0.03, y_cur, "Top Relation Types",
            transform=ax_voc.transAxes,
            fontsize=9, fontweight="bold", color=ACC1, va="top")
y_cur -= 0.045
for rel, cnt in rel_counts.most_common(12):
    ax_voc.text(0.06, y_cur, f"• {rel}", transform=ax_voc.transAxes,
                fontsize=7.5, color=MUTED, va="top")
    ax_voc.text(0.82, y_cur, f"{cnt:,}", transform=ax_voc.transAxes,
                fontsize=7.5, color=ACC1, va="top", ha="right")
    y_cur -= 0.038

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.005,
    "Namespace: http://tourism.ontology.org/  |  Serialisation: OWL2 Turtle + JSON-LD  |  "
    "Validation: SHACL  |  Extraction: LLM-CoT/GoT/Metacognitive  |  v0.7.0",
    ha="center", va="bottom", fontsize=7.5, color=MUTED)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
