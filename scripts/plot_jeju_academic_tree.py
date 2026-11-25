"""
Academic-style hierarchical ontology tree — Jeju Tourism Ontology v0.7.0.
Style: horizontal left-to-right dendrogram, white background, thin black lines,
plain text nodes in minimal boxes.  Modelled after Figures 3 & 4 in
Ontology_Sample.pdf (Agriculture / News Event ontology trees).
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
ONT_DIR = ROOT / "artifacts" / "ontologies" / "0.7.0"
DS_DIR  = ROOT / "dataset"
OUTDIR  = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load ontology artefacts ───────────────────────────────────────────────────
with open(ONT_DIR / "schema.json")   as f: schema   = json.load(f)
with open(ONT_DIR / "concepts.json") as f: concepts = json.load(f)

concept_map: dict[str, dict] = {c["name"].lower(): c for c in concepts}
cls2concepts: dict[str, list[str]] = defaultdict(list)
for c in concepts:
    cls2concepts[c["concept_class"]].append(c["name"])

# ── Jeju filter ───────────────────────────────────────────────────────────────
NON_JEJU = [
    "seoul","busan","incheon","daegu","daejeon","gwangju","suwon","ulsan",
    "singapore","bangkok","tokyo","osaka","beijing","shanghai","hong kong",
    "paris","london","new york","dubai","sydney","bali","phuket","taipei",
    "airbnb","booking.com","tripadvisor","google maps","klook","agoda",
    "expedia","viator","getyourguide",
    "korean air","asiana","jinair","jeju air",   # airlines not island entities
]
JEJU_KW  = [
    "jeju","hallasan","seongsan","manjanggul","hyeopjae","hamdeok",
    "seogwipo","jungmun","cheonjiyeon","olle","daepo","sanbangsan",
    "udo","marado","bijarim","yongduam","dongmun","olle trail",
    "jeju folk","teddy bear","shinhwa","eco land","loveland",
    "jeju national","baekrokdam","yeongsil","eorimok","gwaneumsa",
    "haenyeo","dol hareubang","jeju black pork","heuk dwaeji",
    "chuja","biyangdo","jeju horse","jeju tangerine","hallim",
    "aewol","seocheon","aewol coast","jeju stone park",
]
INCL_CLS = {"Season","TravelerType","DietaryRequirement","TransportMode",
             "SafetyAdvice","WeatherCondition","TravelConstraint",
             "LocalCustom","EmergencyService","Itinerary","Cuisine",
             "Activity","Route"}

def is_jeju(name: str, cls: str) -> bool:
    nl = name.lower()
    if any(b in nl for b in NON_JEJU): return False
    if any(j in nl for j in JEJU_KW):  return True
    if cls in INCL_CLS:                 return True
    return False

# ── Load dataset for entity frequency ────────────────────────────────────────
print("Loading dataset …")
ds_freq: Counter = Counter()
jeju_re = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b')

for fp in sorted(DS_DIR.glob("10kCoT*.json")):
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("data", [])
        for rec in records:
            # Check Jeju relevance at record level
            raw = json.dumps(rec, ensure_ascii=False)
            if "jeju" not in raw.lower():
                continue
            for m in jeju_re.finditer(raw):
                ds_freq[m.group(1)] += 1
    except Exception as e:
        print(f"  skip {fp.name}: {e}")

print(f"  {sum(ds_freq.values()):,} entity mentions extracted")

# Curated Jeju fallbacks — authoritative proper-noun names for every class
CURATED: dict[str, list[str]] = {
    "Island":          ["Jeju Island", "Udo Island", "Marado"],
    "City":            ["Jeju City", "Seogwipo City"],
    "Region":          ["Jungmun Resort Area", "Hallim", "Aewol"],
    "Country":         ["South Korea"],
    "Mountain":        ["Hallasan", "Seongsan Ilchulbong", "Sanbangsan"],
    "Beach":           ["Hamdeok Beach", "Hyeopjae Beach", "Iho Tewoo Beach"],
    "Trail":           ["Hallasan Summit Trail", "Jeju Olle Trail", "Yeongsil Course"],
    "Museum":          ["Jeju Folk Village Museum", "Jeju Haenyeo Museum",
                        "Jeju National Museum"],
    "ReligiousSite":   ["Gwaneumsa Temple", "Sanbangsa Temple"],
    "Park":            ["Hallasan National Park", "Bijarim Forest", "Jeju Stone Park"],
    "Landmark":        ["Manjanggul Cave", "Dol Hareubang", "Cheonjiyeon Falls"],
    "Market":          ["Dongmun Traditional Market", "Seogwipo Olle Market"],
    "Accommodation":   ["Jeju Shilla Hotel", "Lotte Hotel Jeju", "Jeju Hanok Stay"],
    "Restaurant":      ["Heuk Dwaeji Black Pork Restaurant",
                        "Haemul Jjigae Seafood House", "Gogi Guksu Noodle Shop"],
    "Cuisine":         ["Jeju Black Pork (Heuk Dwaeji)", "Hallabong Citrus",
                        "Jeonbokjuk (Abalone Porridge)"],
    "TransportMode":   ["Bus", "Rental Car", "Taxi"],
    "Route":           ["Jeju Olle Route 1", "Hallasan Summit Route"],
    "Activity":        ["Hiking", "Haenyeo Experience", "Snorkeling"],
    "Itinerary":       ["3-Day Jeju Explorer", "Outdoor Adventure Plan"],
    "TravelerType":    ["Family", "Solo Traveler", "Couple"],
    "Season":          ["Spring", "Summer", "Autumn", "Winter"],
    "SafetyAdvice":    ["UV Sun Protection", "Typhoon Warning Check", "Trail Safety"],
    "WeatherCondition":["Typhoon", "Heavy Fog", "Heavy Rainfall"],
    "LocalCustom":     ["Haenyeo Diving Culture", "Dol Hareubang Tradition"],
    "TravelConstraint":["Budget Range", "Entry Fee Required"],
    "EmergencyService":["119 Fire & EMS", "Jeju University Hospital"],
}

# Quality filter: reject generic/possessive descriptions
_GENERIC_ENDS = (
    "beaches","coastline","trails","views","parks","culture","cuisine",
    "seafood","travelers","tourists","services","activities","sites",
    "options","attractions","features","area","areas","regions","spots",
    "information","tips","advice","conditions","requirements","accommodations",
    "restaurants","emergency","parking","fees","tourism","visitors",
)

def _is_quality_name(name: str) -> bool:
    if "'" in name or "'" in name:              return False  # possessives
    if name.lower().endswith(_GENERIC_ENDS):         return False
    if re.search(r'\b(jeju|hallasan)\s*$', name.lower()):
        return False  # reject bare "Jeju" or "Hallasan" with nothing after
    # Reject Korean administrative suffixes (-si, -gun, -do) used as duplicates
    if re.search(r'\b\w+-(si|gun|do)\b', name.lower()):
        return False
    tokens = name.split()
    if len(tokens) == 1:
        return True
    specific = [t for t in tokens
                if t.lower() not in ("jeju","the","a","an","and","of","in","at")]
    return len(specific) >= 1

def top_instances(cls: str, n: int = 3) -> list[str]:
    """Return top-n Jeju instances for a concept class."""
    # 1) Ontology concepts that directly contain a Jeju keyword AND pass quality
    strict = [
        name for name in cls2concepts[cls]
        if any(j in name.lower() for j in JEJU_KW)
        and not any(b in name.lower() for b in NON_JEJU)
        and len(name.split()) <= 5          # skip long phrases
        and name[0].isupper()               # must start with capital
        and _is_quality_name(name)
    ]
    # De-duplicate very similar names (keep shorter)
    seen_lower: set[str] = set()
    deduped: list[str] = []
    for name in strict:
        key = re.sub(r'\s+', ' ', name.lower().strip())
        if not any(key in s or s in key for s in seen_lower):
            deduped.append(name)
            seen_lower.add(key)

    def ds_score(name: str) -> int:
        nl = name.lower()
        best = 0
        for k, v in ds_freq.items():
            if k.lower() == nl or nl in k.lower() or k.lower() in nl:
                best = max(best, v)
        return best

    deduped.sort(key=lambda x: (-ds_score(x), len(x)))

    # 2) Pad with curated fallbacks if needed
    result = deduped[:n]
    if len(result) < n and cls in CURATED:
        for fb in CURATED[cls]:
            if len(result) >= n:
                break
            if not any(fb.lower() in r.lower() or r.lower() in fb.lower()
                       for r in result):
                result.append(fb)

    # 3) Full curated fallback if still empty
    if not result and cls in CURATED:
        result = CURATED[cls][:n]

    return result[:n]

# ── Tree structure definition ─────────────────────────────────────────────────
# Each node: dict with keys: label, node_type, children
# node_type: "root" | "group" | "class" | "abstract" | "instance"

def inst_nodes(cls: str, n: int = 3) -> list[dict]:
    return [{"label": name, "node_type": "instance", "children": []}
            for name in top_instances(cls, n)]

def cls_node(label: str, cls: str, n: int = 3,
             children: list | None = None) -> dict:
    ch = children if children is not None else inst_nodes(cls, n)
    return {"label": label, "node_type": "class", "children": ch}

tree_root = {
    "label": "Jeju Tourism\nOntology",
    "node_type": "root",
    "children": [
        # ── Group 1: Geography ────────────────────────────────────────────
        {"label": "Geography", "node_type": "group", "children": [
            {"label": "Destination", "node_type": "abstract", "children": [
                cls_node("Island",  "Island"),
                cls_node("City",    "City"),
                cls_node("Region",  "Region"),
                cls_node("Country", "Country", n=2),
            ]},
        ]},
        # ── Group 2: Attraction ───────────────────────────────────────────
        {"label": "Attraction", "node_type": "group", "children": [
            {"label": "NaturalAttraction", "node_type": "abstract", "children": [
                cls_node("Mountain", "Mountain"),
                cls_node("Beach",    "Beach"),
                cls_node("Trail",    "Trail"),
            ]},
            {"label": "CulturalAttraction", "node_type": "abstract", "children": [
                cls_node("Museum",       "Museum"),
                cls_node("ReligiousSite","ReligiousSite", n=2),
            ]},
            cls_node("Park",     "Park"),
            cls_node("Landmark", "Landmark"),
            cls_node("Market",   "Market", n=2),
        ]},
        # ── Group 3: Services ─────────────────────────────────────────────
        {"label": "Services", "node_type": "group", "children": [
            cls_node("Accommodation", "Accommodation"),
            cls_node("Restaurant",    "Restaurant"),
            cls_node("Cuisine",       "Cuisine"),
            cls_node("TransportMode", "TransportMode"),
        ]},
        # ── Group 4: Experience ───────────────────────────────────────────
        {"label": "Experience", "node_type": "group", "children": [
            cls_node("Activity",     "Activity"),
            cls_node("Itinerary",    "Itinerary", n=2),
            cls_node("TravelerType", "TravelerType"),
            cls_node("Season",       "Season"),
        ]},
        # ── Group 5: Safety ───────────────────────────────────────────────
        {"label": "Safety", "node_type": "group", "children": [
            cls_node("SafetyAdvice",     "SafetyAdvice"),
            cls_node("WeatherCondition", "WeatherCondition"),
            cls_node("LocalCustom",      "LocalCustom", n=2),
            cls_node("TravelConstraint", "TravelConstraint", n=2),
            cls_node("EmergencyService", "EmergencyService", n=2),
        ]},
    ]
}

# ── Layout algorithm ──────────────────────────────────────────────────────────
def count_leaves(node: dict) -> int:
    if not node["children"]:
        return 1
    return sum(count_leaves(c) for c in node["children"])

def assign_y(node: dict, y_start: float, leaf_h: float = 1.0) -> float:
    """Assign .y to every node. Returns y_end after this subtree."""
    if not node["children"]:
        node["y"] = y_start + leaf_h * 0.5
        return y_start + leaf_h
    cur = y_start
    ys = []
    for child in node["children"]:
        end = assign_y(child, cur, leaf_h)
        ys.append(child["y"])
        cur = end
    node["y"] = (ys[0] + ys[-1]) / 2.0
    return cur

# Fixed x-levels (data units, not axes fraction)
X_LEVELS = {
    "root":     0.30,
    "group":    1.80,
    "abstract": 3.60,
    "class":    5.40,
    "instance": 7.50,
}

def assign_x(node: dict) -> None:
    node["x"] = X_LEVELS.get(node["node_type"], 5.40)
    for child in node["children"]:
        assign_x(child)

total_leaves = count_leaves(tree_root)
assign_y(tree_root, 0.0, leaf_h=1.0)
assign_x(tree_root)
print(f"Tree: {total_leaves} leaf nodes")

# ── Collect all nodes & edges ─────────────────────────────────────────────────
all_nodes: list[dict] = []
edges: list[tuple[dict, dict]] = []

def collect(node: dict, parent: dict | None = None) -> None:
    all_nodes.append(node)
    if parent is not None:
        edges.append((parent, node))
    for child in node["children"]:
        collect(child, node)

collect(tree_root)

# Flip y so Geography (first group) appears at TOP of figure
for node in all_nodes:
    node["y"] = total_leaves - node["y"]

# ── Colour palette (light, matching academic style but subtle) ────────────────
TYPE_STYLE = {
    #          box_fc   box_ec   text_c   lw
    "root":    ("#1A237E","#1A237E","white", 1.8),
    "group":   ("#37474F","#37474F","white", 1.4),
    "abstract":("#455A64","#B0BEC5","#1A1A2E",1.0),
    "class":   ("#FAFAFA","#607D8B","#1A1A2E",0.8),
    "instance":("#FFFFFF","#9E9E9E","#424242",0.6),
}

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W  = 18.0
FIG_H  = max(16.0, total_leaves * 0.31 + 3.0)
BOX_H  = 0.52          # box height in data units
LINE_C = "#455A64"      # connector line colour
LINE_W = 0.9

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")

# Data limits
Y_PAD = 1.2
ax.set_xlim(-0.10, 9.10)
ax.set_ylim(-Y_PAD, total_leaves + Y_PAD)

# Box widths per node type (data units)
level_widths = {
    "root":     2.40,
    "group":    1.55,
    "abstract": 2.00,
    "class":    1.80,
    "instance": 2.10,
}

# ── Draw elbow connectors ─────────────────────────────────────────────────────
for parent, child in edges:
    px, py = parent["x"], parent["y"]
    cx, cy = child["x"],  child["y"]
    p_hw = level_widths.get(parent["node_type"], 1.80) / 2
    c_hw = level_widths.get(child["node_type"],  1.80) / 2
    x0   = px + p_hw          # parent right edge
    x1   = cx - c_hw          # child  left  edge
    mid_x = x0 + (x1 - x0) * 0.45
    ax.plot([x0, mid_x, mid_x, x1],
            [py, py,    cy,    cy],
            color=LINE_C, linewidth=LINE_W,
            solid_capstyle="round", solid_joinstyle="round",
            zorder=1)

# ── Draw nodes ────────────────────────────────────────────────────────────────

for node in all_nodes:
    nx_  = node["x"]
    ny_  = node["y"]
    ntype = node["node_type"]
    label = node["label"]
    fc, ec, tc, lw = TYPE_STYLE.get(ntype, TYPE_STYLE["class"])
    bw   = level_widths.get(ntype, 1.80)
    bh   = BOX_H * (1.35 if ntype in ("root","group") else 1.0)
    # Centred box
    x0 = nx_ - bw / 2
    y0 = ny_ - bh / 2

    box = FancyBboxPatch((x0, y0), bw, bh,
                          boxstyle="round,pad=0.05",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, zorder=3)
    ax.add_patch(box)

    fs = 8.5 if ntype in ("root","group") else (7.8 if ntype in ("abstract","class") else 7.0)
    fw = "bold" if ntype in ("root","group","abstract") else "normal"
    ax.text(nx_, ny_, label,
            ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=4,
            multialignment="center",
            fontfamily="DejaVu Sans")

# ── Column header labels ──────────────────────────────────────────────────────
headers = [
    (X_LEVELS["root"],     "Ontology Root"),
    (X_LEVELS["group"],    "Concept Group"),
    (X_LEVELS["abstract"], "Parent Class"),
    (X_LEVELS["class"],    "Concept Class"),
    (X_LEVELS["instance"], "Sample Instances\n(Jeju-specific)"),
]
for hx, hl in headers:
    ax.text(hx, total_leaves + 0.50, hl,
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color="#546E7A",
            fontfamily="DejaVu Sans")
    ax.plot([hx - 0.70, hx + 0.70],
            [total_leaves + 0.25, total_leaves + 0.25],
            color="#B0BEC5", linewidth=0.8, zorder=2)

# ── Title & caption ───────────────────────────────────────────────────────────
fig.text(0.50, 0.995,
    "Figure — Jeju Tourism Ontology: Hierarchical Class and Instance Tree",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#1A1A2E")
fig.text(0.50, 0.979,
    "Root → Concept Groups → OWL Classes (with is-a hierarchy) → Jeju-specific instance examples  |  "
    "v0.7.0",
    ha="center", va="top", fontsize=8.5, color="#546E7A")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc="#1A237E", ec="#1A237E", label="Ontology Root"),
    mpatches.Patch(fc="#37474F", ec="#37474F", label="Concept Group (5)"),
    mpatches.Patch(fc="#FAFAFA", ec="#455A64", lw=1.0, label="Abstract Class (is_a parent)"),
    mpatches.Patch(fc="#FAFAFA", ec="#607D8B", lw=0.8, label="Concept Class"),
    mpatches.Patch(fc="white",   ec="#9E9E9E", lw=0.6, label="Instance (Jeju-specific)"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=7.5, framealpha=0.95, edgecolor="#BDBDBD",
          title="Node Types", title_fontsize=8,
          bbox_to_anchor=(0.99, 0.01), bbox_transform=ax.transAxes,
          handlelength=1.5, handleheight=0.95)

plt.tight_layout(pad=0.4)
out = OUTDIR / "jeju_ontology_academic_tree.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
