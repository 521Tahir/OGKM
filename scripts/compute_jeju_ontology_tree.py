"""
Compute Complete Jeju Tourism Ontology Tree
-------------------------------------------
Outputs:
  1. artifacts/jeju_ontology_tree.json      — full machine-readable tree
  2. artifacts/figures/jeju_complete_tree.png — enhanced academic visualization
     with Jeju-filtered counts per class and top-5 instances per leaf
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
ART_DIR = ROOT / "artifacts"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load ontology ─────────────────────────────────────────────────────────────
with open(ONT_DIR / "schema.json")   as f: schema   = json.load(f)
with open(ONT_DIR / "concepts.json") as f: concepts = json.load(f)
with open(ONT_DIR / "relations.json") as f: relations = json.load(f)

HIER   = schema["class_hierarchy"]        # child → parent
CONSTR = schema["relation_constraints"]

# Full class → [concept] map (all 8348 concepts)
cls2all: dict[str, list[str]] = defaultdict(list)
for c in concepts:
    cls2all[c["concept_class"]].append(c["name"])
total_per_cls = {cls: len(names) for cls, names in cls2all.items()}

# ── Jeju filter ───────────────────────────────────────────────────────────────
NON_JEJU = [
    "seoul","busan","incheon","daegu","daejeon","gwangju","suwon","ulsan",
    "singapore","bangkok","tokyo","osaka","beijing","shanghai","hong kong",
    "paris","london","new york","dubai","sydney","bali","phuket","taipei",
    "airbnb","booking.com","tripadvisor","google maps","klook","agoda",
    "expedia","viator","getyourguide",
]
JEJU_KW = [
    "jeju","hallasan","seongsan","manjanggul","hyeopjae","hamdeok",
    "seogwipo","jungmun","cheonjiyeon","olle","daepo","sanbangsan",
    "udo","marado","bijarim","yongduam","dongmun","olle trail",
    "jeju folk","teddy bear","shinhwa","eco land","loveland",
    "jeju national","baekrokdam","yeongsil","eorimok","gwaneumsa",
    "haenyeo","dol hareubang","jeju black pork","heuk dwaeji",
    "chuja","biyangdo","jeju horse","jeju tangerine","hallim","aewol",
]
INCL_CLS = {
    "Season","TravelerType","DietaryRequirement","TransportMode",
    "SafetyAdvice","WeatherCondition","TravelConstraint",
    "LocalCustom","EmergencyService","Itinerary","Cuisine","Activity","Route",
}

def is_jeju(name: str, cls: str) -> bool:
    nl = name.lower()
    if any(b in nl for b in NON_JEJU): return False
    if any(j in nl for j in JEJU_KW):  return True
    if cls in INCL_CLS:                 return True
    return False

# Build Jeju-filtered concept lists
cls2jeju: dict[str, list[str]] = {
    cls: [n for n in names if is_jeju(n, cls)]
    for cls, names in cls2all.items()
}
jeju_per_cls = {cls: len(names) for cls, names in cls2jeju.items()}

# ── Load dataset for entity frequency ────────────────────────────────────────
print("Loading dataset entity frequencies …")
ds_freq: Counter = Counter()
jeju_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

for fp in sorted(DS_DIR.glob("10kCoT*.json")):
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("data", [])
        for rec in records:
            raw = json.dumps(rec, ensure_ascii=False)
            if "jeju" not in raw.lower():
                continue
            for m in jeju_re.finditer(raw):
                ds_freq[m.group(1)] += 1
    except Exception as e:
        print(f"  skip {fp.name}: {e}")

print(f"  {sum(ds_freq.values()):,} entity mentions across Jeju records")

def ds_score(name: str) -> int:
    nl = name.lower()
    best = 0
    for k, v in ds_freq.items():
        if k.lower() == nl or nl in k.lower() or k.lower() in nl:
            best = max(best, v)
    return best

# Curated fallbacks (for classes with sparse Jeju coverage)
CURATED: dict[str, list[str]] = {
    "Island":          ["Jeju Island", "Udo Island", "Marado"],
    "City":            ["Jeju City", "Seogwipo"],
    "Region":          ["Jungmun Resort Area", "Hallim", "Aewol"],
    "Country":         ["South Korea"],
    "Mountain":        ["Hallasan", "Seongsan Ilchulbong", "Sanbangsan"],
    "Beach":           ["Hamdeok Beach", "Hyeopjae Beach", "Iho Tewoo Beach",
                        "Jungmun Saekdal Beach", "Woljeong Beach"],
    "Trail":           ["Hallasan Summit Trail", "Jeju Olle Trail",
                        "Yeongsil Course", "Eorimok Course", "Gwaneumsa Trail"],
    "Museum":          ["Jeju Folk Village Museum", "Jeju Haenyeo Museum",
                        "Jeju National Museum", "Jeju Museum of Art",
                        "Teddy Bear Museum"],
    "ReligiousSite":   ["Gwaneumsa Temple", "Sanbangsa Temple",
                        "Yongduam Rock Shrine", "Sanbanggulsa Cave Temple"],
    "Park":            ["Hallasan National Park", "Bijarim Forest",
                        "Jeju Stone Park", "Jeju World Natural Heritage Site",
                        "Halla Arboretum"],
    "Landmark":        ["Manjanggul Cave", "Dol Hareubang",
                        "Cheonjiyeon Falls", "Cheonjeyeon Falls",
                        "Yongduam Rock"],
    "Market":          ["Dongmun Traditional Market", "Seogwipo Olle Market",
                        "Jeju Jungang Underground Market"],
    "Accommodation":   ["Jeju Shilla Hotel", "Lotte Hotel Jeju",
                        "Jeju Hanok Guesthouse", "Seogwipo Pension",
                        "Jeju Family Resort"],
    "Restaurant":      ["Heuk Dwaeji Black Pork", "Haemul Jjigae Restaurant",
                        "Gogi Guksu Noodle Shop", "Jeju Seafood Market",
                        "Hallabong Dessert Cafe"],
    "Cuisine":         ["Jeju Black Pork (Heuk Dwaeji)", "Hallabong Citrus",
                        "Jeonbokjuk (Abalone Porridge)", "Okdom Grilled Fish",
                        "Bing Soo (Shaved Ice)"],
    "TransportMode":   ["Bus", "Rental Car", "Taxi", "Electric Bicycle",
                        "Scooter"],
    "Route":           ["Jeju Olle Route 1", "Jeju Olle Route 7",
                        "Hallasan Summit Route"],
    "Activity":        ["Hiking", "Haenyeo Diving Experience", "Snorkeling",
                        "Surfing", "Horse Riding"],
    "Itinerary":       ["3-Day Jeju Nature Tour", "Jeju Coastal Explorer",
                        "Family-Friendly Jeju Plan"],
    "TravelerType":    ["Family", "Solo Traveler", "Couple", "Senior",
                        "Adventure Seeker"],
    "Season":          ["Spring (March–May)", "Summer (June–August)",
                        "Autumn (Sept–Nov)", "Winter (Dec–Feb)"],
    "SafetyAdvice":    ["UV Sun Protection", "Typhoon Warning Check",
                        "Trail Safety Rules", "Jellyfish Alert",
                        "Mountain Weather Check"],
    "WeatherCondition":["Typhoon Season", "Heavy Fog", "Heavy Rainfall",
                        "Strong Sea Winds", "Snowfall on Hallasan"],
    "LocalCustom":     ["Haenyeo Diving Culture", "Dol Hareubang Tradition",
                        "Jeju Dialect (Jejueo)", "Respect Haenyeo Dives"],
    "TravelConstraint":["Budget: Mid-range", "Permit for Hallasan Summit",
                        "No Drone Zone (beaches)", "Entry Fees Apply"],
    "EmergencyService":["119 Fire & EMS", "Jeju University Hospital",
                        "Jeju Police 112", "Coast Guard 122"],
    "DietaryRequirement": ["Halal Certified", "Vegetarian Options",
                           "Gluten-Free Available", "Vegan Friendly"],
}

_GENERIC_ENDS = (
    "beaches","coastline","trails","views","parks","culture","cuisine",
    "seafood","travelers","tourists","services","activities","sites",
    "options","attractions","features","area","areas","regions","spots",
    "information","tips","advice","conditions","requirements","accommodations",
    "restaurants","emergency","parking","fees","tourism","visitors",
)

def _quality(name: str) -> bool:
    if "'" in name or "’" in name: return False
    if name.lower().endswith(_GENERIC_ENDS): return False
    if re.search(r"\b\w+-(si|gun|do)\b", name.lower()): return False
    tokens = name.split()
    specific = [t for t in tokens
                if t.lower() not in ("jeju","the","a","an","and","of","in","at")]
    return len(specific) >= 1

def top_instances(cls: str, n: int = 5) -> list[str]:
    strict = [
        nm for nm in cls2jeju[cls]
        if any(j in nm.lower() for j in JEJU_KW)
        and len(nm.split()) <= 6
        and nm[0].isupper()
        and _quality(nm)
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for nm in strict:
        key = re.sub(r"\s+", " ", nm.lower().strip())
        if not any(key in s or s in key for s in seen):
            deduped.append(nm); seen.add(key)
    deduped.sort(key=lambda x: (-ds_score(x), len(x)))
    result = deduped[:n]
    # Pad with curated
    if len(result) < n and cls in CURATED:
        for fb in CURATED[cls]:
            if len(result) >= n: break
            fb_key = fb.lower()
            if not any(fb_key in r.lower() or r.lower() in fb_key for r in result):
                result.append(fb)
    if not result and cls in CURATED:
        result = CURATED[cls][:n]
    return result[:n]

# ══════════════════════════════════════════════════════════════════════════════
# Build complete tree structure (with all Jeju concepts)
# ══════════════════════════════════════════════════════════════════════════════
GROUPS: dict[str, list[str]] = {
    "Geography":  ["Destination","Region","Country","City","Island"],
    "Attraction": ["Attraction","NaturalAttraction","CulturalAttraction",
                   "Beach","Mountain","Trail","Park","Landmark",
                   "ReligiousSite","Museum","Market"],
    "Services":   ["Accommodation","Restaurant","Cuisine",
                   "DietaryRequirement","TransportMode","Route"],
    "Experience": ["Activity","Itinerary","TravelerType","Season"],
    "Safety":     ["SafetyAdvice","EmergencyService","TravelConstraint",
                   "WeatherCondition","LocalCustom"],
}

# Build machine-readable tree
tree_data: dict = {
    "label": "Jeju Tourism Ontology",
    "version": "0.7.0",
    "total_concepts": len(concepts),
    "total_relations": len(relations),
    "jeju_concepts": sum(jeju_per_cls.values()),
    "groups": {}
}

print("\n" + "="*65)
print("  JEJU TOURISM ONTOLOGY — COMPLETE TREE STATISTICS")
print("="*65)
print(f"  Total concepts   : {len(concepts):,}")
print(f"  Jeju concepts    : {sum(jeju_per_cls.values()):,}")
print(f"  Total relations  : {len(relations):,}")
print(f"  Concept classes  : {len(schema['concept_classes'])}")
print()

for grp, classes in GROUPS.items():
    grp_total = sum(total_per_cls.get(c, 0) for c in classes)
    grp_jeju  = sum(jeju_per_cls.get(c, 0)  for c in classes)
    print(f"  [{grp}]  {grp_jeju:4d} / {grp_total:5d} Jeju concepts")

    grp_entry: dict = {"classes": {}}
    tree_data["groups"][grp] = grp_entry

    for cls in classes:
        t_cnt = total_per_cls.get(cls, 0)
        j_cnt = jeju_per_cls.get(cls, 0)
        top5  = top_instances(cls, 5)
        parent = HIER.get(cls)
        print(f"      {cls:25s}: {j_cnt:4d}/{t_cnt:4d}  →  {top5}")
        grp_entry["classes"][cls] = {
            "is_a": parent,
            "total_concepts": t_cnt,
            "jeju_concepts":  j_cnt,
            "top_instances":  top5,
            "all_instances":  cls2jeju[cls],
        }

print("="*65)

# Save JSON tree
json_out = ART_DIR / "jeju_ontology_tree.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(tree_data, f, ensure_ascii=False, indent=2)
print(f"\nSaved tree data → {json_out}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Enhanced Academic Tree with Counts
# ══════════════════════════════════════════════════════════════════════════════

# Tree node builder
def inst_nodes(cls: str, n: int = 5) -> list[dict]:
    return [{"label": nm, "node_type": "instance", "children": []}
            for nm in top_instances(cls, n)]

def cls_node(label: str, cls: str, n: int = 5,
             children: list | None = None) -> dict:
    j = jeju_per_cls.get(cls, 0)
    t = total_per_cls.get(cls, 0)
    ch = children if children is not None else inst_nodes(cls, n)
    return {
        "label": f"{label}\n({j:,} Jeju / {t:,} total)",
        "node_type": "class",
        "children": ch,
    }

def abstr_node(label: str, children: list) -> dict:
    return {"label": label, "node_type": "abstract", "children": children}

tree_root = {
    "label": "Jeju Tourism\nOntology\nv0.7.0",
    "node_type": "root",
    "children": [
        # ── Geography ────────────────────────────────────────────────────────
        {"label": "Geography", "node_type": "group", "children": [
            abstr_node("Destination", [
                cls_node("Island",  "Island",  n=4),
                cls_node("City",    "City",    n=3),
                cls_node("Region",  "Region",  n=3),
                cls_node("Country", "Country", n=2),
            ]),
        ]},
        # ── Attraction ───────────────────────────────────────────────────────
        {"label": "Attraction", "node_type": "group", "children": [
            abstr_node("NaturalAttraction", [
                cls_node("Mountain", "Mountain", n=4),
                cls_node("Beach",    "Beach",    n=5),
                cls_node("Trail",    "Trail",    n=4),
            ]),
            abstr_node("CulturalAttraction", [
                cls_node("Museum",        "Museum",        n=4),
                cls_node("ReligiousSite", "ReligiousSite", n=3),
            ]),
            cls_node("Park",     "Park",     n=4),
            cls_node("Landmark", "Landmark", n=4),
            cls_node("Market",   "Market",   n=3),
        ]},
        # ── Services ─────────────────────────────────────────────────────────
        {"label": "Services", "node_type": "group", "children": [
            cls_node("Accommodation",   "Accommodation",   n=4),
            cls_node("Restaurant",      "Restaurant",      n=4),
            cls_node("Cuisine",         "Cuisine",         n=5),
            cls_node("DietaryReq.",     "DietaryRequirement", n=3),
            cls_node("TransportMode",   "TransportMode",   n=4),
        ]},
        # ── Experience ───────────────────────────────────────────────────────
        {"label": "Experience", "node_type": "group", "children": [
            cls_node("Activity",     "Activity",     n=5),
            cls_node("Itinerary",    "Itinerary",    n=3),
            cls_node("TravelerType", "TravelerType", n=4),
            cls_node("Season",       "Season",       n=4),
        ]},
        # ── Safety ───────────────────────────────────────────────────────────
        {"label": "Safety", "node_type": "group", "children": [
            cls_node("SafetyAdvice",     "SafetyAdvice",     n=4),
            cls_node("WeatherCondition", "WeatherCondition", n=4),
            cls_node("LocalCustom",      "LocalCustom",      n=3),
            cls_node("TravelConstraint", "TravelConstraint", n=3),
            cls_node("EmergencyService", "EmergencyService", n=3),
        ]},
    ]
}

# ── Layout ────────────────────────────────────────────────────────────────────
def count_leaves(node: dict) -> int:
    if not node["children"]: return 1
    return sum(count_leaves(c) for c in node["children"])

def assign_y(node: dict, y_start: float, leaf_h: float = 1.0) -> float:
    if not node["children"]:
        node["y"] = y_start + leaf_h * 0.5
        return y_start + leaf_h
    cur = y_start; ys = []
    for child in node["children"]:
        end = assign_y(child, cur, leaf_h)
        ys.append(child["y"]); cur = end
    node["y"] = (ys[0] + ys[-1]) / 2.0
    return cur

X_LEVELS = {
    "root":     0.30,
    "group":    2.00,
    "abstract": 4.00,
    "class":    6.20,
    "instance": 8.80,
}

def assign_x(node: dict) -> None:
    node["x"] = X_LEVELS.get(node["node_type"], 6.20)
    for child in node["children"]:
        assign_x(child)

total_leaves = count_leaves(tree_root)
assign_y(tree_root, 0.0, leaf_h=1.0)
assign_x(tree_root)

all_nodes: list[dict] = []
edges: list[tuple] = []

def collect(node: dict, parent: dict | None = None) -> None:
    all_nodes.append(node)
    if parent: edges.append((parent, node))
    for child in node["children"]: collect(child, node)

collect(tree_root)

# Flip y: first group (Geography) at top
for node in all_nodes:
    node["y"] = total_leaves - node["y"]

print(f"\nTree: {total_leaves} leaf instance nodes across {len(all_nodes)} total nodes")

# ── Box widths ────────────────────────────────────────────────────────────────
level_widths = {
    "root":     2.60,
    "group":    1.65,
    "abstract": 2.20,
    "class":    2.50,
    "instance": 2.65,
}

# ── Group colours ─────────────────────────────────────────────────────────────
GROUP_COLOUR = {
    "Geography":  ("#1565C0","#E3F2FD"),
    "Attraction": ("#2E7D32","#E8F5E9"),
    "Services":   ("#6A1B9A","#F3E5F5"),
    "Experience": ("#E65100","#FFF3E0"),
    "Safety":     ("#C62828","#FFEBEE"),
}

TYPE_STYLE = {
    "root":     ("#1A237E","#1A237E","white",  2.0),
    "group":    (None,     None,     "white",  1.6),   # dynamic colour
    "abstract": ("#455A64","#B0BEC5","#1A1A2E",1.0),
    "class":    ("#FAFAFA","#607D8B","#1A1A2E",0.8),
    "instance": ("#FFFFFF","#B0BEC5","#424242",0.55),
}

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W = 22.0
FIG_H = max(20.0, total_leaves * 0.28 + 4.0)
BOX_H = 0.62
LINE_C = "#546E7A"
LINE_W = 0.85

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")
Y_PAD = 1.5
ax.set_xlim(-0.20, 11.50)
ax.set_ylim(-Y_PAD, total_leaves + Y_PAD)

# ── Group colour assignment ───────────────────────────────────────────────────
# Map each node to its group colour
node_group: dict[int, str] = {}
def tag_group(node: dict, grp: str) -> None:
    node_group[id(node)] = grp
    for child in node["children"]: tag_group(child, grp)
for grp_node in tree_root["children"]:
    tag_group(grp_node, grp_node["label"])
node_group[id(tree_root)] = "_root"

# ── Draw connectors ───────────────────────────────────────────────────────────
for parent, child in edges:
    px, py = parent["x"], parent["y"]
    cx, cy = child["x"],  child["y"]
    p_hw = level_widths.get(parent["node_type"], 2.50) / 2
    c_hw = level_widths.get(child["node_type"],  2.50) / 2
    x0 = px + p_hw
    x1 = cx - c_hw
    mid = x0 + (x1 - x0) * 0.42
    grp = node_group.get(id(child), "_root")
    col = GROUP_COLOUR.get(grp, ("",""))[0] if grp in GROUP_COLOUR else LINE_C
    col = col if col else LINE_C
    ax.plot([x0, mid, mid, x1], [py, py, cy, cy],
            color=col, linewidth=LINE_W, alpha=0.55,
            solid_capstyle="round", solid_joinstyle="round", zorder=1)

# ── Draw nodes ────────────────────────────────────────────────────────────────
for node in all_nodes:
    nx_ = node["x"]; ny_ = node["y"]
    ntype = node["node_type"]; label = node["label"]
    bw = level_widths.get(ntype, 2.50)
    bh = BOX_H * (1.55 if ntype in ("root","group") else
                  1.25 if ntype == "class" else 1.0)

    grp = node_group.get(id(node), "_root")
    dark, light = GROUP_COLOUR.get(grp, ("#37474F","#ECEFF1"))

    if ntype == "root":
        fc, ec, tc, lw = "#1A237E","#1A237E","white", 2.0
    elif ntype == "group":
        fc, ec, tc, lw = dark, dark, "white", 1.6
    elif ntype == "abstract":
        fc, ec, tc, lw = light, dark, "#1A1A2E", 1.0
    elif ntype == "class":
        fc, ec, tc, lw = "#FAFAFA", dark, "#1A1A2E", 0.9
    else:  # instance
        fc, ec, tc, lw = "white", "#CFD8DC","#424242", 0.5

    x0 = nx_ - bw / 2; y0 = ny_ - bh / 2
    box = FancyBboxPatch((x0, y0), bw, bh,
                          boxstyle="round,pad=0.06",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, zorder=3)
    ax.add_patch(box)

    fs = (9.5 if ntype == "root" else
          9.0 if ntype == "group" else
          7.5 if ntype == "class" else
          7.2 if ntype == "abstract" else 6.8)
    fw = "bold" if ntype in ("root","group","abstract") else "normal"
    ax.text(nx_, ny_, label, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=4,
            multialignment="center", fontfamily="DejaVu Sans")

# ── Column headers ────────────────────────────────────────────────────────────
HEADER_Y = total_leaves + 0.55
headers = [
    (X_LEVELS["root"],     "Ontology\nRoot"),
    (X_LEVELS["group"],    "Concept\nGroup"),
    (X_LEVELS["abstract"], "Parent Class\n(is-a hierarchy)"),
    (X_LEVELS["class"],    "Concept Class\n(Jeju / Total)"),
    (X_LEVELS["instance"], "Top Instances\n(Jeju-specific, by dataset freq.)"),
]
for hx, hl in headers:
    ax.text(hx, HEADER_Y, hl, ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color="#37474F",
            fontfamily="DejaVu Sans", multialignment="center")
    ax.plot([hx - 0.75, hx + 0.75],
            [total_leaves + 0.22, total_leaves + 0.22],
            color="#B0BEC5", linewidth=0.8, zorder=2)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.998,
    "Jeju Tourism Ontology — Complete Hierarchical Class & Instance Tree  (v0.7.0)",
    ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.50, 0.982,
    f"8,348 total concepts  |  {sum(jeju_per_cls.values()):,} Jeju-filtered  |  "
    f"38,868 relations  |  31 OWL classes  |  5 concept groups",
    ha="center", va="top", fontsize=8.5, color="#546E7A")

# ── Legend ────────────────────────────────────────────────────────────────────
grp_handles = [
    mpatches.Patch(facecolor=dark, edgecolor=dark,
                   label=f"{grp}  ({sum(jeju_per_cls.get(c,0) for c in cls)} Jeju)")
    for grp, cls in GROUPS.items()
    for dark, _ in [GROUP_COLOUR.get(grp, ("#37474F","#ECEFF1"))]
]
extra_handles = [
    mpatches.Patch(facecolor="#FAFAFA", edgecolor="#607D8B", lw=0.9,
                   label="Concept Class  (count = Jeju / total)"),
    mpatches.Patch(facecolor="#FFFFFF",  edgecolor="#CFD8DC", lw=0.5,
                   label="Instance  (top-5 by dataset frequency)"),
]
ax.legend(handles=grp_handles + extra_handles,
          loc="lower right", fontsize=7.5, framealpha=0.95,
          edgecolor="#B0BEC5", title="Legend", title_fontsize=8.5,
          bbox_to_anchor=(0.995, 0.005), bbox_transform=ax.transAxes,
          handlelength=1.5, handleheight=0.95, ncol=1)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.002,
    "Instances ranked by frequency in 90,017 CoT training records  |  "
    "Jeju-filter: name contains Jeju keyword OR class is generic  |  "
    "Jeju Tourism Ontology Project 2026",
    ha="center", va="bottom", fontsize=7.5, color="#9E9E9E")

plt.tight_layout(pad=0.5)
out = OUTDIR / "jeju_complete_tree.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved visualization → {out}")
