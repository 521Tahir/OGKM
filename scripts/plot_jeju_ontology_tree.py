"""
Jeju Tourism Ontology — Ring 0 + Ring 1 Hierarchical Tree
Full dataset: dataset/10kCoT*.json  (~90k records, 13k+ Jeju-specific)
Only Jeju-specific entities. Top 40 Ring-1 nodes by dataset frequency.
"""
from __future__ import annotations
import json, os, re, math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT   = Path(__file__).resolve().parents[1]
ONT    = ROOT / "artifacts" / "ontologies" / "0.7.0"
DS_DIR = ROOT / "dataset"
OUTDIR = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load dataset entity frequencies ────────────────────────────────────────
print("Loading dataset …")
JEJU_KW = {
    "jeju","hallasan","seogwipo","hyeopjae","hamdeok","jungmun","seongsan",
    "ilchulbong","manjanggul","osulloc","samseonghyeol","olle","haenyeo",
    "udo island","marado","bijarim","aewol","hallim","songaksan","sanbangsan",
    "yongduam","cheonjiyeon","cheonjeyeon","dongmun","loveland","ecoland",
    "seopjikoji","camellia hill","hallim park","jeju folk village",
    "hallabong","black pork","jeju island",
}
ds_freq: Counter = Counter()
ds_total = 0; ds_jeju = 0
for fname in sorted(os.listdir(DS_DIR)):
    if not fname.endswith(".json"): continue
    try:
        with open(DS_DIR / fname, encoding="utf-8", errors="replace") as f:
            data = json.loads(f.read())
    except Exception as e:
        print(f"  SKIP {fname}: {e}"); continue
    if not isinstance(data, list): continue
    for rec in data:
        if not isinstance(rec, dict): continue
        text = " ".join(str(v) for v in rec.values() if v)
        ds_total += 1
        if not any(kw in text.lower() for kw in JEJU_KW): continue
        ds_jeju += 1
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', text):
            e = m.group(1)
            if len(e) > 3: ds_freq[e] += 1

print(f"  {ds_total:,} total records | {ds_jeju:,} Jeju records")
print(f"  {len(ds_freq):,} entity candidates")

def ds_score(name):
    nl = name.lower()
    best = 0
    for k, v in ds_freq.items():
        if k.lower() == nl or nl in k.lower() or k.lower() in nl:
            best = max(best, v)
    return best

# ── 2. Load + filter ontology ─────────────────────────────────────────────────
with open(ONT / "concepts.json")  as f: ALL_C = json.load(f)
with open(ONT / "relations.json") as f: ALL_R = json.load(f)

NON_JEJU = {
    "seoul","busan","incheon","daegu","daejeon","gwangju","suwon","jeonju",
    "yeosu","gyeongju","ulsan","changwon","pohang","andong","gangneung",
    "sokcho","chuncheon","cheongju","wonju","gunsan","mokpo","suncheon",
    "jinju","geoje","gangnam","hongdae","itaewon","myeongdong","dongdaemun",
    "gyeongbokgung","namsan tower","bukhansan","bukchon","insadong",
    "lotte world","coex","gyeonggi","gyeongsang","chungcheong","jeolla",
    "gangwon","dmz","singapore","tokyo","osaka","kyoto","hiroshima",
    "beijing","shanghai","guangzhou","hong kong","taipei","taiwan","macau",
    "bangkok","phuket","chiang mai","bali","jakarta","hanoi","ho chi minh",
    "saigon","da nang","kuala lumpur","manila","cebu","boracay",
    "google maps","kakaomaps","airbnb","booking.com","agoda",
    "tripadvisor","grab","gojek","uber",
}
INCL_CLS = {"Attraction","NaturalAttraction","CulturalAttraction","Beach",
            "Mountain","Trail","Park","Landmark","ReligiousSite","Museum",
            "Market","Accommodation","Restaurant","Cuisine","Destination",
            "Region","City","Island","Activity","Season","WeatherCondition",
            "TravelerType","DietaryRequirement","SafetyAdvice","TravelConstraint",
            "LocalCustom","Itinerary","TransportMode","Route"}
GENERIC_CLS = {"Activity","Season","WeatherCondition","TravelerType",
               "DietaryRequirement","SafetyAdvice","TravelConstraint",
               "LocalCustom","Itinerary","TransportMode","Route"}
GENERIC_KW  = {"beach","hiking","waterfall","cave","lava","volcanic","oreum",
               "park","museum","restaurant","cafe","hotel","resort","guesthouse",
               "pension","hostel","villa","lodge","black pork","tangerine",
               "abalone","seafood","snorkel","diving","cycling","kayak",
               "surfing","horseback","folk","heritage","temple","shrine",
               "ferry","rental","scooter","bicycle","family","solo","couple",
               "senior","adventure","budget","luxury","typhoon","sunny",
               "halal","vegan","vegetarian","traditional","local","authentic"}

def is_jeju(name, cls):
    n = name.lower()
    if any(b in n for b in NON_JEJU): return False
    if any(j in n for j in JEJU_KW):  return True
    if cls in GENERIC_CLS:             return True
    if any(g in n for g in GENERIC_KW): return True
    return cls in INCL_CLS

jC  = [c for c in ALL_C if is_jeju(c["name"], c.get("concept_class",""))]
jN  = {c["name"] for c in jC}
jR  = [r for r in ALL_R if r["subject"] in jN and r["object"] in jN]
jM  = {c["name"]: c for c in jC}
print(f"  Jeju ontology: {len(jC):,} concepts | {len(jR):,} relations")

# ── 3. Adjacency ──────────────────────────────────────────────────────────────
adj = defaultdict(set)
rel_lut = defaultdict(list)
for r in jR:
    s, o, p = r["subject"], r["object"], r["predicate"]
    adj[s].add(o); adj[o].add(s)
    rel_lut[(s,o)].append(p); rel_lut[(o,s)].append(p)

PRIORITY = ["located_in","has_activity","near","part_of","is_a","suitable_for",
            "reachable_by","best_for","served_at","available_in","requires",
            "connected_to","has_seasonality","has_risk"]

def best_rel(u, v):
    preds = rel_lut.get((u,v),[]) + rel_lut.get((v,u),[])
    for p in PRIORITY:
        if p in preds: return p
    return preds[0] if preds else "related"

# ── 4. Ring 0 ─────────────────────────────────────────────────────────────────
RING0 = [n for n in [
    "Hallasan","Jeju","Seongsan Ilchulbong","Hallasan National Park",
    "Jeju City","Manjanggul Cave","Haenyeo","Jeju Folk Village Museum",
] if n in jM]
seed_set = set(RING0)

# ── 5. Ring 1 (top 40 by dataset score + ontology degree) ────────────────────
r1_scores = {}
for seed in RING0:
    for nb in adj.get(seed, set()):
        if nb in seed_set or nb not in jM: continue
        sc = ds_score(nb) * 0.6 + len(adj.get(nb,set())) * 0.4
        r1_scores[nb] = max(r1_scores.get(nb,0), sc)

# Take top 40 globally by score, then guarantee at least 1 per class (up to 40)
RING1_MAX = 40
global_top = sorted(r1_scores.items(), key=lambda x: -x[1])

# Ensure at least 1 per class, then fill by score
seen_cls: set[str] = set()
selected: list[str] = []
# Pass 1: one per class (in score order)
for nb, sc in global_top:
    cls = jM[nb].get("concept_class","Other") if nb in jM else "Other"
    if cls not in seen_cls:
        selected.append(nb); seen_cls.add(cls)
    if len(seen_cls) >= 15: break   # cover top 15 classes
# Pass 2: fill to RING1_MAX by score
sel_set = set(selected)
for nb, sc in global_top:
    if len(selected) >= RING1_MAX: break
    if nb not in sel_set:
        selected.append(nb); sel_set.add(nb)

GROUP_ORDER = ["Geography","Culture","Services","Activity","Temporal","Safety","Other"]

CLS_CFG = {
    "Beach":             ("#0288D1","#01579B","Geography"),
    "Mountain":          ("#388E3C","#1B5E20","Geography"),
    "Trail":             ("#66BB6A","#2E7D32","Geography"),
    "NaturalAttraction": ("#00ACC1","#006064","Geography"),
    "Park":              ("#7CB342","#33691E","Geography"),
    "Attraction":        ("#1976D2","#0D47A1","Geography"),
    "Destination":       ("#1565C0","#0D47A1","Geography"),
    "Region":            ("#42A5F5","#1565C0","Geography"),
    "City":              ("#29B6F6","#0277BD","Geography"),
    "Island":            ("#26C6DA","#006064","Geography"),
    "Landmark":          ("#7E57C2","#4527A0","Culture"),
    "CulturalAttraction":("#FFB300","#E65100","Culture"),
    "ReligiousSite":     ("#FF8F00","#BF360C","Culture"),
    "Museum":            ("#EF6C00","#E65100","Culture"),
    "Market":            ("#E91E63","#880E4F","Culture"),
    "LocalCustom":       ("#DCEDC8","#558B2F","Culture"),
    "Accommodation":     ("#8D6E63","#4E342E","Services"),
    "Restaurant":        ("#E53935","#B71C1C","Services"),
    "Cuisine":           ("#F06292","#880E4F","Services"),
    "TransportMode":     ("#78909C","#37474F","Services"),
    "Route":             ("#90A4AE","#455A64","Services"),
    "DietaryRequirement":("#AED581","#33691E","Services"),
    "Activity":          ("#43A047","#1B5E20","Activity"),
    "TravelerType":      ("#AB47BC","#6A1B9A","Activity"),
    "Itinerary":         ("#FDD835","#F57F17","Activity"),
    "Season":            ("#FFD54F","#F9A825","Temporal"),
    "WeatherCondition":  ("#B0BEC5","#546E7A","Temporal"),
    "SafetyAdvice":      ("#EF9A9A","#C62828","Safety"),
    "TravelConstraint":  ("#D7CCC8","#4E342E","Safety"),
    "EmergencyService":  ("#EF5350","#B71C1C","Safety"),
}
DEF_CFG = ("#E0E0E0","#9E9E9E","Other")
GROUP_FILL = {"Geography":"#E3F2FD","Culture":"#FFF8E1","Services":"#F3E5F5",
              "Activity":"#E8F5E9","Temporal":"#FFFDE7","Safety":"#FFEBEE","Other":"#F5F5F5"}
GROUP_DARK = {"Geography":"#1565C0","Culture":"#E65100","Services":"#6A1B9A",
              "Activity":"#2E7D32","Temporal":"#F57F17","Safety":"#C62828","Other":"#757575"}

def cfg(n):
    cls = jM[n].get("concept_class","Other") if n in jM else "Other"
    return CLS_CFG.get(cls, DEF_CFG)

def grp(n): return cfg(n)[2]

# Sort ring-1 by group then dataset score
RING1 = sorted(selected,
    key=lambda n: (GROUP_ORDER.index(grp(n)) if grp(n) in GROUP_ORDER else 9,
                   -ds_score(n)))

# Parent map
r1_parent = {}; r1_rel = {}
for r1n in RING1:
    best_seed = None; best_sc = -1
    for seed in RING0:
        if r1n in adj.get(seed,set()):
            sc = ds_score(seed)
            if sc > best_sc: best_sc = sc; best_seed = seed
    if best_seed: r1_parent[r1n] = best_seed; r1_rel[r1n] = best_rel(best_seed, r1n)

print(f"  Ring 0: {len(RING0)} | Ring 1: {len(RING1)}")

# ── 6. Layout ─────────────────────────────────────────────────────────────────
NR0 = len(RING0); NR1 = len(RING1)
PAD = 1.2
W = NR1 * 2.4 + 2 * PAD   # total canvas width
H = 22.0

# Ring-0: centred top row
r0_y = H - 2.8
r0_xs = np.linspace(W/2 - (NR0-1)*3.2/2, W/2 + (NR0-1)*3.2/2, NR0)

# Ring-1: group-clustered with gap between groups
r1_y = 5.2
r1_xs = []
x_cur = PAD
prev_g = None
for n in RING1:
    g = grp(n)
    if prev_g and g != prev_g: x_cur += 0.8
    r1_xs.append(x_cur)
    x_cur += 2.4
    prev_g = g

# Centre ring-1
r1_total_w = r1_xs[-1] - r1_xs[0]
r1_shift = W/2 - (r1_xs[0] + r1_total_w/2)
r1_xs = [x + r1_shift for x in r1_xs]

pos_r0 = {n: (x, r0_y) for n, x in zip(RING0, r0_xs)}
pos_r1 = {n: (x, r1_y) for n, x in zip(RING1, r1_xs)}

# ── 7. Figure ─────────────────────────────────────────────────────────────────
PRED_C = {
    "located_in":"#1565C0","has_activity":"#2E7D32","near":"#7B1FA2",
    "part_of":"#006064","is_a":"#90A4AE","suitable_for":"#E65100",
    "reachable_by":"#C62828","best_for":"#F9A825","served_at":"#AD1457",
    "requires":"#37474F","available_in":"#0097A7","connected_to":"#558B2F",
    "has_seasonality":"#5C6BC0","has_risk":"#B71C1C","related":"#BDBDBD",
}

FIG_W = max(28, NR1 * 1.05 + 3)
FIG_H = 20
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")
ax.axis("off")
ax.set_xlim(0, W); ax.set_ylim(0, H)

# ── Row background bands ──────────────────────────────────────────────────────
ax.add_patch(plt.Rectangle((0, r0_y - 2.2), W, 4.6,
    facecolor="#E8EAF6", alpha=0.5, zorder=0))
ax.add_patch(plt.Rectangle((0, r1_y - 3.8), W, 8.5,
    facecolor="#F1F8E9", alpha=0.40, zorder=0))

# Row labels
ax.text(0.5, r0_y + 2.1, "RING 0 — Primary Seed Concepts",
    fontsize=16, fontweight="bold", color="#283593", va="bottom")
ax.text(0.5, r1_y + 4.4, "RING 1 — Direct Neighbours  (ranked by dataset mention frequency, Jeju-only)",
    fontsize=14, fontweight="bold", color="#1B5E20", va="bottom")

# ── Group-coloured strips under Ring-1 ───────────────────────────────────────
prev_g  = None; g_xs = []
for i, n in enumerate(RING1):
    g = grp(n)
    x = r1_xs[i]
    if g != prev_g:
        if prev_g and g_xs:
            lo, hi = g_xs[0]-1.0, g_xs[-1]+1.0
            ax.add_patch(plt.Rectangle((lo, r1_y - 3.6), hi-lo, 8.0,
                facecolor=GROUP_FILL.get(prev_g,"#F5F5F5"),
                edgecolor=GROUP_DARK.get(prev_g,"#BDBDBD"),
                linewidth=1.0, linestyle="--", alpha=0.55, zorder=1))
            ax.text((lo+hi)/2, r1_y + 4.6, prev_g,
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=GROUP_DARK.get(prev_g,"#455A64"), alpha=0.85)
        g_xs = [x]; prev_g = g
    else:
        g_xs.append(x)
if prev_g and g_xs:
    lo, hi = g_xs[0]-1.0, g_xs[-1]+1.0
    ax.add_patch(plt.Rectangle((lo, r1_y-3.6), hi-lo, 8.0,
        facecolor=GROUP_FILL.get(prev_g,"#F5F5F5"),
        edgecolor=GROUP_DARK.get(prev_g,"#BDBDBD"),
        linewidth=1.0, linestyle="--", alpha=0.55, zorder=1))
    ax.text((lo+hi)/2, r1_y+4.6, prev_g,
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        color=GROUP_DARK.get(prev_g,"#455A64"), alpha=0.85)

# ── Edges ─────────────────────────────────────────────────────────────────────
for r1n, seed in r1_parent.items():
    rel = r1_rel.get(r1n,"related")
    col = PRED_C.get(rel,"#BDBDBD")
    x0, y0 = pos_r0[seed]
    x1, y1 = pos_r1[r1n]
    rad = 0.18 if abs(x1-x0) < 3 else 0.06
    ax.annotate("", xy=(x1, y1 + 0.7), xytext=(x0, y0 - 0.7),
        arrowprops=dict(arrowstyle="-|>", color=col,
                        lw=1.8, alpha=0.65,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2)
    # Relation label at 35% along the edge
    tx = x0 + 0.35*(x1-x0); ty = y0 + 0.35*(y1-y0)
    ax.text(tx, ty, rel, fontsize=6, color=col, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                  alpha=0.82, edgecolor="none"), zorder=3)

# ── Ring-0 nodes ──────────────────────────────────────────────────────────────
R0R = 0.80
for n in RING0:
    x, y = pos_r0[n]
    fill, border, group = cfg(n)
    freq = ds_score(n)
    # Glow
    ax.add_patch(plt.Circle((x,y), R0R+0.15, fill=False,
        edgecolor=border, lw=1.8, alpha=0.25, zorder=4))
    # Node
    ax.add_patch(plt.Circle((x,y), R0R, facecolor=fill,
        edgecolor=border, lw=3.0, zorder=5))
    # Name
    lines = n.replace(" ","\n",1) if len(n)>12 else n
    ax.text(x, y+0.15, lines, ha="center", va="center",
        fontsize=10, fontweight="bold", color="#1A1A2E",
        zorder=6, multialignment="center")
    # Class tag above
    cls = jM[n].get("concept_class","?") if n in jM else "?"
    ax.text(x, y+R0R+0.12, cls, ha="center", va="bottom",
        fontsize=8, color=border, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=fill+"CC",
                  edgecolor=border, lw=0.8), zorder=6)
    # Freq below
    ax.text(x, y-R0R-0.12, f"✦ {freq:,} dataset mentions",
        ha="center", va="top", fontsize=8.5, color="#455A64",
        fontweight="bold", zorder=6)

# ── Ring-1 nodes — alternating label up/down for readability ─────────────────
R1R = 0.52
for i, n in enumerate(RING1):
    x, y = pos_r1[n]
    fill, border, group = cfg(n)
    freq = ds_score(n)
    r = R1R + min(0.15, freq / 10000 * 0.18)

    ax.add_patch(plt.Circle((x,y), r, facecolor=fill,
        edgecolor=border, lw=2.0, zorder=5))

    cls = jM[n].get("concept_class","?") if n in jM else "?"
    rel = r1_rel.get(n, "")

    # Relation label inside node
    if rel:
        rc = PRED_C.get(rel,"#90A4AE")
        ax.text(x, y, rel[:10], ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.08", facecolor=rc,
                      edgecolor="none", alpha=0.92))

    # Name label — alternating above/below
    label_up = (i % 2 == 0)
    short = n if len(n) <= 22 else n[:20] + "…"
    if label_up:
        ly = y + r + 0.12
        va_n = "bottom"
        # Class tag above name
        ax.text(x, ly + 0.55, short, ha="center", va=va_n,
            fontsize=8, fontweight="bold", color="#1A1A2E",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor=border, lw=0.7, alpha=0.92), zorder=6)
        ax.text(x, ly + 0.55 + 0.48, cls, ha="center", va="bottom",
            fontsize=6.5, color=border, fontweight="bold", zorder=6)
        if freq > 0:
            ax.text(x, ly + 0.10, f"×{freq:,}", ha="center", va="bottom",
                fontsize=6.5, color="#78909C", zorder=6)
    else:
        ly = y - r - 0.12
        ax.text(x, ly - 0.55, short, ha="center", va="top",
            fontsize=8, fontweight="bold", color="#1A1A2E",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor=border, lw=0.7, alpha=0.92), zorder=6)
        ax.text(x, ly - 0.55 - 0.48, cls, ha="center", va="top",
            fontsize=6.5, color=border, fontweight="bold", zorder=6)
        if freq > 0:
            ax.text(x, ly - 0.15, f"×{freq:,}", ha="center", va="top",
                fontsize=6.5, color="#78909C", zorder=6)

# ── Legends ───────────────────────────────────────────────────────────────────
grp_hdl = [mpatches.Patch(facecolor=GROUP_FILL[g], edgecolor=GROUP_DARK[g],
    linewidth=1.5, label=g) for g in GROUP_ORDER if g != "Other"]
leg1 = ax.legend(handles=grp_hdl, title="Concept Category",
    title_fontsize=10, fontsize=9.5, loc="upper right",
    framealpha=0.97, edgecolor="#B0BEC5", handlelength=1.5,
    bbox_to_anchor=(0.99, 0.99), bbox_transform=ax.transAxes)
ax.add_artist(leg1)

rel_hdl = [mpatches.Patch(color=c, label=l)
    for l,c in PRED_C.items() if l != "related"]
ax.legend(handles=rel_hdl, title="Relation Type",
    title_fontsize=9.5, fontsize=8.5, loc="lower right",
    framealpha=0.97, edgecolor="#B0BEC5", handlelength=1.2, ncol=2,
    bbox_to_anchor=(0.99, 0.01), bbox_transform=ax.transAxes)

# ── Info boxes ────────────────────────────────────────────────────────────────
stats = (f"Full Dataset\n"
         f"━━━━━━━━━━━━━━━━━━━━\n"
         f"Files   : {len([f for f in os.listdir(DS_DIR) if f.endswith('.json')])}\n"
         f"Total   : {ds_total:,} records\n"
         f"Jeju    : {ds_jeju:,} records\n"
         f"━━━━━━━━━━━━━━━━━━━━\n"
         f"Concepts: {len(jC):,}\n"
         f"Relations: {len(jR):,}\n"
         f"━━━━━━━━━━━━━━━━━━━━\n"
         f"Ring 0  : {len(RING0)} seeds\n"
         f"Ring 1  : {len(RING1)} nodes")
ax.text(0.01, 0.01, stats, fontsize=9, va="bottom", ha="left",
    fontfamily="monospace", color="#1A237E", transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8EAF6",
              edgecolor="#3F51B5", lw=1.8, alpha=0.97))

geo = ("Jeju Island · South Korea\n"
       "━━━━━━━━━━━━━━━━━━━━━━━━\n"
       "Center   33°25′N 126°30′E\n"
       "Jeju City 33.51°N 126.52°E\n"
       "Seogwipo  33.25°N 126.56°E\n"
       "Mt. Halla 33.36°N 126.53°E\n"
       "Bounds: 33°11′–33°34′N\n"
       "        126°08′–126°58′E")
ax.text(0.18, 0.01, geo, fontsize=9, va="bottom", ha="left",
    fontfamily="monospace", color="#1A237E", transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD",
              edgecolor="#1565C0", lw=1.8, alpha=0.97))

# ── Title & footer ────────────────────────────────────────────────────────────
fig.suptitle(
    "Jeju Island Tourism Ontology — Ring 0 & Ring 1 Complete Tree\n"
    f"Source: {ds_jeju:,} Jeju records from full dataset  ·  "
    f"{len(jC):,} Jeju-only concepts  ·  {len(jR):,} relations  ·  "
    f"{len(RING0)} seeds + {len(RING1)} direct neighbours shown",
    fontsize=15, fontweight="bold", color="#1A1A2E", y=0.995)

fig.text(0.5, 0.002,
    "Non-Jeju entities excluded (Seoul · Singapore · Bangkok …)  ·  "
    "Node size ∝ dataset frequency  ·  "
    "Edge colour = relation type  ·  Sector background = concept category  ·  "
    "Jeju Tourism Ontology v0.7.0  ·  © 2026",
    ha="center", va="bottom", fontsize=8.5, color="#9E9E9E")

plt.tight_layout(rect=[0, 0.012, 1, 0.990])
out = OUTDIR / "jeju_ontology_tree_ring0_ring1.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {out}")
