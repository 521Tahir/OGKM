"""
Jeju Tourism Ontology — Clean hub-and-spoke subgraph, slide-ready.
Seeds centre · Ring-1 (1-hop) · Ring-2 (2-hop) with sector colouring.
Only Jeju Island concepts — no Seoul / Singapore / Bangkok etc.
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import numpy as np

try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable,"-m","pip","install","networkx","-q"])
    import networkx as nx

ROOT   = Path(__file__).resolve().parents[1]
ONT    = ROOT / "artifacts" / "ontologies" / "0.7.0"
OUTDIR = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(ONT / "concepts.json")  as f: ALL_C = json.load(f)
with open(ONT / "relations.json") as f: ALL_R = json.load(f)

# ── Jeju-only filter ──────────────────────────────────────────────────────────
NON_JEJU = {
    "seoul","busan","incheon","daegu","daejeon","gwangju","suwon","jeonju",
    "yeosu","gyeongju","ulsan","changwon","pohang","andong","gangneung",
    "sokcho","chuncheon","cheongju","wonju","gunsan","mokpo","suncheon",
    "jinju","geoje","gangnam","hongdae","itaewon","myeongdong","dongdaemun",
    "gyeongbokgung","namsan tower","bukhansan","bukchon","insadong",
    "lotte world","coex","gyeonggi","gyeongsang","chungcheong","jeolla",
    "gangwon","dmz","singapore","tokyo","osaka","kyoto","hiroshima","beijing",
    "shanghai","guangzhou","hong kong","taipei","taiwan","macau","bangkok",
    "phuket","chiang mai","bali","jakarta","hanoi","ho chi minh","saigon",
    "da nang","kuala lumpur","manila","cebu","boracay",
    "google maps","kakaomaps","airbnb","booking.com","agoda","tripadvisor",
    "grab","gojek","uber",
}
JEJU_KW = {
    "jeju","hallasan","halla","seogwipo","hyeopjae","hamdeok","jungmun",
    "seongsan","ilchulbong","manjanggul","osulloc","samseonghyeol","olle",
    "haenyeo","udo","marado","bijarim","aewol","hallim","hanrim","songaksan",
    "sanbangsan","yongduam","jusangjeolli","cheonjiyeon","cheonjeyeon",
    "dongmun","woljeong","gimnyeong","loveland","ecoland","seopjikoji",
    "pyoseon","namwon","camellia hill","halla arboretum","hallim park",
    "jeju folk","jeju international airport","jeju haenyeo","jeju volcanic",
    "teddy bear museum","jeju pearl","jeju black pork","hallabong",
    "horse park","jeju olle","gwakji","jocheon","seongeup","oreum",
}
GENERIC_CLS = {"Activity","Season","WeatherCondition","TravelerType",
               "DietaryRequirement","SafetyAdvice","TravelConstraint",
               "LocalCustom","Itinerary","TransportMode","Route"}
INCL_CLS = {"Attraction","NaturalAttraction","CulturalAttraction","Beach",
            "Mountain","Trail","Park","Landmark","ReligiousSite","Museum",
            "Market","Accommodation","Restaurant","Cuisine","Destination",
            "Region","City","Island"}
GENERIC_KW = {"beach","hiking","waterfall","cave","lava","volcanic","oreum",
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
print(f"Jeju-only: {len(jC):,} concepts, {len(jR):,} relations")

# ── Adjacency + relation lookup ───────────────────────────────────────────────
adj = defaultdict(set)
rel_lookup = defaultdict(list)
for r in jR:
    s, o, p = r["subject"], r["object"], r["predicate"]
    adj[s].add(o); adj[o].add(s)
    rel_lookup[(s,o)].append(p)
    rel_lookup[(o,s)].append(p)

PRIORITY = ["located_in","has_activity","near","part_of","is_a","served_at",
            "suitable_for","reachable_by","best_for","available_in",
            "requires","connected_to","has_seasonality","has_risk"]

def best_pred(u, v):
    preds = rel_lookup.get((u,v), []) + rel_lookup.get((v,u), [])
    for p in PRIORITY:
        if p in preds: return p
    return preds[0] if preds else "related"

def directed_uv(u, v):
    """Return (u,v) in the direction the relation is stored."""
    if (u,v) in rel_lookup: return u, v
    return v, u

# ── Node selection ────────────────────────────────────────────────────────────
SEEDS = [s for s in ["Hallasan","Jeju","Hallasan National Park",
                     "Seongsan Ilchulbong","Jeju City"] if s in jM]

r1_score = {}
for seed in SEEDS:
    for nb in adj.get(seed,set()):
        if nb not in SEEDS and nb in jM:
            r1_score[nb] = r1_score.get(nb,0) + len(adj.get(nb,set()))
R1 = sorted(r1_score, key=lambda n:-r1_score[n])[:24]
r1_set = set(R1)

r2_score = {}
for n1 in R1:
    for nb in adj.get(n1,set()):
        if nb not in SEEDS and nb not in r1_set and nb in jM:
            r2_score[nb] = r2_score.get(nb,0) + len(adj.get(nb,set()))
R2 = sorted(r2_score, key=lambda n:-r2_score[n])[:22]

all_nodes  = SEEDS + R1 + R2
n_layer    = {n:0 for n in SEEDS} | {n:1 for n in R1} | {n:2 for n in R2}
print(f"Nodes: {len(SEEDS)} seeds + {len(R1)} ring-1 + {len(R2)} ring-2 = {len(all_nodes)}")

# ── Class → style ─────────────────────────────────────────────────────────────
CLS_CFG = {
    "Beach":             ("#0288D1","#01579B","Geography"),
    "Mountain":          ("#43A047","#1B5E20","Geography"),
    "Trail":             ("#AED581","#388E3C","Geography"),
    "NaturalAttraction": ("#00BCD4","#006064","Geography"),
    "Park":              ("#8BC34A","#33691E","Geography"),
    "Attraction":        ("#1E88E5","#0D47A1","Geography"),
    "Destination":       ("#1565C0","#0D47A1","Geography"),
    "Region":            ("#42A5F5","#1565C0","Geography"),
    "City":              ("#4FC3F7","#0277BD","Geography"),
    "Island":            ("#26C6DA","#006064","Geography"),
    "Landmark":          ("#7E57C2","#4527A0","Culture"),
    "CulturalAttraction":("#FFB300","#E65100","Culture"),
    "ReligiousSite":     ("#FF8F00","#BF360C","Culture"),
    "Museum":            ("#EF6C00","#E65100","Culture"),
    "Market":            ("#E91E63","#880E4F","Culture"),
    "LocalCustom":       ("#DCEDC8","#558B2F","Culture"),
    "Accommodation":     ("#8D6E63","#3E2723","Services"),
    "Restaurant":        ("#E53935","#B71C1C","Services"),
    "Cuisine":           ("#F06292","#880E4F","Services"),
    "TransportMode":     ("#78909C","#263238","Services"),
    "Route":             ("#90A4AE","#37474F","Services"),
    "DietaryRequirement":("#AED581","#33691E","Services"),
    "Activity":          ("#66BB6A","#1B5E20","Activity"),
    "TravelerType":      ("#CE93D8","#6A1B9A","Activity"),
    "Itinerary":         ("#FFF176","#F57F17","Activity"),
    "Season":            ("#FFD54F","#F9A825","Temporal"),
    "WeatherCondition":  ("#CFD8DC","#546E7A","Temporal"),
    "SafetyAdvice":      ("#EF9A9A","#C62828","Safety"),
    "TravelConstraint":  ("#D7CCC8","#4E342E","Safety"),
    "EmergencyService":  ("#EF5350","#B71C1C","Safety"),
}
DEF_CFG = ("#E0E0E0","#757575","Other")
GROUP_ORDER  = ["Geography","Culture","Services","Activity","Temporal","Safety","Other"]
GROUP_FILL   = {"Geography":"#E3F2FD","Culture":"#FFF8E1","Services":"#F3E5F5",
                "Activity":"#E8F5E9","Temporal":"#FFFDE7","Safety":"#FFEBEE","Other":"#F5F5F5"}
GROUP_DARK   = {"Geography":"#1565C0","Culture":"#E65100","Services":"#6A1B9A",
                "Activity":"#2E7D32","Temporal":"#F57F17","Safety":"#C62828","Other":"#757575"}

def get_style(name):
    cls = jM[name].get("concept_class","Other") if name in jM else "Other"
    return CLS_CFG.get(cls, DEF_CFG)   # (fill, border, group)

# ── Layout radii ──────────────────────────────────────────────────────────────
RC = 0.15   # seed cluster radius
R1r = 1.0   # ring-1 radius
R2r = 2.0   # ring-2 radius

def grp(n): return get_style(n)[2]

R1_sorted = sorted(R1, key=lambda n: GROUP_ORDER.index(grp(n)) if grp(n) in GROUP_ORDER else 99)
R2_sorted = sorted(R2, key=lambda n: GROUP_ORDER.index(grp(n)) if grp(n) in GROUP_ORDER else 99)

np.random.seed(5)
pos = {}
ang_s = np.linspace(0, 2*math.pi, len(SEEDS), endpoint=False)
for n, a in zip(SEEDS, ang_s):
    pos[n] = (RC*math.cos(a), RC*math.sin(a))

ang_1 = np.linspace(math.pi/10, math.pi/10+2*math.pi, len(R1_sorted), endpoint=False)
for n, a in zip(R1_sorted, ang_1):
    j = np.random.uniform(-0.05,0.05)
    pos[n] = ((R1r+j)*math.cos(a), (R1r+j)*math.sin(a))

ang_2 = np.linspace(0, 2*math.pi, len(R2_sorted), endpoint=False)
for n, a in zip(R2_sorted, ang_2):
    j = np.random.uniform(-0.07,0.07)
    pos[n] = ((R2r+j)*math.cos(a), (R2r+j)*math.sin(a))

# ── Edge selection ────────────────────────────────────────────────────────────
edges = []
def add_edge(u, v):
    pred = best_pred(u,v)
    a, b = directed_uv(u,v)
    edges.append((a, b, pred))

# seed → ring-1
for seed in SEEDS:
    for n1 in R1:
        if n1 in adj.get(seed,set()):
            add_edge(seed, n1)

# ring-1 → ring-2 (one per ring-2 node)
for n2 in R2:
    for n1 in R1:
        if n1 in adj.get(n2,set()):
            add_edge(n1, n2); break

# ring-1 ↔ ring-1 (max 18, skip is_a to reduce clutter)
r1r1 = 0
for i, a in enumerate(R1_sorted):
    for b in R1_sorted[i+1:]:
        if b in adj.get(a,set()) and r1r1 < 18:
            p = best_pred(a,b)
            if p != "is_a":
                add_edge(a, b); r1r1 += 1

print(f"Edges: {len(edges)}")

# ── Build graph ───────────────────────────────────────────────────────────────
G = nx.DiGraph()
for n in all_nodes:
    f, b, g = get_style(n)
    G.add_node(n, fill=f, border=b, grp=g, layer=n_layer[n])
for u, v, pred in edges:
    if u in G and v in G:
        G.add_edge(u, v, pred=pred)

# ── Figure ────────────────────────────────────────────────────────────────────
PRED_C = {"located_in":"#1565C0","has_activity":"#2E7D32","near":"#7B1FA2",
           "part_of":"#00695C","is_a":"#90A4AE","suitable_for":"#E65100",
           "reachable_by":"#C62828","best_for":"#F9A825","served_at":"#AD1457",
           "requires":"#37474F","available_in":"#0097A7","connected_to":"#558B2F",
           "has_seasonality":"#5C6BC0","has_risk":"#B71C1C"}

fig, ax = plt.subplots(figsize=(24, 22))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off"); ax.set_aspect("equal")

# ── Sector wedge backgrounds ──────────────────────────────────────────────────
group_ang = defaultdict(list)
for n, a in zip(R1_sorted, ang_1):
    group_ang[grp(n)].append(math.degrees(a))

for g, angs in group_ang.items():
    if len(angs) < 1: continue
    a_lo, a_hi = min(angs)-10, max(angs)+10
    if a_hi - a_lo > 340: continue
    w = Wedge((0,0), R2r+0.35, a_lo, a_hi, width=R2r+0.25,
              facecolor=GROUP_FILL.get(g,"#F5F5F5"), alpha=0.14,
              edgecolor=GROUP_DARK.get(g,"#BDBDBD"), linewidth=0.8, linestyle="--")
    ax.add_patch(w)
    # Sector label
    mid = math.radians((a_lo+a_hi)/2)
    lx = (R2r+0.55)*math.cos(mid)
    ly = (R2r+0.55)*math.sin(mid)
    ax.text(lx, ly, g, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=GROUP_DARK.get(g,"#455A64"), alpha=0.80,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=GROUP_FILL.get(g,"#F5F5F5"),
                      edgecolor=GROUP_DARK.get(g,"#BDBDBD"),
                      linewidth=0.8, alpha=0.90))

# Guide circles
for r_, ls, al in [(R1r,"--",0.25),(R2r,":",0.18)]:
    ax.add_patch(plt.Circle((0,0), r_, fill=False, color="#B0BEC5",
                              linestyle=ls, linewidth=1.0, alpha=al))

# ── Edges ─────────────────────────────────────────────────────────────────────
for u, v, data in G.edges(data=True):
    pred = data.get("pred","")
    col  = PRED_C.get(pred,"#BDBDBD")
    lu, lv = G.nodes[u]["layer"], G.nodes[v]["layer"]
    s_edge = lu==0 or lv==0
    lw = 2.8 if s_edge else (1.4 if lu==1 and lv==1 else 1.0)
    al = 0.85 if s_edge else (0.65 if lu==1 and lv==1 else 0.40)
    rad = 0.14 if lu==lv else 0.05
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], ax=ax,
        edge_color=col, width=lw, alpha=al,
        connectionstyle=f"arc3,rad={rad}",
        arrows=True, arrowsize=15, arrowstyle="-|>",
        min_source_margin=18, min_target_margin=18)

# ── Nodes ─────────────────────────────────────────────────────────────────────
SZ = {0:5500, 1:1600, 2:520}
LW = {0:4.0,  1:2.4,  2:1.6}
for n in G.nodes():
    l = G.nodes[n]["layer"]
    nx.draw_networkx_nodes(G, pos, nodelist=[n], ax=ax,
        node_color=[G.nodes[n]["fill"]],
        edgecolors=[G.nodes[n]["border"]],
        node_size=[SZ[l]], linewidths=LW[l])

# Seed outline glow
for n in SEEDS:
    if n in G:
        x,y = pos[n]
        ax.add_patch(plt.Circle((x,y), 0.19, fill=False,
                                  color="#1565C0", lw=2.2, alpha=0.28))

# ── Labels ────────────────────────────────────────────────────────────────────
for n in SEEDS:
    if n in G:
        x,y = pos[n]
        ax.text(x, y, n, ha="center", va="center", fontsize=10,
                fontweight="bold", color="#0D1117",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.78))

for n in R1_sorted:
    if n in G:
        x,y = pos[n]; a = math.atan2(y,x)
        ox = x + 0.24*math.cos(a); oy = y + 0.24*math.sin(a)
        ha = "left" if math.cos(a)>=0 else "right"
        short = (n if len(n)<=22 else n[:20]+"…")
        ax.text(ox, oy, short, ha=ha, va="center",
                fontsize=8.5, fontweight="bold", color="#1A237E",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.82))

for n in R2_sorted:
    if n in G:
        x,y = pos[n]; a = math.atan2(y,x)
        ox = x + 0.22*math.cos(a); oy = y + 0.22*math.sin(a)
        ha = "left" if math.cos(a)>=0 else "right"
        short = (n if len(n)<=18 else n[:16]+"…")
        ax.text(ox, oy, short, ha=ha, va="center",
                fontsize=7.0, color="#37474F",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.80))

# ── Seed-edge labels ──────────────────────────────────────────────────────────
for u, v, data in G.edges(data=True):
    if G.nodes[u]["layer"]==0 or G.nodes[v]["layer"]==0:
        pred = data.get("pred","")
        ux,uy = pos[u]; vx,vy = pos[v]
        mx,my = (ux+vx)/2+0.03, (uy+vy)/2+0.03
        ax.text(mx, my, pred, fontsize=5.5, color=PRED_C.get(pred,"#546E7A"),
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                          alpha=0.78, edgecolor="none"))

# ── Ring labels ───────────────────────────────────────────────────────────────
ax.text(0, -0.22, "Seeds\n(Ring 0)", ha="center", va="top",
        fontsize=8, color="#90A4AE", alpha=0.7, style="italic")
ax.text(0, -(R1r+0.12), "Ring 1 · Direct neighbours",
        ha="center", va="top", fontsize=8, color="#90A4AE", alpha=0.7, style="italic")
ax.text(0, -(R2r+0.14), "Ring 2 · Extended neighbourhood",
        ha="center", va="top", fontsize=8, color="#90A4AE", alpha=0.7, style="italic")

# ── Info boxes ────────────────────────────────────────────────────────────────
geo = ("Jeju Island · South Korea\n"
       "━━━━━━━━━━━━━━━━━━━━━━━━\n"
       "Center   : 33°25′N  126°30′E\n"
       "Jeju City: 33.51°N  126.52°E\n"
       "Seogwipo : 33.25°N  126.56°E\n"
       "Mt. Halla: 33.36°N  126.53°E\n"
       "Bounds:  33°11′–33°34′N\n"
       "         126°08′–126°58′E")
ax.text(-2.62,-2.48, geo, fontsize=9, va="bottom", ha="left",
        fontfamily="monospace", color="#1A237E",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#E3F2FD",
                  edgecolor="#1565C0", linewidth=2.0, alpha=0.97))

stats = (f"Jeju-only Ontology\n"
         f"━━━━━━━━━━━━━━━━━━━\n"
         f"Concepts : {len(jC):,}\n"
         f"Relations: {len(jR):,}\n"
         f"━━━━━━━━━━━━━━━━━━━\n"
         f"Ring 0   : {len(SEEDS)} seeds\n"
         f"Ring 1   : {len(R1)} nodes\n"
         f"Ring 2   : {len(R2)} nodes\n"
         f"Edges    : {len(G.edges())}")
ax.text(1.72,-2.48, stats, fontsize=9, va="bottom", ha="left",
        fontfamily="monospace", color="#1B5E20",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#E8F5E9",
                  edgecolor="#2E7D32", linewidth=2.0, alpha=0.97))

# ── Legends (using axes-fraction coordinates) ─────────────────────────────────
grp_hdl = [mpatches.Patch(facecolor=GROUP_FILL[g], edgecolor=GROUP_DARK[g],
                           linewidth=1.5, label=g, alpha=0.85)
           for g in GROUP_ORDER if g!="Other"]
grp_hdl += [mpatches.Patch(facecolor="#1565C0", edgecolor="#0D47A1",
                             linewidth=3.0, label="★  Seed node (Ring 0)")]
leg1 = ax.legend(handles=grp_hdl, title="Concept Category",
    title_fontsize=9.5, fontsize=9, loc="upper left", framealpha=0.97,
    edgecolor="#B0BEC5", handlelength=1.5, ncol=1,
    bbox_to_anchor=(0.01, 0.99), bbox_transform=ax.transAxes)
ax.add_artist(leg1)

rel_hdl = [mpatches.Patch(color=c, label=l) for l,c in PRED_C.items()]
ax.legend(handles=rel_hdl, title="Relation Type",
    title_fontsize=9.5, fontsize=8.5, loc="upper right", framealpha=0.97,
    edgecolor="#B0BEC5", handlelength=1.2, ncol=1,
    bbox_to_anchor=(0.99, 0.99), bbox_transform=ax.transAxes)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Jeju Island Tourism Ontology — Knowledge Connection Graph\n"
    f"Seeds: {', '.join(SEEDS[:5])}  ·  "
    f"{len(G.nodes())} concepts  ·  {len(G.edges())} relations  ·  "
    "Jeju Island only",
    fontsize=16, fontweight="bold", color="#1A1A2E", pad=14)

fig.text(0.5, 0.006,
    "Non-Jeju places excluded (Seoul · Singapore · Bangkok · …)  ·  "
    "Sector colour = concept category  ·  Edge colour = relation type  ·  "
    "Jeju Tourism Ontology v0.7.0  ·  © 2026",
    ha="center", va="bottom", fontsize=8.5, color="#9E9E9E")

ax.text(0, 0, "JEJU\nISLAND", ha="center", va="center",
        fontsize=88, color="#1565C0", alpha=0.022,
        fontweight="bold", rotation=20)

ax.set_xlim(-2.72, 2.72); ax.set_ylim(-2.72, 2.72)
out = OUTDIR / "subgraph_Hallasan_Jeju.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
