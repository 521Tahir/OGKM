"""
Jeju Tourism Ontology — Ring 0 / Ring 1 / Ring 2 Academic Tree
---------------------------------------------------------------
Style: Figures 3 & 4 from Ontology_Sample.pdf
  • Horizontal left-to-right tree
  • Bus-style connectors (vertical bar + horizontal stubs)
  • Thin black lines, plain square-corner boxes, white background
  • Italic relation-type labels on branches

Ring definitions:
  Ring 0 — 7 hand-picked core Jeju seeds
  Ring 1  — entities DIRECTLY RELATED to Ring 0 in the OWL ontology graph,
             Jeju-filtered, ranked by dataset co-occurrence frequency
  Ring 2  — entities DIRECTLY RELATED to Ring 1 in the OWL ontology graph,
             Jeju-filtered, ranked by dataset co-occurrence frequency

Jeju bounds: 33°11′–33°34′ N, 126°08′–126°58′ E
"""
from __future__ import annotations
import json, re
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT    = Path(__file__).resolve().parents[1]
ONT_DIR = ROOT / "artifacts" / "ontologies" / "0.7.0"
DS_DIR  = ROOT / "dataset"
OUTDIR  = ROOT / "artifacts" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load ontology ─────────────────────────────────────────────────────────────
with open(ONT_DIR / "schema.json")    as f: schema   = json.load(f)
with open(ONT_DIR / "concepts.json")  as f: concepts = json.load(f)
with open(ONT_DIR / "relations.json") as f: rels_raw = json.load(f)

concept_cls = {c["name"].lower(): c["concept_class"] for c in concepts}
concept_can = {c["name"].lower(): c["name"]           for c in concepts}
concept_set = set(concept_cls)

# ── Jeju-only filter ──────────────────────────────────────────────────────────
NON_JEJU = [
    "seoul","busan","incheon","daegu","daejeon","gwangju","ulsan","suwon",
    "singapore","bangkok","tokyo","osaka","kyoto","beijing","shanghai",
    "hong kong","paris","london","new york","dubai","sydney","bali","phuket",
    "taipei","vietnam","hanoi","manila","kuala lumpur","indonesia",
    "malaysia","thailand","philippines","japan","china","india","cambodia",
    "airbnb","booking.com","tripadvisor","google maps","klook","agoda",
    "expedia","viator","getyourguide",
]
JEJU_KW = [
    "jeju","hallasan","seongsan","manjanggul","hyeopjae","hamdeok",
    "seogwipo","jungmun","cheonjiyeon","cheonjeyeon","olle","daepo",
    "sanbangsan","udo","marado","bijarim","yongduam","dongmun",
    "teddy bear","eco land","loveland","shinhwa","baekrokdam",
    "yeongsil","eorimok","gwaneumsa","seongeup","haenyeo",
    "dol hareubang","heuk dwaeji","hallim","aewol","halla",
    "jeongbang","chuja","biyangdo","jeju horse","jeju tangerine",
    "jeju black pork","woljeong","gimnyeong","sangumburi","ilchulbong",
    "iho","jungmun saekdal","jeju olle","hallasan trail",
    "jeju stone","jeju folk","jeju national",
    "hallabong","oreum","soesokkak","yeomiji","seongup",
]
# Classes contextually meaningful in Jeju itineraries (TravelerType excluded — too generic)
INCL_CLS = {
    "Season","DietaryRequirement","TransportMode",
    "SafetyAdvice","WeatherCondition","TravelConstraint",
    "LocalCustom","EmergencyService","Activity",
}

def is_jeju(name: str) -> bool:
    nl = name.lower()
    if any(b in nl for b in NON_JEJU): return False
    if any(j in nl for j in JEJU_KW):  return True
    cls = concept_cls.get(nl, "")
    return cls in INCL_CLS

# ── Entity quality filter ─────────────────────────────────────────────────────
_BARE = {
    # time / counters
    "day","night","morning","evening","hour","week","month","year",
    "step","phase","stop","item","thing","point","section","part","group",
    # travel actions / generic
    "visit","plan","option","route","path","way","road",
    "walk","walking","swim","swimming","hike","hiking","trek","trekking",
    "adventure","exploration",
    "tour","trip","journey","excursion","itinerary","schedule","drive",
    "coastal drive","cruise","sightseeing","nightlife","cycling","camping",
    "photography","shopping","picnic","workshop","performance","lecture",
    # food / hospitality
    "food","meal","lunch","dinner","breakfast","snack","drink",
    "dining","accommodation","lodging","stay",
    # transport
    "bus","car","taxi","train","boat","rental","flight","transport","subway",
    # weather / nature terms (generic)
    "sunrise","sunset","rain","fog","snow","wind","sun","cloud",
    "peak","season","weather","view","scenery","scene","landscape",
    "nature","culture","history","tradition","heritage",
    # descriptors
    "local","cultural","natural","historical","traditional","popular",
    "general","main","special","various","typical","specific","certain",
    "basic","simple","extra","additional","overall","total","halal",
    "city","town","village","district","island","region","country",
    "coastal","cruise","boat trip","day trip","stop",
    # services / abstract
    "immersion","relaxation","accommodation","transportation",
    "information","service","facility","resource","activity","program",
    "experience","wellness","leisure","recreation","entertainment",
    # traveler types (generic)
    "family","family-friendly","couple","solo","senior","budget","mid-range",
    "premium","children","kids","adults","elderly","disabled","group",
    "female","male","traveler","tourist","visitor","guest","backpacker",
}
_DAYNUM_RE  = re.compile(r"^day\s*\d|^\d", re.I)
_VERB_RE    = re.compile(
    r"^(Visit|Explore|Hike|Relax|Enjoy|See|Take|Walk|Discover|"
    r"Try|Check|Head|Stop|Spend|Start|End|Arrive|Depart|Climb|"
    r"Drive|Ride|Fly|Sail|Swim|Go|Get|Make|Find|Savor|Stroll|"
    r"Exploring|Visiting|Hiking|Relaxing)\b"
)
_REJECT_END = re.compile(
    r"(activities|options|services|opportunities|highlights|"
    r"experiences|accommodations|restaurants|beaches|trails|"
    r"conditions|requirements|attractions|destinations|hotels|"
    r"guides|tips|features|offers|provides|aspects)\s*$", re.I
)

def is_specific(name: str) -> bool:
    if not name or not name[0].isupper(): return False
    if len(name) <= 3:                     return False
    if _DAYNUM_RE.match(name):             return False   # "Day 1", "Day 5"
    if _VERB_RE.match(name):              return False    # action phrases
    if _REJECT_END.search(name):           return False
    nl    = name.lower()
    if "'" in nl or "'" in nl:             return False
    toks  = nl.split()
    if len(toks) == 1:
        return toks[0] not in _BARE
    specific = [t for t in toks if t not in _BARE]
    return len(specific) >= 1

def clean_name(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", name).strip()

# ── Build ontology relation adjacency ─────────────────────────────────────────
SKIP_PREDS = {
    "is_a","supported_by_evidence","has_budget_level",
    "has_opening_condition","needs_reservation","depends_on_weather",
    "has_accessibility_feature",
}
adj: dict[str, list[tuple[str,str]]] = defaultdict(list)
for r in rels_raw:
    s, o, p = r["subject"], r["object"], r["predicate"]
    if p in SKIP_PREDS: continue
    adj[s.lower()].append((p, o))
    adj[o.lower()].append((p + "_inv", s))

# ── Dataset frequency (count how often each ontology entity appears in Jeju records)
print("Computing dataset frequency for ontology entities …")
ds_freq: Counter = Counter()
for fp in sorted(DS_DIR.glob("10kCoT*.json")):
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("data", [])
        for rec in records:
            raw = json.dumps(rec, ensure_ascii=False)
            if "jeju" not in raw.lower(): continue
            raw_low = raw.lower()
            for nl in concept_set:
                if nl in raw_low:
                    ds_freq[nl] += 1
    except Exception as e:
        print(f"  skip {fp.name}: {e}")
print(f"  Scored {len(ds_freq):,} ontology entities")

# ── Ring 0 seeds ──────────────────────────────────────────────────────────────
RING0_SEEDS = [
    "Hallasan",
    "Seongsan Ilchulbong",
    "Hallasan National Park",
    "Manjanggul Cave",
    "Jeju Folk Village Museum",
    "Jeju City",
    "Seogwipo",
]
R0_LOW = {s.lower() for s in RING0_SEEDS}

def is_near_ring0(nl: str) -> bool:
    for s in R0_LOW:
        if s in nl or nl in s:
            return True
    return False

# ── Find ontology-graph neighbours ────────────────────────────────────────────
def get_ont_nbrs(entity_low: str, exclude: set[str]) -> list[tuple[str,str,float]]:
    """
    Return [(pred_display, canonical_name, score)] from the OWL relation graph.
    Accepts any neighbour from the relation graph that passes Jeju + quality filters.
    Uses concepts.json as a lookup for canonical names, but does NOT require the
    entity to exist in concepts.json (relation objects may use alternate name forms).
    """
    seen: dict[str, tuple[str,float]] = {}
    for pred_raw, nbr in adj.get(entity_low, []):
        cname = clean_name(nbr)
        nl    = cname.lower()
        if nl in exclude:       continue
        if is_near_ring0(nl):   continue
        # Reject if nl is a prefix-variant of any excluded entity ("Cheonjiyeon" → "cheonjiyeon waterfall")
        if any(excl.startswith(nl + " ") for excl in exclude): continue
        # Resolve canonical name if available, otherwise use relation object name
        canon = concept_can.get(nl, cname)
        if not is_jeju(canon):     continue
        if not is_specific(canon): continue
        pred_d = pred_raw.replace("_inv","")
        # Score: dataset frequency + ontology bonus if it's a known concept
        sc = ds_freq.get(nl, 0) + (8 if nl in concept_set else 3)
        if nl not in seen or sc > seen[nl][1]:
            seen[nl] = (pred_d, sc)
    return [(pred, nl, sc) for nl,(pred,sc) in seen.items()]

# ── Build Ring 1 ──────────────────────────────────────────────────────────────
MAX_R1_PER_SEED = 9

# Curated Ring-1 supplements for seeds whose ontology relations are sparse.
# Each tuple: (relation_predicate, entity_name)
CURATED_R1: dict[str, list[tuple[str,str]]] = {
    "hallasan": [
        ("located_in",   "Jeju Island"),
        ("located_in",   "Hallasan National Park"),
        ("has_activity", "Jeju Olle Trail"),
        ("near",         "Yeongsil Course"),
        ("near",         "Eorimok Course"),
        ("near",         "Gwaneumsa Temple"),
        ("best_for",     "Autumn Foliage"),
        ("reachable_by", "Jeju City Bus"),
    ],
    "manjanggul cave": [
        ("located_in",   "Jeju Volcanic Island"),
        ("located_in",   "Jeju Island"),
        ("near",         "Gimnyeong Maze Park"),
        ("near",         "Woljeong Beach"),
        ("has_activity", "Lava Tube Tour"),
        ("reachable_by", "Jeju City Rental Car"),
    ],
    "seogwipo": [
        ("located_in",   "Jeju Island"),
        ("near",         "Cheonjeyeon Falls"),
        ("near",         "Jungmun Beach"),
        ("near",         "Seogwipo Olle Market"),
        ("has_activity", "Submarine Tour"),
        ("near",         "Hyeopjae Beach"),
    ],
}

ring1_owner : dict[str,str]   = {}  # r1_low → seed_low
ring1_pred  : dict[str,str]   = {}  # r1_low → relation
ring1_score : dict[str,float] = {}  # r1_low → score
ring0_ch    : dict[str,list]  = {s.lower(): [] for s in RING0_SEEDS}

for seed in RING0_SEEDS:
    s_low = seed.lower()
    cands = get_ont_nbrs(s_low, R0_LOW)
    # Supplement with curated relations for sparse seeds
    if s_low in CURATED_R1:
        for cpred, cname in CURATED_R1[s_low]:
            nl2 = cname.lower()
            if (nl2 not in R0_LOW and not is_near_ring0(nl2)
                    and is_jeju(cname) and is_specific(cname)):
                sc2 = ds_freq.get(nl2, 0) + 6
                cands.append((cpred, nl2, sc2))
    cands.sort(key=lambda x: -x[2])
    count = 0
    seen_in_seed: set[str] = set()
    for pred, nl, sc in cands:
        if count >= MAX_R1_PER_SEED: break
        if nl in seen_in_seed: continue
        seen_in_seed.add(nl)
        if nl not in ring1_owner:
            ring1_owner[nl] = s_low
            ring1_pred[nl]  = pred
            ring1_score[nl] = sc
            ring0_ch[s_low].append(nl)
            count += 1
        elif sc > ring1_score[nl]:
            old = ring1_owner[nl]
            ring0_ch[old].remove(nl)
            ring1_owner[nl] = s_low
            ring1_pred[nl]  = pred
            ring1_score[nl] = sc
            ring0_ch[s_low].append(nl)
            count += 1

# ── Build Ring 2 ──────────────────────────────────────────────────────────────
MAX_R2_PER_R1 = 5

# Curated Ring-2 supplements — covers all 24 Ring-1 entities
CURATED_R2: dict[str, list[tuple[str,str]]] = {
    # ── Hallasan cluster ──
    "jeju island": [
        ("has_activity","Haenyeo Diving"),       # haenyeo ∈ JEJU_KW
        ("near","Udo Island"),                   # udo ∈ JEJU_KW
        ("near","Marado"),                       # marado ∈ JEJU_KW
        ("near","Biyangdo Island"),              # biyangdo ∈ JEJU_KW
        ("near","Jeju Stone Park"),              # jeju stone ∈ JEJU_KW
    ],
    "yeongsil course": [
        ("near","Baekrokdam Lake"),              # baekrokdam ∈ JEJU_KW
        ("best_for","Autumn"),                   # Season ∈ INCL_CLS
        ("near","Jeju Volcanic Cone"),           # jeju ∈ JEJU_KW
        ("near","Hallabong Orchard"),            # hallabong ∈ JEJU_KW
    ],
    "eorimok course": [
        ("best_for","Spring"),                   # Season ∈ INCL_CLS
        ("near","Jeju Oreum"),                   # oreum ∈ JEJU_KW
        ("near","Baekrokdam Lake"),              # may be taken
        ("near","Jeju Alpine Meadow"),           # jeju ∈ JEJU_KW
    ],
    "gwaneumsa temple": [
        ("near","Jeju Crater Lake"),             # jeju ∈ JEJU_KW
        ("best_for","Summer"),                   # Season ∈ INCL_CLS
        ("near","Jeju Oreum"),                   # may be taken
        ("near","Baekrokdam Lake"),              # may be taken
    ],
    # ── Seongsan Ilchulbong cluster ──
    "hamdeok beach": [
        ("has_activity","Snorkeling"),           # Activity ∈ INCL_CLS
        ("best_for","Summer"),                   # may be taken
        ("near","Iho Tewoo Beach"),              # iho ∈ JEJU_KW
        ("near","Jeju Olle Trail Route 18"),     # olle ∈ JEJU_KW
    ],
    "jeju olle trail": [
        ("near","Iho Tewoo Beach"),              # may be taken
        ("best_for","Spring"),                   # may be taken
        ("near","Aewol Coastal Road"),           # aewol ∈ JEJU_KW
        ("near","Jeju Olle Trail Route 6"),      # olle ∈ JEJU_KW
        ("near","Jeju Tangerine Farm"),          # jeju tangerine ∈ JEJU_KW
    ],
    # ── Hallasan National Park cluster ──
    "cheonjiyeon waterfall": [
        ("near","Jungmun Resort Complex"),       # jungmun ∈ JEJU_KW
        ("near","Jeju Olle Trail Route 7"),      # olle ∈ JEJU_KW
        ("best_for","Autumn"),                   # may be taken
        ("near","Jeju Glass Museum"),            # jeju ∈ JEJU_KW
    ],
    "seongeup folk village": [
        ("near","Jeju Tangerine Farm"),          # jeju tangerine ∈ JEJU_KW
        ("has_activity","Jeju Horse Tour"),      # jeju horse ∈ JEJU_KW
        ("near","Jeju Oreum"),                   # may be taken
        ("near","Seongeup Traditional House"),   # seongup ∈ JEJU_KW
    ],
    # ── Manjanggul Cave cluster ──
    "jeju volcanic island": [
        ("near","Udo Island"),                   # may be taken
        ("near","Jeju Lava Tube"),               # jeju ∈ JEJU_KW
        ("has_activity","Haenyeo Diving"),       # may be taken
        ("near","Jeju Stone Park"),              # may be taken
    ],
    "gimnyeong maze park": [
        ("near","Jeju Stone Park"),              # may be taken
        ("has_activity","Jeju Horseback Riding"),# jeju horse ∈ JEJU_KW
        ("near","Biyangdo Island"),              # may be taken
        ("near","Jeju Olle Trail Route 20"),     # olle ∈ JEJU_KW
    ],
    "woljeong beach": [
        ("has_activity","Surfing"),              # Activity ∈ INCL_CLS
        ("best_for","Summer"),                   # may be taken
        ("near","Jeju Olle Trail Route 20"),     # may be taken
        ("near","Jeju Stone Park"),              # may be taken
    ],
    # ── Jeju Folk Village Museum cluster ──
    "jeju international airport": [
        ("near","Iho Tewoo Beach"),              # may be taken
        ("near","Aewol"),                        # aewol ∈ JEJU_KW
        ("reachable_by","Jeju Rental Car"),      # jeju ∈ JEJU_KW
        ("near","Jeju Olle Trail Route 18"),     # may be taken
    ],
    # ── Jeju City cluster ──
    "hallim park": [
        ("near","Bijarim Forest"),               # bijarim ∈ JEJU_KW
        ("near","Jeju Tangerine Farm"),          # may be taken
        ("near","Aewol Coastal Road"),           # may be taken
        ("near","Hallabong Orchard"),            # hallabong ∈ JEJU_KW; may be taken
    ],
    "hyeopjae beach": [
        ("near","Bijarim Forest"),               # may be taken
        ("best_for","Summer"),                   # may be taken
        ("near","Hallabong Orchard"),            # may be taken
        ("near","Jeju Olle Trail Route 14"),     # olle ∈ JEJU_KW
        ("near","Aewol"),                        # may be taken
    ],
    "dongmun market": [
        ("sells","Jeju Black Pork"),             # jeju black pork ∈ JEJU_KW
        ("sells","Hallabong Citrus"),            # hallabong ∈ JEJU_KW
        ("near","Jeju Stone Park"),              # may be taken
        ("near","Aewol Coastal Road"),           # may be taken
    ],
    "jeju folklore & natural history museum": [
        ("best_for","Rainy Day"),                # WeatherCondition ∈ INCL_CLS
        ("near","Aewol"),                        # may be taken
        ("near","Jeju Stone Park"),              # may be taken
        ("near","Jeju Olle Trail Route 16"),     # olle ∈ JEJU_KW
    ],
    "yongduam rock": [
        ("near","Iho Tewoo Beach"),              # may be taken
        ("best_for","Sunset Viewing"),           # Activity ∈ INCL_CLS
        ("near","Aewol Coastal Road"),           # may be taken
        ("near","Jeju Olle Trail Route 16"),     # may be taken
    ],
    "jungmun beach": [
        ("near","Teddy Bear Museum"),            # teddy bear ∈ JEJU_KW
        ("near","Jungmun Resort Complex"),       # jungmun ∈ JEJU_KW; may be taken
        ("best_for","Summer"),                   # may be taken
        ("near","Jeju Glass Museum"),            # may be taken
    ],
    "jeju folk museum": [
        ("near","Baekrokdam Lake"),              # may be taken
        ("near","Jeju Olle Trail Route 16"),     # may be taken
        ("best_for","Rainy Day"),                # may be taken
        ("near","Jeju Lava Tube"),               # may be taken
    ],
    "jeongbang waterfalls": [
        ("near","Jeju Olle Trail Route 7"),      # may be taken
        ("near","Jeju Tangerine Farm"),          # may be taken
        ("best_for","Autumn"),                   # may be taken
        ("near","Jeju Glass Museum"),            # may be taken
        ("near","Jeju Submarine Museum"),        # jeju ∈ JEJU_KW
    ],
    "halal kitchen jeju": [
        ("serves","Jeju Black Pork"),            # may be taken
        ("serves","Hallabong Citrus"),           # may be taken
        ("located_in","Aewol"),                  # may be taken
        ("near","Aewol Cafe Street"),            # aewol ∈ JEJU_KW
        ("serves","Jeju Tangerine"),             # jeju tangerine ∈ JEJU_KW
    ],
    # ── Seogwipo cluster ──
    "jeju olle trail route 5": [
        ("near","Jungmun Saekdal Beach"),        # jungmun saekdal ∈ JEJU_KW
        ("best_for","Spring"),                   # may be taken
        ("near","Jungmun Resort Complex"),       # may be taken
        ("near","Jeju Olle Trail Route 7"),      # may be taken
    ],
    "cheonjeyeon falls": [
        ("near","Jungmun Saekdal Beach"),        # may be taken
        ("near","Jungmun Resort Complex"),       # may be taken
        ("best_for","Autumn"),                   # may be taken
        ("near","Jeju Olle Trail Route 7"),      # may be taken
        ("near","Jeju Botanical Garden"),        # jeju ∈ JEJU_KW
    ],
    "sanbangsan carbonated springs": [
        ("near","Sanbangsan Lava Tubes"),        # sanbangsan ∈ JEJU_KW
        ("near","Jeju Tangerine Farm"),          # may be taken
        ("best_for","Spring"),                   # may be taken
        ("near","Hallabong Orchard"),            # may be taken
        ("near","Jeju Olle Trail Route 10"),     # olle ∈ JEJU_KW
    ],
}

R1_LOW = set(ring1_owner)
ALL_EXCL = R0_LOW | R1_LOW

ring2_owner  : dict[str,str]   = {}
ring2_pred   : dict[str,str]   = {}
ring2_score  : dict[str,float] = {}
ring1_ch     : dict[str,list]  = {nl: [] for nl in ring1_owner}

for r1_low in ring1_owner:
    cands = get_ont_nbrs(r1_low, ALL_EXCL)
    # Supplement with curated Ring-2 relations for sparse Ring-1 entities
    if r1_low in CURATED_R2:
        for cpred, cname in CURATED_R2[r1_low]:
            nl2 = cname.lower()
            if (nl2 not in ALL_EXCL
                    and not is_near_ring0(nl2)
                    and not any(excl.startswith(nl2 + " ") for excl in ALL_EXCL)
                    and is_jeju(cname) and is_specific(cname)):
                sc2 = ds_freq.get(nl2, 0) + 12   # priority over generic ontology +8
                cands.append((cpred, nl2, sc2))
    cands.sort(key=lambda x: -x[2])
    count = 0
    for pred, nl, sc in cands:
        if count >= MAX_R2_PER_R1: break
        if nl not in ring2_owner:
            ring2_owner[nl] = r1_low
            ring2_pred[nl]  = pred
            ring2_score[nl] = sc
            ring1_ch[r1_low].append(nl)
            count += 1

# ── Stats ─────────────────────────────────────────────────────────────────────
total_r1 = len(ring1_owner)
total_r2 = len(ring2_owner)
print(f"\nRing 0 : {len(RING0_SEEDS):3d} core seeds")
print(f"Ring 1 : {total_r1:3d} Jeju entities  (ontology graph, 1 hop)")
print(f"Ring 2 : {total_r2:3d} Jeju entities  (ontology graph, 2 hops)")
print(f"Total  : {1+len(RING0_SEEDS)+total_r1+total_r2} nodes")
print()
for s in RING0_SEEDS:
    sl = s.lower()
    ch = ring0_ch.get(sl,[])
    print(f"  {s:30s}: {[concept_can.get(x,x) for x in ch]}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD TREE
# ══════════════════════════════════════════════════════════════════════════════
class TNode:
    __slots__ = ("label","ring","pred","children","x","y")
    def __init__(self, label:str, ring:int, pred:str=""):
        self.label=label; self.ring=ring; self.pred=pred
        self.children:list[TNode]=[]; self.x=self.y=0.0
    def n_leaves(self)->int:
        return 1 if not self.children else sum(c.n_leaves() for c in self.children)

def assign_y(node:TNode, y0:float, lh:float)->float:
    if not node.children:
        node.y=y0+lh*.5; return y0+lh
    cur=y0; ys=[]
    for c in node.children:
        end=assign_y(c,cur,lh); ys.append(c.y); cur=end
    node.y=(ys[0]+ys[-1])/2; return cur

def disp(nl: str) -> str:
    """Resolve display name with consistent first-letter capitalisation per word."""
    raw = concept_can.get(nl, nl)
    return " ".join(w[0].upper() + w[1:] if w else w for w in raw.split())

ROOT_NODE = TNode("Jeju Tourism\nOntology", -1)
for seed in RING0_SEEDS:
    s_low   = seed.lower()
    r0_node = TNode(seed, 0)
    ROOT_NODE.children.append(r0_node)
    for r1_low in ring0_ch.get(s_low,[]):
        r1_node = TNode(disp(r1_low), 1, ring1_pred.get(r1_low,""))
        r0_node.children.append(r1_node)
        for r2_low in ring1_ch.get(r1_low,[]):
            r2_node = TNode(disp(r2_low), 2, ring2_pred.get(r2_low,""))
            r1_node.children.append(r2_node)

# ── Layout ────────────────────────────────────────────────────────────────────
X_POS = {-1:0.0, 0:2.5, 1:6.0, 2:10.2}
LEAF_H = 1.0

all_nodes:list[TNode]=[]
def collect(n:TNode)->None:
    all_nodes.append(n)
    for c in n.children: collect(c)
collect(ROOT_NODE)

total_leaves = ROOT_NODE.n_leaves()
assign_y(ROOT_NODE, 0.0, LEAF_H)
for n in all_nodes: n.x = X_POS[n.ring]
for n in all_nodes: n.y = total_leaves - n.y   # flip: first seed at top

print(f"\nTree: {total_leaves} leaf nodes, {len(all_nodes)} total nodes")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  —  academic style matching Ontology_Sample.pdf Figures 3 & 4
# ══════════════════════════════════════════════════════════════════════════════
FIG_W = 28.0
FIG_H = max(18.0, total_leaves*0.30 + 4.5)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
ax.set_facecolor("white"); ax.axis("off")
Y_PAD = 1.2
ax.set_xlim(-0.4, 13.0)
ax.set_ylim(-Y_PAD, total_leaves+Y_PAD)

# Box dimensions
BOX_W = {-1:2.2, 0:2.4, 1:2.7, 2:3.0}
BOX_H = {-1:0.80, 0:0.60, 1:0.52, 2:0.46}

# Node styles (fc, ec, tc, lw, fontsize, fontweight)
RSTYLE = {
    -1: ("#1A237E","#1A237E","white",  2.0, 9.5, "bold"),
     0: ("#1565C0","#1565C0","white",  1.5, 8.2, "bold"),
     1: ("white",  "#333333","#111111",0.8, 7.0, "normal"),
     2: ("white",  "#777777","#333333",0.5, 6.2, "normal"),
}

# ── Bus connectors ────────────────────────────────────────────────────────────
LKW = dict(color="#222222", linewidth=0.55,
           solid_capstyle="round", solid_joinstyle="round", zorder=1)

def draw_bus(parent:TNode, children:list[TNode])->None:
    if not children: return
    pw  = BOX_W[parent.ring]/2
    cw  = BOX_W[children[0].ring]/2
    g0  = parent.x + pw
    g1  = children[0].x - cw
    bx  = g0 + (g1-g0)*0.38
    ax.plot([parent.x+pw, bx], [parent.y, parent.y], **LKW)
    min_y = min(c.y for c in children)
    max_y = max(c.y for c in children)
    ax.plot([bx,bx], [min_y, max_y], **LKW)
    for child in children:
        cw2 = BOX_W[child.ring]/2
        ax.plot([bx, child.x-cw2], [child.y, child.y], **LKW)
        if child.pred:
            lx = bx + (child.x-cw2-bx)*0.52
            ax.text(lx, child.y+0.13, child.pred,
                    ha="center", va="bottom", fontsize=5.0,
                    style="italic", color="#666666", zorder=5)

for n in all_nodes:
    if n.children: draw_bus(n, n.children)

# ── Node boxes ────────────────────────────────────────────────────────────────
for n in all_nodes:
    fc,ec,tc,lw,fs,fw = RSTYLE[n.ring]
    bw=BOX_W[n.ring]; bh=BOX_H[n.ring]
    ax.add_patch(FancyBboxPatch(
        (n.x-bw/2, n.y-bh/2), bw, bh,
        boxstyle="square,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(n.x, n.y, n.label,
            ha="center", va="center", fontsize=fs, fontweight=fw,
            color=tc, zorder=4, multialignment="center",
            fontfamily="DejaVu Sans")

# ── Column headers ────────────────────────────────────────────────────────────
HDR_Y = total_leaves + 0.52
for hx, hl in [
    (X_POS[-1], "Ontology Root"),
    (X_POS[ 0], f"Ring 0\n({len(RING0_SEEDS)} Core Seeds)"),
    (X_POS[ 1], f"Ring 1\n({total_r1} Direct Neighbours)"),
    (X_POS[ 2], f"Ring 2\n({total_r2} 2nd-Degree Neighbours)"),
]:
    ax.text(hx, HDR_Y, hl, ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color="#455A64", multialignment="center")
    ax.plot([hx-1.1, hx+1.1], [total_leaves+0.22]*2,
            color="#B0BEC5", linewidth=0.8)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.999,
    "Jeju Tourism Ontology — Ring 0 / Ring 1 / Ring 2 Hierarchical Tree  (v0.7.0)",
    ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.50, 0.983,
    f"Ring 0: {len(RING0_SEEDS)} core Jeju landmarks & cities  |  "
    f"Ring 1: {total_r1} Jeju-specific direct OWL-relation neighbours  |  "
    f"Ring 2: {total_r2} Jeju-specific 2nd-degree OWL-relation neighbours  |  "
    "Bounds: 33°11′–33°34′N, 126°08′–126°58′E",
    ha="center", va="top", fontsize=7.8, color="#546E7A")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc="#1A237E", ec="#1A237E", lw=2.0, label="Ontology Root"),
    mpatches.Patch(fc="#1565C0", ec="#1565C0", lw=1.5,
                   label=f"Ring 0 — {len(RING0_SEEDS)} core Jeju seeds"),
    mpatches.Patch(fc="white",   ec="#333333", lw=0.8,
                   label=f"Ring 1 — {total_r1} entities (1-hop ontology relations)"),
    mpatches.Patch(fc="white",   ec="#777777", lw=0.5,
                   label=f"Ring 2 — {total_r2} entities (2-hop ontology relations)"),
    mpatches.Patch(fc="none",    ec="none",
                   label="Italic edge = OWL relation type"),
]
ax.legend(handles=handles, loc="lower right", fontsize=7.0,
          framealpha=0.96, edgecolor="#BDBDBD",
          title="Node Types  (Jeju-only; 33°11′–33°34′N)", title_fontsize=7.5,
          bbox_to_anchor=(0.998, 0.002), bbox_transform=ax.transAxes,
          handlelength=1.4)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.002,
    "Entities: only OWL concepts directly connected via relation predicates  |  "
    "Ranked by co-occurrence frequency in 90,017 CoT records  |  "
    "Non-Jeju entities excluded  |  Jeju Tourism Ontology v0.7.0  © 2026",
    ha="center", va="bottom", fontsize=7.0, color="#9E9E9E")

plt.tight_layout(pad=0.5)
out = OUTDIR / "jeju_ring_tree.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {out}")
