"""
Jeju Tourism Ontology Q&A — ontology-grounded answers with real-time weather.
Uses local Llama 3.1-8B-Instruct (free).

Usage
-----
  python scripts/ontology_rules_qa.py extract              # one-time rule extraction
  python scripts/ontology_rules_qa.py show-rules
  python scripts/ontology_rules_qa.py query "your question"
  python scripts/ontology_rules_qa.py interactive
"""
from __future__ import annotations

# ── Silence everything BEFORE any framework imports ───────────────────────────
import os, sys, warnings, logging as _logging

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"   # force-override, not setdefault
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
_logging.disable(_logging.CRITICAL)               # nuclear option: silences ALL loggers

# ── Project root + .env ───────────────────────────────────────────────────────
PROJECT_ROOT = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
)
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def _load_env() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

import json
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Optional

from src.graph.graph_store import TourismGraphStore
from src.retrieval.concept_linker import ConceptLinker
from src.llm.client import LLMClient
from src.utils.jeju_guard import jeju_guard as _jeju_guard_shared

ONTOLOGY_VERSION = os.environ.get("ONTOLOGY_VERSION", "0.7.0")
RULES_PATH = PROJECT_ROOT / "artifacts" / "rules"  / "rules_auto.json"
ONT_DIR    = PROJECT_ROOT / "artifacts" / "ontologies" / ONTOLOGY_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# Weather — Open-Meteo (free, no API key)
# ══════════════════════════════════════════════════════════════════════════════

_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=33.5097&longitude=126.5219"
    "&current=temperature_2m,precipitation,windspeed_10m,weathercode,relative_humidity_2m"
    "&timezone=Asia%2FSeoul"
)

_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Rain showers", 82: "Heavy rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm",
}

def _wmo_desc(code: int) -> str:
    return _WMO.get(code, _WMO.get((code // 10) * 10, "Variable"))

def _activity_advice(code: int, wind_kmh: float, precip_mm: float) -> tuple[str, list[str]]:
    """Return (short condition label, list of recommended activity types)."""
    if code == 0 or code <= 3:
        if wind_kmh > 30:
            return "Windy & clear", ["inland hiking", "scenic drives", "cultural sites"]
        return "Sunny", ["beaches", "Olle trails", "Hallasan hiking", "cycling", "snorkeling"]
    if code <= 48:
        return "Foggy", ["Seongsan Ilchulbong", "mountain viewpoints", "indoor cafes"]
    if code <= 55:
        return "Drizzly", ["Manjanggul Cave", "Jeju Folk Museum", "Dongmun Market", "indoor restaurants"]
    if code <= 65 or precip_mm > 2:
        return "Rainy", ["Jeju Museum of Art", "Haenyeo show (indoor)", "Seongeup Folk Village", "shopping markets"]
    if code <= 75:
        return "Snowy", ["Hallasan snow views (lower trails)", "Jeju Stone Park", "indoor cultural sites"]
    if code <= 82:
        return "Showery", ["Jeju Botanical Garden (covered)", "aquarium", "Loveland", "indoor markets"]
    return "Stormy", ["stay indoors", "hotel spa", "local cafes", "museum visits"]


def fetch_jeju_weather() -> dict:
    """Fetch current Jeju weather. Returns {} on failure."""
    try:
        with urllib.request.urlopen(_WEATHER_URL, timeout=5) as resp:
            data = json.loads(resp.read())
        cur = data["current"]
        code     = int(cur["weathercode"])
        wind     = float(cur["windspeed_10m"])
        precip   = float(cur["precipitation"])
        cond, activities = _activity_advice(code, wind, precip)
        return {
            "temp_c":    cur["temperature_2m"],
            "humidity":  cur["relative_humidity_2m"],
            "wind_kmh":  wind,
            "precip_mm": precip,
            "code":      code,
            "condition": cond,
            "desc":      _wmo_desc(code),
            "activities": activities,
        }
    except Exception:
        return {}


def _weather_block(w: dict) -> str:
    if not w:
        return ""
    acts = ", ".join(w["activities"])
    return (
        f"Current Jeju Weather: {w['desc']} ({w['condition']}) | "
        f"{w['temp_c']}°C | Wind {w['wind_kmh']} km/h | "
        f"Humidity {w['humidity']}%\n"
        f"Best activities right now: {acts}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pattern:
    subject_class: str
    predicate:     str
    object_class:  str
    frequency:     int

    def __str__(self) -> str:
        return f"[{self.subject_class}] --{self.predicate}--> [{self.object_class}]: {self.frequency}"


@dataclass
class Rule:
    rule_id:            str
    description:        str
    condition:          str
    consequence:        str
    rule_type:          str
    applicable_classes: list[str]
    reasoning:          str
    source:             str   = "auto"
    confidence:         float = 0.9

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("rule_id","description","condition","consequence",
                 "rule_type","applicable_classes","reasoning","source","confidence")}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(**{k: d.get(k, v) for k, v in
                      [("rule_id",""),("description",""),("condition",""),
                       ("consequence",""),("rule_type","soft"),
                       ("applicable_classes",[]),("reasoning",""),
                       ("source","auto"),("confidence",0.9)]})


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Pattern Mining
# ══════════════════════════════════════════════════════════════════════════════

class PatternMiner:
    def mine(self, concepts: list[dict], relations: list[dict],
             min_freq: int = 5, top_k: int = 80) -> list[Pattern]:
        cls_lkp: dict[str, str] = {
            c["name"].lower(): c.get("concept_class", "") for c in concepts
        }
        ctr: Counter = Counter()
        for r in relations:
            sc   = cls_lkp.get(r.get("subject","").lower(), "")
            oc   = cls_lkp.get(r.get("object", "").lower(), "")
            pred = r.get("predicate","")
            if sc and oc and pred:
                ctr[(sc, pred, oc)] += 1
        out: list[Pattern] = []
        for (sc, pred, oc), freq in ctr.most_common(top_k):
            if freq < min_freq:
                break
            out.append(Pattern(sc, pred, oc, freq))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — LLM Rule Extraction
# ══════════════════════════════════════════════════════════════════════════════

_EXT_SYS = (
    "You are a knowledge graph expert. "
    "Output ONLY a valid JSON array — no markdown, no prose."
)

_EXT_TMPL = """\
Jeju tourism knowledge graph patterns (SubjectClass --predicate--> ObjectClass: count):
{patterns}

Generate exactly {n_rules} rules (AUTO_{start:03d} to AUTO_{end:03d}).
Return a JSON array only:
[
  {{
    "rule_id": "AUTO_{start:03d}",
    "description": "<15 words max>",
    "condition": "entity.class = X AND predicate EXISTS",
    "consequence": "INFER or SUGGEST or SUPPRESS or WARN result",
    "rule_type": "hard",
    "applicable_classes": ["ClassA"],
    "reasoning": "one sentence"
  }}
]"""


def _extract_json(text: str) -> list[dict]:
    text = text.strip()
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    data = json.loads(text[start: i + 1])
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
                break
    for obj in re.findall(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            d = json.loads(obj)
            if isinstance(d, dict) and "rule_id" in d:
                return [d]
        except json.JSONDecodeError:
            pass
    return []


class LLMRuleExtractor:
    BATCH_SIZE      = 15
    RULES_PER_BATCH = 5

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(self, patterns: list[Pattern]) -> list[Rule]:
        all_rules: list[Rule] = []
        seen: set[str] = set()
        ctr  = 1
        n_b  = (len(patterns) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for b, b0 in enumerate(range(0, len(patterns), self.BATCH_SIZE)):
            batch  = patterns[b0: b0 + self.BATCH_SIZE]
            prompt = _EXT_TMPL.format(
                patterns="\n".join(f"  {p}" for p in batch),
                n_rules=self.RULES_PER_BATCH,
                start=ctr, end=ctr + self.RULES_PER_BATCH - 1,
            )
            try:
                raw   = self.client.call(prompt, system=_EXT_SYS, max_tokens=1000)
                items = _extract_json(raw)
                new   = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if not all(k in item for k in ("rule_id","description","condition","consequence")):
                        continue
                    rule = Rule.from_dict(item)
                    rule.source = "auto"
                    if rule.rule_id and rule.rule_id not in seen:
                        seen.add(rule.rule_id)
                        all_rules.append(rule)
                        new += 1
                print(f"  Batch {b+1}/{n_b}: {new} rules")
            except Exception as exc:
                print(f"  Batch {b+1}/{n_b}: failed — {exc}")
            ctr += self.RULES_PER_BATCH
        return all_rules


# ══════════════════════════════════════════════════════════════════════════════
# Seed rules
# ══════════════════════════════════════════════════════════════════════════════

SEED_RULES: list[Rule] = [
    Rule("S001","Storm/rain → suppress outdoor water activities",
         "weather IN [heavy_rain, storm, typhoon]",
         "SUPPRESS beach_swimming AND WARN outdoor_activities",
         "hard",["Beach","NaturalAttraction","Weather"],
         "Safety: severe weather + sea = hazard","seed",1.0),
    Rule("S002","Halal/vegan traveler → filter restaurants to dietary match",
         "traveler.dietary IN [halal, kosher, vegan, vegetarian]",
         "FILTER restaurants WHERE supports_dietary_need MATCHES traveler.dietary",
         "hard",["Restaurant","DietaryRequirement","Cuisine"],
         "Dietary compliance","seed",1.0),
    Rule("S003","High-risk activity + family → suppress unless explicit",
         "traveler.type = family AND activity.has_risk.severity = high",
         "SUPPRESS activity UNLESS explicitly_requested",
         "soft",["Activity"],"Family safety","seed",1.0),
    Rule("S004","Windy + beach → warn about swimming conditions",
         "windspeed > 30kmh AND destination.class = Beach",
         "WARN swimming_conditions AND SUGGEST sheltered_beaches",
         "soft",["Beach","Season"],"Wind safety","seed",1.0),
    Rule("S005","Critical-risk attraction → always include safety warning",
         "entity.has_risk.severity = critical",
         "ALWAYS_INCLUDE safety_warning",
         "hard",["Attraction","NaturalAttraction"],"Critical safety","seed",1.0),
    Rule("S006","No ontology evidence → do not fabricate",
         "evidence_sources IS EMPTY",
         "RETURN uncertainty_statement INSTEAD OF fabricated_fact",
         "hard",[],"Grounding requirement","seed",1.0),
]

def save_rules(rules: list[Rule], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in rules], indent=2))

def load_rules(path: Path) -> list[Rule]:
    if not path.exists():
        return []
    return [Rule.from_dict(d) for d in json.loads(path.read_text())]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Rule-Grounded Q&A
# ══════════════════════════════════════════════════════════════════════════════

_QA_SYS = (
    "You are a Jeju Island tourism expert. "
    "Answer using ONLY the ontology context and rules provided. "
    "Only mention real Jeju places found in the ontology context. "
    "Never invent place names or facts not in the context. "
    "When weather data is provided, tailor your recommendation accordingly. "
    "Cite rule IDs when applying them (e.g. 'Per S001: ...')."
)

_QA_TMPL = """\
=== CURRENT JEJU WEATHER ===
{weather}

=== ONTOLOGY CONTEXT ===
{subgraph}

=== APPLICABLE RULES ===
{rules}

=== QUESTION ===
{question}

Answer concisely. Only name Jeju places from the context above. Apply HARD rules always. \
Mention SOFT rules as recommendations. If weather affects the answer, say so.\n\nAnswer:"""


class RuleGroundedQA:
    MAX_NODES = 50
    MAX_RELS  = 25
    MAX_RULES = 6

    _JEJU_KW = frozenset([
        "jeju","hallasan","seongsan","manjanggul","hyeopjae","hamdeok",
        "seogwipo","jungmun","cheonjiyeon","cheonjeyeon","olle","sanbangsan",
        "udo","marado","bijarim","yongduam","dongmun","hallim","aewol",
        "seongeup","haenyeo","dol hareubang","heuk dwaeji","woljeong",
        "gimnyeong","sangumburi","iho","hallabong","oreum",
    ])
    _UTIL_CLS = frozenset([
        "Season","Activity","SafetyAdvice","DietaryRequirement",
        "TransportMode","Weather","WeatherCondition","LocalCustom",
        "EmergencyService","TravelConstraint",
    ])
    _GEO_CLS = frozenset([
        "City","Island","Region","Destination","Beach","Attraction",
        "NaturalAttraction","CulturalAttraction","ReligiousSite",
        "Accommodation","Restaurant","Market","Park",
    ])
    _KW_CLS: dict[str, str] = {
        "beach":"Beach","beaches":"Beach",
        "restaurant":"Restaurant","food":"Restaurant",
        "eat":"Restaurant","dining":"Restaurant","halal":"Restaurant",
        "hotel":"Accommodation","stay":"Accommodation","accommodation":"Accommodation",
        "hike":"Activity","hiking":"Activity","trail":"Activity",
        "market":"Market","shopping":"Market",
        "waterfall":"NaturalAttraction","cave":"NaturalAttraction",
        "museum":"CulturalAttraction","park":"Attraction",
    }

    def __init__(self, graph: TourismGraphStore, rules: list[Rule], client: LLMClient) -> None:
        self.graph  = graph
        self.rules  = rules
        self.client = client
        self.linker = ConceptLinker(graph=graph)

    @classmethod
    def _is_jeju(cls, name: str) -> bool:
        nl = name.lower()
        return any(kw in nl for kw in cls._JEJU_KW)

    @classmethod
    def _allowed(cls, concept: dict) -> bool:
        name = concept.get("name", "")
        kls  = concept.get("concept_class", "")
        if kls in cls._UTIL_CLS:
            return True
        if kls in cls._GEO_CLS:
            return cls._is_jeju(name)
        return True

    def _kw_seeds(self, question: str) -> list[str]:
        q = question.lower()
        extra, seen = [], set()
        for kw, cls_name in self._KW_CLS.items():
            if kw in q:
                for c in self.graph.get_concepts_by_class(cls_name):
                    if self._is_jeju(c.get("name","")) and c["name"] not in seen:
                        seen.add(c["name"])
                        extra.append(c["name"])
        return extra[:10]

    def _subgraph(self, seeds: list[str]) -> tuple[list[dict], list[dict]]:
        visited:   set[str]   = set()
        concepts:  list[dict] = []
        relations: list[dict] = []
        frontier:  list[str]  = []

        for s in seeds:
            c = self.graph.get_concept(s)
            if c and c["name"].lower() not in visited and self._allowed(c):
                frontier.append(c["name"])
                visited.add(c["name"].lower())
                concepts.append(c)

        for _hop in range(2):
            if len(visited) >= self.MAX_NODES:
                break
            candidates: list[tuple[int,str,dict,dict]] = []
            for name in frontier:
                for r in (self.graph._out_edges.get(name,[]) +
                          self.graph._in_edges.get(name,[]))[:20]:
                    nbr = r["object"] if r["subject"] == name else r["subject"]
                    if nbr.lower() not in visited:
                        c = self.graph.get_concept(nbr)
                        if c:
                            candidates.append((int(self._is_jeju(nbr)), nbr, c, r))
            candidates.sort(key=lambda x: -x[0])
            nxt: list[str] = []
            for _, nbr, c, r in candidates:
                if len(visited) >= self.MAX_NODES:
                    break
                if nbr.lower() not in visited and self._allowed(c):
                    concepts.append(c); visited.add(nbr.lower()); nxt.append(nbr)
                if len(relations) < self.MAX_RELS * 3:
                    relations.append(r)
            frontier = nxt

        # Keep only relations where both endpoints were accepted
        seen_k: set[tuple] = set()
        deduped: list[dict] = []
        for r in relations:
            if r["subject"].lower() not in visited or r["object"].lower() not in visited:
                continue
            key = (r["subject"], r["predicate"], r["object"])
            if key not in seen_k:
                seen_k.add(key); deduped.append(r)
        deduped.sort(key=lambda r: r.get("provenance",{}).get("confidence",0), reverse=True)
        return concepts, deduped[:self.MAX_RELS]

    def _select_rules(self, classes: set[str], question: str, weather: dict) -> list[Rule]:
        q = question.lower()
        scored: list[tuple[float,Rule]] = []
        for rule in self.rules:
            overlap  = len(set(rule.applicable_classes) & classes)
            kw_hits  = sum(1 for kw in re.findall(r"\b\w{4,}\b", rule.description.lower()) if kw in q)
            priority = (3 if rule.rule_type=="hard" else 0) + (1 if rule.source=="seed" else 0)
            # Boost weather rules when conditions are bad
            if weather and rule.rule_id in ("S001","S004"):
                if weather.get("code",0) >= 61 or weather.get("wind_kmh",0) > 30:
                    priority += 4
            score = overlap * 3 + kw_hits * 2 + priority
            if score > 0:
                scored.append((score, rule))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:self.MAX_RULES]]

    def _fmt_subgraph(self, concepts: list[dict], relations: list[dict]) -> str:
        if not concepts:
            return "[No relevant Jeju ontology context found]"
        lines = [f"Concepts ({len(concepts)}):"]
        for c in concepts[:18]:
            cls = c.get("concept_class","?")
            als = c.get("aliases",[])[:1]
            lines.append(f"  • {c['name']} [{cls}]" + (f" | {als[0]}" if als else ""))
        if relations:
            lines.append(f"\nRelations ({len(relations)}):")
            for r in relations:
                conf = r.get("provenance",{}).get("confidence",0)
                lines.append(f"  {r['subject']} --[{r['predicate']}]--> {r['object']}  ({conf:.2f})")
        return "\n".join(lines)

    def _fmt_rules(self, rules: list[Rule]) -> str:
        if not rules:
            return "[No specific rules apply]"
        lines: list[str] = []
        for r in rules:
            tag = "HARD" if r.rule_type=="hard" else "SOFT"
            lines += [
                f"[{r.rule_id}] ({tag}) {r.description}",
                f"  IF: {r.condition}",
                f"  THEN: {r.consequence}",
            ]
        return "\n".join(lines)

    def stream_answer(self, question: str, weather: dict) -> None:
        """Stream answer to stdout, then print TTFT / TPOT metrics."""
        linked = self.linker.link(question, top_k=12)
        seeds: list[str] = []
        for lc in linked:
            c = lc.get("concept") if isinstance(lc, dict) else None
            if c and self._allowed(c):
                seeds.append(c["name"])

        jeju = self.graph.get_concept("Jeju")
        if jeju and jeju["name"] not in seeds:
            seeds.insert(0, jeju["name"])
        for s in self._kw_seeds(question):
            if s not in seeds:
                seeds.append(s)

        concepts, relations = self._subgraph(seeds)
        classes = {c.get("concept_class","") for c in concepts}
        rules   = self._select_rules(classes, question, weather)

        prompt = _QA_TMPL.format(
            weather=_weather_block(weather) or "Weather data unavailable.",
            subgraph=self._fmt_subgraph(concepts, relations),
            rules=self._fmt_rules(rules),
            question=question,
        )
        self.client.stream_call(prompt, system=_QA_SYS, max_tokens=350)

        # ── Print latency metrics ─────────────────────────────────────────────
        m = self.client.get_last_metrics()
        if m:
            ttft  = f"{m['ttft_ms']:,.0f} ms"  if m.get("ttft_ms")  is not None else "n/a"
            tpot  = f"{m['tpot_ms']:,.1f} ms/tok" if m.get("tpot_ms") is not None else "n/a"
            total = f"{m['total_ms']/1000:.1f} s"
            tps   = f"{m.get('throughput_tps', 0):.1f} tok/s"
            itok  = m.get("input_tokens", 0)
            otok  = m.get("output_tokens", 0)
            print(
                f"\n\n{'─'*58}\n"
                f"  TTFT {ttft}  │  TPOT {tpot}  │  {tps}\n"
                f"  Total {total}  │  In {itok} tok  │  Out {otok} tok\n"
                f"{'─'*58}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Jeju query guard — delegates to src.utils.jeju_guard (shared with API)
# ══════════════════════════════════════════════════════════════════════════════

def _jeju_guard(question: str) -> tuple[bool, str]:
    return _jeju_guard_shared(question)


# ══════════════════════════════════════════════════════════════════════════════
# LLM client factory — 4-bit, single GPU for speed
# ══════════════════════════════════════════════════════════════════════════════

def _make_client(max_tokens: int) -> LLMClient:
    return LLMClient(
        max_tokens=max_tokens,
        temperature=0.0,
        load_in_4bit=True,   # ~3x faster, fits in 5 GB VRAM
        gpu_ids=[0],          # single GPU — no multi-GPU overhead
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI commands
# ══════════════════════════════════════════════════════════════════════════════

def cmd_extract(argv: list[str]) -> None:
    force = "--force" in argv
    if RULES_PATH.exists() and not force:
        rules = load_rules(RULES_PATH)
        print(f"Rules already extracted ({len(rules)} rules). Use --force to redo.")
        return

    print(f"Loading ontology v{ONTOLOGY_VERSION}...")
    concepts  = json.loads((ONT_DIR / "concepts.json").read_text())
    relations = json.loads((ONT_DIR / "relations.json").read_text())
    print(f"  {len(concepts):,} concepts, {len(relations):,} relations\n")

    print("Mining patterns...")
    patterns = PatternMiner().mine(concepts, relations, min_freq=5, top_k=80)
    print(f"  {len(patterns)} patterns found\n")

    n_b = (len(patterns) + LLMRuleExtractor.BATCH_SIZE - 1) // LLMRuleExtractor.BATCH_SIZE
    print(f"Extracting rules via local LLM ({n_b} batches)...")
    client     = _make_client(max_tokens=1000)
    auto_rules = LLMRuleExtractor(client).extract(patterns)
    all_rules  = SEED_RULES + auto_rules
    save_rules(all_rules, RULES_PATH)
    print(f"\nDone: {len(all_rules)} rules saved ({len(SEED_RULES)} seed + {len(auto_rules)} auto)")


def cmd_show_rules(argv: list[str]) -> None:
    rules = load_rules(RULES_PATH)
    if not rules:
        print("No rules found. Run: python scripts/ontology_rules_qa.py extract")
        return
    ft = next((a for a in argv if a in ("hard","soft")), None)
    fs = next((a for a in argv if a in ("seed","auto")), None)
    shown = [r for r in rules
             if (ft is None or r.rule_type==ft) and (fs is None or r.source==fs)]
    print(f"\n{len(shown)}/{len(rules)} rules\n")
    print(f"{'ID':<12} {'TYPE':<5} {'SRC':<5}  Description")
    print("─" * 76)
    for r in shown:
        print(f"{r.rule_id:<12} {'HARD' if r.rule_type=='hard' else 'soft':<5} {r.source:<5}  {r.description}")


def _build_qa() -> tuple[RuleGroundedQA, dict]:
    rules = load_rules(RULES_PATH)
    if not rules:
        rules = list(SEED_RULES)
    graph  = TourismGraphStore.from_version(
        ONTOLOGY_VERSION, str(PROJECT_ROOT / "artifacts" / "ontologies")
    )
    client = _make_client(max_tokens=350)
    qa     = RuleGroundedQA(graph=graph, rules=rules, client=client)
    weather = fetch_jeju_weather()
    return qa, weather


def cmd_query(argv: list[str]) -> None:
    question = " ".join(argv).strip()
    if not question:
        print("Usage: python scripts/ontology_rules_qa.py query 'your question'")
        return

    allowed, msg = _jeju_guard(question)
    if not allowed:
        print(f"\nQ: {question}\n")
        print(f"A: {msg}\n")
        return

    qa, weather = _build_qa()
    if weather:
        print(f"\nJeju weather: {weather['desc']} | {weather['temp_c']}°C | "
              f"Wind {weather['wind_kmh']} km/h\n")
    print(f"Q: {question}\n")
    print("A: ", end="", flush=True)
    qa.stream_answer(question, weather)
    print()


def cmd_interactive(argv: list[str]) -> None:
    qa, weather = _build_qa()
    rules = load_rules(RULES_PATH) or SEED_RULES
    auto  = sum(1 for r in rules if r.source == "auto")

    print("\n" + "═" * 58)
    print("  Jeju Tourism Assistant  (Ontology + Local LLM)")
    print("═" * 58)
    if weather:
        print(f"  Weather : {weather['desc']} | {weather['temp_c']}°C | "
              f"Wind {weather['wind_kmh']} km/h")
        print(f"  Best now: {', '.join(weather['activities'][:3])}")
    print(f"  Rules   : {len(rules)} ({len(SEED_RULES)} seed + {auto} auto)")
    print(f"  Type 'quit' to exit\n" + "─" * 58)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        allowed, msg = _jeju_guard(question)
        if not allowed:
            print(f"\nAssistant: {msg}")
            print("─" * 58)
            continue
        print("\nAssistant: ", end="", flush=True)
        qa.stream_answer(question, weather)
        print("\n" + "─" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

_CMDS = {
    "extract":     cmd_extract,
    "show-rules":  cmd_show_rules,
    "query":       cmd_query,
    "interactive": cmd_interactive,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        print(__doc__)
        print(f"Commands: {', '.join(_CMDS)}")
        sys.exit(1)
    _CMDS[sys.argv[1]](sys.argv[2:])
