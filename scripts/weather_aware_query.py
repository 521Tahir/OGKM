"""
Weather-Aware Jeju Tourism Planner (KMA-integrated)
====================================================
Takes a tourist-plan query (optionally with a date), fetches the Jeju
weather forecast from KMA (live if KMA_API_KEY + internet; else Jeju
climatology), and returns:
  - the weather condition for that date,
  - activated ontology constraint rules,
  - weather-based recommendations (what to do / avoid / indoor backups).

Usage (VS Code terminal):
  cd /home/tahirahmad/Ontology_Project
  set -a && source .env && set +a

  # Auto-detect date from the query text
  python scripts/weather_aware_query.py --query "Plan a beach day in Jeju in August"

  # Explicit date + region
  python scripts/weather_aware_query.py \
      --query "Is it good for hiking Hallasan?" --date 2026-10-15 --region "jeju city"

  # Interactive
  python scripts/weather_aware_query.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.jeju_guard import jeju_guard
from src.graph.graph_store import TourismGraphStore
from src.integrations.weather_client import KMAWeatherClient, JEJU_GRID
from src.generation.weather_recommender import recommend

GRN="\033[92m"; YLW="\033[93m"; RED="\033[91m"; CYN="\033[96m"
BLD="\033[1m"; RST="\033[0m"; MGN="\033[95m"; BLU="\033[94m"
DIV = f"{BLD}{'═'*72}{RST}"
THIN = f"{'─'*72}"

_MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,
    "august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sep":9,
    "sept":9,"oct":10,"nov":11,"dec":12,
}

def extract_date(query: str) -> tuple[str | None, str]:
    """Extract a target date from query text. Returns (YYYY-MM-DD or None, how)."""
    q = query.lower()
    today = datetime.now()

    # ISO date
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)
    if m:
        return m.group(0), "explicit ISO date"

    # "August 15", "15 August", "Aug 20"
    for name, mon in _MONTHS.items():
        m = re.search(rf"\b{name}\s+(\d{{1,2}})\b", q)
        if m:
            day = int(m.group(1))
            year = today.year if mon >= today.month else today.year + 1
            try:
                return datetime(year, mon, min(day, 28)).strftime("%Y-%m-%d"), f"{name} {day}"
            except ValueError:
                pass
    # bare month name → 15th of that month
    for name, mon in _MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            year = today.year if mon >= today.month else today.year + 1
            return datetime(year, mon, 15).strftime("%Y-%m-%d"), f"{name} (mid-month)"

    # relative
    if "today" in q:        return today.strftime("%Y-%m-%d"), "today"
    if "tomorrow" in q:     return (today+timedelta(days=1)).strftime("%Y-%m-%d"), "tomorrow"
    if "this weekend" in q:
        days = (5 - today.weekday()) % 7
        return (today+timedelta(days=days)).strftime("%Y-%m-%d"), "this weekend"
    if "next week" in q:    return (today+timedelta(days=7)).strftime("%Y-%m-%d"), "next week"

    # seasons → representative month
    if "summer" in q:   return datetime(today.year, 8, 1).strftime("%Y-%m-%d"), "summer"
    if "winter" in q:   return datetime(today.year, 1, 15).strftime("%Y-%m-%d"), "winter"
    if "spring" in q:   return datetime(today.year, 4, 15).strftime("%Y-%m-%d"), "spring"
    if "autumn" in q or "fall" in q: return datetime(today.year,10,15).strftime("%Y-%m-%d"), "autumn"

    return None, "no date found → using today"

def detect_topics(query: str) -> dict:
    q = query.lower()
    return {k: v for k, v in {
        "beach":   any(w in q for w in ["beach","swim","snorkel","coast"]),
        "hiking":  any(w in q for w in ["hike","hiking","trail","hallasan","trek","climb"]),
        "itinerary": any(w in q for w in ["plan","itinerary","trip","day","schedule"]),
        "outdoor": any(w in q for w in ["outdoor","tour","sightsee","walk","oreum"]),
    }.items() if v}

def detect_region(query: str) -> str:
    q = query.lower()
    for region in JEJU_GRID:
        if region in q:
            return region
    return "jeju city"

def print_report(query, date_str, how, region, rec):
    fc = rec.forecast
    print(f"\n{DIV}")
    print(f"{BLD}  WEATHER-AWARE JEJU PLAN{RST}")
    print(f"{THIN}")
    print(f"  Query  : {query}")
    print(f"  Date   : {date_str}  ({how})")
    print(f"  Region : {region.title()}")
    print(f"  Source : {GRN if fc.source=='kma_live' else YLW}{fc.source}{RST}"
          f"  {'(live KMA API)' if fc.source=='kma_live' else '(Jeju climatology fallback)'}")

    print(f"\n{MGN}{BLD}  ▌ WEATHER CONDITION{RST}")
    print(f"  {fc.summary()}")
    if fc.note:
        print(f"  {CYN}{fc.note}{RST}")

    suit_col = {"excellent":GRN,"good":GRN,"fair":YLW,"poor":RED}.get(rec.suitability, YLW)
    print(f"\n{MGN}{BLD}  ▌ OUTDOOR SUITABILITY{RST}")
    print(f"  {suit_col}{BLD}{rec.suitability.upper()}{RST}")

    if rec.activated_rules:
        print(f"\n{MGN}{BLD}  ▌ ACTIVATED CONSTRAINT RULES (ontology){RST}")
        for r in rec.activated_rules:
            print(f"  • {r}")

    if rec.warnings:
        print(f"\n{MGN}{BLD}  ▌ WARNINGS{RST}")
        for w in rec.warnings:
            print(f"  {RED}⚠{RST} {w}")

    if rec.recommended:
        print(f"\n{MGN}{BLD}  ▌ RECOMMENDED (weather-appropriate){RST}")
        for r in rec.recommended:
            print(f"  {GRN}✓{RST} {r}")

    if rec.discouraged:
        print(f"\n{MGN}{BLD}  ▌ AVOID / WEATHER-DEPENDENT{RST}")
        for d in rec.discouraged:
            print(f"  {RED}✗{RST} {d}")

    if rec.indoor_alternatives:
        print(f"\n{MGN}{BLD}  ▌ INDOOR ALTERNATIVES{RST}")
        for a in rec.indoor_alternatives:
            print(f"  {BLU}⌂{RST} {a}")
    print(f"\n{DIV}")

def run_once(query, date_arg, region_arg, graph, client, save):
    ok, msg = jeju_guard(query)
    if not ok:
        print(f"\n  {RED}⛔ BLOCKED: {msg}{RST}")
        print("  This planner only answers Jeju Island questions.")
        return

    if date_arg:
        date_str, how = date_arg, "explicit --date"
    else:
        date_str, how = extract_date(query)
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

    region = region_arg or detect_region(query)
    topics = detect_topics(query)

    forecast = client.get_forecast(date=date_str, region=region)
    rec = recommend(forecast, query=query, graph=graph, topics=topics)
    print_report(query, date_str, how, region, rec)

    if save:
        out = {"query": query, "date": date_str, "region": region,
               "recommendation": rec.to_dict()}
        p = Path("artifacts/reports") / f"weather_plan_{int(datetime.now().timestamp())}.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"  {GRN}Saved → {p}{RST}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query",  type=str, default=None)
    ap.add_argument("--date",   type=str, default=None, help="YYYY-MM-DD (overrides query date)")
    ap.add_argument("--region", type=str, default=None, help=f"one of: {', '.join(JEJU_GRID)}")
    ap.add_argument("--save",   action="store_true")
    args = ap.parse_args()

    print(f"\n{DIV}")
    print(f"{BLD}{'  JEJU WEATHER-AWARE TOURISM PLANNER (KMA-integrated)':^72}{RST}")
    print(DIV)

    print(f"\n  Loading ontology...", end=" ", flush=True)
    graph = TourismGraphStore()
    graph.load(
        json.loads(Path("artifacts/ontologies/1.0.0/concepts.json").read_text()),
        json.loads(Path("artifacts/ontologies/1.0.0/relations.json").read_text()),
    )
    print(f"{GRN}ready{RST}")

    client = KMAWeatherClient()  # reads KMA_API_KEY from env; falls back to climatology
    mode = "LIVE KMA API" if client.enable_live else "Jeju climatology (offline fallback)"
    print(f"  Weather source: {GRN if client.enable_live else YLW}{mode}{RST}")
    if not client.enable_live:
        print(f"  {CYN}Tip: set KMA_API_KEY in .env (free at data.go.kr) to enable live forecasts.{RST}")

    if args.query:
        run_once(args.query, args.date, args.region, graph, client, args.save)
        return

    print(f"\n  {BLD}Enter a Jeju tourism query (with a date/month) or 'quit':{RST}")
    while True:
        print(f"  {YLW}> {RST}", end="", flush=True)
        try:
            q = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {YLW}Goodbye!{RST}"); break
        if q.lower() in ("quit","exit","q"):
            print(f"  {YLW}Goodbye!{RST}"); break
        if not q:
            continue
        run_once(q, None, args.region, graph, client, args.save)

if __name__ == "__main__":
    main()
