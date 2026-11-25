"""
Weather-Aware Recommendation Engine.

Given a KMA WeatherForecast (live or climatology) and a user's query, this
module:
  1. Activates ontology constraint rules that depend on weather
     (R001 storm→suppress swimming, R006 monsoon→warn, plus typhoon/seasonal).
  2. Produces weather-appropriate Jeju activity/attraction recommendations,
     drawn from the ontology where possible.
  3. Returns a structured, explainable WeatherRecommendation object.

It is deliberately ontology-grounded: when a TourismGraphStore is supplied,
recommended attractions are pulled from real ontology concepts (by class),
matching the project's "no fabrication" constraint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..integrations.weather_client import WeatherForecast
from ..utils.logging_utils import get_logger

logger = get_logger("generation.weather_recommender")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class WeatherRecommendation:
    forecast:           WeatherForecast
    suitability:        str               # excellent | good | fair | poor
    activated_rules:    list[str] = field(default_factory=list)
    warnings:           list[str] = field(default_factory=list)
    recommended:        list[str] = field(default_factory=list)   # do these
    discouraged:        list[str] = field(default_factory=list)   # avoid these
    indoor_alternatives:list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "forecast": self.forecast.to_dict(),
            "suitability": self.suitability,
            "activated_rules": self.activated_rules,
            "warnings": self.warnings,
            "recommended": self.recommended,
            "discouraged": self.discouraged,
            "indoor_alternatives": self.indoor_alternatives,
        }


# ── Recommendation engine ─────────────────────────────────────────────────────

# Static, curated Jeju fallbacks per weather regime (used if no ontology supplied
# or ontology lacks a class). All are real Jeju places/activities.
_INDOOR_JEJU = [
    "Jeju Folk Village Museum", "Jeju National Museum", "O'sulloc Tea Museum",
    "Manjanggul Lava Tube (cave, weather-independent)", "Jeju Haenyeo Museum",
    "Aqua Planet Jeju aquarium", "Spirited Garden indoor sections",
]
_RAINY_OK = [
    "Manjanggul Lava Tube", "Jeju Folk Village Museum", "O'sulloc Tea Museum",
    "Aqua Planet Jeju", "local black-pork restaurants", "café touring in Aewol",
]
_CLEAR_WARM = [
    "Hyeopjae Beach", "Hamdeok Beach", "Hallasan day hike (check trail status)",
    "Seongsan Ilchulbong sunrise", "Udo Island ferry tour", "Jeju Olle coastal trails",
]
_COLD_CLEAR = [
    "Hallasan winter hiking (with gear)", "Sanbangsan", "Seongsan Ilchulbong",
    "Camellia Hill", "Spirited Garden", "Jeju Folk Village Museum",
]


def recommend(
    forecast: WeatherForecast,
    query: str = "",
    graph=None,                      # optional TourismGraphStore
    topics: Optional[dict] = None,   # optional detected query topics
) -> WeatherRecommendation:
    """
    Build a weather-aware recommendation from a forecast.

    If `graph` (TourismGraphStore) is provided, attraction suggestions are
    drawn from real ontology concepts; otherwise curated Jeju fallbacks
    are used.
    """
    topics = topics or {}
    rec = WeatherRecommendation(forecast=forecast, suitability="good")

    temp   = forecast.avg_temp_c
    rainy  = forecast.is_rainy_season or forecast.precip_type in ("rain", "showers", "drizzle")
    wet_prob = forecast.precip_prob is not None and forecast.precip_prob >= 60
    typhoon  = forecast.typhoon_risk in ("moderate", "high")
    high_typhoon = forecast.typhoon_risk == "high"

    # ── Constraint-rule activation (ties to ontology rules.json) ──────────────
    if high_typhoon or (typhoon and wet_prob):
        rec.activated_rules.append(
            "R001 — Coastal attraction + severe storm → suppress swimming")
        rec.warnings.append(
            "Typhoon season risk is elevated: swimming and coastal activities "
            "may be suspended on warning days. Verify live KMA status each morning.")
        rec.discouraged += ["beach swimming", "boat/ferry tours (Udo)", "exposed coastal trails"]
    if forecast.is_rainy_season or rainy:
        rec.activated_rules.append(
            "R006 — Monsoon/rainy season + coastal island → warn about weather-dependent activities")
        rec.warnings.append(
            "Rainy season conditions likely: pack rain gear and plan indoor backups.")
    if forecast.is_typhoon_season:
        rec.activated_rules.append(
            "SeasonalSafety_WeatherDependent — coastal/mountain activities conditional on KMA status")
    if forecast.advisory:
        rec.warnings.append(forecast.advisory)

    # ── Suitability score ─────────────────────────────────────────────────────
    if high_typhoon or wet_prob:
        rec.suitability = "poor"
    elif typhoon or rainy or temp < 5 or temp > 31:
        rec.suitability = "fair"
    elif 12 <= temp <= 27 and not rainy:
        rec.suitability = "excellent"
    else:
        rec.suitability = "good"

    # ── Recommendations by weather regime ─────────────────────────────────────
    indoor = _ontology_indoor(graph) or _INDOOR_JEJU

    if high_typhoon or wet_prob or rainy:
        rec.recommended += (_RAINY_OK[:5])
        rec.indoor_alternatives += indoor[:5]
        rec.discouraged += ["Hallasan summit hike", "long outdoor trails"]
    elif temp < 8:
        rec.recommended += _ontology_attractions(graph, warm=False)[:4] or _COLD_CLEAR[:4]
        rec.indoor_alternatives += indoor[:4]
        rec.warnings.append("Cold and possibly windy — dress warmly; sea may be rough.")
    elif forecast.is_beach_season and temp >= 22 and not typhoon:
        rec.recommended += _ontology_beaches(graph)[:3] or _CLEAR_WARM[:3]
        rec.recommended += ["Hallasan day hike (check trail status)", "Jeju Olle coastal trails"]
    else:
        rec.recommended += _ontology_attractions(graph, warm=True)[:4] or _CLEAR_WARM[:4]

    # Beach-specific guidance when the user asked about beaches/swimming.
    if topics.get("beach") or topics.get("swimming") or "beach" in query.lower():
        if not forecast.is_beach_season:
            rec.warnings.append(
                f"Note: {forecast.date[:7]} is outside Jeju's main beach season "
                "(swimming comfortable ~June–September). Sea temperature "
                f"~{forecast.sea_temp_c:.0f}°C.")
        elif high_typhoon:
            rec.warnings.append(
                "Beach swimming is risky during peak typhoon season — only swim "
                "in designated zones on green-status days.")

    # De-duplicate while preserving order.
    rec.recommended          = _dedup(rec.recommended)
    rec.discouraged          = _dedup(rec.discouraged)
    rec.indoor_alternatives  = _dedup(rec.indoor_alternatives)
    rec.warnings             = _dedup(rec.warnings)
    rec.activated_rules      = _dedup(rec.activated_rules)
    return rec


# ── ontology helpers (optional graph) ─────────────────────────────────────────

# Strict Jeju markers: a concept name must contain one of these to be used,
# preventing non-Jeju places (Phuket, Busan, Bali, …) from the multi-destination
# ontology leaking into Jeju recommendations.
_JEJU_PLACES = (
    "jeju", "hallasan", "haenyeo", "hyeopjae", "hamdeok", "seongsan", "udo",
    "sanbangsan", "manjanggul", "seogwipo", "olle", "oreum", "jungmun",
    "seopjikoji", "camellia", "o'sulloc", "osulloc",
)
# Word-boundary patterns so short tokens (udo, olle, oreum) don't match inside
# unrelated words (e.g. "Son-obudo-yo Museum" must NOT match "udo").
_JEJU_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in _JEJU_PLACES) + r")\b", re.I)

def _is_jeju(name: str) -> bool:
    return bool(_JEJU_RE.search(name))


def _ontology_beaches(graph) -> list[str]:
    if graph is None:
        return []
    out = [c["name"] for c in graph.get_concepts_by_class("Beach") if _is_jeju(c.get("name", ""))]
    return out[:6]


def _ontology_indoor(graph) -> list[str]:
    if graph is None:
        return []
    out = []
    for cls in ("Museum", "CulturalAttraction"):
        out += [c["name"] for c in graph.get_concepts_by_class(cls) if _is_jeju(c.get("name", ""))]
    return _dedup(out)[:6]


def _ontology_attractions(graph, warm: bool) -> list[str]:
    if graph is None:
        return []
    out = []
    for cls in ("NaturalAttraction", "Attraction", "Landmark", "CulturalAttraction"):
        out += [c["name"] for c in graph.get_concepts_by_class(cls) if _is_jeju(c.get("name", ""))]
    return _dedup(out)[:6]


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
