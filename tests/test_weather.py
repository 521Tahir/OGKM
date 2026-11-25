"""Tests for the KMA weather integration + weather-aware recommender."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.integrations.weather_client import KMAWeatherClient, WeatherForecast, JEJU_GRID
from src.integrations import jeju_climatology as climo
from src.generation.weather_recommender import recommend, _is_jeju


# ── climatology ───────────────────────────────────────────────────────────────

def test_climate_all_months_present():
    for m in range(1, 13):
        cm = climo.get_month_climate(m)
        assert cm.month == m
        assert -5 <= cm.avg_temp_c <= 35
        assert cm.typhoon_risk in {"none", "low", "moderate", "high"}

def test_climate_invalid_month():
    with pytest.raises(ValueError):
        climo.get_month_climate(13)

def test_typhoon_season_window():
    assert climo.is_typhoon_season(8) is True      # August = peak
    assert climo.is_typhoon_season(1) is False     # January = none
    assert climo.typhoon_risk_level(8) == "high"

def test_beach_season():
    assert climo.is_beach_season(7) is True
    assert climo.is_beach_season(8) is True
    assert climo.is_beach_season(1) is False


# ── weather client (offline / climatology mode) ───────────────────────────────

@pytest.fixture
def client():
    # Force climatology mode (no live API) for deterministic tests.
    return KMAWeatherClient(service_key="", enable_live=False)

def test_forecast_climatology_august(client):
    fc = client.get_forecast(date="2026-08-15", region="jeju city")
    assert isinstance(fc, WeatherForecast)
    assert fc.source == "climatology"
    assert fc.typhoon_risk == "high"
    assert fc.is_typhoon_season is True
    assert fc.avg_temp_c > 25

def test_forecast_climatology_january(client):
    fc = client.get_forecast(date="2027-01-15", region="seogwipo")
    assert fc.typhoon_risk == "none"
    assert fc.is_typhoon_season is False
    assert fc.avg_temp_c < 12

def test_forecast_default_region_and_today(client):
    fc = client.get_forecast()           # no date → today, default region
    assert fc.region in JEJU_GRID
    assert fc.date == datetime.now().strftime("%Y-%m-%d")

def test_forecast_unknown_region_falls_back(client):
    fc = client.get_forecast(date="2026-05-01", region="atlantis")
    assert fc.region == "jeju city"      # default

def test_forecast_cache(client):
    a = client.get_forecast(date="2026-10-10", region="jeju city")
    b = client.get_forecast(date="2026-10-10", region="jeju city")
    assert a is b                         # same cached object

def test_forecast_summary_string(client):
    fc = client.get_forecast(date="2026-08-15")
    s = fc.summary()
    assert "°C" in s and "typhoon" in s.lower()


# ── recommender ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph():
    from src.graph.graph_store import TourismGraphStore
    g = TourismGraphStore()
    g.load(
        json.loads(Path("artifacts/ontologies/1.0.0/concepts.json").read_text()),
        json.loads(Path("artifacts/ontologies/1.0.0/relations.json").read_text()),
    )
    return g

def test_recommend_august_typhoon_suppresses_swimming(client, graph):
    fc = client.get_forecast(date="2026-08-15", region="jeju city")
    rec = recommend(fc, query="beach day in August", graph=graph,
                    topics={"beach": True, "swimming": True})
    assert rec.suitability == "poor"
    # R001 must fire during high typhoon risk
    assert any("R001" in r for r in rec.activated_rules)
    assert "beach swimming" in rec.discouraged
    # indoor alternatives offered
    assert len(rec.indoor_alternatives) > 0

def test_recommend_october_good_for_touring(client, graph):
    fc = client.get_forecast(date="2026-10-15", region="jeju city")
    rec = recommend(fc, query="hiking and sightseeing", graph=graph,
                    topics={"hiking": True})
    assert rec.suitability in {"good", "fair", "excellent"}
    assert len(rec.recommended) > 0

def test_recommend_recommendations_are_jeju_only(client, graph):
    fc = client.get_forecast(date="2026-08-15")
    rec = recommend(fc, query="rainy day plan", graph=graph)
    # No non-Jeju leakage in indoor alternatives.
    for item in rec.indoor_alternatives:
        assert _is_jeju(item), f"non-Jeju item leaked: {item}"

def test_is_jeju_word_boundary():
    assert _is_jeju("Jeju Haenyeo Museum") is True
    assert _is_jeju("Udo Island") is True
    assert _is_jeju("Sonobudoyo Museum") is False   # 'udo' substring must not match
    assert _is_jeju("Busan Museum") is False

def test_recommend_rainy_season_warns(client, graph):
    fc = client.get_forecast(date="2026-07-05", region="jeju city")  # rainy season
    rec = recommend(fc, query="outdoor plan", graph=graph)
    assert any("R006" in r for r in rec.activated_rules)
    assert len(rec.warnings) > 0
