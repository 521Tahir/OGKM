"""
KMA Weather Client — Korea Meteorological Administration (기상청) integration.

Provides Jeju weather forecasts for tourism planning. Two data sources:

  1. LIVE  — the official KMA open API via data.go.kr (free service key).
             Used when KMA_API_KEY is set AND the server has internet.
             - Short-term village forecast (getVilageFcst): up to ~3 days, hourly.
  2. OFFLINE — Jeju climate normals (src/integrations/jeju_climatology.py).
             Always-available fallback so the system is reproducible without
             internet or an API key, and so tests run deterministically.

Getting a FREE KMA service key:
  1. Sign up at https://www.data.go.kr (Korea public-data portal — free).
  2. Search "기상청_단기예보 ((구)_동네예보) 조회서비스" and request use.
  3. Copy the issued service key into .env as:
         KMA_API_KEY=your_decoded_service_key

The client NEVER hard-fails: any network/parse error falls back to climatology.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from . import jeju_climatology as climo
from ..utils.logging_utils import get_logger

logger = get_logger("integrations.weather")

# ── KMA village-forecast grid coordinates for Jeju regions (nx, ny) ───────────
JEJU_GRID: dict[str, tuple[int, int]] = {
    "jeju city":  (53, 38),
    "jeju":       (53, 38),
    "seogwipo":   (52, 33),
    "seongsan":   (55, 37),
    "jungmun":    (52, 33),
    "hallasan":   (53, 37),
    "udo":        (55, 38),
}
DEFAULT_REGION = "jeju city"

# KMA short-term forecast base times (the API publishes 8 times/day).
_BASE_TIMES = ["2300", "2000", "1700", "1400", "1100", "0800", "0500", "0200"]

_VILAGE_FCST_URL = (
    "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
)

# KMA SKY codes → text;  PTY (precipitation type) codes → text.
_SKY = {"1": "clear", "3": "partly cloudy", "4": "cloudy"}
_PTY = {"0": "none", "1": "rain", "2": "rain/snow", "3": "snow",
        "4": "showers", "5": "drizzle", "6": "drizzle/snow", "7": "snow flurries"}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class WeatherForecast:
    """Unified weather forecast result (live or climatology)."""
    date:           str           # YYYY-MM-DD
    region:         str
    source:         str           # "kma_live" | "climatology"
    avg_temp_c:     float
    high_temp_c:    float
    low_temp_c:     float
    precip_prob:    Optional[int]  # % chance of precipitation (live only)
    precip_type:    str            # none | rain | showers | snow | ...
    sky:            str            # clear | partly cloudy | cloudy | ...
    sea_temp_c:     float
    typhoon_risk:   str            # none | low | moderate | high
    is_beach_season:   bool
    is_typhoon_season: bool
    is_rainy_season:   bool
    advisory:       str = ""       # active weather warning text, if any
    note:           str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date, "region": self.region, "source": self.source,
            "avg_temp_c": self.avg_temp_c, "high_temp_c": self.high_temp_c,
            "low_temp_c": self.low_temp_c, "precip_prob": self.precip_prob,
            "precip_type": self.precip_type, "sky": self.sky,
            "sea_temp_c": self.sea_temp_c, "typhoon_risk": self.typhoon_risk,
            "is_beach_season": self.is_beach_season,
            "is_typhoon_season": self.is_typhoon_season,
            "is_rainy_season": self.is_rainy_season,
            "advisory": self.advisory, "note": self.note,
        }

    def summary(self) -> str:
        parts = [
            f"{self.region.title()} on {self.date}: {self.sky}, "
            f"{self.low_temp_c:.0f}–{self.high_temp_c:.0f}°C (avg {self.avg_temp_c:.0f}°C)"
        ]
        if self.precip_prob is not None:
            parts.append(f"{self.precip_prob}% chance of {self.precip_type}")
        elif self.precip_type != "none":
            parts.append(self.precip_type)
        parts.append(f"sea {self.sea_temp_c:.0f}°C")
        if self.typhoon_risk not in ("none", "low"):
            parts.append(f"typhoon risk: {self.typhoon_risk.upper()}")
        if self.advisory:
            parts.append(f"⚠ {self.advisory}")
        return " | ".join(parts)


# ── Client ────────────────────────────────────────────────────────────────────

class KMAWeatherClient:
    """
    KMA weather client with live API + offline climatology fallback.

    Parameters
    ----------
    service_key : str, optional
        KMA data.go.kr service key. Defaults to env KMA_API_KEY.
    timeout : float
        HTTP timeout in seconds for live calls.
    enable_live : bool
        If False, always use climatology (useful for offline reproducibility).
    """

    def __init__(
        self,
        service_key: Optional[str] = None,
        timeout: float = 6.0,
        enable_live: bool = True,
    ) -> None:
        self.service_key = service_key or os.getenv("KMA_API_KEY", "")
        self.timeout = timeout
        self.enable_live = enable_live and bool(self.service_key)
        self._cache: dict[str, WeatherForecast] = {}
        logger.info(
            f"KMAWeatherClient ready (live={'on' if self.enable_live else 'off — climatology'})"
        )

    # ── public API ────────────────────────────────────────────────────────────

    def get_forecast(
        self,
        date: Optional[str | datetime] = None,
        region: str = DEFAULT_REGION,
    ) -> WeatherForecast:
        """
        Return a weather forecast for a Jeju region and date.

        date   : 'YYYY-MM-DD' string, datetime, or None (=today).
        region : key in JEJU_GRID (jeju city, seogwipo, seongsan, ...).
        """
        target = self._parse_date(date)
        region = region.lower().strip()
        if region not in JEJU_GRID:
            region = DEFAULT_REGION

        key = f"{target.strftime('%Y%m%d')}|{region}"
        if key in self._cache:
            return self._cache[key]

        forecast: Optional[WeatherForecast] = None

        # Live KMA only covers ~3 days ahead; beyond that use climatology.
        days_ahead = (target.date() - datetime.now().date()).days
        if self.enable_live and 0 <= days_ahead <= 2:
            try:
                forecast = self._fetch_live(target, region)
            except Exception as e:  # noqa: BLE001 — never hard-fail
                logger.warning(f"KMA live fetch failed ({e}); using climatology.")

        if forecast is None:
            forecast = self._climatology_forecast(target, region)

        self._cache[key] = forecast
        return forecast

    # ── live KMA call ─────────────────────────────────────────────────────────

    def _fetch_live(self, target: datetime, region: str) -> WeatherForecast:
        nx, ny = JEJU_GRID[region]
        base_date, base_time = self._latest_base(target)

        params = {
            "serviceKey": self.service_key,
            "pageNo": "1",
            "numOfRows": "1000",
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": str(nx),
            "ny": str(ny),
        }
        url = _VILAGE_FCST_URL + "?" + urllib.parse.urlencode(params, safe="")
        req = urllib.request.Request(url, headers={"User-Agent": "jeju-ontology/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        items = (
            payload.get("response", {}).get("body", {})
            .get("items", {}).get("item", [])
        )
        if not items:
            raise ValueError("KMA returned no forecast items")

        fdate = target.strftime("%Y%m%d")
        temps, highs, lows, pops, ptys, skies = [], [], [], [], [], []
        for it in items:
            if it.get("fcstDate") != fdate:
                continue
            cat, val = it.get("category"), it.get("fcstValue")
            if cat == "TMP":
                try: temps.append(float(val))
                except ValueError: pass
            elif cat == "TMX":
                try: highs.append(float(val))
                except ValueError: pass
            elif cat == "TMN":
                try: lows.append(float(val))
                except ValueError: pass
            elif cat == "POP":
                try: pops.append(int(float(val)))
                except ValueError: pass
            elif cat == "PTY":
                ptys.append(_PTY.get(str(val), "none"))
            elif cat == "SKY":
                skies.append(_SKY.get(str(val), "partly cloudy"))

        if not temps:
            raise ValueError("KMA forecast missing temperature for target date")

        month = target.month
        cm = climo.get_month_climate(month)
        avg = round(sum(temps) / len(temps), 1)
        high = max(highs) if highs else max(temps)
        low = min(lows) if lows else min(temps)
        precip_prob = max(pops) if pops else 0
        precip_type = next((p for p in ptys if p != "none"), "none")
        sky = max(set(skies), key=skies.count) if skies else cm.sky

        return WeatherForecast(
            date=target.strftime("%Y-%m-%d"), region=region, source="kma_live",
            avg_temp_c=avg, high_temp_c=round(high, 1), low_temp_c=round(low, 1),
            precip_prob=precip_prob, precip_type=precip_type, sky=sky,
            sea_temp_c=cm.sea_temp_c, typhoon_risk=cm.typhoon_risk,
            is_beach_season=climo.is_beach_season(month),
            is_typhoon_season=climo.is_typhoon_season(month),
            is_rainy_season=climo.is_rainy_season(month),
            advisory=self._infer_advisory(precip_prob, precip_type, cm.typhoon_risk),
            note=cm.note,
        )

    # ── climatology fallback ──────────────────────────────────────────────────

    def _climatology_forecast(self, target: datetime, region: str) -> WeatherForecast:
        month = target.month
        cm = climo.get_month_climate(month)
        advisory = ""
        if cm.typhoon_risk == "high":
            advisory = ("Peak typhoon season — typhoon warnings possible; "
                        "verify live KMA status before coastal/mountain activity.")
        elif climo.is_rainy_season(month):
            advisory = "Rainy season (jangma) — frequent rain; plan indoor backups."
        return WeatherForecast(
            date=target.strftime("%Y-%m-%d"), region=region, source="climatology",
            avg_temp_c=cm.avg_temp_c, high_temp_c=cm.high_temp_c, low_temp_c=cm.low_temp_c,
            precip_prob=None, precip_type=("rain" if climo.is_rainy_season(month) else "none"),
            sky=cm.sky, sea_temp_c=cm.sea_temp_c, typhoon_risk=cm.typhoon_risk,
            is_beach_season=climo.is_beach_season(month),
            is_typhoon_season=climo.is_typhoon_season(month),
            is_rainy_season=climo.is_rainy_season(month),
            advisory=advisory, note=cm.note,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_advisory(precip_prob: int, precip_type: str, typhoon_risk: str) -> str:
        if typhoon_risk == "high" and precip_prob >= 60:
            return "Heavy rain likely during peak typhoon season — check KMA special warnings."
        if precip_prob >= 70:
            return f"High chance of {precip_type} — outdoor plans weather-dependent."
        return ""

    @staticmethod
    def _parse_date(date: Optional[str | datetime]) -> datetime:
        if date is None:
            return datetime.now()
        if isinstance(date, datetime):
            return date
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(date.strip(), fmt)
            except ValueError:
                continue
        # Fall back to today if unparseable.
        logger.warning(f"Could not parse date '{date}'; using today.")
        return datetime.now()

    @staticmethod
    def _latest_base(target: datetime) -> tuple[str, str]:
        """Pick the most recent KMA base_date/base_time available now."""
        now = datetime.now()
        # KMA publishes each base time ~10 min later; step back to be safe.
        ref = now - timedelta(minutes=15)
        hhmm = ref.strftime("%H%M")
        for bt in _BASE_TIMES:
            if hhmm >= bt:
                return ref.strftime("%Y%m%d"), bt
        # Before 02:15 → use 2300 of previous day.
        prev = ref - timedelta(days=1)
        return prev.strftime("%Y%m%d"), "2300"
