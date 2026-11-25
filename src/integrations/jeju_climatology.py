"""
Jeju Climatology — offline fallback knowledge base.

Real public climate-normal data for Jeju Island (sourced from KMA climate
normals). Used when the live KMA API is unavailable (no service key, no
internet, or API error) so the weather feature is always functional and
the project stays fully reproducible offline.

All values are monthly normals for Jeju City. Sea temperature, typhoon
risk windows, and the official beach season reflect documented Jeju
tourism/meteorological facts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonthClimate:
    month:          int       # 1-12
    name:           str
    avg_temp_c:     float     # monthly mean air temperature
    high_temp_c:    float
    low_temp_c:     float
    precip_mm:      float     # monthly total precipitation normal
    rain_days:      int       # days with measurable precipitation
    sea_temp_c:     float     # coastal sea surface temperature
    typhoon_risk:   str       # none | low | moderate | high
    sky:            str       # typical sky condition
    season:         str       # spring | summer | autumn | winter
    note:           str       # tourism-relevant note


# ── Jeju monthly climate normals ──────────────────────────────────────────────
JEJU_CLIMATE: dict[int, MonthClimate] = {
    1:  MonthClimate(1,  "January",   6.0,  8.0,  3.0,  65.0, 9,  15.0, "none",     "partly cloudy, windy", "winter",
                     "Cold and windy; sea often rough. Good for indoor sites, hot springs, winter camellia."),
    2:  MonthClimate(2,  "February",  7.0,  9.0,  3.5,  63.0, 8,  14.0, "none",     "partly cloudy",        "winter",
                     "Still cold; early camellia and plum blossoms begin late month."),
    3:  MonthClimate(3,  "March",    10.0, 12.5,  6.0,  88.0, 9,  14.0, "none",     "variable",             "spring",
                     "Spring begins; cherry blossoms late March. Mild for hiking."),
    4:  MonthClimate(4,  "April",    14.0, 17.0, 10.0,  89.0, 9,  16.0, "none",     "mostly clear",         "spring",
                     "Canola (yellow rape) flowers peak; cherry blossoms. Excellent for outdoor touring."),
    5:  MonthClimate(5,  "May",      18.0, 21.0, 14.0,  96.0, 9,  18.0, "low",      "mostly clear",         "spring",
                     "Warm and pleasant; one of the best months for Hallasan and Olle trails."),
    6:  MonthClimate(6,  "June",     21.5, 24.0, 18.0, 182.0, 12, 21.0, "low",      "cloudy, rainy",        "summer",
                     "Rainy season (jangma) usually starts late June; humid. Carry rain gear."),
    7:  MonthClimate(7,  "July",     26.5, 29.0, 23.0, 240.0, 13, 25.0, "moderate", "humid, showers",       "summer",
                     "Hot, humid, peak rainy season. Typhoon season begins. Beach season opens ~Jul 1."),
    8:  MonthClimate(8,  "August",   28.0, 31.0, 25.0, 257.0, 13, 27.0, "high",     "hot, thunderstorms",   "summer",
                     "Hottest month and peak typhoon season. Prime beach season but check warnings daily."),
    9:  MonthClimate(9,  "September",24.0, 27.0, 21.0, 222.0, 11, 25.0, "high",     "variable, storms",     "autumn",
                     "Typhoon season continues; warm sea. Beach season ends ~Sep 1 (lifeguards withdraw)."),
    10: MonthClimate(10, "October",  19.0, 22.0, 15.0,  80.0, 7,  22.0, "moderate", "clear, crisp",         "autumn",
                     "Pleasant, clear autumn weather; typhoon risk tapering. Great for hiking and touring."),
    11: MonthClimate(11, "November", 14.0, 16.5, 10.0,  70.0, 8,  19.0, "low",      "partly cloudy",        "autumn",
                     "Cool autumn; silver grass (eulalia) at oreum. Comfortable for walking."),
    12: MonthClimate(12, "December",  9.0, 11.0,  5.0,  55.0, 8,  16.0, "none",     "windy, cold",          "winter",
                     "Cold and windy; sea rough. Winter scenery on Hallasan; indoor attractions."),
}

# Official Jeju designated beach (swimming) season — lifeguards on duty.
BEACH_SEASON_MONTHS = {7, 8}          # peak; some beaches open mid-Jun to early-Sep
BEACH_COMFORTABLE_MONTHS = {6, 7, 8, 9}  # sea warm enough for swimming

# Korean/Jeju typhoon season window.
TYPHOON_SEASON_MONTHS = {7, 8, 9, 10}
TYPHOON_PEAK_MONTHS = {8, 9}

# Rainy season (jangma).
RAINY_SEASON_MONTHS = {6, 7}

# Months generally considered best for tourism (mild, clear, scenic).
BEST_TOURISM_MONTHS = {4, 5, 10, 11}


def get_month_climate(month: int) -> MonthClimate:
    """Return climate normals for a month (1-12)."""
    if month < 1 or month > 12:
        raise ValueError(f"month must be 1-12, got {month}")
    return JEJU_CLIMATE[month]


def is_beach_season(month: int) -> bool:
    return month in BEACH_COMFORTABLE_MONTHS


def is_typhoon_season(month: int) -> bool:
    return month in TYPHOON_SEASON_MONTHS


def is_rainy_season(month: int) -> bool:
    return month in RAINY_SEASON_MONTHS


def typhoon_risk_level(month: int) -> str:
    """Coarse typhoon risk for a month based on Jeju climatology."""
    return get_month_climate(month).typhoon_risk
