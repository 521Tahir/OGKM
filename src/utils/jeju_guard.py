"""
Jeju-only query guard — pure Python, zero LLM cost.

Blocks queries that explicitly name a non-Jeju location.
Allows queries that mention a Jeju landmark/keyword, or that mention no
location at all (we assume the user wants Jeju info).

Usage:
    from src.utils.jeju_guard import jeju_guard

    allowed, msg = jeju_guard("Best beaches in Bali?")
    # allowed=False, msg="I can only answer questions about Jeju Island..."

    allowed, msg = jeju_guard("What hikes near Hallasan?")
    # allowed=True, msg=""
"""
from __future__ import annotations

import re

# ── Terms that confirm a query is Jeju-related ────────────────────────────────

JEJU_TERMS: frozenset[str] = frozenset([
    "jeju", "hallasan", "seongsan", "manjanggul", "hyeopjae", "hamdeok",
    "seogwipo", "jungmun", "cheonjiyeon", "cheonjeyeon", "olle",
    "sanbangsan", "udo", "marado", "bijarim", "yongduam", "dongmun",
    "hallim", "aewol", "seongeup", "haenyeo", "dol hareubang",
    "woljeong", "gimnyeong", "sangumburi", "iho", "hallabong", "oreum",
    "jeju island", "jeju-do",
])

# ── Non-Jeju destinations — query blocked if one appears with no Jeju keyword ─

NON_JEJU_PLACES: dict[str, str] = {
    # Korean cities (not Jeju)
    "seoul":        "Seoul",        "busan":        "Busan",
    "incheon":      "Incheon",      "daegu":        "Daegu",
    "gwangju":      "Gwangju",      "ulsan":        "Ulsan",
    "daejeon":      "Daejeon",      "suwon":        "Suwon",
    "jeonju":       "Jeonju",       "gyeongju":     "Gyeongju",
    "pohang":       "Pohang",       "changwon":     "Changwon",
    "namsan":       "Namsan",
    # Asia
    "bali":         "Bali",         "phuket":       "Phuket",
    "bangkok":      "Bangkok",      "singapore":    "Singapore",
    "tokyo":        "Tokyo",        "osaka":        "Osaka",
    "kyoto":        "Kyoto",        "beijing":      "Beijing",
    "shanghai":     "Shanghai",     "hong kong":    "Hong Kong",
    "taipei":       "Taipei",       "ho chi minh":  "Ho Chi Minh",
    "hanoi":        "Hanoi",        "kuala lumpur": "Kuala Lumpur",
    "jakarta":      "Jakarta",      "manila":       "Manila",
    "dubai":        "Dubai",        "mumbai":       "Mumbai",
    "delhi":        "Delhi",        "kathmandu":    "Kathmandu",
    "colombo":      "Colombo",
    # Countries
    "pakistan":     "Pakistan",     "india":        "India",
    "china":        "China",        "japan":        "Japan",
    "indonesia":    "Indonesia",    "thailand":     "Thailand",
    "malaysia":     "Malaysia",     "vietnam":      "Vietnam",
    "nepal":        "Nepal",        "bangladesh":   "Bangladesh",
    "sri lanka":    "Sri Lanka",    "myanmar":      "Myanmar",
    # Western
    "paris":        "Paris",        "london":       "London",
    "rome":         "Rome",         "barcelona":    "Barcelona",
    "amsterdam":    "Amsterdam",    "berlin":       "Berlin",
    "new york":     "New York",     "los angeles":  "Los Angeles",
    "sydney":       "Sydney",       "toronto":      "Toronto",
    "australia":    "Australia",    "canada":       "Canada",
    "usa":          "USA",          "uk":           "UK",
    "france":       "France",
}

_REDIRECT_MSG = (
    "I can only answer questions about Jeju Island, South Korea.\n"
    "Your question mentions {place}, which is outside my scope.\n\n"
    "Try asking about Jeju instead, for example:\n"
    '  "Plan a 3-day trip for Jeju Island"\n'
    '  "What are the best beaches in Jeju?"\n'
    '  "What food should I try in Jeju?"'
)


def jeju_guard(question: str) -> tuple[bool, str]:
    """
    Return (allowed, rejection_message).
    Runs in <1 ms — no LLM cost.

    Logic:
    - Any Jeju keyword present → allow immediately.
    - Known non-Jeju place word-boundary match → block with redirect message.
    - No location at all → allow (assume Jeju context).
    """
    q = question.lower()

    if any(term in q for term in JEJU_TERMS):
        return True, ""

    for term, display in NON_JEJU_PLACES.items():
        if re.search(r"\b" + re.escape(term) + r"\b", q):
            return False, _REDIRECT_MSG.format(place=display)

    return True, ""
