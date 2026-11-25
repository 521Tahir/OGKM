"""
Query Understanding — parse tourist queries into structured intent objects.

Responsibilities:
  - Detect task type (recommendation, fact_lookup, comparison, itinerary, safety)
  - Extract destination entities from query text
  - Identify traveler constraints (budget, season, accessibility, group type)
  - Detect theme preferences (nature, culture, food, adventure, history)
  - Flag urgency / sentiment modifiers

Used by: context_assembler.py, constrained_generator.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.query_understanding")


# ── Vocabulary tables ────────────────────────────────────────────────────────

_TASK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("recommendation", re.compile(
        r"\b(recommend|suggest|best|top|where (should|can) i|what (is|are) (the best|good)|help me (find|choose))\b",
        re.I)),
    ("comparison", re.compile(
        r"\b(compare|vs\.?|versus|difference between|which is better|pros and cons)\b",
        re.I)),
    ("itinerary", re.compile(
        r"\b(itinerary|plan|schedule|trip plan|day.by.day|how many days|route|tour plan)\b",
        re.I)),
    ("safety", re.compile(
        r"\b(safe|safety|danger|risk|warning|travel advisory|crime|scam)\b",
        re.I)),
    ("fact_lookup", re.compile(
        r"\b(what (is|are|time|hours)|when (does|is)|how (much|far|long|do)|tell me about|info (on|about))\b",
        re.I)),
]

_BUDGET_PATTERNS: dict[str, re.Pattern] = {
    "budget":   re.compile(r"\b(cheap|budget|affordable|low.cost|inexpensive|backpacker)\b", re.I),
    "midrange": re.compile(r"\b(mid.?range|moderate|reasonable|value for money)\b", re.I),
    "luxury":   re.compile(r"\b(luxury|luxurious|high.end|five.star|5.star|premium|exclusive)\b", re.I),
}

_SEASON_PATTERNS: dict[str, re.Pattern] = {
    "spring": re.compile(r"\b(spring|march|april|may)\b", re.I),
    "summer": re.compile(r"\b(summer|june|july|august|hot season)\b", re.I),
    "autumn": re.compile(r"\b(autumn|fall|september|october|november|foliage)\b", re.I),
    "winter": re.compile(r"\b(winter|december|january|february|snow|cold)\b", re.I),
}

_GROUP_PATTERNS: dict[str, re.Pattern] = {
    "solo":      re.compile(r"\b(solo|alone|by myself|single traveler)\b", re.I),
    "couple":    re.compile(r"\b(couple|honeymoon|romantic|partner|spouse)\b", re.I),
    "family":    re.compile(r"\b(family|kids|children|child|toddler|baby|stroller)\b", re.I),
    "group":     re.compile(r"\b(group|friends|party of|team|corporate)\b", re.I),
    "elderly":   re.compile(r"\b(elderly|senior|older adult|retirement|grandparent)\b", re.I),
    "disabled":  re.compile(r"\b(wheelchair|disabled|accessibility|accessible|mobility)\b", re.I),
}

_THEME_PATTERNS: dict[str, re.Pattern] = {
    "nature":    re.compile(r"\b(nature|hiking|trekking|beach|mountain|forest|outdoor|wildlife|national park)\b", re.I),
    "culture":   re.compile(r"\b(culture|museum|heritage|traditional|local life|folk|art|festival)\b", re.I),
    "food":      re.compile(r"\b(food|cuisine|restaurant|eat|dining|local dish|seafood|street food)\b", re.I),
    "adventure": re.compile(r"\b(adventure|extreme|scuba|diving|paragliding|bungee|rafting|surfing)\b", re.I),
    "history":   re.compile(r"\b(history|historic|historical|ancient|ruins|castle|monument|UNESCO)\b", re.I),
    "shopping":  re.compile(r"\b(shopping|market|mall|souvenir|duty.free|bazaar)\b", re.I),
    "wellness":  re.compile(r"\b(spa|wellness|relax|yoga|meditation|hot spring|thermal)\b", re.I),
}

# Destination-related tokens (Jeju-specific + general geography)
_DESTINATION_STOPWORDS = {
    "the", "a", "an", "in", "at", "on", "to", "of", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "i", "we", "you", "they", "he", "she", "it", "this", "that", "these",
    "those", "my", "your", "our", "their", "me", "us", "him", "her", "them",
}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TravelerConstraints:
    budget: Optional[str] = None          # "budget" | "midrange" | "luxury"
    season: Optional[str] = None          # "spring" | "summer" | "autumn" | "winter"
    group_type: Optional[str] = None      # "solo" | "couple" | "family" | "group" | "elderly" | "disabled"
    duration_days: Optional[int] = None   # extracted from "3 days", "a week", etc.
    themes: list[str] = field(default_factory=list)  # ["nature", "culture", ...]


@dataclass
class QueryIntent:
    """Structured understanding of a tourist query."""
    raw_query: str
    task_type: str                            # "recommendation" | "comparison" | "itinerary" | "safety" | "fact_lookup" | "general"
    destination_mentions: list[str]           # raw text spans that look like destination names
    constraints: TravelerConstraints
    keywords: list[str]                       # meaningful non-stop-word tokens
    is_jeju_specific: bool = False            # True if query mentions Jeju explicitly
    confidence: float = 1.0                   # parsing confidence 0–1


# ── Public API ───────────────────────────────────────────────────────────────

class QueryUnderstanding:
    """
    Parse a raw natural-language tourist query into a QueryIntent.

    Pure rule-based — no LLM call, no model weights. Fast enough for
    every request without adding latency.
    """

    # Jeju-related terms (used to set is_jeju_specific)
    _JEJU_RE = re.compile(
        r"\b(jeju|제주|hallasan|seongsan|manjanggul|haenyeo|olle|gotjawal|udo island|mara island)\b",
        re.I,
    )

    # Duration extraction: "3 days", "2 nights", "a week", "one week"
    _DURATION_RE = re.compile(
        r"(\d+|one|two|three|four|five|six|seven)\s*(day|days|night|nights|week|weeks)",
        re.I,
    )
    _WORD_TO_INT = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7,
    }

    # Candidate destination: capitalized word (2+ chars), not a stop word
    _CAPITALIZED_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,})*)\b")

    def parse(self, query: str) -> QueryIntent:
        """Return a QueryIntent for the given query string."""
        task_type = self._detect_task_type(query)
        destinations = self._extract_destinations(query)
        constraints = self._extract_constraints(query)
        keywords = self._extract_keywords(query)
        is_jeju = bool(self._JEJU_RE.search(query))

        intent = QueryIntent(
            raw_query=query,
            task_type=task_type,
            destination_mentions=destinations,
            constraints=constraints,
            keywords=keywords,
            is_jeju_specific=is_jeju,
        )
        logger.debug("Parsed query: task=%s dest=%s themes=%s", task_type, destinations, constraints.themes)
        return intent

    # ── Private helpers ──────────────────────────────────────────────────────

    def _detect_task_type(self, query: str) -> str:
        for task, pattern in _TASK_PATTERNS:
            if pattern.search(query):
                return task
        return "general"

    def _extract_destinations(self, query: str) -> list[str]:
        """Extract capitalized spans as potential destination names."""
        candidates = self._CAPITALIZED_TOKEN_RE.findall(query)
        seen: set[str] = set()
        result: list[str] = []
        for c in candidates:
            lower = c.lower()
            if lower not in _DESTINATION_STOPWORDS and lower not in seen:
                seen.add(lower)
                result.append(c)
        return result

    def _extract_constraints(self, query: str) -> TravelerConstraints:
        budget = next(
            (level for level, pat in _BUDGET_PATTERNS.items() if pat.search(query)),
            None
        )
        season = next(
            (s for s, pat in _SEASON_PATTERNS.items() if pat.search(query)),
            None
        )
        group_type = next(
            (g for g, pat in _GROUP_PATTERNS.items() if pat.search(query)),
            None
        )
        themes = [t for t, pat in _THEME_PATTERNS.items() if pat.search(query)]
        duration = self._extract_duration(query)

        return TravelerConstraints(
            budget=budget,
            season=season,
            group_type=group_type,
            duration_days=duration,
            themes=themes,
        )

    def _extract_duration(self, query: str) -> Optional[int]:
        m = self._DURATION_RE.search(query)
        if not m:
            return None
        num_str = m.group(1).lower()
        days_per_unit = 7 if "week" in m.group(2).lower() else 1
        try:
            num = int(num_str)
        except ValueError:
            num = self._WORD_TO_INT.get(num_str, 1)
        return num * days_per_unit

    def _extract_keywords(self, query: str) -> list[str]:
        """Lowercase meaningful tokens, removing stop words and punctuation."""
        tokens = re.findall(r"[a-zA-Z']+", query.lower())
        return [t for t in tokens if t not in _DESTINATION_STOPWORDS and len(t) > 2]


# Module-level singleton
_default_qu: Optional[QueryUnderstanding] = None


def get_query_understanding() -> QueryUnderstanding:
    global _default_qu
    if _default_qu is None:
        _default_qu = QueryUnderstanding()
    return _default_qu
