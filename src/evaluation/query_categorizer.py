"""
Rule-based query categorizer for tourism benchmark (Week 9).

Maps a query string to one of 11 tourism categories using regex patterns.
Designed for Jeju Island tourism domain but generalizable.

Categories
----------
1.  itinerary_planning
2.  attraction_recommendation
3.  restaurant_recommendation
4.  accessibility_aware
5.  weather_aware
6.  safety_advice
7.  etiquette
8.  dietary_constraints
9.  family_travel
10. emergency_info
11. local_transportation

Paper: Table 9 — Per-Category Benchmark Performance (Week 9)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.query_categorizer")

# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = [
    "itinerary_planning",
    "attraction_recommendation",
    "restaurant_recommendation",
    "accessibility_aware",
    "weather_aware",
    "safety_advice",
    "etiquette",
    "dietary_constraints",
    "family_travel",
    "emergency_info",
    "local_transportation",
]

# Each entry: (category_name, list_of_regex_patterns, priority_weight)
# Higher priority_weight wins when multiple categories match.
# Patterns are matched case-insensitively against the query.

_CATEGORY_RULES: list[tuple[str, list[str], int]] = [
    (
        "emergency_info",
        [
            r"\bhospital\b", r"\bclinic\b", r"\bpharmacy\b",
            r"\bambulance\b", r"\bpolice\s+station\b", r"\bfire\s+station\b",
            r"\bemergency\s+(number|contact|service|call)\b",
            r"\b(911|119|112|118)\b",
            r"\burgent\s+(care|help|medical)\b",
            r"\bfirst\s+aid\b", r"\binjur(ed|y)\b",
        ],
        10,
    ),
    (
        "accessibility_aware",
        [
            r"\bwheelchair\b", r"\bdisabilit(y|ies)\b", r"\bdisabled\b",
            r"\baccessib(le|ility)\b", r"\bramp\b", r"\belevator\b", r"\blift\b",
            r"\bmobilit(y|ies)\b", r"\bhandicap\b",
            r"\bvisually?\s+impaired\b", r"\bhearing\s+impaired\b",
        ],
        9,
    ),
    (
        "dietary_constraints",
        [
            r"\bvegan\b", r"\bvegetar(ian|ians)\b",
            r"\bhalal\b", r"\bkosher\b",
            r"\ballerg(y|ies|ic)\b", r"\bgluten[\s-]free\b",
            r"\bdairy[\s-]free\b", r"\bnut[\s-]free\b",
            r"\bdietary\s+(restriction|requirement|need)\b",
            r"\bno\s+(meat|pork|beef|seafood|shellfish)\b",
        ],
        8,
    ),
    (
        "safety_advice",
        [
            r"\bsafe(ty|ly)?\b", r"\bdangerous?\b", r"\bcrime\b",
            r"\bscam\b", r"\btheft\b", r"\bpickpocket\b",
            r"\brisk\b", r"\bhazard\b", r"\bwarning\b",
            r"\bprotect\b", r"\bsecurity\b",
            r"\btravel\s+advisory\b", r"\bsafe\s+to\s+(travel|visit|go)\b",
        ],
        7,
    ),
    (
        "etiquette",
        [
            r"\betiquette\b", r"\bcustom\b", r"\btradition\b",
            r"\bcultur(e|al)\b", r"\brespect\b", r"\boffensive\b",
            r"\btaboo\b", r"\bdress\s+code\b", r"\bmanners?\b",
            r"\bbehavior\b", r"\bnorms?\b",
            r"\bdo(s|es)?\s+and\s+don'?ts?\b", r"\bpolite\b",
        ],
        6,
    ),
    (
        "weather_aware",
        [
            r"\bweather\b", r"\bclimate\b", r"\btemperature\b",
            r"\brain(y|fall)?\b", r"\bsnow\b", r"\bwind\b",
            r"\btyphoon\b", r"\bmonsoon\b", r"\bhumid(ity)?\b",
            r"\bsunny\b", r"\bovercast\b", r"\bforecast\b",
            r"\bseason(al)?\b", r"\bspring|summer|autumn|fall|winter\b",
            r"\bbest\s+time\s+(to\s+visit|of\s+year)\b",
        ],
        5,
    ),
    (
        "family_travel",
        [
            r"\bfamily\b", r"\bchild(ren)?\b", r"\bkid(s)?\b",
            r"\btoddler\b", r"\bbaby\b", r"\binfant\b",
            r"\bstroller\b", r"\bplayground\b",
            r"\bfamily[\s-]friendly\b", r"\bkid[\s-]friendly\b",
            r"\bschool[\s-]age\b", r"\bteen(ager)?s?\b",
        ],
        4,
    ),
    (
        "local_transportation",
        [
            r"\bbus\b", r"\btaxi\b", r"\bcab\b",
            r"\bferry\b", r"\bboat\b", r"\bsubway\b", r"\bmetro\b",
            r"\brent[\s-]?a?[\s-]?car\b", r"\bcar\s+rental\b",
            r"\bscooter\b", r"\bbike\s+rental\b", r"\bcycle\b",
            r"\bshutt?le\b", r"\btrain\b", r"\bairport\s+transfer\b",
            r"\bgetting\s+(around|there|to)\b", r"\btransport(ation)?\b",
        ],
        3,
    ),
    (
        "restaurant_recommendation",
        [
            r"\brestaurant\b", r"\bcafe\b", r"\bcoffee\s+shop\b",
            r"\beat(ing)?\b", r"\bdine\b", r"\bdining\b",
            r"\bfood\b", r"\bcuisine\b", r"\bmeal\b",
            r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b",
            r"\bsnack\b", r"\bstreet\s+food\b",
            r"\bwhere\s+to\s+eat\b", r"\bgood\s+place\s+to\s+eat\b",
        ],
        2,
    ),
    (
        "itinerary_planning",
        [
            r"\bitinerary\b", r"\bschedule\b", r"\bday\s+\d+\b",
            r"\btrip\s+plan\b", r"\bplan\s+(my\s+)?trip\b",
            r"\b(\d+[-\s]day|\d+\s+days?)\s+(trip|tour|visit|itinerary)\b",
            r"\bwhat\s+to\s+do\s+(in|on|during)\b",
            r"\bfull[\s-]day\b", r"\bhalf[\s-]day\b",
            r"\bmorning.*afternoon\b", r"\bday\s+trip\b",
            r"\bvisit\s+multiple\b", r"\broute\b",
        ],
        2,
    ),
    (
        "attraction_recommendation",
        [
            r"\battraction\b", r"\blandmark\b", r"\bsightseeing\b",
            r"\btourist\s+spot\b", r"\bmust[\s-]see\b", r"\bmust[\s-]visit\b",
            r"\bbest\s+(place|spot|site|thing)\s+to\s+(visit|see|do)\b",
            r"\bpopular\s+(place|destination|spot)\b",
            r"\bhike\b", r"\btrail\b", r"\bbeach\b", r"\bvolcano\b",
            r"\bwaterfall\b", r"\bcastle\b", r"\btemple\b", r"\bmuseum\b",
            r"\bnational\s+park\b", r"\bscenic\b", r"\bviewpoint\b",
        ],
        1,
    ),
]

# ── Main categorizer ──────────────────────────────────────────────────────────

@dataclass
class CategoryResult:
    """Result of categorizing a single query."""
    query:         str
    category:      str        # primary assigned category
    confidence:    float      # fraction of patterns matched (0-1) for primary
    all_matches:   dict[str, int]  # category → n_patterns matched

    def to_dict(self) -> dict:
        return {
            "query":       self.query,
            "category":    self.category,
            "confidence":  round(self.confidence, 3),
            "all_matches": self.all_matches,
        }


class QueryCategorizer:
    """
    Rule-based query→category classifier for tourism queries.

    Uses a priority-weighted regex matching system. When multiple categories
    match, the one with the highest priority wins. Within the same priority
    level, the one with more pattern matches wins.

    Parameters
    ----------
    default_category : str
        Category to assign when no rules match (default: 'attraction_recommendation').
    """

    def __init__(self, default_category: str = "attraction_recommendation") -> None:
        self._default = default_category
        # Pre-compile all patterns
        self._compiled: list[tuple[str, list[re.Pattern], int]] = [
            (
                cat,
                [re.compile(p, re.I) for p in patterns],
                priority,
            )
            for cat, patterns, priority in _CATEGORY_RULES
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def categorize(self, query: str) -> CategoryResult:
        """Assign a single query to its best-matching category."""
        matches: dict[str, tuple[int, int]] = {}  # category → (priority, n_matches)

        for cat, compiled_patterns, priority in self._compiled:
            n_matches = sum(1 for p in compiled_patterns if p.search(query))
            if n_matches > 0:
                matches[cat] = (priority, n_matches)

        if not matches:
            return CategoryResult(
                query=query,
                category=self._default,
                confidence=0.0,
                all_matches={},
            )

        # Pick winner: highest priority, break ties by n_matches
        winner = max(matches, key=lambda c: (matches[c][0], matches[c][1]))
        priority_w, n_w = matches[winner]

        # Confidence: fraction of category's patterns matched
        n_patterns = next(
            len(pats) for cat, pats, _ in self._compiled if cat == winner
        )
        confidence = n_w / n_patterns

        all_match_counts = {cat: n for cat, (_, n) in matches.items()}
        return CategoryResult(
            query=query,
            category=winner,
            confidence=confidence,
            all_matches=all_match_counts,
        )

    def categorize_batch(self, queries: list[str]) -> list[CategoryResult]:
        """Categorize a list of queries."""
        return [self.categorize(q) for q in queries]

    def category_distribution(
        self, results: list[CategoryResult]
    ) -> dict[str, int]:
        """Count queries per category."""
        dist: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        for r in results:
            dist[r.category] = dist.get(r.category, 0) + 1
        return dist

    def print_distribution(self, results: list[CategoryResult]) -> None:
        dist = self.category_distribution(results)
        total = sum(dist.values())
        logger.info(f"Query category distribution (n={total}):")
        for cat in CATEGORIES:
            n = dist.get(cat, 0)
            bar = "#" * (n // 2)
            logger.info(f"  {cat:<35} {n:>4} ({100*n/total:.1f}%)  {bar}")
