"""
Temporal ontology layer — explicit time model for the tourism ontology.

Motivation (comments.txt / Temporal Ontology Research):
    Traditional ontologies are static and cannot represent dynamic, time-varying
    facts. This module adds an explicit temporal dimension so concepts and
    relations can state *when* they hold — recurring seasonal windows (blooms,
    festivals, weather), validity intervals, and "as-of" facts — enabling the
    LLM to give time-aware, context-correct answers.

Design
------
A ``TemporalAnnotation`` attaches a time scope to any fact:
    - ``months``      : list[int] (1–12) when the fact is active (recurring)
    - ``season``      : optional season label (spring/summer/autumn/winter)
    - ``recurrence``  : "annual" | "seasonal" | "once" | "always"
    - ``valid_from`` / ``valid_to`` : ISO dates for one-off / bounded validity
    - ``note``        : human-readable phrasing (e.g. "late March to early April")

Months use the Northern-hemisphere calendar (Jeju, South Korea).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Northern-hemisphere season → months (meteorological)
SEASON_MONTHS: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "fall":   [9, 10, 11],
}

# Lowercased month name / abbreviation → month number
_MONTH_LOOKUP: dict[str, int] = {}
for _i, _n in enumerate(MONTH_NAMES):
    if _n:
        _MONTH_LOOKUP[_n.lower()] = _i
        _MONTH_LOOKUP[_n.lower()[:3]] = _i


def season_to_months(season: str) -> list[int]:
    """Map a season label to its month numbers (empty if unknown)."""
    return list(SEASON_MONTHS.get((season or "").strip().lower(), []))


def months_to_phrase(months: list[int]) -> str:
    """Render a month list as a compact human phrase (e.g. 'March–April')."""
    if not months:
        return ""
    ms = sorted(set(m for m in months if 1 <= m <= 12))
    if not ms:
        return ""
    if len(ms) == 1:
        return MONTH_NAMES[ms[0]]
    # contiguous run (handles Dec–Feb wrap)
    if _is_contiguous(ms):
        return f"{MONTH_NAMES[ms[0]]}–{MONTH_NAMES[ms[-1]]}"
    if _is_contiguous_wrap(ms):
        ordered = _wrap_order(ms)
        return f"{MONTH_NAMES[ordered[0]]}–{MONTH_NAMES[ordered[-1]]}"
    return ", ".join(MONTH_NAMES[m] for m in ms)


def _is_contiguous(ms: list[int]) -> bool:
    return all(ms[i] + 1 == ms[i + 1] for i in range(len(ms) - 1))


def _is_contiguous_wrap(ms: list[int]) -> bool:
    """True if months form a contiguous block crossing the year boundary."""
    full = set(ms)
    for start in range(1, 13):
        run = {((start - 1 + k) % 12) + 1 for k in range(len(ms))}
        if run == full:
            return True
    return False


def _wrap_order(ms: list[int]) -> list[int]:
    full = set(ms)
    for start in range(1, 13):
        run = [((start - 1 + k) % 12) + 1 for k in range(len(ms))]
        if set(run) == full:
            return run
    return sorted(ms)


def parse_months(value) -> list[int]:
    """Coerce months given as ints, names, or season labels into a month list."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip().lower()
        if v in SEASON_MONTHS:
            return season_to_months(v)
        if v in _MONTH_LOOKUP:
            return [_MONTH_LOOKUP[v]]
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, int) and 1 <= item <= 12:
            out.append(item)
        elif isinstance(item, str):
            out.extend(parse_months(item))
    return sorted(set(out))


@dataclass
class TemporalAnnotation:
    """A time scope attached to a fact."""
    months:     list[int] = field(default_factory=list)   # 1–12, recurring
    season:     Optional[str] = None
    recurrence: str = "annual"                              # annual|seasonal|once|always
    valid_from: Optional[str] = None                        # ISO date
    valid_to:   Optional[str] = None                        # ISO date
    note:       str = ""

    def __post_init__(self) -> None:
        if not self.months and self.season:
            self.months = season_to_months(self.season)

    def is_active(self, month: Optional[int] = None, on: Optional[date] = None) -> bool:
        """
        Is this fact active in the given month (1–12) or on the given date?
        ``always`` facts are always active; if no time is given, returns True.
        """
        if self.recurrence == "always":
            return True
        if on is not None:
            if self.valid_from and on.isoformat() < self.valid_from:
                return False
            if self.valid_to and on.isoformat() > self.valid_to:
                return False
            month = month or on.month
        if month is None:
            return True
        return month in self.months if self.months else True

    def phrase(self) -> str:
        """Best human-readable time phrase for this annotation."""
        if self.note:
            return self.note
        if self.months:
            return months_to_phrase(self.months)
        if self.season:
            return self.season
        if self.valid_from or self.valid_to:
            return f"{self.valid_from or '…'} to {self.valid_to or '…'}"
        return "year-round"

    def to_dict(self) -> dict:
        return {
            "months": self.months,
            "season": self.season,
            "recurrence": self.recurrence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalAnnotation":
        d = d or {}
        return cls(
            months=parse_months(d.get("months")) or parse_months(d.get("season")),
            season=d.get("season"),
            recurrence=d.get("recurrence", "annual"),
            valid_from=d.get("valid_from"),
            valid_to=d.get("valid_to"),
            note=d.get("note", ""),
        )


@dataclass
class TemporalFact:
    """A subject–predicate–object fact with a temporal scope."""
    subject:   str
    predicate: str
    object:    str
    temporal:  TemporalAnnotation
    category:  str = "general"        # bloom|festival|weather|harvest|hiking|...
    confidence: float = 0.9
    source:    str = "curated"

    def is_active(self, month: Optional[int] = None, on: Optional[date] = None) -> bool:
        return self.temporal.is_active(month=month, on=on)

    def to_sentence(self) -> str:
        """Render the fact as a natural-language, time-scoped statement."""
        when = self.temporal.phrase()
        verb = self.predicate.replace("_", " ")
        return f"{self.subject} {verb} {self.object} ({when})."

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "temporal": self.temporal.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalFact":
        return cls(
            subject=d.get("subject", ""),
            predicate=d.get("predicate", ""),
            object=d.get("object", ""),
            temporal=TemporalAnnotation.from_dict(d.get("temporal", {})),
            category=d.get("category", "general"),
            confidence=d.get("confidence", 0.9),
            source=d.get("source", "curated"),
        )


def current_month() -> int:
    return date.today().month
