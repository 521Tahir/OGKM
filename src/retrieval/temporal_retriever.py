"""
Temporal retriever — selects time-scoped facts from the temporal layer for a
query and formats them for prompt injection (Case B of the temporal demo).

Given a natural-language question (and an optional reference month/date), it:
  1. links the question to relevant temporal facts by keyword overlap;
  2. optionally prioritises facts that are *active* in the reference month;
  3. returns the top-K facts plus an injectable context block.

This is the ontology-grounding step that lets a vanilla LLM answer time-varying
questions correctly (e.g. "when do cherry blossoms bloom in Jeju?").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from ..ontology.temporal import TemporalFact
from ..utils.io import load_json
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.temporal_retriever")

_STOP = {
    "the", "a", "an", "in", "of", "to", "for", "and", "or", "is", "are", "do",
    "does", "can", "i", "my", "you", "when", "what", "which", "how", "best",
    "time", "during", "at", "on", "season", "months", "month", "visit", "see",
    "go", "should", "will", "me", "about", "tell", "jeju", "there",
}


def _stem(t: str) -> str:
    """Light singular/plural normalisation so 'camellias'~'camellia', 'flowers'~'flower'."""
    for suf in ("ies", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def _tokens(text: str) -> set[str]:
    raw = [t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in _STOP]
    return {_stem(t) for t in raw}


@dataclass
class TemporalRetrievalResult:
    facts: list[TemporalFact]
    context: str
    n_active: int

    def is_empty(self) -> bool:
        return not self.facts


class TemporalRetriever:
    """Keyword + recency retriever over the temporal layer."""

    def __init__(self, facts: list[TemporalFact]):
        self.facts = facts
        # precompute keyword sets per fact
        self._fact_tokens: list[set[str]] = [
            _tokens(f"{f.subject} {f.object} {f.temporal.note}") for f in facts
        ]

    @classmethod
    def from_version(cls, version: str = "1.2.0",
                     ontologies_dir: str | Path = "artifacts/ontologies") -> "TemporalRetriever":
        path = Path(ontologies_dir) / version / "temporal_facts.json"
        payload = load_json(path)
        facts = [TemporalFact.from_dict(d) for d in payload.get("facts", [])]
        logger.info("Loaded %d temporal facts from v%s", len(facts), version)
        return cls(facts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reference_month: Optional[int] = None,
        on: Optional[date] = None,
    ) -> TemporalRetrievalResult:
        """Return the top-K temporal facts relevant to the query."""
        q_tokens = _tokens(query)
        if on is not None and reference_month is None:
            reference_month = on.month

        scored: list[tuple[float, TemporalFact]] = []
        for f, ftoks in zip(self.facts, self._fact_tokens):
            overlap = len(q_tokens & ftoks)
            if overlap == 0:
                continue
            # score: keyword overlap, weighted by confidence; curated > derived
            score = overlap + f.confidence
            if f.source == "curated":
                score += 0.5
            if reference_month is not None and f.is_active(month=reference_month):
                score += 0.25  # mild boost for facts active "now"
            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [f for _, f in scored[:top_k]]
        n_active = sum(
            1 for f in top
            if reference_month is None or f.is_active(month=reference_month)
        )
        return TemporalRetrievalResult(
            facts=top,
            context=self.format_context(top, reference_month=reference_month),
            n_active=n_active,
        )

    @staticmethod
    def format_context(facts: list[TemporalFact], reference_month: Optional[int] = None) -> str:
        """Render facts as an injectable, time-scoped evidence block."""
        if not facts:
            return "[No temporal ontology facts found for this query]"
        lines = ["Temporal ontology facts (authoritative, time-scoped):"]
        for f in facts:
            lines.append(f"- {f.to_sentence()}")
        return "\n".join(lines)
