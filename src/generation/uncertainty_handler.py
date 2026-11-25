"""
Uncertainty Handler — detect and communicate model uncertainty in generated responses.

Provides:
  1. Confidence estimation — token-level and semantic indicators of uncertainty
  2. Uncertainty classification — types: temporal, factual, entity, relational
  3. Hedge detection — identify existing hedges in the answer
  4. Calibration helpers — map raw scores to calibrated confidence levels
  5. Uncertainty-aware prompt injection — add calibration instructions to prompts

Used by: constrained_generator.py, exp005/run_constraints.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("generation.uncertainty_handler")


# ── Linguistic uncertainty markers ────────────────────────────────────────────

# Strong hedges → high uncertainty
_STRONG_HEDGE_RE = re.compile(
    r"\b(i('m| am) not (sure|certain)|i (don'?t|do not) know|"
    r"unclear|uncertain|i (cannot|can'?t) (confirm|verify|say)|"
    r"this (may|might|could) (be )?incorrect|i apologize|no (information|data) (available|found))\b",
    re.I,
)

# Moderate hedges → medium uncertainty
_MODERATE_HEDGE_RE = re.compile(
    r"\b(might|may|could|possibly|perhaps|probably|likely|"
    r"i (believe|think|assume|suspect)|as far as i know|"
    r"to my knowledge|generally|typically|usually|often)\b",
    re.I,
)

# Weak hedges → low uncertainty
_WEAK_HEDGE_RE = re.compile(
    r"\b(sometimes|occasionally|can be|tends to|in (most|many) cases|"
    r"depending on|varies|approximately|roughly|around)\b",
    re.I,
)

# Temporal uncertainty markers
_TEMPORAL_RE = re.compile(
    r"\b(as of|at the time|recently|currently|at present|"
    r"this (information|data) (may|might) (be|have) (outdated|changed)|"
    r"please (check|verify|confirm) (the )?latest)\b",
    re.I,
)


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class HedgeSpan:
    text: str
    strength: str   # "strong" | "moderate" | "weak"
    start: int
    end: int


@dataclass
class UncertaintyReport:
    """Uncertainty analysis for a single generated response."""
    answer: str

    # Estimated confidence (0=very uncertain, 1=very confident)
    confidence: float

    # Uncertainty level bucket
    level: str   # "high" | "medium" | "low" | "none"

    # Uncertainty types detected
    types: list[str]   # e.g., ["temporal", "entity", "factual"]

    # Detected hedge spans
    hedges: list[HedgeSpan] = field(default_factory=list)

    # Recommended action
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "level": self.level,
            "types": self.types,
            "n_hedges": len(self.hedges),
            "recommendation": self.recommendation,
        }


# ── Calibration helpers ───────────────────────────────────────────────────────

def _level_from_confidence(conf: float) -> str:
    if conf >= 0.80:
        return "none"
    if conf >= 0.60:
        return "low"
    if conf >= 0.40:
        return "medium"
    return "high"


def _recommendation_for_level(level: str) -> str:
    return {
        "none":   "Answer appears confident. No additional verification needed.",
        "low":    "Minor hedges detected. Answer is likely reliable but verify specific details.",
        "medium": "Moderate uncertainty. Cross-check key claims with authoritative sources.",
        "high":   "High uncertainty. Answer may be incomplete or incorrect. Human review recommended.",
    }.get(level, "")


# ── Main class ────────────────────────────────────────────────────────────────

class UncertaintyHandler:
    """
    Analyze and handle uncertainty in LLM-generated answers.

    Workflow:
      1. `analyze(answer)` → UncertaintyReport (always safe to call)
      2. `inject_calibration_prompt(prompt)` → augmented prompt asking LLM to express uncertainty
      3. `annotate_answer(answer, report)` → optionally append uncertainty notice to answer

    Parameters
    ----------
    strong_weight : float   Confidence penalty per strong hedge (default 0.20)
    moderate_weight : float  Confidence penalty per moderate hedge (default 0.08)
    weak_weight : float      Confidence penalty per weak hedge (default 0.03)
    base_confidence : float  Starting confidence before hedge analysis (default 0.95)
    """

    def __init__(
        self,
        strong_weight:   float = 0.20,
        moderate_weight: float = 0.08,
        weak_weight:     float = 0.03,
        base_confidence: float = 0.95,
    ):
        self.strong_weight   = strong_weight
        self.moderate_weight = moderate_weight
        self.weak_weight     = weak_weight
        self.base_confidence = base_confidence

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, answer: str) -> UncertaintyReport:
        """
        Analyze the given answer text for uncertainty signals.

        Returns an UncertaintyReport with confidence score and level.
        """
        hedges = self._extract_hedges(answer)
        confidence = self._estimate_confidence(hedges, answer)
        level = _level_from_confidence(confidence)
        types = self._detect_types(answer, hedges)
        rec = _recommendation_for_level(level)

        report = UncertaintyReport(
            answer=answer,
            confidence=confidence,
            level=level,
            types=types,
            hedges=hedges,
            recommendation=rec,
        )
        logger.debug(
            "Uncertainty: conf=%.2f level=%s types=%s hedges=%d",
            confidence, level, types, len(hedges),
        )
        return report

    def inject_calibration_prompt(self, system_prompt: str) -> str:
        """
        Append calibration instructions to a system prompt.

        Instructs the LLM to:
          - Express uncertainty with explicit hedges when not confident
          - Distinguish between confirmed ontology facts and inferred claims
          - Avoid fabricating specific details not in context
        """
        calibration_suffix = (
            "\n\n[CALIBRATION INSTRUCTIONS]\n"
            "- If you are not certain about a specific fact, say so explicitly using phrases like "
            "'I believe', 'This may vary', or 'Please verify'.\n"
            "- Ground all specific claims (names, numbers, hours, prices) in the provided context.\n"
            "- Do NOT invent proper nouns, location names, or statistics not in the context.\n"
            "- Prefer a shorter accurate answer over a longer one that includes guesses.\n"
            "- If the context does not cover the query, say what you know and flag the gap.\n"
        )
        return system_prompt + calibration_suffix

    def annotate_answer(
        self, answer: str, report: UncertaintyReport, append: bool = True
    ) -> str:
        """
        Optionally append an uncertainty notice to the answer.

        Only appends if level is "medium" or "high".
        Set append=False to skip appending (e.g., for API responses where you
        want the report but not the annotation).
        """
        if not append or report.level in ("none", "low"):
            return answer

        notice = self._format_notice(report)
        return answer.rstrip() + "\n\n" + notice

    def calibrate_score(self, raw_score: float, n_violations: int) -> float:
        """
        Adjust a raw grounding/confidence score by violation count.

        Used to produce a final calibrated score for evaluation metrics.
        """
        penalty = min(0.5, n_violations * 0.05)
        return max(0.0, min(1.0, raw_score - penalty))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_hedges(self, text: str) -> list[HedgeSpan]:
        hedges: list[HedgeSpan] = []
        for strength, pattern in [
            ("strong",   _STRONG_HEDGE_RE),
            ("moderate", _MODERATE_HEDGE_RE),
            ("weak",     _WEAK_HEDGE_RE),
        ]:
            for m in pattern.finditer(text):
                hedges.append(HedgeSpan(
                    text=m.group(0),
                    strength=strength,
                    start=m.start(),
                    end=m.end(),
                ))
        # De-duplicate overlapping spans (keep first)
        hedges.sort(key=lambda h: h.start)
        deduped: list[HedgeSpan] = []
        last_end = -1
        for h in hedges:
            if h.start >= last_end:
                deduped.append(h)
                last_end = h.end
        return deduped

    def _estimate_confidence(self, hedges: list[HedgeSpan], text: str) -> float:
        """Compute confidence score starting from base and applying hedge penalties."""
        conf = self.base_confidence
        for h in hedges:
            if h.strength == "strong":
                conf -= self.strong_weight
            elif h.strength == "moderate":
                conf -= self.moderate_weight
            else:
                conf -= self.weak_weight
        # Additional penalty for very short answers
        if len(text.strip()) < 30:
            conf -= 0.10
        return max(0.0, min(1.0, conf))

    def _detect_types(self, text: str, hedges: list[HedgeSpan]) -> list[str]:
        """Classify what dimension of uncertainty is expressed."""
        types: list[str] = []

        if _TEMPORAL_RE.search(text):
            types.append("temporal")

        # Entity uncertainty: strong hedges near proper-noun-like patterns
        entity_re = re.compile(r"\b[A-Z][a-zA-Z]+\b")
        if any(h.strength == "strong" for h in hedges) and entity_re.search(text):
            types.append("entity")

        # Factual uncertainty: moderate hedges with factual language
        factual_re = re.compile(r"\b(cost|price|hours|open|close|distance|km|mile|height|year|date)\b", re.I)
        if factual_re.search(text) and any(h.strength in ("strong", "moderate") for h in hedges):
            types.append("factual")

        # Relational: hedges near relation keywords
        relation_re = re.compile(r"\b(located|found|near|part of|belongs|contains|includes)\b", re.I)
        if relation_re.search(text) and hedges:
            types.append("relational")

        if not types and hedges:
            types.append("general")

        return types

    def _format_notice(self, report: UncertaintyReport) -> str:
        """Format a human-readable uncertainty notice."""
        type_str = ", ".join(report.types) if report.types else "general"
        return (
            f"[Uncertainty Notice — {report.level.upper()}] "
            f"Confidence: {report.confidence:.0%}. "
            f"Type(s): {type_str}. "
            f"{report.recommendation}"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_default_handler: Optional[UncertaintyHandler] = None


def get_uncertainty_handler() -> UncertaintyHandler:
    global _default_handler
    if _default_handler is None:
        _default_handler = UncertaintyHandler()
    return _default_handler
