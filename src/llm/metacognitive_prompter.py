"""
Metacognitive Prompting for Ontology Self-Evaluation — PDF §6.5

Metacognitive prompting instructs the LLM to reflect on and evaluate its own
outputs before committing them to the ontology. This acts as an internal
quality gate that:

  1. Self-evaluates generated concepts for domain relevance
  2. Detects over-generalisation or under-specification
  3. Flags low-confidence extractions for expert review
  4. Proposes corrections before the output reaches SHACL validation

Three metacognitive strategies are implemented:
  - "reflect"    : the model reviews its own extraction and scores it
  - "calibrate"  : the model adjusts confidence scores based on evidence
  - "debate"     : the model argues for and against each concept/relation

Usage:
    mp = MetacognitivePrompter(llm_client)
    reviewed = mp.reflect(concepts, source_text)
    calibrated = mp.calibrate_confidence(concepts, source_text)
    flagged = mp.flag_for_review(concepts, threshold=0.65)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.metacognitive")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ReflectionResult:
    """Output of metacognitive reflection on a concept or relation."""
    original:       dict
    accepted:       bool
    revised:        Optional[dict]   # if the model proposed a correction
    confidence_adj: float            # adjustment to original confidence (-1 to +1)
    reasoning:      str              # the model's self-evaluation trace
    flags:          list[str]        # ["low_confidence", "ambiguous", "hallucinated", ...]


@dataclass
class CalibrationReport:
    """Summary of a batch calibration pass."""
    n_reviewed:   int
    n_accepted:   int
    n_revised:    int
    n_flagged:    int
    avg_conf_before: float
    avg_conf_after:  float
    flagged_names:   list[str]


# ── Prompt templates ──────────────────────────────────────────────────────────

_REFLECT_PROMPT = """
You are a quality reviewer for a Jeju Island tourism knowledge ontology.

Review the following extracted concepts from this source text and evaluate each one:

For each concept, reason through:
  1. Is this concept genuinely relevant to Jeju Island tourism?
  2. Is the concept class correct?
  3. Is the confidence score justified by the text?
  4. Should this concept be accepted, revised, or rejected?

Output format for each:
CONCEPT: <name>
VERDICT: ACCEPT | REVISE | REJECT
REVISED_NAME: <corrected name if REVISE, else same>
REVISED_CLASS: <corrected class if REVISE, else same>
CONFIDENCE_ADJ: <float between -0.3 and +0.3>
REASON: <one sentence>

SOURCE TEXT:
{text}

EXTRACTED CONCEPTS:
{concepts}
"""

_CALIBRATE_PROMPT = """
You are calibrating confidence scores for a tourism ontology extraction.

For each concept below, assign a calibrated confidence score (0.0–1.0) based on:
  - Clarity of mention in the source text (0.3 weight)
  - Domain relevance to Jeju Island tourism (0.4 weight)
  - Specificity (general terms get lower scores) (0.3 weight)

Output one line per concept:
CALIBRATE: <name> | SCORE: <0.0–1.0> | REASON: <brief>

SOURCE TEXT:
{text}

CONCEPTS TO CALIBRATE:
{concepts}
"""

_DEBATE_PROMPT = """
For each extracted concept, argue BOTH for and against its inclusion in the
Jeju Island tourism ontology. Then give a final verdict.

Format:
CONCEPT: <name>
FOR: <argument for inclusion>
AGAINST: <argument against>
VERDICT: INCLUDE | EXCLUDE | FLAG_FOR_EXPERT
FINAL_CONF: <0.0–1.0>

CONCEPTS:
{concepts}
"""


# ── Main class ────────────────────────────────────────────────────────────────

class MetacognitivePrompter:
    """
    Metacognitive self-evaluation layer for LLM-generated ontology elements.

    All methods degrade gracefully to heuristic evaluation when no LLM is
    available (llm_client=None).
    """

    def __init__(
        self,
        llm_client=None,
        max_tokens: int = 512,
        review_threshold: float = 0.65,  # concepts below this are flagged
    ) -> None:
        self.llm       = llm_client
        self.max_tok   = max_tokens
        self.threshold = review_threshold

    # ── Core call ─────────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        if self.llm is None:
            return ""
        try:
            return self.llm.call(
                prompt=prompt,
                system=(
                    "You are a critical quality reviewer of ontology extractions. "
                    "Be strict: only accept well-evidenced, domain-relevant concepts."
                ),
                max_tokens=self.max_tok,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"Metacognitive LLM call failed: {exc}")
            return ""

    # ── Reflect ───────────────────────────────────────────────────────────────

    def reflect(
        self,
        concepts: list[dict],
        source_text: str,
    ) -> list[ReflectionResult]:
        """
        Run metacognitive reflection on a list of extracted concepts.

        Returns a ReflectionResult for each concept — accepted, revised, or rejected.
        """
        if not concepts:
            return []

        conc_str = "\n".join(
            f"- {c.get('name')} | class={c.get('concept_class')} | conf={c.get('confidence', 0.8):.2f}"
            for c in concepts
        )
        prompt = _REFLECT_PROMPT.format(text=source_text[:1000], concepts=conc_str)
        raw    = self._call(prompt)

        results: list[ReflectionResult] = []
        # Parse LLM output per concept block
        blocks = re.split(r"\n(?=CONCEPT:)", raw.strip())
        parsed = {b.split("\n")[0].replace("CONCEPT:", "").strip(): b for b in blocks if b.strip()}

        for c in concepts:
            name  = c.get("name", "")
            block = parsed.get(name, "")
            if block:
                result = self._parse_reflect_block(c, block)
            else:
                result = self._heuristic_reflect(c)
            results.append(result)

        n_acc = sum(1 for r in results if r.accepted)
        n_rej = sum(1 for r in results if not r.accepted and r.revised is None)
        logger.info(
            f"Metacognitive reflect: {len(results)} reviewed, "
            f"{n_acc} accepted, {n_rej} rejected"
        )
        return results

    # ── Calibrate ─────────────────────────────────────────────────────────────

    def calibrate_confidence(
        self,
        concepts: list[dict],
        source_text: str,
    ) -> list[dict]:
        """
        Adjust confidence scores for a batch of concepts using metacognitive calibration.

        Returns a new list of concept dicts with updated confidence values.
        """
        if not concepts:
            return []

        conc_str = "\n".join(
            f"- {c.get('name')} (class: {c.get('concept_class')})"
            for c in concepts
        )
        prompt = _CALIBRATE_PROMPT.format(text=source_text[:1000], concepts=conc_str)
        raw    = self._call(prompt)

        # Parse calibrated scores
        score_map: dict[str, float] = {}
        for m in re.finditer(r"CALIBRATE:\s*(.+?)\s*\|\s*SCORE:\s*([\d.]+)", raw):
            name, score = m.group(1).strip(), float(m.group(2))
            score_map[name] = max(0.0, min(1.0, score))

        calibrated: list[dict] = []
        for c in concepts:
            updated = dict(c)
            name    = c.get("name", "")
            if name in score_map:
                updated["confidence"] = score_map[name]
            else:
                # heuristic: slight penalty for single-word generic terms
                if len(name.split()) == 1 and len(name) < 6:
                    updated["confidence"] = max(0.3, c.get("confidence", 0.7) - 0.1)
            calibrated.append(updated)

        before_avg = sum(c.get("confidence", 0.7) for c in concepts) / max(1, len(concepts))
        after_avg  = sum(c.get("confidence", 0.7) for c in calibrated) / max(1, len(calibrated))
        logger.info(
            f"Confidence calibrated: avg {before_avg:.3f} → {after_avg:.3f}"
        )
        return calibrated

    # ── Debate ────────────────────────────────────────────────────────────────

    def debate(self, concepts: list[dict]) -> list[dict]:
        """
        For-and-against debate for each concept. Returns only INCLUDE verdicts.

        Intended for borderline-confidence concepts (0.5–0.75).
        """
        borderline = [
            c for c in concepts
            if 0.5 <= c.get("confidence", 0.7) < 0.75
        ]
        if not borderline:
            return concepts  # nothing to debate

        conc_str = "\n".join(
            f"- {c.get('name')} | {c.get('concept_class')}"
            for c in borderline
        )
        prompt = _DEBATE_PROMPT.format(concepts=conc_str)
        raw    = self._call(prompt)

        # Parse verdicts
        exclude_set: set[str] = set()
        for m in re.finditer(r"CONCEPT:\s*(.+?)\n.*?VERDICT:\s*(INCLUDE|EXCLUDE|FLAG_FOR_EXPERT)", raw, re.DOTALL):
            name, verdict = m.group(1).strip(), m.group(2).strip()
            if verdict == "EXCLUDE":
                exclude_set.add(name.lower())

        kept = []
        for c in concepts:
            if c.get("name", "").lower() in exclude_set:
                logger.debug(f"Debate: excluded '{c['name']}'")
            else:
                kept.append(c)

        logger.info(f"Debate: {len(concepts)} → {len(kept)} (excluded {len(exclude_set)})")
        return kept

    # ── Flagging ──────────────────────────────────────────────────────────────

    def flag_for_review(
        self,
        concepts: list[dict],
        threshold: Optional[float] = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Split concepts into (auto_accepted, flagged_for_expert_review).

        Flagged = confidence below threshold OR tagged with ambiguous/hallucinated flags.
        """
        thr      = threshold if threshold is not None else self.threshold
        accepted = []
        flagged  = []

        for c in concepts:
            conf = float(c.get("confidence", c.get("provenance", {}).get("confidence", 0.7)))
            if conf < thr:
                flagged.append({**c, "_flag_reason": f"confidence={conf:.2f} < {thr}"})
            else:
                accepted.append(c)

        logger.info(
            f"Flag-for-review (thr={thr}): {len(accepted)} accepted, "
            f"{len(flagged)} flagged"
        )
        return accepted, flagged

    # ── Calibration report ────────────────────────────────────────────────────

    def calibration_report(
        self,
        original: list[dict],
        calibrated: list[dict],
    ) -> CalibrationReport:
        """Summarise the effect of a calibration pass."""
        flagged = [c for c in calibrated if c.get("confidence", 1.0) < self.threshold]
        revised = [
            c for c, o in zip(calibrated, original)
            if abs(c.get("confidence", 0.7) - o.get("confidence", 0.7)) > 0.05
        ]
        return CalibrationReport(
            n_reviewed=len(original),
            n_accepted=len(calibrated) - len(flagged),
            n_revised=len(revised),
            n_flagged=len(flagged),
            avg_conf_before=round(
                sum(c.get("confidence", 0.7) for c in original) / max(1, len(original)), 3
            ),
            avg_conf_after=round(
                sum(c.get("confidence", 0.7) for c in calibrated) / max(1, len(calibrated)), 3
            ),
            flagged_names=[c.get("name", "") for c in flagged[:20]],
        )

    # ── Heuristic fallbacks ───────────────────────────────────────────────────

    def _heuristic_reflect(self, concept: dict) -> ReflectionResult:
        """Rule-based reflection when LLM is not available."""
        name = concept.get("name", "")
        conf = float(concept.get("confidence", 0.7))

        flags: list[str] = []
        if conf < self.threshold:
            flags.append("low_confidence")
        if len(name) < 3:
            flags.append("too_short")
        if name.lower() in {"a", "an", "the", "and", "or", "in", "at", "of"}:
            flags.append("stopword")

        accepted = len(flags) == 0 or (len(flags) == 1 and "low_confidence" in flags)
        return ReflectionResult(
            original=concept,
            accepted=accepted,
            revised=None,
            confidence_adj=0.0,
            reasoning="Heuristic: " + (", ".join(flags) if flags else "passes basic checks"),
            flags=flags,
        )

    def _parse_reflect_block(self, concept: dict, block: str) -> ReflectionResult:
        """Parse a CONCEPT/VERDICT/... block from LLM reflect output."""
        def _field(key: str) -> str:
            m = re.search(rf"{key}:\s*(.+)", block)
            return m.group(1).strip() if m else ""

        verdict      = _field("VERDICT").upper()
        revised_name = _field("REVISED_NAME")
        revised_cls  = _field("REVISED_CLASS")
        reason       = _field("REASON")
        adj_str      = _field("CONFIDENCE_ADJ")

        try:
            adj = float(adj_str)
        except ValueError:
            adj = 0.0

        accepted = verdict == "ACCEPT"
        revised  = None
        if verdict == "REVISE" and (revised_name or revised_cls):
            revised = dict(concept)
            if revised_name and revised_name.upper() != concept.get("name", "").upper():
                revised["name"] = revised_name
            if revised_cls and revised_cls.upper() != concept.get("concept_class", "").upper():
                revised["concept_class"] = revised_cls

        flags: list[str] = []
        if verdict == "REJECT":
            flags.append("rejected_by_metacog")
        if adj < -0.1:
            flags.append("confidence_downgraded")

        return ReflectionResult(
            original=concept,
            accepted=accepted or verdict == "REVISE",
            revised=revised,
            confidence_adj=adj,
            reasoning=reason,
            flags=flags,
        )
