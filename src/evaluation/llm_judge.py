"""
LLM-as-Judge evaluation module (Week 10).

Uses Llama 3.1-8B-Instruct as an automatic judge to score system responses
on three dimensions (factual correctness, constraint satisfaction, fluency).

Paper: Table 10 — LLM-as-Judge Scores by Mode

Dimensions
----------
1. factual_correctness    (1–5): Are stated facts verifiable / ontology-grounded?
2. constraint_satisfaction (1–5): Does the response obey the query constraints?
3. fluency_coherence       (1–5): Is the response fluent and well-structured?

Usage
-----
    judge = LLMJudge(llm_client)   # pass loaded LocalLLMClient
    score = judge.judge_factual_correctness(query, answer, context)
    scores = judge.judge_batch(items, mode)
    agg = judge.aggregate(scores)
    print(judge.compare_modes(agg_by_mode))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.llm_judge")

# ── Scoring prompts ────────────────────────────────────────────────────────────

_FACTUAL_PROMPT = """\
You are an expert tourism fact-checker evaluating a response about Jeju Island tourism.

Query: {query}

Response to evaluate:
{answer}

Reference context (verified ontology facts):
{context}

Rate the FACTUAL CORRECTNESS of the response on a scale of 1–5:
1 = Many fabricated facts not in context
2 = Several unsupported claims
3 = Mix of supported and unsupported claims
4 = Mostly supported with minor gaps
5 = All claims verifiable from context

Respond with ONLY a single integer (1, 2, 3, 4, or 5), followed by a brief reason.
Format: <score>\\n<reason>"""

_CONSTRAINT_PROMPT = """\
You are evaluating whether a tourism response satisfies the user's query requirements.

Query: {query}

Response to evaluate:
{answer}

Rate CONSTRAINT SATISFACTION on a scale of 1–5:
1 = Ignores key requirements
2 = Addresses few requirements
3 = Partially addresses requirements
4 = Mostly satisfies requirements
5 = Fully satisfies all stated requirements

Respond with ONLY a single integer (1, 2, 3, 4, or 5), followed by a brief reason.
Format: <score>\\n<reason>"""

_FLUENCY_PROMPT = """\
You are evaluating the quality of a tourism information response.

Response to evaluate:
{answer}

Rate FLUENCY AND COHERENCE on a scale of 1–5:
1 = Incoherent, grammatically broken
2 = Mostly incoherent or repetitive
3 = Understandable but awkward
4 = Fluent with minor issues
5 = Fluent, well-structured, natural

Respond with ONLY a single integer (1, 2, 3, 4, or 5), followed by a brief reason.
Format: <score>\\n<reason>"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    """Scores for a single response from the LLM judge."""
    query_id:               int
    mode:                   str
    factual_correctness:    float   # 1–5
    constraint_satisfaction: float  # 1–5
    fluency_coherence:      float   # 1–5
    factual_reason:         str = ""
    constraint_reason:      str = ""
    fluency_reason:         str = ""

    @property
    def overall(self) -> float:
        return round(mean([
            self.factual_correctness,
            self.constraint_satisfaction,
            self.fluency_coherence,
        ]), 2)

    def to_dict(self) -> dict:
        return {
            "query_id":               self.query_id,
            "mode":                   self.mode,
            "factual_correctness":    round(self.factual_correctness, 2),
            "constraint_satisfaction": round(self.constraint_satisfaction, 2),
            "fluency_coherence":      round(self.fluency_coherence, 2),
            "overall":                self.overall,
            "factual_reason":         self.factual_reason,
            "constraint_reason":      self.constraint_reason,
            "fluency_reason":         self.fluency_reason,
        }


@dataclass
class JudgeAggregate:
    """Aggregated judge scores for one mode."""
    mode:                        str
    n_scored:                    int
    avg_factual_correctness:     float
    avg_constraint_satisfaction: float
    avg_fluency_coherence:       float
    avg_overall:                 float

    def to_dict(self) -> dict:
        return {
            "mode":                         self.mode,
            "n_scored":                     self.n_scored,
            "avg_factual_correctness":      round(self.avg_factual_correctness, 3),
            "avg_constraint_satisfaction":  round(self.avg_constraint_satisfaction, 3),
            "avg_fluency_coherence":        round(self.avg_fluency_coherence, 3),
            "avg_overall":                  round(self.avg_overall, 3),
        }


# ── Score parser ───────────────────────────────────────────────────────────────

def _parse_score(raw: str) -> tuple[float, str]:
    """
    Extract numeric score (1–5) and reason from LLM judge output.

    Handles formats:
      - "4\\nReason text"
      - "Score: 4. Because..."
      - "4/5 ..."
    Returns (score, reason). Falls back to 3.0 if unparseable.
    """
    raw = raw.strip()
    # Try first character / first line for the integer
    first_line = raw.split("\n")[0].strip()
    m = re.search(r"\b([1-5])\b", first_line)
    if not m:
        # Try full text
        m = re.search(r"\b([1-5])\b", raw)
    score = float(m.group(1)) if m else 3.0

    # Reason = everything after the first line, or after a period/newline
    parts = raw.split("\n", 1)
    reason = parts[1].strip() if len(parts) > 1 else ""
    if not reason:
        # Try after first sentence
        idx = raw.find(".")
        reason = raw[idx + 1:].strip() if idx != -1 else ""

    return score, reason[:300]


# ── Main judge class ───────────────────────────────────────────────────────────

class LLMJudge:
    """
    LLM-as-Judge evaluator using Llama 3.1-8B-Instruct.

    Parameters
    ----------
    llm_client : LocalLLMClient | None
        A loaded LLM client. If None, uses fallback heuristic scoring (offline).
    max_tokens : int
        Max tokens for judge response (keep short — just score + reason).
    """

    def __init__(
        self,
        llm_client=None,
        max_tokens: int = 64,
    ):
        self._llm = llm_client
        self._max_tokens = max_tokens

    # ── Public API ─────────────────────────────────────────────────────────────

    def judge_factual_correctness(
        self,
        query: str,
        answer: str,
        context: str = "",
    ) -> tuple[float, str]:
        """Return (score 1-5, reason)."""
        prompt = _FACTUAL_PROMPT.format(
            query=query[:300],
            answer=answer[:600],
            context=context[:400] or "(no context provided)",
        )
        return self._call_judge(prompt)

    def judge_constraint_satisfaction(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, str]:
        """Return (score 1-5, reason)."""
        prompt = _CONSTRAINT_PROMPT.format(
            query=query[:300],
            answer=answer[:600],
        )
        return self._call_judge(prompt)

    def judge_fluency_coherence(
        self,
        answer: str,
    ) -> tuple[float, str]:
        """Return (score 1-5, reason)."""
        prompt = _FLUENCY_PROMPT.format(answer=answer[:600])
        return self._call_judge(prompt)

    def judge_response(
        self,
        query_id: int,
        mode: str,
        query: str,
        answer: str,
        context: str = "",
    ) -> JudgeScore:
        """Score all three dimensions for a single response."""
        fc, fr = self.judge_factual_correctness(query, answer, context)
        cs, cr = self.judge_constraint_satisfaction(query, answer)
        fl, lr = self.judge_fluency_coherence(answer)

        return JudgeScore(
            query_id=query_id,
            mode=mode,
            factual_correctness=fc,
            constraint_satisfaction=cs,
            fluency_coherence=fl,
            factual_reason=fr,
            constraint_reason=cr,
            fluency_reason=lr,
        )

    def judge_batch(
        self,
        items: list[dict],
        mode: str,
        log_every: int = 10,
    ) -> list[JudgeScore]:
        """
        Score a batch of items.

        Each item dict must have:
          - query_id: int
          - query: str
          - answer: str
          - context: str (optional)
        """
        scores = []
        for i, item in enumerate(items):
            if i % log_every == 0:
                logger.info(f"Judging item {i+1}/{len(items)} (mode={mode})")
            score = self.judge_response(
                query_id=item.get("query_id", i),
                mode=mode,
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                context=item.get("context", ""),
            )
            scores.append(score)
        return scores

    def aggregate(
        self,
        scores: list[JudgeScore],
        mode: str,
    ) -> JudgeAggregate:
        """Aggregate per-response scores into mode-level summary."""
        n = len(scores)
        if n == 0:
            return JudgeAggregate(
                mode=mode, n_scored=0,
                avg_factual_correctness=0,
                avg_constraint_satisfaction=0,
                avg_fluency_coherence=0,
                avg_overall=0,
            )
        return JudgeAggregate(
            mode=mode,
            n_scored=n,
            avg_factual_correctness=mean(s.factual_correctness for s in scores),
            avg_constraint_satisfaction=mean(s.constraint_satisfaction for s in scores),
            avg_fluency_coherence=mean(s.fluency_coherence for s in scores),
            avg_overall=mean(s.overall for s in scores),
        )

    def compare_modes(
        self,
        agg_by_mode: dict[str, JudgeAggregate],
    ) -> str:
        """Return a markdown table comparing judge scores across modes."""
        lines = [
            "| Mode | Factual (1-5) | Constraint (1-5) | Fluency (1-5) | Overall (1-5) |",
            "|------|:------------:|:----------------:|:-------------:|:-------------:|",
        ]
        for mode, agg in agg_by_mode.items():
            lines.append(
                f"| {mode} "
                f"| {agg.avg_factual_correctness:.2f} "
                f"| {agg.avg_constraint_satisfaction:.2f} "
                f"| {agg.avg_fluency_coherence:.2f} "
                f"| {agg.avg_overall:.2f} |"
            )
        return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _call_judge(self, prompt: str) -> tuple[float, str]:
        """Call LLM judge. Falls back to heuristic if no LLM loaded."""
        if self._llm is None:
            return self._heuristic_score(prompt)
        try:
            raw = self._llm.call(
                prompt=prompt,
                system=(
                    "You are an expert evaluator. Respond with ONLY a score (1-5) "
                    "on the first line, then a one-sentence reason."
                ),
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            return _parse_score(raw)
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e} — using heuristic fallback")
            return self._heuristic_score(prompt)

    @staticmethod
    def _heuristic_score(prompt: str) -> tuple[float, str]:
        """
        Offline heuristic scoring when no LLM is available.

        Based on simple text signals present in the prompt/answer.
        Used for structural testing only — real evaluation requires LLM.
        """
        prompt_lower = prompt.lower()

        # Extract the answer section
        answer_start = prompt_lower.find("response to evaluate:")
        if answer_start == -1:
            answer_start = prompt_lower.find("response:")
        answer_text = prompt[answer_start:answer_start + 600] if answer_start != -1 else prompt

        score = 3.0  # default neutral

        # Positive signals
        if any(k in answer_text.lower() for k in [
            "hallasan", "seongsan", "jeju", "manjanggul", "olle", "udo island"
        ]):
            score += 0.5
        if len(answer_text.split()) > 30:
            score += 0.3
        if any(k in answer_text.lower() for k in [
            "verified", "ontology", "confirmed", "knowledge base"
        ]):
            score += 0.4

        # Negative signals
        if any(k in answer_text.lower() for k in [
            "cannot confirm", "not found", "unable to verify"
        ]):
            score -= 0.3
        if len(answer_text.split()) < 10:
            score -= 0.5

        score = max(1.0, min(5.0, score))
        return round(score, 1), "heuristic offline estimate"
