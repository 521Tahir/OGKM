"""
Context Assembler — builds structured prompt context from retrieval results.

Takes a HybridContext (ontology subgraph + text passages) and a QueryIntent
and produces a formatted string ready to be injected into the LLM prompt.

Design goals:
  - Token-budget aware: truncates to max_tokens estimate (rough 4-char/token)
  - Prioritizes ontology facts over raw text passages
  - Injects SHACL rules most relevant to the query task
  - Formats for Llama-3 chat template (no special tokens added here)

Used by: constrained_generator.py
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional, Any

from .query_understanding import QueryIntent
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.context_assembler")


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class AssembledContext:
    """Ready-to-inject context string plus diagnostics."""
    text: str                        # full context block for prompt injection
    n_concepts: int                  # how many ontology concepts included
    n_relations: int                 # how many ontology relations included
    n_passages: int                  # how many text passages included
    n_rules: int                     # how many SHACL rules included
    estimated_tokens: int            # rough token estimate (len/4)
    was_truncated: bool = False      # True if context was cut to fit budget


# ── Section builders ─────────────────────────────────────────────────────────

def _format_concept(c: dict) -> str:
    """Single-line concept summary."""
    name = c.get("name", "")
    cls  = c.get("class", "")
    desc = c.get("description", "")
    attrs = c.get("attributes", {})

    parts = [f"[{cls}] {name}"]
    if desc:
        parts.append(f"— {desc[:120]}")
    if attrs:
        attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:4])
        parts.append(f"({attr_str})")
    return " ".join(parts)


def _format_relation(r: dict) -> str:
    """Single-line relation summary."""
    subj = r.get("subject", r.get("head", ""))
    pred = r.get("predicate", r.get("relation", ""))
    obj  = r.get("object", r.get("tail", ""))
    conf = r.get("confidence", None)
    conf_str = f" [conf={conf:.2f}]" if conf is not None else ""
    return f"  {subj} —[{pred}]→ {obj}{conf_str}"


def _format_rule(rule: dict) -> str:
    """Single-line rule summary."""
    name     = rule.get("name", rule.get("id", "rule"))
    severity = rule.get("severity", "Warning")
    msg      = rule.get("message", rule.get("description", ""))
    return f"  [{severity}] {name}: {msg[:100]}"


def _format_passage(p: dict, idx: int) -> str:
    """Text passage with score."""
    text  = p.get("text", p.get("content", ""))[:300]
    score = p.get("score", 0.0)
    rid   = p.get("record_id", "")
    header = f"Passage {idx+1}" + (f" (id={rid}, score={score:.2f})" if rid else f" (score={score:.2f})")
    return f"  {header}:\n  {textwrap.fill(text, width=90, subsequent_indent='  ')}"


# ── Main class ────────────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Assemble an LLM-ready context block from retrieval results.

    Parameters
    ----------
    max_tokens : int
        Approximate token budget for the context block (default 1200).
        Uses 4-chars-per-token heuristic.
    concept_limit : int
        Max number of ontology concepts to include (default 20).
    relation_limit : int
        Max number of relations to include (default 30).
    passage_limit : int
        Max number of text passages to include (default 5).
    rule_limit : int
        Max number of SHACL rules to include (default 6).
    """

    def __init__(
        self,
        max_tokens: int = 1200,
        concept_limit: int = 20,
        relation_limit: int = 30,
        passage_limit: int = 5,
        rule_limit: int = 6,
    ):
        self.max_chars    = max_tokens * 4  # rough 4 chars per token
        self.concept_limit   = concept_limit
        self.relation_limit  = relation_limit
        self.passage_limit   = passage_limit
        self.rule_limit      = rule_limit

    # ── Public API ────────────────────────────────────────────────────────────

    def assemble(
        self,
        hybrid_context: Any,   # HybridContext (avoid circular import)
        query_intent: Optional[QueryIntent] = None,
        rules: Optional[list[dict]] = None,
    ) -> AssembledContext:
        """
        Build context from a HybridContext object.

        Parameters
        ----------
        hybrid_context : HybridContext
            Output of HybridRetriever.retrieve()
        query_intent : QueryIntent, optional
            Parsed query intent for theme-aware rule selection
        rules : list[dict], optional
            SHACL rules to inject; filtered by relevance if intent provided

        Returns
        -------
        AssembledContext
        """
        subgraph = hybrid_context.subgraph if hybrid_context else None
        passages = getattr(hybrid_context, "text_passages", []) or []
        mode     = getattr(hybrid_context, "mode", "unknown")

        # Collect ontology concepts and relations
        concepts  = []
        relations = []
        if subgraph:
            concepts  = list(getattr(subgraph, "concepts",  []) or [])
            relations = list(getattr(subgraph, "relations", []) or [])

        # Select relevant rules
        selected_rules = self._select_rules(rules or [], query_intent)

        # Build sections
        sections: list[str] = []
        sections.append(f"=== ONTOLOGY CONTEXT (mode: {mode}) ===")

        # ── Concepts ─────────────────────────────────────────────────────────
        shown_concepts = concepts[: self.concept_limit]
        if shown_concepts:
            sections.append("\n-- Relevant Concepts --")
            for c in shown_concepts:
                sections.append(_format_concept(c))

        # ── Relations ────────────────────────────────────────────────────────
        shown_relations = relations[: self.relation_limit]
        if shown_relations:
            sections.append("\n-- Relations --")
            for r in shown_relations:
                sections.append(_format_relation(r))

        # ── SHACL Rules ───────────────────────────────────────────────────────
        shown_rules = selected_rules[: self.rule_limit]
        if shown_rules:
            sections.append("\n-- Constraints & Rules --")
            for rule in shown_rules:
                sections.append(_format_rule(rule))

        # ── Text passages ─────────────────────────────────────────────────────
        shown_passages = passages[: self.passage_limit]
        if shown_passages:
            sections.append("\n-- Source Passages --")
            for i, p in enumerate(shown_passages):
                sections.append(_format_passage(p, i))

        sections.append("\n=== END CONTEXT ===")

        full_text = "\n".join(sections)

        # Truncate if over budget
        was_truncated = False
        if len(full_text) > self.max_chars:
            full_text = full_text[: self.max_chars] + "\n[... context truncated ...]"
            was_truncated = True

        return AssembledContext(
            text=full_text,
            n_concepts=len(shown_concepts),
            n_relations=len(shown_relations),
            n_passages=len(shown_passages),
            n_rules=len(shown_rules),
            estimated_tokens=len(full_text) // 4,
            was_truncated=was_truncated,
        )

    def assemble_minimal(self, query: str, concepts: list[dict]) -> AssembledContext:
        """
        Build a minimal context from a plain concept list (no retriever needed).
        Used when only concept linking results are available.
        """
        lines = ["=== ONTOLOGY CONTEXT (minimal) ==="]
        shown = concepts[: self.concept_limit]
        for c in shown:
            lines.append(_format_concept(c))
        lines.append("=== END CONTEXT ===")
        text = "\n".join(lines)
        return AssembledContext(
            text=text,
            n_concepts=len(shown),
            n_relations=0,
            n_passages=0,
            n_rules=0,
            estimated_tokens=len(text) // 4,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _select_rules(
        self, rules: list[dict], intent: Optional[QueryIntent]
    ) -> list[dict]:
        """
        Filter / rank rules by relevance to the detected task type and themes.
        Falls back to all rules if no intent is provided.
        """
        if not rules:
            return []
        if intent is None:
            return rules

        # Priority keywords per task type
        task_keywords: dict[str, list[str]] = {
            "safety":         ["risk", "danger", "safe", "warning", "hazard", "emergency"],
            "recommendation": ["suitable_for", "recommended", "season", "budget", "theme"],
            "itinerary":      ["duration", "distance", "sequence", "travel_time", "located_in"],
            "comparison":     ["rating", "price", "attribute", "difference"],
            "fact_lookup":    [],  # all rules equally relevant
        }

        # Theme keywords
        theme_kw = {
            "nature":    ["outdoor", "park", "beach", "mountain", "trail"],
            "culture":   ["heritage", "museum", "traditional", "festival"],
            "food":      ["restaurant", "cuisine", "dining"],
            "adventure": ["extreme", "dive", "surf", "climb"],
        }

        priority_terms: set[str] = set()
        task_kw = task_keywords.get(intent.task_type, [])
        priority_terms.update(t.lower() for t in task_kw)
        for theme in intent.constraints.themes:
            priority_terms.update(t.lower() for t in theme_kw.get(theme, []))

        if not priority_terms:
            return rules  # no filtering possible

        def _score(rule: dict) -> int:
            text = " ".join(str(v) for v in rule.values()).lower()
            return sum(1 for term in priority_terms if term in text)

        scored = sorted(rules, key=_score, reverse=True)
        return scored
