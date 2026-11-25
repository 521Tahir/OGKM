"""
Graph-of-Thought (GoT) Prompting for Ontology Construction — PDF §6.4

GoT extends chain-of-thought by modeling knowledge as interconnected graph
structures rather than linear chains. This is especially suited for:
  - Hierarchy induction (concepts as nodes, is-a as edges)
  - Structural coherence checking (detecting redundant/conflicting paths)
  - Subgraph-level consistency validation

The prompter produces structured graph fragments that can be directly merged
into the in-memory TourismGraphStore.

Usage:
    prompter = GoTPrompter(llm_client)
    graph_fragment = prompter.induce_subgraph(text, seed_concepts)
    consistency   = prompter.check_structural_coherence(concepts, relations)
    hierarchy     = prompter.build_hierarchy(concepts)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.got_prompter")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class GraphFragment:
    """A sub-graph returned by a GoT prompting call."""
    nodes: list[dict]         # [{name, concept_class, confidence}]
    edges: list[dict]         # [{subject, predicate, object, confidence}]
    reasoning_trace: str = "" # the LLM's step-by-step graph construction trace
    coherence_score: float = 1.0  # 0–1, estimated by self-evaluation step

    def to_dict(self) -> dict:
        return {
            "nodes":          self.nodes,
            "edges":          self.edges,
            "reasoning_trace": self.reasoning_trace,
            "coherence_score": self.coherence_score,
        }


@dataclass
class HierarchyNode:
    """One node in an inferred concept taxonomy."""
    name:     str
    parent:   Optional[str]  = None
    children: list[str]      = field(default_factory=list)
    depth:    int            = 0
    concept_class: str       = "Attraction"


@dataclass
class CoherenceReport:
    """Output of structural coherence checking."""
    is_coherent:  bool
    score:        float           # 0–1
    redundant:    list[str]       # description of redundant links
    conflicting:  list[str]       # description of conflicting relations
    missing:      list[str]       # suggested missing connections
    suggestions:  list[str]       # natural language improvement suggestions


# ── Prompt templates ──────────────────────────────────────────────────────────

_SUBGRAPH_PROMPT = """
You are a knowledge graph builder for a Jeju Island tourism ontology.

Given the text below, construct a concept graph using Graph-of-Thought reasoning:

STEP 1 — Identify NODES (tourism concepts):
  List each concept as: [CONCEPT: <name> | CLASS: <class> | CONFIDENCE: <0-1>]
  Classes: Attraction, Restaurant, Hotel, Transportation, Activity, Culture,
           Nature, Food, Event, Service, Destination, Accommodation

STEP 2 — Identify EDGES (semantic relations between concepts):
  List each relation as: [EDGE: <subject> -> <predicate> -> <object> | CONF: <0-1>]
  Predicates: locatedIn, hasActivity, isTypeOf, relatedTo, servedAt,
              nearBy, partOf, hasAttribute, operatesAt, requires

STEP 3 — Check graph coherence:
  - Are there cycles that shouldn't exist?
  - Are any edges redundant (A->B->C when A->C exists and is weaker)?
  - Are there missing obvious connections?
  State: COHERENCE_SCORE: <0-1>

TEXT:
{text}

SEED CONCEPTS (already known): {seed_concepts}

Now reason step by step, then output the graph:
"""

_HIERARCHY_PROMPT = """
You are building a concept taxonomy for Jeju Island tourism.

Using Graph-of-Thought, organize these concepts into a clean hierarchy:

STEP 1 — Identify the broadest categories (depth=0):
STEP 2 — Place each concept under its most specific parent:
STEP 3 — Check for concepts that belong in multiple places (flag them):
STEP 4 — Verify: no concept is its own ancestor

Output format (one per line):
HIER: <concept> | PARENT: <parent_or_ROOT> | DEPTH: <int>

CONCEPTS:
{concepts}
"""

_COHERENCE_CHECK_PROMPT = """
You are a knowledge graph validator for a Jeju Island tourism ontology.

Review this graph for structural issues using Graph-of-Thought:

STEP 1 — Find REDUNDANT edges (weaker A->C when A->B->C exists):
STEP 2 — Find CONFLICTING edges (same subject+object, contradictory predicates):
STEP 3 — Identify MISSING obvious connections from domain knowledge:
STEP 4 — Give an overall COHERENCE_SCORE (0=broken, 1=perfect):

CONCEPTS:
{concepts_summary}

RELATIONS (sample):
{relations_summary}

Output:
REDUNDANT: <list or "none">
CONFLICTING: <list or "none">
MISSING: <list or "none">
COHERENCE_SCORE: <0-1>
SUGGESTIONS: <improvement ideas>
"""


# ── Main class ────────────────────────────────────────────────────────────────

class GoTPrompter:
    """
    Graph-of-Thought prompting engine for ontology construction.

    All methods work offline (using heuristics) if no LLM client is provided.
    """

    def __init__(self, llm_client=None, max_tokens: int = 512) -> None:
        self.llm     = llm_client
        self.max_tok = max_tokens

    # ── Core GoT call ─────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        if self.llm is None:
            return ""
        try:
            return self.llm.call(
                prompt=prompt,
                system="You are a structured knowledge graph builder. Follow each step carefully.",
                max_tokens=self.max_tok,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"GoT LLM call failed: {exc}")
            return ""

    # ── Subgraph induction ────────────────────────────────────────────────────

    def induce_subgraph(
        self,
        text: str,
        seed_concepts: Optional[list[str]] = None,
    ) -> GraphFragment:
        """
        Extract a graph fragment from text using GoT step-by-step reasoning.

        Falls back to empty fragment if LLM is unavailable.
        """
        seeds_str = ", ".join(seed_concepts or []) or "none"
        prompt    = _SUBGRAPH_PROMPT.format(text=text[:1500], seed_concepts=seeds_str)
        raw       = self._call(prompt)

        nodes  = self._parse_nodes(raw)
        edges  = self._parse_edges(raw)
        coh    = self._parse_coherence_score(raw)

        logger.debug(
            f"GoT subgraph: {len(nodes)} nodes, {len(edges)} edges, coherence={coh:.2f}"
        )
        return GraphFragment(
            nodes=nodes, edges=edges, reasoning_trace=raw, coherence_score=coh
        )

    # ── Hierarchy induction ───────────────────────────────────────────────────

    def build_hierarchy(self, concepts: list[dict]) -> list[HierarchyNode]:
        """
        Organize concepts into a taxonomy using GoT hierarchy induction.

        Returns a flat list of HierarchyNode with parent pointers.
        """
        if not concepts:
            return []

        concept_list = "\n".join(
            f"- {c.get('name','?')} ({c.get('concept_class','?')})"
            for c in concepts[:100]
        )
        prompt = _HIERARCHY_PROMPT.format(concepts=concept_list)
        raw    = self._call(prompt)

        nodes: list[HierarchyNode] = []
        for line in raw.splitlines():
            line = line.strip()
            m = re.match(
                r"HIER:\s*(.+?)\s*\|\s*PARENT:\s*(.+?)\s*\|\s*DEPTH:\s*(\d+)", line
            )
            if m:
                name, parent, depth = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
                parent = None if parent.upper() == "ROOT" else parent
                nodes.append(HierarchyNode(
                    name=name, parent=parent, depth=depth,
                    concept_class=self._infer_class(name, concepts),
                ))

        # If LLM returned nothing, fall back to flat class-based grouping
        if not nodes:
            nodes = self._heuristic_hierarchy(concepts)

        # Build children lists
        name_map = {n.name: n for n in nodes}
        for node in nodes:
            if node.parent and node.parent in name_map:
                name_map[node.parent].children.append(node.name)

        logger.info(f"GoT hierarchy: {len(nodes)} nodes, max_depth={max((n.depth for n in nodes), default=0)}")
        return nodes

    # ── Coherence checking ────────────────────────────────────────────────────

    def check_structural_coherence(
        self,
        concepts: list[dict],
        relations: list[dict],
    ) -> CoherenceReport:
        """
        Evaluate graph structural coherence using GoT self-evaluation.

        Works as a pure heuristic check when LLM is unavailable.
        """
        # Always run heuristic checks
        heuristic = self._heuristic_coherence(concepts, relations)

        if self.llm is None:
            return heuristic

        # LLM-enhanced check
        conc_summary = "\n".join(
            f"- {c.get('name')} ({c.get('concept_class')})"
            for c in concepts[:30]
        )
        rel_sample = "\n".join(
            f"- {r.get('subject')} --[{r.get('predicate')}]--> {r.get('object')}"
            for r in relations[:30]
        )
        prompt = _COHERENCE_CHECK_PROMPT.format(
            concepts_summary=conc_summary,
            relations_summary=rel_sample,
        )
        raw = self._call(prompt)

        redundant   = self._extract_list_field(raw, "REDUNDANT")
        conflicting = self._extract_list_field(raw, "CONFLICTING")
        missing     = self._extract_list_field(raw, "MISSING")
        suggestions = self._extract_list_field(raw, "SUGGESTIONS")
        score       = self._parse_coherence_score(raw)

        # Merge with heuristic findings
        return CoherenceReport(
            is_coherent=score >= 0.7 and len(conflicting) == 0,
            score=round((score + heuristic.score) / 2, 3),
            redundant=redundant or heuristic.redundant,
            conflicting=conflicting or heuristic.conflicting,
            missing=missing or heuristic.missing,
            suggestions=suggestions,
        )

    # ── Heuristic helpers (offline fallbacks) ─────────────────────────────────

    def _heuristic_hierarchy(self, concepts: list[dict]) -> list[HierarchyNode]:
        """Group concepts by class into a 2-level flat taxonomy."""
        class_groups: dict[str, list[str]] = {}
        for c in concepts:
            cls  = c.get("concept_class", "Other")
            name = c.get("name", "")
            if name:
                class_groups.setdefault(cls, []).append(name)

        nodes: list[HierarchyNode] = []
        for cls, names in class_groups.items():
            nodes.append(HierarchyNode(name=cls, parent=None, depth=0, concept_class=cls))
            for n in names:
                nodes.append(HierarchyNode(name=n, parent=cls, depth=1, concept_class=cls))
        return nodes

    def _heuristic_coherence(
        self, concepts: list[dict], relations: list[dict]
    ) -> CoherenceReport:
        """Quick structural checks without LLM."""
        concept_names = {c.get("name", "") for c in concepts}

        # Detect self-loops
        self_loops = [
            f"{r['subject']} --[{r['predicate']}]--> {r['object']}"
            for r in relations
            if r.get("subject") == r.get("object")
        ]
        # Detect duplicate edges
        seen: set[tuple] = set()
        duplicates: list[str] = []
        for r in relations:
            key = (r.get("subject"), r.get("predicate"), r.get("object"))
            if key in seen:
                duplicates.append(f"{key[0]} --[{key[1]}]--> {key[2]}")
            seen.add(key)

        # Dangling edges (subject/object not in concept set)
        dangling = [
            f"{r.get('subject')} | {r.get('object')}"
            for r in relations[:500]
            if r.get("subject") not in concept_names or r.get("object") not in concept_names
        ][:10]

        issues = len(self_loops) + len(duplicates) + len(dangling)
        total  = max(1, len(relations))
        score  = max(0.0, 1.0 - issues / total)

        return CoherenceReport(
            is_coherent=issues == 0,
            score=round(score, 3),
            redundant=duplicates[:5],
            conflicting=self_loops[:5],
            missing=[],
            suggestions=[
                f"Remove {len(duplicates)} duplicate edges",
                f"Fix {len(self_loops)} self-loops",
                f"Resolve {len(dangling)} dangling references",
            ] if issues else ["Graph structure looks coherent"],
        )

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_nodes(self, raw: str) -> list[dict]:
        nodes: list[dict] = []
        for m in re.finditer(
            r"\[CONCEPT:\s*(.+?)\s*\|\s*CLASS:\s*(.+?)\s*\|\s*CONFIDENCE:\s*([\d.]+)\]",
            raw,
        ):
            nodes.append({
                "name":          m.group(1).strip(),
                "concept_class": m.group(2).strip(),
                "confidence":    float(m.group(3)),
                "source":        "got",
            })
        return nodes

    def _parse_edges(self, raw: str) -> list[dict]:
        edges: list[dict] = []
        for m in re.finditer(
            r"\[EDGE:\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+?)\s*\|\s*CONF:\s*([\d.]+)\]",
            raw,
        ):
            edges.append({
                "subject":    m.group(1).strip(),
                "predicate":  m.group(2).strip(),
                "object":     m.group(3).strip(),
                "confidence": float(m.group(4)),
                "source":     "got",
            })
        return edges

    def _parse_coherence_score(self, raw: str) -> float:
        m = re.search(r"COHERENCE_SCORE:\s*([\d.]+)", raw)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        return 0.8  # default if not found

    def _extract_list_field(self, raw: str, field: str) -> list[str]:
        m = re.search(rf"{field}:\s*(.+?)(?:\n[A-Z]|\Z)", raw, re.DOTALL)
        if not m:
            return []
        content = m.group(1).strip()
        if content.lower() in ("none", "n/a", ""):
            return []
        return [s.strip("- ").strip() for s in content.splitlines() if s.strip()]

    def _infer_class(self, name: str, concepts: list[dict]) -> str:
        for c in concepts:
            if c.get("name", "").lower() == name.lower():
                return c.get("concept_class", "Attraction")
        return "Attraction"
