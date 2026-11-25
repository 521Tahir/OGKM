"""
Concept normalizer and canonicalizer.

Merges duplicate concepts, resolves aliases, and standardizes names
across the global ontology extracted from all records.

Strategies:
1. Exact string match (case-insensitive)
2. Fuzzy match using rapidfuzz (edit distance)
3. Known alias table lookup
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schema import OntologyConcept
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.normalizer")

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not available — fuzzy dedup disabled.")

# Manual alias table for well-known tourism entities
KNOWN_ALIASES: dict[str, str] = {
    "bali island": "bali",
    "ball island": "bali",
    "jeju island": "jeju",
    "jeju-do": "jeju",
    "south korea": "south korea",
    "korea": "south korea",
    "rok": "south korea",
    "jakarta city": "jakarta",
    "yogyakarta city": "yogyakarta",
    "jogja": "yogyakarta",
    "chiang-mai": "chiang mai",
    "koh samui": "ko samui",
    "phuket island": "phuket",
    "borobudur temple": "borobudur",
    "prambanan temple": "prambanan",
    "komodo national park": "komodo",
    "mt. bromo": "mount bromo",
    "mt bromo": "mount bromo",
    "rinjani": "mount rinjani",
    "n seoul tower": "namsan tower",
    "namsan tower": "namsan tower",
    "seoul tower": "namsan tower",
    "gyeongbok palace": "gyeongbokgung palace",
    "gyeongbokgung": "gyeongbokgung palace",
}


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation and extra spaces."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class ConceptNormalizer:
    """
    Canonicalizes a set of OntologyConcept objects:
    - resolves aliases to canonical names
    - merges near-duplicates (fuzzy match)
    - keeps highest-confidence version
    - accumulates all aliases onto the canonical concept
    """

    def __init__(
        self,
        fuzzy_threshold: int = 88,  # 0–100; higher = stricter
        use_fuzzy: bool = True,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy and _HAS_RAPIDFUZZ

    def normalize(
        self, concepts: list[OntologyConcept]
    ) -> tuple[list[OntologyConcept], dict[str, Any]]:
        """
        Normalize a list of concepts.

        Returns:
            (canonical_concepts, normalization_report)
        """
        # Step 1: apply alias table
        resolved = [self._resolve_alias(c) for c in concepts]

        # Step 2: group by normalized name × class
        groups: dict[str, list[OntologyConcept]] = defaultdict(list)
        for c in resolved:
            key = f"{_normalize_text(c.name)}::{c.concept_class}"
            groups[key].append(c)

        # Step 3: fuzzy merge within same class
        if self.use_fuzzy:
            groups = self._fuzzy_merge(groups)

        # Step 4: pick canonical representative per group
        canonical: list[OntologyConcept] = []
        merge_count = 0
        for members in groups.values():
            if len(members) == 1:
                canonical.append(members[0])
            else:
                best = max(members, key=lambda c: c.provenance.confidence)
                # Merge all aliases into best
                all_aliases: set[str] = set(best.aliases)
                for m in members:
                    if m is not best:
                        all_aliases.add(m.name)
                        all_aliases.update(m.aliases)
                        merge_count += 1
                all_aliases.discard(best.name)
                best.aliases = sorted(all_aliases)
                canonical.append(best)

        report = {
            "input_count": len(concepts),
            "output_count": len(canonical),
            "merged_count": merge_count,
            "alias_resolved_count": sum(
                1 for c in resolved if _normalize_text(c.name) != _normalize_text(
                    next((o for o in concepts if o.concept_class == c.concept_class and
                          _normalize_text(o.name) != _normalize_text(c.name)), c).name
                )
            ),
        }
        logger.info(
            f"Normalization: {len(concepts)} → {len(canonical)} concepts "
            f"({merge_count} merged)."
        )
        return canonical, report

    def _resolve_alias(self, concept: OntologyConcept) -> OntologyConcept:
        """Map known aliases to canonical names."""
        key = _normalize_text(concept.name)
        if key in KNOWN_ALIASES:
            concept.aliases.append(concept.name)
            concept.name = KNOWN_ALIASES[key].title()
        return concept

    def _fuzzy_merge(
        self, groups: dict[str, list[OntologyConcept]]
    ) -> dict[str, list[OntologyConcept]]:
        """Merge groups whose names are fuzzy-similar within the same class."""
        if not _HAS_RAPIDFUZZ:
            return groups

        keys = list(groups.keys())
        merged_into: dict[str, str] = {}  # key → canonical key

        for i, k1 in enumerate(keys):
            if k1 in merged_into:
                continue
            name1, class1 = k1.split("::", 1)
            for k2 in keys[i + 1 :]:
                if k2 in merged_into:
                    continue
                name2, class2 = k2.split("::", 1)
                if class1 != class2:
                    continue
                score = fuzz.token_sort_ratio(name1, name2)
                if score >= self.fuzzy_threshold:
                    # Merge k2 into k1
                    groups[k1].extend(groups[k2])
                    merged_into[k2] = k1

        # Remove merged keys
        for k in merged_into:
            groups.pop(k, None)

        return groups
