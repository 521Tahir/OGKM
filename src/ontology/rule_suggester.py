"""
Rule suggestion module.

Infers candidate constraint/inference rules from:
1. Pattern analysis over extracted concepts and relations (offline)
2. LLM-based rule suggestion (when client is provided)

The offline engine uses frequency analysis to surface high-signal patterns.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import OntologyRule, OntologyConcept, OntologyRelation, ProvenanceMetadata
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.rule_suggester")

# Predefined seed rules grounded in tourism domain knowledge
SEED_RULES: list[dict] = [
    {
        "rule_id": "R001",
        "description": "Coastal attraction + severe storm → suppress swimming",
        "condition": "attraction.type IN [Beach, NaturalAttraction] AND weather = severe_storm",
        "consequence": "SUPPRESS swimming_related_activities",
        "rule_type": "hard",
    },
    {
        "rule_id": "R002",
        "description": "Halal/kosher/vegan traveler → filter dining to matching DietaryRequirement",
        "condition": "traveler.dietary IN [halal, kosher, vegan, vegetarian]",
        "consequence": "FILTER restaurants WHERE supports_dietary_need MATCHES traveler.dietary",
        "rule_type": "hard",
    },
    {
        "rule_id": "R003",
        "description": "Family travel → remove high-risk activities unless explicit",
        "condition": "traveler.type = family AND activity.has_risk.severity = high",
        "consequence": "SUPPRESS activity UNLESS explicitly_requested",
        "rule_type": "soft",
    },
    {
        "rule_id": "R004",
        "description": "Low confidence concept → answer cautiously",
        "condition": "concept.confidence < 0.5",
        "consequence": "ADD uncertainty_qualifier TO response",
        "rule_type": "soft",
    },
    {
        "rule_id": "R005",
        "description": "No ontology evidence → do not fabricate",
        "condition": "evidence_sources IS EMPTY",
        "consequence": "RETURN 'No verified information available for this claim'",
        "rule_type": "hard",
    },
    {
        "rule_id": "R006",
        "description": "Monsoon season + coastal island → warn about weather-dependent activities",
        "condition": "destination.has_seasonality = monsoon AND destination.type IN [Island, Beach]",
        "consequence": "ADD weather_warning FOR water_activities",
        "rule_type": "soft",
    },
    {
        "rule_id": "R007",
        "description": "Critical safety risk → always include warning",
        "condition": "attraction.has_risk.severity = critical",
        "consequence": "ALWAYS_INCLUDE safety_warning IN response",
        "rule_type": "hard",
    },
    {
        "rule_id": "R008",
        "description": "No restaurant has required dietary support → suggest alternatives",
        "condition": "restaurant.supports_dietary_need NOT CONTAINS traveler.dietary",
        "consequence": "SUGGEST alternative_area OR nearby_halal_market",
        "rule_type": "soft",
    },
]


class RuleSuggester:
    """
    Suggests constraint rules from extracted ontology data.

    Offline mode: returns seed rules + data-driven pattern rules.
    LLM mode: additionally queries LLM for domain-specific rule candidates.
    """

    PROMPT_VERSION = "v1.0"

    def __init__(
        self,
        client=None,
        prompt_manager=None,
        strategy: str = "cot",
        min_pattern_frequency: int = 5,
    ):
        self.client = client
        self.pm = prompt_manager
        self.strategy = strategy
        self.min_pattern_frequency = min_pattern_frequency

    def suggest(
        self,
        concepts: list[OntologyConcept],
        relations: list[OntologyRelation],
    ) -> list[OntologyRule]:
        """
        Generate rule candidates.

        Returns list of OntologyRule objects (seed + pattern-mined).
        """
        rules: list[OntologyRule] = []

        # Always include seed rules
        rules.extend(self._load_seed_rules())

        # Mine additional rules from data patterns
        mined = self._mine_patterns(concepts, relations)
        rules.extend(mined)

        logger.info(f"RuleSuggester: {len(rules)} total rules ({len(mined)} mined).")
        return rules

    def _load_seed_rules(self) -> list[OntologyRule]:
        return [
            OntologyRule(
                rule_id=r["rule_id"],
                description=r["description"],
                condition=r["condition"],
                consequence=r["consequence"],
                rule_type=r["rule_type"],
                provenance=ProvenanceMetadata(
                    extraction_prompt_version="seed_v1.0",
                    confidence=1.0,
                    validator_status="pass",
                    approved_by="domain_expert",
                    notes="Seed rule from tourism domain knowledge",
                ),
            )
            for r in SEED_RULES
        ]

    def _mine_patterns(
        self,
        concepts: list[OntologyConcept],
        relations: list[OntologyRelation],
    ) -> list[OntologyRule]:
        """
        Find frequent (predicate, object_class) co-occurrence patterns
        and convert them to candidate rules.
        """
        # Build concept class lookup
        class_lookup: dict[str, str] = {
            c.name.lower(): c.concept_class for c in concepts
        }

        # Count (subject_class, predicate, object_class) patterns
        pattern_counter: Counter = Counter()
        for rel in relations:
            s_class = class_lookup.get(rel.subject.lower(), "Unknown")
            o_class = class_lookup.get(rel.obj.lower(), "Unknown")
            if s_class != "Unknown" and o_class != "Unknown":
                pattern_counter[(s_class, rel.predicate, o_class)] += 1

        mined: list[OntologyRule] = []
        rule_idx = len(SEED_RULES) + 1
        for (s_class, pred, o_class), freq in pattern_counter.most_common(20):
            if freq < self.min_pattern_frequency:
                break
            rule_id = f"R_mined_{rule_idx:03d}"
            mined.append(OntologyRule(
                rule_id=rule_id,
                description=f"Frequent pattern: {s_class} --{pred}--> {o_class} (freq={freq})",
                condition=f"entity.type = {s_class} AND {pred} EXISTS",
                consequence=f"INFER {o_class} context; apply {pred} constraints",
                rule_type="soft",
                provenance=ProvenanceMetadata(
                    extraction_prompt_version="pattern_mining_v1.0",
                    confidence=min(0.5 + freq / 100, 0.95),
                    validator_status="pending",
                    notes=f"Mined from {freq} occurrences",
                ),
            ))
            rule_idx += 1

        return mined
