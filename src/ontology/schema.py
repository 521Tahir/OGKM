"""
Tourism Ontology Schema — v0.1.0

Defines the formal class hierarchy, properties, relations, controlled
vocabularies, and provenance structures for the tourism domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ── Controlled Vocabularies ─────────────────────────────────────────────────

class ConceptClass(str, Enum):
    """Ontology class hierarchy for tourism domain."""
    # Top-level
    DESTINATION = "Destination"
    ATTRACTION = "Attraction"
    ACCOMMODATION = "Accommodation"
    RESTAURANT = "Restaurant"
    TRANSPORT_MODE = "TransportMode"
    ACTIVITY = "Activity"
    ITINERARY = "Itinerary"
    TRAVELER_TYPE = "TravelerType"
    SEASON = "Season"
    WEATHER_CONDITION = "WeatherCondition"
    SAFETY_ADVICE = "SafetyAdvice"
    EMERGENCY_SERVICE = "EmergencyService"
    LOCAL_CUSTOM = "LocalCustom"
    TRAVEL_CONSTRAINT = "TravelConstraint"
    CUISINE = "Cuisine"
    ROUTE = "Route"
    DIETARY_REQUIREMENT = "DietaryRequirement"

    # Destination subclasses
    REGION = "Region"
    COUNTRY = "Country"
    CITY = "City"
    ISLAND = "Island"

    # Attraction subclasses
    NATURAL_ATTRACTION = "NaturalAttraction"
    CULTURAL_ATTRACTION = "CulturalAttraction"
    RELIGIOUS_SITE = "ReligiousSite"
    MUSEUM = "Museum"
    BEACH = "Beach"
    MOUNTAIN = "Mountain"
    TRAIL = "Trail"
    PARK = "Park"
    MARKET = "Market"
    LANDMARK = "Landmark"


# Class hierarchy: child → parent
CLASS_HIERARCHY: dict[str, str] = {
    "Region": "Destination",
    "Country": "Destination",
    "City": "Destination",
    "Island": "Destination",
    "NaturalAttraction": "Attraction",
    "CulturalAttraction": "Attraction",
    "ReligiousSite": "CulturalAttraction",
    "Museum": "CulturalAttraction",
    "Beach": "NaturalAttraction",
    "Mountain": "NaturalAttraction",
    "Trail": "NaturalAttraction",
    "Park": "Attraction",
    "Market": "Attraction",
    "Landmark": "Attraction",
}

# Allowed relation types
RELATION_TYPES: set[str] = {
    "located_in",
    "part_of",
    "near",
    "best_for",
    "suitable_for",
    "requires",
    "avoid_during",
    "available_in",
    "served_at",
    "reachable_by",
    "connected_to",
    "has_activity",
    "has_risk",
    "has_seasonality",
    "has_budget_level",
    "supports_dietary_need",
    "has_accessibility_feature",
    "recommended_duration",
    "has_opening_condition",
    "needs_reservation",
    "depends_on_weather",
    "supported_by_evidence",
    "is_a",
    "instance_of",
    "subclass_of",   # rdfs:subClassOf — class→parent subsumption (taxonomy backbone)
}

# Meta-class for ontology class nodes themselves (owl:Class), used when the
# class taxonomy is materialised as concept nodes for graph connectivity.
META_CLASS = "OntologyClass"

# Domain/range constraints: relation → (allowed_subject_classes, allowed_object_classes)
RELATION_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "located_in": ({"Attraction", "City", "Island", "Accommodation", "Restaurant"}, {"Destination", "Region", "Country", "City", "Island"}),
    "part_of": ({"City", "Island", "Region"}, {"Country", "Region", "Destination"}),
    "reachable_by": ({"Destination", "Attraction"}, {"TransportMode", "Route"}),
    "served_at": ({"Cuisine"}, {"Restaurant"}),
    "has_activity": ({"Destination", "Attraction"}, {"Activity"}),
    "suitable_for": ({"Attraction", "Activity", "Destination"}, {"TravelerType"}),
    "has_risk": ({"Destination", "Activity", "Attraction"}, {"SafetyAdvice"}),
    "avoid_during": ({"Destination", "Attraction", "Activity"}, {"Season", "WeatherCondition"}),
    "supports_dietary_need": ({"Restaurant", "Cuisine"}, {"DietaryRequirement"}),
}

BUDGET_LEVELS = {"budget", "mid-range", "luxury", "backpacker"}
TRAVELER_TYPES = {"family", "solo", "couple", "group", "senior", "budget_traveler", "luxury_traveler", "adventure_seeker", "cultural_tourist"}
DIETARY_REQUIREMENTS = {"halal", "kosher", "vegan", "vegetarian", "gluten_free", "nut_free"}
TRANSPORT_MODES = {"bus", "train", "metro", "taxi", "car_rental", "scooter", "ferry", "domestic_flight", "tuk_tuk", "bicycle", "cable_car"}


# ── Provenance ───────────────────────────────────────────────────────────────

@dataclass
class ProvenanceMetadata:
    """Tracks the origin and trustworthiness of an ontology element."""
    source_record_id: Optional[int] = None
    extraction_prompt_version: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    confidence: float = 1.0       # 0.0–1.0
    validator_status: str = "pending"  # pending | pass | warning | fail | expert_review
    approved_by: Optional[str] = None  # agent or human reviewer id
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "source_record_id": self.source_record_id,
            "extraction_prompt_version": self.extraction_prompt_version,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "validator_status": self.validator_status,
            "approved_by": self.approved_by,
            "notes": self.notes,
        }


# ── Core Ontology Elements ────────────────────────────────────────────────────

@dataclass
class OntologyConcept:
    """A concept (class instance) in the tourism ontology."""
    name: str                          # canonical label, e.g. "Ubud"
    concept_class: str                 # e.g. "City"
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "concept_class": self.concept_class,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OntologyConcept":
        prov = ProvenanceMetadata(**d.get("provenance", {}))
        return cls(
            name=d["name"],
            concept_class=d["concept_class"],
            aliases=d.get("aliases", []),
            attributes=d.get("attributes", {}),
            provenance=prov,
        )


@dataclass
class OntologyRelation:
    """A typed relation between two concepts."""
    subject: str       # concept name
    predicate: str     # relation type from RELATION_TYPES
    obj: str           # concept name
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OntologyRelation":
        prov = ProvenanceMetadata(**d.get("provenance", {}))
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            obj=d["object"],
            provenance=prov,
        )


@dataclass
class OntologyRule:
    """A constraint or inference rule in the tourism ontology."""
    rule_id: str
    description: str
    condition: str       # natural language or SPARQL-like condition
    consequence: str     # what happens when condition is met
    rule_type: str = "soft"  # "soft" | "hard"
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "condition": self.condition,
            "consequence": self.consequence,
            "rule_type": self.rule_type,
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class OntologyCandidate:
    """
    A raw extraction candidate produced by the LLM before validation.
    Holds lists of concepts, relations, and rules extracted from a single record.
    """
    record_id: int
    concepts: list[OntologyConcept] = field(default_factory=list)
    relations: list[OntologyRelation] = field(default_factory=list)
    rules: list[OntologyRule] = field(default_factory=list)
    extraction_prompt: str = ""
    raw_llm_output: str = ""
    parse_success: bool = True
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "concepts": [c.to_dict() for c in self.concepts],
            "relations": [r.to_dict() for r in self.relations],
            "rules": [r.to_dict() for r in self.rules],
            "extraction_prompt": self.extraction_prompt,
            "raw_llm_output": self.raw_llm_output,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }


# ── Schema Container ─────────────────────────────────────────────────────────

class TourismOntologySchema:
    """
    Container for the full tourism ontology including class hierarchy,
    relation types, constraints, and controlled vocabularies.
    """
    version: str = "0.1.0"
    namespace: str = "http://tourism.ontology.org/"

    class_hierarchy = CLASS_HIERARCHY
    relation_types = RELATION_TYPES
    relation_constraints = RELATION_CONSTRAINTS
    budget_levels = BUDGET_LEVELS
    traveler_types = TRAVELER_TYPES
    dietary_requirements = DIETARY_REQUIREMENTS
    transport_modes = TRANSPORT_MODES
    concept_classes = {c.value for c in ConceptClass}

    @classmethod
    def is_valid_class(cls, class_name: str) -> bool:
        return class_name in cls.concept_classes or class_name == META_CLASS

    @classmethod
    def is_valid_relation(cls, relation: str) -> bool:
        return relation in cls.relation_types

    @classmethod
    def get_parent_class(cls, class_name: str) -> Optional[str]:
        return cls.class_hierarchy.get(class_name)

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "version": cls.version,
            "namespace": cls.namespace,
            "concept_classes": sorted(cls.concept_classes),
            "class_hierarchy": cls.class_hierarchy,
            "relation_types": sorted(cls.relation_types),
            "relation_constraints": {
                k: {"subject_classes": sorted(v[0]), "object_classes": sorted(v[1])}
                for k, v in cls.relation_constraints.items()
            },
            "controlled_vocabularies": {
                "budget_levels": sorted(cls.budget_levels),
                "traveler_types": sorted(cls.traveler_types),
                "dietary_requirements": sorted(cls.dietary_requirements),
                "transport_modes": sorted(cls.transport_modes),
            },
        }
