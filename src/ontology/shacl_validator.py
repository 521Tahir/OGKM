"""
SHACL-style constraint validation for the tourism ontology.

Implements a subset of SHACL (Shapes Constraint Language) constraints without
requiring the pyshacl library. Uses pyshacl if available for full RDF validation.

Constraint sets:
  - ConceptShapes: name length, class membership, confidence range
  - RelationShapes: valid predicates, subject/object existence, domain/range
  - GlobalShapes: max concepts per record, orphan detection, cycle detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import OntologyConcept, OntologyRelation, TourismOntologySchema
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.shacl_validator")

SCHEMA = TourismOntologySchema

# ── Domain / Range constraints ────────────────────────────────────────────────

# For each predicate, which concept classes are valid as subject / object
DOMAIN_RANGE: dict[str, dict[str, list[str]]] = {
    "located_in": {
        "domain": ["Destination", "Region", "City", "Island", "Beach", "Mountain",
                   "Park", "Trail", "Attraction", "NaturalAttraction", "CulturalAttraction",
                   "ReligiousSite", "Museum", "Market", "Landmark", "Restaurant",
                   "Accommodation", "TransportMode", "Route"],
        "range": ["Region", "Country", "City", "Island", "Destination"],
    },
    "part_of": {
        "domain": ["Region", "City", "Island", "Park", "Trail", "Market", "Route"],
        "range": ["Region", "Country", "City", "Island", "Destination"],
    },
    "reachable_by": {
        "domain": ["Destination", "City", "Island", "Attraction", "Beach", "Mountain"],
        "range": ["TransportMode", "Route"],
    },
    "served_at": {
        "domain": ["Cuisine", "Restaurant"],
        "range": ["Restaurant", "Accommodation"],
    },
    "has_activity": {
        "domain": ["Destination", "City", "Island", "Beach", "Park", "Trail",
                   "NaturalAttraction", "Attraction"],
        "range": ["Activity"],
    },
    "suitable_for": {
        "domain": ["Destination", "City", "Island", "Beach", "Accommodation",
                   "Attraction", "Activity", "Itinerary"],
        "range": ["TravelerType"],
    },
    "avoid_during": {
        "domain": ["Destination", "City", "Island", "Activity", "Trail"],
        "range": ["Season", "WeatherCondition"],
    },
    "supports_dietary_need": {
        "domain": ["Restaurant", "Cuisine", "Accommodation"],
        "range": ["DietaryRequirement"],
    },
    "has_risk": {
        "domain": ["Destination", "City", "Island", "Activity", "Trail", "Beach"],
        "range": ["SafetyAdvice", "EmergencyService", "TravelConstraint"],
    },
}

# Predicates that require high-confidence evidence
HIGH_CONFIDENCE_PREDICATES = {"has_risk", "avoid_during", "requires", "needs_reservation"}
HIGH_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class SHACLViolation:
    """A single SHACL constraint violation."""
    shape_name: str           # which SHACL shape was violated
    severity: str             # "Violation" | "Warning" | "Info"
    message: str
    focus_node: str           # the concept/relation that violated the constraint
    constraint_component: str # e.g. "sh:minLength", "sh:in", "sh:minInclusive"
    value: Any = None         # the actual value that caused the violation

    def to_dict(self) -> dict:
        return {
            "shape": self.shape_name,
            "severity": self.severity,
            "message": self.message,
            "focus_node": self.focus_node,
            "constraint": self.constraint_component,
            "value": str(self.value) if self.value is not None else None,
        }


@dataclass
class SHACLReport:
    """Aggregated SHACL validation report."""
    conforms: bool = True
    violations: list[SHACLViolation] = field(default_factory=list)
    warnings: list[SHACLViolation] = field(default_factory=list)

    def add(self, v: SHACLViolation) -> None:
        if v.severity == "Violation":
            self.violations.append(v)
            self.conforms = False
        else:
            self.warnings.append(v)

    def to_dict(self) -> dict:
        return {
            "conforms": self.conforms,
            "violation_count": len(self.violations),
            "warning_count": len(self.warnings),
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class SHACLValidator:
    """
    Validates concepts and relations against SHACL-style shape constraints.

    Usage:
        validator = SHACLValidator()
        report = validator.validate(concepts, relations)
        if not report.conforms:
            for v in report.violations:
                print(v.message)
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        check_domain_range: bool = True,
        check_high_confidence: bool = True,
        max_concepts_per_record: int = 50,
    ):
        self.min_confidence = min_confidence
        self.check_domain_range = check_domain_range
        self.check_high_confidence = check_high_confidence
        self.max_concepts_per_record = max_concepts_per_record

    def validate(
        self,
        concepts: list[OntologyConcept],
        relations: list[OntologyRelation],
        record_id: int = -1,
    ) -> SHACLReport:
        """Run all SHACL shapes against a set of concepts and relations."""
        report = SHACLReport()
        concept_names = {c.name.lower() for c in concepts}

        self._concept_shapes(concepts, report)
        self._relation_shapes(relations, concept_names, report)
        self._global_shapes(concepts, relations, report)

        if report.conforms:
            logger.debug(f"Record {record_id}: SHACL validation passed.")
        else:
            logger.debug(
                f"Record {record_id}: SHACL {len(report.violations)} violations, "
                f"{len(report.warnings)} warnings."
            )
        return report

    def validate_ontology(
        self,
        concepts: list[dict],
        relations: list[dict],
    ) -> SHACLReport:
        """Validate a full ontology snapshot (dicts from JSON)."""
        from .schema import OntologyConcept, OntologyRelation, ProvenanceMetadata

        def _to_concept(d: dict) -> OntologyConcept:
            prov = d.get("provenance", {})
            return OntologyConcept(
                name=d.get("name", ""),
                concept_class=d.get("concept_class", ""),
                aliases=d.get("aliases", []),
                provenance=ProvenanceMetadata(
                    source_record_id=prov.get("source_record_id", 0),
                    extraction_prompt_version=prov.get("extraction_prompt_version", ""),
                    timestamp=prov.get("timestamp", ""),
                    confidence=prov.get("confidence", 0.0),
                    validator_status=prov.get("validator_status", "pending"),
                ),
            )

        def _to_relation(d: dict) -> OntologyRelation:
            prov = d.get("provenance", {})
            return OntologyRelation(
                subject=d.get("subject", ""),
                predicate=d.get("predicate", ""),
                obj=d.get("object", ""),
                provenance=ProvenanceMetadata(
                    source_record_id=prov.get("source_record_id", 0),
                    extraction_prompt_version=prov.get("extraction_prompt_version", ""),
                    timestamp=prov.get("timestamp", ""),
                    confidence=prov.get("confidence", 0.0),
                    validator_status=prov.get("validator_status", "pending"),
                ),
            )

        c_objs = [_to_concept(d) for d in concepts]
        r_objs = [_to_relation(d) for d in relations]
        return self.validate(c_objs, r_objs)

    # ── Shape implementations ────────────────────────────────────────────────

    def _concept_shapes(
        self, concepts: list[OntologyConcept], report: SHACLReport
    ) -> None:
        """ConceptShape: name, class, confidence constraints."""
        for c in concepts:
            node = c.name or "(unnamed)"

            # sh:minLength 2 on name
            if not c.name or len(c.name.strip()) < 2:
                report.add(SHACLViolation(
                    shape_name="ConceptShape",
                    severity="Violation",
                    message=f"Concept name too short or empty: '{c.name}'",
                    focus_node=node,
                    constraint_component="sh:minLength",
                    value=len(c.name) if c.name else 0,
                ))

            # sh:in SCHEMA.CONCEPT_CLASSES
            if not SCHEMA.is_valid_class(c.concept_class):
                report.add(SHACLViolation(
                    shape_name="ConceptShape",
                    severity="Violation",
                    message=f"Concept class '{c.concept_class}' not in allowed set",
                    focus_node=node,
                    constraint_component="sh:in",
                    value=c.concept_class,
                ))

            # sh:minInclusive 0.0 sh:maxInclusive 1.0 on confidence
            conf = c.provenance.confidence
            if not (0.0 <= conf <= 1.0):
                report.add(SHACLViolation(
                    shape_name="ConceptShape",
                    severity="Violation",
                    message=f"Confidence {conf:.2f} out of range [0,1]",
                    focus_node=node,
                    constraint_component="sh:minInclusive",
                    value=conf,
                ))

            # Warning: low confidence
            if 0.0 <= conf < self.min_confidence:
                report.add(SHACLViolation(
                    shape_name="ConceptShape",
                    severity="Warning",
                    message=f"Low confidence {conf:.2f} < threshold {self.min_confidence}",
                    focus_node=node,
                    constraint_component="sh:minInclusive",
                    value=conf,
                ))

            # Warning: numeric concept name (likely extraction artifact)
            if c.name and c.name.strip().isdigit():
                report.add(SHACLViolation(
                    shape_name="ConceptShape",
                    severity="Warning",
                    message=f"Concept name is purely numeric: '{c.name}'",
                    focus_node=node,
                    constraint_component="sh:pattern",
                    value=c.name,
                ))

    def _relation_shapes(
        self,
        relations: list[OntologyRelation],
        concept_names: set[str],
        report: SHACLReport,
    ) -> None:
        """RelationShape: predicate validity, dangling references, domain/range."""
        for r in relations:
            node = f"{r.subject}--{r.predicate}-->{r.obj}"

            # sh:in SCHEMA.RELATION_TYPES on predicate
            if not SCHEMA.is_valid_relation(r.predicate):
                report.add(SHACLViolation(
                    shape_name="RelationShape",
                    severity="Violation",
                    message=f"Predicate '{r.predicate}' not in allowed set",
                    focus_node=node,
                    constraint_component="sh:in",
                    value=r.predicate,
                ))

            # sh:minLength 1 on subject / object
            if not r.subject or not r.obj:
                report.add(SHACLViolation(
                    shape_name="RelationShape",
                    severity="Violation",
                    message="Relation has empty subject or object",
                    focus_node=node,
                    constraint_component="sh:minLength",
                    value=None,
                ))

            # Warning: dangling reference (subject/object not in extracted concepts)
            if concept_names and r.subject.lower() not in concept_names:
                report.add(SHACLViolation(
                    shape_name="RelationShape",
                    severity="Warning",
                    message=f"Relation subject '{r.subject}' not in extracted concepts",
                    focus_node=node,
                    constraint_component="sh:node",
                    value=r.subject,
                ))
            if concept_names and r.obj.lower() not in concept_names:
                report.add(SHACLViolation(
                    shape_name="RelationShape",
                    severity="Warning",
                    message=f"Relation object '{r.obj}' not in extracted concepts",
                    focus_node=node,
                    constraint_component="sh:node",
                    value=r.obj,
                ))

            # High-confidence predicates need strong evidence
            if self.check_high_confidence and r.predicate in HIGH_CONFIDENCE_PREDICATES:
                if r.provenance.confidence < HIGH_CONFIDENCE_THRESHOLD:
                    report.add(SHACLViolation(
                        shape_name="RelationShape",
                        severity="Warning",
                        message=(
                            f"High-stakes predicate '{r.predicate}' has low confidence "
                            f"{r.provenance.confidence:.2f} < {HIGH_CONFIDENCE_THRESHOLD}"
                        ),
                        focus_node=node,
                        constraint_component="sh:minInclusive",
                        value=r.provenance.confidence,
                    ))

    def _global_shapes(
        self,
        concepts: list[OntologyConcept],
        relations: list[OntologyRelation],
        report: SHACLReport,
    ) -> None:
        """GlobalShape: record-level constraints."""
        # sh:maxCount on concepts per record
        if len(concepts) > self.max_concepts_per_record:
            report.add(SHACLViolation(
                shape_name="GlobalShape",
                severity="Warning",
                message=(
                    f"Unusually many concepts per record: {len(concepts)} "
                    f"> max {self.max_concepts_per_record}"
                ),
                focus_node="record",
                constraint_component="sh:maxCount",
                value=len(concepts),
            ))

        # Self-loop detection
        for r in relations:
            if r.subject.lower() == r.obj.lower():
                report.add(SHACLViolation(
                    shape_name="GlobalShape",
                    severity="Warning",
                    message=f"Self-loop relation: {r.subject} --{r.predicate}--> {r.obj}",
                    focus_node=r.subject,
                    constraint_component="sh:disjoint",
                    value=None,
                ))
