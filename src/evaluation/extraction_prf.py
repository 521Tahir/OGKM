"""
Extraction Precision / Recall / F1 — real scoring for Phase 10·A.

This module computes the *ground-truth* concept and relation quality metrics
required by the CLAUDE_PROMPT (Phase 10·A "Ontology quality metrics"):

  - concept precision / recall / F1
  - relation precision / recall / F1

It replaces the previously hardcoded placeholder values in
``pdf_aligned_metrics_report.json`` with numbers computed directly from data.

Provenance model
----------------
Two complementary annotation sources feed the metrics:

  * **Precision** is computed from a *sampled-extraction* annotation set
    (``manual_annotation_sample.json``) in which each extracted concept/relation
    carries a ``correct: true|false|null`` judgement.  Precision = TP / (TP+FP)
    over the *labelled* subset.  Human (``gold``) labels always take priority;
    where a sample is unlabelled (``correct: null``) the scorer can optionally
    fall back to a documented heuristic ``silver`` label.  The provenance of
    every precision number is reported explicitly.

  * **Recall** is computed *directly and without human labels* from a per-query
    gold set (``gold_annotations_100.json``) carrying ``expected_concepts``:
    recall = (expected concepts present in the ontology) / (all expected
    concepts).  This is a genuine measurement, not an estimate.

If a metric cannot be computed (no labels, no gold) the scorer returns ``None``
with a machine-readable reason rather than fabricating a value.

Paper table: Table 3 — Ontology Quality Metrics (Section 11.4.1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.extraction_prf")


# Concept classes considered "schema-valid" for the silver heuristic. Kept in
# sync with ontology_metrics.TARGET_SCHEMA_CLASSES plus the v1.0.0 schema labels.
SCHEMA_CLASSES = {
    "Destination", "Region", "City", "Island", "Beach", "Mountain", "Trail",
    "NaturalSite", "NaturalAttraction", "NationalPark", "Museum", "CulturalSite",
    "CulturalAttraction", "ReligiousSite", "Monument", "Heritage", "Festival",
    "Restaurant", "LocalFood", "Cuisine", "Hotel", "Accommodation", "Activity",
    "Sport", "Transportation", "TransportMode", "Route", "Attraction",
    "TouristAttraction", "Market", "TravelInfo", "Season", "WeatherCondition",
    "Budget", "TravelerType", "DietaryRequirement", "SafetyAdvice",
    "EmergencyService", "LocalCustom", "TravelConstraint", "Itinerary",
}

# Predicate vocabulary considered valid for the silver relation heuristic.
KNOWN_PREDICATES = {
    "located_in", "part_of", "near", "best_for", "suitable_for", "requires",
    "avoid_during", "available_in", "served_at", "reachable_by", "connected_to",
    "has_activity", "has_risk", "has_seasonality", "has_budget_level",
    "supports_dietary_need", "has_accessibility_feature", "recommended_duration",
    "has_opening_condition", "needs_reservation", "depends_on_weather",
    "supported_by_evidence", "is_a", "subclass_of", "instance_of",
}

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_term(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for matching."""
    if not text:
        return ""
    t = _PUNCT_RE.sub(" ", text.lower())
    t = _WS_RE.sub(" ", t).strip()
    return t


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class PRFScore:
    """A single precision/recall/F1 result with provenance."""
    precision:        Optional[float] = None
    recall:           Optional[float] = None
    f1:               Optional[float] = None
    # Precision support
    n_labeled:        int = 0          # annotations with a non-null judgement
    n_total:          int = 0          # total annotations sampled
    true_positives:   int = 0
    false_positives:  int = 0
    precision_source: str = "none"     # gold | silver | mixed | none
    # Recall support
    n_expected:       int = 0          # distinct gold expected items
    n_matched:        int = 0          # expected items found in ontology
    recall_source:    str = "none"     # gold | none
    # Notes / reasons when a metric is missing
    notes:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "precision":        round(self.precision, 4) if self.precision is not None else None,
            "recall":           round(self.recall, 4) if self.recall is not None else None,
            "f1":               round(self.f1, 4) if self.f1 is not None else None,
            "n_labeled":        self.n_labeled,
            "n_total":          self.n_total,
            "true_positives":   self.true_positives,
            "false_positives":  self.false_positives,
            "precision_source": self.precision_source,
            "n_expected":       self.n_expected,
            "n_matched":        self.n_matched,
            "recall_source":    self.recall_source,
            "notes":            self.notes,
        }


def f1_score(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """Harmonic mean of precision and recall; None if either is missing."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Scorer ────────────────────────────────────────────────────────────────────

class ExtractionPRFScorer:
    """
    Compute concept/relation precision, recall, and F1 for an ontology version.

    Parameters
    ----------
    concepts : list[dict]
        Ontology concept dicts (``name``, ``concept_class``, ``aliases``).
    relations : list[dict]
        Ontology relation dicts (``subject``/``head``, ``predicate``,
        ``object``/``tail``).
    concept_annotations : list[dict]
        Sampled extracted concepts with ``correct`` judgement (for precision).
    relation_annotations : list[dict]
        Sampled extracted relations with ``correct`` judgement (for precision).
    gold_annotations : list[dict]
        Per-query gold records carrying ``expected_concepts`` (for recall).
    soft_match : bool
        If True, recall also counts token-subset matches (e.g. "beach" matches
        "black sand beach"); strict normalized/alias matching otherwise.
    """

    def __init__(
        self,
        concepts: list[dict],
        relations: list[dict],
        concept_annotations: Optional[list[dict]] = None,
        relation_annotations: Optional[list[dict]] = None,
        gold_annotations: Optional[list[dict]] = None,
        soft_match: bool = True,
    ):
        self.concepts = concepts or []
        self.relations = relations or []
        self.concept_annotations = concept_annotations or []
        self.relation_annotations = relation_annotations or []
        self.gold_annotations = gold_annotations or []
        self.soft_match = soft_match

        # Build normalized ontology term lookup (names + aliases).
        self._terms: set[str] = set()
        self._term_tokens: list[set[str]] = []
        for c in self.concepts:
            for raw in [c.get("name", "")] + list(c.get("aliases", []) or []):
                norm = normalize_term(raw)
                if norm:
                    self._terms.add(norm)
        # token sets are only needed for soft matching — build lazily-ish here
        if self.soft_match:
            self._term_tokens = [set(t.split()) for t in self._terms]

    # ── Matching ───────────────────────────────────────────────────────────────

    def _matches_ontology(self, expected: str) -> bool:
        """True if an expected concept term is represented in the ontology."""
        norm = normalize_term(expected)
        if not norm:
            return False
        if norm in self._terms:
            return True
        if not self.soft_match:
            return False
        # Soft match: a multi-word term on one side is fully contained in the
        # term on the other side, e.g. expected "luxury beach resort" is covered
        # by ontology "beach resort", and expected "beach resort" is covered by
        # ontology "black sand beach resort". The *subset* side must have ≥2
        # tokens: allowing a single shared common word (e.g. a 1-token alias
        # "beach") to match explodes recall toward 1.0 given the large alias
        # vocabulary, so it is excluded.
        exp_tokens = set(norm.split())
        if not exp_tokens:
            return False
        for tokens in self._term_tokens:
            if len(exp_tokens) >= 2 and exp_tokens <= tokens:
                return True
            if len(tokens) >= 2 and tokens <= exp_tokens:
                return True
        return False

    # ── Precision ────────────────────────────────────────────────────────────────

    @staticmethod
    def _precision_from_annotations(
        annotations: list[dict],
        use_silver: bool,
        silver_labeler,
    ) -> tuple[Optional[float], int, int, int, int, str, list[str]]:
        """
        Compute precision from a list of annotation dicts.

        Returns (precision, n_labeled, n_total, tp, fp, source, notes).
        Human labels (``correct`` already set) take priority; silver labels are
        applied only to unlabelled items when ``use_silver`` is True.
        """
        notes: list[str] = []
        n_total = len(annotations)
        tp = fp = 0
        n_human = n_silver = 0

        for ann in annotations:
            verdict = ann.get("correct", None)
            source_is_silver = False
            if verdict is None and use_silver:
                verdict = silver_labeler(ann)
                source_is_silver = True
            if verdict is None:
                continue  # still unlabelled — excluded from precision
            if verdict:
                tp += 1
            else:
                fp += 1
            if source_is_silver:
                n_silver += 1
            else:
                n_human += 1

        n_labeled = tp + fp
        if n_labeled == 0:
            notes.append(
                "precision unavailable: no human or silver labels present "
                "(all annotations have correct=null)"
            )
            return None, 0, n_total, 0, 0, "none", notes

        precision = tp / n_labeled
        if n_human and n_silver:
            source = "mixed"
        elif n_silver:
            source = "silver"
        else:
            source = "gold"
        if n_silver:
            notes.append(
                f"{n_silver} silver (heuristic) labels used; "
                f"{n_human} human labels used"
            )
        return precision, n_labeled, n_total, tp, fp, source, notes

    @staticmethod
    def silver_label_concept(ann: dict) -> bool:
        """
        Documented heuristic for whether an extracted concept is 'correct'.

        A concept is judged correct when it is a plausible canonical term:
          - its class is a recognised schema class, AND
          - its name is 1–4 words, AND
          - it is not an article-prefixed noun phrase ("A budget-friendly stay").

        This is a transparent stand-in for human judgement, never a substitute
        for it; labels produced here are tagged ``silver`` in all reports.
        """
        name = (ann.get("name") or "").strip()
        cls = ann.get("concept_class") or ann.get("class") or ""
        if not name:
            return False
        if _ARTICLE_RE.match(name):
            return False
        word_count = len(name.split())
        if word_count == 0 or word_count > 4:
            return False
        if cls not in SCHEMA_CLASSES:
            return False
        return True

    def silver_label_relation(self, ann: dict) -> bool:
        """
        Documented heuristic for whether an extracted relation triple is 'correct'.

        A triple is judged correct when its predicate is in the known vocabulary
        and both endpoints resolve to ontology concepts.  Tagged ``silver``.
        """
        pred = ann.get("predicate", "")
        subj = ann.get("subject", ann.get("head", ""))
        obj = ann.get("object", ann.get("tail", ""))
        if pred not in KNOWN_PREDICATES:
            return False
        return self._matches_ontology(subj) and self._matches_ontology(obj)

    # ── Recall ─────────────────────────────────────────────────────────────────

    def concept_recall(self) -> tuple[Optional[float], int, int, list[str]]:
        """
        Recall = distinct expected concepts present in the ontology / all
        distinct expected concepts, computed from the gold per-query set.

        Returns (recall, n_matched, n_expected, notes).
        """
        notes: list[str] = []
        if not self.gold_annotations:
            notes.append("recall unavailable: no gold annotations with expected_concepts")
            return None, 0, 0, notes

        expected: set[str] = set()
        for g in self.gold_annotations:
            for ec in g.get("expected_concepts", []) or []:
                norm = normalize_term(ec)
                if norm:
                    expected.add(norm)

        if not expected:
            notes.append("recall unavailable: gold annotations carry no expected_concepts")
            return None, 0, 0, notes

        matched = sum(1 for e in expected if self._matches_ontology(e))
        recall = matched / len(expected)
        notes.append(
            f"recall over {len(expected)} distinct expected concepts "
            f"({'soft' if self.soft_match else 'strict'} matching)"
        )
        return recall, matched, len(expected), notes

    # ── Top-level computation ─────────────────────────────────────────────────────

    def compute(self, use_silver: bool = False) -> dict:
        """
        Compute concept and relation P/R/F1.

        Parameters
        ----------
        use_silver : bool
            Apply the heuristic silver labeler to unlabelled precision samples.
            Human labels are always preferred where present.

        Returns
        -------
        dict with ``concept`` and ``relation`` PRFScore dicts plus a summary.
        """
        # Concept precision
        (c_p, c_lab, c_tot, c_tp, c_fp, c_src, c_notes) = self._precision_from_annotations(
            self.concept_annotations, use_silver, self.silver_label_concept
        )
        # Concept recall
        c_recall, c_matched, c_expected, c_rnotes = self.concept_recall()
        concept = PRFScore(
            precision=c_p, recall=c_recall, f1=f1_score(c_p, c_recall),
            n_labeled=c_lab, n_total=c_tot, true_positives=c_tp,
            false_positives=c_fp, precision_source=c_src,
            n_expected=c_expected, n_matched=c_matched,
            recall_source="gold" if c_recall is not None else "none",
            notes=c_notes + c_rnotes,
        )

        # Relation precision
        (r_p, r_lab, r_tot, r_tp, r_fp, r_src, r_notes) = self._precision_from_annotations(
            self.relation_annotations, use_silver, self.silver_label_relation
        )
        # Relation recall — requires a gold relation set, which is not provided.
        r_recall_notes = [
            "relation recall unavailable: no gold relation set (gold annotations "
            "carry expected_concepts only). Add expected_relations to enable."
        ]
        relation = PRFScore(
            precision=r_p, recall=None, f1=None,
            n_labeled=r_lab, n_total=r_tot, true_positives=r_tp,
            false_positives=r_fp, precision_source=r_src,
            recall_source="none",
            notes=r_notes + r_recall_notes,
        )

        logger.info(
            "Concept P/R/F1 = %s / %s / %s  (prec source=%s, %d/%d labeled; "
            "recall %d/%d matched)",
            _fmt(concept.precision), _fmt(concept.recall), _fmt(concept.f1),
            concept.precision_source, concept.n_labeled, concept.n_total,
            concept.n_matched, concept.n_expected,
        )
        logger.info(
            "Relation P/R/F1 = %s / %s / %s  (prec source=%s, %d/%d labeled)",
            _fmt(relation.precision), _fmt(relation.recall), _fmt(relation.f1),
            relation.precision_source, relation.n_labeled, relation.n_total,
        )

        return {
            "concept":  concept.to_dict(),
            "relation": relation.to_dict(),
            "use_silver": use_silver,
            "soft_match": self.soft_match,
        }


def _fmt(x: Optional[float]) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
