"""
Offline (no-LLM) concept and relation extractor.

Uses spaCy NER + regex patterns + vocabulary lookups to extract
tourism entities deterministically. Activated when no LLM API key
is configured. Produces lower-precision but immediately runnable results.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from .schema import (
    OntologyConcept,
    OntologyRelation,
    ProvenanceMetadata,
    TourismOntologySchema,
)
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.offline_extractor")
SCHEMA = TourismOntologySchema

# ── Vocabulary lookups ────────────────────────────────────────────────────────
CITY_VOCAB: dict[str, str] = {
    "jeju": "Island", "bali": "Island", "lombok": "Island",
    "seoul": "City", "busan": "City", "bangkok": "City",
    "chiang mai": "City", "phuket": "Island", "jakarta": "City",
    "yogyakarta": "City", "bandung": "City", "surabaya": "City",
    "ubud": "City", "seminyak": "City", "canggu": "City",
    "hanoi": "City", "ho chi minh": "City", "singapore": "City",
    "kuala lumpur": "City", "tokyo": "City", "kyoto": "City",
    "osaka": "City", "manila": "City", "cebu": "City",
    "labuan bajo": "City", "flores": "Island", "komodo": "Island",
    "nusa penida": "Island", "gili islands": "Island",
}
COUNTRY_VOCAB: dict[str, str] = {
    "indonesia": "Country", "south korea": "Country", "korea": "Country",
    "thailand": "Country", "vietnam": "Country", "malaysia": "Country",
    "philippines": "Country", "japan": "Country", "cambodia": "Country",
    "singapore": "Country", "india": "Country", "china": "Country",
    "myanmar": "Country", "laos": "Country", "nepal": "Country",
}
ATTRACTION_VOCAB: dict[str, str] = {
    "borobudur": "CulturalAttraction", "prambanan": "CulturalAttraction",
    "angkor wat": "ReligiousSite", "mount bromo": "Mountain",
    "mount rinjani": "Mountain", "mount fuji": "Mountain",
    "hallasan": "Mountain", "seongsan ilchulbong": "NaturalAttraction",
    "manjanggul": "NaturalAttraction", "komodo national park": "Park",
    "tegallalang": "NaturalAttraction", "monkey forest": "Park",
    "kuta beach": "Beach", "seminyak beach": "Beach", "nusa dua": "Beach",
    "uluwatu": "CulturalAttraction", "tanah lot": "ReligiousSite",
    "gyeongbokgung": "CulturalAttraction", "bukchon": "CulturalAttraction",
    "n seoul tower": "Landmark", "namsan": "Landmark",
    "grand palace": "ReligiousSite", "wat pho": "ReligiousSite",
    "wat arun": "ReligiousSite",
}
ACTIVITY_VOCAB: set[str] = {
    "surfing", "diving", "snorkeling", "hiking", "trekking",
    "cooking class", "cycling", "kayaking", "island hopping",
    "temple visit", "cultural tour", "spa treatment",
}
TRANSPORT_VOCAB: set[str] = {
    "bus", "train", "metro", "subway", "taxi", "grab", "tuk-tuk",
    "ferry", "boat", "domestic flight", "scooter", "bicycle",
    "car rental", "cable car", "jeep",
}
ACCOMMODATION_VOCAB: set[str] = {
    "hotel", "hostel", "guesthouse", "resort", "villa", "homestay",
    "lodge", "bungalow", "airbnb", "boutique hotel",
}
CUISINE_VOCAB: set[str] = {
    "nasi goreng", "satay", "pad thai", "tom yum", "kimchi",
    "bibimbap", "pho", "banh mi", "rendang", "gado-gado",
    "sushi", "ramen", "street food", "seafood", "local cuisine",
}
DIETARY_VOCAB: set[str] = {"halal", "kosher", "vegan", "vegetarian", "gluten-free"}
TRAVELER_VOCAB: set[str] = {"family", "solo", "couple", "backpacker", "senior"}
SEASON_VOCAB: set[str] = {
    "monsoon", "dry season", "rainy season", "peak season",
    "spring", "summer", "autumn", "winter",
}


def _ts() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _prov(record_id: int, confidence: float = 0.7) -> ProvenanceMetadata:
    return ProvenanceMetadata(
        source_record_id=record_id,
        extraction_prompt_version="offline_v1.0",
        timestamp=_ts(),
        confidence=confidence,
        validator_status="pending",
    )


def _find_in_vocab(
    text: str,
    vocab: dict[str, str] | set[str],
    record_id: int,
    class_override: str | None = None,
    confidence: float = 0.75,
) -> list[OntologyConcept]:
    """Return concept objects for all vocab entries found in text."""
    lower = text.lower()
    concepts: list[OntologyConcept] = []
    items = vocab.items() if isinstance(vocab, dict) else ((v, class_override or "Activity") for v in vocab)
    for name, cclass in items:
        if name in lower:
            canonical = name.title()
            if SCHEMA.is_valid_class(cclass):
                concepts.append(OntologyConcept(
                    name=canonical,
                    concept_class=cclass,
                    provenance=_prov(record_id, confidence),
                ))
    return concepts


class OfflineExtractor:
    """
    Extract tourism concepts and relations without LLM.
    Used when ANTHROPIC_API_KEY is not set.
    """

    def extract_concepts(self, text: str, record_id: int) -> list[OntologyConcept]:
        """Return concept list using vocabulary + spaCy NER."""
        seen: dict[str, OntologyConcept] = {}

        def _add(c: OntologyConcept) -> None:
            key = c.name.lower()
            if key not in seen:
                seen[key] = c

        # Vocabulary hits
        for c in _find_in_vocab(text, CITY_VOCAB, record_id, confidence=0.80):
            _add(c)
        for c in _find_in_vocab(text, COUNTRY_VOCAB, record_id, confidence=0.80):
            _add(c)
        for c in _find_in_vocab(text, ATTRACTION_VOCAB, record_id, confidence=0.82):
            _add(c)
        for c in _find_in_vocab(text, ACTIVITY_VOCAB, record_id, "Activity", 0.70):
            _add(c)
        for c in _find_in_vocab(text, TRANSPORT_VOCAB, record_id, "TransportMode", 0.72):
            _add(c)
        for c in _find_in_vocab(text, ACCOMMODATION_VOCAB, record_id, "Accommodation", 0.70):
            _add(c)
        for c in _find_in_vocab(text, CUISINE_VOCAB, record_id, "Cuisine", 0.70):
            _add(c)
        for c in _find_in_vocab(text, DIETARY_VOCAB, record_id, "DietaryRequirement", 0.85):
            _add(c)
        for c in _find_in_vocab(text, TRAVELER_VOCAB, record_id, "TravelerType", 0.75):
            _add(c)
        for c in _find_in_vocab(text, SEASON_VOCAB, record_id, "Season", 0.72):
            _add(c)

        # spaCy NER for GPE (geo-political entities) not caught by vocab
        try:
            import spacy
            if not hasattr(self, "_nlp"):
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            doc = self._nlp(text[:1000])  # limit for speed
            for ent in doc.ents:
                if ent.label_ == "GPE" and ent.text.lower() not in seen:
                    c = OntologyConcept(
                        name=ent.text.strip(),
                        concept_class="Destination",
                        provenance=_prov(record_id, 0.65),
                    )
                    _add(c)
        except Exception:
            pass

        return list(seen.values())

    def extract_relations(
        self,
        text: str,
        concepts: list[OntologyConcept],
        record_id: int,
    ) -> list[OntologyRelation]:
        """Infer relations from co-occurrence + positional patterns."""
        lower = text.lower()
        relations: list[OntologyRelation] = []
        seen_triples: set[tuple] = set()

        concept_names = {c.name.lower(): c for c in concepts}
        class_to_names: dict[str, list[str]] = {}
        for c in concepts:
            class_to_names.setdefault(c.concept_class, []).append(c.name.lower())

        def _add_rel(s: str, p: str, o: str, conf: float = 0.65) -> None:
            triple = (s, p, o)
            if triple not in seen_triples and s != o:
                seen_triples.add(triple)
                relations.append(OntologyRelation(
                    subject=s, predicate=p, obj=o,
                    provenance=_prov(record_id, conf),
                ))

        # Rule: City/Island located_in Country
        for city_name in class_to_names.get("City", []) + class_to_names.get("Island", []):
            for country_name in class_to_names.get("Country", []):
                _add_rel(city_name.title(), "located_in", country_name.title(), 0.72)

        # Rule: Attraction located_in City/Island
        for attr_name in (
            class_to_names.get("Attraction", []) +
            class_to_names.get("NaturalAttraction", []) +
            class_to_names.get("CulturalAttraction", []) +
            class_to_names.get("Beach", []) +
            class_to_names.get("Mountain", []) +
            class_to_names.get("Museum", [])
        ):
            for city_name in class_to_names.get("City", []) + class_to_names.get("Island", []):
                if city_name in lower and attr_name in lower:
                    _add_rel(attr_name.title(), "located_in", city_name.title(), 0.65)

        # Rule: Activity available_in City
        for act_name in class_to_names.get("Activity", []):
            for city_name in class_to_names.get("City", []) + class_to_names.get("Island", []):
                _add_rel(act_name.title(), "available_in", city_name.title(), 0.60)

        # Rule: Accommodation located_in City
        for acc_name in class_to_names.get("Accommodation", []):
            for city_name in class_to_names.get("City", []) + class_to_names.get("Island", []):
                _add_rel(acc_name.title(), "located_in", city_name.title(), 0.60)

        # Rule: reachable_by for transport modes
        for dest_name in class_to_names.get("City", []) + class_to_names.get("Island", []):
            for transport in class_to_names.get("TransportMode", []):
                if transport in lower and dest_name in lower:
                    _add_rel(dest_name.title(), "reachable_by", transport.title(), 0.62)

        # Rule: dietary requirement → supports_dietary_need
        for diet in class_to_names.get("DietaryRequirement", []):
            for restaurant_kw in ["restaurant", "dining", "food", "cuisine"]:
                if restaurant_kw in lower:
                    _add_rel("Restaurant", "supports_dietary_need", diet.title(), 0.60)
                    break

        # Rule: TravelerType suitable_for Destination
        for trav in class_to_names.get("TravelerType", []):
            for dest in class_to_names.get("City", []) + class_to_names.get("Island", []):
                _add_rel(dest.title(), "suitable_for", trav.title(), 0.58)

        return relations[:50]  # cap per record to avoid noise
