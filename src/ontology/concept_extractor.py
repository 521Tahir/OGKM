"""
LLM-driven concept extractor.

Supports multiple prompting strategies:
1. naive
2. cot (chain-of-thought)
3. metacognitive
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schema import OntologyConcept, ProvenanceMetadata, TourismOntologySchema
from ..llm.client import LLMClient
from ..llm.prompt_manager import PromptManager
from ..llm.parsers import JSONOutputParser
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.concept_extractor")
SCHEMA = TourismOntologySchema


class ConceptExtractor:
    """Extract tourism concepts from text using LLM prompts."""

    PROMPT_VERSION = "v1.0"

    def __init__(
        self,
        client: LLMClient,
        prompt_manager: PromptManager,
        strategy: str = "cot",   # "naive" | "cot" | "metacognitive"
        min_confidence: float = 0.5,
    ):
        self.client = client
        self.pm = prompt_manager
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.parser = JSONOutputParser()

    def extract(self, text: str, record_id: int) -> list[OntologyConcept]:
        """
        Extract concepts from text for a given record_id.

        Returns:
            List of OntologyConcept objects (validated against schema).
        """
        # Truncate to avoid slow generation on very long records
        text = text[:1500]

        try:
            prompt = self.pm.render_template(
                "concept_extraction", self.strategy, {"text": text}
            )
        except FileNotFoundError:
            logger.warning(f"Prompt template 'concept_extraction/{self.strategy}' not found. Using naive.")
            prompt = self._fallback_prompt(text)

        raw_output = self.client.call(prompt, max_tokens=800)
        items, ok, err = self.parser.parse_list(raw_output)

        if not ok:
            logger.warning(
                f"Record {record_id}: concept parsing failed — {err}. "
                "Attempting repair..."
            )
            items = self._repair_parse(raw_output, record_id)

        concepts: list[OntologyConcept] = []
        for item in items:
            try:
                concept = self._build_concept(item, record_id)
                if concept:
                    concepts.append(concept)
            except Exception as e:
                logger.debug(f"Skipping malformed concept item: {e}")

        logger.debug(f"Record {record_id}: extracted {len(concepts)} concepts.")
        return concepts

    def _build_concept(self, item: dict, record_id: int) -> OntologyConcept | None:
        """Validate and build an OntologyConcept from a raw LLM item dict."""
        name = str(item.get("name", "")).strip()
        cclass = str(item.get("concept_class", "")).strip()
        confidence = float(item.get("confidence", 0.5))

        if not name or len(name) < 2:
            return None
        if not SCHEMA.is_valid_class(cclass):
            # Try to find closest match — if not found, default to Attraction
            logger.debug(f"Unknown class '{cclass}' for concept '{name}' — defaulting to Attraction.")
            cclass = "Attraction"
        if confidence < self.min_confidence:
            return None

        return OntologyConcept(
            name=name,
            concept_class=cclass,
            aliases=item.get("aliases", []),
            attributes=item.get("attributes", {}),
            provenance=ProvenanceMetadata(
                source_record_id=record_id,
                extraction_prompt_version=self.PROMPT_VERSION,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                confidence=confidence,
                validator_status="pending",
            ),
        )

    def _repair_parse(self, raw_output: str, record_id: int) -> list[dict]:
        """Attempt LLM-based JSON repair."""
        try:
            repair_prompt = self.pm.render_template(
                "repair", "json_repair", {"broken_json": raw_output[:2000]}
            )
            repaired = self.client.call(repair_prompt)
            items, ok, _ = self.parser.parse_list(repaired)
            if ok:
                logger.info(f"Record {record_id}: JSON repair successful.")
                return items
        except Exception as e:
            logger.error(f"Record {record_id}: repair failed — {e}")
        return []

    def _fallback_prompt(self, text: str) -> str:
        """Inline fallback prompt when template files are missing."""
        return (
            f"Extract tourism concepts from the following text and return a JSON array. "
            f"Each item: {{\"name\": str, \"concept_class\": str, \"aliases\": [], \"confidence\": 0.8}}\n\n"
            f"TEXT:\n{text}\n\nJSON:"
        )
