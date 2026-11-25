"""LLM-driven relation extractor for tourism ontology."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schema import OntologyRelation, ProvenanceMetadata, TourismOntologySchema
from ..llm.client import LLMClient
from ..llm.prompt_manager import PromptManager
from ..llm.parsers import JSONOutputParser
from ..utils.logging_utils import get_logger

logger = get_logger("ontology.relation_extractor")
SCHEMA = TourismOntologySchema


class RelationExtractor:
    """Extract semantic relations between tourism concepts using LLM."""

    PROMPT_VERSION = "v1.0"

    def __init__(
        self,
        client: LLMClient,
        prompt_manager: PromptManager,
        strategy: str = "cot",
        min_confidence: float = 0.5,
    ):
        self.client = client
        self.pm = prompt_manager
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.parser = JSONOutputParser()

    def extract(
        self,
        text: str,
        concepts: list[dict],
        record_id: int,
    ) -> list[OntologyRelation]:
        """
        Extract relations given text and already-identified concepts.

        Args:
            text: original input/output text
            concepts: list of concept dicts (name, concept_class)
            record_id: source record identifier

        Returns:
            List of OntologyRelation objects.
        """
        if not concepts:
            return []

        concept_summary = ", ".join(
            f"{c['name']} ({c.get('concept_class','?')})" for c in concepts[:20]
        )

        # Truncate text to keep prompt focused
        text = text[:1000]

        try:
            prompt = self.pm.render_template(
                "relation_extraction",
                self.strategy,
                {"text": text, "concepts": concept_summary},
            )
        except FileNotFoundError:
            prompt = self._fallback_prompt(text, concept_summary)

        raw_output = self.client.call(prompt, max_tokens=800)
        items, ok, err = self.parser.parse_list(raw_output)

        if not ok:
            logger.warning(f"Record {record_id}: relation parsing failed — {err}")
            return []

        relations: list[OntologyRelation] = []
        for item in items:
            try:
                rel = self._build_relation(item, record_id)
                if rel:
                    relations.append(rel)
            except Exception as e:
                logger.debug(f"Skipping malformed relation item: {e}")

        logger.debug(f"Record {record_id}: extracted {len(relations)} relations.")
        return relations

    def _build_relation(self, item: dict, record_id: int) -> OntologyRelation | None:
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        confidence = float(item.get("confidence", 0.5))

        if not subj or not pred or not obj:
            return None
        if not SCHEMA.is_valid_relation(pred):
            logger.debug(f"Invalid relation predicate '{pred}' — skipping.")
            return None
        if confidence < self.min_confidence:
            return None

        return OntologyRelation(
            subject=subj,
            predicate=pred,
            obj=obj,
            provenance=ProvenanceMetadata(
                source_record_id=record_id,
                extraction_prompt_version=self.PROMPT_VERSION,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                confidence=confidence,
                validator_status="pending",
            ),
        )

    def _fallback_prompt(self, text: str, concept_summary: str) -> str:
        return (
            f"Extract relations between tourism concepts. "
            f"Concepts: {concept_summary}\n\nText: {text}\n\n"
            f"Return JSON array: [{{\"subject\":str, \"predicate\":str, \"object\":str, \"confidence\":0.8}}]"
        )
