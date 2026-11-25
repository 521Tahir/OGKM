"""
Orchestrates the full ontology extraction pipeline for a batch of records.

Per-record steps:
1. Concept extraction  (LLM or offline fallback)
2. Relation extraction (LLM or offline fallback)
3. Validate candidate
4. Merge into global store

Post-batch steps:
5. Hierarchy induction
6. Rule suggestion
7. Concept normalization
8. Export (Turtle + JSON-LD)
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .schema import OntologyConcept, OntologyRelation, OntologyCandidate, TourismOntologySchema
from .concept_extractor import ConceptExtractor
from .relation_extractor import RelationExtractor
from .offline_extractor import OfflineExtractor
from .validator import OntologyValidator, CandidateValidationResult
from .normalizer import ConceptNormalizer
from .hierarchy_inducer import HierarchyInducer
from .rule_suggester import RuleSuggester
from .versioning import OntologyVersionManager, OntologyChange
from .exporters import OntologyExporter
from ..data.loaders import TourismRecord
from ..utils.logging_utils import get_logger
from ..utils.io import save_json, ensure_dir
from ..utils.misc import chunk_list

logger = get_logger("ontology.builder")
SCHEMA = TourismOntologySchema


class OntologyBuilder:
    """
    Runs the full ontology construction pipeline over a dataset.
    Produces a merged, normalised, validated global ontology.
    """

    def __init__(
        self,
        client=None,
        prompt_manager=None,
        concept_strategy: str = "cot",
        relation_strategy: str = "cot",
        output_dir: str | Path = "artifacts/ontologies",
        checkpoint_dir: str | Path = "artifacts/runs/checkpoints",
        version: str = "0.1.0",
    ):
        # Use LLM mode if a live client is provided (local or Anthropic) with a prompt manager
        has_llm = (
            client is not None
            and prompt_manager is not None
            and getattr(client, "is_available", False)
        )

        if has_llm:
            self.concept_extractor: Any = ConceptExtractor(
                client, prompt_manager, strategy=concept_strategy
            )
            self.relation_extractor: Any = RelationExtractor(
                client, prompt_manager, strategy=relation_strategy
            )
            self._mode = "llm"
            backend = getattr(client, "backend", "unknown")
            logger.info(f"OntologyBuilder: LLM mode ({concept_strategy} strategy, backend={backend})")
        else:
            self._offline = OfflineExtractor()
            self.concept_extractor = None
            self.relation_extractor = None
            self._mode = "offline"
            logger.info("OntologyBuilder: offline mode (no LLM available)")

        self.validator = OntologyValidator()
        self.normalizer = ConceptNormalizer()
        self.hierarchy_inducer = HierarchyInducer()
        self.rule_suggester = RuleSuggester()
        self.version_manager = OntologyVersionManager(output_dir)
        self.exporter = OntologyExporter(output_dir=output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.version = version
        self._concept_strategy = concept_strategy  # used in post-batch GoT/metacog steps
        self._llm_client = client                  # kept for post-batch LLM calls
        ensure_dir(self.checkpoint_dir)

        # Global merged ontology
        self._concepts: dict[str, OntologyConcept] = {}   # key: name.lower()
        self._relations: set[tuple] = set()                # (subj, pred, obj)
        self._relation_objects: list[OntologyRelation] = []
        self._validation_results: list[dict] = []
        self._hierarchy_nodes: list[dict] = []
        self._rules: list[dict] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def process_records(
        self,
        records: list[TourismRecord],
        batch_size: int = 50,
        max_records: Optional[int] = None,
        save_every_n: int = 500,
        llm_max_records: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Process records and build the global ontology. Returns summary.

        llm_max_records: if set and LLM is available, only first N records use LLM;
                         the rest fall back to offline extraction automatically.
        """
        if max_records:
            records = records[:max_records]

        logger.info(
            f"OntologyBuilder [{self._mode}]: processing {len(records):,} records"
        )
        if llm_max_records is not None and self._mode == "llm":
            logger.info(
                f"  Hybrid: LLM for first {llm_max_records:,}, offline for remainder"
            )

        for i, record in enumerate(records):
            # Switch to offline after llm_max_records threshold
            use_offline = (
                self._mode == "llm"
                and llm_max_records is not None
                and i >= llm_max_records
            )
            try:
                candidate = self._process_record(record, force_offline=use_offline)
                result = self.validator.validate_candidate(candidate)
                self._validation_results.append(result.to_dict())
                self._merge_candidate(candidate, result)
            except Exception as e:
                logger.error(f"Record {record.record_id}: pipeline error — {e}")

            if (i + 1) % save_every_n == 0:
                self._checkpoint(i + 1)
                mode_tag = "offline" if use_offline else self._mode
                logger.info(
                    f"  [{i+1}/{len(records)}] mode={mode_tag} "
                    f"concepts={len(self._concepts):,} "
                    f"relations={len(self._relations):,}"
                )

        # Post-batch: normalise, induce hierarchy, suggest rules
        logger.info("Running post-batch steps: normalisation, hierarchy, rules ...")
        self._run_post_batch()
        self._checkpoint(len(records))
        return self._build_summary(len(records))

    def save_ontology(self) -> dict[str, str]:
        """Persist current merged ontology. Returns paths."""
        concepts_list = [c.to_dict() for c in self._concepts.values()]
        relations_list = [r.to_dict() for r in self._relation_objects]

        version_dir = self.version_manager.save_version(
            version=self.version,
            concepts=concepts_list,
            relations=relations_list,
            rules=self._rules,
            schema=SCHEMA.to_dict(),
            changelog=[],
        )

        ttl_path = self.exporter.export_turtle(
            concepts_list, relations_list, self._rules, version=self.version
        )
        jsonld_path = self.exporter.export_jsonld(
            concepts_list, relations_list, version=self.version
        )

        # Save hierarchy and rules separately
        ensure_dir(version_dir)
        save_json(self._hierarchy_nodes, Path(version_dir) / "hierarchy.json")

        return {
            "version": self.version,
            "version_dir": str(version_dir),
            "turtle": ttl_path,
            "jsonld": jsonld_path,
            "n_concepts": len(concepts_list),
            "n_relations": len(relations_list),
            "n_rules": len(self._rules),
            "n_hierarchy_nodes": len(self._hierarchy_nodes),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _process_record(self, record: TourismRecord, force_offline: bool = False) -> OntologyCandidate:
        combined = f"{record.input_text}\n{record.output_text}"

        if self._mode == "llm" and not force_offline:
            concepts = self.concept_extractor.extract(combined, record.record_id)
            concept_dicts = [{"name": c.name, "concept_class": c.concept_class} for c in concepts]
            relations = self.relation_extractor.extract(combined, concept_dicts, record.record_id)
        else:
            if not hasattr(self, "_offline"):
                self._offline = OfflineExtractor()
            concepts = self._offline.extract_concepts(combined, record.record_id)
            relations = self._offline.extract_relations(combined, concepts, record.record_id)

        return OntologyCandidate(
            record_id=record.record_id,
            concepts=concepts,
            relations=relations,
        )

    def _merge_candidate(
        self,
        candidate: OntologyCandidate,
        result: CandidateValidationResult,
    ) -> None:
        if result.status == "fail":
            return

        for concept in candidate.concepts:
            key = concept.name.lower().strip()
            if key not in self._concepts:
                self._concepts[key] = concept
            else:
                existing = self._concepts[key]
                if concept.provenance.confidence > existing.provenance.confidence:
                    self._concepts[key] = concept
                else:
                    for alias in concept.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

        for rel in candidate.relations:
            triple = (rel.subject.lower(), rel.predicate, rel.obj.lower())
            if triple not in self._relations:
                self._relations.add(triple)
                self._relation_objects.append(rel)

    def _run_post_batch(self) -> None:
        """Normalise, induce hierarchy, suggest rules, then apply GoT/metacognitive if configured."""
        concept_list = list(self._concepts.values())

        # Step 1: Normalise
        normalised, norm_report = self.normalizer.normalize(concept_list)
        self._concepts = {c.name.lower(): c for c in normalised}
        logger.info(
            f"Normalisation: {norm_report['input_count']} → "
            f"{norm_report['output_count']} ({norm_report['merged_count']} merged)"
        )

        # Step 2: Hierarchy
        nodes, is_a_rels = self.hierarchy_inducer.induce(normalised)
        self._hierarchy_nodes = [n.to_dict() for n in nodes]
        for rel in is_a_rels:
            triple = (rel.subject.lower(), rel.predicate, rel.obj.lower())
            if triple not in self._relations:
                self._relations.add(triple)
                self._relation_objects.append(rel)

        # Step 3: Rules
        rules = self.rule_suggester.suggest(normalised, self._relation_objects)
        self._rules = [r.to_dict() for r in rules]

        # Step 4: GoT structural coherence check (§6.4) — applies for "got" or "unified"
        if self._concept_strategy in ("got", "unified"):
            self._apply_got_coherence(normalised)

        # Step 5: Metacognitive filtering (§6.6) — applies for "metacognitive" or "unified"
        if self._concept_strategy in ("metacognitive", "unified"):
            self._apply_metacognitive_filter(normalised)

    def _apply_got_coherence(self, concepts: list) -> None:
        """Run GoT structural coherence check and remove self-loops / duplicates (§6.4)."""
        try:
            from src.llm.got_prompter import GoTPrompter
            got = GoTPrompter(llm_client=self._llm_client)
            concept_dicts = [{"name": c.name, "concept_class": c.concept_class,
                              "confidence": c.provenance.confidence} for c in concepts]
            rel_dicts = [{"subject": r.subject, "predicate": r.predicate, "object": r.obj}
                         for r in self._relation_objects]
            report = got.check_structural_coherence(concept_dicts, rel_dicts)
            logger.info(
                f"GoT coherence: score={report.score:.3f}, "
                f"redundant={len(report.redundant)}, conflicting={len(report.conflicting)}"
            )
            # Remove self-loop relations flagged by GoT
            before = len(self._relation_objects)
            self._relation_objects = [
                r for r in self._relation_objects
                if r.subject.lower() != r.obj.lower()
            ]
            removed = before - len(self._relation_objects)
            if removed:
                self._relations = {
                    (r.subject.lower(), r.predicate, r.obj.lower())
                    for r in self._relation_objects
                }
                logger.info(f"GoT: removed {removed} self-loop relations")
        except Exception as e:
            logger.warning(f"GoT coherence check skipped: {e}")

    def _apply_metacognitive_filter(self, concepts: list) -> None:
        """Run metacognitive confidence calibration and flag low-confidence concepts (§6.6)."""
        try:
            from src.llm.metacognitive_prompter import MetacognitivePrompter
            mp = MetacognitivePrompter(llm_client=self._llm_client, review_threshold=0.5)
            concept_dicts = [{"name": c.name, "concept_class": c.concept_class,
                              "confidence": c.provenance.confidence} for c in concepts]
            accepted, flagged = mp.flag_for_review(concept_dicts, threshold=0.5)
            logger.info(
                f"Metacognitive filter: {len(accepted)} accepted, {len(flagged)} flagged for review"
            )
            # Lower confidence for flagged concepts rather than removing them
            flagged_names = {f["name"].lower() for f in flagged}
            for c in self._concepts.values():
                if c.name.lower() in flagged_names:
                    c.provenance.validator_status = "warning"
        except Exception as e:
            logger.warning(f"Metacognitive filter skipped: {e}")

    def _checkpoint(self, processed_count: int) -> None:
        save_json(
            {
                "processed_count": processed_count,
                "n_concepts": len(self._concepts),
                "n_relations": len(self._relations),
                "mode": self._mode,
            },
            self.checkpoint_dir / "ontology_build_checkpoint.json",
        )

    def _build_summary(self, total: int) -> dict[str, Any]:
        val = self._validation_results
        return {
            "total_records": total,
            "total_concepts": len(self._concepts),
            "total_relations": len(self._relations),
            "total_rules": len(self._rules),
            "total_hierarchy_nodes": len(self._hierarchy_nodes),
            "extraction_mode": self._mode,
            "validation": {
                "pass": sum(1 for r in val if r.get("status") == "pass"),
                "warning": sum(1 for r in val if r.get("status") == "warning"),
                "fail": sum(1 for r in val if r.get("status") == "fail"),
            },
        }
