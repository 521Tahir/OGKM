"""
Tourism QA Benchmark.

Constructs a fixed benchmark of (query, expected_facts) pairs sampled
from the test split. Used to evaluate all four pipeline conditions.

Benchmark construction strategy:
  - Sample from test records (6,197 total)
  - Derive query from Input field
  - Derive expected entity set from Output field (via simple extraction)
  - Store ground-truth concept names where they overlap with the ontology

Week 9 additions:
  - CategorizedBenchmarkItem: extends BenchmarkItem with category field
  - TourismBenchmark.build_categorized(): 11-category benchmark (≥20/category)
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils.io import load_json, save_json, ensure_dir
from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.benchmark")


@dataclass
class CategorizedBenchmarkItem:
    """A benchmark item with tourism category label."""
    item_id:           int
    query:             str
    expected_answer:   str
    expected_concepts: list[str] = field(default_factory=list)
    record_id:         int = -1
    source_file:       str = ""
    category:          str = "attraction_recommendation"
    category_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "item_id":              self.item_id,
            "query":                self.query,
            "expected_answer":      self.expected_answer,
            "expected_concepts":    self.expected_concepts,
            "record_id":            self.record_id,
            "source_file":          self.source_file,
            "category":             self.category,
            "category_confidence":  round(self.category_confidence, 3),
        }


@dataclass
class BenchmarkItem:
    """A single benchmark query with ground truth."""
    item_id:    int
    query:      str
    expected_answer:  str       # the Output field (reference answer)
    expected_concepts: list[str] = field(default_factory=list)  # known ontology entities
    record_id:  int = -1
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "item_id":           self.item_id,
            "query":             self.query,
            "expected_answer":   self.expected_answer,
            "expected_concepts": self.expected_concepts,
            "record_id":         self.record_id,
            "source_file":       self.source_file,
        }


class TourismBenchmark:
    """
    Creates and manages the tourism QA benchmark.

    Usage:
        bench = TourismBenchmark.build(test_path, ontology_concepts, n=200)
        bench.save("artifacts/reports/benchmark.json")
        items = bench.items
    """

    def __init__(self, items: list[BenchmarkItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def build(
        cls,
        test_split_path: str | Path,
        ontology_concepts: list[dict],
        n: int = 200,
        seed: int = 42,
        min_query_len: int = 20,
    ) -> "TourismBenchmark":
        """
        Sample n benchmark items from test split.

        Filters out records with very short Input fields.
        Matches Output text against known ontology concept names
        to build the expected_concepts set.
        """
        path = Path(test_split_path)
        if path.suffix == ".json":
            records = load_json(path)
        else:
            # JSONL
            import json
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]

        # Build concept name set for matching
        concept_names = [c.get("name", "") for c in ontology_concepts if c.get("name")]
        concept_set   = {n.lower(): n for n in concept_names}

        # Filter records with sufficient input length
        eligible = [
            r for r in records
            if len(r.get("Input", "")) >= min_query_len
        ]
        random.seed(seed)
        sampled = random.sample(eligible, min(n, len(eligible)))

        items: list[BenchmarkItem] = []
        for i, r in enumerate(sampled):
            query  = r.get("Input", "").strip()
            answer = r.get("Output", "").strip()
            # Find ontology entities mentioned in the reference answer
            answer_lower = answer.lower()
            expected_concepts = [
                canonical for lower, canonical in concept_set.items()
                if lower in answer_lower
            ][:10]

            items.append(BenchmarkItem(
                item_id=i,
                query=query,
                expected_answer=answer,
                expected_concepts=expected_concepts,
                record_id=r.get("record_id", -1),
                source_file=r.get("source_file", ""),
            ))

        logger.info(
            f"Benchmark built: {len(items)} items from {len(eligible)} eligible records "
            f"(total {len(records)})"
        )
        return cls(items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        save_json(
            {
                "n_items": len(self.items),
                "items": [it.to_dict() for it in self.items],
            },
            path,
        )
        logger.info(f"Benchmark saved → {path}")

    @classmethod
    def build_categorized(
        cls,
        test_split_path: str | Path,
        ontology_concepts: list[dict],
        n_per_category: int = 20,
        seed: int = 42,
        min_query_len: int = 20,
    ) -> tuple["TourismBenchmark", list[CategorizedBenchmarkItem]]:
        """
        Build an 11-category benchmark with ≥n_per_category items each.

        Uses QueryCategorizer to assign categories to all eligible records,
        then samples n_per_category from each category. Returns:
          - A TourismBenchmark (for API compatibility)
          - A list of CategorizedBenchmarkItem (category-labelled)

        Requires: query_categorizer module in same package.
        """
        from .query_categorizer import QueryCategorizer, CATEGORIES

        path = Path(test_split_path)
        if path.suffix == ".json":
            records = load_json(path)
        else:
            import json
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]

        concept_names = [c.get("name", "") for c in ontology_concepts if c.get("name")]
        concept_set   = {n.lower(): n for n in concept_names}

        eligible = [r for r in records if len(r.get("Input", "")) >= min_query_len]

        # Categorize all eligible records
        categorizer = QueryCategorizer()
        cat_buckets: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
        cat_results = categorizer.categorize_batch([r.get("Input", "") for r in eligible])

        for r, cr in zip(eligible, cat_results):
            cat_buckets[cr.category].append((r, cr))

        logger.info("Category bucket sizes before sampling:")
        for cat in CATEGORIES:
            logger.info(f"  {cat:<35} {len(cat_buckets[cat]):>4}")

        random.seed(seed)
        all_items: list[CategorizedBenchmarkItem] = []
        item_id = 0
        for cat in CATEGORIES:
            bucket = cat_buckets[cat]
            sample = random.sample(bucket, min(n_per_category, len(bucket)))
            for r, cr in sample:
                query  = r.get("Input", "").strip()
                answer = r.get("Output", "").strip()
                answer_lower = answer.lower()
                expected_concepts = [
                    canonical for lower, canonical in concept_set.items()
                    if lower in answer_lower
                ][:10]
                all_items.append(CategorizedBenchmarkItem(
                    item_id=item_id,
                    query=query,
                    expected_answer=answer,
                    expected_concepts=expected_concepts,
                    record_id=r.get("record_id", -1),
                    source_file=r.get("source_file", ""),
                    category=cat,
                    category_confidence=cr.confidence,
                ))
                item_id += 1

        # Also produce a TourismBenchmark from the same items for API compatibility
        base_items = [
            BenchmarkItem(
                item_id=ci.item_id,
                query=ci.query,
                expected_answer=ci.expected_answer,
                expected_concepts=ci.expected_concepts,
                record_id=ci.record_id,
                source_file=ci.source_file,
            )
            for ci in all_items
        ]
        logger.info(
            f"Categorized benchmark: {len(all_items)} items across {len(CATEGORIES)} categories "
            f"(~{len(all_items)//len(CATEGORIES)} per category)"
        )
        return cls(base_items), all_items

    @classmethod
    def save_categorized(
        cls,
        categorized_items: list[CategorizedBenchmarkItem],
        path: str | Path,
    ) -> None:
        """Save categorized benchmark items to JSON."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(
            {
                "n_items": len(categorized_items),
                "n_categories": len({ci.category for ci in categorized_items}),
                "items": [ci.to_dict() for ci in categorized_items],
            },
            path,
        )
        logger.info(f"Categorized benchmark saved → {path}")

    @classmethod
    def load_categorized(cls, path: str | Path) -> list[CategorizedBenchmarkItem]:
        """Load categorized benchmark from JSON."""
        data = load_json(path)
        return [
            CategorizedBenchmarkItem(
                item_id=d["item_id"],
                query=d["query"],
                expected_answer=d["expected_answer"],
                expected_concepts=d.get("expected_concepts", []),
                record_id=d.get("record_id", -1),
                source_file=d.get("source_file", ""),
                category=d.get("category", "attraction_recommendation"),
                category_confidence=d.get("category_confidence", 0.0),
            )
            for d in data["items"]
        ]

    @classmethod
    def load(cls, path: str | Path) -> "TourismBenchmark":
        data = load_json(path)
        items = [
            BenchmarkItem(
                item_id=d["item_id"],
                query=d["query"],
                expected_answer=d["expected_answer"],
                expected_concepts=d.get("expected_concepts", []),
                record_id=d.get("record_id", -1),
                source_file=d.get("source_file", ""),
            )
            for d in data["items"]
        ]
        return cls(items)
