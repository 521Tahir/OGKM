"""
Concept Linker — maps free-text query terms to ontology concepts.

Linking pipeline (in priority order):
  1. Exact name match (case-insensitive)
  2. Token overlap search (via GraphStore.search_concepts)
  3. Sentence-transformer embedding similarity (if model loaded)

Returns a ranked list of (concept_dict, score, method) tuples.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..graph.graph_store import TourismGraphStore
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.concept_linker")


class ConceptLinker:
    """
    Links query text to ontology concepts.

    Usage:
        linker = ConceptLinker(graph)
        links = linker.link("best restaurants in Jeju Island", top_k=5)
    """

    def __init__(
        self,
        graph: TourismGraphStore,
        embedding_model: Optional[Any] = None,  # sentence_transformers.SentenceTransformer
        min_score: float = 0.3,
    ) -> None:
        self.graph = graph
        self.min_score = min_score
        self._embed_model = embedding_model
        self._concept_embeddings: Optional[Any] = None
        self._concept_names_indexed: list[str] = []

        if embedding_model is not None:
            self._build_embeddings()

    def _build_embeddings(self) -> None:
        """Pre-compute embeddings for all concept names."""
        try:
            names = list(self.graph._concepts.keys())
            logger.info(f"Building embeddings for {len(names):,} concepts …")
            self._concept_embeddings = self._embed_model.encode(
                names, batch_size=512, show_progress_bar=False
            )
            self._concept_names_indexed = names
            logger.info("Concept embeddings ready.")
        except Exception as e:
            logger.warning(f"Could not build embeddings: {e}")
            self._embed_model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def link(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Link query to ontology concepts.

        Returns list of dicts:
          { concept, score, method }
        Sorted by score descending, truncated to top_k.
        """
        results: dict[str, dict] = {}  # name → best result so far

        # 1. Exact match
        for token in _query_ngrams(query):
            concept = self.graph.get_concept(token)
            if concept:
                name = concept["name"]
                if name not in results or results[name]["score"] < 1.0:
                    results[name] = {"concept": concept, "score": 1.0, "method": "exact"}

        # 2. Token overlap
        for concept in self.graph.search_concepts(query, top_k=top_k * 3):
            name = concept["name"]
            if name not in results:
                score = _overlap_score(query, name)
                if score >= self.min_score:
                    results[name] = {"concept": concept, "score": score, "method": "token_overlap"}

        # 3. Embedding similarity (if available)
        if self._embed_model is not None and self._concept_embeddings is not None:
            emb_results = self._embed_link(query, top_k=top_k * 2)
            for item in emb_results:
                name = item["concept"]["name"]
                if name not in results or results[name]["score"] < item["score"]:
                    results[name] = item

        # Sort and return top_k
        ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def link_batch(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Link a batch of queries."""
        return [self.link(q, top_k=top_k) for q in queries]

    # ── Embedding linking ─────────────────────────────────────────────────────

    def _embed_link(self, query: str, top_k: int = 10) -> list[dict]:
        try:
            import numpy as np
            q_emb = self._embed_model.encode([query], show_progress_bar=False)
            # Cosine similarity
            norms = np.linalg.norm(self._concept_embeddings, axis=1, keepdims=True)
            normed = self._concept_embeddings / (norms + 1e-9)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            scores = (normed @ q_norm.T).flatten()
            top_idx = scores.argsort()[::-1][:top_k]
            results = []
            for idx in top_idx:
                score = float(scores[idx])
                if score >= self.min_score:
                    name = self._concept_names_indexed[idx]
                    concept = self.graph._concepts.get(name)
                    if concept:
                        results.append({"concept": concept, "score": score, "method": "embedding"})
            return results
        except Exception as e:
            logger.debug(f"Embedding link failed: {e}")
            return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_ngrams(query: str, max_n: int = 4) -> list[str]:
    """Generate word n-grams from query for exact matching (longest first)."""
    words = re.split(r"\s+", query.strip())
    ngrams: list[str] = []
    for n in range(min(max_n, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def _overlap_score(query: str, name: str) -> float:
    """Token Jaccard overlap between query and concept name."""
    q_tokens = set(re.split(r"\W+", query.lower())) - {""}
    n_tokens = set(re.split(r"\W+", name.lower())) - {""}
    if not q_tokens or not n_tokens:
        return 0.0
    intersection = q_tokens & n_tokens
    union = q_tokens | n_tokens
    return len(intersection) / len(union)
