"""
Hybrid Retriever — combines ontology subgraph retrieval with dense text retrieval.

Two retrieval modes:
  - ontology_only : SubgraphRetriever only
  - text_only     : FAISS dense retrieval over record texts
  - hybrid        : ontology subgraph + top-k text passages, merged

The hybrid mode is the target condition for the RAG experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from ..graph.graph_store import TourismGraphStore
from .concept_linker import ConceptLinker
from .subgraph_retriever import SubgraphRetriever, SubgraphContext
from ..utils.logging_utils import get_logger

logger = get_logger("retrieval.hybrid")


@dataclass
class HybridContext:
    """Combined retrieval result from ontology + text retrieval."""
    query: str
    subgraph: SubgraphContext
    text_passages: list[dict]   # list of {text, score, record_id}
    mode: str                   # "ontology_only" | "text_only" | "hybrid"

    def to_text(self, max_passages: int = 3, max_relations: int = 20) -> str:
        """Render combined context as text block for LLM."""
        parts: list[str] = []

        if self.mode in ("ontology_only", "hybrid") and not self.subgraph.is_empty():
            parts.append(self.subgraph.to_text(max_relations=max_relations))

        if self.mode in ("text_only", "hybrid") and self.text_passages:
            parts.append("\n=== Retrieved Text Passages ===")
            for i, p in enumerate(self.text_passages[:max_passages], 1):
                parts.append(f"\n[Passage {i}] (score={p['score']:.3f})")
                parts.append(p["text"][:500])

        return "\n".join(parts) if parts else "[No context retrieved]"

    def is_empty(self) -> bool:
        return self.subgraph.is_empty() and not self.text_passages

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode":  self.mode,
            "ontology": self.subgraph.to_dict(),
            "text_passages": len(self.text_passages),
        }


class HybridRetriever:
    """
    Unified retriever supporting ontology-only, text-only, and hybrid modes.

    For text retrieval: uses FAISS index if available; falls back to BM25-style
    substring search over in-memory records.
    """

    def __init__(
        self,
        graph: TourismGraphStore,
        records: Optional[list[dict]] = None,
        embedding_model: Optional[Any] = None,
        faiss_index: Optional[Any] = None,
        max_hops: int = 2,
        max_nodes: int = 50,
        top_k_text: int = 5,
    ) -> None:
        self.linker   = ConceptLinker(graph, embedding_model=embedding_model)
        self.onto_ret = SubgraphRetriever(
            graph, linker=self.linker, max_hops=max_hops, max_nodes=max_nodes
        )
        self.records       = records or []
        self.faiss_index   = faiss_index
        self.embed_model   = embedding_model
        self.top_k_text    = top_k_text
        self._record_texts = [
            r.get("Input", "") + " " + r.get("Output", "")
            for r in self.records
        ]

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
    ) -> HybridContext:
        """
        Retrieve context for query.

        mode: "ontology_only" | "text_only" | "hybrid"
        """
        # Ontology retrieval
        if mode in ("ontology_only", "hybrid"):
            subgraph = self.onto_ret.retrieve(query)
        else:
            subgraph = SubgraphContext(
                query=query, seeds=[], concepts=[], relations=[], linked_concepts=[]
            )

        # Text retrieval
        if mode in ("text_only", "hybrid") and self.records:
            passages = self._text_retrieve(query)
        else:
            passages = []

        return HybridContext(
            query=query, subgraph=subgraph, text_passages=passages, mode=mode
        )

    def retrieve_batch(
        self,
        queries: list[str],
        mode: str = "hybrid",
    ) -> list[HybridContext]:
        return [self.retrieve(q, mode=mode) for q in queries]

    # ── Text retrieval ────────────────────────────────────────────────────────

    def _text_retrieve(self, query: str) -> list[dict]:
        """Dense retrieval (FAISS) or fallback substring search."""
        if self.faiss_index is not None and self.embed_model is not None:
            return self._faiss_retrieve(query)
        return self._bm25_retrieve(query)

    def _faiss_retrieve(self, query: str) -> list[dict]:
        try:
            import numpy as np
            q_emb = self.embed_model.encode([query], show_progress_bar=False)
            distances, indices = self.faiss_index.search(
                q_emb.astype("float32"), self.top_k_text
            )
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.records):
                    continue
                r = self.records[idx]
                results.append({
                    "text":      r.get("Input", ""),
                    "score":     float(1 / (1 + dist)),
                    "record_id": r.get("record_id", idx),
                })
            return results
        except Exception as e:
            logger.debug(f"FAISS retrieval failed: {e}")
            return self._bm25_retrieve(query)

    def _bm25_retrieve(self, query: str) -> list[dict]:
        """Simple token-overlap retrieval over in-memory records."""
        import re
        q_tokens = set(re.split(r"\W+", query.lower())) - {""}
        scored: list[tuple[float, int]] = []
        for i, text in enumerate(self._record_texts):
            t_tokens = set(re.split(r"\W+", text.lower())) - {""}
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens) / (len(q_tokens | t_tokens) + 1e-9)
            if overlap > 0:
                scored.append((overlap, i))
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[: self.top_k_text]:
            r = self.records[idx]
            results.append({
                "text":      r.get("Input", "")[:500],
                "score":     score,
                "record_id": r.get("record_id", idx),
            })
        return results
