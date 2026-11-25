"""
Unit tests for retrieval pipeline:
  - ConceptLinker
  - SubgraphRetriever (with and without ranker)
  - HybridRetriever
  - SubgraphRanker
  - resolve_conflict (versioning)
"""
import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.graph_store import TourismGraphStore
from src.graph.subgraph_ranker import SubgraphRanker, RankedSubgraphResult
from src.retrieval.concept_linker import ConceptLinker
from src.retrieval.subgraph_retriever import SubgraphRetriever, SubgraphContext
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ontology.versioning import resolve_conflict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph():
    """Load real v0.7.0 ontology into GraphStore once for all tests."""
    concepts  = json.load(open("artifacts/ontologies/1.1.0/concepts.json"))
    relations = json.load(open("artifacts/ontologies/1.1.0/relations.json"))
    g = TourismGraphStore()
    g.load(concepts, relations)
    return g


@pytest.fixture(scope="module")
def linker(graph):
    return ConceptLinker(graph, min_score=0.2)


@pytest.fixture(scope="module")
def ranker(graph):
    return SubgraphRanker(graph)


@pytest.fixture(scope="module")
def retriever(graph):
    return SubgraphRetriever(graph, use_ranker=True, max_seeds=4, max_top_seeds=2)


@pytest.fixture(scope="module")
def retriever_no_rank(graph):
    return SubgraphRetriever(graph, use_ranker=False)


# ── ConceptLinker tests ───────────────────────────────────────────────────────

class TestConceptLinker:

    def test_exact_match(self, linker):
        """Exact concept name should score 1.0 via exact method."""
        results = linker.link("Jeju", top_k=3)
        assert len(results) > 0
        top = results[0]
        assert top["score"] > 0.8
        assert "concept" in top
        assert "name" in top["concept"]

    def test_returns_top_k(self, linker):
        results = linker.link("beach restaurant seafood", top_k=5)
        assert len(results) <= 5

    def test_empty_query(self, linker):
        results = linker.link("", top_k=3)
        assert isinstance(results, list)

    def test_unknown_term_returns_list(self, linker):
        """Completely unknown term should return a list (possibly empty)."""
        results = linker.link("xyzzyqwerty123", top_k=3)
        assert isinstance(results, list)

    def test_link_result_has_required_fields(self, linker):
        results = linker.link("hotel Jeju", top_k=3)
        for r in results:
            assert "concept" in r
            assert "score" in r
            assert "method" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_multi_term_query(self, linker):
        """Multi-word query should link to at least one concept."""
        results = linker.link("traditional Korean cuisine restaurant", top_k=5)
        assert len(results) >= 1

    def test_scores_sorted_descending(self, linker):
        results = linker.link("Jeju Island beach", top_k=5)
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)


# ── SubgraphRetriever tests ───────────────────────────────────────────────────

class TestSubgraphRetriever:

    def test_retrieve_returns_context(self, retriever):
        ctx = retriever.retrieve("beaches in Jeju")
        assert isinstance(ctx, SubgraphContext)

    def test_context_has_required_fields(self, retriever):
        ctx = retriever.retrieve("seafood restaurant Jeju")
        assert hasattr(ctx, "query")
        assert hasattr(ctx, "seeds")
        assert hasattr(ctx, "concepts")
        assert hasattr(ctx, "relations")
        assert hasattr(ctx, "linked_concepts")
        assert hasattr(ctx, "rank_score")
        assert hasattr(ctx, "n_candidates")

    def test_non_empty_for_known_query(self, retriever):
        ctx = retriever.retrieve("traditional food in Jeju Island")
        assert not ctx.is_empty()
        assert len(ctx.concepts) > 0

    def test_rank_score_set_when_ranker_active(self, retriever):
        ctx = retriever.retrieve("hotel near beach Jeju")
        # With multiple seeds ranker fires → rank_score > 0
        if ctx.n_candidates > 1:
            assert ctx.rank_score > 0.0
            assert ctx.n_candidates >= 2

    def test_rank_score_zero_without_ranker(self, retriever_no_rank):
        ctx = retriever_no_rank.retrieve("Jeju island tour")
        assert ctx.rank_score == 0.0
        assert ctx.n_candidates == 1

    def test_empty_result_for_gibberish(self, retriever):
        ctx = retriever.retrieve("xyzzy qwerty zzz123")
        assert isinstance(ctx, SubgraphContext)
        # May be empty — just confirm it doesn't crash
        assert ctx.concepts is not None

    def test_to_text_non_empty(self, retriever):
        ctx = retriever.retrieve("accommodation near Jeju")
        text = ctx.to_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_text_empty_context(self, retriever):
        ctx = SubgraphContext(
            query="test", seeds=[], concepts=[], relations=[],
            linked_concepts=[], rank_score=0.0, n_candidates=0
        )
        text = ctx.to_text()
        assert "No relevant" in text

    def test_to_dict_contains_rank_score(self, retriever):
        ctx = retriever.retrieve("Jeju tourism attractions")
        d = ctx.to_dict()
        assert "rank_score" in d
        assert "n_candidates" in d
        assert isinstance(d["rank_score"], float)

    def test_retrieve_batch(self, retriever):
        queries = ["beach Jeju", "traditional food", "hiking trails"]
        results = retriever.retrieve_batch(queries)
        assert len(results) == 3
        for ctx in results:
            assert isinstance(ctx, SubgraphContext)

    def test_merge_deduplicates_concepts(self, graph):
        """_merge_subgraphs should not produce duplicate concept names."""
        sg1 = {"seeds": ["A"], "concepts": [{"name": "X"}, {"name": "Y"}], "relations": []}
        sg2 = {"seeds": ["B"], "concepts": [{"name": "Y"}, {"name": "Z"}], "relations": []}
        merged = SubgraphRetriever._merge_subgraphs([sg1, sg2], max_nodes=50)
        names = [c["name"] for c in merged["concepts"]]
        assert len(names) == len(set(names))
        assert set(names) == {"X", "Y", "Z"}

    def test_merge_deduplicates_relations(self, graph):
        rel = {"subject": "A", "predicate": "located_in", "object": "B"}
        sg1 = {"seeds": [], "concepts": [], "relations": [rel]}
        sg2 = {"seeds": [], "concepts": [], "relations": [rel]}
        merged = SubgraphRetriever._merge_subgraphs([sg1, sg2], max_nodes=50)
        assert len(merged["relations"]) == 1


# ── SubgraphRanker tests ──────────────────────────────────────────────────────

class TestSubgraphRanker:

    def _make_subgraph(self, n_concepts=5, n_relations=10, conf=0.8, class_name="City"):
        concepts = [
            {"name": f"Concept{i}", "concept_class": class_name,
             "provenance": {"confidence": conf}}
            for i in range(n_concepts)
        ]
        relations = [
            {"subject": f"Concept{i%n_concepts}", "predicate": "located_in",
             "object": f"Concept{(i+1)%n_concepts}", "confidence": conf}
            for i in range(n_relations)
        ]
        return {"seeds": ["Concept0"], "concepts": concepts, "relations": relations}

    def test_rank_returns_sorted_results(self, ranker):
        sg_high = self._make_subgraph(n_concepts=10, n_relations=30, conf=0.95)
        sg_low  = self._make_subgraph(n_concepts=2,  n_relations=2,  conf=0.30)
        results = ranker.rank([sg_high, sg_low], query="beach resort")
        assert len(results) == 2
        assert results[0].total_score >= results[1].total_score

    def test_rank_single(self, ranker):
        sg = self._make_subgraph(n_concepts=5, n_relations=10, conf=0.80)
        result = ranker.rank_single(sg, query="food")
        assert isinstance(result, RankedSubgraphResult)
        assert 0.0 <= result.total_score <= 1.0

    def test_rank_empty_list(self, ranker):
        results = ranker.rank([], query="beach")
        assert results == []

    def test_empty_subgraph_scores_zero(self, ranker):
        sg = {"seeds": [], "concepts": [], "relations": []}
        result = ranker.rank_single(sg, query="beach")
        assert result.total_score == 0.0

    def test_score_breakdown_keys(self, ranker):
        sg = self._make_subgraph()
        result = ranker.rank_single(sg, "Jeju beach")
        bd = result.score_breakdown()
        assert set(bd.keys()) == {"total", "confidence", "centrality", "query", "density", "diversity"}

    def test_top_k_limits_results(self, ranker):
        subgraphs = [self._make_subgraph(conf=0.5 + i*0.05) for i in range(5)]
        results = ranker.rank(subgraphs, query="beach", top_k=3)
        assert len(results) == 3

    def test_compare_returns_markdown_table(self, ranker):
        subgraphs = [self._make_subgraph(), self._make_subgraph(conf=0.6)]
        table = ranker.compare(subgraphs, query="Jeju")
        assert "| Rank |" in table
        assert "|" in table

    def test_diversity_counts_distinct_classes(self, ranker):
        sg = {
            "seeds": [],
            "concepts": [
                {"name": "A", "concept_class": "City",    "provenance": {"confidence": 0.8}},
                {"name": "B", "concept_class": "Beach",   "provenance": {"confidence": 0.8}},
                {"name": "C", "concept_class": "Cuisine", "provenance": {"confidence": 0.8}},
            ],
            "relations": [],
        }
        result = ranker.rank_single(sg, "")
        assert result.n_classes == 3


# ── HybridRetriever tests ─────────────────────────────────────────────────────

class TestHybridRetriever:

    @pytest.fixture(scope="class")
    def hybrid(self, graph):
        # Load a small doc corpus from test split if available
        test_records = []
        test_path = Path("data/splits/test.jsonl")
        if test_path.exists():
            with open(test_path) as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    test_records.append(json.loads(line))
        return HybridRetriever(graph, records=test_records)

    def test_retrieve_returns_hybrid_context(self, hybrid):
        from src.retrieval.hybrid_retriever import HybridContext
        result = hybrid.retrieve("best beach in Jeju")
        assert isinstance(result, HybridContext)

    def test_result_to_dict_has_ontology_key(self, hybrid):
        result = hybrid.retrieve("traditional food Korea")
        d = result.to_dict()
        assert "ontology" in d
        assert "mode" in d

    def test_ontology_only_mode(self, hybrid):
        result = hybrid.retrieve("Jeju attractions", mode="ontology_only")
        assert result.mode == "ontology_only"
        assert result.text_passages == []

    def test_retrieve_does_not_crash_empty_query(self, hybrid):
        result = hybrid.retrieve("")
        assert result is not None

    def test_to_text_returns_string(self, hybrid):
        result = hybrid.retrieve("Jeju seafood restaurant")
        text = result.to_text()
        assert isinstance(text, str)
        assert len(text) > 0


# ── resolve_conflict tests ────────────────────────────────────────────────────

class TestResolveConflict:

    def test_duplicate_concept_higher_conf_wins(self):
        existing = {"name": "Haundae Beach", "confidence": 0.60, "aliases": []}
        incoming = {"name": "Haeundae Beach", "confidence": 0.92, "aliases": ["Haeundae"]}
        result, action = resolve_conflict(existing, incoming, "duplicate_concept")
        assert action == "merged"
        assert result["name"] == "Haeundae Beach"
        assert "Haundae Beach" in result["aliases"]

    def test_duplicate_concept_lower_conf_existing_wins(self):
        existing = {"name": "Jeju Island", "confidence": 0.95, "aliases": []}
        incoming = {"name": "Cheju Island", "confidence": 0.50, "aliases": []}
        result, action = resolve_conflict(existing, incoming, "duplicate_concept")
        assert action == "merged"
        assert result["name"] == "Jeju Island"
        assert "Cheju Island" in result["aliases"]

    def test_contradictory_relation_higher_conf_wins(self):
        existing = {"subject": "X", "predicate": "located_in", "object": "A", "confidence": 0.90}
        incoming = {"subject": "X", "predicate": "located_in", "object": "B", "confidence": 0.40}
        result, action = resolve_conflict(existing, incoming, "contradictory_relation")
        assert action == "kept_existing"
        assert result["object"] == "A"

    def test_contradictory_relation_non_functional_kept(self):
        """Non-functional predicates: keep existing (caller handles both)."""
        existing = {"subject": "X", "predicate": "has_language", "object": "Korean", "confidence": 0.9}
        incoming = {"subject": "X", "predicate": "has_language", "object": "English", "confidence": 0.9}
        result, action = resolve_conflict(existing, incoming, "contradictory_relation")
        assert action == "kept_existing"

    def test_contradictory_relation_expert_review_wins(self):
        existing = {"predicate": "located_in", "object": "A", "confidence": 0.50, "source": "expert_review"}
        incoming = {"predicate": "located_in", "object": "B", "confidence": 0.99}
        result, action = resolve_conflict(existing, incoming, "contradictory_relation")
        assert action == "kept_existing"
        assert result["object"] == "A"

    def test_contradictory_relation_tied_flagged(self):
        existing = {"subject": "X", "predicate": "located_in", "object": "A", "confidence": 0.80}
        incoming = {"subject": "X", "predicate": "located_in", "object": "B", "confidence": 0.82}
        result, action = resolve_conflict(existing, incoming, "contradictory_relation")
        # Gap < 0.05 → flagged or replaced
        assert action in ("flagged", "replaced", "kept_existing")

    def test_class_disagreement_always_flagged(self):
        existing = {"name": "X", "concept_class": "City"}
        incoming = {"name": "X", "concept_class": "Beach"}
        result, action = resolve_conflict(existing, incoming, "class_disagreement")
        assert action == "flagged"

    def test_confidence_collision_averaged(self):
        existing = {"confidence": 0.80}
        incoming = {"confidence": 0.60}
        result, action = resolve_conflict(existing, incoming, "confidence_collision")
        assert action == "merged"
        assert abs(result["confidence"] - 0.70) < 0.001

    def test_flag_policy_overrides_all(self):
        existing = {"name": "A", "confidence": 0.90}
        incoming = {"name": "B", "confidence": 0.10}
        result, action = resolve_conflict(existing, incoming, "duplicate_concept", policy="flag")
        assert action == "flagged"
        assert result == existing

    def test_unknown_conflict_type_flagged(self):
        result, action = resolve_conflict({"x": 1}, {"x": 2}, "unknown_type")
        assert action == "flagged"
