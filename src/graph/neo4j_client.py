"""
Neo4j client for the Tourism Ontology graph.

Provides the same query interface as TourismGraphStore so it can be used
as a drop-in backend: get_concept, get_concepts_by_class, neighbours,
extract_subgraph, search_concepts, stats.

Additionally exposes bulk-load methods for ingesting ontology versions.

Usage (with a real Neo4j instance):
    client = Neo4jClient.from_env()          # reads NEO4J_URI / _USER / _PASSWORD
    client.load_ontology_version("0.7.0")    # one-time bulk import
    concept = client.get_concept("Hallasan")

Usage (offline / test mode — no Neo4j required):
    client = Neo4jClient(uri=None)           # auto-enters offline mode
    # All query methods return empty results gracefully.

Connection parameters can also be supplied via environment variables:
    NEO4J_URI      — bolt://localhost:7687
    NEO4J_USER     — neo4j
    NEO4J_PASSWORD — <password>
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ..utils.logging_utils import get_logger
from . import cypher_queries as CQ

logger = get_logger("graph.neo4j")

ROOT = Path(__file__).resolve().parents[3]


# ── Result helpers ────────────────────────────────────────────────────────────

def _node_to_concept(node) -> dict:
    """Convert a neo4j Node object to a plain concept dict."""
    d = dict(node.items())
    aliases = d.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    d["aliases"] = list(aliases)
    return d


def _rel_to_dict(a_node, rel, b_node) -> dict:
    """Convert a neo4j Relationship to a plain relation dict."""
    return {
        "subject":    dict(a_node.items()).get("name", ""),
        "predicate":  rel.get("predicate", ""),
        "object":     dict(b_node.items()).get("name", ""),
        "confidence": float(rel.get("confidence", 0.0)),
        "source":     rel.get("source", ""),
    }


# ── Main client ───────────────────────────────────────────────────────────────

class Neo4jClient:
    """
    Thin wrapper around the neo4j Python driver exposing tourism-ontology
    specific query operations and bulk-load utilities.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        uri:      Optional[str] = "bolt://localhost:7687",
        user:     str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_retry: int = 3,
    ) -> None:
        self._uri      = uri
        self._user     = user
        self._password = password
        self._database = database
        self._driver   = None
        self._offline  = uri is None

        if not self._offline:
            self._connect(max_retry)

    @classmethod
    def from_env(cls, **kwargs) -> "Neo4jClient":
        """Construct from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars."""
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            **kwargs,
        )

    def _connect(self, max_retry: int = 3) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("neo4j package not installed — entering offline mode")
            self._offline = True
            return

        for attempt in range(1, max_retry + 1):
            try:
                self._driver = GraphDatabase.driver(
                    self._uri, auth=(self._user, self._password)
                )
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self._uri}")
                return
            except Exception as exc:
                logger.warning(f"Neo4j connect attempt {attempt}/{max_retry}: {exc}")
                if attempt < max_retry:
                    time.sleep(2 ** attempt)

        logger.error("Could not connect to Neo4j — entering offline mode")
        self._offline = True

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Low-level execution ───────────────────────────────────────────────────

    def _run(self, cypher: str, **params) -> list[Any]:
        """Execute a Cypher query and return all records as a list."""
        if self._offline or not self._driver:
            return []
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return list(result)

    def _run_write(self, cypher: str, **params) -> None:
        """Execute a write Cypher query inside an explicit write transaction."""
        if self._offline or not self._driver:
            return
        with self._driver.session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(cypher, **params))

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        if self._offline:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ── Schema initialisation ─────────────────────────────────────────────────

    def create_indexes(self) -> None:
        """Create all schema indexes (idempotent — safe to re-run)."""
        if self._offline:
            return
        with self._driver.session(database=self._database) as session:
            for stmt in CQ.CREATE_INDEXES:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug(f"Index creation skipped: {exc}")
        logger.info("Neo4j indexes verified")

    # ── Bulk loading ──────────────────────────────────────────────────────────

    def bulk_load_concepts(
        self,
        concepts: list[dict],
        batch_size: int = 500,
    ) -> int:
        """
        Upsert (MERGE) concepts into Neo4j in batches.

        Returns the number of concepts processed.
        """
        if self._offline or not concepts:
            return 0

        loaded = 0
        with self._driver.session(database=self._database) as session:
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i : i + batch_size]
                for c in batch:
                    prov = c.get("provenance", {})
                    session.execute_write(
                        lambda tx, c=c, prov=prov: tx.run(
                            CQ.MERGE_CONCEPT,
                            name=c.get("name", ""),
                            concept_class=c.get("concept_class", ""),
                            confidence=float(
                                prov.get("confidence", c.get("confidence", 0.0))
                            ),
                            aliases=list(c.get("aliases", []) or []),
                            source=str(prov.get("source", c.get("source", ""))),
                        )
                    )
                loaded += len(batch)
                logger.info(f"Loaded concepts {loaded}/{len(concepts)}")

        return loaded

    def bulk_load_relations(
        self,
        relations: list[dict],
        batch_size: int = 500,
    ) -> int:
        """
        Upsert (MERGE) relations into Neo4j in batches.

        Skips relations whose subject or object node does not yet exist.
        Returns the number of relations processed.
        """
        if self._offline or not relations:
            return 0

        loaded = 0
        skipped = 0
        with self._driver.session(database=self._database) as session:
            for i in range(0, len(relations), batch_size):
                batch = relations[i : i + batch_size]
                for r in batch:
                    try:
                        prov = r.get("provenance", {})
                        session.execute_write(
                            lambda tx, r=r, prov=prov: tx.run(
                                CQ.MERGE_RELATION,
                                subject=r.get("subject", ""),
                                predicate=r.get("predicate", ""),
                                object=r.get("object", ""),
                                confidence=float(
                                    prov.get("confidence", r.get("confidence", 0.0))
                                ),
                                source=str(prov.get("source", r.get("source", ""))),
                            )
                        )
                        loaded += 1
                    except Exception:
                        skipped += 1
                if loaded % (batch_size * 10) == 0 and loaded > 0:
                    logger.info(f"Loaded relations {loaded}/{len(relations)}")

        logger.info(f"Relations loaded: {loaded}, skipped: {skipped}")
        return loaded

    def load_ontology_version(
        self,
        version: str,
        ontology_dir: str | Path = "artifacts/ontologies",
        clear_existing: bool = False,
    ) -> dict:
        """
        Full pipeline: read concepts+relations from disk → upsert to Neo4j.

        Args:
            version:        ontology version string, e.g. "0.7.0"
            ontology_dir:   root directory of versioned ontologies
            clear_existing: if True, delete all nodes before loading (destructive)

        Returns:
            {"n_concepts": int, "n_relations": int, "version": str}
        """
        from ..utils.io import load_json

        ont_path = Path(ontology_dir) / version
        if not ont_path.exists():
            raise FileNotFoundError(f"Ontology version not found: {ont_path}")

        concepts  = load_json(ont_path / "concepts.json")
        relations = load_json(ont_path / "relations.json")

        logger.info(
            f"Loading v{version}: {len(concepts):,} concepts, "
            f"{len(relations):,} relations into Neo4j"
        )

        self.create_indexes()

        if clear_existing:
            logger.warning("Clearing existing graph data before load")
            self._run_write(CQ.DELETE_ALL)

        t0 = time.perf_counter()
        n_c = self.bulk_load_concepts(concepts)
        n_r = self.bulk_load_relations(relations)
        elapsed = round(time.perf_counter() - t0, 1)

        logger.info(
            f"Load complete: {n_c:,} concepts, {n_r:,} relations in {elapsed}s"
        )
        return {"n_concepts": n_c, "n_relations": n_r, "version": version}

    # ── Read queries (mirrors TourismGraphStore interface) ────────────────────

    def get_concept(self, name: str) -> Optional[dict]:
        """Exact concept lookup by name (case-insensitive). Returns None if not found."""
        records = self._run(CQ.GET_CONCEPT_BY_NAME, name=name)
        if records:
            return _node_to_concept(records[0]["c"])
        return None

    def get_concepts_by_class(
        self,
        concept_class: str,
        min_conf: float = 0.0,
        top_k: int = 50,
    ) -> list[dict]:
        """Return concepts of a given class, ranked by confidence."""
        records = self._run(
            CQ.GET_CONCEPTS_BY_CLASS,
            concept_class=concept_class,
            min_conf=min_conf,
            limit=top_k,
        )
        return [_node_to_concept(r["c"]) for r in records]

    def neighbours(
        self,
        name: str,
        direction: str = "out",
        predicates: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Return one-hop neighbour relations.

        Args:
            name:       concept name
            direction:  "out" | "in" | "both"
            predicates: filter to these predicate types (None = all)
            limit:      max relations to return

        Returns:
            List of relation dicts with keys: subject, predicate, object,
            confidence, source.
        """
        query_map = {
            "out":  CQ.OUTGOING_NEIGHBOURS,
            "in":   CQ.INCOMING_NEIGHBOURS,
            "both": CQ.BOTH_NEIGHBOURS,
        }
        cypher = query_map.get(direction, CQ.OUTGOING_NEIGHBOURS)

        # Run once per predicate filter (or once with None)
        if predicates:
            results: list[dict] = []
            for pred in predicates:
                records = self._run(cypher, name=name, predicate=pred, limit=limit)
                results.extend(_rel_to_dict(r["a"], r["r"], r["b"]) for r in records)
            return results[:limit]

        records = self._run(cypher, name=name, predicate=None, limit=limit)
        return [_rel_to_dict(r["a"], r["r"], r["b"]) for r in records]

    def extract_subgraph(
        self,
        seed_names: list[str],
        max_hops: int = 2,
        max_nodes: int = 50,
        predicates: Optional[list[str]] = None,
    ) -> dict:
        """
        BFS subgraph extraction from seed concepts.

        Uses the 2-hop Cypher expansion (no APOC required).

        Returns:
            {"concepts": [...], "relations": [...], "seeds": [...], "hops": max_hops}
        """
        if self._offline or not seed_names:
            return {"concepts": [], "relations": [], "seeds": [], "hops": 0}

        records = self._run(
            CQ.SUBGRAPH_2HOP,
            seed_names=seed_names,
        )

        if not records:
            return {"concepts": [], "relations": [], "seeds": seed_names, "hops": 0}

        row = records[0]
        concepts  = [_node_to_concept(n)   for n in (row.get("concepts")  or [])]
        relations = [_rel_to_dict(r.start_node, r, r.end_node)
                     for r in (row.get("relations") or [])]

        # Apply predicate filter if requested
        if predicates:
            pred_set  = set(predicates)
            relations = [r for r in relations if r["predicate"] in pred_set]

        return {
            "concepts":  concepts[:max_nodes],
            "relations": relations,
            "seeds":     seed_names,
            "hops":      max_hops,
        }

    def shortest_path(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
    ) -> dict:
        """
        Find the shortest path between two concepts using Cypher shortestPath().

        Returns:
            {
                "source": str, "target": str, "found": bool,
                "length": int,
                "nodes": [concept_name, ...],
                "relations": [predicate, ...]
            }
        """
        records = self._run(
            CQ.SHORTEST_PATH,
            source=source,
            target=target,
            max_hops=max_hops,
        )

        if not records:
            return {"source": source, "target": target, "found": False,
                    "length": 0, "nodes": [], "relations": []}

        path = records[0]["path"]
        nodes = [dict(n.items()).get("name", "") for n in path.nodes]
        rels  = [dict(r.items()).get("predicate", "") for r in path.relationships]

        return {
            "source":    source,
            "target":    target,
            "found":     True,
            "length":    len(path.relationships),
            "nodes":     nodes,
            "relations": rels,
        }

    def search_concepts(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Fulltext search over concept names.

        Falls back to CONTAINS search if the fulltext index is unavailable.
        """
        # Try fulltext index first
        try:
            records = self._run(
                CQ.FULLTEXT_SEARCH,
                query=query,
                limit=top_k,
            )
            if records:
                return [_node_to_concept(r["c"]) for r in records]
        except Exception:
            pass

        # CONTAINS fallback: search for the first meaningful token
        import re
        tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        token  = tokens[0] if tokens else query.lower()
        records = self._run(CQ.CONTAINS_SEARCH, token=token, limit=top_k)
        return [_node_to_concept(r["c"]) for r in records]

    def stats(self) -> dict:
        """Return aggregate graph statistics."""
        counts = self._run(CQ.TOTAL_COUNTS)
        if not counts:
            return {
                "n_concepts": 0, "n_relations": 0,
                "n_classes": 0, "n_predicates": 0,
                "class_counts": {},
            }

        row = counts[0]
        class_records = self._run(CQ.CONCEPT_STATS)
        class_counts  = {r["cls"]: r["cnt"] for r in class_records if r["cls"]}

        return {
            "n_concepts":   row["n_concepts"],
            "n_relations":  row["n_relations"],
            "n_classes":    row["n_classes"],
            "n_predicates": row["n_predicates"],
            "class_counts": class_counts,
        }

    def clear(self) -> None:
        """Delete all Concept nodes and their relationships. Use with caution."""
        self._run_write(CQ.DELETE_ALL)
        logger.info("Neo4j graph cleared")

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return not self._offline and self._driver is not None

    @property
    def is_offline(self) -> bool:
        return self._offline

    def __repr__(self) -> str:
        mode = "offline" if self._offline else f"connected @ {self._uri}"
        return f"Neo4jClient({mode})"
