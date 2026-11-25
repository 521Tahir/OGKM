"""Neo4j client for the hybrid ontology (read-only graph queries for the UI).

Uses NEO4J_URI/USER/PASSWORD from .env. Falls back gracefully if unavailable.
"""
from __future__ import annotations
import os


class Neo4jClient:
    def __init__(self):
        self.driver = None
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "jeju1234")))
            self.driver.verify_connectivity()
        except Exception:
            self.driver = None

    def neighbours(self, concept: str, limit: int = 15) -> list[dict]:
        if not self.driver:
            return []
        q = ("MATCH (c {name:$n})-[r]-(m) RETURN type(r) AS rel, m.name AS other LIMIT $k")
        try:
            with self.driver.session() as s:
                return [dict(rec) for rec in s.run(q, n=concept, k=limit)]
        except Exception:
            return []
