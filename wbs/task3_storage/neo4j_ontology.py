"""WBS Task 3.1.1 - the ontology in Neo4j, built the way `Ontology_Project` builds it.

`/home/tahirahmad/Ontology_Project` is the companion project, and it already had
a Neo4j graph layer: `src/graph/neo4j_client.py` plus `src/graph/cypher_queries.py`,
driven by `scripts/ingest_to_neo4j.py`. This module is that design carried over to
the v1.4.0 Jeju ontology, so the two projects hold the same graph under the same
schema and a query written against one runs against the other.

What is carried over, deliberately unchanged:

    (:Concept {name, concept_class, confidence, aliases, source})
    (:Concept)-[:RELATES_TO {predicate, confidence, source}]->(:Concept)

    five indexes - name, concept_class, confidence, RELATES_TO.predicate, and a
    fulltext index over name - and MERGE rather than CREATE, so a re-run updates
    in place instead of duplicating.

    the read interface: get_concept, concepts_by_class, neighbours,
    extract_subgraph, shortest_path, search_concepts, stats.

Two things are different, and both are changes I would defend:

  * their loader opens one write transaction per concept and per relation, which
    is 9,474 round trips for this ontology. This one keeps MERGE semantics and
    the same property set but sends batches through UNWIND, which is the same
    graph in a fraction of the time. Measured in `graph_db_neo4j_portable.json`.

  * their `SHORTEST_PATH` interpolates `$max_hops` into a variable-length
    pattern. Neo4j does not accept a parameter there - the query raises at
    parse time - so the bound is formatted into the Cypher after being checked
    as an integer, which is the only way to write it and still refuse input
    that is not a number.

This schema is our "generic" mapping from `neo4j_store.py`: one label, one
relationship type, the predicate as a property. The other mapping, a label per
class and a relationship type per predicate, is the one `neo4j_store.py` leaves
in the database by default. Neo4j Community serves a single user database, so
the two cannot both be live; `--clear` swaps which one is.

  python -m wbs.task3_storage.neo4j_ontology --clear      # load, replacing
  python -m wbs.task3_storage.neo4j_ontology --dry-run    # counts only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
ONTO = ROOT / "artifacts" / "ontologies"
DELIV = ROOT / "wbs" / "deliverables" / "task3_storage"

NODE_LABEL = "Concept"
REL_TYPE = "RELATES_TO"

# ---------------------------------------------------------------- the schema

CREATE_INDEXES = [
    f"CREATE INDEX concept_name IF NOT EXISTS "
    f"FOR (c:{NODE_LABEL}) ON (c.name)",
    f"CREATE INDEX concept_class IF NOT EXISTS "
    f"FOR (c:{NODE_LABEL}) ON (c.concept_class)",
    f"CREATE INDEX concept_confidence IF NOT EXISTS "
    f"FOR (c:{NODE_LABEL}) ON (c.confidence)",
    f"CREATE INDEX rel_predicate IF NOT EXISTS "
    f"FOR ()-[r:{REL_TYPE}]-() ON (r.predicate)",
    f"CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS "
    f"FOR (c:{NODE_LABEL}) ON EACH [c.name]",
]

# UNWIND around the companion project's own MERGE, so the write semantics are
# identical and the transaction count is not.
MERGE_CONCEPTS = f"""
UNWIND $rows AS row
MERGE (c:{NODE_LABEL} {{name: row.name}})
SET   c.concept_class = row.concept_class,
      c.confidence    = row.confidence,
      c.aliases       = row.aliases,
      c.source        = row.source
"""

MERGE_RELATIONS = f"""
UNWIND $rows AS row
MATCH  (a:{NODE_LABEL} {{name: row.subject}}),
       (b:{NODE_LABEL} {{name: row.object}})
MERGE  (a)-[r:{REL_TYPE} {{predicate: row.predicate}}]->(b)
SET    r.confidence = row.confidence,
       r.source     = row.source,
       r.ordinal    = row.ordinal
"""

GET_CONCEPT_BY_NAME = f"""
MATCH (c:{NODE_LABEL}) WHERE toLower(c.name) = toLower($name)
RETURN c LIMIT 1
"""

GET_CONCEPTS_BY_CLASS = f"""
MATCH (c:{NODE_LABEL} {{concept_class: $concept_class}})
WHERE c.confidence >= $min_conf
RETURN c ORDER BY c.confidence DESC LIMIT $limit
"""

FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes('concept_fulltext', $query)
YIELD node, score
RETURN node AS c, score LIMIT $limit
"""

CONTAINS_SEARCH = f"""
MATCH (c:{NODE_LABEL}) WHERE toLower(c.name) CONTAINS toLower($token)
RETURN c LIMIT $limit
"""

OUTGOING_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]->(b:{NODE_LABEL})
WHERE toLower(a.name) = toLower($name)
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b ORDER BY r.confidence DESC LIMIT $limit
"""

INCOMING_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]->(b:{NODE_LABEL})
WHERE toLower(b.name) = toLower($name)
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b ORDER BY r.confidence DESC LIMIT $limit
"""

BOTH_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]-(b:{NODE_LABEL})
WHERE (toLower(a.name) = toLower($name) OR toLower(b.name) = toLower($name))
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b ORDER BY r.confidence DESC LIMIT $limit
"""

# The hop bound cannot be a parameter in a variable-length pattern, so it is
# formatted in - after int() has refused anything that is not a number.
SHORTEST_PATH = f"""
MATCH (src:{NODE_LABEL}), (tgt:{NODE_LABEL})
WHERE toLower(src.name) = toLower($source) AND toLower(tgt.name) = toLower($target)
MATCH path = shortestPath((src)-[:{REL_TYPE}*1..%d]-(tgt))
RETURN path LIMIT 1
"""

# The companion project's version caps the node list in Python *after* the
# relations are collected, so a 50-node subgraph comes back carrying thousands of
# relations whose endpoints were cut - here, 7,001 for 50 nodes, because two hops
# out of any Jeju place passes through the `Jeju` hub and reaches most of the
# graph. This one caps inside Cypher and then returns only the edges induced on
# the nodes it kept, ranked by how many paths from the seeds reach them, which is
# deterministic and prefers what the seeds are actually connected to.
SUBGRAPH_NHOP = f"""
MATCH (seed:{NODE_LABEL}) WHERE seed.name IN $seed_names
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS s
MATCH (s)-[:{REL_TYPE}*1..%d]-(nb:{NODE_LABEL})
WHERE NOT nb IN seeds
WITH seeds, nb, count(*) AS paths
ORDER BY paths DESC, nb.name ASC
LIMIT $max_extra
WITH seeds, collect(nb) AS extra
WITH seeds + extra AS nodes
CALL (nodes) {{
    UNWIND nodes AS n
    MATCH (n)-[r:{REL_TYPE}]->(m:{NODE_LABEL}) WHERE m IN nodes
    RETURN collect(DISTINCT r) AS rels
}}
RETURN nodes, rels AS relations
"""

CONCEPT_STATS = f"""
MATCH (c:{NODE_LABEL}) WITH c.concept_class AS cls, count(*) AS cnt
RETURN cls, cnt ORDER BY cnt DESC
"""

TOTAL_COUNTS = f"""
MATCH (c:{NODE_LABEL})
WITH count(c) AS n_concepts, count(DISTINCT c.concept_class) AS n_classes
OPTIONAL MATCH ()-[r:{REL_TYPE}]->()
RETURN n_concepts, n_classes, count(r) AS n_relations,
       count(DISTINCT r.predicate) AS n_predicates
"""

DELETE_ALL = "MATCH (n) DETACH DELETE n"

# Cards are what the retrieval arm consumes: one row per subject, its class, its
# degree and its out-edges, assembled in the database rather than in Python.
CARD_ROWS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]->(b:{NODE_LABEL})
WHERE toLower(a.name) <> toLower(b.name)
WITH a, r, b ORDER BY r.ordinal
WITH a, collect({{predicate: r.predicate, object: b.name}}) AS edges
RETURN a.name AS name, a.concept_class AS concept_class, edges
"""

DEGREE_ROWS = f"""
MATCH (c:{NODE_LABEL})
OPTIONAL MATCH (c)-[r:{REL_TYPE}]-(o:{NODE_LABEL})
WHERE toLower(c.name) <> toLower(o.name)
RETURN c.name AS name, count(r) AS degree
"""


def _concept(node) -> dict:
    d = dict(node.items())
    return {"name": d.get("name", ""), "concept_class": d.get("concept_class", ""),
            "confidence": d.get("confidence"), "aliases": list(d.get("aliases") or []),
            "source": d.get("source", "")}


def _relation(a, r, b) -> dict:
    rd = dict(r.items())
    return {"subject": dict(a.items()).get("name", ""),
            "predicate": rd.get("predicate", ""),
            "object": dict(b.items()).get("name", ""),
            "confidence": rd.get("confidence"), "source": rd.get("source", "")}


class OntologyGraph:
    """The companion project's Neo4jClient interface, over our ontology."""

    def __init__(self, uri: str = None, user: str = None, password: str = None,
                 database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.database = database
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                  password or os.environ.get("NEO4J_PASSWORD", "jeju-ontology-2026")))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _read(self, cypher: str, **params) -> list:
        with self._driver.session(database=self.database) as s:
            return list(s.run(cypher, **params))

    def _write(self, cypher: str, **params):
        with self._driver.session(database=self.database) as s:
            return s.execute_write(lambda tx: tx.run(cypher, **params).consume())

    # ------------------------------------------------------------- schema

    def create_indexes(self) -> list[str]:
        made = []
        with self._driver.session(database=self.database) as s:
            for stmt in CREATE_INDEXES:
                s.run(stmt)
                made.append(stmt.split()[2] if stmt.startswith("CREATE INDEX")
                            else stmt.split()[3])
            # A fulltext index is populated asynchronously; querying it before it
            # comes online returns nothing and looks like an empty ontology.
            s.run("CALL db.awaitIndexes(120)")
        return made

    def clear(self):
        self._write(DELETE_ALL)

    # ------------------------------------------------------------- loading

    @staticmethod
    def _concept_rows(concepts: list[dict]) -> list[dict]:
        rows = []
        for c in concepts:
            prov = c.get("provenance") or {}
            rows.append({
                "name": c.get("name", ""),
                "concept_class": c.get("concept_class") or "",
                "confidence": float(prov.get("confidence",
                                             c.get("confidence", 0.0)) or 0.0),
                "aliases": [str(a) for a in (c.get("aliases") or [])],
                "source": str(prov.get("extraction_prompt_version",
                                       prov.get("source", c.get("source", "")))),
            })
        return rows

    @staticmethod
    def _relation_rows(relations: list[dict]) -> list[dict]:
        """One property beyond the companion project's schema: `ordinal`.

        The retrieval arm keeps the first 8 facts of an entity in the order the
        relation file asserts them, so a card is a function of that order. Cypher
        makes no promise about the order `collect` returns edges in, which means a
        graph without the assertion index cannot reproduce the published contexts
        and the store would become a confound in every comparison that uses it.
        """
        rows = []
        for ordinal, r in enumerate(relations):
            prov = r.get("provenance") or {}
            rows.append({
                "subject": r.get("subject", ""),
                "predicate": r.get("predicate", ""),
                "object": r.get("object", ""),
                "confidence": float(prov.get("confidence",
                                             r.get("confidence", 0.0)) or 0.0),
                "source": str(prov.get("extraction_prompt_version",
                                       prov.get("source", r.get("source", "")))),
                "ordinal": ordinal,
            })
        return rows

    def load_ontology_version(self, version: str, ontology_dir: Path = ONTO,
                              clear_existing: bool = False,
                              batch_size: int = 500) -> dict:
        path = Path(ontology_dir) / version
        if not path.exists():
            raise FileNotFoundError(f"ontology version not found: {path}")
        concepts = json.loads((path / "concepts.json").read_text(encoding="utf-8"))
        relations = json.loads((path / "relations.json").read_text(encoding="utf-8"))

        if clear_existing:
            self.clear()
        t_idx = time.perf_counter()
        self.create_indexes()
        idx_s = time.perf_counter() - t_idx

        crows = self._concept_rows(concepts)
        rrows = self._relation_rows(relations)

        t0 = time.perf_counter()
        for i in range(0, len(crows), batch_size):
            self._write(MERGE_CONCEPTS, rows=crows[i:i + batch_size])
        t_nodes = time.perf_counter() - t0

        t1 = time.perf_counter()
        for i in range(0, len(rrows), batch_size):
            self._write(MERGE_RELATIONS, rows=rrows[i:i + batch_size])
        t_edges = time.perf_counter() - t1

        st = self.stats()
        return {
            "version": version,
            "concepts_sent": len(crows), "relations_sent": len(rrows),
            "concepts_in_graph": st["n_concepts"],
            "relations_in_graph": st["n_relations"],
            # MERGE on (a)-[:RELATES_TO {predicate}]->(b) collapses a repeated
            # triple onto one edge; saying so beats a silent shortfall.
            "relations_collapsed_by_merge": len(rrows) - st["n_relations"],
            "index_seconds": round(idx_s, 2),
            "node_seconds": round(t_nodes, 2),
            "edge_seconds": round(t_edges, 2),
            "batch_size": batch_size,
            "transactions": (len(crows) + batch_size - 1) // batch_size
                            + (len(rrows) + batch_size - 1) // batch_size,
        }

    # ------------------------------------------------------------- reading

    def get_concept(self, name: str) -> Optional[dict]:
        rec = self._read(GET_CONCEPT_BY_NAME, name=name)
        return _concept(rec[0]["c"]) if rec else None

    def concepts_by_class(self, concept_class: str, min_conf: float = 0.0,
                          limit: int = 100) -> list[dict]:
        return [_concept(r["c"]) for r in
                self._read(GET_CONCEPTS_BY_CLASS, concept_class=concept_class,
                           min_conf=min_conf, limit=limit)]

    def neighbours(self, name: str, direction: str = "out",
                   predicates: list[str] = None, limit: int = 100) -> list[dict]:
        cypher = {"out": OUTGOING_NEIGHBOURS, "in": INCOMING_NEIGHBOURS,
                  "both": BOTH_NEIGHBOURS}.get(direction, OUTGOING_NEIGHBOURS)
        if predicates:
            out = []
            for p in predicates:
                out += [_relation(r["a"], r["r"], r["b"]) for r in
                        self._read(cypher, name=name, predicate=p, limit=limit)]
            return out[:limit]
        return [_relation(r["a"], r["r"], r["b"]) for r in
                self._read(cypher, name=name, predicate=None, limit=limit)]

    def extract_subgraph(self, seed_names: list[str], max_hops: int = 2,
                         max_nodes: int = 50,
                         predicates: list[str] = None) -> dict:
        if not seed_names:
            return {"concepts": [], "relations": [], "seeds": [], "hops": 0}
        hops = max(1, min(int(max_hops), 4))
        seeds = list(seed_names)
        extra = max(0, int(max_nodes) - len(seeds))
        rec = self._read(SUBGRAPH_NHOP % hops, seed_names=seeds, max_extra=extra)
        if not rec:
            return {"concepts": [], "relations": [], "seeds": seeds, "hops": hops}
        nodes = [_concept(n) for n in (rec[0]["nodes"] or [])]
        rels = [_relation(r.start_node, r, r.end_node)
                for r in (rec[0]["relations"] or []) if r is not None]
        if predicates:
            keep = set(predicates)
            rels = [r for r in rels if r["predicate"] in keep]
        # Every relation returned must join two concepts that are also returned,
        # or the caller is handed a context referring to entities it cannot see.
        kept = {c["name"] for c in nodes}
        rels = [r for r in rels if r["subject"] in kept and r["object"] in kept]
        return {"concepts": nodes, "relations": rels,
                "seeds": seeds, "hops": hops}

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> dict:
        hops = max(1, min(int(max_hops), 8))
        rec = self._read(SHORTEST_PATH % hops, source=source, target=target)
        if not rec:
            return {"source": source, "target": target, "found": False,
                    "length": 0, "nodes": [], "relations": []}
        path = rec[0]["path"]
        return {"source": source, "target": target, "found": True,
                "length": len(path.relationships),
                "nodes": [dict(n.items()).get("name", "") for n in path.nodes],
                "relations": [dict(r.items()).get("predicate", "")
                              for r in path.relationships]}

    # Lucene gives these characters syntactic meaning; a place name containing
    # one would otherwise raise instead of being searched for.
    _LUCENE = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')

    def search_concepts(self, query: str, top_k: int = 10) -> list[dict]:
        escaped = self._LUCENE.sub(r"\\\1", query).strip()
        if escaped:
            try:
                rec = self._read(FULLTEXT_SEARCH, query=escaped, limit=top_k)
                if rec:
                    return [dict(_concept(r["c"]), score=r["score"]) for r in rec]
            except Exception:
                pass
        tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        token = tokens[0] if tokens else query.lower()
        return [_concept(r["c"]) for r in
                self._read(CONTAINS_SEARCH, token=token, limit=top_k)]

    def stats(self) -> dict:
        rec = self._read(TOTAL_COUNTS)
        if not rec:
            return {"n_concepts": 0, "n_relations": 0, "n_classes": 0,
                    "n_predicates": 0, "class_counts": {}}
        row = rec[0]
        return {"n_concepts": row["n_concepts"], "n_relations": row["n_relations"],
                "n_classes": row["n_classes"], "n_predicates": row["n_predicates"],
                "class_counts": {r["cls"]: r["cnt"] for r in
                                 self._read(CONCEPT_STATS) if r["cls"]}}

    def cards(self, max_facts: int = 8) -> list[dict]:
        """The retrieval arm's entity cards, assembled from Cypher rows.

        Same shape as `OntoV3ContextBuilder._build_cards`, so the experiment can
        take its cards from the graph database instead of from the JSON files.
        """
        degree = {r["name"]: r["degree"] for r in self._read(DEGREE_ROWS)}
        out = []
        for r in self._read(CARD_ROWS):
            name = r["name"]
            if len(name) < 3:
                continue
            klass = r["concept_class"] or "Concept"
            seen, facts = set(), []
            for e in r["edges"]:
                key = (str(e["predicate"]).lower(), str(e["object"]).lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(f"{name} "
                             f"{str(e['predicate']).replace('_', ' ').strip()} "
                             f"{e['object']}")
                if len(facts) >= max_facts:
                    break
            out.append({"name": name, "klass": klass,
                        "degree": degree.get(name, 0), "facts": facts,
                        "text": f"{name} ({klass}). " + "; ".join(facts)})
        return out


# --------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Load the Jeju ontology into Neo4j under the companion "
                    "project's schema")
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--bolt", default=os.environ.get("NEO4J_URI",
                                                     "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD",
                                                         "jeju-ontology-2026"))
    ap.add_argument("--clear", action="store_true",
                    help="delete everything first; Neo4j Community serves one "
                         "database, so this is how the typed mapping is swapped "
                         "out for this one")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--out", default=str(DELIV / "graph_db_neo4j_portable.json"))
    a = ap.parse_args()

    path = ONTO / a.version
    concepts = json.loads((path / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((path / "relations.json").read_text(encoding="utf-8"))

    if a.dry_run:
        from collections import Counter
        print(f"ontology v{a.version}: {len(concepts):,} concepts, "
              f"{len(relations):,} relations")
        for label, counter in (
                ("class", Counter(c.get("concept_class") or "?" for c in concepts)),
                ("predicate", Counter(r.get("predicate") or "?" for r in relations))):
            print(f"\n  top {label}s")
            for k, n in counter.most_common(10):
                print(f"    {k:<28}{n:>6,}")
        print("\nnothing written (--dry-run)")
        return

    g = OntologyGraph(a.bolt, a.user, a.password)
    try:
        print(f"loading v{a.version} into {a.bolt} under the companion project's "
              f"schema: (:{NODE_LABEL})-[:{REL_TYPE} {{predicate}}]->(:{NODE_LABEL})")
        res = g.load_ontology_version(a.version, clear_existing=a.clear,
                                      batch_size=a.batch_size)
        st = g.stats()

        # Exercise every read primitive, so "the interface works" is measured
        # rather than asserted by the fact that it imported.
        t = {}
        t0 = time.perf_counter()
        jeju = g.get_concept("Jeju")
        t["get_concept"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        beaches = g.concepts_by_class("Beach", limit=100)
        t["concepts_by_class"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        nb = g.neighbours("Jeju", direction="both", limit=500)
        t["neighbours"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        found = g.search_concepts("Seongsan Ilchulbong", top_k=5)
        t["search_concepts"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        sub = g.extract_subgraph(["Seongsan Ilchulbong", "Hallasan"], max_hops=2,
                                 max_nodes=50)
        t["extract_subgraph"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        sp = g.shortest_path("Hallasan", "Jeju International Airport", max_hops=4)
        t["shortest_path"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        cards = g.cards()
        t["cards"] = round((time.perf_counter() - t0) * 1000, 2)

        out = {
            "schema": {"node_label": NODE_LABEL, "relationship_type": REL_TYPE,
                       "node_properties": ["name", "concept_class", "confidence",
                                           "aliases", "source"],
                       "relationship_properties": ["predicate", "confidence",
                                                   "source", "ordinal"],
                       "indexes": CREATE_INDEXES},
            "ported_from": "/home/tahirahmad/Ontology_Project/src/graph",
            "bolt": a.bolt, "load": res, "stats": st, "read_ms": t,
            "probe": {
                "get_concept_Jeju": jeju,
                "concepts_by_class_Beach": len(beaches),
                "neighbours_Jeju_both": len(nb),
                "search_concepts_top1": found[0]["name"] if found else None,
                "subgraph_2hop_nodes": len(sub["concepts"]),
                "subgraph_2hop_relations": len(sub["relations"]),
                "shortest_path": sp,
                "cards": len(cards),
            },
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                               encoding="utf-8")

        print(f"\nloaded in {res['node_seconds']}s + {res['edge_seconds']}s "
              f"({res['transactions']} transactions, batch {res['batch_size']})")
        print(f"  concepts  {st['n_concepts']:,}   classes    {st['n_classes']}")
        print(f"  relations {st['n_relations']:,}   predicates {st['n_predicates']}")
        if res["relations_collapsed_by_merge"]:
            print(f"  {res['relations_collapsed_by_merge']} relation rows merged "
                  f"onto an existing edge (same subject, predicate and object)")
        print("\nread interface, against the live graph:")
        for k, ms in t.items():
            print(f"  {k:<20}{ms:>8.2f} ms")
        print(f"\n  Jeju: {jeju['concept_class']}, confidence "
              f"{jeju['confidence']}, {len(nb)} neighbours")
        print(f"  2-hop subgraph from 2 seeds: {len(sub['concepts'])} concepts, "
              f"{len(sub['relations'])} relations")
        print(f"  Hallasan -> Jeju International Airport: "
              + (" -> ".join(sp["nodes"]) if sp["found"] else "no path"))
        print(f"\nwrote {a.out}")
    finally:
        g.close()


if __name__ == "__main__":
    main()
