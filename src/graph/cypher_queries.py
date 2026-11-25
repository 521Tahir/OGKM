"""
Cypher query templates for the Tourism Ontology Neo4j graph.

Schema
------
Nodes:
    (:Concept {name, concept_class, confidence, aliases, source})

Relationships:
    (:Concept)-[:RELATES_TO {predicate, confidence, source}]->(:Concept)

All queries are parameterised — never use f-strings to embed user values.
"""
from __future__ import annotations


# ── Node/relationship labels ──────────────────────────────────────────────────

NODE_LABEL = "Concept"
REL_TYPE   = "RELATES_TO"


# ── Schema / indexes ──────────────────────────────────────────────────────────

CREATE_INDEXES = [
    f"CREATE INDEX concept_name IF NOT EXISTS FOR (c:{NODE_LABEL}) ON (c.name)",
    f"CREATE INDEX concept_class IF NOT EXISTS FOR (c:{NODE_LABEL}) ON (c.concept_class)",
    f"CREATE INDEX concept_confidence IF NOT EXISTS FOR (c:{NODE_LABEL}) ON (c.confidence)",
    f"CREATE INDEX rel_predicate IF NOT EXISTS FOR ()-[r:{REL_TYPE}]-() ON (r.predicate)",
    f"CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS "
    f"FOR (c:{NODE_LABEL}) ON EACH [c.name]",
]


# ── Upsert (MERGE) queries ────────────────────────────────────────────────────

MERGE_CONCEPT = f"""
MERGE (c:{NODE_LABEL} {{name: $name}})
SET   c.concept_class = $concept_class,
      c.confidence    = $confidence,
      c.aliases       = $aliases,
      c.source        = $source
"""

MERGE_RELATION = f"""
MATCH  (a:{NODE_LABEL} {{name: $subject}}),
       (b:{NODE_LABEL} {{name: $object}})
MERGE  (a)-[r:{REL_TYPE} {{predicate: $predicate}}]->(b)
SET    r.confidence = $confidence,
       r.source     = $source
"""


# ── Concept lookup ────────────────────────────────────────────────────────────

GET_CONCEPT_BY_NAME = f"""
MATCH (c:{NODE_LABEL})
WHERE toLower(c.name) = toLower($name)
RETURN c
LIMIT 1
"""

GET_CONCEPTS_BY_CLASS = f"""
MATCH (c:{NODE_LABEL} {{concept_class: $concept_class}})
WHERE c.confidence >= $min_conf
RETURN c
ORDER BY c.confidence DESC
LIMIT $limit
"""

FULLTEXT_SEARCH = f"""
CALL db.index.fulltext.queryNodes('concept_fulltext', $query)
YIELD node, score
RETURN node AS c, score
LIMIT $limit
"""

# Fallback when fulltext index unavailable
CONTAINS_SEARCH = f"""
MATCH (c:{NODE_LABEL})
WHERE toLower(c.name) CONTAINS toLower($token)
RETURN c
LIMIT $limit
"""


# ── Neighbour traversal ───────────────────────────────────────────────────────

OUTGOING_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]->(b:{NODE_LABEL})
WHERE toLower(a.name) = toLower($name)
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b
ORDER BY r.confidence DESC
LIMIT $limit
"""

INCOMING_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]->(b:{NODE_LABEL})
WHERE toLower(b.name) = toLower($name)
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b
ORDER BY r.confidence DESC
LIMIT $limit
"""

BOTH_NEIGHBOURS = f"""
MATCH (a:{NODE_LABEL})-[r:{REL_TYPE}]-(b:{NODE_LABEL})
WHERE (toLower(a.name) = toLower($name) OR toLower(b.name) = toLower($name))
  AND ($predicate IS NULL OR r.predicate = $predicate)
RETURN a, r, b
ORDER BY r.confidence DESC
LIMIT $limit
"""


# ── Shortest path ─────────────────────────────────────────────────────────────

SHORTEST_PATH = f"""
MATCH (src:{NODE_LABEL}), (tgt:{NODE_LABEL})
WHERE toLower(src.name) = toLower($source)
  AND toLower(tgt.name) = toLower($target)
MATCH path = shortestPath((src)-[:{REL_TYPE}*1..$max_hops]-(tgt))
RETURN path
LIMIT 1
"""


# ── Subgraph extraction ───────────────────────────────────────────────────────

SUBGRAPH_FROM_SEEDS = f"""
MATCH (seed:{NODE_LABEL})
WHERE seed.name IN $seed_names
CALL apoc.path.subgraphAll(seed, {{
    relationshipFilter: '{REL_TYPE}',
    maxLevel: $max_hops,
    limit: $max_nodes
}}) YIELD nodes, relationships
RETURN nodes, relationships
"""

# Fallback without APOC: multi-hop expansion (up to 3 hops hardcoded)
SUBGRAPH_2HOP = f"""
MATCH (seed:{NODE_LABEL})
WHERE seed.name IN $seed_names
MATCH (seed)-[r1:{REL_TYPE}*1..2]-(neighbor:{NODE_LABEL})
WITH COLLECT(DISTINCT seed) + COLLECT(DISTINCT neighbor) AS nodes
UNWIND nodes AS n
OPTIONAL MATCH (n)-[r:{REL_TYPE}]-(m:{NODE_LABEL})
WHERE m IN nodes
RETURN COLLECT(DISTINCT n) AS concepts,
       COLLECT(DISTINCT r) AS relations
"""


# ── Aggregate stats ───────────────────────────────────────────────────────────

CONCEPT_STATS = f"""
MATCH (c:{NODE_LABEL})
WITH c.concept_class AS cls, COUNT(*) AS cnt
RETURN cls, cnt
ORDER BY cnt DESC
"""

TOTAL_COUNTS = f"""
MATCH (c:{NODE_LABEL})
WITH COUNT(c) AS n_concepts
MATCH ()-[r:{REL_TYPE}]->()
WITH n_concepts, COUNT(r) AS n_relations
MATCH (c:{NODE_LABEL})
WITH n_concepts, n_relations, COLLECT(DISTINCT c.concept_class) AS classes
MATCH ()-[r:{REL_TYPE}]->()
RETURN n_concepts, n_relations,
       SIZE(classes) AS n_classes,
       COUNT(DISTINCT r.predicate) AS n_predicates
"""


# ── Utility ───────────────────────────────────────────────────────────────────

DELETE_ALL = f"MATCH (n:{NODE_LABEL}) DETACH DELETE n"

COUNT_CONCEPTS  = f"MATCH (c:{NODE_LABEL}) RETURN COUNT(c) AS n"
COUNT_RELATIONS = f"MATCH ()-[r:{REL_TYPE}]->() RETURN COUNT(r) AS n"
