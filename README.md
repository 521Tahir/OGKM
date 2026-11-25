# Jeju Tourism Ontology + LLM Knowledge Management System



---

## Overview

An **ontology-grounded RAG system** for Jeju Island tourism that:
- Externalises LLM knowledge into a versioned, validated ontology (8,348 concepts, 38,868 relations)
- Reduces hallucinations via constraint-based generation (V5: 8.5% hallucination vs 100% baseline)
- Grounds every answer in an explainable ontology trace
- Enforces Jeju-only responses — non-Jeju queries are blocked before the LLM is called
- Stores the full knowledge graph in **Neo4j** for visual exploration

---

## Quick Start

```bash
# 1. Clone and enter project
cd /home/tahirahmad/Ontology_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load environment (HF_TOKEN, Neo4j creds)
set -a && source .env && set +a

# 4. Start Neo4j (knowledge graph browser)
bash scripts/start_neo4j.sh start
# → open http://localhost:7474  (neo4j / jeju1234)

# 5. Ask a Jeju question (single query)
python scripts/ontology_rules_qa.py query "What hiking trails are on Hallasan?"

# 6. Interactive chat mode
python scripts/ontology_rules_qa.py interactive

# 7. Run all tests
python -m pytest tests/ -q
```

---

## Jeju Guard — Non-Jeju Queries Are Blocked

```
ALLOWED  → "What are the best beaches in Jeju?"
ALLOWED  → "Tell me about Hallasan trails"
ALLOWED  → "What is the weather like?"       (no location = pass)
BLOCKED  → "Hotels in Seoul?"                → redirected to Jeju
BLOCKED  → "Best restaurants in Paris?"      → redirected to Jeju
BLOCKED  → "Things to do in Tokyo?"          → redirected to Jeju
```

Blocking happens at zero LLM cost — `src/utils/jeju_guard.py` runs in <1 ms.

---



---

## Neo4j Knowledge Graph

```bash
bash scripts/start_neo4j.sh start     # start
bash scripts/start_neo4j.sh stop      # stop
bash scripts/start_neo4j.sh status    # check
```

**Browser:** http://localhost:7474 — user: `neo4j` / password: `jeju1234`

```cypher
-- Schema overview
CALL db.schema.visualization()

-- Hallasan subgraph
MATCH (n:Concept {name:"Hallasan"})-[r:RELATES_TO]-(m) RETURN n,r,m LIMIT 50

-- Class distribution
MATCH (c:Concept) RETURN c.concept_class AS Class, COUNT(*) AS Count ORDER BY Count DESC
```

**Graph stats:** 8,348 concepts · 16,742 relations · 31 classes · 23 predicates

---

## Project Structure

```
Ontology_Project/
├── src/
│   ├── api/              FastAPI service (9 endpoints, rate limiting, Jeju guard)
│   ├── data/             Loader, cleaner, splitter (61,963 records)
│   ├── evaluation/       Metrics, benchmark, hallucination eval, LLM judge
│   ├── generation/       ConstrainedGenerator, ResponseValidator, UncertaintyHandler
│   ├── graph/            TourismGraphStore, Neo4jClient, SubgraphRanker
│   ├── llm/              LLMClient (Llama 3.1-8B 4-bit, free local)
│   ├── ontology/         Extractor, SHACL validator, versioning, OWL exporter
│   ├── retrieval/        HybridRetriever, SubgraphRetriever (3-hop + keyword seeding)
│   └── utils/            JejuGuard, logging, IO
├── experiments/          exp001–exp019 (all run, reports saved)
├── scripts/
│   ├── ontology_rules_qa.py    Main QA CLI
│   ├── generate_figures.py     All paper figures
│   ├── failure_analysis.py     Failure categorisation
│   ├── ingest_to_neo4j.py      Load ontology into Neo4j
│   └── start_neo4j.sh          Start/stop Neo4j
├── artifacts/
│   ├── ontologies/1.0.0/       8,348 concepts, 38,868 relations, v1.0.0 TTL
│   ├── figures/                25 publication figures (300 DPI)
│   └── reports/                35 JSON experiment reports
├── docs/                       Schema, API docs, case studies, runbook
├── tests/                      185 tests (all passing)
└── configs/                    API config, runtime settings
```

---

## Ontology v1.0.0

| Property | Value |
|----------|-------|
| Concepts | 8,348 |
| Relations | 38,868 |
| Classes | 31 |
| Relation types | 23 |
| SHACL violations | 0 |
| OWL export | `artifacts/ontologies/tourism_v1.0.0.ttl` |
| JSON-LD export | `artifacts/ontologies/tourism_v1.0.0.jsonld` |

**Top classes:** Cuisine (1,152) · Attraction (949) · Activity (860) · Accommodation (636) · Destination (621)

---

## API Service

```bash
bash scripts/start_api.sh              # start on port 8000
LOAD_LLM=1 bash scripts/start_api.sh   # with Llama 3.1-8B

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What beaches are near Seogwipo?","mode":"constrained"}'
```

Endpoints: `/query` `/health` `/ontology/stats` `/ontology/concept/{name}`
`/ontology/search` `/ontology/subgraph` `/rules` `/metrics` `/explain`

---

## LLM — Free Local Only (No Paid API)

| Setting | Value |
|---------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| GPU | Set `CUDA_VISIBLE_DEVICES` in `.env` |
| HF token | `HF_TOKEN` in `.env` (free, for model download) |
| Weather | Open-Meteo API (free, no key needed) |

---

## Run Experiments

```bash
# Main 5-variant comparison (Table 3)
python -m experiments.exp018_main_results.run_main_results

# Failure analysis
python scripts/failure_analysis.py

# Robustness evaluation (100 items × 5 variants)
python -m experiments.exp019_robustness.run_robustness

# Generate all paper figures
python scripts/generate_figures.py

# Scaling runner (checkpoint/resume)
python -m experiments.exp007_scaling_100k.run_scaling --dry-run
```

---

## Research Questions & Answers

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | Can LLM CoT prompting produce quality tourism ontologies? | Yes — 8,348 concepts, 0 SHACL violations |
| RQ2 | Does ontology validation improve quality? | Yes — v0.3→v1.0: 0 violations, 31 OWL classes |
| RQ3 | Does ontology-RAG outperform doc-RAG? | Yes — +1.8% precision, +1.4% grounding (V3 vs V2) |
| RQ4 | Do constraints reduce hallucinations? | Yes — 91.5pp reduction (V5 vs V1) |
| RQ5 | Does the system provide explainable answers? | Yes — every V5 answer has grounding score + evidence trace |

---

## Tests

```bash
python -m pytest tests/ -q          # 185 tests, ~3s
python -m pytest tests/test_api.py  # API tests (includes Jeju guard)
python -m pytest tests/test_retrieval.py
python -m pytest tests/test_generation.py
```

---

## Reproduce All Results

```bash
bash scripts/reproduce_all.sh
```

See `docs/full_data_runbook.md` for step-by-step instructions.
