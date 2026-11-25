# Jeju Tourism Intelligence

An interactive comparison of **Retrieval-Augmented Generation** and **Ontology-Grounded
Question Answering** for complex Jeju tourism questions.

Everything in this folder is generated from recorded runs of the real pipeline.
No score, prompt, retrieval result, ontology fact or answer was written by hand.

---

## Headline results

### Three-query qualitative demonstration

| Query | Simple RAG | Ontology | Winner |
|---|---|---|---|
| Q1 — Geographic + multi-constraint recommendation | 83.3 | 25.0 | RAG |
| Q2 — Multi-hop cultural and geographic classification | 64.6 | 88.8 | **Ontology** |
| Q3 — Accessibility-constrained planning (abstention stress test) | 70.6 | 10.0 | RAG |
| **Mean** | **72.8** | **41.3** | — |

Ontology wins Q2, the factual-classification question, with a perfect factual score and
zero unsupported claims. Three queries is a behavioural demonstration, not statistical
evidence.

### Full held-out benchmark

| Task | Winner |
|---|---|
| Free-form generation, 4,607 items x 6 models | **Simple RAG**, all five metrics, every model |
| Factual benchmark, 200 questions | **Ontology grounding** wins all five metrics on Llama-3.2-1B |

Both are reported. Reporting only one would misrepresent the work.

---

## What is in here

```
Jeju_Tourism_Research_Demo/
├── application/
│   ├── frontend/index.html          the interactive page (open directly, no server)
│   └── backend/server.py            optional FastAPI live inference
├── ontology/
│   ├── original/                    ontology v1.3.0 source files, copied verbatim
│   └── graph_exports/               nodes.csv, edges.csv, class_tree.json
├── experiments/
│   ├── query_1..3/trace.json        full per-query execution traces
│   └── evaluation/                  scores + the data-contract test
├── reports/
│   ├── accuracy_improvement_report.md
│   ├── qualitative_analysis_report.md
│   ├── evaluation_methodology.md
│   ├── ontology_structure_report.md
│   └── rag_vs_ontology_comparison.csv
├── demo/
│   ├── interactive_application/index.html
│   ├── recorded_traces/bundle.json  every figure the app and video read
│   └── screenshots/
└── video/
    ├── Jeju_Tourism_Full_HD_Demo.mp4
    ├── prompts/                     full prompt + answer text for all 6 cells
    ├── narration_script.md
    └── storyboard.md
```

---

## Running it

### The interactive application — no install

```bash
xdg-open Jeju_Tourism_Research_Demo/application/frontend/index.html
```

It is a single self-contained page. The data bundle is embedded, so it opens from
`file://` with no server and no network except the Cytoscape CDN script.

Seven views: Home · Live Query Comparison · RAG Pipeline · Ontology Pipeline ·
**Ontology Explorer** · Qualitative Analysis · Evaluation Dashboard.

**Ontology Explorer** loads the real graph — 1,585 concepts,
8,324 relations, 32 classes. Click a node to
expand its relations; double-click to expand another level; search any of the
1,585 concepts; filter by predicate or class; export PNG. Four modes:

- **A** class hierarchy (31 subclass edges)
- **B** instance network, progressive expansion
- **C** query subgraph — exactly the entities retrieved for a question
- **D** reasoning path — relations present verbatim in the prompt the model received

### Live inference (optional)

```bash
pip install fastapi uvicorn
python Jeju_Tourism_Research_Demo/application/backend/server.py --gpu 1 --model llama3.1-8b
curl -s localhost:8077/health
curl -s localhost:8077/ask -H 'content-type: application/json' \
     -d '{"question":"Which Jeju beaches are near Hallim?"}' | python -m json.tool
```

Loads the same base model and the same arm-dedicated adapters used for the recorded traces.

### The video

`video/Jeju_Tourism_Full_HD_Demo.mp4` — 1920x1080, H.264, 30 fps, 6 min 50 s.

---

## Reproducing from scratch

```bash
# 1. capture real traces for the three queries (needs a GPU)
python -m experiments.exp041_demo_traces.capture_traces --gpu 1 --model llama3.1-8b

# 2. score them with the auditable rubric (CPU)
python -m experiments.exp041_demo_traces.score_queries

# 3. assemble the data bundle from ontology + traces + benchmark cells
python -m experiments.exp041_demo_traces.build_bundle

# 4. build the single-page app
python -m experiments.exp041_demo_traces.build_app

# 5. generate every report
python -m experiments.exp041_demo_traces.build_reports

# 6. render the Full HD video
python -m experiments.exp041_demo_traces.render_video

# 7. verify the app's data contract
python Jeju_Tourism_Research_Demo/experiments/evaluation/test_bundle_contract.py
```

---

## Experimental configuration

| Factor | Setting |
|---|---|
| Base model | Llama-3.1-8B (`meta-llama/Llama-3.1-8B-Instruct`) |
| Adapters | arm-dedicated (exp037) — one per arm, so neither is advantaged |
| Decoding | greedy, max_new_tokens=512 |
| Context budget | 3,000 characters, identical for both arms |
| Retriever | `BAAI/bge-base-en-v1.5` + FAISS IndexFlatIP |
| Ontology | v1.3.0 — 1,585 concepts, 8,324 relations |
| Ground truth | 69-place gazetteer with coordinates |
| Traces captured | 2026-10-08 01:22:37 |

Only the retrieval strategy differs, because that is the method under comparison.

---

## Honest limitations

1. **The video is not a screen recording.** This machine has no browser; Chromium needs
   `libatk-1.0`, `libgbm`, `libxkbcommon` and seven more system libraries that require
   root, and conda-forge is unreachable from this host. Every frame is rendered directly
   from `demo/recorded_traces/bundle.json` — the same file the application reads — so the
   content is real, but it is a rendering, not a capture. The method card at the end of the
   video states this.
2. **The app could not be click-tested here** for the same reason. Instead
   `experiments/evaluation/test_bundle_contract.py` asserts every data path the JavaScript
   dereferences, and the page's script was parsed for syntax errors. Both pass.
3. **Three queries demonstrate behaviour, not general superiority.** The full benchmark is
   reported alongside and disagrees with the three-query ranking.
4. **Q3 has no ground truth by design.** Zero of 69 gazetteer entries carry
   accessibility data, so it is an abstention stress test. Both systems fail it.
5. **The hallucination component is a ratio**, so a verbose answer dilutes a fixed number of
   unsupported claims. Raw counts are reported next to it.
6. **Ontology grounding on free-form generation does not beat retrieval**, and seventeen
   attempts did not change that. The measured reason is in
   `reports/accuracy_improvement_report.md`.
