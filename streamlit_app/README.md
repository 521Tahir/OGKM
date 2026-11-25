# Ontology-Augmented LLM Benchmark (Streamlit)

Side-by-side comparison of vanilla LLMs (**Case A — no ontology**) vs
ontology-enhanced LLMs (**Case B — with ontology**) for tourism questions.
Four models answer **in parallel** across GPUs; metrics run on a dedicated GPU.

## Run
```bash
bash streamlit_app/run_app.sh           # http://localhost:8501
bash streamlit_app/run_app.sh 8600      # custom port
```
Models load once on the first **Generate** (~1–2 min), then stay warm.

## Layout
- **Left = Case A (red, no ontology)**, **Right = Case B (green, with ontology)** — always side by side, never tabs.
- Each side shows all 4 models (Llama-3.1-8B, DeepSeek-R1-8B, Qwen3-8B, Mistral-7B) as cards with answer, metrics, latency, tokens, and (Case B) an expandable **ontology evidence** section.
- Below: **summary table**, **composite-score leaderboard**, **Plotly charts** (quality bar, latency, radar), and **export** (CSV / JSON / report).

## GPU plan (application.txt)
| GPU | Role |
|----|------|
| 5 | Llama-3.1-8B |
| 6 | DeepSeek-R1-8B |
| 7 | Qwen3-8B |
| 8 | Mistral-7B |
| 9 | metrics (BERT-F1, COMET) + main process |

Each model runs in its own GPU-pinned worker process; all jobs dispatch together
so total latency ≈ `max(model_times)`, not the sum (validated: 8 answers in ~3.4 s).

## Metrics
BERT-F1, METEOR, COMET, Key-token, LLM-Judge, TTFT, TPOT. The retrieved **ontology
fact is used as the gold reference**, so the metrics measure how close each answer
is to the authoritative ontology — which is exactly what Case B is grounded in.

## Structure
```
app.py            config.py          run_app.sh
ui/    layout(cards,charts,styles)
models/ llama deepseek qwen mistral
ontology/ retriever temporal_retriever neo4j_client
evaluation/ metrics bertscore comet meteor llm_judge keytoken
utils/ model_server async_runner gpu_manager logging_utils cache
results/ generated_answers/ metrics/ logs/
```
