# exp024 — Simple RAG vs Ontology grounding (LoRA fine-tuned, full test split)

Answers one question: **given the same fine-tuned model, does grounding on the
Jeju tourism ontology beat plain dense RAG?**

## Design

| | |
|---|---|
| Data | `data/jeju_full/splits/` — train 36,847 / val 4,605 / **test 4,607** |
| Models | Llama-3.1-8B · DeepSeek-R1-8B · Qwen3-8B · Mistral-7B |
| Adaptation | LoRA r=16 α=32 on all attention+MLP projections, 1 epoch, bf16, max_len 2048 |
| Gold reference | the dataset's own `Output` field — **identical for both arms** |
| Metrics | BERT-F1 · METEOR · COMET · Key-token · Response Accuracy (LLM-judge) |
| Decoding | greedy, `max_new_tokens=512` |

### The two arms

| Arm | Context injected |
|---|---|
| `rag` | top-5 dense passages (`BAAI/bge-base-en-v1.5`, FAISS IP) over the 36,847 train answers |
| `onto` | ontology v1.2.0 facts + temporal layer, predicate-weighted, ≤15 facts |

Controls: `none` (fine-tuned, closed-book) and `base` (vanilla instruct model).

Everything except the context block is held constant — same base weights, same
LoRA recipe, same seed, same prompt template, same decoder, same gold, same judge.

### One adapter, two arms

Both arms are served by a **single** adapter per model, fine-tuned on a 50/50
mix of RAG and ontology contexts over the same 36,847 examples (seeded, so every
model sees the identical assignment). Two consequences:

- the compared weights are *identical*, so the measured difference is
  attributable to the retrieved context and not to two separately-converged
  adapters;
- neither arm is out-of-distribution at test time, because the model has been
  trained to consume both context formats.

## Two methodology corrections vs the Streamlit demo

1. **Gold reference.** `streamlit_app/ontology/retriever.py` builds its reference
   *from the same ontology facts it injects*, so a grounded answer is closer to
   gold by construction. exp024 scores both arms against the dataset `Output`.
2. **Ontology choice.** The sibling `Ontology_Project/` ontology is larger
   (8,362 concepts) but **17.9% of its relations mention non-Jeju places**
   (Bali, Cebu, Jakarta, Boracay, Seoul). This project's v1.2.0 is Jeju-clean
   (0.04%) and is what the ontology arm uses.

## Running it

```bash
# 1. contexts for every split and arm (~20 min, one GPU for the embedder)
python -m experiments.exp024_rag_vs_ontology.build_contexts --splits train val test

# 2. everything else, resumable, dispatched onto GPUs with live headroom
python -m experiments.exp024_rag_vs_ontology.run_all --stage all

# individual stages
python -m experiments.exp024_rag_vs_ontology.run_all --stage train
python -m experiments.exp024_rag_vs_ontology.run_all --stage generate score report

# end-to-end smoke test on a tiny slice, in its own namespace
bash experiments/exp024_rag_vs_ontology/pilot.sh 4
```

`EXP024_TAG=<name>` namespaces adapters/generations/scores/reports.
`EXP024_4BIT=1` switches training to QLoRA for a GPU with < 26 GB free.

Every stage skips work already on disk, so a killed run resumes where it stopped.

## Shared-machine etiquette

GPUs 5–9 run the GROUND project's AMBER jobs and another tenant intermittently
claims 0–3. `gpu.py` places work only on GPUs with enough free memory, so a run
never displaces someone else's job. Measured cost: **5.97 h per training run** on
an uncontended A6000 (1.71 samples/s), ~4× slower on a shared one.

## Outputs

```
outputs/contexts/{split}_{arm}.jsonl     precomputed context blocks
outputs/generations/{split}__{model}__{arm}.jsonl
outputs/scores/{split}__{model}__{arm}.jsonl
outputs/reports/results.{md,tex,json}    main table, deltas, paired tests
```

Significance is item-level paired t-test + Wilcoxon signed-rank over the 4,607
shared test questions (`src/evaluation/stats_tests.py`).


---

## Why the ontology arm lost, and what fixed it

The v1 result was a clean loss for the ontology on all five metrics, every model,
all significant. Diagnosing it on the generations already on disk (4,589 scored
items) located the defect precisely — **it was the retriever, not the ontology.**

### Gold-answer content actually present in each arm's context

| Arm | all key-tokens | real entities only¹ | zero-coverage items | budget used |
|---|---|---|---|---|
| `rag` | 0.726 | 0.552 | 1.1% | 89% |
| `onto` (v1) | 0.093 | 0.141 | 52.2% | 30% |
| `onto_v2` | 0.091 | — | 56.4% | 68% |
| `onto_v3` | **0.191** | **0.275** | **25.2%** | 99% |
| *ontology ceiling* | *0.677* | — | *0.9%* | — |

¹ excluding itinerary boilerplate — see the caveat below.

The ceiling row is the point: ontology v1.2.0 **contains** 67.7% of the gold
entities. v1's substring scan surfaced 9.3% of them and filled under a third of
the prompt budget.

### `onto_v2` — dense retrieval over triples (negative result)

Verbalise all 9,208 relations, embed with the same `bge-base` the RAG arm uses,
retrieve by cosine. Coverage did **not** move: 0.091 vs 0.093.

The failure is instructive. For *"plan a 3-day trip, major natural attractions,
traditional food"* it returns the generic hub node — `Jeju Island best for
Sightseeing`, `Jeju Island suitable for Tourist` — while the gold answer names
Hallim Park, Cheonjiyeon Waterfall and Olle Trail, all present in the ontology.
**A generic question underdetermines which specific entities the answer needs.**
RAG escapes this only because its index units are whole answers, which already
carry a plausible entity set; an atomic triple never can. Kept in the codebase,
not carried into the results table — its coverage never justified generation.

### `onto_v3` — entity cards + intent→class routing (adopted)

1. **Entity cards.** One indexed unit per entity carrying all its relations, so
   the retrieval unit resembles a passage rather than a 5-word triple.
2. **Intent → class routing.** Map question phrasing to ontology classes and pull
   the highest-degree instances: *"natural attractions"* → `NaturalAttraction`
   (Seongsan Ilchulbong, Cheonjiyeon Waterfall, Manjanggul Cave), *"traditional
   food"* → `Cuisine` (omegi-tteok, Hallabong citrus). This supplies the
   specifics a generic question implies but never names — the thing v1 and v2
   could not reach.

Result: coverage 0.093 → 0.191 (0.141 → 0.275 on real entities), zero-coverage
items 52.2% → 25.2%, budget utilisation 30% → 99%. Still below RAG's 0.552, but
it doubles the ontology's grounding without touching the ontology itself.

Runs on the existing `mixed` adapter — same header wording, so no retraining.

## Caveat on the metrics

**36.4% of the Key-token metric is itinerary boilerplate.** The most frequent
"gold entities" are `Morning` (2,598), `Afternoon` (2,432), `Visit`, `Evening`,
`Explore`, `Arrival` — section headers in the gold itineraries. RAG reproduces
them for free by copying another itinerary's structure; an ontology cannot
retrieve "Morning" as a fact. Key-token, and likely METEOR (unigram overlap,
where the gap is widest at −15%), substantially measure **format mimicry rather
than factual grounding**.

None of the five metrics tests faithfulness, which is the ontology's actual
claim. `src/evaluation/hallucination_eval.py` and `ontology_metrics.py` are in
the repo and run over generations already on disk, with no regeneration. Report
the overlap table either way — but it is not, on its own, evidence about
grounding.
