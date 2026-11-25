# Business-plan deck and demo film

Both artefacts answer the brief in `Ontology-Comment19 October.txt`: say why the
project needs an ontology, and show it as a before-and-after rather than as
architecture. Both are generated, so a re-run of the pipeline moves them instead
of leaving them stale.

| file | what it is |
|---|---|
| `Ontology_Business_Plan.pptx` | 32 slides, 16:9 |
| `Ontology_Demo.mp4` | narrated demo film, 1920x1080 |
| `demo_evidence.json` | every number and quoted answer the two are allowed to use |
| `demo_policy_before_after__*.json` | the policy edit run at two sizes: 16 questions x 2 ontology versions, verbatim |
| `demo_live_edit.json` | the earlier live ontology-edit run: contexts, answers, timings, adapter digests |
| `deck_fig_*.png` | the two charts drawn for a business audience |
| `_video/` | narration wavs, per-scene stills, render log (intermediate) |

## Rebuilding

```bash
python -m wbs.task2_formalization.extract_policy_constraints --write 1.6.0   # the policy layer
python -m wbs.task2_formalization.validate_ontology --version 1.6.0 --out-tag __1.6.0
EXP024_TAG=ded python -m wbs.task5_demo.demo_policy_before_after --gpu 5 --model llama3.2-3b
python -m wbs.task4_evaluation.repair_policy        # the R19 cells the deck reads
python -m wbs.task4_evaluation.verifier_across_models --split test   # the six-model check
python -m wbs.task5_demo.demo_live_edit --gpu 5     # optional: re-runs the live edit
python -m wbs.task5_demo.build_demo_evidence        # collects the numbers
python -m wbs.task5_demo.make_deck_figures          # the two charts
python -m wbs.task5_demo.make_slides                # the deck
python -m wbs.task5_demo.check_deck                 # geometry and overflow
python -m wbs.task5_demo.make_demo_video            # the film
python -m wbs.task5_demo.make_demo_video --stills   # one PNG per scene, for review
```

`build_demo_evidence.py` reads only `wbs/deliverables/`; the slide footers name
the file each figure came from. The before/after answers are the stored
generations: where a slide cannot hold a whole answer it quotes the window
containing the claim, marked with ellipses, and `tests/test_rates.py` checks
that the window is a substring of the stored answer and contains the claim it
is shown to prove.

## The corrections the demo shows

The brief asks for the demo as a before-and-after: a confusing or incorrect
answer becoming a clear, correct one because the ontology was changed. There
are two kinds here, and the deck labels which kind each slide is.

**Edited live** — the fact was not in the graph, we extracted it from the
corpus, wrote it into v1.6.0 beside the published version, and asked again.
Same adapter loaded once, greedy decoding, same order, only the ontology
version differing; the adapter digest `0012c1e8e69e68d2`
is recomputed after the run and compared.

| # | the rule added | before | after |
|---|---|---|---|
| 1 | `Jeju Folk Village requires Shoes Removed` | does not state it | states it, backed by an approved fact |
| 2 | `Jungmun Saekdal Beach requires Reef-safe Sunscreen` | does not state it | states it, backed by an approved fact |
| 3 | `Dongmun Market requires Cash Payment` | states it, unbacked | states it, backed by an approved fact |
| 4 | `Gwaneumsa Temple requires Modest Dress` | states it, unbacked | states it, backed by an approved fact |

**Never there to begin with** — the stored cases, where the grounded arm
refuses to invent what the graph does not hold.

| # | item | what the base model said | what the graph says |
|---|---|---|---|
| 5 | `B0190` | drives the traveller from Jeju to Busan and Gyeongju | no entity, no edge: a mainland city is not reachable from the island by rental car |
| 6 | `B0276` | "the Jeju Subway ... with 10 lines" | `subway is_a transportmode` and no edge to Jeju at all |
| 7 | live | — | `avoid_during` added in v1.5.0; the same question, the same weights, and the safety claim is now backed |

The live runs use one non-default setting, `safety_first=True`, which orders
constraint predicates first in an entity card so a late-asserted fact can reach
the model at all; the published arms were run without it and are unaffected.
R18 in `docs/risk_management_plan.md` records why that flag has to exist.

## How the film is cut

The brief asks for the demo as a before-and-after, and gives the four parts it
wants per correction. Every correction scene in the film is cut to those four
parts, in that order:

| part | what is on screen |
|---|---|
| **The problem** | the customer's question, then the answer the system gave, quoted |
| **The ontology fix** | the structure that was added, drawn as nodes and edges — what the graph already held in grey, the new edges in amber, animated in |
| **The result** | the answer after the edit, with the words the new fact bought highlighted |
| **Why it improved** | one paragraph, beside the corpus sentence the fact was read from |

Two of those scenes are the brief's **own** examples — the Flagship Software V1
incompatibility and the contractor remote-work boundary — drawn exactly as the
brief writes them, with the ontology structure each one needs: the
incompatibility edge plus the successor version and the patch, and the staff
taxonomy with the disjointness and the attribute on each side. They are
labelled on screen and in the narration as illustrations of the shape, **not**
runs of this system, because they are not: no model produced those two answers.
They come first so the two shapes are clear, and the real runs that follow are
the same two shapes on this project's own graph.

## The policy edit, counted (R21, R22)

`extract_policy_constraints.py` reads 46,059 corpus records
for the two shapes the brief's own examples use — a hard requirement and a hard
restriction — and keeps **22 triples** over
16 places at support >= 3 within
120 characters, each carrying the sentence it was
read from. Writing them as v1.6.0 costs 0 GPU-hours and
0 validation errors, with SHACL conforming.

The probe set is 10 questions the edit speaks to and
6 controls it does not:

| | before v1.5.0 | after v1.6.0 |
|---|--:|--:|
| the rule reached the model's context | 0/10 | 10/10 |
| the answer states the rule | 6/10 | 8/10 |
| control answers changed at all | — | 4/6 (75% word overlap) |

Both of the uncomfortable numbers are on the slide and in the narration. The
fact reached the prompt in 10 of
10 and was stated in 8 — putting a fact in
front of a model is not the same as the model using it (**R22**). And an
additive edit reworded 4 of 6
answers it had nothing to say about, because new concepts become new entity
cards and the retrieval index shifts under questions the edit never touched
(**R21**).

Against that, what the edit cost: 75 facts over 2 edits,
41 seconds of CPU to re-index, 0
GPU-hours. The 15 LoRA adapters this project trained cost
135 GPU-hours between them, and every one would
have to be paid again to teach the same six rules by fine-tuning.

## The deployed configuration the deck describes (R19)

Both artefacts describe the gated validation layer with **surgical repair**: a
flagged answer loses the clause that carries the violation and keeps every
other byte, rather than being handed back to the model to rewrite. That is what
`repair_policy.md` measures and what `demo_evidence.json` → `gated` now reads
from; `demo_evidence.json` → `repair` keeps the previous re-prompting cell
beside it, so the deck shows what changed instead of quietly restating a better
number.

| | re-prompting repair | surgical repair |
|---|---|---|
| token F1 vs Simple RAG, 1B / 3B / 8B | -2.37 / -1.75 / -3.44 | -0.73 / -0.12 / -0.86 |
| ungrounded, published detector (8B) | 1.26% | 0.00% |
| ungrounded, independent strict rule (8B) | 2.89% | 2.15% (Simple RAG 5.66%) |
| cost per repaired answer | a second generation | ~1.3 ms of CPU |

The deck and the film also carry the cross-model check that R19 made cheap:
the verifier is now ordinary post-processing, so it was run unchanged over
4,607 stored Simple RAG answers from each of six models - Mistral-7B, Qwen3-8B
and DeepSeek-R1-8B among them, none of which this pipeline was tuned for - and
the strict-rule ungrounded rate more than halves for all six
(`task4_evaluation/verifier_across_models__test.md`).

The 0.00% is constructional - the pass removes exactly the spans the published
detector counts - and both the deck and the film say so on the slide and in the
narration, with the strict-rule figure beside it. R20 in
`docs/risk_management_plan.md` records why that matters and what would close it.

## What the two say, and what they refuse to say

The case is maintenance, safety and small-model economics: a knowledge
correction costs an ontology edit rather than a fine-tune; a 1B model with the
graph is ahead of an 8B model without it; the deployed configuration cuts
unsupported claims below the retrieval baseline at all three sizes.

Neither artefact claims the ontology arms beat plain retrieval on reference
overlap. They do not, at any size - though with surgical repair the gap is
under one point everywhere and, at llama3.2-3b, no longer distinguishable from
zero on any metric. Both the deck and the film say so with the number and the
reason. A test pins that slide in place.
