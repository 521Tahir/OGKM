#!/usr/bin/env bash
# exp040 - onto_v10: COVERAGE + COMPOSITION with a dedicated adapter (1B).
#
# Coverage is exhausted. The ontology holds 0.599 token-weighted coverage of test
# gold entities and onto_v6 already retrieves 0.512 - 85% of everything available.
# exp038 proved more of it does nothing: pairing that coverage with a dedicated
# adapter moved free-form BERT-F1 by +0.0003.
#
# What the graph holds and no arm has used as primary content is COMPOSITION:
# 10,124 PMI-filtered `co_occurs_with` edges mined from splits/train.jsonl - the
# same corpus the RAG arm indexes. A gold answer IS a composition; a list of facts
# about individual places cannot express one.
#
# onto_v7 attempted this and discarded 95% of the signal (10,124 edges -> 527)
# because its place filter demanded a gazetteer hit or a place-name suffix.
# Filtering on the ontology's own concept_class instead yields a clean 419-edge /
# 143-anchor graph, and exposes that `Region` was wrongly excluded - Andeok,
# Seongsan, Hallim and Aewol are real Jeju regions.
#
# onto_v10 = popularity facts (55% of the 3,000-char budget) + co-occurrence
# pairings (45%), matched budget, served by an adapter trained on those contexts.
# Control: rag @ dedicated from exp037.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 EXP024_TAG=ded PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/exp041_onto_blend
AD=/ceph_data/tahirahmad/ontology_exp024/adapters_ded
SD=experiments/exp024_rag_vs_ontology/outputs/scores_ded
M=llama3.2-1b

pick() { local need=$1; shift; local busy=" $* "; local g
  for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --need "$need"); do
    [[ "$busy" == *" $g "* ]] || { echo "$g"; return; }
  done; }

if [ ! -f "$AD/${M}__onto_v10/adapter_config.json" ]; then
  G=""; while [ -z "$G" ]; do G=$(pick 34000); [ -z "$G" ] && sleep 120; done
  echo "[$(date '+%T')] TRAIN $M/onto_v10 on GPU $G"
  python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm onto_v10 \
    --gpu "$G" > "$LOG/train__${M}__onto_v10.log" 2>&1
  [ -f "$AD/${M}__onto_v10/adapter_config.json" ] || { echo "FATAL adapter missing"; exit 1; }
fi
echo "[$(date '+%T')] adapter ready"

GPUS=(); PIDS=()
for S in bench test; do
  G=""; while [ -z "$G" ]; do G=$(pick 30000 "${GPUS[@]:-}"); [ -z "$G" ] && sleep 120; done
  echo "[$(date '+%T')] GEN $M/onto_v10 $S on GPU $G"
  python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm onto_v10 \
    --adapter onto_v10 --gpu "$G" --split "$S" > "$LOG/gen__${S}.log" 2>&1 &
  GPUS+=("$G"); PIDS+=("$!"); sleep 90
done
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || echo "[$(date '+%T')] WARN gen GPU ${GPUS[$i]} rc=$?"; done

for S in bench test; do
  [ -f "$SD/${S}__${M}__onto_v10.jsonl" ] && continue
  for attempt in 1 2 3; do
    G=""; while [ -z "$G" ]; do G=$(pick 34000); [ -z "$G" ] && sleep 120; done
    echo "[$(date '+%T')] SCORE $S on GPU $G (attempt $attempt)"
    python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$G" --split "$S" \
      --cells "${M}__onto_v10" >> "$LOG/score__${S}.log" 2>&1
    [ -f "$SD/${S}__${M}__onto_v10.jsonl" ] && break
    sleep 120
  done
done
echo "=== exp041 1B complete ($(date '+%F %T')) ==="
