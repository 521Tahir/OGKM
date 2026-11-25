#!/usr/bin/env bash
# exp037 - the confound eleven approaches never controlled: THE ADAPTER.
#
# Every ontology arm (onto, onto_c, v3, v4, v5, v6, v7, v8, onto_g) was generated
# with the `mixed` adapter, fine-tuned on a 50/50 curriculum of RAG prose and
# ontology-v1 bullets. Half its capacity learns "COPY the itinerary out of the
# context" - a great policy when the context holds an itinerary, a destructive one
# when it holds a fact list, which is the only regime ontology arms are scored in.
#
# `train_onto_v3.jsonl` did not exist until 2026-09-24: onto_v3 was ALWAYS served
# by weights that had never seen an onto_v3 context. Eleven approaches varied the
# test-time context while the training distribution stayed frozen. That is why
# onto_v6 could double gold-entity coverage (0.270 -> 0.512) and move nothing:
# more facts do not help a policy whose learned move is to copy prose.
#
# Arm-dedicated adapters, both trained so the fight stays fair:
#   onto_v3 -> 100% ontology contexts: learns to EXPAND facts into an itinerary
#   rag     -> 100% RAG contexts: the matched control
#
# GPUs are PINNED. The free-memory dispatcher put training on GPUs already busy
# with exp036 generation and the step time went to 8.8s (~5.6h for a 1B LoRA);
# pinning to idle devices removes the contention.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 EXP024_TAG=ded
LOG=experiments/exp037_dedicated_adapter
M="${MODEL:-llama3.2-1b}"
GPU_ONTO="${GPU_ONTO:-8}"
GPU_RAG="${GPU_RAG:-9}"

echo "[$(date '+%T')] TRAIN $M/onto_v3 on GPU $GPU_ONTO (pinned)"
python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm onto_v3 \
  --gpu "$GPU_ONTO" > "$LOG/train__${M}__onto_v3.log" 2>&1 &
P1=$!
echo "[$(date '+%T')] TRAIN $M/rag on GPU $GPU_RAG (pinned)"
python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm rag \
  --gpu "$GPU_RAG" > "$LOG/train__${M}__rag.log" 2>&1 &
P2=$!
wait $P1; echo "[$(date '+%T')] onto_v3 adapter done (rc=$?)"
wait $P2; echo "[$(date '+%T')] rag adapter done (rc=$?)"

for A in onto_v3 rag; do
  [ -f "/ceph_data/tahirahmad/ontology_exp024/adapters_ded/${M}__${A}/adapter_config.json" ] \
    || { echo "FATAL: ${M}__${A} adapter missing - aborting before generation"; exit 1; }
done
echo "=== training complete ($(date '+%F %T')) ==="

# ── stage 2: generate both arms, dedicated adapter, full test + bench ─────
for JOB in "onto_v3:bench:$GPU_ONTO" "rag:bench:$GPU_RAG" "onto_v3:test:$GPU_ONTO" "rag:test:$GPU_RAG"; do
  A="${JOB%%:*}"; rest="${JOB#*:}"; S="${rest%%:*}"; G="${rest##*:}"
  echo "[$(date '+%T')] GEN $M/$A $S on GPU $G (dedicated adapter)"
  python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
    --adapter "$A" --gpu "$G" --split "$S" \
    > "$LOG/gen__${M}__${A}__${S}.log" 2>&1 &
  eval "PID_${A}_${S}=\$!"
done
wait
echo "=== exp037 pilot complete ($(date '+%F %T')) ==="
