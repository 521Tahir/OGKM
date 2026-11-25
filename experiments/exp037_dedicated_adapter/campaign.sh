#!/usr/bin/env bash
# exp037 campaign - arm-dedicated adapters for every remaining model.
#
# The 1B pilot showed the shared `mixed` adapter is biased against the ontology:
# a dedicated adapter is worth nothing to the RAG arm (0.8913 -> 0.8917 BERT-F1)
# and lifts onto_v3 on 5/5 metrics, significant on 4/5, closing 13-23% of the gap.
# Each model gets BOTH a dedicated onto_v3 adapter and a dedicated rag control so
# the comparison stays matched.
#
# Two bugs in the first version of this script, both fixed here:
#   1. the busy-GPU test compared GPU ids against a list of PIDs, so it never
#      matched and both trainings of a pair landed on the SAME GPU;
#   2. the pid map was keyed by GPU, so that collision overwrote the first pid -
#      the script then waited on only one job, orphaned the other, and reported it
#      FATAL while it was still training.
# GPU/pid pairs are now kept in parallel indexed arrays and every pid is waited on.
#
# Also: an adapter already on disk is skipped, and an adapter whose training log is
# still held open by a live process is WAITED FOR rather than retrained.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 EXP024_TAG=ded PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/exp037_dedicated_adapter
AD=/ceph_data/tahirahmad/ontology_exp024/adapters_ded
SD=experiments/exp024_rag_vs_ontology/outputs/scores_ded
GD=experiments/exp024_rag_vs_ontology/outputs/generations_ded

MODELS=(llama3.2-3b mistral-7b llama3.1-8b qwen3-8b deepseek-r1-8b)

# a free GPU that is not already claimed by this script's in-flight jobs
pick() {
  local need=$1; shift
  local busy=" $* "
  local g
  for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --need "$need"); do
    [[ "$busy" == *" $g "* ]] || { echo "$g"; return; }
  done
}

have_adapter() { [ -f "$AD/$1__$2/adapter_config.json" ]; }
inflight()     { [ -n "$(lsof -t "$LOG/train__$1__$2.log" 2>/dev/null)" ]; }

for M in "${MODELS[@]}"; do
  # ── train the pair, one GPU each ──────────────────────────────────────
  GPUS=(); PIDS=()
  for A in onto_v3 rag; do
    if have_adapter "$M" "$A"; then echo "[$(date '+%T')] $M/$A adapter present - skip"; continue; fi
    if inflight "$M" "$A"; then
      echo "[$(date '+%T')] $M/$A already training elsewhere - waiting for it"
      while inflight "$M" "$A"; do sleep 120; done
      continue
    fi
    G=""
    while [ -z "$G" ]; do
      G=$(pick 34000 "${GPUS[@]:-}")
      [ -z "$G" ] && sleep 120
    done
    echo "[$(date '+%T')] TRAIN $M/$A on GPU $G"
    python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm "$A" \
      --gpu "$G" > "$LOG/train__${M}__${A}.log" 2>&1 &
    GPUS+=("$G"); PIDS+=("$!")
    sleep 90
  done
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[$(date '+%T')] WARN $M train on GPU ${GPUS[$i]} rc=$?"
  done

  ok=1
  for A in onto_v3 rag; do
    have_adapter "$M" "$A" || { echo "[$(date '+%T')] FATAL $M/$A adapter missing - skipping $M"; ok=0; }
  done
  [ $ok -eq 0 ] && continue
  echo "[$(date '+%T')] $M adapters ready"

  # ── generate: bench pair, then test pair, distinct GPUs within a pair ──
  for PAIRSPLIT in bench test; do
    GPUS=(); PIDS=()
    for A in onto_v3 rag; do
      G=""
      while [ -z "$G" ]; do
        G=$(pick 30000 "${GPUS[@]:-}")
        [ -z "$G" ] && sleep 120
      done
      echo "[$(date '+%T')] GEN $M/$A $PAIRSPLIT on GPU $G"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
        --adapter "$A" --gpu "$G" --split "$PAIRSPLIT" \
        > "$LOG/gen__${M}__${A}__${PAIRSPLIT}.log" 2>&1 &
      GPUS+=("$G"); PIDS+=("$!")
      sleep 90
    done
    for i in "${!PIDS[@]}"; do
      wait "${PIDS[$i]}" || echo "[$(date '+%T')] WARN $M gen $PAIRSPLIT on GPU ${GPUS[$i]} rc=$?"
    done
  done

  # ── score, sequential, gated, retried ─────────────────────────────────
  for S in bench test; do
    if [ -f "$SD/${S}__${M}__rag.jsonl" ] && [ -f "$SD/${S}__${M}__onto_v3.jsonl" ]; then continue; fi
    for attempt in 1 2 3; do
      G=""
      while [ -z "$G" ]; do
        G=$(pick 34000)
        [ -z "$G" ] && sleep 120
      done
      echo "[$(date '+%T')] SCORE $M $S on GPU $G (attempt $attempt)"
      python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$G" --split "$S" \
        --cells "${M}__onto_v3" "${M}__rag" >> "$LOG/score__${M}__${S}.log" 2>&1
      { [ -f "$SD/${S}__${M}__rag.jsonl" ] && [ -f "$SD/${S}__${M}__onto_v3.jsonl" ]; } && break
      echo "[$(date '+%T')] $M $S attempt $attempt did not produce scores - retrying"
      sleep 120
    done
  done
  echo "=== $M COMPLETE ($(date '+%F %T')) ==="
done
echo "=== exp037 campaign complete ($(date '+%F %T')) ==="
