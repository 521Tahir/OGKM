#!/usr/bin/env bash
# Score hybrid20 for all six models, test + bench.
#
# Sequenced AFTER exp037's test scoring: both jobs want ~34GB for BERTScore +
# COMET + the 8B judge, and running them together is what OOM'd the first
# attempt. Each cell is gated on the free-memory picker and retried, because
# this box is shared and "idle a second ago" is not "idle now".
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/exp036_hybrid20
SD=experiments/exp024_rag_vs_ontology/outputs/scores

echo "[$(date '+%T')] waiting for exp037 test scoring to release its GPU …"
until grep -q 'exp037 test scoring finished' \
      experiments/exp037_dedicated_adapter/score_test_driver.log 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%T')] exp037 clear - starting hybrid20 scoring"

MODELS=(llama3.2-1b llama3.2-3b llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b)
for S in bench test; do
  for M in "${MODELS[@]}"; do
    [ -f "$SD/${S}__${M}__hybrid20.jsonl" ] && { echo "[$(date '+%T')] $S $M already scored"; continue; }
    for attempt in 1 2 3 4 5; do
      G=""
      while [ -z "$G" ]; do
        G=$(python -m experiments.exp024_rag_vs_ontology.gpu --need 34000 --n 1)
        [ -z "$G" ] && sleep 90
      done
      echo "[$(date '+%T')] score $S $M hybrid20 on GPU $G (attempt $attempt)"
      python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$G" --split "$S" \
        --cells "${M}__hybrid20" >> "$LOG/score__${S}__${M}.log" 2>&1
      [ -f "$SD/${S}__${M}__hybrid20.jsonl" ] && break
      echo "[$(date '+%T')] $S $M attempt $attempt failed - retrying"
      sleep 120
    done
  done
done
echo "=== exp036 hybrid20 scoring complete ($(date '+%F %T')) ==="
