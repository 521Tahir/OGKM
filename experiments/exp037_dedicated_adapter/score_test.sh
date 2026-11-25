#!/usr/bin/env bash
# Score the dedicated-adapter TEST cells. The first attempt was launched on a
# hand-picked GPU and OOM'd: nvidia-smi showed the device idle moments earlier,
# but two exp036 generators landed on it in between. Never hand-pick - ask the
# free-memory picker, and retry rather than die, because the box is shared.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 EXP024_TAG=ded PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/exp037_dedicated_adapter
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  G=""
  while [ -z "$G" ]; do
    G=$(python -m experiments.exp024_rag_vs_ontology.gpu --need 34000 --n 1)
    [ -z "$G" ] && sleep 90
  done
  echo "[$(date '+%T')] attempt $attempt: scoring test cells on GPU $G"
  python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$G" --split test \
    --cells llama3.2-1b__onto_v3 llama3.2-1b__rag >> "$LOG/score__test.log" 2>&1
  rc=$?
  if [ -f experiments/exp024_rag_vs_ontology/outputs/scores_ded/test__llama3.2-1b__rag.jsonl ] \
     && [ -f experiments/exp024_rag_vs_ontology/outputs/scores_ded/test__llama3.2-1b__onto_v3.jsonl ]; then
    echo "[$(date '+%T')] test scoring complete (rc=$rc)"; break
  fi
  echo "[$(date '+%T')] attempt $attempt failed (rc=$rc) - waiting for headroom"
  sleep 120
done
echo "=== exp037 test scoring finished ==="
