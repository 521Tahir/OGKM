#!/usr/bin/env bash
# mistral-7b bench OOM'd when exp037's pinned training landed on the same device
# as the dispatcher's placement. Wait for genuine headroom, then re-run; the
# generator resumes by record_id, so nothing already written is recomputed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
while :; do
  G=$(python -m experiments.exp024_rag_vs_ontology.gpu --need 30000 --n 1)
  if [ -n "$G" ]; then
    echo "[$(date '+%T')] mistral-7b bench hybrid20 on GPU $G"
    python -u -m experiments.exp024_rag_vs_ontology.generate --model mistral-7b \
      --arm hybrid20 --gpu "$G" --split bench \
      > experiments/exp036_hybrid20/gen__mistral-7b__bench.log 2>&1
    rc=$?
    echo "[$(date '+%T')] mistral bench exited rc=$rc"
    [ $rc -eq 0 ] && break
    echo "retrying after failure"
  fi
  sleep 60
done
echo "=== mistral bench rerun complete ==="
