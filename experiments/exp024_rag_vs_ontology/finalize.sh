#!/usr/bin/env bash
# Wait for every expected generation cell to reach full length, then score the
# new cells and re-render the tables. Idempotent: score.py skips cells whose
# metrics are already on disk.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
G=experiments/exp024_rag_vs_ontology/outputs/generations
EXPECT=4607
CELLS="test__llama3.1-8b__onto_v3 test__deepseek-r1-8b__onto_v3 test__qwen3-8b__onto_v3 test__mistral-7b__onto_v3 test__mistral-7b__base"

while :; do
  missing=0
  for c in $CELLS; do
    n=$( [ -f "$G/$c.jsonl" ] && wc -l < "$G/$c.jsonl" || echo 0 )
    [ "$n" -lt "$EXPECT" ] && missing=$((missing+1))
  done
  [ "$missing" -eq 0 ] && break
  echo "[$(date '+%T')] waiting on $missing cell(s)"
  sleep 300
done

echo "[$(date '+%T')] all cells complete — scoring"
GPU=$(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 5 6 7 8 9 --need 26000 --n 1)
python -u -m experiments.exp024_rag_vs_ontology.score --gpu "${GPU:-4}" --metrics all
python -u -m experiments.exp024_rag_vs_ontology.report
echo "=== finalize complete ($(date '+%F %T')) ==="
