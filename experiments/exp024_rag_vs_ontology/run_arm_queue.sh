#!/usr/bin/env bash
# Dispatch one generation cell per free GPU across the full 0-9 pool, waiting for
# headroom rather than displacing anyone. Tahir authorised 5-9 alongside GROUND
# (2026-09-10); the free-memory gate still decides placement.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
ARM="${1:?usage: run_arm_queue.sh <arm> [models...]}"; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b)
NEED=22000
declare -A PIDGPU=()

pending=("${MODELS[@]}")
while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(date '+%T')] cell on GPU ${PIDGPU[$pid]} finished (pid $pid)"
      unset 'PIDGPU[$pid]'
    fi
  done
  if [ ${#pending[@]} -gt 0 ]; then
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 3 4 5 6 7 8 9 --need $NEED); do
      [ ${#pending[@]} -eq 0 ] && break
      [[ "$busy" == *" $g "* ]] && continue
      M="${pending[0]}"; pending=("${pending[@]:1}")
      echo "[$(date '+%T')] launching $ARM/$M on GPU $g"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$ARM" --gpu "$g" \
        > "experiments/exp024_rag_vs_ontology/outputs/logs/gen__${M}__${ARM}.log" 2>&1 &
      PIDGPU[$!]=$g; busy=" ${PIDGPU[*]} "
      sleep 90   # let it allocate before re-reading free memory
    done
  fi
  sleep 60
done
echo "=== $ARM queue complete ($(date '+%F %T')) ==="
