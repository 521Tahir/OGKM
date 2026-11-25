#!/usr/bin/env bash
# Evaluate every model on the 200-item benchmark, one cell per free GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
MODELS=(llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b)
ARMS=(rag onto onto_v3 base)
LOGD=experiments/exp025_multihop/outputs/logs; mkdir -p "$LOGD"

declare -A PIDGPU=()
pending=()
for M in "${MODELS[@]}"; do for A in "${ARMS[@]}"; do pending+=("$M:$A"); done; done

while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date '+%T')] done on GPU ${PIDGPU[$pid]}"; unset 'PIDGPU[$pid]'; }
  done
  if [ ${#pending[@]} -gt 0 ]; then
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 3 4 5 6 7 8 9 --need 22000); do
      [ ${#pending[@]} -eq 0 ] && break
      [[ "$busy" == *" $g "* ]] && continue
      IFS=: read -r M A <<< "${pending[0]}"; pending=("${pending[@]:1}")
      echo "[$(date '+%T')] launching $A/$M on GPU $g"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
        --gpu "$g" --split bench > "$LOGD/gen__${M}__${A}.log" 2>&1 &
      PIDGPU[$!]=$g; busy=" ${PIDGPU[*]} "
      sleep 150
    done
  fi
  sleep 30
done
echo "=== all cells generated ($(date '+%F %T')) ==="
