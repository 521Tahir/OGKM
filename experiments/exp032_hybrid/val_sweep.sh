#!/usr/bin/env bash
# Tune the hybrid's fact/passage split on the VALIDATION split.
#
# The hybrid at a 30% fact share trailed Simple RAG by only 0.0008-0.0036 on the
# full test set - close enough that the allocation itself may be the deciding
# factor rather than the idea. Val (4,605 items) has never been used in this
# project, so choosing the share there and reporting the winner on test is the
# textbook-correct way to tune without contaminating the test estimate.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
M="${MODEL:-llama3.1-8b}"
N="${NITEMS:-1200}"
ARMS=(${ARMS:-rag hybrid10 hybrid15 hybrid20})
declare -A PIDGPU=()
pending=("${ARMS[@]}")

while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date '+%T')] done GPU ${PIDGPU[$pid]}"; unset 'PIDGPU[$pid]'; }
  done
  if [ ${#pending[@]} -gt 0 ]; then
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --need 22000); do
      [ ${#pending[@]} -eq 0 ] && break
      [[ "$busy" == *" $g "* ]] && continue
      A="${pending[0]}"; pending=("${pending[@]:1}")
      echo "[$(date '+%T')] $A on GPU $g"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
        --gpu "$g" --split val --limit "$N" \
        > "experiments/exp032_hybrid/val__${A}.log" 2>&1 &
      PIDGPU[$!]=$g; busy=" ${PIDGPU[*]} "; sleep 120; break
    done
  fi
  sleep 30
done
echo "=== val sweep generation complete ($(date '+%F %T')) ==="
