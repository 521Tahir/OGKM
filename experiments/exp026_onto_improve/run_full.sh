#!/usr/bin/env bash
# Run a given arm across models on a given split, GPUs 5-9 only, memory gate
# scaled per model size.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
SPLIT="${SPLIT:-test}"; ARMS=(${ARMS:-onto_c})
MODELS=(${MODELS:-llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b})
LOGD=experiments/exp026_onto_improve/outputs/logs; mkdir -p "$LOGD"
declare -A PIDGPU=(); pending=()
for M in "${MODELS[@]}"; do for A in "${ARMS[@]}"; do pending+=("$M:$A"); done; done
while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date '+%T')] done GPU ${PIDGPU[$pid]}"; unset 'PIDGPU[$pid]'; }
  done
  if [ ${#pending[@]} -gt 0 ]; then
    IFS=: read -r M A <<< "${pending[0]}"
    NEED=$(python -c "import experiments.exp024_rag_vs_ontology.config as c; print(c.gen_mib('$M'))")
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 3 4 5 6 7 8 9 --need "$NEED"); do
      [[ "$busy" == *" $g "* ]] && continue
      pending=("${pending[@]:1}")
      echo "[$(date '+%T')] launching $A/$M on GPU $g (need ${NEED}MiB)"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
        --gpu "$g" --split "$SPLIT" > "$LOGD/gen__${SPLIT}__${M}__${A}.log" 2>&1 &
      PIDGPU[$!]=$g; sleep 150; break
    done
  fi
  sleep 45
done
echo "=== $SPLIT / ${ARMS[*]} generation complete ($(date '+%F %T')) ==="
