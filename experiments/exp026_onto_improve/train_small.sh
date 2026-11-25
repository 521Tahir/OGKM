#!/usr/bin/env bash
# LoRA fine-tune the 1B and 3B models on the same mixed curriculum the 8B models
# used. Restricted to GPUs 5-9; memory requirement scales with model size.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
LOGD=experiments/exp024_rag_vs_ontology/outputs/logs
declare -A PIDGPU=(); pending=(llama3.2-3b llama3.2-1b)
while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date '+%T')] train done GPU ${PIDGPU[$pid]}"; unset 'PIDGPU[$pid]'; }
  done
  if [ ${#pending[@]} -gt 0 ]; then
    M="${pending[0]}"
    NEED=$(python -c "import experiments.exp024_rag_vs_ontology.config as c; print(c.train_mib('$M'))")
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 5 6 7 8 9 --need "$NEED"); do
      [[ "$busy" == *" $g "* ]] && continue
      pending=("${pending[@]:1}")
      if [ -f "/ceph_data/tahirahmad/ontology_exp024/adapters/${M}__mixed/adapter_model.safetensors" ]; then
        echo "[$(date '+%T')] $M adapter exists - skip"; break
      fi
      echo "[$(date '+%T')] training $M on GPU $g (need ${NEED}MiB)"
      python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm mixed \
        --gpu "$g" > "$LOGD/train__${M}__mixed.log" 2>&1 &
      PIDGPU[$!]=$g; sleep 150; break
    done
  fi
  sleep 45
done
echo "=== small-model training complete ($(date '+%F %T')) ==="
