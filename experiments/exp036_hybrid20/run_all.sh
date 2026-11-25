#!/usr/bin/env bash
# exp036 - confirm hybrid20 across all six models on the full test split + bench.
#
# Why this arm and this share. The val sweep (exp032, 1,200 held-out val items,
# Llama-3.1-8B) tuned the ontology fact share without touching test:
#
#   share   BERT-F1   METEOR    COMET    Key-tok    Judge
#   10%     p=0.051   p=0.092   p=0.066  p=0.591    p=0.456
#   15%     p=0.005*  p=0.016*  p=0.017* p=0.764    p=0.309
#   20%     p=0.072   p=0.575   p=0.557  p=0.937    p=0.699
#
# At 20% NOTHING differs significantly from Simple RAG (largest |delta| 0.0011),
# and the two length-insensitive metrics lean ontology-positive. Every pure
# ontology arm loses 5/5 significantly; hybrid20 is the first arm that gives up
# nothing on free-form while keeping ontology grounding for multi-hop.
#
# This run confirms that on the full 4,607-item test split for all six models,
# and on the 200-item multi-hop bench.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
LOG=experiments/exp036_hybrid20
MODELS=(llama3.2-1b llama3.2-3b llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b)

pending=()
for m in "${MODELS[@]}"; do pending+=("$m:bench" "$m:test"); done

declare -A PIDGPU=()
while [ ${#pending[@]} -gt 0 ] || [ ${#PIDGPU[@]} -gt 0 ]; do
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date '+%T')] done GPU ${PIDGPU[$pid]}"; unset 'PIDGPU[$pid]'; }
  done
  if [ ${#pending[@]} -gt 0 ]; then
    busy=" ${PIDGPU[*]} "
    for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --need 24000); do
      [ ${#pending[@]} -eq 0 ] && break
      [[ "$busy" == *" $g "* ]] && continue
      job="${pending[0]}"; pending=("${pending[@]:1}")
      M="${job%%:*}"; S="${job##*:}"
      echo "[$(date '+%T')] $M $S hybrid20 on GPU $g  (${#pending[@]} queued)"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm hybrid20 \
        --gpu "$g" --split "$S" > "$LOG/gen__${M}__${S}.log" 2>&1 &
      PIDGPU[$!]=$g; busy=" ${PIDGPU[*]} "; sleep 90; break
    done
  fi
  sleep 30
done
echo "=== exp036 hybrid20 generation complete ($(date '+%F %T')) ==="
