#!/usr/bin/env bash
# Wait for the v4/v5 bench cells, score them, then rank every ontology variant
# against Simple RAG on real metrics.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
G=experiments/exp024_rag_vs_ontology/outputs/generations
while :; do
  full=0
  for M in llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b; do
    for A in onto_v4 onto_v5; do
      f="$G/bench__${M}__${A}.jsonl"
      [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 200 ] && full=$((full+1))
    done
  done
  [ "$full" -ge 8 ] && break
  echo "[$(date '+%T')] $full/8 variant cells complete"; sleep 120
done
echo "[$(date '+%T')] scoring"
GPU=$(python -m experiments.exp024_rag_vs_ontology.gpu --candidates 5 6 7 8 9 --need 26000 --n 1)
python -u -m experiments.exp024_rag_vs_ontology.score --split bench --metrics all --gpu "${GPU:-3}"
python -u -m experiments.exp026_onto_improve.rank_variants
echo "=== variant comparison complete ($(date '+%F %T')) ==="
