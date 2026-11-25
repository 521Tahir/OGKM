#!/usr/bin/env bash
# Vanilla-LLM control arm (no adapter, no context), pinned to ONE gpu and run
# sequentially so it never races the main orchestrator for placement.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
GPU="${1:?usage: run_base_arm.sh <gpu> [models...]}"; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b)
for M in "${MODELS[@]}"; do
  echo "=== base arm: $M on GPU $GPU ($(date '+%F %T')) ==="
  python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm base --gpu "$GPU" \
    || echo "!! $M base FAILED (rc=$?), continuing"
done
echo "=== base arm complete ($(date '+%F %T')) ==="
