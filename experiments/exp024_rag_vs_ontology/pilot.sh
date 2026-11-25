#!/usr/bin/env bash
# End-to-end smoke of every stage on a tiny slice, in its own EXP024_TAG
# namespace. Validates the wiring (adapters, prompts, COMET/BERTScore/judge
# downloads, report rendering) before the full campaign is committed.
set -euo pipefail
cd "$(dirname "$0")/../.."
export EXP024_TAG=pilot
export PYTHONUNBUFFERED=1
GPU="${1:-4}"
M=llama3.1-8b

echo "=== [1/4] train mixed adapter (512 examples) ==="
python -u -m experiments.exp024_rag_vs_ontology.train_lora --model $M --arm mixed --gpu "$GPU" --limit 512

for ARM in rag onto; do
  echo "=== [2/4] generate $ARM (60 items) ==="
  python -u -m experiments.exp024_rag_vs_ontology.generate --model $M --arm $ARM --gpu "$GPU" --limit 60
done

echo "=== [3/4] score ==="
python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$GPU" --metrics all

echo "=== [4/4] report ==="
python -u -m experiments.exp024_rag_vs_ontology.report
