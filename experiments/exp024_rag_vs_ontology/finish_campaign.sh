#!/usr/bin/env bash
# Waits for the in-flight Mistral training to land its adapter, then finishes the
# campaign (generate both arms -> score every cell -> render the tables) using
# the full GPU candidate pool 0-9. Safe to run alongside the base-arm job: the
# free-memory gate in gpu.py decides placement.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
ADAPTER=/ceph_data/tahirahmad/ontology_exp024/adapters/mistral-7b__mixed/adapter_model.safetensors

echo "waiting for $ADAPTER ($(date '+%F %T'))"
while [ ! -f "$ADAPTER" ]; do
  sleep 120
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -qx 383948; then
    [ -f "$ADAPTER" ] || { echo "!! mistral training pid 383948 gone and no adapter — aborting"; exit 1; }
  fi
done
echo "adapter present ($(date '+%F %T')) — finishing campaign"
python -u -m experiments.exp024_rag_vs_ontology.run_all --stage generate score report
echo "=== campaign complete ($(date '+%F %T')) ==="
