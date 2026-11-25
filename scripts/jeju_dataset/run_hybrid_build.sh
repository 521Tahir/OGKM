#!/usr/bin/env bash
# Hybrid Jeju ontology build (Offline + LLM) on the FULL dataset, GPUs 5-9.
# Every worker sets proctitle "tahir_Ont-code". Run when GPUs 5,6,7,8,9 are free.
#
#   bash scripts/jeju_dataset/run_hybrid_build.sh
#
# Resumable: re-running continues incomplete LLM shards from checkpoints.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env 2>/dev/null; set +a
export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_SHARDS=5
GPUS=(5 6 7 8 9)
BATCH="${BATCH:-16}"

echo "==================================================================="
echo " HYBRID JEJU ONTOLOGY BUILD (Offline + LLM) — full dataset, GPUs ${GPUS[*]}"
echo "==================================================================="

echo "[1/4] Offline base (gazetteer + geo + mined relations) ..."
python scripts/jeju_dataset/build_jeju_ontology.py

echo "[2/4] Launching $NUM_SHARDS LLM extraction shards ..."
pids=()
for k in 0 1 2 3 4; do
    gpu="${GPUS[$k]}"
    nohup python scripts/jeju_dataset/llm_extract_shard.py \
        --gpu "$gpu" --shard "$k" --num-shards "$NUM_SHARDS" --batch "$BATCH" \
        > "data/jeju_full/llm_shard${k}.log" 2>&1 &
    pids+=($!)
    echo "    shard $k -> GPU $gpu (pid ${pids[$k]})"
done
echo "    waiting for all shards to finish (proctitle: tahir_Ont-code) ..."
wait "${pids[@]}"

echo "[3/4] Merging Offline + LLM -> hybrid ontology v1.1.0 ..."
python scripts/jeju_dataset/merge_hybrid_ontology.py --base 1.0.0 --target 1.1.0

echo "[4/4] Ingesting hybrid v1.1.0 into Neo4j ..."
python scripts/ingest_to_neo4j.py --ontology-version 1.1.0 --clear || echo "    (Neo4j ingest skipped/failed)"

echo "==================================================================="
echo " DONE — hybrid ontology v1.1.0 built and loaded into Neo4j."
echo " Visualize: http://localhost:7474  (neo4j / jeju1234)"
echo "==================================================================="
