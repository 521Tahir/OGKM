#!/usr/bin/env bash
# Launch the LLM → Ontology → Neo4j knowledge demo.
# Extraction runs a 4-bit Llama-3.1-8B on KG_DEMO_GPU (default 7).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null; set +a
export HF_HOME="${HF_HOME:-/ceph_data/tahirahmad/hf_home}"
export HF_TOKEN="${HF_TOKEN:-local-cache}"
export KG_DEMO_GPU="${KG_DEMO_GPU:-7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT="${1:-8502}"
echo "Starting KG demo on http://localhost:${PORT}  (extraction on GPU ${KG_DEMO_GPU}; Neo4j on 7687/7474)"
exec streamlit run streamlit_app/kg_demo.py \
    --server.port "${PORT}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
