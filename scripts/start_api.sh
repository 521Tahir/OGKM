#!/usr/bin/env bash
# start_api.sh — Launch the Tourism Ontology RAG API
#
# Usage:
#   bash scripts/start_api.sh                    # ontology only, no LLM
#   LOAD_LLM=1 bash scripts/start_api.sh         # load Llama 3.1-8B on startup
#   LOAD_LLM=1 CUDA_VISIBLE_DEVICES=5,6 bash scripts/start_api.sh
#
# All settings default to config/api_config.yaml; env vars below override them.

set -euo pipefail

# ── Project root (directory containing this script's parent) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "[start_api] Loaded .env"
fi

# ── Resolve settings (env vars override api_config.yaml defaults) ─────────────
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-1}"
RELOAD="${API_RELOAD:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOAD_LLM="${LOAD_LLM:-0}"

# GPU: default from api_config.yaml llm.gpu_ids; can be overridden via env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"

# ── Validate environment ──────────────────────────────────────────────────────
if ! python -c "import fastapi, uvicorn, slowapi, yaml" 2>/dev/null; then
    echo "[start_api] ERROR: Missing dependencies. Run: pip install fastapi uvicorn slowapi pyyaml"
    exit 1
fi

if [[ "$LOAD_LLM" == "1" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[start_api] WARNING: HF_TOKEN not set — gated model download may fail"
    fi
    echo "[start_api] LLM enabled on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p logs

# ── Print config summary ──────────────────────────────────────────────────────
echo "========================================"
echo " Tourism Ontology RAG API"
echo "========================================"
echo "  Host:     $HOST:$PORT"
echo "  Workers:  $WORKERS"
echo "  Reload:   $RELOAD"
echo "  LLM:      $LOAD_LLM"
echo "  GPU(s):   $CUDA_VISIBLE_DEVICES"
echo "  Log:      logs/api.log"
echo "  Docs:     http://$HOST:$PORT/docs"
echo "========================================"

# ── Launch uvicorn ────────────────────────────────────────────────────────────
RELOAD_FLAG=""
if [[ "$RELOAD" == "true" || "$RELOAD" == "1" ]]; then
    RELOAD_FLAG="--reload"
fi

exec uvicorn src.api.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    $RELOAD_FLAG
