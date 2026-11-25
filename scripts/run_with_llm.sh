#!/usr/bin/env bash
# ============================================================
# run_with_llm.sh — Download Llama-3.1-8B and rerun all experiments
#
# Usage:
#   bash scripts/run_with_llm.sh            # download + run all
#   bash scripts/run_with_llm.sh --skip-download   # skip download if already cached
#   bash scripts/run_with_llm.sh --only-eval       # only run eval experiments (004,005,006)
#
# Prerequisites:
#   - .env must have HF_TOKEN set and LOCAL_MODEL_KEY=llama3.1-8b
#   - Conda env with transformers, bitsandbytes, accelerate installed
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

SKIP_DOWNLOAD=0
ONLY_EVAL=0
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --only-eval)     ONLY_EVAL=1 ;;
    esac
done

MODEL_KEY="${LOCAL_MODEL_KEY:-llama3.1-8b}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
LOG_DIR="artifacts/runs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Tourism Ontology — LLM Run"
echo "  Model   : $MODEL_KEY"
echo "  GPUs    : $CUDA_VISIBLE_DEVICES"
echo "  Date    : $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"

# ── Step 1: Download model ────────────────────────────────────
if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
    echo ""
    echo "[1/4] Downloading $MODEL_KEY ..."
    python scripts/download_model.py \
        --model "$MODEL_KEY" \
        --verify \
        --gpu "${GPUS%%,*}" \
        2>&1 | tee "$LOG_DIR/download_${MODEL_KEY}.log"
    echo "Download complete."
else
    echo "[1/4] Skipping download (--skip-download set)"
fi

# ── Step 2: Re-extract ontology with LLM (exp002 rerun) ───────
if [ "$ONLY_EVAL" -eq 0 ]; then
    echo ""
    echo "[2/4] Running exp002: Ontology extraction with Llama-3.1-8B ..."
    echo "  Strategy: cot | LLM records: 1000 | Full dataset: 61,963"
    python -u experiments/exp002_ontology_bootstrap/run_extraction.py \
        --strategy cot \
        --version 0.4.0 \
        --llm-records 1000 \
        --load-in-4bit \
        --gpus "$GPUS" \
        --save-every 200 \
        2>&1 | tee "$LOG_DIR/exp002_llama31.log"
    echo "exp002 complete."

    # Re-run validation on new ontology
    echo ""
    echo "[2b] Running exp003: Validate + repair ontology v0.4.0 ..."
    python experiments/exp003_validation/run_validation.py \
        --source-version 0.4.0 \
        --target-version 0.5.0 \
        --diff-against 0.3.0 \
        2>&1 | tee "$LOG_DIR/exp003_llama31.log"
    echo "exp003 complete."
else
    echo "[2/4] Skipping exp002/003 (--only-eval set)"
fi

# ── Step 3: RAG pipeline with real LLM ────────────────────────
ONTO_VER="0.5.0"
if [ "$ONLY_EVAL" -eq 1 ]; then
    ONTO_VER="0.3.0"
fi

echo ""
echo "[3/4] Running exp004: RAG pipeline with $MODEL_KEY (ontology $ONTO_VER) ..."
python experiments/exp004_ontology_rag/run_rag.py \
    --ontology-version "$ONTO_VER" \
    --n-queries 50 \
    2>&1 | tee "$LOG_DIR/exp004_llama31.log"

echo ""
echo "[4/4] Running exp005 + exp006: Constraint enforcement + full evaluation ..."
python experiments/exp005_constraints/run_constraints.py \
    --ontology-version "$ONTO_VER" \
    --n-queries 100 \
    2>&1 | tee "$LOG_DIR/exp005_llama31.log"

python experiments/exp006_evaluation/run_evaluation.py \
    --ontology-version "$ONTO_VER" \
    --n-queries 200 \
    2>&1 | tee "$LOG_DIR/exp006_llama31.log"

echo ""
echo "============================================================"
echo " ALL DONE"
echo "  Logs     : $LOG_DIR/"
echo "  Reports  : artifacts/reports/"
echo "  Figures  : artifacts/figures/exp006_comparison_table.md"
echo "============================================================"
