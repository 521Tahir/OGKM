#!/usr/bin/env bash
# Launch the Ontology-Augmented LLM Benchmark.
#
# 1) Start 4 independent model microservices (one per GPU 5-8) if not already up.
# 2) Wait until all 4 report /health ready (models loaded, ~1-2 min first time).
# 3) Launch the Streamlit app (metrics on GPU 9), which is just an HTTP client.
#
# Decoupling model serving from Streamlit avoids the multiprocessing-spawn
# deadlock that produced "Models ready: none" / "[model unavailable]".
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null; set +a
export HF_HOME="${HF_HOME:-/ceph_data/tahirahmad/hf_home}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PORT="${1:-8501}"
LOGDIR="streamlit_app/results/logs"; mkdir -p "$LOGDIR"
PYBIN="$(command -v python3 || command -v python)"

# model_key | display name | gpu | port | reasoning(0/1)
SERVICES=(
  "llama3.1-8b|Llama-3.1-8B|5|8101|0"
  "deepseek-r1-8b|DeepSeek-R1-8B|6|8102|1"
  "qwen3-8b|Qwen3-8B|7|8103|1"
  "mistral-7b|Mistral-7B|8|8104|0"
)

health() { curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/health" 2>/dev/null; }

echo "Starting model microservices…"
for s in "${SERVICES[@]}"; do
  IFS='|' read -r key name gpu sport reason <<< "$s"
  if [ "$(health "$sport")" = "200" ]; then
    echo "  [$name] already healthy on $sport — reusing"
    continue
  fi
  args=(--key "$key" --name "$name" --gpu "$gpu" --port "$sport")
  [ "$reason" = "1" ] && args+=(--reasoning)
  echo "  [$name] launching on GPU $gpu port $sport"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_HOME" \
    setsid "$PYBIN" streamlit_app/model_service.py "${args[@]}" \
    > "$LOGDIR/service_${sport}.log" 2>&1 < /dev/null &
done

echo "Waiting for all services to load (up to 5 min)…"
deadline=$((SECONDS + 300))
while [ $SECONDS -lt $deadline ]; do
  ok=0
  for s in "${SERVICES[@]}"; do
    sport="$(echo "$s" | cut -d'|' -f4)"
    [ "$(health "$sport")" = "200" ] && ok=$((ok+1))
  done
  echo "  ready: $ok/4"
  [ "$ok" = "4" ] && break
  sleep 5
done
if [ "$ok" != "4" ]; then
  echo "WARNING: only $ok/4 services healthy — check $LOGDIR/service_*.log"
fi

export CUDA_VISIBLE_DEVICES=9   # Streamlit process: metrics on GPU 9
echo "Starting benchmark app on http://localhost:${PORT}"
exec streamlit run streamlit_app/app.py \
    --server.port "${PORT}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
