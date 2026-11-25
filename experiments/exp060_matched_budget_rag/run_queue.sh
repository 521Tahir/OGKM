#!/usr/bin/env bash
# exp060 - run the matched-budget RAG work for one GPU, model by model.
#
# Each model is trained to the pass count its ontology counterpart already had,
# then generates the held-out test answers. Training resumes from the last
# checkpoint, so a crash on this shared box costs the last few hundred steps and
# not the run; a model that fails three times is skipped and the queue moves on
# rather than blocking the models behind it.
#
#   ./run_queue.sh 6 qwen2.5-3b llama3.2-1b
set -u
cd /home/tahirahmad/Ontology_Project_JejuFull
GPU="$1"; shift
HERE=experiments/exp060_matched_budget_rag
LOGS="$HERE/outputs/logs"
mkdir -p "$LOGS"

for MODEL in "$@"; do
  ADAPTER="$HERE/outputs/adapters/${MODEL}__rag_pass"*
  echo "=== $(date '+%F %T') gpu$GPU $MODEL: train ===" >> "$LOGS/queue_gpu$GPU.log"
  ok=0
  for attempt in 1 2 3; do
    python3 "$HERE/train_rag_extended.py" --model "$MODEL" --gpu "$GPU" \
      >> "$LOGS/train_$MODEL.log" 2>&1 && { ok=1; break; }
    echo "    attempt $attempt failed, retrying from checkpoint" >> "$LOGS/queue_gpu$GPU.log"
    sleep 120
  done
  if [ "$ok" -ne 1 ]; then
    echo "!!! $(date '+%F %T') gpu$GPU $MODEL: training failed 3x, skipping" >> "$LOGS/queue_gpu$GPU.log"
    continue
  fi
  echo "=== $(date '+%F %T') gpu$GPU $MODEL: generate ===" >> "$LOGS/queue_gpu$GPU.log"
  for attempt in 1 2 3; do
    python3 "$HERE/generate_rag_extended.py" --model "$MODEL" --gpu "$GPU" \
      >> "$LOGS/generate_$MODEL.log" 2>&1 && break
    echo "    generate attempt $attempt failed, resuming" >> "$LOGS/queue_gpu$GPU.log"
    sleep 120
  done
  echo "=== $(date '+%F %T') gpu$GPU $MODEL: done ===" >> "$LOGS/queue_gpu$GPU.log"
done
echo "=== $(date '+%F %T') gpu$GPU queue complete ===" >> "$LOGS/queue_gpu$GPU.log"
