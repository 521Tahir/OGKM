#!/usr/bin/env bash
# exp060 - GPU 5 follow-on. qwen2.5-1.5b training was launched by hand before the
# queue existed; wait for it, generate its answers, then take llama3.2-3b through
# the normal train-and-generate path.
set -u
cd /home/tahirahmad/Ontology_Project_JejuFull
HERE=experiments/exp060_matched_budget_rag
LOGS="$HERE/outputs/logs"
WAIT_PID="${1:?pid of the running qwen2.5-1.5b trainer}"

say() { echo "=== $(date '+%F %T') gpu5 $*" >> "$LOGS/queue_gpu5.log"; }

say "waiting for pid $WAIT_PID (qwen2.5-1.5b training)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
say "qwen2.5-1.5b training process exited"

if [ -f "$HERE/outputs/adapters/qwen2.5-1.5b__rag_pass3/adapter_config.json" ]; then
  say "qwen2.5-1.5b: generate"
  for attempt in 1 2 3; do
    python3 "$HERE/generate_rag_extended.py" --model qwen2.5-1.5b --gpu 5 \
      >> "$LOGS/generate_qwen2.5-1.5b.log" 2>&1 && break
    say "  generate attempt $attempt failed, resuming"; sleep 120
  done
  say "qwen2.5-1.5b: done"
else
  say "qwen2.5-1.5b: no adapter written - training did not finish; retrying it"
  for attempt in 1 2 3; do
    python3 "$HERE/train_rag_extended.py" --model qwen2.5-1.5b --gpu 5 \
      >> "$LOGS/train_qwen2.5-1.5b.log" 2>&1 && break
    say "  retrain attempt $attempt failed"; sleep 120
  done
  python3 "$HERE/generate_rag_extended.py" --model qwen2.5-1.5b --gpu 5 \
    >> "$LOGS/generate_qwen2.5-1.5b.log" 2>&1
  say "qwen2.5-1.5b: done"
fi

exec "$HERE/run_queue.sh" 5 llama3.2-3b
