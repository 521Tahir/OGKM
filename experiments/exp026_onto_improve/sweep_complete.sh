#!/usr/bin/env bash
# Completion sweep: find every (model, arm) cell on a split that has fewer than
# the expected DISTINCT record_ids and regenerate it, one process at a time.
#
# Why one at a time: two writers appending to the same cell is what corrupted
# llama3.2-3b__rag earlier (4,620 lines for 3,604 records). Sequential is slower
# but cannot interleave. generate.py resumes by record_id, so restarts are cheap.
#
# Usage: sweep_complete.sh <split> <arm> [arm...]
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
SPLIT="${1:?usage: sweep_complete.sh <split> <arm>...}"; shift
ARMS=("$@")
MODELS="llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b llama3.2-3b llama3.2-1b"
GEN=experiments/exp024_rag_vs_ontology/outputs/generations
EXP=$([ "$SPLIT" = "bench" ] && echo 200 || echo 4607)
say(){ echo "[$(date '+%F %T')] $*"; }

uniq_ids(){ python3 -c "
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print(0); raise SystemExit
ids=set()
for l in open(p,encoding='utf-8'):
    try: ids.add(json.loads(l)['record_id'])
    except Exception: pass
print(len(ids))" "$1"; }

dedupe(){ python3 -c "
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit
seen,kept=set(),[]
for l in open(p,encoding='utf-8'):
    try: r=json.loads(l)
    except Exception: continue
    if r.get('record_id') in seen: continue
    seen.add(r['record_id']); kept.append(l.rstrip('\n'))
p.write_text('\n'.join(kept)+'\n',encoding='utf-8')" "$1"; }

for pass_no in 1 2 3 4 5; do
  remaining=0
  for A in "${ARMS[@]}"; do
    for M in $MODELS; do
      F="$GEN/${SPLIT}__${M}__${A}.jsonl"
      N=$(uniq_ids "$F")
      [ "$N" -ge "$EXP" ] && continue
      remaining=$((remaining+1))
      NEED=$(python -c "import experiments.exp024_rag_vs_ontology.config as c; print(c.gen_mib('$M'))")
      G=""
      for _ in $(seq 1 60); do
        G=$(python -m experiments.exp024_rag_vs_ontology.gpu --need "$NEED" --n 1)
        [ -n "$G" ] && break
        say "waiting for a GPU with ${NEED}MiB (${SPLIT}/${M}/${A} at ${N}/${EXP})"; sleep 180
      done
      [ -z "$G" ] && { say "no GPU for ${M}/${A}; skipping this pass"; continue; }
      dedupe "$F"
      say "pass $pass_no: ${M}/${A} at ${N}/${EXP} -> generating on GPU $G"
      python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm "$A" \
        --gpu "$G" --split "$SPLIT" \
        > "experiments/exp026_onto_improve/outputs/logs/sweep__${SPLIT}__${M}__${A}.log" 2>&1
      dedupe "$F"
      say "   -> now $(uniq_ids "$F")/${EXP}"
    done
  done
  [ "$remaining" -eq 0 ] && { say "sweep complete - every cell full"; break; }
done

say "=== SWEEP DONE ==="
for A in "${ARMS[@]}"; do
  for M in $MODELS; do
    printf '  %-44s %s/%s\n' "${SPLIT}__${M}__${A}" "$(uniq_ids "$GEN/${SPLIT}__${M}__${A}.jsonl")" "$EXP"
  done
done
