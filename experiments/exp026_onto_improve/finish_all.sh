#!/usr/bin/env bash
# Single driver for everything that remains:
#   1. wait until every expected generation cell is COMPLETE (row-count checked,
#      not process-exit checked - a cell that OOMs mid-run exits "successfully"
#      with a short file, which is how the 3B rag arm silently stalled);
#   2. score both splits, retrying through OOM and GPU contention;
#   3. regenerate every report;
#   4. rebuild the Word report, the slides and the demo page.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
GEN=experiments/exp024_rag_vs_ontology/outputs/generations
say(){ echo "[$(date '+%F %T')] $*"; }

BIG="llama3.1-8b deepseek-r1-8b qwen3-8b mistral-7b"
SMALL="llama3.2-1b llama3.2-3b"

# Count DISTINCT record_ids, not lines. Two resume processes once ran
# concurrently on the same cell and appended interleaved duplicates, producing
# 4,620 lines for only 3,604 records - a plain `wc -l >= 4607` gate passed that
# corrupted file. Never trust line count as a completeness signal here.
uniq_ids(){ python3 -c "
import json,sys
ids=set()
for l in open(sys.argv[1],encoding='utf-8'):
    try: ids.add(json.loads(l)['record_id'])
    except Exception: pass
print(len(ids))" "$1"; }

incomplete_test(){
  local n out=0
  for m in $BIG; do
    for a in rag onto onto_c base; do
      f="$GEN/test__${m}__${a}.jsonl"
      [ -f "$f" ] || continue
      n=$(uniq_ids "$f"); [ "$n" -lt 4607 ] && { echo "test__${m}__${a}=$n"; out=1; }
    done
  done
  for m in $SMALL; do
    for a in rag onto_c base; do
      f="$GEN/test__${m}__${a}.jsonl"
      [ -f "$f" ] || { echo "test__${m}__${a}=MISSING"; out=1; continue; }
      n=$(uniq_ids "$f"); [ "$n" -lt 4607 ] && { echo "test__${m}__${a}=$n"; out=1; }
    done
  done
  return $out
}

say "STAGE 1 - waiting for every test cell to reach 4607 rows"
while :; do
  short=$(incomplete_test) && { say "all test cells complete"; break; }
  say "still short: $(echo "$short" | tr '\n' ' ')"
  sleep 300
done

say "STAGE 2 - scoring"
bash experiments/exp026_onto_improve/final_score.sh test 600
bash experiments/exp026_onto_improve/final_score.sh bench 120

say "STAGE 3 - reports"
python -u -m experiments.exp024_rag_vs_ontology.report --split test  || say "!! exp024 report"
python -u -m experiments.exp025_multihop.report_bench               || say "!! bench report"
python -u -m experiments.exp026_onto_improve.rank_variants          || say "!! variant ranking"
python -u -m experiments.exp029_scale.report_scale --split bench    || say "!! scale bench"
python -u -m experiments.exp029_scale.report_scale --split test     || say "!! scale test"
python -u -m experiments.exp027_qualitative.analyse --split bench --onto-arm onto_c || say "!! qual bench"
python -u -m experiments.exp027_qualitative.report  --split bench --onto-arm onto_c || say "!! qual report"
python -u -m experiments.exp027_qualitative.analyse --split test  --onto-arm onto_c || say "!! qual test"

say "STAGE 4 - documents"
python -u -m experiments.exp025_multihop.make_report_docx || say "!! docx"
python -u -m experiments.exp025_multihop.make_slides      || say "!! slides"
python -u -m experiments.exp028_demo.make_demo --onto-arm onto_c --n 6 || say "!! demo"

say "=== EVERYTHING COMPLETE ==="
