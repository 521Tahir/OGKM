#!/usr/bin/env bash
# exp038 - the two partial fixes, COMBINED, on Llama-3.2-1B.
#
# Why this is not another repeat. The sibling project's exp009 measured ontology
# retrieval against document retrieval on 6,197 queries and separated the two
# failure modes cleanly:
#
#   mode        Recall    MRR     Hit@5   avg concepts
#   doc_only    0.752     0.382   0.605   314.8
#   onto_only   0.109     0.613   0.749    70.3
#
# Ontology retrieval has the BEST ranking (MRR 0.613 vs 0.382, Hit@5 0.749 vs
# 0.605) and the WORST recall (0.109 vs 0.752). That is exactly why it wins the
# 200-question benchmark, where one correct fact answers the question, and loses
# the 4,607-item free-form split, where a gold itinerary names ~18 places and the
# score is driven by how many of them appear at all.
#
# So the free-form deficit is a RECALL problem, and two separate things were
# capping it. Each was fixed once, alone, and neither was enough:
#
#   * coverage      onto_v6's popularity prior lifted gold-entity coverage
#                   0.270 -> 0.512 (RAG is 0.560) - but was served by the `mixed`
#                   adapter, whose learned policy is to copy retrieved prose, so
#                   the extra facts had nothing to act on.
#   * policy        a dedicated adapter lifted onto_v3 on 27/30 cells - but
#                   onto_v3 retrieves at coverage 0.270, so there was little to
#                   ground on.
#
# exp038 combines them: a dedicated adapter trained on onto_v6 contexts. High
# recall AND a policy that expands facts instead of copying prose. train_onto_v6
# did not exist until now, so this pairing has never been tested.
#
# Budget stays matched at 3,000 characters, identical to the RAG arm. The control
# is rag @ dedicated, already trained in exp037.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 EXP024_TAG=ded PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/exp038_onto_v6_dedicated
AD=/ceph_data/tahirahmad/ontology_exp024/adapters_ded
SD=experiments/exp024_rag_vs_ontology/outputs/scores_ded
M=llama3.2-1b

pick() { local need=$1; shift; local busy=" $* "; local g
  for g in $(python -m experiments.exp024_rag_vs_ontology.gpu --need "$need"); do
    [[ "$busy" == *" $g "* ]] || { echo "$g"; return; }
  done; }

if [ ! -f "$AD/${M}__onto_v6/adapter_config.json" ]; then
  G=""; while [ -z "$G" ]; do G=$(pick 34000); [ -z "$G" ] && sleep 120; done
  echo "[$(date '+%T')] TRAIN $M/onto_v6 on GPU $G"
  python -u -m experiments.exp024_rag_vs_ontology.train_lora --model "$M" --arm onto_v6 \
    --gpu "$G" > "$LOG/train__${M}__onto_v6.log" 2>&1
  [ -f "$AD/${M}__onto_v6/adapter_config.json" ] || { echo "FATAL adapter missing"; exit 1; }
fi
echo "[$(date '+%T')] adapter ready"

GPUS=(); PIDS=()
for S in bench test; do
  G=""; while [ -z "$G" ]; do G=$(pick 30000 "${GPUS[@]:-}"); [ -z "$G" ] && sleep 120; done
  echo "[$(date '+%T')] GEN $M/onto_v6 $S on GPU $G"
  python -u -m experiments.exp024_rag_vs_ontology.generate --model "$M" --arm onto_v6 \
    --adapter onto_v6 --gpu "$G" --split "$S" > "$LOG/gen__${S}.log" 2>&1 &
  GPUS+=("$G"); PIDS+=("$!"); sleep 90
done
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || echo "[$(date '+%T')] WARN gen GPU ${GPUS[$i]} rc=$?"; done

for S in bench test; do
  [ -f "$SD/${S}__${M}__onto_v6.jsonl" ] && continue
  for attempt in 1 2 3; do
    G=""; while [ -z "$G" ]; do G=$(pick 34000); [ -z "$G" ] && sleep 120; done
    echo "[$(date '+%T')] SCORE $S on GPU $G (attempt $attempt)"
    python -u -m experiments.exp024_rag_vs_ontology.score --gpu "$G" --split "$S" \
      --cells "${M}__onto_v6" >> "$LOG/score__${S}.log" 2>&1
    [ -f "$SD/${S}__${M}__onto_v6.jsonl" ] && break
    sleep 120
  done
done
echo "=== exp038 1B complete ($(date '+%F %T')) ==="
