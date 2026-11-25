"""
LLM extraction worker (one shard) — the LLM half of the hybrid Jeju ontology.

Processes a shard of the Jeju dataset with a vanilla LLM (Llama-3.1-8B) to extract
tourism concepts and semantic relations the deterministic/offline pass misses
(has_risk, avoid_during, requires, recommended_duration, supports_dietary_need,
needs_reservation, best_for, …). Batched (call_batch) for speed; sharded by
record index so 5 workers run in parallel across GPUs 5-9.

Output: data/jeju_full/llm_onto/llm_shard{S}.jsonl  (one JSON per record:
        {"concepts":[{name,class}], "relations":[{s,p,o}], "rid":...})

Usage (one shard):
  python scripts/jeju_dataset/llm_extract_shard.py --gpu 5 --shard 0 --num-shards 5 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.proctitle import set_proctitle
from src.utils.logging_utils import get_logger

logger = get_logger("llm_extract")

RECORDS = ROOT / "data" / "processed" / "cleaned_records.jsonl"
OUT_DIR = ROOT / "data" / "jeju_full" / "llm_onto"

# Allowed vocabulary (kept aligned with src/ontology/schema.py)
CLASSES = [
    "Destination", "Region", "City", "Island", "Beach", "Mountain", "Trail",
    "NaturalAttraction", "CulturalAttraction", "ReligiousSite", "Museum", "Market",
    "Landmark", "Park", "Attraction", "Restaurant", "Cuisine", "Accommodation",
    "Activity", "TransportMode", "Route", "Itinerary", "TravelerType", "Season",
    "WeatherCondition", "SafetyAdvice", "EmergencyService", "LocalCustom",
    "TravelConstraint", "DietaryRequirement",
]
PREDICATES = [
    "located_in", "part_of", "near", "best_for", "suitable_for", "requires",
    "avoid_during", "available_in", "served_at", "reachable_by", "connected_to",
    "has_activity", "has_risk", "has_seasonality", "has_budget_level",
    "supports_dietary_need", "has_accessibility_feature", "recommended_duration",
    "needs_reservation", "depends_on_weather",
]
CLASS_SET = {c.lower(): c for c in CLASSES}
PRED_SET = set(PREDICATES)

SYSTEM = "You are a Jeju Island tourism knowledge extractor. Output strict JSON only."
PROMPT = """Extract Jeju Island tourism knowledge from the text below.

Allowed concept classes: {classes}
Allowed relation predicates: {preds}

Return STRICT JSON:
{{"concepts": [{{"name": "...", "class": "<one allowed class>"}}],
  "relations": [{{"s": "subject concept", "p": "<one allowed predicate>", "o": "object concept"}}]}}

Rules: only real Jeju entities; use the allowed classes/predicates exactly; 3-8 concepts and
2-8 relations; concept names short (1-4 words); no text outside the JSON.

TEXT:
{text}

JSON:"""


def parse(raw: str) -> dict | None:
    if not raw:
        return None
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    s = t.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[s:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def clean(obj: dict, rid) -> dict:
    concepts, relations = [], []
    for c in obj.get("concepts", []) or []:
        if not isinstance(c, dict):
            continue
        nm = str(c.get("name", "")).strip()
        cl = CLASS_SET.get(str(c.get("class", "")).strip().lower())
        if nm and 2 <= len(nm) <= 40 and cl:
            concepts.append({"name": nm, "class": cl})
    for r in obj.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        s, p, o = str(r.get("s", "")).strip(), str(r.get("p", "")).strip(), str(r.get("o", "")).strip()
        if s and o and s != o and p in PRED_SET:
            relations.append({"s": s, "p": p, "o": o})
    return {"rid": rid, "concepts": concepts, "relations": relations}


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser(description="LLM ontology extraction shard")
    ap.add_argument("--gpu", type=str, default="5")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--model-key", default="llama3.1-8b")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all in this shard")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"llm_shard{args.shard}.jsonl"

    # load this shard's records
    recs = []
    with open(RECORDS) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and i % args.num_shards == args.shard:
                recs.append(json.loads(line))
    if args.max_records:
        recs = recs[: args.max_records]

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["rid"])
                except Exception:
                    pass
    todo = [r for r in recs if r.get("record_id") not in done]
    logger.info("shard %d/%d: %d records, %d done, %d todo",
                args.shard, args.num_shards, len(recs), len(done), len(todo))

    from src.llm.client import LLMClient
    llm = LLMClient(backend="local", model_key=args.model_key,
                    max_tokens=args.max_tokens, temperature=0.3)
    if not llm.is_available:
        logger.error("LLM unavailable"); sys.exit(1)
    client = llm._client

    cls_s = ", ".join(CLASSES)
    prd_s = ", ".join(PREDICATES)
    n = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as w:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b: b + args.batch]
            prompts = [PROMPT.format(classes=cls_s, preds=prd_s,
                                     text=f"{r.get('Input','')} {r.get('Output','')}"[:1400])
                       for r in chunk]
            try:
                raws = client.call_batch(prompts, system=SYSTEM, temperature=0.3)
            except Exception as e:  # pragma: no cover
                logger.warning("batch err: %s", e); continue
            for r, raw in zip(chunk, raws):
                obj = parse(raw)
                rec = clean(obj, r.get("record_id")) if obj else {"rid": r.get("record_id"), "concepts": [], "relations": []}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            w.flush()
            if (b // args.batch) % 10 == 0:
                rate = n / max(time.time() - t0, 1e-9)
                eta = (len(todo) - n) / rate if rate else 0
                logger.info("shard %d: %d/%d (%.2f rec/s, ETA %.0f min)",
                            args.shard, n, len(todo), rate, eta / 60)

    logger.info("shard %d DONE: %d records in %.1f min → %s",
                args.shard, n, (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
