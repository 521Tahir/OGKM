"""
Jeju-specific synthetic data generation with DeepSeek-R1-Distill-Llama-8B.

Produces Input/CoT/Output triplets grounded in real Jeju places (lat/long from
the gazetteer), matching the schema/provenance of the original DeepSeek-made data.

Correctness safeguards (pilot review feedback)
  - location-first selection: the task is chosen to be COMPATIBLE with the anchor
    (no "dining at a mountain summit" or "scuba diving at an inland forest");
  - anti-fabrication prompt: the model must not invent specific business names and
    must not mention places outside Jeju;
  - validation rejects samples that name mainland-Korea places (leakage) or fail
    grounding/length checks.

Throughput
  - batched generation (--batch) via LocalLLMClient.call_batch (only GPU 0 free);
  - checkpoint/resume (append-only JSONL), dedup, partial save each batch;
  - GPU sharding (--shard i --num-shards N) to parallelise when GPUs free up.

Usage
  python scripts/jeju_dataset/generate_jeju_dataset.py --n 300 --gpu 0 --batch 12 --tag pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger("gen_jeju")

GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
OUT_DIR = ROOT / "data" / "jeju_full" / "generated"
REAL = ROOT / "data" / "jeju_full" / "real_jeju.jsonl"

# Which task types make sense for each location category (anchor-first design)
LOC_TASKS: dict[str, list[str]] = {
    "Beach":             ["attraction_recommendation", "water_activity", "safety", "weather_aware", "accessibility", "transportation", "itinerary"],
    "Island":            ["attraction_recommendation", "water_activity", "transportation", "itinerary", "safety", "weather_aware"],
    "Mountain":          ["attraction_recommendation", "hiking", "safety", "weather_aware", "accessibility", "transportation", "emergency", "itinerary"],
    "Trail":             ["hiking", "safety", "weather_aware", "attraction_recommendation", "accessibility", "itinerary"],
    "NaturalAttraction": ["attraction_recommendation", "activity", "safety", "weather_aware", "accessibility", "transportation", "emergency", "itinerary"],
    "Museum":            ["attraction_recommendation", "etiquette", "accessibility", "transportation", "emergency", "itinerary"],
    "CulturalAttraction":["attraction_recommendation", "etiquette", "accessibility", "transportation", "emergency", "itinerary"],
    "ReligiousSite":     ["attraction_recommendation", "etiquette", "accessibility", "transportation", "itinerary"],
    "Attraction":        ["attraction_recommendation", "activity", "accessibility", "transportation", "emergency", "itinerary"],
    "Market":            ["dining", "dietary", "attraction_recommendation", "etiquette", "transportation"],
    "City":              ["dining", "dietary", "transportation", "itinerary", "attraction_recommendation", "emergency", "accessibility", "etiquette"],
    "Region":            ["dining", "dietary", "transportation", "itinerary", "attraction_recommendation", "weather_aware", "activity"],
    "Transportation":    ["transportation", "emergency", "accessibility"],
}

TASK_DESC: dict[str, str] = {
    "itinerary": "create a short Jeju itinerary that includes the anchor location(s)",
    "attraction_recommendation": "recommend what to see or do at the anchor location",
    "dining": "recommend the kind of Jeju food to eat in the anchor area (refer to general areas/markets, not invented restaurant names)",
    "dietary": "advise what/where to eat in the anchor area for a specific dietary need (general areas, not invented restaurant names)",
    "transportation": "explain how to reach or get around the anchor location in Jeju",
    "hiking": "give hiking advice for the anchor mountain/trail (route, difficulty, time)",
    "water_activity": "advise on water activities (swimming/snorkeling/surfing/diving) at the anchor coast",
    "activity": "suggest suitable activities or experiences at the anchor location",
    "safety": "give safety advice for the anchor location or its activities",
    "weather_aware": "give season/weather-aware advice for visiting the anchor location",
    "accessibility": "give accessibility advice (mobility, families, seniors) for the anchor location",
    "etiquette": "explain local etiquette and dos and don'ts at the anchor location",
    "emergency": "give practical visiting info (opening hours, tickets, what to bring) for the anchor location",
}

# Mainland-Korea / non-Jeju place names that indicate leakage → reject sample
NON_JEJU = [
    "seoul", "busan", "incheon", "daegu", "gwangju", "daejeon", "gyeongju", "jeonju",
    "gangneung", "sokcho", "suwon", "gangnam", "gyeongbokgung", "namsan", "myeongdong",
    "hongdae", "itaewon", "haeundae", "gamcheon", "everland", "lotte world", "nami island",
    "bukhansan", "seoraksan",
]

PROMPT_TPL = """You are a knowledgeable Jeju Island (South Korea) travel expert creating ONE training example.

Anchor location(s) in Jeju (real, with coordinates):
{anchors}
Task: {task}
{extras}

Write ONE realistic example as STRICT JSON with exactly these keys:
  "Input":  a natural question a real traveler would ask about the anchor location(s) in Jeju,
  "CoT":    a concise expert reasoning trace (2-4 sentences),
  "Output": the final accurate, specific answer (2-5 sentences).

Hard rules:
- Everything must be about JEJU ISLAND only. Never mention any place outside Jeju.
- Do NOT invent specific business, restaurant, cafe, or hotel names. Refer to areas, markets,
  or the named anchor places generally (e.g. "restaurants around Dongmun Market").
- Be factually correct and consistent with what the anchor location actually is.
- Output must be concise and directly useful. Return ONLY the JSON object, nothing else.

JSON:"""


def _hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def load_gazetteer() -> dict:
    return json.loads(GAZ.read_text())


def build_seed(rng: random.Random, gaz: dict) -> dict:
    primary = rng.choice(gaz["locations"])
    tasks = LOC_TASKS.get(primary["category"], ["attraction_recommendation", "itinerary", "transportation"])
    task = rng.choice(tasks)
    locs = [primary]
    if task == "itinerary":
        pool = [l for l in gaz["locations"] if l["name"] != primary["name"]]
        locs += rng.sample(pool, k=rng.choice([1, 2]))
    extras = []
    if task in ("dining", "dietary"):
        extras.append(f"Cuisine focus: {rng.choice(gaz['cuisine'])}")
    if task == "dietary":
        extras.append(f"Dietary requirement: {rng.choice(gaz['dietary'])}")
    if task in ("itinerary", "activity", "attraction_recommendation"):
        extras.append(f"Traveler: {rng.choice(gaz['traveler_types'])}")
    if task in ("weather_aware", "safety", "itinerary", "hiking", "water_activity"):
        extras.append(f"Season: {rng.choice(gaz['seasons'])}")
    return {"task": task, "locations": locs, "extras": extras}


def build_prompt(seed: dict) -> str:
    anchors = "\n".join(
        f"  - {l['name']} ({l['category']}) at lat {l['lat']}, lon {l['lon']} — {l.get('note','')}"
        for l in seed["locations"]
    )
    extras = "\n".join(seed["extras"]) if seed["extras"] else ""
    return PROMPT_TPL.format(anchors=anchors, task=TASK_DESC[seed["task"]], extras=extras)


def _clean(s: str) -> str:
    """Strip markdown artifacts and collapse whitespace."""
    s = re.sub(r"\*\*|__|`|^#{1,6}\s*", "", s.strip())
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_triplet(raw: str) -> dict | None:
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).replace("</think>", "")
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                for attempt in (blob, blob.replace("\n", " ")):
                    try:
                        obj = json.loads(attempt)
                        if all(k in obj for k in ("Input", "CoT", "Output")):
                            return {k: _clean(str(obj[k])) for k in ("Input", "CoT", "Output")}
                    except Exception:
                        continue
                return None
    return None


def valid(rec: dict, loc_names: list[str]) -> tuple[bool, str]:
    if not rec:
        return False, "unparseable"
    inp, cot, out = rec["Input"], rec["CoT"], rec["Output"]
    if len(inp) < 12 or len(out) < 20 or len(cot) < 12:
        return False, "too_short"
    blob = f"{inp} {out}".lower()
    if "jeju" not in blob and not any(n.lower() in blob for n in loc_names):
        return False, "not_grounded"
    for bad in NON_JEJU:
        if re.search(rf"\b{re.escape(bad)}\b", blob):
            return False, f"non_jeju:{bad}"
    return True, "ok"


def existing_inputs(path: Path) -> set[str]:
    seen = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(_hash(json.loads(line)["Input"]))
                    except Exception:
                        pass
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Jeju-specific dataset with DeepSeek-R1")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=700)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"shard{args.shard}"
    out_path = OUT_DIR / f"gen_{tag}.jsonl"

    gaz = load_gazetteer()
    seen = existing_inputs(REAL) | existing_inputs(out_path)
    already = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    logger.info("Resume: %d existing; dedup pool=%d; target this run=%d (batch=%d)",
                already, len(seen), args.n, args.batch)

    from src.llm.client import LLMClient
    llm = LLMClient(backend="local", model_key="deepseek-r1-8b",
                    max_tokens=args.max_tokens, temperature=0.7)
    if not llm.is_available:
        logger.error("DeepSeek backend unavailable."); sys.exit(1)
    client = llm._client  # underlying LocalLLMClient (for call_batch)

    rng = random.Random(args.seed + args.shard * 100000 + already)
    produced = 0
    attempts = 0
    rejects: dict[str, int] = {}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as w:
        while produced < args.n:
            seeds = [build_seed(rng, gaz) for _ in range(args.batch)]
            prompts = [build_prompt(s) for s in seeds]
            try:
                raws = client.call_batch(prompts, system="You output only strict JSON.", temperature=0.7)
            except Exception as e:  # pragma: no cover
                logger.warning("batch error: %s", e); continue
            for seed, raw in zip(seeds, raws):
                attempts += 1
                rec = parse_triplet(raw)
                ok, why = valid(rec, [l["name"] for l in seed["locations"]])
                if not ok:
                    rejects[why.split(":")[0]] = rejects.get(why.split(":")[0], 0) + 1
                    continue
                h = _hash(rec["Input"])
                if h in seen:
                    rejects["dup"] = rejects.get("dup", 0) + 1
                    continue
                seen.add(h)
                rec.update({
                    "_provenance": "deepseek-r1-generated",
                    "_category": seed["task"],
                    "_anchors": [{"name": l["name"], "lat": l["lat"], "lon": l["lon"]} for l in seed["locations"]],
                    "_global_id": f"gen_{tag}_{already + produced}",
                    "_generated_at": datetime.now(timezone.utc).isoformat(),
                })
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                produced += 1
                if produced >= args.n:
                    break
            w.flush()
            rate = produced / max(time.time() - t0, 1e-9)
            eta = (args.n - produced) / rate if rate else 0
            logger.info("produced %d/%d (%.2f/s, accept=%.0f%%, ETA %.0f min) rejects=%s",
                        produced, args.n, rate, 100 * produced / max(attempts, 1), eta / 60, rejects)

    logger.info("DONE: %d samples in %.1f min → %s", produced, (time.time() - t0) / 60, out_path)
    print(f"\n[generate_jeju_dataset] produced={produced} file={out_path} "
          f"accept={100*produced/max(attempts,1):.0f}% rejects={rejects}")


if __name__ == "__main__":
    main()
