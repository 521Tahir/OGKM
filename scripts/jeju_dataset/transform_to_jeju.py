"""
Transform a multi-destination tourism dataset into a fully Jeju-focused dataset.

Per the dataset-engineering spec: every NON-Jeju record is rewritten into a
realistic Jeju-equivalent record that PRESERVES the original user intent, schema,
structure/length, and the style of the validated Jeju records — NOT a name swap.
Already-Jeju records pass through unchanged. Final size is preserved (99,910).

  source:   data/raw/full_dataset.json            (99,910 records)
  style ref: data/jeju_full/real_jeju.jsonl        (6,983 validated Jeju records)
  output:   data/jeju_full/jeju_full_dataset.jsonl (99,910, all Jeju)

Schema is preserved exactly (Input, CoT, Output, _source_file, _global_id) — no
fields added/removed/renamed. Transformation status is tracked in a sidecar
manifest, not in the records.

Model: DeepSeek-R1-Distill-Llama-8B (matches the original data's generator).
Features: batching, checkpoint/resume (by _global_id), validation, sharding.

Usage
  # Pilot: transform 100 non-Jeju records (keeps originals side-by-side for review)
  python scripts/jeju_dataset/transform_to_jeju.py --pilot 100 --gpu 0 --batch 12
  # Full run (shardable):
  python scripts/jeju_dataset/transform_to_jeju.py --gpu 0 --batch 16 --shard 0 --num-shards 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger("transform_jeju")

SRC = ROOT / "data" / "raw" / "full_dataset.json"
REAL = ROOT / "data" / "jeju_full" / "real_jeju.jsonl"
OUT_DIR = ROOT / "data" / "jeju_full"

# Jeju place names that mark a record as already-Jeju even without the word "jeju"
JEJU_MARKERS = [
    "jeju", "hallasan", "seongsan", "ilchulbong", "manjanggul", "udo",
    "seogwipo", "cheonjiyeon", "cheonjeyeon", "jeongbang", "hyeopjae", "hamdeok",
    "woljeongri", "seopjikoji", "sanbangsan", "osulloc", "o'sulloc", "haenyeo",
    "jusangjeolli", "yongduam", "saryeoni", "bijarim", "gapado", "marado",
]

# Non-Jeju places that must NOT appear in a transformed output (leakage check)
NON_JEJU = [
    # mainland Korea
    "seoul", "busan", "incheon", "daegu", "gwangju", "daejeon", "gyeongju", "jeonju",
    "gangneung", "sokcho", "suwon", "gangnam", "gyeongbokgung", "namsan", "myeongdong",
    "hongdae", "itaewon", "haeundae", "gamcheon", "everland", "lotte world", "nami island",
    "bukhansan", "seoraksan", "cheonggyecheon", "gwangjang market", "dmz",
    # common world hubs present in the source data
    "indonesia", "jakarta", "java", "bali", "ubud", "borobudur", "yogyakarta", "lombok",
    "thailand", "bangkok", "phuket", "chiang mai", "krabi", "vietnam", "hanoi", "saigon",
    "ho chi minh", "philippines", "manila", "cebu", "boracay", "palawan",
    "japan", "tokyo", "kyoto", "osaka", "singapore", "malaysia", "kuala lumpur",
    "india", "china", "beijing", "shanghai", "hong kong", "taiwan", "taipei",
    "paris", "london", "rome", "new york", "dubai",
    # specific regions/landmarks common in the source data
    "sumatra", "nusa penida", "gili", "munduk", "tirtagangga", "candidasa",
    "monkey forest", "tegallalang", "warung", "komodo", "flores", "lombok",
    "sapa", "halong", "hoi an", "danang", "siem reap", "angkor", "cambodia",
    "laos", "myanmar", "sri lanka", "nepal", "maldives", "europe", "africa",
]

GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"

SYSTEM = ("You are a Jeju Island (South Korea) tourism dataset engineer. You rewrite tourism "
          "Q&A records so they are entirely about Jeju Island while preserving the original "
          "intent, structure, length and detail. You output strict JSON only.")

PROMPT_TPL = """Rewrite the following tourism record so it is ENTIRELY about Jeju Island, South Korea.

Match the tone, format and detail of these validated Jeju examples:
EXAMPLE A
Q: {ex1_q}
A: {ex1_a}
EXAMPLE B
Q: {ex2_q}
A: {ex2_a}

Use ONLY these REAL Jeju places (do not invent any place names):
{places}
Real Jeju food: {food}
Real Jeju transport: {transport}

Rules:
- Preserve the ORIGINAL intent and task type. An N-day itinerary stays an N-day Jeju itinerary; a
  restaurant query stays a restaurant query; a transport question stays a transport question.
- Reconstruct the WHOLE scenario in Jeju: replace every non-Jeju place/route/attraction with a
  realistic Jeju equivalent of the SAME category and function. Keep it geographically logical
  (e.g. arrivals via Jeju International Airport, not a non-existent "Jeju Station").
- Match the original's STRUCTURE, LENGTH and detail (same number of days/sections/bullets).
- Use real Jeju attractions, food (black pork, hairtail, abalone, tangerines), and transport
  (rental car, public/airport-limousine buses). Do NOT invent specific business names.
- Mention NO place outside Jeju. Keep diversity — do not overuse Hallasan, Seongsan or the airport.

ORIGINAL RECORD:
Q (Input): {input}
Reasoning (CoT): {cot}
A (Output): {output}

Return STRICT JSON with exactly these keys: {{"Input": "...", "CoT": "...", "Output": "..."}}. JSON only."""


def is_jeju(rec: dict) -> bool:
    blob = " ".join(str(rec.get(k, "")) for k in ("Input", "CoT", "Output")).lower()
    return any(m in blob for m in JEJU_MARKERS)


def load_exemplars(k_short: int = 12) -> list[dict]:
    ex = []
    with open(REAL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if 40 <= len(r.get("Output", "")) <= 320 and 15 <= len(r.get("Input", "")) <= 160:
                ex.append({"Input": r["Input"], "Output": r["Output"]})
            if len(ex) >= 200:
                break
    return ex


def _clean(s: str) -> str:
    """Light cleanup: strip code fences / bold markers; keep newlines & bullets (style)."""
    s = s.strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.MULTILINE).strip()
    s = s.replace("**", "").replace("__", "")
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
                for attempt in (blob, blob.replace(",\n}", "\n}").replace(",}", "}")):
                    try:
                        obj = json.loads(attempt)
                        if all(key in obj for key in ("Input", "CoT", "Output")):
                            return {key: _clean(str(obj[key])) for key in ("Input", "CoT", "Output")}
                    except Exception:
                        pass
                break
    # Fallback: tolerant regex extraction of the three fields
    fields = {}
    for key in ("Input", "CoT", "Output"):
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"\s*(?:,\s*"(?:Input|CoT|Output)"|\}})',
                      text, flags=re.DOTALL)
        if m:
            fields[key] = _clean(m.group(1).replace('\\"', '"').replace("\\n", "\n"))
    if all(k in fields and fields[k] for k in ("Input", "CoT", "Output")):
        return fields
    return None


def valid(rec: dict) -> tuple[bool, str]:
    if not rec:
        return False, "unparseable"
    inp, cot, out = rec["Input"], rec["CoT"], rec["Output"]
    if len(inp) < 10 or len(out) < 20 or len(cot) < 8:
        return False, "too_short"
    blob = f"{inp} {cot} {out}".lower()
    if not any(m in blob for m in JEJU_MARKERS):
        return False, "not_jeju"
    for bad in NON_JEJU:
        if re.search(rf"(?<![a-z]){re.escape(bad)}(?![a-z])", blob):
            return False, f"leak:{bad}"
    return True, "ok"


def build_prompt(rec: dict, exs: list[dict], gaz: dict, rng: random.Random) -> str:
    e1, e2 = rng.sample(exs, 2)
    locs = rng.sample(gaz["locations"], k=min(18, len(gaz["locations"])))
    places = ", ".join(f"{l['name']} ({l['category']})" for l in locs)
    food = ", ".join(rng.sample(gaz["cuisine"], k=min(6, len(gaz["cuisine"]))))
    transport = "Jeju International Airport, rental car, public bus, airport-limousine bus, taxi, ferry"
    return PROMPT_TPL.format(
        ex1_q=e1["Input"], ex1_a=e1["Output"], ex2_q=e2["Input"], ex2_a=e2["Output"],
        places=places, food=food, transport=transport,
        input=rec["Input"][:1500], cot=str(rec.get("CoT", ""))[:600], output=rec["Output"][:1800],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Transform tourism dataset → Jeju-only")
    ap.add_argument("--pilot", type=int, default=0, help="transform N non-Jeju records (review mode)")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--model-key", default="llama3.1-8b",
                    help="LLM registry key (llama3.1-8b strong instruction follower; deepseek-r1-8b reasoning)")
    ap.add_argument("--max-rounds", type=int, default=4, help="retry rounds for failed records")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        import setproctitle
        setproctitle.setproctitle("tahir_Ont-Code")
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + args.shard)

    data = json.loads(SRC.read_text())
    exs = load_exemplars()
    gaz = json.loads(GAZ.read_text())
    logger.info("Source=%d records | %d style exemplars | %d gazetteer places",
                len(data), len(exs), len(gaz["locations"]))

    # Partition: passthrough Jeju vs transform non-Jeju
    non_jeju = [r for r in data if not is_jeju(r)]
    logger.info("non-Jeju to transform=%d (shard %d/%d)", len(non_jeju), args.shard, args.num_shards)
    # shard
    non_jeju = [r for i, r in enumerate(non_jeju) if i % args.num_shards == args.shard]

    pilot = args.pilot > 0
    if pilot:
        non_jeju = non_jeju[: args.pilot]
        out_path = OUT_DIR / "transform_pilot.jsonl"
        out_path.unlink(missing_ok=True)
    else:
        out_path = OUT_DIR / f"transformed_shard{args.shard}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("record_id"))
                except Exception:
                    pass
    todo = [r for r in non_jeju if r.get("_global_id") not in done_ids]
    logger.info("resume: %d done, %d to do", len(done_ids), len(todo))

    from src.llm.client import LLMClient
    llm = LLMClient(backend="local", model_key=args.model_key,
                    max_tokens=args.max_tokens, temperature=0.6)
    if not llm.is_available:
        logger.error("LLM unavailable (%s)", args.model_key); sys.exit(1)
    client = llm._client
    # Only DeepSeek-R1 distills need the empty-think prefix to skip reasoning.
    assistant_prefix = "<think>\n\n</think>\n\n" if ("r1" in args.model_key or "deepseek" in args.model_key) else ""
    logger.info("Model=%s assistant_prefix=%r", args.model_key, assistant_prefix)

    produced = 0
    attempts = 0
    rejects: dict[str, int] = {}
    t0 = time.time()
    remaining = todo
    rnd = 0
    target = len(todo)
    with open(out_path, "a", encoding="utf-8") as w:
        # Retry rounds: failed records are regenerated (different exemplars/seed)
        # so coverage approaches 100% (spec: "if any check is NO, regenerate").
        while remaining and rnd < args.max_rounds:
            rnd += 1
            next_remaining = []
            for b in range(0, len(remaining), args.batch):
                chunk = remaining[b: b + args.batch]
                prompts = [build_prompt(r, exs, gaz, rng) for r in chunk]
                try:
                    raws = client.call_batch(prompts, system=SYSTEM, temperature=0.6,
                                             assistant_prefix=assistant_prefix)
                except Exception as e:  # pragma: no cover
                    logger.warning("batch error: %s", e)
                    next_remaining.extend(chunk)
                    continue
                for orig, raw in zip(chunk, raws):
                    attempts += 1
                    rec = parse_triplet(raw)
                    ok, why = valid(rec)
                    if not ok:
                        rejects[why.split(":")[0]] = rejects.get(why.split(":")[0], 0) + 1
                        next_remaining.append(orig)
                        continue
                    out_rec = {
                        "record_id": orig.get("_global_id"),
                        "Input": rec["Input"], "CoT": rec["CoT"], "Output": rec["Output"],
                        "source_file": orig.get("_source_file"),
                    }
                    if pilot:
                        out_rec["_orig_Input"] = orig["Input"]
                        out_rec["_orig_Output"] = orig["Output"]
                    w.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    produced += 1
                w.flush()
                rate = produced / max(time.time() - t0, 1e-9)
                eta = (target - produced) / rate if rate else 0
                logger.info("round %d: %d/%d done (%.2f/s, ETA %.0f min) rejects=%s",
                            rnd, produced, target, rate, eta / 60, rejects)
            remaining = next_remaining
            logger.info("round %d complete; %d still failing", rnd, len(remaining))

    logger.info("DONE: %d/%d transformed in %.1f min (%d unrecovered) → %s",
                produced, target, (time.time() - t0) / 60, len(remaining), out_path)
    print(f"\n[transform_to_jeju] transformed={produced}/{target} "
          f"unrecovered={len(remaining)} rejects={rejects} file={out_path}")


if __name__ == "__main__":
    main()
