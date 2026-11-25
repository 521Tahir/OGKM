"""
Assemble the final all-Jeju dataset from passthrough + transformed shards.

Combines:
  - already-Jeju source records (passed through, schema-normalised), and
  - transformed non-Jeju records (from transformed_shard*.jsonl),
into one dataset of up to 99,910 records in the VALIDATED structure
(record_id, Input, CoT, Output, source_file), then writes train/val/test splits.

Output:
  data/jeju_full/jeju_full_dataset.jsonl   (all records, record_id reassigned 0..N-1)
  data/jeju_full/jeju_full_dataset.json    (same, JSON list)
  data/jeju_full/splits/{train,val,test}.jsonl   (80/10/10)
  data/jeju_full/assembly_report.json

Usage:
  python scripts/jeju_dataset/assemble_jeju_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.jeju_dataset.transform_to_jeju import is_jeju, NON_JEJU, JEJU_MARKERS  # reuse logic
import ast
import re


def _flatten_obj(o) -> str:
    """Render a nested dict/list as readable prose."""
    if isinstance(o, dict):
        parts = []
        for k, v in o.items():
            sub = _flatten_obj(v)
            parts.append(f"{k}: {sub}" if not isinstance(v, (dict, list)) else f"{k} — {sub}")
        return " ".join(parts)
    if isinstance(o, list):
        return " ".join(_flatten_obj(x) for x in o)
    return str(o)


def to_prose(text: str) -> str:
    """If an Output is a serialized JSON/dict, flatten it to natural prose."""
    s = (text or "").strip()
    if not (s.startswith("{") or s.startswith("[")):
        return text
    for parser in (json.loads, ast.literal_eval):
        try:
            return re.sub(r"\s+", " ", _flatten_obj(parser(s))).strip()
        except Exception:
            continue
    return text


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

SRC = ROOT / "data" / "raw" / "full_dataset.json"
GEN_DIR = ROOT / "data" / "jeju_full"
OUT_JSONL = GEN_DIR / "jeju_full_dataset.jsonl"
OUT_JSON = GEN_DIR / "jeju_full_dataset.json"
SPLIT_DIR = GEN_DIR / "splits"


def norm_passthrough(rec: dict) -> dict:
    return {
        "record_id": None,  # reassigned later
        "Input": rec.get("Input", ""),
        "CoT": rec.get("CoT", ""),
        "Output": to_prose(rec.get("Output", "")),
        "source_file": rec.get("_source_file", rec.get("source_file", "")),
    }


def has_leak(rec: dict) -> bool:
    blob = f"{rec['Input']} {rec['CoT']} {rec['Output']}".lower()
    return any(re.search(rf"(?<![a-z]){re.escape(b)}(?![a-z])", blob) for b in NON_JEJU)


def main() -> None:
    data = json.loads(SRC.read_text())
    # Load transformed shards: record_id (orig global id) -> transformed record
    transformed: dict = {}
    shard_files = sorted(GEN_DIR.glob("transformed_shard*.jsonl"))
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    transformed[r["record_id"]] = r
                except Exception:
                    pass
    print(f"Loaded {len(transformed)} transformed records from {len(shard_files)} shards")

    final = []
    n_pass = n_trans = n_missing = 0
    for rec in data:
        if is_jeju(rec):
            final.append(norm_passthrough(rec))
            n_pass += 1
        else:
            gid = rec.get("_global_id")
            if gid in transformed:
                t = transformed[gid]
                final.append({
                    "record_id": None,
                    "Input": t["Input"], "CoT": t["CoT"], "Output": to_prose(t["Output"]),
                    "source_file": t.get("source_file", rec.get("_source_file", "")),
                })
                n_trans += 1
            else:
                n_missing += 1  # not yet transformed (run shards to completion)

    # ── Dedup by normalised Input (keep first occurrence) ──────────────────────
    n_before = len(final)
    seen_inp: set[str] = set()
    deduped = []
    for r in final:
        key = _norm(r["Input"])
        if key in seen_inp:
            continue
        seen_inp.add(key)
        deduped.append(r)
    n_removed = n_before - len(deduped)
    final = deduped

    # ── Drop records with any non-Jeju leakage → fully Jeju-focused set ─────────
    n_pre_leak = len(final)
    final = [r for r in final if not has_leak(r)]
    n_leak_dropped = n_pre_leak - len(final)

    # reassign sequential record_id
    for i, r in enumerate(final):
        r["record_id"] = i

    # leakage audit (sample)
    leak = sum(1 for r in final if has_leak(r))
    jeju_ok = sum(1 for r in final
                  if any(m in f"{r['Input']} {r['Output']}".lower() for m in JEJU_MARKERS))

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for r in final:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_JSON.write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")

    # splits 80/10/10
    rng = random.Random(42)
    idx = list(range(len(final)))
    rng.shuffle(idx)
    n = len(final)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in parts.items():
        with open(SPLIT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as w:
            for i in ids:
                w.write(json.dumps(final[i], ensure_ascii=False) + "\n")

    dictlike = sum(1 for r in final if r["Output"].lstrip().startswith(("{", "[")))
    report = {
        "total_final": len(final),
        "total_before_dedup": n_before,
        "duplicates_removed": n_removed,
        "leak_records_dropped": n_leak_dropped,
        "passthrough_jeju": n_pass,
        "transformed": n_trans,
        "missing_unrecovered": n_missing,
        "leakage_detected": leak,
        "jeju_grounded": jeju_ok,
        "dictlike_outputs_remaining": dictlike,
        "splits": {k: len(v) for k, v in parts.items()},
        "schema": ["record_id", "Input", "CoT", "Output", "source_file"],
    }
    (GEN_DIR / "assembly_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_JSONL} and splits/. (target 99,910; missing={n_missing})")


if __name__ == "__main__":
    main()
