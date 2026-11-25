"""
Precompute the per-arm context blocks once, for every split.

Both fine-tuning and inference read these files, so the context a model is
trained on is produced by exactly the same retriever that serves it at test time.

  outputs/contexts/{train,val,test}_{rag,onto}.jsonl
    {"record_id", "Input", "Output", "context", "n_hits", ...}

Usage:
  python -m experiments.exp024_rag_vs_ontology.build_contexts --splits train val test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import config as C

sys.path.insert(0, str(C.ROOT))
from src.utils.proctitle import set_proctitle

SPLIT_FILES = {"train": C.TRAIN_JSONL, "val": C.VAL_JSONL, "test": C.TEST_JSONL,
               "bench": C.BENCH_JSONL}


def _read(split: str) -> list[dict]:
    return [json.loads(l) for l in open(SPLIT_FILES[split], encoding="utf-8")]


def build_rag(splits: list[str], force: bool) -> None:
    from .contexts import RagContextBuilder

    b = RagContextBuilder(device="cuda")
    t0 = time.time()
    print(f"[rag] building FAISS index over {C.TRAIN_JSONL.name} …", flush=True)
    b.build(force=force)
    b.load()
    print(f"[rag] index ready ({len(b._passages):,} passages) in {time.time()-t0:.0f}s", flush=True)

    for split in splits:
        out = C.CTX_DIR / f"{split}_rag.jsonl"
        if out.exists() and not force:
            print(f"[rag] {out.name} exists — skip", flush=True)
            continue
        recs = _read(split)
        # a train query must not retrieve its own passage
        excl = [r.get("record_id") for r in recs] if split == "train" else None
        t0, chunk = time.time(), 512
        with open(out, "w", encoding="utf-8") as f:
            for i in range(0, len(recs), chunk):
                part = recs[i:i + chunk]
                res = b.retrieve_batch([r["Input"] for r in part],
                                       exclude_ids=excl[i:i + chunk] if excl else None)
                for r, c in zip(part, res):
                    f.write(json.dumps({"record_id": r.get("record_id"), "Input": r["Input"],
                                        "Output": r["Output"], **c}, ensure_ascii=False) + "\n")
                if i % 5120 == 0:
                    print(f"[rag] {split} {i:,}/{len(recs):,} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[rag] {split} done -> {out} ({time.time()-t0:.0f}s)", flush=True)


def build_onto(splits: list[str], force: bool, version: str = None,
               tag: str = "onto", max_facts: int = None) -> None:
    from .contexts import OntoContextBuilder

    if max_facts:
        C.ONTO_MAX_FACTS = max_facts
    b = OntoContextBuilder(onto_version=version)
    print(f"[onto] ontology v{C.ONTO_VERSION}: {b.n_concepts:,} concepts, "
          f"{b.n_relations:,} relations", flush=True)
    for split in splits:
        out = C.CTX_DIR / f"{split}_{tag}.jsonl"
        if out.exists() and not force:
            print(f"[{tag}] {out.name} exists — skip", flush=True)
            continue
        recs = _read(split)
        t0 = time.time()
        with open(out, "w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                c = b.retrieve(r["Input"])
                f.write(json.dumps({"record_id": r.get("record_id"), "Input": r["Input"],
                                    "Output": r["Output"], **c}, ensure_ascii=False) + "\n")
                if i and i % 5000 == 0:
                    print(f"[onto] {split} {i:,}/{len(recs):,} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[onto] {split} done -> {out} ({time.time()-t0:.0f}s)", flush=True)


def build_onto_v2(splits: list[str], force: bool) -> None:
    from .contexts import OntoV2ContextBuilder

    b = OntoV2ContextBuilder(device="cuda")
    t0 = time.time()
    print("[onto_v2] building FAISS index over verbalised ontology facts …", flush=True)
    b.build(force=force)
    b.load()
    print(f"[onto_v2] index ready ({len(b._facts):,} facts) in {time.time()-t0:.0f}s", flush=True)

    for split in splits:
        out = C.CTX_DIR / f"{split}_onto_v2.jsonl"
        if out.exists() and not force:
            print(f"[onto_v2] {out.name} exists — skip", flush=True)
            continue
        recs = _read(split)
        t0, chunk = time.time(), 512
        with open(out, "w", encoding="utf-8") as f:
            for i in range(0, len(recs), chunk):
                part = recs[i:i + chunk]
                res = b.retrieve_batch([r["Input"] for r in part])
                for r, c in zip(part, res):
                    f.write(json.dumps({"record_id": r.get("record_id"), "Input": r["Input"],
                                        "Output": r["Output"], **c}, ensure_ascii=False) + "\n")
                if i % 5120 == 0:
                    print(f"[onto_v2] {split} {i:,}/{len(recs):,} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[onto_v2] {split} done -> {out} ({time.time()-t0:.0f}s)", flush=True)


def build_onto_v3(splits: list[str], force: bool) -> None:
    from .contexts import OntoV3ContextBuilder

    b = OntoV3ContextBuilder(device="cuda")
    t0 = time.time()
    print("[onto_v3] building FAISS index over entity cards …", flush=True)
    b.build(force=force)
    b.load()
    print(f"[onto_v3] index ready ({len(b._cards):,} cards) in {time.time()-t0:.0f}s", flush=True)

    for split in splits:
        out = C.CTX_DIR / f"{split}_onto_v3.jsonl"
        if out.exists() and not force:
            print(f"[onto_v3] {out.name} exists — skip", flush=True)
            continue
        recs = _read(split)
        t0, chunk = time.time(), 512
        with open(out, "w", encoding="utf-8") as f:
            for i in range(0, len(recs), chunk):
                part = recs[i:i + chunk]
                res = b.retrieve_batch([r["Input"] for r in part], n_entities=C.ONTO_V3_ENTITIES)
                for r, c in zip(part, res):
                    f.write(json.dumps({"record_id": r.get("record_id"), "Input": r["Input"],
                                        "Output": r["Output"], **c}, ensure_ascii=False) + "\n")
        print(f"[onto_v3] {split} done -> {out} ({time.time()-t0:.0f}s)", flush=True)


def build_onto_v45(splits: list[str], force: bool, variant: str) -> None:
    from experiments.exp026_onto_improve.onto_v4 import OntoV4ContextBuilder

    kw = C.ONTO_V4 if variant == "onto_v4" else C.ONTO_V5
    b = OntoV4ContextBuilder(onto_version=C.ONTO_CLEAN_VERSION, device="cuda", **kw)
    b.build()
    print(f"[{variant}] graph ready: {len(b.edges):,} edges, {len(b.card_names):,} cards, "
          f"{len(b.hubs)} hubs excluded", flush=True)
    for split in splits:
        out = C.CTX_DIR / f"{split}_{variant}.jsonl"
        if out.exists() and not force:
            print(f"[{variant}] {out.name} exists — skip", flush=True)
            continue
        recs = _read(split)
        t0, chunk = time.time(), 256
        with open(out, "w", encoding="utf-8") as f:
            for i in range(0, len(recs), chunk):
                part = recs[i:i + chunk]
                for r, c in zip(part, b.retrieve_batch([x["Input"] for x in part])):
                    c.pop("seeds", None)
                    f.write(json.dumps({"record_id": r.get("record_id"), "Input": r["Input"],
                                        "Output": r["Output"], **c}, ensure_ascii=False) + "\n")
                if i % 5120 == 0 and i:
                    print(f"[{variant}] {split} {i:,}/{len(recs):,} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{variant}] {split} done -> {out} ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--arms", nargs="+", default=["rag", "onto"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if "onto" in a.arms:
        build_onto(a.splits, a.force)
    if "onto_c" in a.arms:
        build_onto(a.splits, a.force, version=C.ONTO_CLEAN_VERSION, tag="onto_c",
                   max_facts=C.ONTO_C_MAX_FACTS)
    if "onto_v2" in a.arms:
        build_onto_v2(a.splits, a.force)
    if "onto_v3" in a.arms:
        build_onto_v3(a.splits, a.force)
    for v in ("onto_v4", "onto_v5"):
        if v in a.arms:
            build_onto_v45(a.splits, a.force, v)
    if "rag" in a.arms:
        build_rag(a.splits, a.force)

    print("\n── context stats ─────────────────────────────────────────")
    for split in a.splits:
        for arm in a.arms:
            p = C.CTX_DIR / f"{split}_{arm}.jsonl"
            if not p.exists():
                continue
            n, chars, hits, empty = 0, 0, 0, 0
            for l in open(p, encoding="utf-8"):
                d = json.loads(l)
                n += 1; chars += len(d["context"]); hits += d.get("n_hits", 0)
                empty += int(d.get("n_hits", 0) == 0)
            print(f"  {split:5s} {arm:4s}  n={n:6,}  ctx={chars//max(n,1):5,} chars  "
                  f"hits={hits/max(n,1):5.1f}  empty={empty}")


if __name__ == "__main__":
    main()
