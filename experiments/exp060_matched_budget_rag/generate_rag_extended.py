"""exp060 - answers from the budget-matched Simple RAG arm, on the held-out test set.

The decoding settings here are not chosen; they are the ones the stored answers
this will be compared against were produced with, and the script asserts that
rather than trusting the comment. Greedy, 512 new tokens, 1,536-token prompt
budget, the companion project's system prompt and user template, and the same
`test_rag.jsonl` contexts the one-pass RAG answers used - checked by sha256
against the generation manifest the companion project wrote, where one exists.

A different prompt or a different context file would make the new baseline
incomparable with everything already published, in a way no later check would
catch, so both are verified before a single token is generated.

  python3 experiments/exp060_matched_budget_rag/generate_rag_extended.py \
      --model qwen2.5-3b --gpu 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CODEX = Path("/home/tahirahmad/Ontology_Project_Codex")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "generations"

sys.path.insert(0, str(CODEX))

os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MAX_PROMPT = 1536
MAX_NEW = 512
BATCH = 8
SPLIT = "test"

# The companion project's own manifest for the one-pass RAG answers. Where it
# exists, the context file and decoding settings used here must match it.
MANIFEST = {
    "llama3.2-1b": CODEX / "experiments/exp041_small_models/outputs/generations_v2/"
                           "test__llama3.2-1b__rag.manifest.json",
    "llama3.2-3b": CODEX / "experiments/exp041_small_models/outputs/generations_v2/"
                           "test__llama3.2-3b__rag.manifest.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["record_id"])
    return done


def check_manifest(model_key: str, ctx: Path, system: str) -> dict:
    """Refuse to generate under settings the published baseline did not use."""
    path = MANIFEST.get(model_key)
    if path is None or not path.exists():
        return {"status": "no manifest on record for this model"}
    man = json.loads(path.read_text(encoding="utf-8"))
    got = sha256(ctx)
    problems = []
    if man.get("contexts_sha256") and man["contexts_sha256"] != got:
        problems.append(f"context sha256 {got[:16]} != published "
                        f"{man['contexts_sha256'][:16]}")
    if man.get("max_new_tokens") not in (None, MAX_NEW):
        problems.append(f"max_new_tokens {MAX_NEW} != published {man['max_new_tokens']}")
    if man.get("max_prompt_tokens") not in (None, MAX_PROMPT):
        problems.append(f"max_prompt_tokens {MAX_PROMPT} != published "
                        f"{man['max_prompt_tokens']}")
    if man.get("decoding") not in (None, "greedy"):
        problems.append(f"decoding greedy != published {man['decoding']}")
    if man.get("system") and man["system"] != system:
        problems.append("system prompt differs from the published one")
    if problems:
        raise SystemExit("settings do not match the published baseline:\n  "
                         + "\n  ".join(problems))
    return {"status": "checked against the published manifest",
            "manifest": str(path.relative_to(CODEX)), "contexts_sha256": got}


def main() -> int:
    from train_rag_extended import PLAN, configure, dest_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PLAN))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args()

    adapter = dest_for(a.model)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"train it first: {adapter} has no adapter")
    receipt = json.loads((adapter / "train_summary.json").read_text(encoding="utf-8"))

    from experiments.exp024_rag_vs_ontology.gpu import validate_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = validate_gpu(a.gpu)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         strip_think)
    from src.utils.proctitle import set_proctitle

    set_proctitle()
    C = configure(a.model)
    ctx = C.CTX_DIR / f"{SPLIT}_rag.jsonl"
    if not ctx.exists():
        raise SystemExit(f"missing contexts: {ctx}")
    manifest_check = check_manifest(a.model, ctx, C.SYSTEM)
    print(f"[{a.model}] contexts {ctx}\n  {manifest_check}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{SPLIT}__{a.model}__rag_pass{receipt['total_rag_passes']}.jsonl"
    done = load_done(dest)
    rows = [json.loads(line) for line in ctx.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["record_id"] not in done]
    print(f"[{a.model}] {len(rows)} to generate, {len(done)} already done", flush=True)
    if not rows:
        print(f"[{a.model}] complete -> {dest}", flush=True)
        return 0

    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(adapter, local_files_only=True),
                            for_training=False)
    base = AutoModelForCausalLM.from_pretrained(
        C.MODEL_BY_KEY[a.model]["hf_id"], dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval().to("cuda")
    model.config.use_cache = True
    print(f"[{a.model}] rag @ {receipt['total_rag_passes']} passes loaded", flush=True)

    prompts = [render_prompt(tok, row["Input"], "rag", row.get("context"), True)
               for row in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    started = time.time()
    with dest.open("a", encoding="utf-8") as handle:
        for start in range(0, len(order), a.batch):
            idx = order[start:start + a.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_PROMPT,
                      add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**enc, max_new_tokens=MAX_NEW,
                                           do_sample=False, temperature=None, top_p=None,
                                           pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(generated[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for row_i, text in zip(idx, texts):
                row = rows[row_i]
                handle.write(json.dumps({
                    "record_id": row["record_id"],
                    "model": a.model,
                    "arm": "rag",
                    "rag_passes": receipt["total_rag_passes"],
                    "Input": row["Input"],
                    "reference": row.get("Output") or "",
                    "answer": strip_think(text),
                }, ensure_ascii=False) + "\n")
            handle.flush()
            finished = start + len(idx)
            if finished % (a.batch * 20) == 0 or finished == len(order):
                elapsed = time.time() - started
                eta = elapsed / finished * (len(order) - finished) / 60
                print(f"  {finished}/{len(order)} {elapsed / 60:.1f}m eta {eta:.0f}m",
                      flush=True)
            del generated, enc

    (OUT / f"{SPLIT}__{a.model}__rag_pass{receipt['total_rag_passes']}.manifest.json"
     ).write_text(json.dumps({
         "model": a.model, "arm": "rag",
         "rag_passes": receipt["total_rag_passes"],
         "adapter": str(adapter),
         "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
         "contexts": str(ctx), "contexts_sha256": sha256(ctx),
         "decoding": "greedy", "max_new_tokens": MAX_NEW,
         "max_prompt_tokens": MAX_PROMPT, "batch_size": a.batch,
         "dtype": "bfloat16", "system": C.SYSTEM,
         "published_baseline_check": manifest_check,
         "generation_s": round(time.time() - started, 1),
     }, indent=2) + "\n", encoding="utf-8")
    print(f"[{a.model}] complete -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
