"""
Generate test-set answers for one (model, arm) cell.

Arms:
  rag  / onto  — LoRA adapter tuned on that arm's contexts, served that context
  none         — LoRA adapter tuned closed-book, served no context
  base         — vanilla instruct model, no adapter, no context

Greedy decoding, identical settings for every cell. Resumable: reruns skip
record_ids already present in the output file.

Usage:
  python -m experiments.exp024_rag_vs_ontology.generate --model llama3.1-8b --arm rag --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    # choices come from config so adding an arm there is enough (this list used
    # to be hardcoded and silently rejected onto_v3)
    from . import config as _C
    ap.add_argument("--arm", required=True, choices=list(dict.fromkeys(_C.ALL_ARMS + ["onto_v2"])))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--adapter", default=None,
                    help="adapter arm to load (default: config.ADAPTER_FOR_ARM[arm])")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import config as C
    from .chat import prepare_tokenizer, render_prompt, strip_think, supports_system_role

    sys.path.insert(0, str(C.ROOT))
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    cfg = C.MODEL_BY_KEY[a.model]
    batch_size = a.batch or C.GEN_BATCH
    out_path = C.GEN_DIR / f"{a.split}__{a.model}__{a.arm}.jsonl"

    # ── inputs ────────────────────────────────────────────────────────────
    # Any arm that is not an explicit no-context control must be served its own
    # precomputed context. This used to read `if a.arm in ("rag", "onto")`, which
    # silently ran onto_v3/v4/v5 CLOSED-BOOK and made them look like failures.
    if a.arm not in ("none", "base"):
        src = C.CTX_DIR / f"{a.split}_{a.arm}.jsonl"
        if not src.exists():
            raise SystemExit(f"missing contexts: {src}")
        recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    else:
        f = {"test": C.TEST_JSONL, "val": C.VAL_JSONL, "train": C.TRAIN_JSONL,
             "bench": C.BENCH_JSONL}[a.split]
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        for r in recs:
            r["context"] = None
    if a.limit:
        recs = recs[: a.limit]

    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["record_id"])
            except Exception:
                pass
    todo = [r for r in recs if r.get("record_id") not in done]
    print(f"[{a.model}/{a.arm}] {len(todo):,} to generate ({len(done):,} already done)", flush=True)
    if not todo:
        return

    # ── model ─────────────────────────────────────────────────────────────
    # the served context (a.arm) is decoupled from the adapter that serves it:
    # rag and onto both run on the single "mixed" adapter unless overridden.
    adapter_arm = a.adapter or C.ADAPTER_FOR_ARM.get(a.arm)
    adapter = C.ADAPTER_DIR / f"{a.model}__{adapter_arm}"
    tok_src = str(adapter) if (a.arm != "base" and (adapter / "tokenizer_config.json").exists()) else cfg["hf_id"]
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(tok_src), for_training=False)
    sys_ok = supports_system_role(tok)

    print(f"[{a.model}/{a.arm}] loading {cfg['hf_id']} on GPU {a.gpu} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if a.arm != "base":
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"missing adapter: {adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()          # fold LoRA in for fast inference
    model.eval()
    model.config.use_cache = True

    # ── generation (length-sorted batches, original order restored on write) ──
    prompts = [render_prompt(tok, r["Input"], a.arm, r.get("context"), sys_ok) for r in todo]
    order = sorted(range(len(todo)), key=lambda i: len(prompts[i]))
    t0, n_done = time.time(), 0
    fout = open(out_path, "a", encoding="utf-8")
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        # This box is shared and other tenants grow after placement, so a fixed
        # batch size loses whole cells to OOM hours into a run. Halve the batch
        # and retry instead; a slower cell beats a dead one.
        texts, gen, enc = None, None, None
        sub = len(idx)
        while texts is None:
            try:
                parts_txt, parts_gen, prompt_len = [], [], 0
                for s0 in range(0, len(idx), sub):
                    chunk = idx[s0:s0 + sub]
                    enc = tok([prompts[i] for i in chunk], return_tensors="pt", padding=True,
                              truncation=True, max_length=C.GEN_MAX_PROMPT_TOKENS,
                              add_special_tokens=False).to("cuda")
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=C.MAX_NEW_TOKENS,
                                             do_sample=False, temperature=None, top_p=None,
                                             top_k=None, pad_token_id=tok.pad_token_id)
                    g = out[:, enc["input_ids"].shape[1]:]
                    prompt_len = int(enc["input_ids"].shape[1])
                    parts_gen.extend(list(g))
                    parts_txt.extend(tok.batch_decode(g, skip_special_tokens=True))
                    del out, enc
                texts, gen = parts_txt, parts_gen
                enc = {"input_ids": torch.zeros((1, prompt_len), dtype=torch.long)}
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if sub == 1:
                    print(f"[{a.model}/{a.arm}] OOM even at batch 1 - skipping batch", flush=True)
                    texts, gen = [], []
                    enc = {"input_ids": torch.zeros((1, 1), dtype=torch.long)}
                    idx = []
                    break
                sub = max(1, sub // 2)
                print(f"[{a.model}/{a.arm}] OOM - retrying at sub-batch {sub}", flush=True)
        for b, (i, txt) in enumerate(zip(idx, texts)):
            r = todo[i]
            fout.write(json.dumps({
                "record_id": r.get("record_id"), "model": a.model, "arm": a.arm,
                "adapter": adapter_arm,
                "Input": r["Input"], "reference": r["Output"],
                "answer": strip_think(txt),
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "gen_tokens": int((gen[b] != tok.pad_token_id).sum()),
            }, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += len(idx)
        if s % (batch_size * 20) == 0:
            el = time.time() - t0
            eta = el / max(n_done, 1) * (len(order) - n_done) / 60
            print(f"[{a.model}/{a.arm}] {n_done:,}/{len(order):,}  {el/60:.1f}m elapsed, "
                  f"ETA {eta:.0f}m", flush=True)
    fout.close()
    print(f"[{a.model}/{a.arm}] finished {n_done:,} in {(time.time()-t0)/60:.1f}m -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
