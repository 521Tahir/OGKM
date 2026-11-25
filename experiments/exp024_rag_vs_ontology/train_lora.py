"""
LoRA fine-tune one base model for one arm.

Each arm is tuned on prompts carrying its OWN retrieved context, so training and
test prompt distributions match for both arms (RAFT-style). The LoRA recipe,
data order, seed and step budget are identical across arms — only the context
block differs.

Usage:
  python -m experiments.exp024_rag_vs_ontology.train_lora \
      --model llama3.1-8b --arm rag --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Arm names that can have a dedicated adapter, taken from config without importing
# torch/CUDA (config is a plain module; the heavy imports happen inside main()).
def _dedicated_arms() -> list[str]:
    from . import config as C
    return [a for a in C.ALL_ARMS if a not in ("none", "base")]


C_ARMS = _dedicated_arms()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    # Arm-dedicated adapters (exp037): every ontology arm was previously served by
    # the `mixed` adapter, which spends half its capacity learning to COPY from
    # retrieved prose. That behaviour misfires on a fact list, so the ontology arms
    # were being judged on weights tuned against them. Any arm with a
    # train_<arm>.jsonl context file can now get its own adapter.
    ap.add_argument("--arm", required=True,
                    choices=["mixed", "rag", "onto", "none"] + list(C_ARMS))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training examples")
    ap.add_argument("--epochs", type=float, default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    from . import config as C
    from .chat import build_training_example, prepare_tokenizer, supports_system_role

    sys.path.insert(0, str(C.ROOT))
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    cfg = C.MODEL_BY_KEY[a.model]
    out_dir = C.ADAPTER_DIR / f"{a.model}__{a.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(C.TRAIN["seed"])

    # ── data ──────────────────────────────────────────────────────────────
    if a.arm == "mixed":
        # one adapter serving both arms: each example carries EITHER its RAG or
        # its ontology context, 50/50, assigned by a fixed seed so every model
        # sees the identical assignment.
        import random
        by_arm = {}
        for arm in ("rag", "onto"):
            f = C.CTX_DIR / f"train_{arm}.jsonl"
            if not f.exists():
                raise SystemExit(f"missing contexts: {f} — run build_contexts first")
            by_arm[arm] = [json.loads(l) for l in open(f, encoding="utf-8")]
        if len(by_arm["rag"]) != len(by_arm["onto"]):
            raise SystemExit("rag/onto context files disagree in length")
        rng = random.Random(C.MIX_SEED)
        recs = []
        for i in range(len(by_arm["rag"])):
            arm = "rag" if rng.random() < 0.5 else "onto"
            r = dict(by_arm[arm][i]); r["ctx_arm"] = arm
            recs.append(r)
        n_rag = sum(1 for r in recs if r["ctx_arm"] == "rag")
        print(f"[{a.model}/mixed] curriculum: {n_rag:,} rag / {len(recs)-n_rag:,} onto", flush=True)
    elif a.arm == "none":
        recs = [json.loads(l) for l in open(C.TRAIN_JSONL, encoding="utf-8")]
        for r in recs:
            r["context"] = None
    else:
        f = C.CTX_DIR / f"train_{a.arm}.jsonl"
        if not f.exists():
            raise SystemExit(f"missing contexts: {f} — run build_contexts first")
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]

    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(cfg["hf_id"]), for_training=True)
    sys_ok = supports_system_role(tok)
    print(f"[{a.model}/{a.arm}] tokenising {len(recs):,} examples "
          f"(system_role={sys_ok}) …", flush=True)

    t0 = time.time()
    feats, skipped = [], 0
    for i, r in enumerate(recs):
        ex = build_training_example(tok, r["Input"], r["Output"], r.get("ctx_arm", a.arm),
                                    r.get("context"), sys_ok, C.TRAIN["max_len"])
        if ex is None:
            skipped += 1
            continue
        ex["length"] = len(ex["input_ids"])
        feats.append(ex)
        if i and i % 10000 == 0:
            print(f"  … {i:,}/{len(recs):,} ({time.time()-t0:.0f}s)", flush=True)
    lens = [f["length"] for f in feats]
    print(f"[{a.model}/{a.arm}] {len(feats):,} examples, {skipped} skipped, "
          f"len mean={sum(lens)//len(lens)} max={max(lens)} ({time.time()-t0:.0f}s)", flush=True)

    pad_id = tok.pad_token_id

    def collate(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * k)
            labels.append(b["labels"] + [-100] * k)
            mask.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}

    # ── model ─────────────────────────────────────────────────────────────
    print(f"[{a.model}/{a.arm}] loading {cfg['hf_id']} on GPU {a.gpu} …", flush=True)
    load_kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if C.LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        load_kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **load_kw)
    if C.LOAD_4BIT:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**C.LORA))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        num_train_epochs=a.epochs if a.epochs is not None else C.TRAIN["epochs"],
        per_device_train_batch_size=C.TRAIN["micro_bs"],
        gradient_accumulation_steps=C.TRAIN["grad_accum"],
        learning_rate=C.TRAIN["lr"],
        lr_scheduler_type=C.TRAIN["lr_scheduler"],
        warmup_ratio=C.TRAIN["warmup_ratio"],
        weight_decay=C.TRAIN["weight_decay"],
        logging_steps=C.TRAIN["logging_steps"],
        save_steps=C.TRAIN["save_steps"],
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=C.TRAIN["seed"],
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=feats, data_collator=collate)

    t0 = time.time()
    res = trainer.train()
    dt = time.time() - t0

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "train_summary.json").write_text(json.dumps({
        "model": a.model, "hf_id": cfg["hf_id"], "arm": a.arm,
        "n_examples": len(feats), "skipped": skipped,
        "train_runtime_s": round(dt, 1), "train_loss": res.metrics.get("train_loss"),
        "lora": C.LORA, "train_cfg": C.TRAIN, "load_4bit": C.LOAD_4BIT,
        "ontology_version": C.ONTO_VERSION,
        "embed_model": C.EMBED_MODEL, "rag_top_k": C.RAG_TOP_K,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[{a.model}/{a.arm}] done in {dt/3600:.2f}h  loss={res.metrics.get('train_loss')} "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
