"""exp060 - give Simple RAG the training budget the ontology arm already had.

`cross_model_results.md` reports four models on which the ontology arm is worse
than Simple RAG on nothing measured. All four carry a 3:1 or 4:1 training
advantage: the companion project trained the ontology arm for three or four
passes over the 36,847 training records and left the Simple RAG baseline at one.
Its own receipts say so (`rag_passes: 1`). That confound cannot be removed by
re-scoring stored answers - the matching baseline does not exist - so it is
trained here.

What this does, per model: continue LoRA training of the companion project's
one-pass `__rag` adapter for exactly the number of further epochs that brings the
Simple RAG arm to the same total pass count as the ontology arm it will be
compared against.

  model          ontology passes   rag passes now   further epochs here
  llama3.2-1b    4                 1                3
  llama3.2-3b    3                 1                2
  qwen2.5-1.5b   3                 1                2
  qwen2.5-3b     4                 1                3

Everything else is held: the same training records, the same RAG contexts the
one-pass adapter saw, the same LoRA recipe, learning rate, scheduler, warmup,
sequence budget and seed, read out of the companion project's config rather than
restated. Only the epoch count changes, because only the epoch count is the
confound.

Two things this module refuses to do. It will not write anything into the
companion project - every output path is asserted to be under this repository
before training starts - and it will not initialise from an adapter that is not
the published one-pass RAG adapter, because continuing from the wrong weights
would produce a baseline that looks matched and is not.

  python3 experiments/exp060_matched_budget_rag/train_rag_extended.py \
      --model qwen2.5-3b --gpu 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CODEX = Path("/home/tahirahmad/Ontology_Project_Codex")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"

sys.path.insert(0, str(CODEX))

os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

E41 = CODEX / "experiments/exp041_small_models/outputs"
E56 = CODEX / "experiments/exp056_qwen/outputs"

# ontology passes the companion project gave each model, and therefore the total
# the Simple RAG arm has to reach. `source` is the published one-pass adapter.
PLAN = {
    "llama3.2-1b": dict(family="llama", onto_passes=4, source=E41 / "adapters_v2/llama3.2-1b__rag"),
    "llama3.2-3b": dict(family="llama", onto_passes=3, source=E41 / "adapters_v2/llama3.2-3b__rag"),
    "qwen2.5-1.5b": dict(family="qwen", onto_passes=3, source=E56 / "adapters/qwen2.5-1.5b__rag"),
    "qwen2.5-3b": dict(family="qwen", onto_passes=4, source=E56 / "adapters/qwen2.5-3b__rag"),
}


def configure(model_key: str):
    """Hand back the companion project's own config object for this family."""
    plan = PLAN[model_key]
    if plan["family"] == "qwen":
        from experiments.exp056_qwen.train import configure as qwen_configure
        return qwen_configure(model_key)
    from experiments.exp041_small_models.settings import configure as llama_configure
    from experiments.exp024_rag_vs_ontology import config as C
    llama_configure()
    return C


def dest_for(model_key: str) -> Path:
    plan = PLAN[model_key]
    return OUT / "adapters" / f"{model_key}__rag_pass{plan['onto_passes']}"


def guard_paths(dest: Path, source: Path) -> None:
    """The companion project is evidence for published numbers. Do not write to it."""
    resolved = dest.resolve() if dest.exists() else dest.parent.resolve() / dest.name
    if CODEX in resolved.parents or resolved == CODEX:
        raise SystemExit(f"refusing to write inside the companion project: {resolved}")
    if ROOT not in resolved.parents:
        raise SystemExit(f"output must live in this repository, got {resolved}")
    if not (source / "adapter_config.json").exists():
        raise SystemExit(f"missing source adapter: {source}")
    summary = source / "train_summary.json"
    if not summary.exists():
        raise SystemExit(f"source adapter has no training receipt: {summary}")
    prior = json.loads(summary.read_text(encoding="utf-8"))
    if prior.get("arm") != "rag":
        raise SystemExit(f"source adapter is not the RAG arm: {prior.get('arm')!r}")
    epochs = (prior.get("train_cfg") or {}).get("epochs")
    if epochs != 1:
        raise SystemExit(f"source adapter is not the one-pass baseline: epochs={epochs}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PLAN))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--micro-bs", type=int, default=None,
                    help="override only to fit a busy GPU; accumulation compensates "
                         "so the effective batch is unchanged")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training examples")
    a = ap.parse_args()

    plan = PLAN[a.model]
    source, dest = plan["source"], dest_for(a.model)
    guard_paths(dest, source)

    from experiments.exp024_rag_vs_ontology.gpu import validate_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = validate_gpu(a.gpu)

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    from experiments.exp024_rag_vs_ontology.chat import (build_training_example,
                                                         prepare_tokenizer,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.integrity import aligned, read_records
    from src.utils.proctitle import set_proctitle

    set_proctitle()
    C = configure(a.model)
    set_seed(C.TRAIN["seed"])

    prior = json.loads((source / "train_summary.json").read_text(encoding="utf-8"))
    done_passes = (prior.get("train_cfg") or {}).get("epochs", 1)
    extra = plan["onto_passes"] - done_passes
    if extra <= 0:
        raise SystemExit(f"{a.model}: nothing to do, rag already at {done_passes} passes")

    dest.mkdir(parents=True, exist_ok=True)
    ctx = C.CTX_DIR / "train_rag.jsonl"
    if not ctx.exists():
        raise SystemExit(f"missing RAG contexts: {ctx}")
    records = [json.loads(line) for line in ctx.open(encoding="utf-8") if line.strip()]
    if a.limit:
        records = records[: a.limit]
    if len({r["record_id"] for r in records}) != len(records):
        raise SystemExit("duplicate training ids in the RAG contexts")
    aligned({r["record_id"]: r for r in records}, read_records(C.TRAIN_JSONL),
            complete=not bool(a.limit))

    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(source, local_files_only=True),
                            for_training=True)
    sys_ok = supports_system_role(tok)
    print(f"[{a.model}] continuing the RAG arm from {done_passes} to "
          f"{plan['onto_passes']} passes (+{extra} epochs), {len(records):,} records, "
          f"system_role={sys_ok}", flush=True)

    started_tok = time.time()
    feats = []
    for index, row in enumerate(records):
        example = build_training_example(tok, row["Input"], row["Output"], "rag",
                                         row.get("context"), sys_ok, C.TRAIN["max_len"])
        if example is None:
            raise SystemExit(f"answer exceeds the token budget for {row['record_id']}; "
                             f"refusing to drop data silently")
        example["length"] = len(example["input_ids"])
        feats.append(example)
        if index and index % 10000 == 0:
            print(f"  tokenised {index:,}/{len(records):,} "
                  f"({time.time() - started_tok:.0f}s)", flush=True)
    lens = [f["length"] for f in feats]
    print(f"[{a.model}] {len(feats):,} examples, len mean={sum(lens) // len(lens)} "
          f"max={max(lens)} ({time.time() - started_tok:.0f}s)", flush=True)

    pad_id = tok.pad_token_id

    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        ids, labels, mask = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}

    micro = a.micro_bs or C.TRAIN["micro_bs"]
    accum = C.TRAIN["grad_accum"] * C.TRAIN["micro_bs"] // micro
    cfg = C.MODEL_BY_KEY[a.model]
    print(f"[{a.model}] loading {cfg['hf_id']} on GPU {a.gpu} "
          f"(micro_bs={micro}, grad_accum={accum})", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True)
    model = PeftModel.from_pretrained(base, str(source), is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(dest / "hf"),
            num_train_epochs=float(extra),
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=accum,
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
        ),
        train_dataset=feats,
        data_collator=collate,
    )
    # This box is shared and these runs take tens of hours each; HF writes a
    # checkpoint every save_steps, so a crash costs the last few hundred steps
    # rather than the whole pass.
    resume = any((dest / "hf").glob("checkpoint-*")) if (dest / "hf").exists() else False
    if resume:
        print(f"[{a.model}] resuming from the last checkpoint in {dest / 'hf'}",
              flush=True)
    started = time.time()
    result = trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))

    summary = {
        "model": a.model,
        "arm": "rag",
        "purpose": "matched training budget for the extended-ontology comparison",
        "initialized_from": str(source),
        "initializer_passes": done_passes,
        "additional_epochs": extra,
        "total_rag_passes": plan["onto_passes"],
        "matches_ontology_passes": plan["onto_passes"],
        "n_examples": len(feats),
        "micro_bs": micro,
        "grad_accum": accum,
        "effective_batch": micro * accum,
        "train_cfg": dict(C.TRAIN),
        "train_runtime_s": round(time.time() - started, 1),
        "train_loss": result.metrics.get("train_loss"),
        "contexts": str(ctx),
        "wrote": str(dest),
    }
    (dest / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"[{a.model}] done: rag now at {plan['onto_passes']} passes, "
          f"loss={summary['train_loss']}, {summary['train_runtime_s'] / 3600:.2f} h "
          f"-> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
