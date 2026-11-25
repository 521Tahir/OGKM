"""
Score generated answers against the dataset's own `Output` field.

Metrics (identical definitions to streamlit_app/evaluation/metrics.py, batched
for the full test set):
  BERT-F1           roberta-large BERTScore F1
  METEOR            NLTK meteor_score
  COMET             Unbabel/wmt22-comet-da (src = question)
  Key-token         fraction of salient gold tokens (months + named entities)
                    present in the candidate
  Response Accuracy LLM-judge 0-10 -> [0,1], one fixed judge for every cell

Usage:
  python -m experiments.exp024_rag_vs_ontology.score --gpu 4 --metrics all
  python -m experiments.exp024_rag_vs_ontology.score --gpu 4 --metrics judge
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _cells(split: str) -> list[Path]:
    from . import config as C
    return sorted(C.GEN_DIR.glob(f"{split}__*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metrics", nargs="+",
                    default=["all"], help="all | bertf1 meteor comet keytok judge")
    ap.add_argument("--cells", nargs="*", default=None, help="restrict to <model>__<arm> cells")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from . import config as C

    sys.path.insert(0, str(C.ROOT))
    from src.utils.proctitle import set_proctitle
    from streamlit_app.evaluation.metrics import (derive_key_tokens, key_token_score,
                                                  meteor_score_one, parse_judge)
    set_proctitle()

    want = set(C.METRICS if "all" in a.metrics else a.metrics)
    files = _cells(a.split)
    if a.cells:
        files = [f for f in files if any(c in f.name for c in a.cells)]
    if not files:
        raise SystemExit("no generation files found — run generate.py first")
    print(f"scoring {len(files)} cells: {[f.stem for f in files]}", flush=True)

    # ── shared model handles, loaded once for all cells ───────────────────
    handles: dict = {}

    def bert_f1(cands, refs):
        if "bert" not in handles:
            from bert_score import BERTScorer
            handles["bert"] = BERTScorer(model_type=C.BERTSCORE_MODEL, lang="en",
                                         rescale_with_baseline=False, batch_size=16,
                                         device="cuda")
        _, _, f = handles["bert"].score(cands, refs, verbose=False)
        return [round(float(x), 4) for x in f.tolist()]

    def comet(srcs, cands, refs):
        if "comet" not in handles:
            from comet import download_model, load_from_checkpoint
            handles["comet"] = load_from_checkpoint(download_model(C.COMET_MODEL))
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, cands, refs)]
        out = handles["comet"].predict(data, batch_size=16, gpus=1, progress_bar=False)
        return [round(float(x), 4) for x in out.scores]

    def judge(qs, refs, answers):
        """One fixed judge model scores every cell, so scores are comparable."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .chat import prepare_tokenizer, supports_system_role
        if "judge" not in handles:
            for k in ("bert", "comet"):        # reclaim before the 8B judge loads
                handles.pop(k, None)
            torch.cuda.empty_cache()
            hf_id = C.MODEL_BY_KEY[C.JUDGE_MODEL_KEY]["hf_id"]
            tok = prepare_tokenizer(AutoTokenizer.from_pretrained(hf_id), for_training=False)
            m = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to("cuda").eval()
            handles["judge"] = (tok, m, supports_system_role(tok))
        tok, m, sys_ok = handles["judge"]

        prompts = []
        for q, r, ans in zip(qs, refs, answers):
            u = C.JUDGE.format(q=q, ref=r[:C.JUDGE_REF_CHARS], ans=(ans or "")[:C.JUDGE_ANS_CHARS])
            msgs = ([{"role": "system", "content": C.JUDGE_SYSTEM}, {"role": "user", "content": u}]
                    if sys_ok else [{"role": "user", "content": f"{C.JUDGE_SYSTEM}\n\n{u}"}])
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        scores, bs = [], 8
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        buf = [None] * len(prompts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = m.generate(**enc, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for b, i in enumerate(idx):
                txt = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                buf[i] = parse_judge(txt)
        scores = buf
        return scores

    # ── per-cell scoring ──────────────────────────────────────────────────
    for path in files:
        cell = path.stem.split("__", 1)[1]
        out_path = C.SCORE_DIR / f"{a.split}__{cell}.jsonl"
        # A run killed mid-write leaves a blank or truncated line behind; skip
        # those rather than aborting the whole scoring pass on one bad row.
        rows = []
        for _line in open(path, encoding="utf-8"):
            if not _line.strip():
                continue
            try:
                rows.append(json.loads(_line))
            except json.JSONDecodeError:
                continue
        prev = {}
        if out_path.exists() and not a.force:
            for l in open(out_path, encoding="utf-8"):
                d = json.loads(l)
                prev[d["record_id"]] = d
        missing = [k for k in want if any(k not in prev.get(r["record_id"], {}) for r in rows)]
        if not missing:
            print(f"  {cell}: all metrics present — skip", flush=True)
            continue
        print(f"\n  {cell}: n={len(rows):,}  computing {sorted(missing)}", flush=True)

        qs = [r["Input"] for r in rows]
        refs = [r["reference"] for r in rows]
        cands = [(r["answer"] or "").strip() or "[empty]" for r in rows]
        res = {r["record_id"]: dict(prev.get(r["record_id"], {})) for r in rows}
        for r in rows:
            res[r["record_id"]].update(record_id=r["record_id"], model=r["model"], arm=r["arm"])

        if "keytok" in missing:
            t0 = time.time()
            for r, c in zip(rows, cands):
                res[r["record_id"]]["keytok"] = key_token_score(c, derive_key_tokens(r["reference"]))
            print(f"    keytok  {time.time()-t0:.0f}s", flush=True)
        if "meteor" in missing:
            t0 = time.time()
            for r, c in zip(rows, cands):
                res[r["record_id"]]["meteor"] = meteor_score_one(c, r["reference"])
            print(f"    meteor  {time.time()-t0:.0f}s", flush=True)
        if "bertf1" in missing:
            t0 = time.time()
            for r, v in zip(rows, bert_f1(cands, refs)):
                res[r["record_id"]]["bertf1"] = v
            print(f"    bertf1  {time.time()-t0:.0f}s", flush=True)
        if "comet" in missing:
            t0 = time.time()
            for r, v in zip(rows, comet(qs, cands, refs)):
                res[r["record_id"]]["comet"] = v
            print(f"    comet   {time.time()-t0:.0f}s", flush=True)
        if "judge" in missing:
            t0 = time.time()
            for r, v in zip(rows, judge(qs, refs, cands)):
                res[r["record_id"]]["judge"] = v
            print(f"    judge   {time.time()-t0:.0f}s", flush=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(res[r["record_id"]], ensure_ascii=False) + "\n")
        summary = {k: round(statistics.mean([res[r["record_id"]][k] for r in rows
                                             if isinstance(res[r["record_id"]].get(k), (int, float))]), 4)
                   for k in C.METRICS
                   if any(isinstance(res[r["record_id"]].get(k), (int, float)) for r in rows)}
        print(f"    -> {out_path.name}  {summary}", flush=True)


if __name__ == "__main__":
    main()
