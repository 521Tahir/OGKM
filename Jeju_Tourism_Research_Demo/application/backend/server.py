"""
Optional live-inference backend for the Jeju Tourism Intelligence demo.

The single-page application works with no server: it reads the recorded traces
embedded in the page and is labelled RECORDED RUN. This backend adds genuine live
inference for arbitrary questions, and the page labels those responses LIVE RUN.

It loads the same base model and the same arm-dedicated adapters used to record
the traces, so a live answer is directly comparable to a recorded one.

    pip install fastapi uvicorn
    python Jeju_Tourism_Research_Demo/application/backend/server.py --gpu 1
    # then open the page and POST to http://127.0.0.1:8077/ask

GPU note: loading both adapters needs roughly 2 x 17 GB in bf16 for an 8B model.
Pass --model llama3.2-1b for a small-footprint demo machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ["EXP024_TAG"] = "ded"
    sys.path.insert(0, str(ROOT))

    import torch
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder)

    cfg = C.MODEL_BY_KEY[a.model]
    print(f"[backend] loading retrievers …", flush=True)
    rag = RagContextBuilder(device="cuda"); rag.load()
    onto = OntoV3ContextBuilder(); onto.load()

    models, toks, sysok = {}, {}, {}
    for arm in ("rag", "onto_v3"):
        adapter = C.ADAPTER_DIR / f"{a.model}__{arm}"
        if not (adapter / "adapter_config.json").exists():
            print(f"[backend] missing adapter {adapter} — skipping arm {arm}")
            continue
        print(f"[backend] loading {arm} adapter …", flush=True)
        tk = prepare_tokenizer(AutoTokenizer.from_pretrained(
            str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]),
            for_training=False)
        m = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload().eval()
        models[arm], toks[arm], sysok[arm] = m, tk, supports_system_role(tk)

    app = FastAPI(title="Jeju Tourism Intelligence — live inference")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    class Ask(BaseModel):
        question: str
        max_new_tokens: int = 512

    @app.get("/health")
    def health():
        return {"ok": True, "model": cfg["name"], "arms": sorted(models),
                "ontology": C.ONTO_CLEAN_VERSION, "mode": "live"}

    @app.post("/ask")
    def ask(req: Ask):
        q = req.question.strip()
        out = {"question": q, "mode": "live", "model": cfg["name"], "arms": {}}

        t0 = time.perf_counter()
        rctx = rag.retrieve_batch([q])[0]
        r_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        octx = onto.retrieve_batch([q], n_entities=C.ONTO_V3_ENTITIES)[0]
        o_ms = (time.perf_counter() - t0) * 1000

        for arm, ctx, rms in (("rag", rctx["context"], r_ms),
                              ("onto_v3", octx["context"], o_ms)):
            if arm not in models:
                continue
            tk, m = toks[arm], models[arm]
            prompt = render_prompt(tk, q, arm, ctx, sysok[arm])
            enc = tk(prompt, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=req.max_new_tokens, do_sample=False,
                               pad_token_id=tk.pad_token_id or tk.eos_token_id)
            dt = time.perf_counter() - t0
            out["arms"][arm] = {
                "context": ctx, "prompt": prompt,
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "answer": tk.decode(g[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip(),
                "answer_tokens": int(g[0].shape[0] - enc["input_ids"].shape[1]),
                "retrieval_ms": round(rms, 1), "generation_s": round(dt, 2),
            }
        return out

    print(f"[backend] ready on http://{a.host}:{a.port}  arms={sorted(models)}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
