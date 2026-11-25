"""
WBS Task 3.2.1 / 4.1.2 - Does a better ontology context produce a better answer?

`improve_onto_context` raised the ontology arm's gold-entity coverage from 38.9%
to 53.1% with a question-dependent policy, and to 96.9% with a question-
INDEPENDENT static roster of the most-connected cards. Coverage is a proxy the
Task 4.1.2 ablation already showed to be weak, so this script runs the
shortlisted policies through real generation and scores them with the same
scorer the five variants were scored with.

Everything except the context is held at the published V3 settings: the same
dedicated `onto_v3` LoRA adapter, the same prompt template, greedy decoding,
max_new_tokens=512, the same 3,000-character budget and the same 347-item
benchmark. The `baseline` policy reproduces the published V3 context byte for
byte (asserted, not assumed), so a drift in its score would indict the harness
rather than the policy.

  python -m wbs.task3_retrieval.generate_tuned_onto --gpu 5 \
      --policies baseline routing_deep_hop2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
OUT = ROOT / "wbs" / "deliverables" / "task3_retrieval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    # the budget result is explained by a train/test mismatch - the adapters were
    # all fitted on full-budget contexts. Running the same cut on the vanilla
    # model is the falsification test for that explanation.
    ap.add_argument("--no-adapter", action="store_true",
                    help="skip the onto_v3 LoRA and run the base model")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["EXP024_TAG"] = "ded"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import OntoV3ContextBuilder
    from wbs.task3_retrieval.tuned_context import POLICIES, TunedOntoContext
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.score_variants import agg, score_item
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        bench = bench[: a.limit]
    qs = [b["Input"] for b in bench]
    print(f"benchmark: {len(bench)} items | policies: {', '.join(a.policies)}", flush=True)

    onto = TunedOntoContext(device="cuda").prepare(qs)
    CTX = {p: [c["context"] for c in onto.build_all(p)] for p in a.policies}

    # the baseline policy must reproduce the published V3 context exactly, or the
    # comparison is against something other than the published arm
    if "baseline" in a.policies:
        ref = OntoV3ContextBuilder(device="cuda"); ref.load()
        pub = [r["context"] for r in ref.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)]
        same = sum(1 for x, y in zip(pub, CTX["baseline"]) if x == y)
        print(f"baseline policy reproduces published V3 context on "
              f"{same}/{len(qs)} questions", flush=True)
        if same != len(qs):
            sys.exit(f"baseline policy diverges from the published V3 builder on "
                     f"{len(qs) - same} questions - fix before trusting the deltas")
        del ref
    del onto
    torch.cuda.empty_cache()

    cfg = C.MODEL_BY_KEY[a.model]
    adapter = C.ADAPTER_DIR / f"{a.model}__onto_v3"
    if a.no_adapter:
        adapter = Path("/nonexistent")
    src = str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
    sys_ok = supports_system_role(tok)
    print(f"loading {cfg['hf_id']}"
          + (" WITHOUT an adapter (falsification run)" if a.no_adapter
             else " + onto_v3 adapter") + " …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()

    def gen(text: str, max_new=512) -> str:
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    kb = KnowledgeBase()
    tag = f"__{a.tag}" if a.tag else ""
    summary_p = OUT / f"tuned_onto_generation__{a.model}{tag}.json"
    summary = json.loads(summary_p.read_text(encoding="utf-8")) if summary_p.exists() else {}

    for pol in a.policies:
        ans_p = OUT / f"tuned_onto__{a.model}__{pol}.jsonl"
        if pol in summary and ans_p.exists():
            print(f"[{pol}] already scored, skipping", flush=True)
            continue
        rows, t0 = [], time.perf_counter()
        for i, (b, ctx) in enumerate(zip(bench, CTX[pol]), 1):
            prompt = render_prompt(tok, b["Input"], "onto_v3", ctx, sys_ok)
            ts = time.perf_counter()
            draft = gen(prompt)
            rows.append({"bench_id": b["bench_id"], "category": b["category"],
                         "record_id": b.get("record_id"), "Input": b["Input"],
                         "reference": b["Output"], "answer": draft,
                         "ctx_chars": len(ctx),
                         "prompt_tokens": int(tok(prompt, return_tensors="pt")
                                              ["input_ids"].shape[1]),
                         "generation_s": round(time.perf_counter() - ts, 2)})
            if i % 50 == 0 or i == len(bench):
                el = time.perf_counter() - t0
                print(f"  [{pol}] {i}/{len(bench)}  {el/60:.1f} min "
                      f"(ETA {(el/i)*(len(bench)-i)/60:.1f} min)", flush=True)
        with open(ans_p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        items = [score_item(r, kb) for r in rows]
        o = agg(items)
        o.update({"policy": pol, "params": POLICIES[pol],
                  "wall_min": round((time.perf_counter() - t0) / 60, 1),
                  "per_item": {i["bench_id"]: {k: i[k] for k in
                                               ("token_f1", "entity_f1", "ungrounded_rate",
                                                "clean", "entity_hit")} for i in items}})
        summary[pol] = o
        summary_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  [{pol}] token F1 {o['token_f1']*100:.1f} | entity F1 "
              f"{o['entity_f1']*100:.1f} | ungrounded {o['ungrounded_rate']*100:.1f} "
              f"| {o['wall_min']} min", flush=True)

    print("\npolicy                      tokF1  entF1  entHit  ungr  clean")
    for pol, o in summary.items():
        print(f"  {pol:<26} {o['token_f1']*100:5.1f}  {o['entity_f1']*100:5.1f}  "
              f"{o['entity_hit']*100:5.1f}  {o['ungrounded_rate']*100:4.1f}  "
              f"{o['clean']*100:5.1f}")
    print(f"\nwrote {summary_p.name}")


if __name__ == "__main__":
    main()
