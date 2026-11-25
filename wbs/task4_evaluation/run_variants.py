"""
WBS Task 4.1.2 - Evaluate the five system variants on the categorized benchmark.

Variants, exactly as named in the WBS:

  V1 llm                    no retrieval at all - the base model answers from
                            parametric memory (arm `base`, no adapter)
  V2 rag                    top-5 dense passages over the train split
  V3 ontology_only          entity cards + intent/class routing over the ontology
  V4 ontology_rag           both, at a matched 3,000-character budget (hybrid20)
  V5 ontology_rag_validated V4 followed by the Task 3.2.2 constrained reasoning
                            layer: validate -> repair -> trace -> uncertainty fallback

Every answer is produced by real inference. V1-V4 share base weights, decoding and
context budget; only retrieval differs. V5 differs from V4 only by the post-hoc
governance layer, so the V4->V5 delta isolates what validation buys.

  python -m wbs.task4_evaluation.run_variants --gpu 3 --model llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VARIANTS = [
    ("V1_llm", "base", "generations"),
    ("V2_rag", "rag", "generations_ded"),
    ("V3_ontology_only", "onto_v3", "generations_ded"),
    ("V4_ontology_rag", "hybrid20", "generations"),
    ("V5_ontology_rag_validated", "hybrid20", "generations"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-repairs", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--only", nargs="*", default=None)
    # V4/V5 are served by the shared `mixed` adapter by default, which is the
    # published setting. Pass hybrid20d to use the adapter trained on this exact
    # hybrid context instead, which removes the mixed-adapter confound.
    ap.add_argument("--hybrid-adapter", default="mixed",
                    choices=["mixed", "hybrid20d"])
    # the selective policy `selective_validation.md` measured: spend the
    # validation pass only on drafts with at least this many audit-grade
    # ungrounded spans. 0 is the published behaviour (validate everything).
    ap.add_argument("--min-violations", type=int, default=0)
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output files, so a re-run does not "
                         "overwrite the published ones")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["EXP024_TAG"] = "ded"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder, _truncate)
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase, constrained_answer
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    D = ROOT / "wbs" / "deliverables" / "task4_evaluation"
    bench = [json.loads(l) for l in
             open(D / "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        bench = bench[: a.limit]
    print(f"benchmark: {len(bench):,} items", flush=True)

    # ── contexts, built once for every benchmark question ─────────────────
    print("building retrieval contexts …", flush=True)
    qs = [b["Input"] for b in bench]
    rag = RagContextBuilder(device="cuda"); rag.load()
    rag_ctx = [r["context"] for r in rag.retrieve_batch(qs)]
    onto = OntoV3ContextBuilder(); onto.load()
    onto_ctx = [r["context"] for r in onto.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)]
    # hybrid at the tuned 20% fact share, matched total budget
    fact_budget = int(C.CTX_MAX_CHARS * 0.20)
    hyb_ctx = []
    for rc, oc in zip(rag_ctx, onto_ctx):
        facts = _truncate(oc if not oc.startswith("[no ontology") else "", fact_budget)
        passages = _truncate(rc, C.CTX_MAX_CHARS - len(facts))
        hyb_ctx.append((f"Ontology facts:\n{facts}\n\n" if facts else "") +
                       f"Retrieved reference passages:\n{passages}")
    CTX = {"base": [None] * len(bench), "rag": rag_ctx,
           "onto_v3": onto_ctx, "hybrid20": hyb_ctx}
    del rag, onto
    torch.cuda.empty_cache()
    print(f"  rag mean {sum(len(c) for c in rag_ctx)//len(bench):,} chars | "
          f"onto {sum(len(c) for c in onto_ctx)//len(bench):,} | "
          f"hybrid {sum(len(c) for c in hyb_ctx)//len(bench):,}", flush=True)

    kb = KnowledgeBase()
    cfg = C.MODEL_BY_KEY[a.model]
    todo = [v for v in VARIANTS if not a.only or v[0] in a.only]
    results = {}

    # V1 uses no adapter; V2/V3 use their dedicated adapters; V4/V5 the mixed one
    ADAPTER = {"base": None, "rag": "rag", "onto_v3": "onto_v3",
               "hybrid20": a.hybrid_adapter}
    DEDICATED = ("rag", "onto_v3", "hybrid20d")
    # NOTE: V1 legitimately wants adapter None, so "nothing loaded yet" needs a
    # sentinel distinct from None or the first model load is skipped entirely.
    UNSET = object()
    loaded, tok, model, sys_ok = UNSET, None, None, None

    for vname, arm, _ in todo:
        want = ADAPTER[arm]
        if want != loaded:
            if model is not None:
                del model; torch.cuda.empty_cache()
            tag = want or "no adapter (vanilla)"
            print(f"\nloading {cfg['hf_id']} + {tag} …", flush=True)
            adapter = (C.ADAPTER_DIR if want in DEDICATED
                       else Path(str(C.ADAPTER_DIR).replace("_ded", ""))) / f"{a.model}__{want}" \
                if want else None
            if adapter and not (adapter / "adapter_config.json").exists():
                sys.exit(f"adapter {adapter} not found - train it before running "
                         f"this arm, or the variant silently falls back to the "
                         f"vanilla model and the comparison is meaningless")
            src = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() \
                else cfg["hf_id"]
            tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
            sys_ok = supports_system_role(tok)
            model = AutoModelForCausalLM.from_pretrained(
                cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
            if adapter and (adapter / "adapter_config.json").exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
            model.eval(); loaded = want

        def gen(text: str, max_new=512) -> str:
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        rows, t0 = [], time.perf_counter()
        for i, (b, ctx) in enumerate(zip(bench, CTX[arm]), 1):
            prompt_arm = "rag" if arm == "hybrid20" else arm
            prompt = render_prompt(tok, b["Input"], prompt_arm, ctx, sys_ok)
            ts = time.perf_counter()
            draft = gen(prompt)
            gen_s = time.perf_counter() - ts
            rec = {"bench_id": b["bench_id"], "category": b["category"],
                   "record_id": b.get("record_id"), "Input": b["Input"],
                   "reference": b["Output"], "answer": draft,
                   "prompt_tokens": int(tok(prompt, return_tensors="pt")["input_ids"].shape[1]),
                   "generation_s": round(gen_s, 2)}
            if vname.startswith("V5"):
                cr = constrained_answer(b["Input"], draft, kb,
                                        regenerate=lambda p: gen(p, 400),
                                        max_repairs=a.max_repairs,
                                        min_coverage=a.min_coverage,
                                        min_violations=a.min_violations)
                rec.update({"answer": cr["answer"], "draft": draft,
                            "violations": cr["violations"],
                            "repair_rounds": cr["repair_rounds"],
                            "grounded_coverage": cr["grounded_coverage"],
                            "uncertainty_fallback": cr["uncertainty_fallback"],
                            "explanation_trace": cr["explanation_trace"],
                            "validation_ms": cr["latency_ms"]})
            rows.append(rec)
            if i % 50 == 0 or i == len(bench):
                el = time.perf_counter() - t0
                print(f"  [{vname}] {i}/{len(bench)}  {el/60:.1f} min "
                      f"(ETA {(el/i)*(len(bench)-i)/60:.1f} min)", flush=True)

        p = D / f"variant__{a.model}__{vname}{a.out_tag}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results[vname] = {"n": len(rows), "file": p.name,
                          "wall_min": round((time.perf_counter() - t0) / 60, 1)}
        print(f"  wrote {p.name}  ({len(rows)} items, "
              f"{results[vname]['wall_min']} min)", flush=True)

    (D / f"variant_run_manifest__{a.model}{a.out_tag}.json").write_text(
        json.dumps({"model": cfg["name"], "hf_id": cfg["hf_id"],
                    "hybrid_adapter": a.hybrid_adapter,
                    "min_violations": a.min_violations,
                    "decoding": "greedy, max_new_tokens=512",
                    "ctx_budget_chars": C.CTX_MAX_CHARS,
                    "ontology_version": kb.version,
                    "max_repairs": a.max_repairs, "min_coverage": a.min_coverage,
                    "variants": results,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf-8")
    print("\nall variants complete")


if __name__ == "__main__":
    main()
