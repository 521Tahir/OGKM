"""
Proposal section 11.5 - "Consistency across paraphrased or repeated queries" and
"Stability under prompt variation".

Accuracy against a reference says whether an answer is right. It says nothing
about whether the same system, asked the same thing twice or asked it in
different words, answers the same way - which is what a user in a high-trust
domain notices first, and what the proposal asks for separately.

Two axes, measured per arm:

  PARAPHRASE   the question is restated K ways with the meaning held fixed, and
               the arm answers each. Consistency is the mean pairwise agreement
               between those answers (token F1 and gazetteer-entity Jaccard).
               Paraphrases are produced once by the model at greedy decoding and
               CACHED to `consistency_paraphrases.jsonl`, so they are fixed,
               inspectable and identical across arms and reruns - a paraphrase
               set regenerated per arm would confound the comparison.
  REPEAT       the original question is answered R times at temperature 0.7.
               Agreement across those answers isolates decoding noise from
               wording sensitivity, so the two sources can be compared rather
               than conflated.

Agreement between two answers is symmetric token F1, which for two predictions
is just their harmonic overlap - the same function used for accuracy, so the
numbers sit on the same scale as the accuracy table.

  python -m wbs.task4_evaluation.consistency --gpu 5 --arms rag onto_v3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
PARA_P = EVAL / "consistency_paraphrases.jsonl"

PARA_PROMPT = ("Rewrite the traveller's question below in different words. Keep the "
               "meaning, the specifics and the number of sub-questions identical. "
               "Do not answer it. Do not add or remove requirements. Reply with the "
               "rewritten question only.\n\nQuestion: {q}\n\nRewritten question:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--arms", nargs="+", default=["rag", "onto_v3", "hybrid20"],
                    choices=["rag", "onto_v3", "hybrid20"])
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--paraphrases", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--tag", default="")
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
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder, _truncate)
    from experiments.exp042_prf.compute_prf import ents, prf, toks
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    rng.shuffle(bench)
    bench = bench[: a.n_questions]
    print(f"{len(bench)} questions | arms {', '.join(a.arms)} | "
          f"{a.paraphrases} paraphrases + {a.repeats} sampled repeats", flush=True)

    cfg = C.MODEL_BY_KEY[a.model]
    ADAPTER = {"rag": "rag", "onto_v3": "onto_v3", "hybrid20": "mixed"}

    def load_arm(want: str):
        base = (C.ADAPTER_DIR if want in ("rag", "onto_v3")
                else Path(str(C.ADAPTER_DIR).replace("_ded", "")))
        adapter = base / f"{a.model}__{want}"
        src = str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]
        tk = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
        m = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda")
        if (adapter / "adapter_config.json").exists():
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
        m.eval()
        return tk, m, supports_system_role(tk)

    def generate(tk, m, text, max_new=512, sample=False, seed=0):
        enc = tk(text, return_tensors="pt").to("cuda")
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new,
                           do_sample=sample,
                           temperature=a.temperature if sample else None,
                           top_p=0.95 if sample else None,
                           pad_token_id=tk.pad_token_id or tk.eos_token_id)
        return tk.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ── paraphrases: produced once, cached, shared by every arm ───────────
    cache = {}
    if PARA_P.exists():
        for l in open(PARA_P, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                cache[r["bench_id"]] = r["paraphrases"]
    need = [b for b in bench if len(cache.get(b["bench_id"], [])) < a.paraphrases]
    if need:
        print(f"generating paraphrases for {len(need)} questions (greedy, cached) …",
              flush=True)
        tk, m, _ = load_arm("rag")          # any arm's tokenizer; base weights paraphrase
        for i, b in enumerate(need, 1):
            outs = cache.get(b["bench_id"], [])
            for k in range(len(outs), a.paraphrases):
                # a different instruction per slot, so the K paraphrases differ
                hint = ["", " Use a different sentence order.",
                        " Phrase it as a single direct request."][k % 3]
                txt = PARA_PROMPT.format(q=b["Input"]) + hint
                p = generate(tk, m, txt, max_new=160)
                p = p.split("\n")[0].strip().strip('"')
                outs.append(p if len(p) > 15 else b["Input"])
            cache[b["bench_id"]] = outs
            if i % 10 == 0 or i == len(need):
                print(f"  {i}/{len(need)}", flush=True)
        with open(PARA_P, "w", encoding="utf-8") as f:
            for b in bench:
                f.write(json.dumps({"bench_id": b["bench_id"], "original": b["Input"],
                                    "paraphrases": cache[b["bench_id"]]},
                                   ensure_ascii=False) + "\n")
        del m
        torch.cuda.empty_cache()
        print(f"cached paraphrases -> {PARA_P.name}", flush=True)

    # ── contexts for every question variant, per arm ──────────────────────
    all_q = []
    for b in bench:
        all_q.append(b["Input"])
        all_q += cache[b["bench_id"]][: a.paraphrases]
    print(f"building contexts for {len(all_q)} question variants …", flush=True)
    rag = RagContextBuilder(device="cuda"); rag.load()
    rag_ctx = [r["context"] for r in rag.retrieve_batch(all_q)]
    onto = OntoV3ContextBuilder(device="cuda"); onto.load()
    onto_ctx = [r["context"] for r in onto.retrieve_batch(all_q, n_entities=C.ONTO_V3_ENTITIES)]
    fb = int(C.CTX_MAX_CHARS * 0.20)
    hyb_ctx = []
    for rc, oc in zip(rag_ctx, onto_ctx):
        facts = _truncate(oc if not oc.startswith("[no ontology") else "", fb)
        passages = _truncate(rc, C.CTX_MAX_CHARS - len(facts))
        hyb_ctx.append((f"Ontology facts:\n{facts}\n\n" if facts else "") +
                       f"Retrieved reference passages:\n{passages}")
    CTX = {"rag": rag_ctx, "onto_v3": onto_ctx, "hybrid20": hyb_ctx}
    del rag, onto
    torch.cuda.empty_cache()

    def agree(x: str, y: str) -> float:
        t = prf(toks(x), toks(y))
        return t[2] if t else 0.0

    def ent_jac(x: str, y: str) -> float:
        ex, ey = set(ents(x)), set(ents(y))
        return len(ex & ey) / len(ex | ey) if (ex or ey) else 1.0

    tag = f"__{a.tag}" if a.tag else ""
    out_p = EVAL / f"consistency__{a.model}{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}

    for arm in a.arms:
        if arm in out:
            print(f"[{arm}] already measured, skipping", flush=True)
            continue
        print(f"\n=== {arm} ===", flush=True)
        tk, m, sys_ok = load_arm(ADAPTER[arm])
        rows, t0 = [], time.perf_counter()
        step = a.paraphrases + 1
        for qi, b in enumerate(bench):
            base_i = qi * step
            variants = [all_q[base_i + k] for k in range(step)]
            ctxs = [CTX[arm][base_i + k] for k in range(step)]
            prompt_arm = "rag" if arm == "hybrid20" else arm
            # paraphrase axis: greedy, so only the wording differs
            para_ans = [generate(tk, m, render_prompt(tk, q, prompt_arm, c, sys_ok))
                        for q, c in zip(variants, ctxs)]
            # repeat axis: original question, sampled
            rep_ans = [generate(tk, m, render_prompt(tk, variants[0], prompt_arm,
                                                     ctxs[0], sys_ok),
                                sample=True, seed=a.seed + r)
                       for r in range(a.repeats)]
            rows.append({"bench_id": b["bench_id"], "category": b["category"],
                         "reference": b["Output"], "questions": variants,
                         "paraphrase_answers": para_ans, "repeat_answers": rep_ans})
            if (qi + 1) % 10 == 0 or qi + 1 == len(bench):
                el = time.perf_counter() - t0
                print(f"  {qi+1}/{len(bench)}  {el/60:.1f} min "
                      f"(ETA {(el/(qi+1))*(len(bench)-qi-1)/60:.1f} min)", flush=True)
        del m
        torch.cuda.empty_cache()

        pa, pe, ra, re_, acc_sd, acc = [], [], [], [], [], []
        for r in rows:
            A = r["paraphrase_answers"]
            pa += [agree(x, y) for x, y in itertools.combinations(A, 2)]
            pe += [ent_jac(x, y) for x, y in itertools.combinations(A, 2)]
            R = [A[0]] + r["repeat_answers"]
            ra += [agree(x, y) for x, y in itertools.combinations(R, 2)]
            re_ += [ent_jac(x, y) for x, y in itertools.combinations(R, 2)]
            f1s = [prf(toks(x), toks(r["reference"]))[2] for x in A
                   if prf(toks(x), toks(r["reference"]))]
            if len(f1s) > 1:
                acc_sd.append(st.stdev(f1s)); acc += f1s
        out[arm] = {
            "n_questions": len(rows), "paraphrases": a.paraphrases,
            "repeats": a.repeats, "temperature": a.temperature,
            "paraphrase_agreement_token_f1": st.mean(pa),
            "paraphrase_agreement_entity_jaccard": st.mean(pe),
            "repeat_agreement_token_f1": st.mean(ra),
            "repeat_agreement_entity_jaccard": st.mean(re_),
            "accuracy_mean_over_paraphrases": st.mean(acc) if acc else None,
            "accuracy_sd_within_question": st.mean(acc_sd) if acc_sd else None,
            "wall_min": round((time.perf_counter() - t0) / 60, 1),
        }
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        with open(EVAL / f"consistency_answers__{a.model}__{arm}.jsonl", "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        o = out[arm]
        print(f"  paraphrase agreement {o['paraphrase_agreement_token_f1']*100:.1f} "
              f"(entities {o['paraphrase_agreement_entity_jaccard']*100:.1f}) | "
              f"repeat agreement {o['repeat_agreement_token_f1']*100:.1f} | "
              f"accuracy SD within question {o['accuracy_sd_within_question']*100:.1f}",
              flush=True)

    print(f"\nwrote {out_p.name}")


if __name__ == "__main__":
    main()
