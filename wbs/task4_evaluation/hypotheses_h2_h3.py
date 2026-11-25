"""
Proposal H2 and H3, measured on the answers the five variants already produced.

H2  "Ontology-based constraints integrated into generation pipelines
     significantly reduce hallucination rates WITHOUT DEGRADING RESPONSE FLUENCY."
     The hallucination half was measured (ungrounded entity rate, distinct
     fabricated spans). The fluency half was not, so H2 was half-tested. Fluency
     here is the mean per-token log-likelihood of the answer under the BASE model
     with no context and no adapter - a plain language-model score, independent
     of the arm that produced the text - plus three degeneration counters that
     catch the failure mode a likelihood score rewards (repetition).

H3  "Ontology-integrated systems provide more interpretable and verifiable
     responses." Verifiability is operationalised as attribution: the share of
     the places an answer names that are present in the context that answer was
     given, and the share of its bullet lines carrying at least one attributable
     place. An answer can be accurate and still unverifiable if nothing in its
     own evidence supports it, which is exactly what a high-trust domain cares
     about. Interpretability is the availability and size of the V5 reasoning
     trace.

Contexts are rebuilt with the same builders `run_variants` used, so each answer
is scored against the evidence it actually received. V1 has no context and is
reported as not-applicable rather than zero.

  python -m wbs.task4_evaluation.hypotheses_h2_h3 --gpu 6 --model llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
            "V5_ontology_rag_validated"]
ARM_OF = {"V1_llm": None, "V2_rag": "rag", "V3_ontology_only": "onto_v3",
          "V4_ontology_rag": "hybrid20", "V5_ontology_rag_validated": "hybrid20"}


def distinct_n(text: str, n: int) -> float:
    w = re.findall(r"[a-z0-9']+", text.lower())
    if len(w) < n:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def repeated_line_frac(text: str) -> float:
    lines = [l.strip().lower() for l in text.splitlines() if len(l.strip()) > 12]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder, _truncate)
    from experiments.exp042_prf.compute_prf import ents
    from wbs.task4_evaluation.score_variants import wilcoxon
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    answers = {}
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if not p.exists():
            print(f"missing {p.name}, skipping {v}", flush=True)
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        answers[v] = rows[: a.limit] if a.limit else rows
    if not answers:
        sys.exit("no variant answers found")
    n = len(next(iter(answers.values())))
    print(f"{len(answers)} variants x {n} answers", flush=True)

    # ── contexts, rebuilt exactly as run_variants built them ──────────────
    qs = [r["Input"] for r in next(iter(answers.values()))]
    print("rebuilding retrieval contexts …", flush=True)
    rag = RagContextBuilder(device="cuda"); rag.load()
    rag_ctx = [r["context"] for r in rag.retrieve_batch(qs)]
    onto = OntoV3ContextBuilder(device="cuda"); onto.load()
    onto_ctx = [r["context"] for r in onto.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)]
    fact_budget = int(C.CTX_MAX_CHARS * 0.20)
    hyb_ctx = []
    for rc, oc in zip(rag_ctx, onto_ctx):
        facts = _truncate(oc if not oc.startswith("[no ontology") else "", fact_budget)
        passages = _truncate(rc, C.CTX_MAX_CHARS - len(facts))
        hyb_ctx.append((f"Ontology facts:\n{facts}\n\n" if facts else "") +
                       f"Retrieved reference passages:\n{passages}")
    CTX = {"rag": rag_ctx, "onto_v3": onto_ctx, "hybrid20": hyb_ctx}
    del rag, onto
    torch.cuda.empty_cache()

    # ── H2: fluency under the base model, no adapter, no context ──────────
    cfg = C.MODEL_BY_KEY[a.model]
    print(f"loading {cfg['hf_id']} (vanilla, for scoring only) …", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model.eval()

    def logprob(text: str) -> tuple[float, int] | None:
        """Mean per-token log-likelihood of `text` as a standalone sequence."""
        t = (text or "").strip()
        if len(t) < 20:
            return None
        ids = tok(t, return_tensors="pt", truncation=True, max_length=1024)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        if ids["input_ids"].shape[1] < 8:
            return None
        with torch.no_grad():
            out = model(**ids, labels=ids["input_ids"])
        ntok = int(ids["input_ids"].shape[1]) - 1
        return -float(out.loss), ntok

    results, per_item = {}, {}
    t0 = time.perf_counter()
    for v, rows in answers.items():
        arm = ARM_OF[v]
        lp, nt, d2, d3, rep, chars = [], [], [], [], [], []
        attr_ent, attr_line, traced, trace_len = [], [], [], []
        for i, r in enumerate(rows):
            ans = r.get("answer") or ""
            s = logprob(ans)
            if s:
                lp.append(s[0]); nt.append(s[1])
            d2.append(distinct_n(ans, 2)); d3.append(distinct_n(ans, 3))
            rep.append(repeated_line_frac(ans)); chars.append(len(ans))

            if arm is not None:
                ctx_low = CTX[arm][i].lower()
                pe = set(ents(ans))
                if pe:
                    attr_ent.append(sum(1 for e in pe if e in ctx_low) / len(pe))
                lines = [l for l in ans.splitlines() if len(l.strip()) > 12]
                if lines:
                    ok = 0
                    for l in lines:
                        le = set(ents(l))
                        if le and any(e in ctx_low for e in le):
                            ok += 1
                    attr_line.append(ok / len(lines))
            if "explanation_trace" in r:
                tr = r.get("explanation_trace") or []
                traced.append(1.0 if tr else 0.0)
                trace_len.append(len(tr) if isinstance(tr, (list, tuple)) else 0)

        results[v] = {
            "n": len(rows),
            "fluency_logprob": st.mean(lp) if lp else None,
            "fluency_ppl": math.exp(-st.mean(lp)) if lp else None,
            "answer_tokens": st.mean(nt) if nt else None,
            "answer_chars": st.mean(chars),
            "distinct_2": st.mean(d2), "distinct_3": st.mean(d3),
            "repeated_line_frac": st.mean(rep),
            "attributable_entity_rate": st.mean(attr_ent) if attr_ent else None,
            "attributable_line_rate": st.mean(attr_line) if attr_line else None,
            "trace_available": st.mean(traced) if traced else None,
            "trace_steps": st.mean(trace_len) if trace_len else None,
        }
        per_item[v] = {r["bench_id"]: x for r, x in zip(rows, lp)} if len(lp) == len(rows) else {}
        o = results[v]
        print(f"  {v:<28} logprob {o['fluency_logprob']:.4f} (ppl "
              f"{o['fluency_ppl']:.2f}) | d2 {o['distinct_2']:.3f} | "
              f"attr-ent {('%.3f' % o['attributable_entity_rate']) if o['attributable_entity_rate'] is not None else '  n/a'}",
              flush=True)

    # paired significance on fluency, every variant against V2 RAG
    sig = {}
    base = per_item.get("V2_rag", {})
    for v, pi in per_item.items():
        if v == "V2_rag" or not pi or not base:
            continue
        both = [b for b in pi if b in base]
        w = wilcoxon([pi[b] for b in both], [base[b] for b in both])
        if w:
            sig[v] = {"stat": w[0], "p": w[1], "n": w[2],
                      "delta_logprob": st.mean(pi[b] - base[b] for b in both)}

    out = {"model": cfg["name"], "n": n,
           "fluency_definition": "mean per-token log-likelihood of the answer under "
                                 "the base model, no adapter, no context",
           "results": results, "fluency_vs_V2_rag": sig,
           "wall_min": round((time.perf_counter() - t0) / 60, 1)}
    p = EVAL / f"hypotheses_h2_h3__{a.model}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p.name}  ({out['wall_min']} min)")


if __name__ == "__main__":
    main()
