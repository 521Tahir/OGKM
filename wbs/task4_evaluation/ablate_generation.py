"""
WBS Task 4.1.2 - Ablations that need generation: prompts, the validation layer, and
the retrieval configurations the offline ablation flagged.

Three axes, each run on the same seeded subset of the benchmark so the arms are
paired item by item:

  --axis prompts     four prompt formulations over the identical hybrid context and
                     the identical weights: the deployed template, a bare
                     context+question template, an explicit grounding instruction
                     ("use only these facts; say so when they do not cover it"), and
                     a step-by-step template. This isolates prompt wording from
                     everything else.

  --axis validation  the Task 3.2.2 layer decomposed. The V4 answers already
                     generated are reused as fixed drafts, so repair is the only
                     thing that costs generation and the draft is identical across
                     configurations: validate-only, repair-only, fallback-only, the
                     deployed setting (1 repair, coverage 0.6) and 2 repairs.

  --axis selective   the 2x2 the other two measurements imply but neither tested:
                     whether to gate the layer on its own detector, crossed with
                     whether to abstain on a grounded-coverage floor or only on an
                     unsupportable claim. Run on the dedicated adapter over the
                     whole benchmark, with the detector held at audit-grade in
                     every cell so the two factors are the only things moving.

  --axis retrieval   the ontology context configurations whose gold-entity coverage
                     the offline ablation measured, now carried through to answers,
                     because coverage in the context is not the same thing as use in
                     the answer.

  python -m wbs.task4_evaluation.ablate_generation --axis prompts --gpu 5 --limit 150
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

PROMPTS = {
    "deployed": None,  # the arm's own render_prompt
    "minimal": "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:",
    "strict_grounding":
        "Use ONLY the facts in the context below. If the context does not cover part "
        "of the question, say plainly that you do not have that information rather "
        "than guessing. Do not name any place that is not in the context.\n\n"
        "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:",
    "step_by_step":
        "Context:\n{ctx}\n\nQuestion: {q}\n\nThink through the context step by step, "
        "then give your final answer to the traveller.\nAnswer:",
}

# NOTE on the labels: constrained_answer() abstains when grounded coverage falls
# below min_coverage OR when any UNSUPPORTABLE_CLAIM is present, and the second
# trigger ignores min_coverage entirely. So min_coverage=0.0 does not mean "never
# abstain" - it means "abstain only on unsupportable claims". The labels below say
# what each configuration actually does rather than what the parameters look like.
VALIDATION_CONFIGS = [
    ("none (raw V4 draft)", None, None),
    ("abstain on unsupportable only, no repair", 0, 0.0),
    ("abstain on unsupportable only + 1 repair", 1, 0.0),
    ("abstain on coverage<0.6 or unsupportable, no repair", 0, 0.6),
    ("deployed (1 repair, coverage 0.6)", 1, 0.6),
    ("2 repairs, coverage 0.6", 2, 0.6),
]

# R2's configuration sweep and the selective-validation fit independently name
# the same component as the thing that costs reference overlap: the coverage
# trigger, which abstains whenever grounded coverage is below a floor without
# ever asking whether anything was found wrong. The sweep was run on the shared
# `mixed` adapter and the selective fit on the dedicated one, so the two fixes
# were never put in the same paired design and whether they compose is unknown.
# This axis is that 2x2 - gate the layer on its own detector or spend it on
# everything, crossed with abstaining on the coverage floor or only on an
# unsupportable claim - on one adapter, over the whole benchmark.
#   (label, max_repairs, min_coverage, min_violations)
SELECTIVE_CONFIGS = [
    ("draft only (layer off)", None, None, 0),
    ("every answer, coverage floor (deployed)", 1, 0.6, 0),
    ("every answer, unsupportable only", 1, 0.0, 0),
    ("gated >=1 span, coverage floor", 1, 0.6, 1),
    ("gated >=1 span, unsupportable only", 1, 0.0, 1),
]

RETRIEVAL_CONFIGS = [
    ("full (dense + routing + temporal)", dict()),
    ("- temporal", dict(use_temporal=False)),
    ("- dense search", dict(use_dense=False)),
    ("- class routing", dict(use_routing=False)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["prompts", "validation", "retrieval", "selective"])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260801)
    # R2's open action: the validation axis was measured with the recall-oriented
    # span detector, which the repair loop then spent its turns on. `audit` runs
    # the same configurations with the audit-grade filter inside the detector.
    # The published run is `raw`, and `audit` writes to its own files.
    ap.add_argument("--detector", default="raw", choices=["raw", "audit"])
    # the `selective` axis is about the layer as it would ship, so it defaults to
    # the dedicated adapter and the drafts generated with it; the published
    # `validation` axis keeps the shared adapter it was measured on.
    ap.add_argument("--hybrid-adapter", default="mixed",
                    choices=["mixed", "hybrid20d"])
    # The validation layer has only ever been measured on top of the hybrid arm
    # (V4 -> V5). The project's standing comparison is about the *ontology-only*
    # arm against Simple RAG, and that arm has never been given the layer, so the
    # one configuration the goal actually asks about was never run. --draft-arm
    # V3_ontology_only supplies V3's stored answers as the drafts and regenerates
    # repairs on V3's own dedicated adapter.
    ap.add_argument("--draft-arm", default="V4_ontology_rag",
                    choices=["V4_ontology_rag", "V3_ontology_only"])
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["EXP024_TAG"] = "ded"

    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder, _truncate)
    from experiments.exp042_prf.compute_prf import ents, prf, toks
    from wbs.task3_reasoning.constrained_reasoner import (KnowledgeBase,
                                                          constrained_answer, validate)
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
             if l.strip()]
    rng = random.Random(a.seed)
    sub = sorted(rng.sample(range(len(bench)), min(a.limit, len(bench))))
    items = [bench[i] for i in sub]
    qs = [b["Input"] for b in items]
    print(f"axis={a.axis}  {len(items)} items (seed {a.seed})", flush=True)

    kb = KnowledgeBase()
    cfg = C.MODEL_BY_KEY[a.model]

    def load_model(adapter_name: str | None, ded: bool):
        base = C.ADAPTER_DIR if ded else Path(str(C.ADAPTER_DIR).replace("_ded", ""))
        adapter = (base / f"{a.model}__{adapter_name}") if adapter_name else None
        src = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() \
            else cfg["hf_id"]
        tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
        m = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
        if adapter and (adapter / "adapter_config.json").exists():
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
        m.eval()
        return tok, m, supports_system_role(tok)

    def score(answers: list[str]) -> dict:
        tf, ef, ug, cl, nv = [], [], [], [], []
        for ans, b in zip(answers, items):
            t = prf(toks(ans), toks(b["Output"]))
            e = prf(ents(ans), ents(b["Output"]))
            if t:
                tf.append(t[2])
            if e:
                ef.append(e[2])
            sp = kb.spans(ans)
            bad = [s for s in sp if not kb.known(s)]
            if sp:
                ug.append(len(bad) / len(sp)); cl.append(0.0 if bad else 1.0)
            nv.append(len(validate(ans, kb)))
        return {"token_f1": st.mean(tf) if tf else None,
                "entity_f1": st.mean(ef) if ef else None,
                "ungrounded_rate": st.mean(ug) if ug else None,
                "clean_rate": st.mean(cl) if cl else None,
                "violations_per_item": st.mean(nv),
                "mean_chars": round(st.mean([len(x) for x in answers]))}

    results, raw = [], {}

    # ══ axis: prompts ═════════════════════════════════════════════════════
    if a.axis == "prompts":
        rag = RagContextBuilder(device="cuda"); rag.load()
        rc = [r["context"] for r in rag.retrieve_batch(qs)]
        onto = OntoV3ContextBuilder(); onto.load()
        oc = [r["context"] for r in onto.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)]
        del rag, onto; torch.cuda.empty_cache()
        fb = int(C.CTX_MAX_CHARS * 0.20)
        ctxs = []
        for o, r in zip(oc, rc):
            f = _truncate(o if not o.startswith("[no ontology") else "", fb)
            p = _truncate(r, C.CTX_MAX_CHARS - len(f))
            ctxs.append((f"Ontology facts:\n{f}\n\n" if f else "") +
                        f"Retrieved reference passages:\n{p}")
        tok, model, sys_ok = load_model("mixed", ded=False)
        for name, tpl in PROMPTS.items():
            ans, t0 = [], time.perf_counter()
            for i, b in enumerate(items):
                prompt = (render_prompt(tok, b["Input"], "rag", ctxs[i], sys_ok)
                          if tpl is None else
                          tok.apply_chat_template(
                              [{"role": "user",
                                "content": tpl.format(ctx=ctxs[i], q=b["Input"])}],
                              tokenize=False, add_generation_prompt=True))
                e = tok(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    g = model.generate(**e, max_new_tokens=512, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                ans.append(tok.decode(g[0][e["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip())
                if (i + 1) % 50 == 0:
                    print(f"  [{name}] {i+1}/{len(items)} "
                          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
            raw[name] = ans
            results.append({"config": name, **score(ans),
                            "wall_min": round((time.perf_counter() - t0) / 60, 1)})
            print(f"  {name:20s} tokF1 {results[-1]['token_f1']*100:.1f} "
                  f"entF1 {results[-1]['entity_f1']*100:.1f} "
                  f"ungrounded {results[-1]['ungrounded_rate']*100:.1f}", flush=True)

    # ══ axis: validation ══════════════════════════════════════════════════
    elif a.axis == "validation":
        p = EVAL / f"variant__{a.model}__V4_ontology_rag.jsonl"
        if not p.exists():
            sys.exit("V4 generations are required as the fixed drafts; run run_variants first")
        v4 = {json.loads(l)["bench_id"]: json.loads(l)
              for l in open(p, encoding="utf-8") if l.strip()}
        drafts = [v4[b["bench_id"]]["answer"] for b in items]
        tok, model, sys_ok = load_model("mixed", ded=False)

        def gen(text, max_new=400):
            e = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**e, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return tok.decode(g[0][e["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

        for name, mr, mc in VALIDATION_CONFIGS:
            t0 = time.perf_counter()
            if mr is None:
                ans = list(drafts)
                extra = {"repair_rounds": 0.0, "fallback_rate": 0.0}
            else:
                ans, rr, fb_ = [], [], []
                for i, b in enumerate(items):
                    cr = constrained_answer(b["Input"], drafts[i], kb,
                                            regenerate=gen, max_repairs=mr,
                                            min_coverage=mc,
                                            audit_grade=(a.detector == "audit"))
                    ans.append(cr["answer"]); rr.append(cr["repair_rounds"])
                    fb_.append(1.0 if cr["uncertainty_fallback"] else 0.0)
                    if (i + 1) % 50 == 0:
                        print(f"  [{name}] {i+1}/{len(items)} "
                              f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
                extra = {"repair_rounds": st.mean(rr), "fallback_rate": st.mean(fb_)}
            raw[name] = ans
            changed = sum(1 for x, d in zip(ans, drafts) if x != d)
            results.append({"config": name, **score(ans), **extra,
                            "answers_changed": changed,
                            "wall_min": round((time.perf_counter() - t0) / 60, 1)})
            print(f"  {name:36s} tokF1 {results[-1]['token_f1']*100:.1f} "
                  f"ungrounded {results[-1]['ungrounded_rate']*100:.1f} "
                  f"changed {changed}", flush=True)

    # ══ axis: selective ══════════════════════════════════════════════════
    elif a.axis == "selective":
        onto_only = a.draft_arm == "V3_ontology_only"
        # V3 was published on its own dedicated adapter, so it carries no tag and
        # needs no hybrid adapter; V4 is the tagged dedicated-adapter run
        sfx = "" if (onto_only or a.hybrid_adapter == "mixed") else "__hybded"
        arm_adapter = "onto_v3" if onto_only else a.hybrid_adapter
        p = EVAL / f"variant__{a.model}__{a.draft_arm}{sfx}.jsonl"
        if not p.exists():
            sys.exit(f"{p.name} is required as the fixed drafts; run run_variants "
                     f"with --hybrid-adapter {a.hybrid_adapter} first")
        v4 = {json.loads(l)["bench_id"]: json.loads(l)
              for l in open(p, encoding="utf-8") if l.strip()}
        missing = [b["bench_id"] for b in items if b["bench_id"] not in v4]
        if missing:
            sys.exit(f"{len(missing)} benchmark items have no V4 draft in {p.name}")
        drafts = [v4[b["bench_id"]]["answer"] for b in items]
        tok, model, sys_ok = load_model(
            arm_adapter, ded=(arm_adapter in ("hybrid20d", "onto_v3")))

        def gen(text, max_new=400):
            e = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**e, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return tok.decode(g[0][e["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

        from wbs.task4_evaluation.score_variants import score_item

        def score_canonical(answers: list[str]) -> dict:
            """Mean metrics under the canonical per-item scorer.

            `score()` above is kept byte-for-byte as the published ablation
            tables were computed with it; this axis is compared against
            `selective_validation.md`, so it has to be scored the way that file
            was. `ungrounded_rate` is audit-grade and `ungrounded_rate_raw` is
            the validator's own detector - the same two columns, same
            denominators, same None-handling as the main pipeline.
            """
            per = [score_item({"bench_id": b["bench_id"], "category": b["category"],
                               "reference": b["Output"], "answer": ans}, kb)
                   for ans, b in zip(answers, items)]
            out = {}
            for f in ("token_f1", "entity_f1", "entity_hit", "ungrounded_rate",
                      "ungrounded_rate_raw", "clean", "unsupportable"):
                xs = [r[f] for r in per if r[f] is not None]
                out[f] = (sum(xs) / len(xs)) if xs else None
            out["mean_chars"] = round(st.mean([len(x) for x in answers]))
            return out

        for name, mr, mc, mv in SELECTIVE_CONFIGS:
            t0 = time.perf_counter()
            if mr is None:
                ans = list(drafts)
                extra = {"repair_rounds": 0.0, "fallback_rate": 0.0,
                         "validated": 0, "min_violations": 0}
            else:
                ans, rr, fb_, spent = [], [], [], 0
                for i, b in enumerate(items):
                    cr = constrained_answer(b["Input"], drafts[i], kb,
                                            regenerate=gen, max_repairs=mr,
                                            min_coverage=mc, min_violations=mv,
                                            audit_grade=(a.detector == "audit"))
                    ans.append(cr["answer"]); rr.append(cr["repair_rounds"])
                    fb_.append(1.0 if cr["uncertainty_fallback"] else 0.0)
                    spent += 0 if cr.get("selective_skip") else 1
                    if (i + 1) % 50 == 0:
                        print(f"  [{name}] {i+1}/{len(items)} "
                              f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
                extra = {"repair_rounds": st.mean(rr), "fallback_rate": st.mean(fb_),
                         "validated": spent, "min_violations": mv}
            raw[name] = ans
            changed = sum(1 for x, d in zip(ans, drafts) if x != d)
            results.append({"config": name, "max_repairs": mr, "min_coverage": mc,
                            **score_canonical(ans), **extra,
                            "answers_changed": changed,
                            "wall_min": round((time.perf_counter() - t0) / 60, 1)})
            r = results[-1]
            print(f"  {name:42s} tokF1 {r['token_f1']*100:.1f} "
                  f"ungrounded {r['ungrounded_rate']*100:.1f} "
                  f"clean {r['clean']*100:.1f} "
                  f"validated {r['validated']}/{len(items)} "
                  f"changed {changed}", flush=True)

    # ══ axis: retrieval ═══════════════════════════════════════════════════
    else:
        onto = OntoV3ContextBuilder(device="cuda"); onto.load()
        import numpy as np
        qe = onto._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in qs],
                                       batch_size=128, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        _, idxs = onto._index.search(qe, C.ONTO_V3_ENTITIES)

        # capture what build() needs BEFORE the builder is freed: the closure below
        # outlives `onto`, and deleting the builder first makes the free variable
        # unbound at call time rather than at definition time.
        CARDS = onto._cards
        BY_CLASS = onto._by_class
        TARGET_CLASSES = onto._target_classes
        TEMPORAL = onto._temporal

        def build(use_dense=True, use_routing=True, use_temporal=True):
            out = []
            for qi, q in enumerate(qs):
                picked, seen = [], set()
                if use_dense:
                    for j in idxs[qi]:
                        if j >= 0 and j not in seen:
                            seen.add(j); picked.append(CARDS[j])
                if use_routing:
                    classes = TARGET_CLASSES(q)
                    per = max(2, C.ONTO_V3_ENTITIES // max(len(classes), 1))
                    for klass in classes:
                        added = 0
                        for j in BY_CLASS.get(klass, []):
                            if j in seen:
                                continue
                            seen.add(j); picked.append(CARDS[j]); added += 1
                            if added >= per:
                                break
                tl = []
                if use_temporal and TEMPORAL is not None:
                    try:
                        tl = [f.to_sentence()
                              for f in TEMPORAL.retrieve(q, top_k=6).facts][:4]
                    except Exception:
                        tl = []
                lines = []
                if tl:
                    lines.append("Time-scoped facts:"); lines += [f"- {t}" for t in tl]
                if picked:
                    if lines:
                        lines.append("")
                    lines.append("Knowledge-graph facts:")
                    for c in picked:
                        lines += [f"- {f}" for f in c["facts"][:3]]
                out.append(_truncate("\n".join(lines), C.CTX_MAX_CHARS) if lines
                           else "[no ontology facts found]")
            return out

        del onto; torch.cuda.empty_cache()
        tok, model, sys_ok = load_model("onto_v3", ded=True)
        for name, kw in RETRIEVAL_CONFIGS:
            ctx = build(**kw)
            ans, t0 = [], time.perf_counter()
            for i, b in enumerate(items):
                prompt = render_prompt(tok, b["Input"], "onto_v3", ctx[i], sys_ok)
                e = tok(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    g = model.generate(**e, max_new_tokens=512, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                ans.append(tok.decode(g[0][e["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip())
                if (i + 1) % 50 == 0:
                    print(f"  [{name}] {i+1}/{len(items)} "
                          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
            raw[name] = ans
            results.append({"config": name,
                            "ctx_chars": round(st.mean([len(c) for c in ctx])),
                            **score(ans),
                            "wall_min": round((time.perf_counter() - t0) / 60, 1)})
            print(f"  {name:36s} tokF1 {results[-1]['token_f1']*100:.1f} "
                  f"entF1 {results[-1]['entity_f1']*100:.1f}", flush=True)

    # ── significance vs the first configuration ───────────────────────────
    out = {"axis": a.axis, "model": a.model, "n": len(items), "seed": a.seed,
           "detector": a.detector, "hybrid_adapter": a.hybrid_adapter,
           "draft_arm": a.draft_arm, "results": results}
    # the published runs are the `raw`-detector ones and keep the plain names
    dsfx = "" if a.detector == "raw" else f"__{a.detector}det"
    if a.axis == "selective":
        if a.draft_arm == "V3_ontology_only":
            dsfx += "__ontoonly"
        elif a.hybrid_adapter != "mixed":
            dsfx += f"__{a.hybrid_adapter}"
    (EVAL / f"ablation_{a.axis}__{a.model}{dsfx}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    with open(EVAL / f"ablation_{a.axis}__{a.model}{dsfx}__answers.jsonl", "w",
              encoding="utf-8") as f:
        for i, b in enumerate(items):
            f.write(json.dumps({"bench_id": b["bench_id"], "Input": b["Input"],
                                "reference": b["Output"],
                                "answers": {k: v[i] for k, v in raw.items()}},
                               ensure_ascii=False) + "\n")
    print(f"\nwrote ablation_{a.axis}__{a.model}{dsfx}.json "
          f"({len(results)} configurations, {len(items)} items)")


if __name__ == "__main__":
    main()
