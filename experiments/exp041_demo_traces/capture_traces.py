"""
exp041 - capture REAL per-stage execution traces for the three demo queries.

Both arms run on Llama-3.1-8B with ARM-DEDICATED LoRA adapters (exp037), so the
comparison is matched: same base weights, same decoding, same 3,000-character
context budget, same prompt template. Only the retrieval differs - which is the
method under study.

Everything the demo displays is written here from the actual run: retrieved
passages with their cosine scores, the ontology entities/classes/triples that
were selected, the fully instantiated prompts, token counts, the generated
answers, and wall-clock stage timings. Nothing in the demo is hand-written.

  python -m experiments.exp041_demo_traces.capture_traces --gpu 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--out", default="experiments/exp041_demo_traces/traces.json")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["EXP024_TAG"] = "ded"          # dedicated adapters

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer,
                                                         render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import (OntoV3ContextBuilder,
                                                             RagContextBuilder,
                                                             _truncate)
    from experiments.exp041_demo_traces.queries import QUERIES
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    cfg = C.MODEL_BY_KEY[a.model]
    gaz = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                     .read_text(encoding="utf-8"))
    places = {x["name"]: x for x in gaz["locations"] if isinstance(x, dict)}

    def hav(a_, b_) -> float:
        R, p = 6371.0, math.pi / 180
        dla, dlo = (b_[0] - a_[0]) * p, (b_[1] - a_[1]) * p
        h = (math.sin(dla / 2) ** 2
             + math.cos(a_[0] * p) * math.cos(b_[0] * p) * math.sin(dlo / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(h))

    # ── retrieval stage: RAG ───────────────────────────────────────────────
    print("[trace] loading RAG index …", flush=True)
    rag = RagContextBuilder(device="cuda")
    rag.load()
    print(f"[trace] RAG index: {len(rag._passages):,} passages", flush=True)

    print("[trace] loading ontology entity-card index …", flush=True)
    onto = OntoV3ContextBuilder()
    onto.load()
    print(f"[trace] ontology cards: {len(onto._cards):,}", flush=True)

    rag_stage, onto_stage = {}, {}
    for q in QUERIES:
        qt = q["text"]

        t0 = time.perf_counter()
        qe = rag._load_model().encode([C.EMBED_QUERY_PREFIX + qt], convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False).astype("float32")
        t_embed = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores, idxs = rag._index.search(qe, C.RAG_TOP_K + 5)
        t_search = time.perf_counter() - t0
        hits, seen = [], set()
        for s, j in zip(scores[0], idxs[0]):
            if j < 0:
                continue
            p = rag._passages[j]
            key = p["Output"][:200].lower()
            if key in seen:
                continue
            seen.add(key)
            hits.append({"rank": len(hits) + 1, "cosine": round(float(s), 4),
                         "record_id": p.get("record_id"),
                         "source_question": p["Input"],
                         "passage": _truncate(p["Output"], C.RAG_PASSAGE_CHARS)})
            if len(hits) == C.RAG_TOP_K:
                break
        rag_stage[q["id"]] = {
            "stage_1_query": qt,
            "stage_2_embed_model": C.EMBED_MODEL,
            "stage_2_query_prefix": C.EMBED_QUERY_PREFIX,
            "stage_3_corpus_size": len(rag._passages),
            "stage_4_index": "FAISS IndexFlatIP (cosine, normalised)",
            "stage_5_top_k": C.RAG_TOP_K,
            "stage_6_hits": hits,
            "stage_7_context": rag._format([{"Output": h["passage"]} for h in hits]),
            "timing_ms": {"embed": round(t_embed * 1000, 1),
                          "search": round(t_search * 1000, 1)},
        }

        t0 = time.perf_counter()
        res = onto.retrieve_batch([qt], n_entities=C.ONTO_V3_ENTITIES)[0]
        t_onto = time.perf_counter() - t0
        classes = onto._target_classes(qt)
        oe = onto._load_model().encode([C.EMBED_QUERY_PREFIX + qt], convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        csc, cidx = onto._index.search(oe, C.ONTO_V3_ENTITIES)
        cards = []
        for s, j in zip(csc[0], cidx[0]):
            if j < 0:
                continue
            c = onto._cards[j]
            cards.append({"entity": c.get("name") or c.get("entity"),
                          "concept_class": c.get("concept_class") or c.get("klass"),
                          "cosine": round(float(s), 4),
                          "facts": c["facts"][:3]})
        triples = [ln[2:] for ln in (res["context"] or "").splitlines()
                   if ln.startswith("- ")]
        onto_stage[q["id"]] = {
            "stage_1_query": qt,
            "stage_2_intent_classes": classes,
            "stage_3_card_index_size": len(onto._cards),
            "stage_4_entity_cards": cards,
            "stage_5_triples_in_context": triples,
            "stage_6_context": res["context"],
            "timing_ms": {"retrieve": round(t_onto * 1000, 1)},
        }
        print(f"[trace] {q['id']}: rag {len(hits)} passages / "
              f"onto {len(cards)} cards, {len(triples)} triples", flush=True)

    del rag, onto
    torch.cuda.empty_cache()

    # ── generation stage, both arms, dedicated adapters ────────────────────
    out = {"meta": {"model": cfg["name"], "hf_id": cfg["hf_id"],
                     "adapters": "arm-dedicated (exp037)",
                     "adapter_dir": str(C.ADAPTER_DIR),
                     "decoding": "greedy, max_new_tokens=512",
                     "ctx_budget_chars": C.CTX_MAX_CHARS,
                     "embed_model": C.EMBED_MODEL,
                     "ontology_version": C.ONTO_CLEAN_VERSION,
                     "captured": time.strftime("%Y-%m-%d %H:%M:%S")},
           "queries": {q["id"]: q for q in QUERIES},
           "gazetteer_distances_km": {},
           "rag": rag_stage, "onto": onto_stage, "answers": {}}

    ap_name = "Jeju International Airport"
    if ap_name in places:
        o = (places[ap_name]["lat"], places[ap_name]["lon"])
        out["gazetteer_distances_km"] = {
            n: round(hav(o, (v["lat"], v["lon"])), 1)
            for n, v in places.items() if n != ap_name}

    for arm, stages in (("rag", rag_stage), ("onto_v3", onto_stage)):
        adapter = C.ADAPTER_DIR / f"{a.model}__{arm}"
        print(f"\n[trace] loading {cfg['hf_id']} + {adapter.name} …", flush=True)
        tok = prepare_tokenizer(AutoTokenizer.from_pretrained(str(adapter)
                                if (adapter / "tokenizer_config.json").exists()
                                else cfg["hf_id"]), for_training=False)
        sys_ok = supports_system_role(tok)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.eval()

        for qid, st in stages.items():
            ctx = st["stage_6_context"] if arm != "rag" else st["stage_7_context"]
            prompt = render_prompt(tok, st["stage_1_query"], arm, ctx, sys_ok)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.perf_counter() - t0
            ans = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out["answers"].setdefault(qid, {})[arm] = {
                "prompt": prompt,
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "answer": ans.strip(),
                "answer_tokens": int(gen[0].shape[0] - enc["input_ids"].shape[1]),
                "generation_s": round(dt, 2),
                "retrieval_ms": sum(st["timing_ms"].values()),
                "adapter": adapter.name,
            }
            print(f"[trace] {qid}/{arm}: {enc['input_ids'].shape[1]} prompt tok -> "
                  f"{out['answers'][qid][arm]['answer_tokens']} tok in {dt:.1f}s", flush=True)

        del model
        torch.cuda.empty_cache()

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[trace] wrote {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
