"""Demo: what an ontology edit changes, with no retraining at all.

The business claim this project makes for the ontology layer is that knowledge
can be corrected by editing a graph instead of by fine-tuning a model. That is
cheap to assert and easy to fake in a slide, so it is measured here end to end:

  * one small model (Llama-3.2-1B) with one adapter, loaded once, never retrained;
  * the same question asked twice;
  * between the two runs nothing changes but the ontology version the context
    builder reads - v1.4.0 (no weather state for any coastal place) and v1.5.0
    (the `avoid_during` states extracted from the corpus by
    wbs/task2_formalization/extract_weather_states.py).

The adapter file digests are recorded before and after so the claim "no weights
were touched" is checked rather than stated. Both runs use the production
OntoV3ContextBuilder; `safety_first=True` is the one non-default setting and it
only reorders an entity's own facts before the 8-fact cap applies, so a
constraint asserted late in the relation file can reach the card at all.

    python -m wbs.task5_demo.demo_live_edit --gpu 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "presentation"

QUESTIONS = [
    "A storm warning is out for Jeju. Is it safe to swim at Woljeongri Beach "
    "today, and what should I do instead?",
    "We are visiting Jungmun Saekdal Beach and Yongmeori Coast during the "
    "monsoon. Which water activities should we avoid?",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", default="llama3.2-1b")
    ap.add_argument("--before", default="1.4.0")
    ap.add_argument("--after", default="1.5.0")
    a = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import OntoV3ContextBuilder
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    OUT.mkdir(parents=True, exist_ok=True)

    # ── contexts, one per ontology version. The embedder runs on the CPU so the
    #    1B model has the GPU to itself: the GPUs this project is allowed to use
    #    are shared with other jobs and have ~5 GiB free.
    ctx, cards, build_s = {}, {}, {}
    for v in (a.before, a.after):
        b = OntoV3ContextBuilder(onto_version=v, device="cpu", safety_first=True)
        t0 = time.perf_counter()
        b.build()
        build_s[v] = round(time.perf_counter() - t0, 1)
        b.load()
        ctx[v] = [r["context"] for r in b.retrieve_batch(QUESTIONS,
                                                         n_entities=C.ONTO_V3_ENTITIES)]
        cards[v] = {c["name"]: c["facts"] for c in b._cards}
        print(f"[{v}] index built in {build_s[v]}s, "
              f"context {sum(len(c) for c in ctx[v]) // len(QUESTIONS):,} chars", flush=True)
        del b

    # ── one model, one adapter, loaded once ───────────────────────────────
    cfg = C.MODEL_BY_KEY[a.model]
    adapter = C.ADAPTER_DIR / f"{a.model}__onto_v3"
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"adapter {adapter} not found")
    digest_before = sha(adapter)

    src = str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
    sys_ok = supports_system_role(tok)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()

    def gen(text: str, max_new: int = 320) -> tuple[str, float]:
        enc = tok(text, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        dt = time.perf_counter() - t0
        return tok.decode(g[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip(), dt

    kbs = {v: KnowledgeBase(v) for v in (a.before, a.after)}
    rows = []
    for i, q in enumerate(QUESTIONS):
        row = {"question": q, "runs": {}}
        for v in (a.before, a.after):
            prompt = render_prompt(tok, q, "onto_v3", ctx[v][i], sys_ok)
            ans, dt = gen(prompt)
            kb = kbs[v]
            # which of the new states does the answer actually name, and is the
            # pairing an asserted triple at this version?
            named = sorted({s for s in ("Typhoon", "Storm", "Rough Seas", "rough seas",
                                        "strong current", "high waves")
                            if s.lower() in ans.lower()})
            supported = [list(t) for t in kb.triples if t[1] == "avoid_during"
                         and t[0] in ans.lower()]
            row["runs"][v] = {
                "ontology_version": v,
                "context": ctx[v][i],
                "answer": ans,
                "generation_s": round(dt, 2),
                "weather_terms_in_answer": named,
                "supporting_avoid_during_triples": supported[:12],
                "n_supporting_triples": len(supported),
            }
            print(f"\n=== Q{i + 1} @ v{v} ({dt:.1f}s)\n{ans}\n", flush=True)
        rows.append(row)

    digest_after = sha(adapter)
    onto_counts = {}
    for v in (a.before, a.after):
        rel = json.loads((ROOT / "artifacts" / "ontologies" / v / "relations.json")
                         .read_text(encoding="utf-8"))
        onto_counts[v] = {
            "relations": len(rel),
            "avoid_during": sum(1 for r in rel if r.get("predicate") == "avoid_during"),
        }

    out = {
        "model": cfg["name"], "hf_id": cfg["hf_id"], "adapter": adapter.name,
        "decoding": "greedy, max_new_tokens=320",
        "retraining": {
            "adapter_sha_before": digest_before,
            "adapter_sha_after": digest_after,
            "weights_unchanged": digest_before == digest_after,
            "gpu_hours_spent_on_the_edit": 0.0,
        },
        "ontology": {
            "before": a.before, "after": a.after,
            "counts": onto_counts,
            "card_index_build_s": build_s,
        },
        "cards": {n: {v: cards[v].get(n, []) for v in (a.before, a.after)}
                  for n in ("Woljeongri Beach", "Jungmun Saekdal Beach",
                            "Yongmeori Coast")},
        "items": rows,
    }
    p = OUT / "demo_live_edit.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
