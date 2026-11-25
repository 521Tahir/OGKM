"""
The demo the business-plan brief asks for: a wrong answer becoming a right one
because somebody edited the graph, and nothing else.

The brief (Ontology-Comment, 19 Oct) wants the demo shown as a before-and-after
comparison rather than as architecture, and it wants the two corrections a
customer recognises:

    [Flagship Software V1] --(is_incompatible_with)--> [Cloud Server 2026]
    [Contractor]           --(remote_work)-->          [Not Eligible]

A hard requirement and a hard category rule. `extract_policy_constraints.py`
asserts this ontology's equivalents in v1.6.0 - a rental car needs an
International Driving Permit, a market stall takes cash, a temple expects
covered shoulders and bare feet - from corpus evidence with provenance. This
script asks the same model the same questions on either side of that edit.

What is held constant, so the only explanation left is the graph:
  * one model, one adapter, loaded once, greedy decoding, same prompt template;
  * the same questions in the same order;
  * the same context builder and the same number of entity cards;
  * the adapter's file digest is recomputed after the run and compared.

Half the questions are controls: they ask about something the edit did not
touch. An edit that changes those too is not a correction, it is a
perturbation, and the report counts both.

  EXP024_TAG=ded python -m wbs.task5_demo.demo_policy_before_after --gpu 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "wbs" / "deliverables" / "presentation"

# Each probe: the customer's question, the triple v1.6.0 adds that answers it,
# and the language an answer must contain to count as having applied the rule.
# `wrong` is the language that marks the permissive answer the rule contradicts.
PROBES = [
    dict(id="idp", kind="requirement",
         q="I land at Jeju International Airport next Tuesday and I have booked "
           "a rental car. I have my own national driving licence with me. Is "
           "that all I need to drive on Jeju?",
         triple=("Rental car", "requires", "International Driving Permit"),
         right=r"international driving permit|international driver'?s licen|\bIDP\b",
         wrong=r"\b(that'?s all you need|all you need|your national licence is "
               r"(fine|enough|sufficient)|you can (just )?(drive|pick up)|no "
               r"additional|nothing else)\b",
         rule="A rental car requires an International Driving Permit."),
    dict(id="cash_dongmun", kind="requirement",
         q="We are going to Dongmun Market for dinner tonight. Can we pay the "
           "food stalls with a credit card?",
         triple=("Dongmun Market", "requires", "Cash Payment"),
         right=r"cash",
         wrong=r"(credit cards? (are|is) (widely )?accepted|most (stalls|vendors) "
               r"accept cards|you can pay by card)",
         rule="Dongmun Market requires cash."),
    dict(id="cash_seogwipo", kind="requirement",
         q="Is a credit card fine for shopping at Seogwipo Maeil Olle Market?",
         triple=("Seogwipo Maeil Olle Market", "requires", "Cash Payment"),
         right=r"cash",
         wrong=r"(cards? (are|is) (widely )?accepted|yes,? (you can|it'?s fine))",
         rule="Seogwipo Maeil Olle Market requires cash."),
    dict(id="dress_gwaneumsa", kind="requirement",
         q="We will visit Gwaneumsa Temple straight from the beach, so we will "
           "be in swim shorts and tank tops. Is there any dress rule we should "
           "know about?",
         triple=("Gwaneumsa Temple", "requires", "Modest Dress"),
         right=r"modest|cover(ing)? (your )?(shoulders|knees)|shoulders and knees",
         wrong=r"(no (specific )?dress code|any (clothing|attire) is fine|"
               r"casual (clothing|wear|attire) is (fine|acceptable))",
         rule="Gwaneumsa Temple requires modest dress."),
    dict(id="shoes_sanbanggulsa", kind="requirement",
         q="Can I keep my shoes on when I go inside Sanbanggulsa Temple?",
         triple=("Sanbanggulsa Temple", "requires", "Shoes Removed"),
         right=r"remove (your )?shoes|take off (your )?shoes|shoes off|barefoot",
         wrong=r"(you can keep (your|them)|shoes are (fine|allowed)|no need to "
               r"(remove|take off))",
         rule="Sanbanggulsa Temple requires shoes to be removed."),
    dict(id="shoes_folk", kind="requirement",
         q="At Jeju Folk Village, do we need to do anything before we step into "
           "the traditional houses?",
         triple=("Jeju Folk Village", "requires", "Shoes Removed"),
         right=r"remove (your )?shoes|take off (your )?shoes|shoes off|barefoot",
         wrong=r"(nothing special|just walk in|no (special )?requirement)",
         rule="Jeju Folk Village requires shoes to be removed."),
    dict(id="drone_stone_park", kind="restriction",
         q="I want aerial shots of Jeju Stone Park with my drone. Can I fly it "
           "there?",
         triple=("Drone Flying", "restricted_in", "Jeju Stone Park"),
         right=r"restrict|regulation|not allowed|prohibit|permission|permit",
         wrong=r"(yes,? you can fly|drones? (are|is) allowed|free to fly|"
               r"no restrictions)",
         rule="Drone flying is restricted in Jeju Stone Park."),
    dict(id="drone_sangumburi", kind="restriction",
         q="Is it fine to launch a drone at Sangumburi Crater for photos?",
         triple=("Drone Flying", "restricted_in", "Sangumburi Crater"),
         right=r"restrict|regulation|not allowed|prohibit|permission|permit",
         wrong=r"(yes,? you can|drones? (are|is) allowed|no restrictions)",
         rule="Drone flying is restricted in Sangumburi Crater."),
    dict(id="sunscreen_jungmun", kind="requirement",
         q="Which sunscreen should I take for snorkelling at Jungmun Saekdal "
           "Beach?",
         triple=("Jungmun Saekdal Beach", "requires", "Reef-safe Sunscreen"),
         right=r"reef[- ]safe|reef friendly|reef-friendly",
         wrong=r"(any (sunscreen|brand)|SPF \d+ (is|will be) (fine|enough))",
         rule="Jungmun Saekdal Beach requires reef-safe sunscreen."),
    dict(id="sunscreen_hamdeok", kind="requirement",
         q="We are swimming at Hamdeok Beach with the children. Any advice on "
           "sun protection?",
         triple=("Hamdeok Beach", "requires", "Reef-safe Sunscreen"),
         right=r"reef[- ]safe|reef friendly|reef-friendly",
         wrong=r"(any (sunscreen|brand)|a (high|good) SPF is (all|enough))",
         rule="Hamdeok Beach requires reef-safe sunscreen."),
    # ── controls: nothing in the edit touches these ──────────────────────
    dict(id="ctl_manjanggul", kind="control",
         q="What is Manjanggul Cave and why is it worth visiting?", triple=None,
         right=None, wrong=None, rule=""),
    dict(id="ctl_food", kind="control",
         q="What should we eat in Seogwipo on a rainy evening?", triple=None,
         right=None, wrong=None, rule=""),
    dict(id="ctl_hallasan", kind="control",
         q="How long does the Seongpanak trail up Hallasan take, and how hard "
           "is it?", triple=None, right=None, wrong=None, rule=""),
    dict(id="ctl_transport", kind="control",
         q="How do we get from Jeju International Airport to Seogwipo without "
           "a car?", triple=None, right=None, wrong=None, rule=""),
    dict(id="ctl_itinerary", kind="control",
         q="Plan a two-day east-coast itinerary for a couple in October.",
         triple=None, right=None, wrong=None, rule=""),
    dict(id="ctl_haenyeo", kind="control",
         q="Who are the haenyeo and where can we see them?", triple=None,
         right=None, wrong=None, rule=""),
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
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--before", default="1.5.0")
    ap.add_argument("--after", default="1.6.0")
    ap.add_argument("--max-new", type=int, default=260)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer, render_prompt,
                                                         supports_system_role)
    from experiments.exp024_rag_vs_ontology.contexts import OntoV3ContextBuilder
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    qs = [p["q"] for p in PROBES]
    ctx, cards, build_s = {}, {}, {}
    for v in (a.before, a.after):
        b = OntoV3ContextBuilder(onto_version=v, device="cpu", safety_first=True)
        t0 = time.perf_counter()
        b.build()
        build_s[v] = round(time.perf_counter() - t0, 1)
        b.load()
        ctx[v] = [r["context"] for r in b.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)]
        cards[v] = {c["name"]: c["facts"] for c in b._cards}
        print(f"[{v}] index built in {build_s[v]}s", flush=True)
        del b

    cfg = C.MODEL_BY_KEY[a.model]
    adapter = C.ADAPTER_DIR / f"{a.model}__onto_v3"
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"adapter {adapter} not found (EXP024_TAG=ded?)")
    digest_before = sha(adapter)
    src = str(adapter) if (adapter / "tokenizer_config.json").exists() else cfg["hf_id"]
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src), for_training=False)
    sys_ok = supports_system_role(tok)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()

    def gen(text: str) -> tuple[str, float]:
        enc = tok(text, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return (tok.decode(g[0][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip(),
                time.perf_counter() - t0)

    triples = {}
    for v in (a.before, a.after):
        rel = json.loads((ROOT / "artifacts" / "ontologies" / v / "relations.json")
                         .read_text(encoding="utf-8"))
        triples[v] = {(str(r["subject"]).lower(), str(r["predicate"]),
                       str(r["object"]).lower()) for r in rel}

    rows = []
    for i, p in enumerate(PROBES):
        row = {k: p[k] for k in ("id", "kind", "q", "rule")}
        row["triple"] = list(p["triple"]) if p["triple"] else None
        for v in (a.before, a.after):
            prompt = render_prompt(tok, p["q"], "onto_v3", ctx[v][i], sys_ok)
            ans, dt = gen(prompt)
            asserted = (p["triple"] is not None
                        and (p["triple"][0].lower(), p["triple"][1],
                             p["triple"][2].lower()) in triples[v])
            row[v] = {
                "answer": ans, "generation_s": round(dt, 2),
                "context_chars": len(ctx[v][i]),
                "states_the_rule": bool(p["right"] and re.search(p["right"], ans, re.I)),
                "states_the_opposite": bool(p["wrong"] and re.search(p["wrong"], ans, re.I)),
                "rule_asserted_in_graph": asserted,
                "rule_in_the_context_given_to_the_model": bool(
                    p["triple"] and p["triple"][2].lower() in ctx[v][i].lower()
                    and p["triple"][0].lower() in ctx[v][i].lower()),
            }
        row["changed"] = row[a.before]["answer"] != row[a.after]["answer"]
        rows.append(row)
        tag = {"requirement": "REQ", "restriction": "RES", "control": "CTL"}[p["kind"]]
        print(f"\n### [{tag}] {p['id']}  changed={row['changed']}", flush=True)
        for v in (a.before, a.after):
            r = row[v]
            print(f"  v{v}  rule_in_context={r['rule_in_the_context_given_to_the_model']} "
                  f"states_rule={r['states_the_rule']} "
                  f"opposite={r['states_the_opposite']}", flush=True)
            print(f"    {r['answer'][:300]}", flush=True)

    digest_after = sha(adapter)
    tgt = [r for r in rows if r["kind"] != "control"]
    ctl = [r for r in rows if r["kind"] == "control"]
    fixed = [r["id"] for r in tgt
             if r[a.after]["states_the_rule"] and not r[a.before]["states_the_rule"]]
    out = {
        "model": cfg["name"], "hf_id": cfg["hf_id"], "adapter": adapter.name,
        "gpu": a.gpu, "decoding": f"greedy, max_new_tokens={a.max_new}",
        "before": a.before, "after": a.after,
        "retraining": {"adapter_sha_before": digest_before,
                       "adapter_sha_after": digest_after,
                       "weights_unchanged": digest_before == digest_after,
                       "gpu_hours_spent_on_the_edit": 0.0},
        "edit": {"triples_added": len(triples[a.after] - triples[a.before]),
                 "card_index_build_s": build_s},
        "summary": {
            "targeted": len(tgt), "controls": len(ctl),
            "rule_stated_before": sum(r[a.before]["states_the_rule"] for r in tgt),
            "rule_stated_after": sum(r[a.after]["states_the_rule"] for r in tgt),
            "opposite_before": sum(r[a.before]["states_the_opposite"] for r in tgt),
            "opposite_after": sum(r[a.after]["states_the_opposite"] for r in tgt),
            "rule_reached_context_before": sum(
                r[a.before]["rule_in_the_context_given_to_the_model"] for r in tgt),
            "rule_reached_context_after": sum(
                r[a.after]["rule_in_the_context_given_to_the_model"] for r in tgt),
            "targeted_answers_changed": sum(r["changed"] for r in tgt),
            "control_answers_changed": sum(r["changed"] for r in ctl),
            "corrected": fixed,
        },
        "cards": {n: {v: cards[v].get(n, []) for v in (a.before, a.after)}
                  for n in sorted({p["triple"][0] for p in PROBES if p["triple"]}
                                  | {p["triple"][2] for p in PROBES if p["triple"]})},
        "items": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = Path(a.out) if a.out else OUT / f"demo_policy_before_after__{a.model}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    s = out["summary"]
    print(f"\n=== {cfg['name']}: rule stated {s['rule_stated_before']} → "
          f"{s['rule_stated_after']} of {s['targeted']} targeted; "
          f"permissive-wrong {s['opposite_before']} → {s['opposite_after']}; "
          f"controls changed {s['control_answers_changed']}/{s['controls']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
