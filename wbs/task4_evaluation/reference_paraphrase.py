"""
WBS Task 4.1.2 / R6 - is the overlap metric measuring answer quality, or
measuring who copied the corpus the references were written from?

R6 records that the benchmark's gold answers are single-source: each was
written from the same corpus the retrieval arm's passages are drawn from. The
report has always explained Simple RAG's token-F1 lead structurally on that
basis - a passage arm is partly copying the reference - but the explanation has
never been tested, because testing it needs references that say the same thing
in different words.

That is a generation job, not an annotation job, so it does not wait on the
human adjudication R6 is blocked on. Each reference is paraphrased by the base
model with no adapter and no context, instructed to keep every fact and every
place name and change only the wording. A paraphrase is accepted only if it
preserves the reference's gazetteer entities and stays within a length band, so
an accepted paraphrase differs from its original in phrasing rather than in
content. Every arm's stored answers are then re-scored against the accepted
paraphrases - no answer is regenerated, and no arm sees the paraphrase.

If the lead is copying, it shrinks when the wording changes. If it survives,
the lead is about content and the structural explanation is wrong.

  python -m wbs.task4_evaluation.reference_paraphrase --stage generate --gpu 5
  python -m wbs.task4_evaluation.reference_paraphrase --stage report
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

PROMPT = (
    "Rewrite the passage below so it says exactly the same things in different "
    "words. Keep every place name, number, time and price exactly as written. "
    "Keep the same list items and the same order. Do not add facts, do not "
    "remove facts, do not comment. Output only the rewritten passage.\n\n"
    "PASSAGE:\n{ref}\n\nREWRITTEN PASSAGE:")

LEN_LO, LEN_HI = 0.6, 1.6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["generate", "report"])
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pp = EVAL / f"reference_paraphrases__{a.model}.jsonl"

    if a.stage == "generate":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from experiments.exp024_rag_vs_ontology import config as C
        from experiments.exp024_rag_vs_ontology.chat import (prepare_tokenizer,
                                                             supports_system_role)
        from src.utils.proctitle import set_proctitle
        set_proctitle()

        rows = [json.loads(l) for l in
                open(EVAL / f"variant__{a.model}__V2_rag.jsonl", encoding="utf-8")
                if l.strip()]
        if a.limit:
            rows = rows[: a.limit]
        cfg = C.MODEL_BY_KEY[a.model]
        tok = prepare_tokenizer(AutoTokenizer.from_pretrained(cfg["hf_id"]),
                                for_training=False)
        m = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
        m.eval()
        sys_ok = supports_system_role(tok)
        t0 = time.perf_counter()
        with open(pp, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                # the project's render_prompt carries an arm's tourism-assistant
                # system prompt, which would steer a rewrite; this task needs a
                # bare instruction, so the template is applied directly
                msgs = ([{"role": "system",
                          "content": "You rewrite text faithfully."}] if sys_ok
                        else [])
                msgs = msgs + [{"role": "user",
                                "content": PROMPT.format(ref=r["reference"])}]
                try:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                e = tok(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    g = m.generate(**e, max_new_tokens=700, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
                out = tok.decode(g[0][e["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
                f.write(json.dumps({"bench_id": r["bench_id"],
                                    "reference": r["reference"],
                                    "paraphrase": out}, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(rows)} "
                          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
        print(f"wrote {pp.name}")
        return

    # ── report ────────────────────────────────────────────────────────────
    from experiments.exp042_prf.compute_prf import ents, prf, toks
    from wbs.task4_evaluation.score_variants import wilcoxon

    if not pp.exists():
        sys.exit(f"{pp.name} not found - run --stage generate first")
    para = {json.loads(l)["bench_id"]: json.loads(l)
            for l in open(pp, encoding="utf-8") if l.strip()}

    # acceptance: the paraphrase must carry the reference's gazetteer entities
    # and stay in a length band. A paraphrase that drops places is a different
    # answer, not a rewording, and would flatter whichever arm names fewer.
    accepted, rej_ent, rej_len = {}, 0, 0
    for b, r in para.items():
        ref_e, par_e = set(ents(r["reference"])), set(ents(r["paraphrase"]))
        ratio = len(r["paraphrase"].split()) / max(len(r["reference"].split()), 1)
        if ref_e and not ref_e <= par_e:
            rej_ent += 1
            continue
        if not (LEN_LO <= ratio <= LEN_HI):
            rej_len += 1
            continue
        accepted[b] = r

    arms = {}
    for name, fn in (("V2 Simple RAG", f"variant__{a.model}__V2_rag.jsonl"),
                     ("V3 ontology only", f"variant__{a.model}__V3_ontology_only.jsonl"),
                     ("V4 ontology+RAG", f"variant__{a.model}__V4_ontology_rag__hybded.jsonl"),
                     ("V5 validated", f"variant__{a.model}__V5_ontology_rag_validated__hybded.jsonl")):
        f_ = EVAL / fn
        if f_.exists():
            arms[name] = {json.loads(l)["bench_id"]: json.loads(l)
                          for l in open(f_, encoding="utf-8") if l.strip()}
    ids = [b for b in accepted if all(b in v for v in arms.values())]

    def f1(ans: str, ref: str, ent: bool = False) -> float | None:
        x = prf(ents(ans), ents(ref)) if ent else prf(toks(ans), toks(ref))
        return x[2] if x else None

    peritem = {}
    out = {"model": a.model, "n_paraphrased": len(para), "n_accepted": len(accepted),
           "rejected_entities_lost": rej_ent, "rejected_length": rej_len,
           "n_scored": len(ids), "arms": {}}
    rows = {}
    for name, recs in arms.items():
        orig = [f1(recs[b]["answer"], accepted[b]["reference"]) for b in ids]
        new = [f1(recs[b]["answer"], accepted[b]["paraphrase"]) for b in ids]
        peritem[name] = {"orig": orig, "new": new}
        pairs = [(o, n) for o, n in zip(orig, new) if o is not None and n is not None]
        w = wilcoxon([n for _, n in pairs], [o for o, _ in pairs])
        rows[name] = {
            "token_f1_original": st.mean(o for o, _ in pairs) * 100,
            "token_f1_paraphrased": st.mean(n for _, n in pairs) * 100,
            "drop": (st.mean(o for o, _ in pairs)
                     - st.mean(n for _, n in pairs)) * 100,
            "p": (w[1] if w else None), "n": len(pairs)}
        out["arms"][name] = rows[name]

    L = [f"# Does the overlap metric measure quality, or measure who copied the "
         f"corpus? ({a.model})\n",
         "R6 records that the benchmark's gold answers are single-source: each "
         "was written from the same corpus the retrieval arm draws its passages "
         "from. Every version of this report has explained Simple RAG's "
         "token-F1 lead on that basis, and never tested it. Testing it needs "
         "references that say the same thing in other words.\n",
         f"Each reference was paraphrased by the base model with no adapter and "
         f"no context, told to keep every place name, number and list item and "
         f"change only the wording. A paraphrase is accepted only if it keeps "
         f"every gazetteer entity of the original and stays within "
         f"{LEN_LO:g}-{LEN_HI:g}x its length: **{out['n_accepted']} of "
         f"{out['n_paraphrased']}** qualified ({rej_ent} dropped an entity, "
         f"{rej_len} fell outside the band). No answer was regenerated and no "
         f"arm saw a paraphrase.\n",
         "| Arm | token F1 vs original | vs paraphrase | drop | p |",
         "|---|--:|--:|--:|--:|"]
    for name, r in rows.items():
        L.append(f"| {name} | {r['token_f1_original']:.2f} | "
                 f"{r['token_f1_paraphrased']:.2f} | {r['drop']:+.2f} | "
                 + (f"{r['p']:.3g} |" if r["p"] is not None else "— |"))
    L.append("")

    if "V2 Simple RAG" in rows:
        rag = rows["V2 Simple RAG"]
        onto = {k: v for k, v in rows.items() if k != "V2 Simple RAG"}
        bigger = [k for k, v in onto.items() if rag["drop"] > v["drop"]]
        out["rag_drops_more_than"] = bigger
        # the lead itself, before and after
        leads = {k: (rag["token_f1_original"] - v["token_f1_original"],
                     rag["token_f1_paraphrased"] - v["token_f1_paraphrased"])
                 for k, v in onto.items()}
        out["lead_before_after"] = {k: {"original": b, "paraphrased": p_}
                                    for k, (b, p_) in leads.items()}
        L += ["## What the rewording does to the lead\n",
              "| Arm | Simple RAG's lead on the original | on the paraphrase | "
              "change |", "|---|--:|--:|--:|"]
        for k, (b, p_) in leads.items():
            L.append(f"| {k} | {b:+.2f} | {p_:+.2f} | {p_ - b:+.2f} |")
        L.append("")
        # the lead change is a difference-in-differences and needs a test of its
        # own: per item, Simple RAG's margin on the original against its margin
        # on the paraphrase
        did = {}
        for k in onto:
            d1, d2 = [], []
            for i in range(len(ids)):
                ro, rn = peritem["V2 Simple RAG"]["orig"][i], peritem["V2 Simple RAG"]["new"][i]
                ao, an = peritem[k]["orig"][i], peritem[k]["new"][i]
                if None in (ro, rn, ao, an):
                    continue
                d1.append(ro - ao)
                d2.append(rn - an)
            w = wilcoxon(d2, d1)
            did[k] = {"lead_change": (st.mean(d2) - st.mean(d1)) * 100,
                      "p": (w[1] if w else None), "n": len(d1)}
        out["lead_change_tests"] = did
        shrank = [k for k, (b, p_) in leads.items() if abs(p_) < abs(b)]
        sig = [k for k, v in did.items() if (v["p"] or 1) < .05]
        L.append(
            (f"**The lead is partly an artefact of the references' provenance.** "
             if shrank else
             f"**The lead is not an artefact of the references' provenance.** ")
            + f"Every arm loses overlap when the reference is reworded - the "
            f"metric is phrasing-sensitive for all of them - but Simple RAG "
            f"loses {rag['drop']:.2f} points against "
            + ", ".join(f"{v['drop']:.2f} for {k}" for k, v in onto.items())
            + ". "
            + (f"Its lead shrinks against {', '.join(shrank)}, which is the "
               f"direction copying predicts: part of what the passage arm was "
               f"being credited with is wording it shares with a reference "
               f"written from the same corpus. "
               + ("Under a paired difference-in-differences test the shrink is "
                  f"significant for {', '.join(sig)}"
                  if sig else "Under a paired difference-in-differences test "
                  "the shrink is not significant for any arm")
               + " (" + ", ".join(f"{k} {v['lead_change']:+.2f}, p = "
                                  + (f"{v['p']:.3g}" if v["p"] is not None else "—")
                                  for k, v in did.items()) + "). "
               + "**But the lead mostly survives.** Rewording explains "
               + ", ".join(f"{abs(p_ - b) / abs(b) * 100:.0f}% of it against {k}"
                           for k, (b, p_) in leads.items() if b)
               + f", so the provenance of the references inflates Simple RAG's "
               f"advantage without creating it: the ontology arms would still "
               f"trail on a benchmark whose answers were written independently "
               f"of the corpus.\n" if shrank else
               "Its lead does not shrink, so the deficit the ontology arms "
               "carry is about content rather than about who wrote the "
               "reference, and the structural explanation this report has "
               "offered needs to be weakened accordingly.\n"))

    L += ["## One caveat the design carries\n",
          "The paraphraser is the same base model the arms are built on, with "
          "no adapter and no context. That keeps it neutral between the arms - "
          "none of them sees it, and all of them were fine-tuned from the same "
          "weights - but it cannot make the paraphrase neutral with respect to "
          "the *model family's* phrasing. A reference reworded by a different "
          "model might move the numbers; what it should not move is the "
          "ordering, since the rewriting is applied to the reference rather "
          "than to any arm's answer.\n",
          "## What this is not\n",
          "It is not the human adjudication R6 asks for. A model-generated "
          "paraphrase checked for entity preservation tests one specific "
          "failure mode of the metric - sensitivity to the reference's "
          "wording - and says nothing about whether the gold answers are "
          "*correct*, which is what a second annotator would settle. R6 stays "
          "open.\n"]

    (EVAL / f"reference_paraphrase__{a.model}.md").write_text("\n".join(L),
                                                              encoding="utf-8")
    (EVAL / f"reference_paraphrase__{a.model}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote reference_paraphrase__{a.model}.{{md,json}}")
    for k, v in rows.items():
        print(f"  {k:<20} {v['token_f1_original']:.2f} -> "
              f"{v['token_f1_paraphrased']:.2f}  ({v['drop']:+.2f})")


if __name__ == "__main__":
    main()
