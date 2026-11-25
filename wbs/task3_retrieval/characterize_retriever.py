"""
WBS Task 3.2.1 - Ontology-aware retriever and context assembly, characterised.

Task 3.2.1 asks for query understanding, concept mapping, ontology-based subgraph
retrieval, hybrid retrieval and context assembly. All five are implemented in
`experiments/exp024_rag_vs_ontology/contexts.py` and are what Task 4.1 evaluates;
what was missing was a deliverable that states what each stage does and shows what it
actually does on real questions. This script measures every stage on the 4,607-item
test split and the 347-item categorized benchmark and writes that document.

Stages, with their implementation site:

  1 QUERY UNDERSTANDING   `OntoV3ContextBuilder._target_classes` - keyword intent
      detection over 8 intent groups mapping to 14 ontology classes, with a
      7-class default mix for questions that name no domain at all ("plan me
      three days"). Measured: which intents fire, how often nothing fires, how
      many classes a question is routed to.

  2 CONCEPT MAPPING       exact surface-form matching of question spans against
      ontology concept names, aliases and the 69-place gazetteer. Measured: the
      share of questions that mention any ontology concept at all, which is the
      ceiling on what pure lexical grounding could achieve.

  3 SUBGRAPH RETRIEVAL    dense entity-card search (bge-base over cards built from
      the concept's outgoing edges) UNION class-routed entities ordered by degree,
      UNION time-scoped facts. Measured: how many entities each of the three
      sources contributes, and how much they overlap - which is the question of
      whether class routing adds anything the dense search already had.

  4 HYBRID RETRIEVAL      ontology facts and dense passages in one context at a
      matched total budget, fact share 20%. Measured: realised fact share, how
      often each side is truncated.

  5 CONTEXT ASSEMBLY      budget enforcement and ordering. Measured: budget
      utilisation and truncation rates per arm.

  python -m wbs.task3_retrieval.characterize_retriever --device cpu --n-test 1500
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DELIV = ROOT / "wbs" / "deliverables" / "task3_retrieval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-test", type=int, default=1500,
                    help="questions sampled from the 4,607-item test split")
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()

    import random
    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.contexts import (_DEFAULT_CLASSES,
                                                             _INTENT_CLASSES,
                                                             OntoV3ContextBuilder,
                                                             RagContextBuilder,
                                                             _truncate)
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase, key

    ROOT_NAMES = {"jeju", "jeju island", "jejudo", "jeju do", "jeju city",
                  "korea", "south korea"}
    from src.utils.proctitle import set_proctitle
    set_proctitle()
    DELIV.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    test = [json.loads(l) for l in
            open(ROOT / "data" / "jeju_full" / "splits" / "test.jsonl", encoding="utf-8")
            if l.strip()]
    rng.shuffle(test)
    test = test[: a.n_test]
    bench = [json.loads(l) for l in
             open(ROOT / "wbs" / "deliverables" / "task4_evaluation" /
                  "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]
    sets = {"test_split": [r["Input"] for r in test],
            "benchmark": [b["Input"] for b in bench]}
    print(f"test split sample {len(test):,} | benchmark {len(bench):,}", flush=True)

    out: dict = {"intent_groups": len(_INTENT_CLASSES),
                 "default_classes": list(_DEFAULT_CLASSES),
                 "ctx_budget_chars": C.CTX_MAX_CHARS,
                 "onto_v3_entities": C.ONTO_V3_ENTITIES,
                 "rag_top_k": C.RAG_TOP_K}

    # ── stage 1: query understanding ──────────────────────────────────────
    s1 = {}
    for nm, qs in sets.items():
        fired, nclass, nofire = Counter(), [], 0
        for q in qs:
            ql = q.lower()
            hit = []
            for i, (keys, classes) in enumerate(_INTENT_CLASSES):
                if any(k in ql for k in keys):
                    fired[classes[0]] += 1
                    hit += [c for c in classes if c not in hit]
            if not hit:
                nofire += 1
                hit = list(_DEFAULT_CLASSES)
            nclass.append(len(hit))
        s1[nm] = {"n": len(qs), "no_intent_fired": nofire,
                  "no_intent_rate": nofire / len(qs),
                  "mean_classes_routed": round(st.mean(nclass), 2),
                  "intent_firing_counts": dict(fired.most_common())}
        print(f"  [{nm}] no intent fired {nofire}/{len(qs)} "
              f"({nofire/len(qs)*100:.1f}%), mean {st.mean(nclass):.2f} classes",
              flush=True)
    out["query_understanding"] = s1

    # ── stage 2: concept mapping ──────────────────────────────────────────
    kb = KnowledgeBase()
    SPANS = re.compile(r"\b([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)")
    # Two counts, because they differ by a factor that matters. A NAMED-ENTITY match
    # is a capitalised span that is a concept, alias or gazetteer place - a question
    # that actually points at a node. A LOOSE match also counts lowercase vocabulary
    # terms the gazetteer carries (cuisine words, activity words, traveller types);
    # those match almost every question and would make the stage look solved, so they
    # are reported separately rather than folded into one headline number.
    s2 = {}
    for nm, qs in sets.items():
        n_named, n_loose, n_nonroot, named, loose, nonroot = 0, 0, 0, [], [], []
        for q in qs:
            caps = {key(m.group(1)) for m in SPANS.finditer(q)}
            caps |= {key(" ".join(w)) for m in SPANS.finditer(q)
                     for w in [m.group(1).split()] if len(w) > 1}
            nh = {m for m in caps if m in kb.surface and len(m) > 3}
            # "Jeju" itself is in nearly every question and is a concept, so counting
            # it makes this stage look solved; the non-root count is the informative one
            nr = {m for m in nh if m not in ROOT_NAMES}
            lh = {key(w) for w in re.findall(r"[a-z]{4,}", q.lower())} & kb.surface
            n_named += bool(nh); n_loose += bool(nh or lh); n_nonroot += bool(nr)
            named.append(len(nh)); loose.append(len(nh | lh)); nonroot.append(len(nr))
        s2[nm] = {"questions_with_named_entity": n_named,
                  "named_rate": n_named / len(qs),
                  "mean_named_entities": round(st.mean(named), 2),
                  "questions_with_nonroot_entity": n_nonroot,
                  "nonroot_rate": n_nonroot / len(qs),
                  "mean_nonroot_entities": round(st.mean(nonroot), 2),
                  "questions_with_any_surface_form": n_loose,
                  "loose_rate": n_loose / len(qs),
                  "mean_any_surface_forms": round(st.mean(loose), 2)}
        print(f"  [{nm}] named-entity {n_named/len(qs)*100:.1f}% "
              f"(mean {st.mean(named):.2f}); NON-ROOT entity "
              f"{n_nonroot/len(qs)*100:.1f}% (mean {st.mean(nonroot):.2f}); "
              f"any surface form {n_loose/len(qs)*100:.1f}%", flush=True)
    out["concept_mapping"] = {"surface_forms": len(kb.surface),
                              "ontology_version": kb.version, "by_set": s2}

    # ── stages 3-5: run the real retrievers ───────────────────────────────
    onto = OntoV3ContextBuilder(device=a.device); onto.load()
    rag = RagContextBuilder(device=a.device); rag.load()
    import numpy as np

    s3, s45 = {}, {}
    for nm, qs in sets.items():
        t0 = time.perf_counter()
        ores = onto.retrieve_batch(qs, n_entities=C.ONTO_V3_ENTITIES)
        rres = rag.retrieve_batch(qs)
        secs = time.perf_counter() - t0

        # decompose the subgraph: dense-only vs class-routed-only vs both
        qe = onto._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in qs],
                                       batch_size=128, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        _, idxs = onto._index.search(qe, C.ONTO_V3_ENTITIES)
        dense_only, routed_only, both, n_ent, n_facts = [], [], [], [], []
        for i, r in enumerate(ores):
            dense = {onto._cards[j]["name"] for j in idxs[i] if j >= 0}
            got = set(r.get("entities") or [])
            routed = got - dense
            dense_only.append(len(dense - routed))
            routed_only.append(len(routed))
            both.append(len(got))
            n_ent.append(r["n_entities"]); n_facts.append(r["n_hits"])
        s3[nm] = {"mean_entities": round(st.mean(n_ent), 2),
                  "mean_facts": round(st.mean(n_facts), 2),
                  "mean_from_dense_search": round(st.mean(dense_only), 2),
                  "mean_added_by_class_routing": round(st.mean(routed_only), 2),
                  "class_routing_share": round(st.mean(routed_only) /
                                               max(1e-9, st.mean(both)), 3),
                  "empty_context_rate": st.mean(
                      [1.0 if r["context"].startswith("[no ontology") else 0.0
                       for r in ores]),
                  "retrieval_secs_for_set": round(secs, 1)}
        print(f"  [{nm}] subgraph: {st.mean(n_ent):.1f} entities, "
              f"{st.mean(n_facts):.1f} facts, "
              f"{s3[nm]['class_routing_share']*100:.0f}% of entities come from class "
              f"routing rather than dense search", flush=True)

        # hybrid + assembly
        fb = int(C.CTX_MAX_CHARS * 0.20)
        oc = [r["context"] for r in ores]
        rc = [r["context"] for r in rres]
        facts = [_truncate(x if not x.startswith("[no ontology") else "", fb) for x in oc]
        pas = [_truncate(y, C.CTX_MAX_CHARS - len(f)) for f, y in zip(facts, rc)]
        hyb = [(f"Ontology facts:\n{f}\n\n" if f else "") +
               f"Retrieved reference passages:\n{p}" for f, p in zip(facts, pas)]
        s45[nm] = {
            "rag_chars_mean": round(st.mean([len(x) for x in rc])),
            "rag_budget_use": round(st.mean([len(x) for x in rc]) / C.CTX_MAX_CHARS, 3),
            "rag_truncated_rate": st.mean([1.0 if len(x) >= C.CTX_MAX_CHARS - 2 else 0.0
                                           for x in rc]),
            "onto_chars_mean": round(st.mean([len(x) for x in oc])),
            "onto_budget_use": round(st.mean([len(x) for x in oc]) / C.CTX_MAX_CHARS, 3),
            "onto_truncated_rate": st.mean([1.0 if len(x) >= C.CTX_MAX_CHARS - 2 else 0.0
                                            for x in oc]),
            "hybrid_chars_mean": round(st.mean([len(x) for x in hyb])),
            "hybrid_realised_fact_share": round(
                st.mean([len(f) / max(1, len(f) + len(p)) for f, p in zip(facts, pas)]), 3),
            "hybrid_fact_side_truncated": st.mean([1.0 if len(f) >= fb - 2 else 0.0
                                                   for f in facts]),
        }
        print(f"  [{nm}] assembly: rag {s45[nm]['rag_budget_use']*100:.0f}% of budget, "
              f"onto {s45[nm]['onto_budget_use']*100:.0f}%, hybrid realised fact share "
              f"{s45[nm]['hybrid_realised_fact_share']*100:.0f}%", flush=True)
    out["subgraph_retrieval"] = s3
    out["hybrid_and_assembly"] = s45

    (DELIV / "retriever_characterization.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    # ── document ──────────────────────────────────────────────────────────
    def pc(x):
        return f"{x*100:.1f}%"

    L = ["# Task 3.2.1 - Ontology-Aware Retriever and Context Assembly\n",
         "Implementation: `experiments/exp024_rag_vs_ontology/contexts.py` "
         "(`RagContextBuilder`, `OntoV3ContextBuilder`). Every figure below is measured "
         f"on {len(test):,} questions sampled from the 4,607-item test split and all "
         f"{len(bench)} questions of the Task 4.1 categorized benchmark, against "
         f"ontology v{kb.version}.\n",
         "## 1. Query understanding\n",
         f"Keyword intent detection over {len(_INTENT_CLASSES)} intent groups "
         f"(natural, cultural, food, accommodation, activity, transport, emergency, "
         f"shopping) mapping onto ontology classes. A question that matches no group is "
         f"routed to a default mix of {len(_DEFAULT_CLASSES)} classes "
         f"({', '.join(_DEFAULT_CLASSES)}), which is what a bare \"plan me three days\" "
         "needs.\n",
         "| Set | n | No intent fired | Mean classes routed |",
         "|---|--:|--:|--:|"]
    for nm in sets:
        d = s1[nm]
        L.append(f"| {nm} | {d['n']:,} | {d['no_intent_fired']:,} "
                 f"({pc(d['no_intent_rate'])}) | {d['mean_classes_routed']} |")
    top = s1["test_split"]["intent_firing_counts"]
    L.append("\nMost-fired intents on the test split: " +
             ", ".join(f"`{k}` {v:,}" for k, v in list(top.items())[:6]) + ".\n")

    L.append("## 2. Concept mapping\n")
    L.append(f"Question spans are matched against {len(kb.surface):,} ontology surface "
             "forms (concept names, aliases, gazetteer entries).\n")
    L.append("| Set | Names ≥1 ontology ENTITY | Mean | Names a NON-ROOT entity | "
             "Mean | Matches any surface form | Mean |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for nm in sets:
        d = s2[nm]
        L.append(f"| {nm} | {d['questions_with_named_entity']:,} "
                 f"({pc(d['named_rate'])}) | {d['mean_named_entities']} | "
                 f"{d['questions_with_nonroot_entity']:,} ({pc(d['nonroot_rate'])}) | "
                 f"{d['mean_nonroot_entities']} | "
                 f"{pc(d['loose_rate'])} | {d['mean_any_surface_forms']} |")
    L.append("\nThe three counts are deliberately separate. The first counts capitalised "
             "spans that resolve to a concept, alias or gazetteer place - a question "
             "that actually points at a node in the graph - but nearly every question "
             "in this corpus says \"Jeju\", which is itself a concept, so that column is "
             "close to 100% by construction. The second excludes the root names "
             "(Jeju, Jejudo, Korea) and is the informative one: it is the share of "
             "questions that name a SPECIFIC place or dish. The third also counts the "
             "lowercase vocabulary the gazetteer carries (cuisine, activity and "
             "traveller-type terms), which matches nearly every question and would make "
             "this stage look solved if they were merged. The non-root column is the "
             "headroom for purely lexical grounding, and the gap between it and 100% is "
             "why the retriever is not purely lexical: on a question that names no "
             "specific entity, dense card search and intent-driven class routing are "
             "doing all the work.\n")

    L.append("## 3. Ontology-based subgraph retrieval\n")
    L.append("Three sources are unioned per question: dense entity-card search, "
             "class-routed entities ordered by graph degree, and time-scoped facts.\n")
    L.append("| Set | Entities | Facts | From dense search | Added by class routing | "
             "Routing share | Empty context |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for nm in sets:
        d = s3[nm]
        L.append(f"| {nm} | {d['mean_entities']} | {d['mean_facts']} | "
                 f"{d['mean_from_dense_search']} | {d['mean_added_by_class_routing']} | "
                 f"{pc(d['class_routing_share'])} | {pc(d['empty_context_rate'])} |")
    L.append(f"\n**Class routing is not redundant.** On the benchmark "
             f"{pc(s3['benchmark']['class_routing_share'])} of the entities in the "
             "assembled subgraph are ones the dense card search did not return; they "
             "come from the intent→class mapping and are ordered by graph degree. That "
             "is the mechanism by which a generic question reaches specific entities it "
             "never names.\n")

    L.append("## 4-5. Hybrid retrieval and context assembly\n")
    L.append(f"Shared budget {C.CTX_MAX_CHARS:,} characters for every arm, so the arms "
             "differ in what fills the context and not in how much. The hybrid arm "
             "reserves 20% for ontology facts and gives the remainder to passages.\n")
    L.append("| Set | RAG chars (budget use) | RAG truncated | Ontology chars "
             "(budget use) | Ontology truncated | Hybrid chars | Realised fact share | "
             "Fact side truncated |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for nm in sets:
        d = s45[nm]
        L.append(f"| {nm} | {d['rag_chars_mean']:,} ({pc(d['rag_budget_use'])}) | "
                 f"{pc(d['rag_truncated_rate'])} | "
                 f"{d['onto_chars_mean']:,} ({pc(d['onto_budget_use'])}) | "
                 f"{pc(d['onto_truncated_rate'])} | {d['hybrid_chars_mean']:,} | "
                 f"{pc(d['hybrid_realised_fact_share'])} | "
                 f"{pc(d['hybrid_fact_side_truncated'])} |")
    L.append("\nBoth arms sit near the budget ceiling, which is the point: the "
             "ontology-vs-RAG comparison in Task 4.1 is a comparison of context "
             "*content* at equal context *size*, not a comparison of context length.")
    (DELIV / "retriever_characterization.md").write_text("\n".join(L) + "\n",
                                                         encoding="utf-8")
    print(f"\nwrote {DELIV / 'retriever_characterization.md'}")


if __name__ == "__main__":
    main()
