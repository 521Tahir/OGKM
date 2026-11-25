"""
Proposal H1 - "Ontologies generated using structured prompting techniques exhibit
higher logical consistency and hierarchical quality than those generated using
naive prompting."

The Task 2.1.2 prompt comparison measures throughput (concepts per record),
schema validity and the candidate-level validator verdict. None of those is
logical consistency or hierarchical quality, so H1 as written was not testable
from it. This script closes that: each strategy extracts from the SAME records
at greedy decoding, the per-strategy concept and relation sets are assembled
into an ontology, and the Task 2.2 validation layers are run over it.

  logical consistency   L1 structural errors and warnings, L2 logical findings
                        (type violations, symmetry breaches, cycles), each per
                        100 relations so strategies that extract more are not
                        rewarded or punished for volume
  hierarchical quality  inheritance richness, mean and max hierarchy depth,
                        isolated-concept fraction, taxonomic cycles
  redundancy            duplicate triples and self-loops per 100 relations

Same records, same decoding, same validator for every strategy, so the only
thing that differs is the prompt.

  python -m wbs.task2_extraction.hypothesis_h1 --gpu 6 --n-records 20
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task2_extraction"
STRATEGIES = ["naive", "cot", "graph_of_thought", "metacognitive",
              "repair_after_failure", "unified"]
# `unified` is not an extractor prompt. Proposal 6.8 asks for the strategies
# integrated into ONE multi-stage pipeline, which `OntologyBuilder` implements as
# cot extraction followed by two post-batch passes: the GoT structural coherence
# check and the metacognitive confidence filter. It is benchmarked here through
# the same harness as the single strategies, using the pipeline's own classes
# rather than a reimplementation, so the comparison is like for like.
UNIFIED_BASE = "cot"
STRUCTURED = {"cot", "graph_of_thought", "metacognitive", "repair_after_failure",
              "unified"}


def hierarchy_from(concepts: list[dict], relations: list[dict]) -> list[dict]:
    """Derive the hierarchy records the validator expects from is_a / subclass_of
    edges, rooted at the concept's declared class."""
    parent: dict[str, str] = {}
    for r in relations:
        if str(r.get("predicate", "")).lower() in ("is_a", "subclass_of", "instance_of"):
            s, o = str(r.get("subject", "")).strip(), str(r.get("object", "")).strip()
            if s and o and s.lower() != o.lower():
                parent.setdefault(s.lower(), o)
    out = []
    for c in concepts:
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        path, cur, guard = [], name, 0
        while cur and guard < 12:                 # guard: a cycle must not hang this
            path.append(cur)
            cur = parent.get(cur.lower())
            guard += 1
            if cur and cur.lower() in {p.lower() for p in path}:
                break
        out.append({"concept": name, "concept_class": c.get("concept_class"),
                    "hierarchy_path": list(reversed(path))})
    return out


def taxonomic_cycles(relations: list[dict]) -> int:
    """Count concepts that reach themselves through is_a edges."""
    nxt = defaultdict(list)
    for r in relations:
        if str(r.get("predicate", "")).lower() in ("is_a", "subclass_of", "instance_of"):
            nxt[str(r.get("subject", "")).lower()].append(str(r.get("object", "")).lower())
    cycles = 0
    for start in list(nxt):
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            if cur == start and seen:
                cycles += 1
                break
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(nxt.get(cur, []))
    return cycles


def run_unified_passes(client, concepts: list[dict], relations: list[dict]):
    """The two post-batch passes `OntologyBuilder` runs for `unified` (6.8).

    Both are the pipeline's own classes. The GoT check is heuristic when no LLM
    answer is available, which is what the builder relies on too, and the
    builder's action on its report is to drop self-loop relations. The
    metacognitive pass flags low-confidence concepts rather than deleting them,
    so it is recorded as a count here instead of silently shrinking the set.
    """
    from src.llm.got_prompter import GoTPrompter
    from src.llm.metacognitive_prompter import MetacognitivePrompter

    info = {"base_strategy": UNIFIED_BASE,
            "concepts_in": len(concepts), "relations_in": len(relations)}
    try:
        got = GoTPrompter(llm_client=client)
        rep = got.check_structural_coherence(concepts, relations)
        info["got_score"] = round(float(getattr(rep, "score", 0.0)), 4)
        info["got_redundant"] = len(getattr(rep, "redundant", []) or [])
        info["got_conflicting"] = len(getattr(rep, "conflicting", []) or [])
        before = len(relations)
        relations = [r for r in relations
                     if str(r.get("subject", "")).lower()
                     != str(r.get("object", "")).lower()]
        info["self_loops_removed"] = before - len(relations)
    except Exception as e:                      # the builder swallows this too
        info["got_error"] = f"{type(e).__name__}: {e}"
    try:
        mp = MetacognitivePrompter(llm_client=client, review_threshold=0.5)
        accepted, flagged = mp.flag_for_review(concepts, threshold=0.5)
        info["metacognitive_accepted"] = len(accepted)
        info["metacognitive_flagged"] = len(flagged)
        names = {str(f.get("name", "")).lower() for f in flagged}
        for c in concepts:
            if str(c.get("name", "")).lower() in names:
                c["validator_status"] = "warning"
    except Exception as e:
        info["metacognitive_error"] = f"{type(e).__name__}: {e}"
    info["concepts_out"] = len(concepts)
    info["relations_out"] = len(relations)
    return concepts, relations, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--n-records", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--sample", type=int, default=0,
                    help="disjoint record slice index; sample 1 uses the next "
                         "n-records of the same shuffled pool")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() \
            if (ROOT / ".env").exists() else []:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager
    from src.ontology.concept_extractor import ConceptExtractor
    from src.ontology.relation_extractor import RelationExtractor
    from src.ontology.schema import TourismOntologySchema
    from src.utils.seed import set_seed
    from wbs.task2_formalization.validate_ontology import (l1_structural, l2_logical,
                                                           quality_metrics)
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    recs = [json.loads(l) for l in
            open(ROOT / "data" / "processed" / "cleaned_records.jsonl", encoding="utf-8")
            if l.strip()]
    random.Random(a.seed).shuffle(recs)
    # The Task 2.1.2 sensitivity sweep found that WHICH records a strategy sees moves
    # its metrics more than the decoding draw does, so a verdict on H1 from one record
    # sample is not a verdict. --sample selects a disjoint slice; run it twice and the
    # report says whether the ranking survived.
    lo = a.sample * a.n_records
    recs = recs[lo: lo + a.n_records]
    print(f"{len(recs)} records (sample {a.sample}, records {lo}-{lo+a.n_records-1} of "
          f"the shuffled pool), {len(a.strategies)} strategies, greedy decoding",
          flush=True)

    client = LLMClient(backend="local", load_in_4bit=False, gpu_ids=[0], max_tokens=800)
    client.temperature = 0.0
    pm = PromptManager()
    schema = TourismOntologySchema()
    print("model ready", flush=True)

    tag = f"__{a.tag}" if a.tag else ""
    out_p = OUT / f"hypothesis_h1{tag}.json"
    res = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}

    for s in a.strategies:
        if s in res:
            print(f"[{s}] already measured, skipping", flush=True)
            continue
        print(f"\n=== {s} ===", flush=True)
        base = UNIFIED_BASE if s == "unified" else s
        ce = ConceptExtractor(client, pm, strategy=base)
        re_ = RelationExtractor(client, pm, strategy=base)
        concepts, relations, fails = [], [], 0
        t0 = time.perf_counter()
        for i, rec in enumerate(recs, 1):
            set_seed(a.seed)
            text = f"{rec['Input']}\n{rec['Output']}"
            try:
                cs = ce.extract(text, rec["record_id"])
                cd = [{"name": c.name, "concept_class": c.concept_class} for c in cs]
                rs = re_.extract(text, cd, rec["record_id"])
                concepts += [{"name": c.name, "concept_class": c.concept_class,
                              "aliases": list(getattr(c, "aliases", []) or []),
                              "attributes": dict(getattr(c, "attributes", {}) or {}),
                              "provenance": {"confidence": c.provenance.confidence}}
                             for c in cs]
                # OntologyRelation names the third field `obj`; `to_dict()` is the
                # only place that spells it "object", so go through it
                relations += [{"subject": r.subject, "predicate": r.predicate,
                               "object": r.obj,
                               "provenance": {"confidence": r.provenance.confidence}}
                              for r in rs]
            except Exception as e:
                fails += 1
                print(f"  record {rec['record_id']} failed: "
                      f"{type(e).__name__}: {e}", flush=True)
            if i % 5 == 0:
                print(f"  {i}/{len(recs)} ({(time.perf_counter()-t0)/i:.1f} s/rec)", flush=True)

        pipeline = None
        if s == "unified":
            concepts, relations, pipeline = run_unified_passes(client, concepts,
                                                               relations)
        hier = hierarchy_from(concepts, relations)
        f1 = l1_structural(concepts, relations)
        f2 = l2_logical(concepts, relations, hier)
        qm = quality_metrics(concepts, relations, hier)
        # An empty relation set must not be dressed up as a rate: dividing by
        # max(len, 1) is what let a broken run report plausible numbers once.
        per100 = (lambda n: round(100 * n / len(relations), 2)) if relations \
            else (lambda n: None)
        if not relations:
            print(f"  !! {s}: 0 relations extracted - every per-100 rate is "
                  f"undefined and is written as null", flush=True)
        sev1 = Counter(f["severity"] for f in f1)
        sev2 = Counter(f["severity"] for f in f2)
        code1 = Counter(f["code"] for f in f1)
        res[s] = {
            "structured": s in STRUCTURED,
            "pipeline": pipeline,
            "sample": a.sample,
            "n_records": len(recs), "parse_failures": fails,
            "concepts": len(concepts), "relations": len(relations),
            "distinct_concepts": len({c["name"].lower() for c in concepts}),
            # logical consistency, volume-normalised
            "l1_errors_per_100rel": per100(sev1.get("error", 0)),
            "l1_warnings_per_100rel": per100(sev1.get("warning", 0)),
            "l2_findings_per_100rel": per100(len(f2)),
            "l2_errors_per_100rel": per100(sev2.get("error", 0)),
            "self_loops_per_100rel": per100(code1.get("SELF_LOOP", 0)),
            "duplicate_triples_per_100rel": per100(code1.get("DUPLICATE_TRIPLE", 0)),
            "dangling_endpoints_per_100rel": per100(code1.get("DANGLING_ENDPOINT", 0)),
            "untyped_concepts_frac": rate(code1.get("CONCEPT_UNTYPED", 0),
                                          concepts),
            "taxonomic_cycles": taxonomic_cycles(relations),
            # hierarchical quality
            "inheritance_richness": qm["inheritance_richness"],
            "mean_hierarchy_depth": qm["mean_hierarchy_depth"],
            "max_hierarchy_depth": qm["max_hierarchy_depth"],
            "isolated_fraction": qm["isolated_fraction"],
            "classes": qm["classes"], "distinct_predicates": qm["distinct_predicates"],
            "relationship_richness": qm["relationship_richness"],
            "graph_density": qm["graph_density"], "average_degree": qm["average_degree"],
            "sec_per_record": round((time.perf_counter() - t0) / len(recs), 1),
        }
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        r = res[s]
        print(f"  {len(concepts)} concepts / {len(relations)} relations | "
              f"L1 err {r['l1_errors_per_100rel']}/100 | L2 {r['l2_findings_per_100rel']}/100 "
              f"| depth {r['mean_hierarchy_depth']} | cycles {r['taxonomic_cycles']}",
              flush=True)

    print(f"\nwrote {out_p.name}")


if __name__ == "__main__":
    main()
