"""
WBS Task 2.1.2 - prompt-strategy sensitivity and stability analysis.

`artifacts/reports/prompt_comparison_llm.md` ranks five extraction strategies on a
single 40-record greedy run. That gives a point estimate per strategy and no error bar,
so it cannot say whether the ranking is real or sample noise. The WBS line item asks
for "sensitivity and stability analysis"; this script supplies the two missing axes.

  STABILITY   same records, temperature 0.7, R seeds. Two questions: how much do the
              metrics move between identical re-runs, and do the re-runs extract the
              same concepts? The second is measured as the mean pairwise Jaccard of
              per-record concept-name sets. A strategy whose mean score is high but
              whose output changes every run is not usable for ontology construction.

  SENSITIVITY K disjoint record samples, greedy decoding. How much of a strategy's
              score is the strategy and how much is which records it happened to see?
              Reported as between-sample standard deviation, and as whether the
              ranking survives resampling (per-sample ranks + Kendall's tau).

Decoding differs from the original report on purpose: bf16 rather than 4-bit, because
30 runs at 4-bit was not affordable. Absolute numbers are therefore not comparable with
`prompt_comparison_llm.md`; the variances and the rank agreement, which are what this
script is for, are internally consistent.

  python -m wbs.task2_extraction.prompt_sensitivity --gpu 5 \
      --strategies naive cot graph_of_thought --n-records 20
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import itertools
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task2_extraction"
STRATEGIES = ["naive", "cot", "graph_of_thought", "metacognitive", "repair_after_failure"]
METRICS = ("concepts_per_record", "relations_per_record", "schema_validity_rate",
           "validation_pass_rate", "parse_success_rate", "mean_confidence")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def run_once(records, strategy, client, pm, validator, schema, temperature, seed):
    """One extraction pass. Returns metrics plus the per-record concept-name sets."""
    from src.ontology.concept_extractor import ConceptExtractor
    from src.ontology.relation_extractor import RelationExtractor
    from src.ontology.schema import OntologyCandidate
    from src.utils.seed import set_seed

    # The extractors call client.call() without a temperature, so LLMClient falls back
    # to its own attribute. Set it here or every "sampled" run decodes greedily and the
    # stability numbers come out perfect by construction.
    client.temperature = temperature
    if getattr(client, "_client", None) is not None:
        client._client.temperature = temperature

    ce = ConceptExtractor(client, pm, strategy=strategy)
    re_ = RelationExtractor(client, pm, strategy=strategy)
    nc, nr, ok, parsed, conf, valid = [], [], [], [], [], []
    names: list[set] = []
    t0 = time.perf_counter()

    for rec in records:
        set_seed(seed)                      # makes a sampled run reproducible
        text = f"{rec['Input']}\n{rec['Output']}"
        try:
            cs = ce.extract(text, rec["record_id"])
            cd = [{"name": c.name, "concept_class": c.concept_class} for c in cs]
            rs = re_.extract(text, cd, rec["record_id"])
            res = validator.validate_candidate(
                OntologyCandidate(rec["record_id"], cs, rs))
            nc.append(len(cs)); nr.append(len(rs))
            ok.append(1 if res.status == "pass" else 0); parsed.append(1)
            names.append({c.name.strip().lower() for c in cs})
            if cs:
                conf.append(st.mean(c.provenance.confidence for c in cs))
                valid.append(sum(schema.is_valid_class(c.concept_class)
                                 for c in cs) / len(cs))
        except Exception as e:                        # a parse failure is a data point
            print(f"      record {rec['record_id']} failed: {type(e).__name__}", flush=True)
            nc.append(0); nr.append(0); ok.append(0); parsed.append(0); names.append(set())

    el = time.perf_counter() - t0
    return {"concepts_per_record": st.mean(nc), "relations_per_record": st.mean(nr),
            "schema_validity_rate": st.mean(valid) if valid else 0.0,
            "validation_pass_rate": st.mean(ok), "parse_success_rate": st.mean(parsed),
            "mean_confidence": st.mean(conf) if conf else 0.0,
            "sec_per_record": rate(el, records, 6)}, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--n-records", type=int, default=20)
    ap.add_argument("--stability-runs", type=int, default=3)
    ap.add_argument("--sensitivity-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--tag", default=None, help="suffix for the output file (GPU shard)")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HOME", "/ceph_data/tahirahmad/hf_home")
    OUT.mkdir(parents=True, exist_ok=True)

    # .env carries HF_TOKEN; the extraction client requires it
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    import random
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager
    from src.ontology.schema import TourismOntologySchema
    from src.ontology.validator import OntologyValidator
    from src.utils.io import load_jsonl
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    recs = load_jsonl(ROOT / "data" / "processed" / "cleaned_records.jsonl")
    rng = random.Random(a.seed)
    pool = rng.sample(range(len(recs)), a.n_records * (a.sensitivity_samples + 1))
    for i, idx in enumerate(pool):
        recs[idx].setdefault("record_id", idx)
    # sample 0 is the fixed set the stability runs re-use; 1..K are the disjoint ones
    samples = [[recs[i] for i in pool[k * a.n_records:(k + 1) * a.n_records]]
               for k in range(a.sensitivity_samples + 1)]

    client = LLMClient(backend="local", model_key=os.environ.get("LOCAL_MODEL_KEY",
                                                                 "llama3.1-8b"),
                       load_in_4bit=False, gpu_ids=[0],
                       hf_token=os.environ.get("HF_TOKEN"), max_tokens=800)
    if not client.is_available:
        sys.exit("LLM client unavailable; check HF_TOKEN in .env")
    pm = PromptManager(prompts_root=ROOT / "prompts")
    validator = OntologyValidator()
    print(f"model ready on GPU {a.gpu}; {len(a.strategies)} strategies, "
          f"{a.n_records} records/run, "
          f"{a.stability_runs} stability + {a.sensitivity_samples} sensitivity runs each",
          flush=True)

    out = {"gpu": a.gpu, "n_records": a.n_records, "seed": a.seed,
           "temperature": a.temperature, "stability_runs": a.stability_runs,
           "sensitivity_samples": a.sensitivity_samples,
           "decoding": "bf16, sampling for stability / greedy for sensitivity",
           "strategies": {}}

    for s in a.strategies:
        print(f"\n=== {s} ===", flush=True)
        # ── stability: same records, same prompt, different seeds, sampled ──
        runs, name_sets = [], []
        for r in range(a.stability_runs):
            m, names = run_once(samples[0], s, client, pm, validator,
                                TourismOntologySchema, a.temperature, a.seed + r)
            runs.append(m); name_sets.append(names)
            print(f"  stability run {r+1}/{a.stability_runs}: "
                  f"concepts/rec {m['concepts_per_record']:.2f} "
                  f"parse {m['parse_success_rate']:.2f} "
                  + (f"({m['sec_per_record']:.1f} s/rec)"
                     if m['sec_per_record'] is not None else "(no records)"),
                  flush=True)
        # per-record agreement between every pair of runs
        pair_j = []
        for i, j in itertools.combinations(range(len(name_sets)), 2):
            pair_j.append(st.mean(jaccard(x, y)
                                  for x, y in zip(name_sets[i], name_sets[j])))
        # ── sensitivity: disjoint record samples, greedy ──
        samp = []
        for k in range(1, a.sensitivity_samples + 1):
            m, _ = run_once(samples[k], s, client, pm, validator,
                            TourismOntologySchema, 0.0, a.seed)
            samp.append(m)
            print(f"  sensitivity sample {k}/{a.sensitivity_samples}: "
                  f"concepts/rec {m['concepts_per_record']:.2f} "
                  f"parse {m['parse_success_rate']:.2f}", flush=True)

        def agg(rows):
            return {k: {"mean": st.mean(r[k] for r in rows),
                        "sd": st.stdev([r[k] for r in rows]) if len(rows) > 1 else 0.0,
                        "min": min(r[k] for r in rows), "max": max(r[k] for r in rows)}
                    for k in METRICS}

        out["strategies"][s] = {
            "stability": {"runs": runs, "agg": agg(runs),
                          "concept_set_jaccard_mean":
                              st.mean(pair_j) if pair_j else None,
                          "concept_set_jaccard_pairs": pair_j},
            "sensitivity": {"runs": samp, "agg": agg(samp)},
            "sec_per_record": (st.mean(_spr) if (_spr := [
                r["sec_per_record"] for r in runs + samp
                if r["sec_per_record"] is not None]) else None),
        }
        tag = f"__{a.tag}" if a.tag else ""
        (OUT / f"prompt_sensitivity{tag}.json").write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"  [checkpointed {s}]", flush=True)

    print("\ndone")


if __name__ == "__main__":
    main()
