"""
WBS Task 4.1.2 / 4.2.2 - the ontology-only arm across eight models, re-measured.

The companion project at /home/tahirahmad/Ontology_Project_Codex ran the
ontology-only and Simple RAG arms over the full 4,607-question held-out test set
for eight models in four families, which is far wider than the three Llama
models this project measured on its own 347-item benchmark. Those runs are the
best ontology-arm results on record, so this module brings them in. It does two
things, and keeps them apart:

  HARVEST       reads the aggregate score files the companion project wrote and
                records, for every arm, the path, the sha256 of the generation
                file it scored and - the part that decides whether any of it
                means anything - how many training passes each arm received.

  RE-MEASURE    scores the stored answers again, from the raw generations, with
                THIS project's metric stack: token P/R/F1 (the published primary
                metric), entity P/R/F1 over the Jeju gazetteer, and the
                groundedness audit against ontology v1.4.0. The companion
                project never computed groundedness for these models, and it
                reported every multi-pass result as a bare mean with no paired
                test. Both gaps are closed here.

Why the training budget is the headline and not a footnote. The companion
project's own train_summary.json files record it: the ontology arm was trained
for one, three or four passes over the same 36,847 records while the Simple RAG
baseline was trained once, and for the four 8B models the RAG arm was not
trained at all. A comparison between a four-pass arm and a one-pass arm measures
the passes as much as the context. Every pair here is therefore labelled:

  equal         1 ontology pass vs 1 RAG pass. The only tier that isolates the
                context format. This is the comparison the project's standing
                claim has to rest on.
  extended      3-4 ontology passes vs 1 RAG pass. Informative about what the
                ontology context can reach, not about equal-budget parity.
  untuned_rag   1 ontology pass vs an untrained base model. Reported because the
                companion project published it; it says almost nothing about the
                ontology and the verdict column says so.

Statistics: per-item paired Wilcoxon signed-rank, two-sided, over items where
both arms produced a countable value, Holm-corrected within each pair across the
metric family; plus a seeded 2,000-sample bootstrap interval on the mean paired
delta. A delta whose interval covers zero is reported as indistinguishable, not
as a win.

  python -m wbs.task4_evaluation.cross_model_results            # harvest + re-measure
  python -m wbs.task4_evaluation.cross_model_results --harvest-only
  python -m wbs.task4_evaluation.cross_model_results --pair qwen2.5-3b@p4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CODEX = Path("/home/tahirahmad/Ontology_Project_Codex")
EXPS = CODEX / "experiments"
OUT = ROOT / "wbs" / "deliverables" / "task4_evaluation"

E41 = EXPS / "exp041_small_models" / "outputs"
E53 = EXPS / "exp053_staged_1b" / "outputs"
E54 = EXPS / "exp054_fourth_pass_1b_eval" / "outputs"
E55 = EXPS / "exp055_3b_ontology" / "outputs"
E56 = EXPS / "exp056_qwen" / "outputs"
E58 = EXPS / "exp058_full_metric_table" / "outputs"
# This project's own matched-budget RAG runs (exp060). The only arm in this
# module that is not the companion project's.
E60 = ROOT / "experiments" / "exp060_matched_budget_rag" / "outputs"

# Metric family re-measured here. "lower" marks a metric where less is better.
METRICS = [
    ("token_f1", "Token F1", False),
    ("token_p", "Token precision", False),
    ("token_r", "Token recall", False),
    ("entity_f1", "Entity F1", False),
    ("entity_hit", "Entity hit rate", False),
    ("ungrounded_rate", "Ungrounded rate", True),
    ("clean", "Fully grounded answers", False),
]
BOOT = 2000
SEED = 20261026
TIERS = ("equal", "extended", "matched", "untuned_rag")

# Every arm pair that is legitimately paired: same model, same split, same
# question ids, answers produced under the same prompt and decoding settings.
# `passes` is read back from the companion project's own training receipts and
# cross-checked in verify_budgets().
REGISTRY = [
    # ---- equal budget: one ontology pass against one RAG pass ----
    dict(key="llama3.2-1b@p1", model="llama3.2-1b", family="Llama 3.2", params_b=1.2,
         tier="equal", onto_passes=1, rag_passes=1,
         onto=E41 / "final/generations/test__llama3.2-1b__onto_graph.jsonl",
         rag=E41 / "generations_v2/test__llama3.2-1b__rag.jsonl",
         onto_exp="exp041_small_models", rag_exp="exp041_small_models",
         reported=[E41 / "final/test_1b_primary_comparison.json"]),
    dict(key="llama3.2-3b@p1", model="llama3.2-3b", family="Llama 3.2", params_b=3.2,
         tier="equal", onto_passes=1, rag_passes=1,
         onto=E41 / "final/generations/test__llama3.2-3b__onto_graph.jsonl",
         rag=E41 / "generations_v2/test__llama3.2-3b__rag.jsonl",
         onto_exp="exp041_small_models", rag_exp="exp041_small_models",
         reported=[E41 / "final/test_3b_primary_comparison.json"]),
    dict(key="qwen2.5-1.5b@p1", model="qwen2.5-1.5b", family="Qwen 2.5", params_b=1.5,
         tier="equal", onto_passes=1, rag_passes=1,
         onto=E56 / "generations/test__qwen2.5-1.5b__onto_graph.jsonl",
         rag=E56 / "generations/test__qwen2.5-1.5b__rag.jsonl",
         onto_exp="exp056_qwen", rag_exp="exp056_qwen",
         reported=[E56 / "cpu_qwen2.5-1.5b.json"]),
    dict(key="qwen2.5-3b@p1", model="qwen2.5-3b", family="Qwen 2.5", params_b=3.1,
         tier="equal", onto_passes=1, rag_passes=1,
         onto=E56 / "generations/test__qwen2.5-3b__onto_graph.jsonl",
         rag=E56 / "generations/test__qwen2.5-3b__rag.jsonl",
         onto_exp="exp056_qwen", rag_exp="exp056_qwen",
         reported=[E56 / "cpu_qwen2.5-3b.json"]),
    # ---- extended ontology training against the same one-pass RAG baseline ----
    dict(key="llama3.2-1b@p4", model="llama3.2-1b", family="Llama 3.2", params_b=1.2,
         tier="extended", onto_passes=4, rag_passes=1,
         onto=E53 / "generations/test__llama3.2-1b__onto_graph.jsonl",
         rag=E41 / "generations_v2/test__llama3.2-1b__rag.jsonl",
         onto_exp="exp053_staged_1b (exp046 fourth-pass adapter)", rag_exp="exp041_small_models",
         reported=[E54 / "cpu_metrics_1b.json", E54 / "bertf1_1b.json",
                   E54 / "entity_test_1b.json"]),
    dict(key="llama3.2-3b@p3", model="llama3.2-3b", family="Llama 3.2", params_b=3.2,
         tier="extended", onto_passes=3, rag_passes=1,
         onto=E55 / "generations/test__llama3.2-3b__onto_graph.jsonl",
         rag=E41 / "generations_v2/test__llama3.2-3b__rag.jsonl",
         onto_exp="exp055_3b_ontology (exp044 third-pass adapter)", rag_exp="exp041_small_models",
         reported=[E55 / "cpu_3b.json", E55 / "bertf1_3b.json", E55 / "comet_3b.json",
                   E55 / "judge_3b.json", E55 / "entity_3b.json"]),
    dict(key="qwen2.5-1.5b@p3", model="qwen2.5-1.5b", family="Qwen 2.5", params_b=1.5,
         tier="extended", onto_passes=3, rag_passes=1,
         onto=E56 / "generations_pass3/test__qwen2.5-1.5b__onto_graph.jsonl",
         rag=E56 / "generations/test__qwen2.5-1.5b__rag.jsonl",
         onto_exp="exp056_qwen pass 3", rag_exp="exp056_qwen",
         reported=[E56 / "pass3_scores/cpu_qwen2.5-1.5b.json",
                   E56 / "pass3_scores/bertf1_qwen2.5-1.5b.json",
                   E56 / "pass3_scores/comet_qwen2.5-1.5b.json",
                   E56 / "pass3_scores/judge_qwen2.5-1.5b.json",
                   E56 / "pass3_scores/entity_qwen2.5-1.5b.json"]),
    dict(key="qwen2.5-3b@p4", model="qwen2.5-3b", family="Qwen 2.5", params_b=3.1,
         tier="extended", onto_passes=4, rag_passes=1,
         onto=E56 / "generations_pass4/test__qwen2.5-3b__onto_graph.jsonl",
         rag=E56 / "generations/test__qwen2.5-3b__rag.jsonl",
         onto_exp="exp056_qwen pass 4", rag_exp="exp056_qwen",
         reported=[E56 / "pass4_scores/cpu_qwen2.5-3b.json",
                   E56 / "pass4_scores/bertf1_qwen2.5-3b.json",
                   E56 / "pass4_scores/comet_qwen2.5-3b.json",
                   E56 / "pass4_scores/judge_qwen2.5-3b.json",
                   E56 / "pass4_scores/entity_qwen2.5-3b.json"]),
    # ---- extended ontology training against a RAG arm trained to match ----
    # exp060 closes the budget confound in the tier above: the same ontology
    # answers, against a Simple RAG arm continued to the same total pass count.
    # These rows appear once exp060 has generated them and are reported as
    # pending until then, rather than quietly left out.
    dict(key="llama3.2-1b@p4m", model="llama3.2-1b", family="Llama 3.2", params_b=1.2,
         tier="matched", onto_passes=4, rag_passes=4,
         onto=E53 / "generations/test__llama3.2-1b__onto_graph.jsonl",
         rag=E60 / "generations/test__llama3.2-1b__rag_pass4.jsonl",
         onto_exp="exp053_staged_1b (exp046 fourth-pass adapter)",
         rag_exp="exp060_matched_budget_rag (this project)",
         reported=[E60 / "adapters/llama3.2-1b__rag_pass4/train_summary.json"]),
    dict(key="llama3.2-3b@p3m", model="llama3.2-3b", family="Llama 3.2", params_b=3.2,
         tier="matched", onto_passes=3, rag_passes=3,
         onto=E55 / "generations/test__llama3.2-3b__onto_graph.jsonl",
         rag=E60 / "generations/test__llama3.2-3b__rag_pass3.jsonl",
         onto_exp="exp055_3b_ontology (exp044 third-pass adapter)",
         rag_exp="exp060_matched_budget_rag (this project)",
         reported=[E60 / "adapters/llama3.2-3b__rag_pass3/train_summary.json"]),
    dict(key="qwen2.5-1.5b@p3m", model="qwen2.5-1.5b", family="Qwen 2.5", params_b=1.5,
         tier="matched", onto_passes=3, rag_passes=3,
         onto=E56 / "generations_pass3/test__qwen2.5-1.5b__onto_graph.jsonl",
         rag=E60 / "generations/test__qwen2.5-1.5b__rag_pass3.jsonl",
         onto_exp="exp056_qwen pass 3",
         rag_exp="exp060_matched_budget_rag (this project)",
         reported=[E60 / "adapters/qwen2.5-1.5b__rag_pass3/train_summary.json"]),
    dict(key="qwen2.5-3b@p4m", model="qwen2.5-3b", family="Qwen 2.5", params_b=3.1,
         tier="matched", onto_passes=4, rag_passes=4,
         onto=E56 / "generations_pass4/test__qwen2.5-3b__onto_graph.jsonl",
         rag=E60 / "generations/test__qwen2.5-3b__rag_pass4.jsonl",
         onto_exp="exp056_qwen pass 4",
         rag_exp="exp060_matched_budget_rag (this project)",
         reported=[E60 / "adapters/qwen2.5-3b__rag_pass4/train_summary.json"]),
    # ---- ontology adapter against an untrained base model ----
    dict(key="llama3.1-8b@p1", model="llama3.1-8b", family="Llama 3.1", params_b=8.0,
         tier="untuned_rag", onto_passes=1, rag_passes=0,
         onto=E58 / "generations/test__llama3.1-8b__onto_graph.jsonl",
         rag=E58 / "generations/test__llama3.1-8b__rag.jsonl",
         onto_exp="exp058_full_metric_table", rag_exp="exp058_full_metric_table",
         reported=[E58 / "scores/llama3.1-8b.json"]),
    dict(key="qwen3-8b@p1", model="qwen3-8b", family="Qwen 3", params_b=8.2,
         tier="untuned_rag", onto_passes=1, rag_passes=0,
         onto=E58 / "generations/test__qwen3-8b__onto_graph.jsonl",
         rag=E58 / "generations/test__qwen3-8b__rag.jsonl",
         onto_exp="exp058_full_metric_table", rag_exp="exp058_full_metric_table",
         reported=[E58 / "scores/qwen3-8b.json"]),
    dict(key="mistral-7b@p1", model="mistral-7b", family="Mistral", params_b=7.2,
         tier="untuned_rag", onto_passes=1, rag_passes=0,
         onto=E58 / "generations/test__mistral-7b__onto_graph.jsonl",
         rag=E58 / "generations/test__mistral-7b__rag.jsonl",
         onto_exp="exp058_full_metric_table", rag_exp="exp058_full_metric_table",
         reported=[E58 / "scores/mistral-7b.json"]),
    dict(key="deepseek-r1-8b@p1", model="deepseek-r1-8b", family="DeepSeek-R1", params_b=8.0,
         tier="untuned_rag", onto_passes=1, rag_passes=0,
         onto=E58 / "generations/test__deepseek-r1-8b__onto_graph.jsonl",
         rag=E58 / "generations/test__deepseek-r1-8b__rag.jsonl",
         onto_exp="exp058_full_metric_table", rag_exp="exp058_full_metric_table",
         reported=[E58 / "scores/deepseek-r1-8b.json"]),
]

# Training receipts, so the pass counts above are read off the companion
# project's own records instead of being asserted here.
RECEIPTS = {
    "llama3.2-1b@p1": [E41 / "adapters_v2/llama3.2-1b__onto_graph/train_summary.json",
                       E41 / "adapters_v2/llama3.2-1b__rag/train_summary.json"],
    "llama3.2-3b@p1": [E41 / "adapters_v2/llama3.2-3b__onto_graph/train_summary.json",
                       E41 / "adapters_v2/llama3.2-3b__rag/train_summary.json"],
    "qwen2.5-1.5b@p1": [E56 / "adapters/qwen2.5-1.5b__onto_graph/train_summary.json",
                        E56 / "adapters/qwen2.5-1.5b__rag/train_summary.json"],
    "qwen2.5-3b@p1": [E56 / "adapters/qwen2.5-3b__onto_graph/train_summary.json",
                      E56 / "adapters/qwen2.5-3b__rag/train_summary.json"],
    "llama3.2-1b@p4": [EXPS / "exp046_ontology_fourth_pass/outputs/schedule_aligned/llama3.2-1b/"
                       "adapters/llama3.2-1b__onto_graph/train_summary.json"],
    "llama3.2-3b@p3": [EXPS / "exp044_ontology_third_pass/outputs/schedule_aligned/llama3.2-3b/"
                       "adapters/llama3.2-3b__onto_graph/train_summary.json"],
    "qwen2.5-1.5b@p3": [E56 / "adapters/qwen2.5-1.5b__onto_graph_pass3/train_summary.json"],
    "qwen2.5-3b@p4": [E56 / "adapters/qwen2.5-3b__onto_graph_pass4/train_summary.json"],
}

# The matched tier's RAG receipts are written by exp060 in this repository. They
# do not exist until that model has finished training, so they verify as
# `pending` rather than as a disagreement.
MATCHED_RECEIPTS = {
    "llama3.2-1b@p4m": E60 / "adapters/llama3.2-1b__rag_pass4/train_summary.json",
    "llama3.2-3b@p3m": E60 / "adapters/llama3.2-3b__rag_pass3/train_summary.json",
    "qwen2.5-1.5b@p3m": E60 / "adapters/qwen2.5-1.5b__rag_pass3/train_summary.json",
    "qwen2.5-3b@p4m": E60 / "adapters/qwen2.5-3b__rag_pass4/train_summary.json",
}

TIER_NOTE = {
    "equal": "One ontology pass against one RAG pass over the same 36,847 training "
             "records, same LoRA recipe, same prompt and decoding. The context "
             "format is the only thing that differs.",
    "extended": "The ontology arm received three or four full training passes; the "
                "Simple RAG baseline received one and was not retrained. A win here "
                "is a win for ontology context plus extra training, and the two "
                "cannot be separated from these runs.",
    "matched": "The same ontology arms as `extended`, against a Simple RAG arm "
               "continued to the same total pass count by exp060 of this project. "
               "Same training records, same RAG contexts, same LoRA recipe, learning "
               "rate, scheduler, sequence budget and seed - only the epoch count was "
               "raised, because only the epoch count was the confound. This is the "
               "tier that says whether the extended result was the ontology or the "
               "training.",
    "untuned_rag": "The ontology arm carries a trained adapter and the Simple RAG arm "
                   "is the untouched base model. This measures fine-tuning, not the "
                   "ontology, and is reported only because the companion project "
                   "published it.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> dict:
    """record_id -> row. Later lines win: two generation files append a re-run."""
    rows = {}
    dupes = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["record_id"] in rows:
                dupes += 1
            rows[row["record_id"]] = row
    return rows, dupes


def score_arm(path_str: str) -> dict:
    """Per-item scores for one stored arm, using this project's metric stack."""
    from experiments.exp042_prf.compute_prf import ents, prf, toks
    from wbs.rates import rate
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    from wbs.task4_evaluation.hallucination_report import audit_spans

    kb = KnowledgeBase()
    rows, dupes = load_rows(Path(path_str))
    per = {}
    for rid, row in rows.items():
        gold = row.get("reference") or ""
        pred = row.get("answer") or ""
        t = prf(toks(pred), toks(gold))
        ge, pe = ents(gold), ents(pred)
        e = prf(pe, ge)
        spans, ungrounded, _generic = audit_spans(pred, kb)
        per[rid] = {
            "token_p": t[0] if t else None,
            "token_r": t[1] if t else None,
            "token_f1": t[2] if t else None,
            "entity_p": e[0] if e else None,
            "entity_r": e[1] if e else None,
            "entity_f1": e[2] if e else None,
            "entity_hit": (1.0 if sum((pe & ge).values()) > 0 else 0.0) if e else None,
            # Same definition as score_variants.score_item: a rate over no
            # entity-like span is undefined, not zero (R15).
            "ungrounded_rate": rate(len(ungrounded), spans),
            "clean": (1.0 if spans and not ungrounded else (0.0 if spans else None)),
            "n_spans": float(len(spans)),
            "answer_chars": float(len(pred)),
        }
    return {"path": path_str, "sha256": sha256(Path(path_str)),
            "n_rows": len(rows), "duplicate_lines": dupes, "per_item": per}


def wilcoxon(deltas: list[float]):
    d = [x for x in deltas if x != 0.0]
    if len(d) < 6:
        return None
    try:
        from scipy.stats import wilcoxon as w
        r = w(d, alternative="two-sided", zero_method="wilcox")
        return float(r.pvalue), len(d)
    except Exception:
        return None


def bootstrap_ci(deltas: list[float], seed: int):
    if len(deltas) < 20:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(BOOT):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def holm(pairs: list[tuple[str, float]]) -> dict:
    """Holm-Bonferroni within one pair's metric family."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        out[name] = running
    return out


def compare(spec: dict, onto: dict, rag: dict) -> dict:
    ids = sorted(set(onto["per_item"]) & set(rag["per_item"]))
    row = {
        "key": spec["key"], "model": spec["model"], "family": spec["family"],
        "params_b": spec["params_b"], "tier": spec["tier"],
        "onto_passes": spec["onto_passes"], "rag_passes": spec["rag_passes"],
        "split": "test", "paired_n": len(ids),
        "onto_only_ids": len(set(onto["per_item"]) - set(rag["per_item"])),
        "rag_only_ids": len(set(rag["per_item"]) - set(onto["per_item"])),
        "sources": {
            "ontology": {"experiment": spec["onto_exp"],
                         "project": rel(onto["path"])[0],
                         "path": rel(onto["path"])[1],
                         "sha256": onto["sha256"], "rows": onto["n_rows"],
                         "duplicate_lines": onto["duplicate_lines"]},
            "simple_rag": {"experiment": spec["rag_exp"],
                           "project": rel(rag["path"])[0],
                           "path": rel(rag["path"])[1],
                           "sha256": rag["sha256"], "rows": rag["n_rows"],
                           "duplicate_lines": rag["duplicate_lines"]},
        },
        "metrics": {},
        "declared_sha_check": check_declared_sha(spec, onto, rag),
    }
    raw_p = []
    for name, _label, lower_is_better in METRICS:
        a = [onto["per_item"][i][name] for i in ids]
        b = [rag["per_item"][i][name] for i in ids]
        kept = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if not kept:
            # R15: a rate over an empty set is undefined, never zero.
            row["metrics"][name] = {"countable_n": 0, "ontology": None,
                                    "simple_rag": None, "delta": None,
                                    "p_raw": None, "ci95": None,
                                    "verdict": "not countable on any item"}
            continue
        ao = [x for x, _ in kept]
        bo = [y for _, y in kept]
        deltas = [x - y for x, y in kept]
        wil = wilcoxon(deltas)
        entry = {
            "countable_n": len(kept),
            "ontology": st.mean(ao), "simple_rag": st.mean(bo),
            "delta": st.mean(deltas),
            "lower_is_better": lower_is_better,
            "nonzero_pairs": wil[1] if wil else 0,
            "p_raw": wil[0] if wil else None,
            "ci95": bootstrap_ci(deltas, SEED + abs(hash(spec["key"] + name)) % 9973),
        }
        row["metrics"][name] = entry
        if wil:
            raw_p.append((name, wil[0]))

    adjusted = holm(raw_p) if raw_p else {}
    for name, entry in row["metrics"].items():
        entry["p_holm"] = adjusted.get(name)
        d, ci, p = entry.get("delta"), entry.get("ci95"), entry.get("p_holm")
        if d is None:
            continue
        better = (d < 0) if entry.get("lower_is_better") else (d > 0)
        sig = (p is not None and p < 0.05) and (ci is None or (ci[0] > 0) == (ci[1] > 0))
        if not sig:
            entry["verdict"] = "indistinguishable"
        else:
            entry["verdict"] = "ontology better" if better else "simple RAG better"
    return row


def _exp060_status(spec: dict) -> str:
    """Where exp060 has got to for this model, read from disk."""
    adapter = E60 / "adapters" / f"{spec['model']}__rag_pass{spec['rag_passes']}"
    gen = Path(spec["rag"])
    if gen.exists():
        n = sum(1 for line in gen.open(encoding="utf-8") if line.strip())
        return f"generating: {n:,} of 4,607 answers written"
    if (adapter / "adapter_config.json").exists():
        return "trained; generation not started"
    ckpts = sorted((adapter / "hf").glob("checkpoint-*")) if (adapter / "hf").exists() else []
    if ckpts:
        return f"training: last checkpoint {ckpts[-1].name}"
    return "training: not started or no checkpoint yet"


def rel(path) -> tuple:
    """Say which project holds a file, and name it relative to that project's root.

    Every arm used to live in the companion project, so a bare CODEX-relative path
    was enough. The matched-budget arms are trained here, so a path alone no longer
    says whether a file is read-only external evidence or this project's own output,
    and the reproducibility manifest has to tell those apart."""
    path = Path(path).resolve()
    for root, tag in ((CODEX, "codex"), (ROOT, "jejufull")):
        if root == path or root in path.parents:
            return tag, str(path.relative_to(root))
    return "elsewhere", str(path)

def harvest_reported(spec: dict) -> dict:
    """The companion project's own aggregate numbers, with provenance."""
    found = {}
    for path in spec["reported"]:
        project, name = rel(path)
        key = name if project == "codex" else f"{project}:{name}"
        if not path.exists():
            found[key] = {"status": "absent"}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test", payload)
        found[key] = {
            "status": "read", "sha256": sha256(path),
            "test_block": test,
        }
    return found


# The companion project's router writes a parallel `hybrid` file in which answers it
# could serve from the graph directly are replaced. How often it fires decides whether
# the hybrid arm is a separate system variant or the same answers under another name,
# so it is measured here rather than read off a "routed" field.
ROUTER_CHECK = [
    ("qwen2.5-3b", "test", E56 / "generations_pass4/test__qwen2.5-3b__onto_graph.jsonl",
     E56 / "pass4_scores/hybrid/test__qwen2.5-3b__onto_graph.jsonl"),
    ("qwen2.5-3b", "temporal1000",
     E56 / "generations_pass4/temporal1000__qwen2.5-3b__onto_graph.jsonl",
     E56 / "pass4_scores/hybrid/temporal1000__qwen2.5-3b__onto_graph.jsonl"),
    ("llama3.2-3b", "test", E55 / "generations/test__llama3.2-3b__onto_graph.jsonl",
     E55 / "hybrid/test__llama3.2-3b__onto_graph.jsonl"),
    ("llama3.2-3b", "temporal1000",
     E55 / "generations/temporal1000__llama3.2-3b__onto_graph.jsonl",
     E55 / "hybrid/temporal1000__llama3.2-3b__onto_graph.jsonl"),
    ("llama3.1-8b", "test", E58 / "generations/test__llama3.1-8b__onto_graph.jsonl",
     E58 / "hybrid/test__llama3.1-8b__onto_graph.jsonl"),
]


# The companion project's own primary-comparison files record the sha256 of the
# generation file each arm was scored from. Where such a record exists, the hash
# computed here must equal it - otherwise this table is scoring a different file
# from the one the published comparison used, and the two cannot be reconciled.
DECLARED_SHA = {
    "llama3.2-1b@p1": E41 / "final/test_1b_primary_comparison.json",
    "llama3.2-3b@p1": E41 / "final/test_3b_primary_comparison.json",
}


def check_declared_sha(spec: dict, onto: dict, rag: dict) -> dict:
    path = DECLARED_SHA.get(spec["key"])
    if path is None or not path.exists():
        return {"status": "no declaration on record"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Two schemas are in use. The 1B file puts the artifacts at `sources`; the 3B
    # file nests them under results/<split>/models/<model>/artifacts. Both carry
    # the same two keys, so read whichever is present rather than assuming one.
    src = payload.get("sources")
    if src is None:
        src = (((payload.get("results") or {}).get("test") or {})
               .get("models") or {}).get(spec["model"], {}).get("artifacts") or {}
    want = {"ontology": (src.get("onto_graph") or {}).get("generation_sha256"),
            "simple_rag": (src.get("rag") or {}).get("generation_sha256")}
    got = {"ontology": onto["sha256"], "simple_rag": rag["sha256"]}
    agree = {k: (want[k] is not None and want[k] == got[k]) for k in want}
    return {"status": "checked", "declared_in": str(path.relative_to(CODEX)),
            "declared": want, "computed": got, "agrees": agree,
            "all_agree": all(agree.values())}


def router_firing() -> list[dict]:
    out = []
    for model, split, gen, hyb in ROUTER_CHECK:
        if not gen.exists() or not hyb.exists():
            out.append({"model": model, "split": split, "status": "absent"})
            continue
        a, _ = load_rows(gen)
        b, _ = load_rows(hyb)
        ids = set(a) & set(b)
        fired = sum(1 for i in ids
                    if (a[i].get("answer") or "") != (b[i].get("answer") or ""))
        out.append({"model": model, "split": split, "n": len(ids), "fired": fired,
                    "fired_rate": rate_or_none(fired, len(ids)),
                    "hybrid_path": str(hyb.relative_to(CODEX))})
    return out


def rate_or_none(num: int, den: int):
    from wbs.rates import rate
    return rate(num, den)


def verify_budgets() -> dict:
    """Read the pass counts back out of the companion project's receipts."""
    out = {}
    for key, paths in RECEIPTS.items():
        spec = next(s for s in REGISTRY if s["key"] == key)
        claims = []
        for path in paths:
            if not path.exists():
                claims.append({"path": str(path.relative_to(CODEX)), "status": "absent"})
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            cont = d.get("continuation", {})
            claims.append({
                "path": str(path.relative_to(CODEX)),
                "arm": d.get("arm"),
                "epochs": (d.get("train_cfg") or {}).get("epochs"),
                "total_ontology_passes": d.get("total_ontology_passes")
                or cont.get("total_training_passes"),
                "rag_passes": d.get("rag_passes") or (1 if d.get("arm") == "rag" else None),
                "comparison_budget": cont.get("comparison_budget"),
                "n_examples": d.get("n_examples"),
            })
        declared = [c.get("total_ontology_passes") for c in claims
                    if c.get("total_ontology_passes")]
        if declared:
            # A continuation receipt states the total directly.
            agrees = spec["onto_passes"] in declared
        else:
            # A first-pass receipt states only its own epoch count; both arms must
            # declare exactly the budget the registry claims for them.
            by_arm = {c.get("arm"): c.get("epochs") for c in claims}
            agrees = (by_arm.get("onto_graph") == spec["onto_passes"]
                      and by_arm.get("rag") == spec["rag_passes"])
        out[key] = {"registry_onto_passes": spec["onto_passes"],
                    "registry_rag_passes": spec["rag_passes"],
                    "receipts": claims, "receipt_agrees": agrees}
    # The matched tier: this project's own RAG receipts, checked the same way.
    for key, path in MATCHED_RECEIPTS.items():
        spec = next(s for s in REGISTRY if s["key"] == key)
        if not path.exists():
            out[key] = {"registry_onto_passes": spec["onto_passes"],
                        "registry_rag_passes": spec["rag_passes"],
                        "receipts": [{"path": str(path), "status": "not written yet"}],
                        "receipt_agrees": None, "status": "pending"}
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        claim = {"path": str(path), "arm": d.get("arm"),
                 "initializer_passes": d.get("initializer_passes"),
                 "additional_epochs": d.get("additional_epochs"),
                 "total_rag_passes": d.get("total_rag_passes"),
                 "matches_ontology_passes": d.get("matches_ontology_passes"),
                 "n_examples": d.get("n_examples"),
                 "effective_batch": d.get("effective_batch")}
        agrees = (d.get("arm") == "rag"
                  and d.get("total_rag_passes") == spec["rag_passes"]
                  and d.get("matches_ontology_passes") == spec["onto_passes"]
                  and spec["onto_passes"] == spec["rag_passes"])
        out[key] = {"registry_onto_passes": spec["onto_passes"],
                    "registry_rag_passes": spec["rag_passes"],
                    "receipts": [claim], "receipt_agrees": agrees, "status": "checked"}

    # The 8B pairs have no RAG receipt at all; that IS the finding.
    for spec in REGISTRY:
        if spec["tier"] == "untuned_rag":
            adir = E58 / "adapters" / f"{spec['model']}__rag"
            out[spec["key"]] = {
                "registry_onto_passes": spec["onto_passes"],
                "registry_rag_passes": 0,
                "receipts": [{"path": str(adir.relative_to(CODEX)),
                              "status": "no adapter directory exists",
                              "exists": adir.exists()}],
                "receipt_agrees": not adir.exists(),
            }
    return out


def fmt_signed(value, places=4):
    """A delta always carries its direction; `fmt` is for levels."""
    if not isinstance(value, (int, float)):
        return fmt(value, places)
    return f"{value:+.{places}f}"


def plural_word(n, word: str) -> str:
    return word if n == 1 else word + "es" if word.endswith("ss") else word + "s"


def budget_effect(rows: list[dict]) -> dict:
    """What matching the training budget actually changed, metric by metric.

    A `matched` pair and its `extended` counterpart share the ontology arm exactly -
    same adapter, same answers, same scores - so the only difference between them is
    the extra training given to the baseline. That isolation is what lets this table
    say which of the extended tier's wins were the ontology context and which were
    the budget, instead of only counting how many survived."""
    label = {k: lab for k, lab, _ in METRICS}
    by_key = {r["key"]: r for r in rows}
    at_equal = {r["model"]: r for r in rows if r["tier"] == "equal"}
    out = {}
    for row in rows:
        if row["tier"] != "matched":
            continue
        base = by_key.get(row["key"][:-1] if row["key"].endswith("m") else None)
        if base is None:
            continue
        lost, kept, gained, baseline_worse, drift = [], [], [], [], []
        newly_lost, to_loss = [], []
        for key, now in row["metrics"].items():
            was = base["metrics"].get(key)
            if was is None:
                continue
            # Four of the seven metrics are undefined where an arm produces nothing
            # countable (no entities, no checkable spans), and a paired comparison has
            # to score both arms on the same items. So improving the baseline changes
            # which items are countable, and the ontology arm's own aggregate shifts a
            # little even though its answers are byte-identical. Small, but it is part
            # of every give-back below and is reported rather than absorbed.
            o_was, o_now = was.get("ontology"), now.get("ontology")
            if (isinstance(o_was, (int, float)) and isinstance(o_now, (int, float))
                    and o_was != o_now):
                drift.append({"metric": key, "label": label.get(key, key),
                              "ontology_was": o_was, "ontology_now": o_now,
                              "countable_was": was.get("countable_n"),
                              "countable_now": now.get("countable_n"),
                              "size": abs(o_now - o_was)})
            won_before = was.get("verdict") == "ontology better"
            won_now = now.get("verdict") == "ontology better"
            if (now.get("verdict") == "simple RAG better"
                    and was.get("verdict") != "simple RAG better"):
                eq = at_equal.get(row["model"], {}).get("metrics", {}).get(key, {})
                newly_lost.append({"metric": key, "label": label.get(key, key),
                                   "was": was.get("delta"), "now": now.get("delta"),
                                   "was_verdict": was.get("verdict"),
                                   "equal_verdict": eq.get("verdict"),
                                   "equal_delta": eq.get("delta")})
            if won_before and not won_now:
                entry = {"metric": key, "label": label.get(key, key),
                         "was": was.get("delta"), "now": now.get("delta"),
                         "now_verdict": now.get("verdict")}
                (to_loss if now.get("verdict") == "simple RAG better"
                 else lost).append(entry)
            elif won_before and won_now:
                sign = -1 if now.get("lower_is_better") else 1
                was_m, now_m = was.get("delta"), now.get("delta")
                own = next((d["size"] for d in drift if d["metric"] == key), 0.0)
                kept.append({
                    "metric": key, "label": label.get(key, key),
                    "was": was_m, "now": now_m, "own_drift": own,
                    "narrowed": (isinstance(was_m, (int, float))
                                 and isinstance(now_m, (int, float))
                                 and sign * now_m < sign * was_m),
                    "give_back": (abs(sign * was_m - sign * now_m)
                                  if isinstance(was_m, (int, float))
                                  and isinstance(now_m, (int, float)) else None)})
            elif won_now and not won_before:
                gained.append(key)
            before, after = was.get("simple_rag"), now.get("simple_rag")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                regressed = after > before if now.get("lower_is_better") else after < before
                if regressed:
                    baseline_worse.append({"metric": key, "label": label.get(key, key),
                                           "before": before, "after": after,
                                           "ontology_leads_now":
                                               now.get("verdict") == "ontology better"})
        out[row["key"]] = {"model": row["model"], "extended_key": base["key"],
                           "extra_passes": row["rag_passes"] - base["rag_passes"],
                           "lost": lost, "kept": kept, "gained": gained,
                           "baseline_worse": baseline_worse,
                           "newly_lost": newly_lost, "to_loss": to_loss,
                           "ontology_drift": drift,
                           "max_ontology_drift": max((d["size"] for d in drift),
                                                     default=0.0)}
    return out


def fmt(value, places=4):
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def fmt_ci(ci):
    if ci is None:
        return "—"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def fmt_p(value):
    if value is None:
        return "—"
    if value < 1e-12:
        return "<1e-12"
    return f"{value:.2g}"


def metric_rollup(rows: list[dict]) -> dict:
    """For each tier and metric: how many models land on each verdict."""
    out = {}
    for tier in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        per_metric = {}
        for name, label, lower in METRICS:
            counts = Counter(r["metrics"][name]["verdict"] for r in tier_rows
                             if r["metrics"][name]["countable_n"])
            per_metric[name] = {
                "label": label, "lower_is_better": lower,
                "ontology_better": counts["ontology better"],
                "indistinguishable": counts["indistinguishable"],
                "simple_rag_better": counts["simple RAG better"],
                "models": len(tier_rows),
                "ontology_better_at": [r["model"] for r in tier_rows
                                       if r["metrics"][name]["verdict"] == "ontology better"],
                "simple_rag_better_at": [r["model"] for r in tier_rows
                                         if r["metrics"][name]["verdict"] == "simple RAG better"],
            }
        out[tier] = per_metric
    return out


def summarise(rows: list[dict]) -> dict:
    out = {}
    for tier in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        sweeps = []
        for r in tier_rows:
            verdicts = [m["verdict"] for m in r["metrics"].values()
                        if m.get("verdict") and m["countable_n"]]
            sweeps.append({
                "key": r["key"],
                "ontology_better": sum(v == "ontology better" for v in verdicts),
                "simple_rag_better": sum(v == "simple RAG better" for v in verdicts),
                "indistinguishable": sum(v == "indistinguishable" for v in verdicts),
                "metrics_tested": len(verdicts),
                "clean_sweep": bool(verdicts) and all(v == "ontology better" for v in verdicts),
                "never_worse": bool(verdicts) and not any(v == "simple RAG better" for v in verdicts),
            })
        out[tier] = {"pairs": len(tier_rows), "note": TIER_NOTE[tier], "per_pair": sweeps,
                     "clean_sweeps": sum(s["clean_sweep"] for s in sweeps),
                     "never_worse": sum(s["never_worse"] for s in sweeps)}
    return out


def write_markdown(payload: dict, path: Path) -> None:
    rows = payload["pairs"]
    L = []
    A = L.append
    A("# The ontology-only arm across eight models")
    A("")
    A("WBS 4.1.2 asks for the ontology-only variant to be evaluated and WBS 4.2.2 asks "
      "for the result to be written up. This project measured three Llama models on its "
      f"own {payload['this_project']['benchmark_n']}-item benchmark. The companion project at "
      "`Ontology_Project_Codex` ran the same two arms over the full "
      f"{payload['split_n']:,}-question held-out test set for **{payload['n_models']} models in "
      f"{payload['n_families']} model lines**, which is the widest ontology-arm evidence on record. "
      "This page brings those runs in, re-scores the stored answers with this project's own "
      "metric stack, and attaches the paired tests the companion project never ran.")
    A("")
    A("## Read the training budget first")
    A("")
    A("Each arm pair is labelled by how much training each side received, because that "
      "decides what a win can mean. The counts are read back out of the companion "
      "project's own training receipts, not asserted here.")
    A("")
    A("| Tier | Pairs | What it isolates |")
    A("|---|--:|---|")
    for tier in TIERS:
        s = payload["summary"][tier]
        A(f"| `{tier}` | {s['pairs']} | {s['note']} |")
    A("")
    if payload.get("pending_pairs"):
        A("### Still running")
        A("")
        A(f"The `matched` tier is being produced by `exp060_matched_budget_rag` in this "
          f"repository: for each model the Simple RAG arm is continued from the companion "
          f"project's published one-pass adapter to the same total pass count its ontology "
          f"counterpart had, then generates the same {payload['split_n']:,} held-out "
          f"questions. {len(payload['pending_pairs'])} of these are not finished, and the "
          f"table below does not include them.")
        A("")
        A("| Pair | Model | Ontology passes | RAG passes being trained | Where it is |")
        A("|---|---|--:|--:|---|")
        for row in payload["pending_pairs"]:
            A(f"| `{row['key']}` | {row['model']} | {row['onto_passes']} | "
              f"{row['rag_passes']} | {row['status']} |")
        A("")
    A("## Re-measured on this project's metrics")
    A("")
    A("Token P/R/F1 is the published primary metric; entity P/R/F1 is computed over the "
      "Jeju gazetteer; the ungrounded rate and the fully-grounded share come from the "
      "ontology v1.4.0 groundedness audit, which the companion project never computed for "
      "these models. Every figure below was computed here from the stored answers.")
    A("")
    A("A metric counts as a win only when the paired Wilcoxon signed-rank test survives Holm "
      "correction **and** the bootstrap interval on the mean delta excludes zero. The two "
      "disagree often enough that one alone would mislead, and in both directions: at "
      "`llama3.2-1b@p4` token F1 gains +0.0066 with an interval of [+0.0034, +0.0100] and a "
      "rank-test p of 0.46, because a minority of items carries the whole gain; at "
      "`llama3.2-3b@p1` the ungrounded rate has a rank-test p of 0.00013 and an interval that "
      "covers zero, because the ontology arm is worse on more items while the items it is "
      "better on are worse by more. Both are reported as indistinguishable. Requiring both "
      "tests costs some real effects and admits no spurious ones, which is the trade this "
      "project has taken everywhere else.")
    A("")
    for tier in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        A(f"### Tier `{tier}`")
        A("")
        A(payload["summary"][tier]["note"])
        A("")
        for r in tier_rows:
            def plural(n, word):
                return f"{n} {word}" + ("" if n == 1 else "es")
            A(f"**{r['model']}** ({r['family']}, {r['params_b']}B) — "
              f"{plural(r['onto_passes'], 'ontology pass')} against "
              f"{plural(r['rag_passes'], 'RAG pass')}, "
              f"paired n = {r['paired_n']:,}")
            A("")
            A("| Metric | Ontology only | Simple RAG | Δ | 95% CI of Δ | Holm p | Verdict |")
            A("|---|--:|--:|--:|:--:|--:|---|")
            for name, label, lower in METRICS:
                m = r["metrics"][name]
                arrow = " ↓better" if lower else ""
                A(f"| {label}{arrow} | {fmt(m['ontology'])} | {fmt(m['simple_rag'])} | "
                  f"{('—' if m['delta'] is None else format(m['delta'], '+.4f'))} | "
                  f"{fmt_ci(m.get('ci95'))} | {fmt_p(m.get('p_holm'))} | {m['verdict']} |")
            A("")
    A("## The same question read down the columns")
    A("")
    A("Per metric, how many of the models in each tier the ontology arm leads, ties and "
      "trails on. This is the view that separates where the ontology helps from where it "
      "costs, and it is stable across both tiers that mean anything.")
    A("")
    for tier in [t for t in ("equal", "extended", "matched")
                 if payload["summary"][t]["pairs"]]:
        roll = payload["metric_rollup"][tier]
        n = payload["summary"][tier]["pairs"]
        A(f"**Tier `{tier}`** ({n} models)")
        A("")
        A("| Metric | Ontology leads | Indistinguishable | Simple RAG leads | Leads at |")
        A("|---|--:|--:|--:|---|")
        for name, label, lower in METRICS:
            m = roll[name]
            A(f"| {label}{' ↓better' if lower else ''} | {m['ontology_better']} | "
              f"{m['indistinguishable']} | {m['simple_rag_better']} | "
              f"{', '.join(m['ontology_better_at']) or '—'} |")
        A("")
    A("## What the tiers say")
    A("")
    for tier in TIERS:
        s = payload["summary"][tier]
        A(f"- **`{tier}`** — {s['clean_sweeps']} of {s['pairs']} models lead on every metric "
          f"tested; {s['never_worse']} of {s['pairs']} are never worse on any metric.")
    A("")
    A("## Verdict")
    A("")
    for line in payload["verdict"]:
        A(f"- {line}")
    A("")
    A("## Limits")
    A("")
    for line in payload["limits"]:
        A(f"- {line}")
    A("")
    A("## Provenance")
    A("")
    A("Every arm is a stored generation file in the companion project, read-only, with its "
      "sha256 recorded so the table can be re-derived or contradicted.")
    A("")
    checked = [r for r in rows if r["declared_sha_check"]["status"] == "checked"]
    if checked:
        A(f"For {len(checked)} of these pairs the companion project recorded, in its own "
          f"primary-comparison file, the sha256 of the generation file each arm was scored "
          f"from. Every one of those "
          f"{2 * len(checked)} hashes matches the hash computed here, so this table and the "
          f"companion project's published comparison are reading the same answers rather "
          f"than two runs that happen to share a filename.")
        A("")
    A("| Pair | Arm | Experiment | File | sha256 (first 16) |")
    A("|---|---|---|---|---|")
    for r in rows:
        for arm, s in r["sources"].items():
            A(f"| `{r['key']}` | {arm} | {s['experiment']} | `{s['path']}` | "
              f"`{s['sha256'][:16]}` |")
    A("")
    A(f"Written by `wbs/task4_evaluation/cross_model_results.py` on {payload['generated']}. "
      f"Companion project root: `{CODEX}`.")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-only", action="store_true")
    ap.add_argument("--pair", action="append", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    specs = REGISTRY if not args.pair else [s for s in REGISTRY if s["key"] in args.pair]
    if not specs:
        print("no pair matched", file=sys.stderr)
        return 2

    # exp060's RAG arms are trained and generated here over tens of GPU-hours. A
    # matched pair whose answers are not on disk yet is reported as pending, with
    # what it is waiting for, instead of being silently absent from the table.
    pending, specs_ready = [], []
    for spec in specs:
        absent = [str(q) for q in (spec["onto"], spec["rag"]) if not Path(q).exists()]
        if not absent:
            specs_ready.append(spec)
            continue
        if spec["tier"] != "matched":
            for path in absent:
                print(f"MISSING {spec['key']}: {path}", file=sys.stderr)
            return 2
        pending.append({
            "key": spec["key"], "model": spec["model"], "tier": spec["tier"],
            "onto_passes": spec["onto_passes"], "rag_passes": spec["rag_passes"],
            "waiting_for": absent,
            "status": _exp060_status(spec),
        })
    if pending:
        print(f"matched pairs still running: {len(pending)}")
        for row in pending:
            print(f"  {row['key']}: {row['status']}")
    specs = specs_ready
    if not specs:
        print("no pair has both arms on disk yet", file=sys.stderr)
        return 2

    router = router_firing()
    for r in router:
        if r.get("status") == "absent":
            print(f"router check absent: {r['model']}/{r['split']}", file=sys.stderr)
        else:
            print(f"router on {r['model']}/{r['split']}: replaced {r['fired']} of {r['n']} "
                  f"answers ({r['fired_rate']})")

    budgets = verify_budgets()
    waiting = [k for k, v in budgets.items() if v.get("status") == "pending"]
    disagree = [k for k, v in budgets.items()
                if v.get("status") != "pending" and not v["receipt_agrees"]]
    print(f"training receipts checked: {len(budgets) - len(waiting)}, "
          f"pending: {len(waiting)}, disagreements: {len(disagree)}")
    for k in disagree:
        print(f"  receipt disagreement: {k}", file=sys.stderr)

    reported = {s["key"]: harvest_reported(s) for s in specs}
    if args.harvest_only:
        print(json.dumps({"budgets": budgets, "reported": reported}, indent=1)[:4000])
        return 0

    arms = sorted({str(p) for s in specs for p in (s["onto"], s["rag"])})
    print(f"re-scoring {len(arms)} distinct stored arms on {args.workers} workers")
    scored = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(score_arm, arms):
            scored[result["path"]] = result
            print(f"  scored {Path(result['path']).name}: {result['n_rows']} rows"
                  f"{' (' + str(result['duplicate_lines']) + ' duplicate lines collapsed)' if result['duplicate_lines'] else ''}")

    rows = [compare(s, scored[str(s["onto"])], scored[str(s["rag"])]) for s in specs]

    declared = [r for r in rows if r["declared_sha_check"]["status"] == "checked"]
    bad = [r["key"] for r in declared if not r["declared_sha_check"]["all_agree"]]
    print(f"sha256 cross-checked against the companion project's own comparison "
          f"files for {len(declared)} pairs, mismatches: {len(bad)}")
    if bad:
        print(f"  SHA MISMATCH: {', '.join(bad)} - this table is scoring a different "
              f"file from the published comparison", file=sys.stderr)
        return 3

    summary = summarise(rows)
    equal = [r for r in rows if r["tier"] == "equal"]
    ext = [r for r in rows if r["tier"] == "extended"]

    def sweepers(group):
        return [s["key"] for s in summary[group]["per_pair"] if s["clean_sweep"]]

    rollup = metric_rollup(rows)
    payload_pending = len(pending)
    eq_sweep, ex_sweep = sweepers("equal"), sweepers("extended")
    eq_never = [s["key"] for s in summary["equal"]["per_pair"] if s["never_worse"]]
    ex_never = [s["key"] for s in summary["extended"]["per_pair"] if s["never_worse"]]

    def leads(tier, metric):
        return rollup[tier][metric]["ontology_better_at"]

    verdict = []
    verdict.append(
        "**The standing claim that the ontology arm beats Simple RAG on every metric is still "
        f"false at equal training budget.** {len(eq_sweep)} of {len(equal)} models lead on every "
        f"metric and {len(eq_never)} of {len(equal)} avoid losing on any; every one of the four "
        "gives up token precision. Thirteen times the benchmark and a second vendor's models "
        "did not change that answer.")
    verdict.append(
        "**What does generalise at equal budget is groundedness.** The ontology arm leads the "
        f"fully-grounded share at {len(leads('equal', 'clean'))} of {len(equal)} models "
        f"({', '.join(leads('equal', 'clean')) or 'none'}) and the ungrounded rate at "
        f"{len(leads('equal', 'ungrounded_rate'))} "
        f"({', '.join(leads('equal', 'ungrounded_rate')) or 'none'}), on the one metric in the set "
        "that cannot be won by copying the reference. It is the same trade this project "
        "published on its own 347-item benchmark, and it holds at 4,607 questions - but only on "
        "the Qwen 2.5 line. Neither Llama 3.2 size reproduces it at one pass, and llama3.2-3b is "
        "worse on it, so this is a result about two of four models and is stated that way.")
    verdict.append(
        "**Extra ontology training buys the overlap back, and the budget is not equal while it "
        f"does so.** With three or four ontology passes against the same one-pass RAG baseline, "
        f"{len(ex_never)} of {len(ext)} models are worse on nothing measured and "
        f"{len(ex_sweep)} leads on everything "
        f"({', '.join(ex_sweep) or 'none'}). That is the strongest ontology-arm result on record "
        "in either project. It is also a 3:1 or 4:1 training advantage, so it bounds what the "
        "ontology context can reach and says nothing about parity at equal cost.")
    mt = summary["matched"]
    if mt["pairs"]:
        m_sweep = [x["key"] for x in mt["per_pair"] if x["clean_sweep"]]
        effect = budget_effect(rows)
        verdict.append(
            "**With the budget matched, the extended result "
            + ("holds" if mt["never_worse"] == mt["pairs"] else "does not hold") + ".** "
            f"Against a Simple RAG arm continued to the same total pass count, "
            f"{mt['never_worse']} of {mt['pairs']} "
            f"{plural_word(mt['pairs'], 'model')} "
            f"{'is' if mt['never_worse'] == 1 else 'are'} worse on nothing measured and "
            f"{len(m_sweep)} {'leads' if len(m_sweep) == 1 else 'lead'} on every metric "
            f"({', '.join(m_sweep) or 'none'}). This is the tier the `extended` rows could "
            f"not supply, and it is the one any parity claim at extended training has to "
            f"rest on.")
        # The counts above cannot distinguish a win that became a tie from a win that
        # became a loss, and that distinction is the whole point of the tier.
        for key, e in sorted(effect.items()):
            moved = ", ".join(
                f"{d['label'].lower()} from {fmt_signed(d['was'])} to "
                f"{fmt_signed(d['now'])} ({d['now_verdict']})" for d in e["lost"])
            won = len(e["kept"]) + len(e["lost"]) + len(e["to_loss"])
            head = (f"**{e['model']}: {len(e['kept'])} of "
                    f"{won} {'win' if won == 1 else 'wins'} survive the correction.** "
                    f"Giving Simple RAG the same {e['extra_passes']} extra "
                    f"{plural_word(e['extra_passes'], 'pass')} ")
            fell = ", ".join(
                f"{d['label'].lower()} from {fmt_signed(d['was'])} to "
                f"{fmt_signed(d['now'])}" for d in e["to_loss"])
            clauses = []
            if e["lost"]:
                clauses.append(f"turns {len(e['lost'])} of them into a tie ({moved})")
            if e["to_loss"]:
                clauses.append(f"turns {len(e['to_loss'])} of them into an outright loss "
                               f"({fell})")
            head += (", ".join(clauses) + ", " if clauses else "changes no verdict, ")
            narrowed = [k for k in e["kept"] if k["narrowed"]]
            widened = [k for k in e["kept"] if not k["narrowed"]]
            head += ("and leaves nothing standing" if not e["kept"]
                     else f"and leaves the other {len(e['kept'])} standing")
            if narrowed:
                worst = max(narrowed, key=lambda k: k["give_back"])
                head += (f", {len(narrowed)} of them at a smaller margin than the "
                         f"`extended` row claimed (most of all {worst['label'].lower()}, "
                         f"{fmt_signed(worst['was'])} down to {fmt_signed(worst['now'])})")
            if widened:
                head += f" and {len(widened)} at a wider one"
            head += (". The ontology arm is the same answers in both rows, so every one of "
                     "those movements is the baseline getting better or worse on its own.")
            if e["newly_lost"]:
                # the wins that fell to a defeat are already named above; what is new
                # here is any metric that was merely level and is now behind
                nl = e["newly_lost"]
                from_tie = [d for d in nl if d["was_verdict"] != "ontology better"]
                if from_tie:
                    head += (f" {len(from_tie)} "
                             f"{'metric that was' if len(from_tie) == 1 else 'metrics that were'}"
                             f" level at the extended budget "
                             f"{'is' if len(from_tie) == 1 else 'are'} behind at this one: "
                             + ", ".join(
                                 f"{d['label'].lower()} {fmt_signed(d['was'])} to "
                                 f"{fmt_signed(d['now'])}" for d in from_tie) + ".")
                foretold = [d for d in nl if d["equal_verdict"] == "simple RAG better"]
                head += " "
                if len(foretold) == len(nl):
                    head += (f"The equal-budget tier already had the ontology arm behind on "
                             f"{'that metric' if len(nl) == 1 else 'every one of those'}, so "
                             f"{'it is' if len(nl) == 1 else 'they are'} not a new weakness - "
                             f"{'it was' if len(nl) == 1 else 'they were'} hidden by the "
                             f"baseline's shorter training and "
                             f"{'reappears' if len(nl) == 1 else 'reappear'} once that is "
                             f"taken away.")
                elif foretold:
                    head += (f"The equal-budget tier already had the ontology arm behind on "
                             f"{len(foretold)} of those "
                             f"({', '.join(d['label'].lower() for d in foretold)}); the rest "
                             f"are new at this budget.")
                else:
                    head += ("None of those was a loss at equal budget, so these are not "
                             "the equal-budget weakness reappearing - they are the baseline "
                             "overtaking the ontology arm as both are trained longer.")
                head += (" Either way this model fails the worse-on-nothing test its "
                         "`extended` row passed.")
            if e["max_ontology_drift"]:
                # Each movement is judged against its own metric's denominator drift;
                # the largest drift in the pair belongs to one metric and says nothing
                # about the others.
                moves = [k for k in e["kept"] if k["give_back"]]
                murky = [k for k in moves if k["give_back"] <= k["own_drift"]]
                ratios = [k["give_back"] / k["own_drift"] for k in moves
                          if k["own_drift"] > 0]
                head += (
                    f" Strictly, {len(e['ontology_drift'])} of the metrics are defined only "
                    f"where both arms produce something countable, so a better baseline also "
                    f"shifts their denominators and the ontology arm's own aggregate moves by "
                    f"up to {fmt(e['max_ontology_drift'])} on its own.")
                if murky:
                    head += (
                        " On "
                        + ", ".join(f"{k['label'].lower()} (movement {fmt(k['give_back'])} "
                                    f"against a drift of {fmt(k['own_drift'])})"
                                    for k in murky)
                        + " that drift is as large as the movement itself, so the direction "
                          "the margin took is not interpreted here - only the verdict, which "
                          "is decided on paired per-question deltas and not on these "
                          "aggregates.")
                else:
                    head += (
                        f" Every movement above is at least "
                        f"{min(ratios):.0f}x its own metric's drift"
                        if ratios else " No movement above shares a metric with that drift"
                    ) + ", so none of them is an artefact of the denominator."
                head += (" No verdict rests on these aggregates in any case: each one is "
                         "decided by a paired test over the per-question deltas within its "
                         "own row, where both arms are scored on the identical item set.")
            if e["baseline_worse"]:
                hurt = ", ".join(f"{d['label'].lower()} {fmt(d['before'])} -> "
                                 f"{fmt(d['after'])}" for d in e["baseline_worse"])
                ahead = [d for d in e["baseline_worse"] if d["ontology_leads_now"]]
                head += (f" Note what the extra training did not buy the baseline: it got "
                         f"worse on {hurt}. ")
                if len(ahead) == len(e["baseline_worse"]):
                    head += ("The ontology lead on those metrics therefore widens rather "
                             "than narrows: more passes over retrieved passages improve "
                             "overlap with the reference and do not improve grounding.")
                elif ahead:
                    head += (f"On {', '.join(d['label'].lower() for d in ahead)} the "
                             f"ontology arm is ahead, so that lead widens rather than "
                             f"narrows; on the rest the baseline fell and the ontology arm "
                             f"still does not lead, so nothing is claimed from it.")
                else:
                    head += ("The ontology arm does not lead those metrics even so, so this "
                             "is not an ontology advantage - it says only that more passes "
                             "over retrieved passages do not improve grounding.")
            verdict.append(head)
    elif payload_pending:
        verdict.append(
            f"**The `matched` tier is not in yet.** {payload_pending} of its pairs are "
            f"still training or generating in `exp060_matched_budget_rag`, so the "
            f"`extended` rows above remain confounded with a 3:1 or 4:1 training "
            f"advantage and are read that way until those runs land.")
    verdict.append(
        "**The four 8B pairs are not evidence about the ontology.** The companion project trained "
        "an adapter for the ontology arm and none for Simple RAG - the RAG adapter directory does "
        "not exist - so the enormous margins there (deepseek-r1-8b gains +0.21 token F1) measure "
        "fine-tuning. They are excluded from every claim above. Worth noting even so: with that "
        "advantage in hand the ontology arm still loses token recall at three of the four and "
        "the entity hit rate at two, which is the same shape as the equal-budget result.")

    limits = [
        "The extended tier is budget-asymmetric by construction and cannot separate the "
        "ontology context from the extra training passes. The equal tier is the only one that "
        "isolates the context format, and it is the tier any parity claim must rest on.",
        "Reference answers in this corpus are synthetic. An arm whose evidence is drawn from "
        "the same corpus the reference was written from has a structural advantage on token "
        "overlap; §5.1g of the final report measures that advantage at 17-22% of the gap.",
        "BERTScore, COMET and the LLM-judge means harvested from the companion project are "
        "reproduced as reported. Their per-item values were not persisted for the Qwen and "
        "8B runs, so no paired test could be attached to them here; only the metrics "
        "re-measured from the stored answers carry a p-value.",
        "The companion project's router (`hybrid`) replaces "
        + "; ".join(f"{r['fired']} of {r['n']} answers on {r['model']}/{r['split']}"
                    for r in router if r.get("status") != "absent")
        + ". On the held-out test split it never fires, so there the hybrid arm and the "
        "ontology arm are literally the same answers and the two must not be counted as "
        "separate evidence. Its large gains live on the temporal splits, where it answers "
        "almost every question - and where the reference and the router read the same graph, "
        "so that margin is a measure of graph coverage, not of generation quality.",
        "Ontology v1.4.0 of THIS project is the groundedness auditor. The companion project's "
        "arms were generated against its own ontology v1.2.0 contexts, so the ungrounded rate "
        "is an audit by a later, larger graph, which is the right direction for a hallucination "
        "check and the wrong direction for crediting the context.",
    ]

    payload = {
        "generated": __import__("datetime").date.today().isoformat(),
        "companion_project": str(CODEX),
        "split": "test (held out)",
        "split_n": max(r["paired_n"] for r in rows),
        "n_models": len({r["model"] for r in rows}),
        "n_families": len({r["family"] for r in rows}),
        "this_project": {"benchmark_n": 347,
                         "models": ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"],
                         "variants": ["V1_llm", "V2_rag", "V3_ontology_only",
                                      "V4_ontology_rag", "V5_ontology_rag_validated"]},
        "metrics_remeasured": [m[0] for m in METRICS],
        "distinct_arms_rescored": len(arms),
        "bootstrap_samples": BOOT, "seed": SEED,
        "pending_pairs": pending,
        "router_firing": router,
        "metric_rollup": rollup,
        "budget_effect": budget_effect(rows),
        "training_budgets": budgets,
        "reported_by_companion": reported,
        "pairs": rows,
        "summary": summary,
        "verdict": verdict,
        "limits": limits,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_model_results.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")
    write_markdown(payload, OUT / "cross_model_results.md")

    print()
    for tier in TIERS:
        s = summary[tier]
        print(f"{tier:12s} pairs={s['pairs']} clean_sweeps={s['clean_sweeps']} "
              f"never_worse={s['never_worse']}")
    print()
    for line in verdict:
        print(f"- {line}")
    print()
    print(f"wrote {OUT / 'cross_model_results.json'}")
    print(f"wrote {OUT / 'cross_model_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
