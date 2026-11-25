"""
Orchestrator: fine-tune -> generate -> score -> report, resumable.

The box is shared, so jobs are dispatched onto whichever GPUs currently have
enough free memory (GROUND's AMBER jobs on 5-9 and another tenant on 0-3 are
never displaced). Every stage skips work that is already on disk, so this can be
killed and rerun at any point.

  python -m experiments.exp024_rag_vs_ontology.run_all --stage all
  python -m experiments.exp024_rag_vs_ontology.run_all --stage train --arms rag onto
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .gpu import pick

sys.path.insert(0, str(C.ROOT))
from src.utils.proctitle import set_proctitle

PY = sys.executable
# Tahir restricted this project to GPUs 5-9 only (2026-09-17). The
# free-memory gate still decides placement, so a job only lands on 5-9 once an
# AMBER run has actually released enough memory — nothing gets displaced.
CAND = [5, 6, 7, 8, 9]
POLL = 60


def _log(job: str) -> Path:
    return C.LOG_DIR / f"{job}.log"


def _launch(cmd: list[str], job: str) -> subprocess.Popen:
    f = open(_log(job), "a", buffering=1, encoding="utf-8")
    f.write(f"\n=== {time.strftime('%F %T')} :: {' '.join(cmd)}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(f"  launch {job}: {' '.join(cmd[-6:])}", flush=True)
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(C.ROOT))


def _run_pool(jobs: list[dict], need_mib: int, max_par: int) -> None:
    """Dispatch jobs onto free GPUs, at most one job per GPU at a time."""
    pending, running = list(jobs), {}          # gpu -> (proc, job)
    while pending or running:
        for g, (proc, job) in list(running.items()):
            if proc.poll() is not None:
                ok = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"  [{time.strftime('%T')}] {job['name']} on GPU {g}: {ok}", flush=True)
                del running[g]
        while pending and len(running) < max_par:
            busy = set(running)
            free = [g for g in pick(CAND, need_mib) if g not in busy]
            if not free:
                break
            g, job = free[0], pending.pop(0)
            running[g] = (_launch(job["cmd"] + ["--gpu", str(g)], job["name"]), job)
        if pending or running:
            time.sleep(POLL)


# ── stages ────────────────────────────────────────────────────────────────
def stage_train(arms: list[str], models: list[str], force: bool) -> None:
    """Train the adapters the requested arms need (rag+onto share one)."""
    need_adapters = []
    for arm in arms:
        ad = C.ADAPTER_FOR_ARM.get(arm)
        if ad and ad not in need_adapters:
            need_adapters.append(ad)
    jobs = []
    for arm in need_adapters:
        for mk in models:
            out = C.ADAPTER_DIR / f"{mk}__{arm}"
            if (out / "adapter_model.safetensors").exists() and not force:
                print(f"  skip train {mk}/{arm} (adapter exists)", flush=True)
                continue
            jobs.append({"name": f"train__{mk}__{arm}",
                         "cmd": [PY, "-m", "experiments.exp024_rag_vs_ontology.train_lora",
                                 "--model", mk, "--arm", arm]})
    if not jobs:
        return
    need = C.MIN_FREE_MIB_4BIT if C.LOAD_4BIT else C.MIN_FREE_MIB_BF16
    print(f"\n== TRAIN: {len(jobs)} jobs, need {need} MiB/GPU "
          f"({'QLoRA 4-bit' if C.LOAD_4BIT else 'bf16 LoRA'})", flush=True)
    _run_pool(jobs, need, max_par=len(CAND))


def stage_generate(arms: list[str], models: list[str], split: str, force: bool) -> None:
    jobs = []
    for arm in arms:
        for mk in models:
            out = C.GEN_DIR / f"{split}__{mk}__{arm}.jsonl"
            n = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
            if n >= 4607 and not force:
                print(f"  skip gen {mk}/{arm} ({n:,} rows)", flush=True)
                continue
            jobs.append({"name": f"gen__{mk}__{arm}",
                         "cmd": [PY, "-m", "experiments.exp024_rag_vs_ontology.generate",
                                 "--model", mk, "--arm", arm, "--split", split]})
    if not jobs:
        return
    print(f"\n== GENERATE: {len(jobs)} jobs", flush=True)
    _run_pool(jobs, 22000, max_par=len(CAND))


def stage_score(split: str) -> None:
    g = pick(CAND, 26000, 1)
    gpu = g[0] if g else 4
    print(f"\n== SCORE on GPU {gpu}", flush=True)
    p = _launch([PY, "-m", "experiments.exp024_rag_vs_ontology.score",
                 "--split", split, "--metrics", "all", "--gpu", str(gpu)], "score")
    rc = p.wait()
    print(f"  score rc={rc}", flush=True)


def stage_report(split: str) -> None:
    print("\n== REPORT", flush=True)
    subprocess.run([PY, "-m", "experiments.exp024_rag_vs_ontology.report", "--split", split],
                   cwd=str(C.ROOT), check=False)


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["all"],
                    choices=["all", "train", "generate", "score", "report"])
    ap.add_argument("--arms", nargs="+", default=["rag", "onto"])
    ap.add_argument("--models", nargs="+", default=[m["key"] for m in C.MODELS])
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--per-model", action="store_true",
                    help="finish one model end-to-end (train->generate->score->report) "
                         "before starting the next, so a complete result lands early")
    a = ap.parse_args()
    stages = ["train", "generate", "score", "report"] if "all" in a.stage else a.stage

    def cycle(models: list[str], report: bool) -> None:
        if "train" in stages:
            stage_train(a.arms, models, a.force)
        if "generate" in stages:
            stage_generate(a.arms, models, a.split, a.force)
        if "score" in stages:
            stage_score(a.split)
        if report and "report" in stages:
            stage_report(a.split)

    if a.per_model:
        for i, mk in enumerate(a.models, 1):
            print(f"\n{'#'*70}\n# MODEL {i}/{len(a.models)}: {mk}\n{'#'*70}", flush=True)
            cycle([mk], report=True)
    else:
        cycle(a.models, report=True)


if __name__ == "__main__":
    main()
