"""Async/parallel dispatch helper — thin wrapper over ModelPool.generate_all.

Total latency is max(model times), not the sum, because each model runs in its
own GPU-pinned worker process and all jobs are dispatched before collection.
"""
from __future__ import annotations


def run_parallel(pool, jobs: dict) -> dict:
    return pool.generate_all(jobs)
