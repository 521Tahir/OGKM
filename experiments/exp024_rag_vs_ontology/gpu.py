"""Free-memory-aware GPU picker.

This box is shared: GPUs 5-9 run the GROUND project's AMBER jobs and another
tenant intermittently claims 0-3. Jobs are placed only on GPUs that currently
have enough headroom, so a run never OOMs someone else's work (or its own).
"""
from __future__ import annotations

# Tahir restricted this project to GPUs 5-9 (2026-09-17). Everything else on
# the box belongs to other work; never place a job outside this set.
# Tahir added GPUs 3 and 4 to the pool (2026-09-22); 0-2 remain other tenants'.
ALLOWED = [3, 4, 5, 6, 7, 8, 9]

import subprocess


def free_mib() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, tot, used = [int(x.strip()) for x in line.split(",")]
        free[i] = tot - used
    return free


def pick(candidates: list[int], need_mib: int, n: int = None) -> list[int]:
    """Candidate GPUs with >= need_mib free, most-free first."""
    f = free_mib()
    ok = sorted([g for g in candidates if f.get(g, 0) >= need_mib],
                key=lambda g: -f[g])
    return ok[:n] if n else ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", type=int, default=ALLOWED)
    ap.add_argument("--need", type=int, required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    print(" ".join(str(g) for g in pick(a.candidates, a.need, a.n or None)))
