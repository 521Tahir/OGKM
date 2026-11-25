"""GPU assignment + availability helpers (per application.txt deployment plan)."""
from __future__ import annotations
import subprocess


def gpu_free_mb() -> dict[int, int]:
    """Return {gpu_index: free_MB} via nvidia-smi (empty on failure)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True)
        return {int(i): int(f) for i, f in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        return {}


def assignment(models: list[dict]) -> dict[str, int]:
    """model name -> GPU id (static plan from config)."""
    return {m["name"]: m["gpu"] for m in models}
