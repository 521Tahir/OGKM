"""Lightweight run logging to results/logs/."""
from __future__ import annotations
import json
from datetime import datetime
from ..config import LOGS


def log_run(payload: dict) -> str:
    path = LOGS / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
