"""I/O helper utilities."""
import json
import os
from pathlib import Path
from typing import Any, Iterator


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize obj to a JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over a JSONL file one record at a time (memory-efficient)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
