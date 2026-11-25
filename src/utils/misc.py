"""Miscellaneous shared utilities."""
import time
import hashlib
from typing import Any
from functools import wraps


def timer(func):
    """Decorator that logs execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def stable_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest for the given string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_list(lst: list, size: int):
    """Yield successive chunks of `size` from lst."""
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def flatten(nested: list[list[Any]]) -> list[Any]:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in sublist]
