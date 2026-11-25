"""Simple in-memory result cache keyed by (question)."""
from __future__ import annotations
_CACHE: dict = {}


def get(key): return _CACHE.get(key)
def put(key, value): _CACHE[key] = value
def clear(): _CACHE.clear()
