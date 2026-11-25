"""
Structured JSON logging for the API layer.

Produces one JSON object per log line with consistent fields:
  timestamp, level, logger, request_id, method, path, status, elapsed_ms, message

Usage:
    from src.utils.json_logger import setup_json_logging, bind_request_id
    setup_json_logging(level="INFO", log_file="logs/api.log")
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Per-request context variable — set in middleware, read in formatter
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_method_var:     ContextVar[str] = ContextVar("method",     default="-")
_path_var:       ContextVar[str] = ContextVar("path",       default="-")


def bind_request(request_id: str, method: str, path: str) -> None:
    """Call once per request (in middleware) to attach context to all log lines."""
    _request_id_var.set(request_id)
    _method_var.set(method)
    _path_var.set(path)


def new_request_id() -> str:
    return str(uuid.uuid4())[:8]


class _JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp":  self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":      record.levelname,
            "logger":     record.name,
            "request_id": _request_id_var.get(),
            "method":     _method_var.get(),
            "path":       _path_var.get(),
            "message":    record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Human-readable coloured text (same style as existing logging_utils.py)."""

    _COLOURS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        ts     = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        rid    = _request_id_var.get()
        msg    = record.getMessage()
        line   = f"{colour}{ts}{self._RESET} | {record.levelname:<8} | {record.name} | [{rid}] {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_json_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = "logs/api.log",
    rotation_bytes: int = 50 * 1024 * 1024,
) -> None:
    """
    Configure the root logger with JSON or text handlers.

    Call once at app startup (before any route handlers run).
    Replaces any existing handlers on the root logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    formatter: logging.Formatter = (
        _JsonFormatter() if fmt == "json" else _TextFormatter()
    )

    # stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # rotating file handler (optional)
    if log_file:
        from logging.handlers import RotatingFileHandler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=rotation_bytes,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "transformers", "bitsandbytes"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
