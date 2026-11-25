"""
FastAPI application factory with lifespan resource management.

Startup loads ontology + optionally the LLM; all heavy objects are stored in
app.state.app_state and injected into route handlers via get_state().

Configuration is read from config/api_config.yaml; individual keys can be
overridden with environment variables (see api_config.yaml for the full map).
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import yaml
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .routes import router
from ..utils.json_logger import bind_request, new_request_id, setup_json_logging
from ..utils.logging_utils import get_logger

ROOT        = Path(__file__).resolve().parents[3]
_CONFIG_PATH = ROOT / "config" / "api_config.yaml"

logger = get_logger("api.app")


# ── Config loading ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load api_config.yaml; fall back to sensible defaults if absent."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _cfg(cfg: dict, *keys: str, env: str | None = None, default: Any = None) -> Any:
    """Drill into nested config dict; env var takes precedence if set."""
    if env and os.environ.get(env) is not None:
        raw = os.environ[env]
        # coerce "1"/"0" → bool for boolean fields
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes")
        if isinstance(default, int):
            return int(raw)
        return raw
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default if k == keys[-1] else {})
    return node


# ── Rate limiter (module-level, shared across requests) ───────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cfg = _load_config()

    # Structured logging
    log_level = _cfg(cfg, "logging", "level", env="LOG_LEVEL", default="INFO")
    log_fmt   = _cfg(cfg, "logging", "format", default="json")
    log_file  = _cfg(cfg, "logging", "file",   env="LOG_FILE",  default="logs/api.log")
    setup_json_logging(level=log_level, fmt=log_fmt, log_file=log_file or None)

    # Rate limiter limits from config
    rl_cfg = cfg.get("rate_limiting", {})
    if rl_cfg.get("enabled", True):
        default_limit = rl_cfg.get("default", "60/minute")
        limiter._default_limits = [default_limit]
    app.state.limiter        = limiter
    app.state.rate_limit_cfg = rl_cfg

    state: dict = {}

    # ── Ontology ──────────────────────────────────────────────────────────────
    ont_version = _cfg(cfg, "ontology", "version", env="ONTOLOGY_VERSION", default="0.7.0")
    ont_dir_rel = _cfg(cfg, "ontology", "dir", default="artifacts/ontologies")
    ont_dir     = ROOT / ont_dir_rel
    ont_path    = ont_dir / ont_version

    try:
        from ..graph.graph_store import TourismGraphStore
        from ..utils.io import load_json

        if ont_path.exists():
            graph     = TourismGraphStore.from_version(ont_version, ont_dir)
            concepts  = load_json(ont_path / "concepts.json")
            relations = load_json(ont_path / "relations.json")
            state.update({
                "graph":            graph,
                "concepts":         concepts,
                "relations":        relations,
                "ontology_version": ont_version,
            })
            logger.info(
                f"Ontology v{ont_version}: {len(concepts):,} concepts, "
                f"{len(relations):,} relations"
            )
        else:
            logger.warning(f"Ontology path not found: {ont_path} — offline mode")
            state.update({"graph": None, "concepts": [], "relations": [],
                          "ontology_version": "none"})
    except Exception as exc:
        logger.error(f"Ontology load error: {exc}")
        state.update({"graph": None, "concepts": [], "relations": [],
                      "ontology_version": "error"})

    # ── Concept linker ────────────────────────────────────────────────────────
    try:
        if state.get("graph") is not None:
            from ..retrieval.concept_linker import ConceptLinker
            state["linker"] = ConceptLinker(graph=state["graph"])
            logger.info("ConceptLinker ready")
        else:
            state["linker"] = None
    except Exception as exc:
        logger.warning(f"ConceptLinker init failed: {exc}")
        state["linker"] = None

    # ── LLM (opt-in) ─────────────────────────────────────────────────────────
    state["llm_client"] = None
    state["model_name"]  = "none"

    load_llm = _cfg(cfg, "llm", "load", env="LOAD_LLM", default=False)
    if load_llm:
        try:
            from ..llm.local_client import LocalLLMClient

            cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_env:
                gpu_ids = [int(g) for g in cuda_env.split(",")]
            else:
                gpu_ids = cfg.get("llm", {}).get("gpu_ids", [5, 6])

            llm = LocalLLMClient(
                model_key=cfg.get("llm", {}).get("model_key", "llama3.1-8b"),
                gpu_ids=gpu_ids,
                load_in_4bit=cfg.get("llm", {}).get("load_in_4bit", True),
                hf_token=os.environ.get("HF_TOKEN"),
            )
            state["llm_client"] = llm
            state["model_name"]  = cfg.get("llm", {}).get("model_key", "llama3.1-8b")
            logger.info(f"LLM loaded on GPU(s) {gpu_ids}")
        except Exception as exc:
            logger.error(f"LLM load failed: {exc}")

    app.state.app_state = state
    logger.info("API startup complete")

    yield

    logger.info("API shutting down")
    if state.get("llm_client") is not None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = _load_config()

    app = FastAPI(
        title="Tourism Ontology RAG API",
        description=(
            "FastAPI service exposing the Jeju tourism ontology-backed RAG pipeline. "
            "Supports four pipeline modes: no_rag, doc_rag, onto_rag, constrained."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_cfg = cfg.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get("allow_origins", ["*"]),
        allow_methods=cors_cfg.get("allow_methods", ["*"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )

    # ── Structured request logging middleware ─────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        rid = new_request_id()
        bind_request(rid, request.method, request.url.path)
        request.state.request_id = rid

        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"status={response.status_code} elapsed_ms={elapsed}"
        )
        response.headers["X-Request-ID"] = rid
        return response

    # ── Global error handler ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail":     "Internal server error",
                "error":      str(exc),
                "request_id": getattr(request.state, "request_id", "-"),
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    prefix = cfg.get("server", {}).get("prefix", "/api/v1")
    app.include_router(router, prefix=prefix)

    return app


app = create_app()
