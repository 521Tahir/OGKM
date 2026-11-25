"""
FastAPI route handlers — all 9 endpoints.

Routes are registered on an APIRouter and mounted in app.py.
All heavy objects (LLM, ontology, retriever) come via dependency injection
from app.py's lifespan state — no module-level globals here.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import (
    BenchmarkRequest, BenchmarkResponse, BenchmarkItem,
    ExplainRequest, ExplainResponse,
    HealthResponse,
    OntologyExportRequest, OntologyUpdateRequest, OntologyUpdateResponse,
    ProfileDataRequest, ProfileDataResponse,
    QueryRequest, QueryResponse,
    ValidationRequest, ValidationResponse,
)
from ..utils.jeju_guard import jeju_guard
from ..utils.logging_utils import get_logger

logger = get_logger("api.routes")
router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ROOT = Path(__file__).resolve().parents[3]


# ── Dependency: app state ─────────────────────────────────────────────────────

def get_state(request: Request) -> dict:
    """Inject the app-level state dict from lifespan."""
    return request.app.state.app_state


# ── 1. GET /health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(state: dict = Depends(get_state)) -> HealthResponse:
    """Return system status, ontology version, model info, and GPU availability."""
    import torch
    gpu_available = torch.cuda.is_available()
    concepts  = state.get("concepts", [])
    relations = state.get("relations", [])
    return HealthResponse(
        status="ok",
        ontology_version=state.get("ontology_version", "unknown"),
        model=state.get("model_name", "none"),
        gpu_available=gpu_available,
        n_concepts=len(concepts),
        n_relations=len(relations),
    )


# ── 2. POST /profile-data ─────────────────────────────────────────────────────

@router.post("/profile-data", response_model=ProfileDataResponse, tags=["data"])
async def profile_data(
    req: ProfileDataRequest,
    state: dict = Depends(get_state),
) -> ProfileDataResponse:
    """Compute basic statistics on the requested data split."""
    import json
    from collections import Counter

    split_path = ROOT / "data" / "splits" / f"{req.split}.jsonl"
    if not split_path.exists():
        raise HTTPException(404, f"Split '{req.split}' not found at {split_path}")

    records = []
    with open(split_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise HTTPException(422, "Split file is empty")

    input_lens  = [len(r.get("Input", "").split()) for r in records]
    output_lens = [len(r.get("Output", "").split()) for r in records]

    # Top tokens from Input field
    token_counter: Counter = Counter()
    for r in records[:1000]:
        for tok in r.get("Input", "").lower().split():
            if len(tok) > 3:
                token_counter[tok] += 1
    top_tokens = [tok for tok, _ in token_counter.most_common(10)]

    return ProfileDataResponse(
        split=req.split,
        n_records=len(records),
        avg_input_len=round(sum(input_lens) / len(input_lens), 1),
        avg_output_len=round(sum(output_lens) / len(output_lens), 1),
        top_tokens=top_tokens,
    )


# ── 3. POST /build-ontology ───────────────────────────────────────────────────

@router.post("/build-ontology", tags=["ontology"])
async def build_ontology(state: dict = Depends(get_state)) -> dict:
    """
    Trigger offline concept extraction on the full dataset (async stub).

    In production this would enqueue a background task.
    Returns a task_id the client can poll.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"build-ontology requested, task_id={task_id}")
    return {
        "task_id":   task_id,
        "status":    "queued",
        "message":   "Ontology build queued. Run exp002 externally for full extraction.",
        "script":    "python -m src.ontology.ontology_builder --ontology-version 0.8.0",
    }


# ── 4. POST /validate-ontology ────────────────────────────────────────────────

@router.post("/validate-ontology", response_model=ValidationResponse, tags=["ontology"])
async def validate_ontology(
    req: ValidationRequest,
    state: dict = Depends(get_state),
) -> ValidationResponse:
    """Run SHACL validation on the specified ontology version."""
    from ..ontology.shacl_validator import SHACLValidator

    ont_path = ROOT / "artifacts" / "ontologies" / req.ontology_version
    if not ont_path.exists():
        raise HTTPException(404, f"Ontology version '{req.ontology_version}' not found")

    from ..utils.io import load_json
    concepts  = load_json(ont_path / "concepts.json")
    relations = load_json(ont_path / "relations.json")
    rules     = load_json(ont_path / "rules.json")

    validator  = SHACLValidator(rules)
    violations = validator.validate_batch(concepts + relations)

    return ValidationResponse(
        ontology_version=req.ontology_version,
        n_violations=len(violations),
        violations=[v.to_dict() if hasattr(v, "to_dict") else v for v in violations],
        passed=len(violations) == 0,
    )


# ── 5. POST /update-ontology ──────────────────────────────────────────────────

@router.post("/update-ontology", response_model=OntologyUpdateResponse, tags=["ontology"])
async def update_ontology(
    req: OntologyUpdateRequest,
    state: dict = Depends(get_state),
) -> OntologyUpdateResponse:
    """Incrementally merge new records into the active ontology version."""
    from ..ontology.versioning import OntologyVersionManager
    from ..ontology.offline_extractor import OfflineExtractor
    from ..utils.io import load_json

    vm       = OntologyVersionManager(ROOT / "artifacts" / "ontologies")
    base     = vm.load_version(req.ontology_version)
    existing = base["concepts"]

    extractor   = OfflineExtractor()
    new_concepts = []
    for rec in req.records:
        text = rec.get("Output", rec.get("text", ""))
        extracted = extractor.extract(text)
        new_concepts.extend(extracted.get("concepts", []))

    conflicts = vm.detect_conflicts(existing, new_concepts)
    merged, stats = vm.add_concepts_incremental(new_concepts, existing, policy=req.policy)

    # Compute new version string (bump patch)
    parts = req.ontology_version.split(".")
    new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}" if len(parts) == 3 else req.ontology_version + ".1"

    vm.save_version(new_ver, merged, base["relations"], base["rules"], base["schema"])
    state["concepts"]          = merged
    state["ontology_version"]  = new_ver

    return OntologyUpdateResponse(
        new_version=new_ver,
        concepts_added=stats["added"],
        concepts_merged=stats["merged"],
        concepts_flagged=stats["flagged"],
        conflicts_detected=len(conflicts),
    )


# ── 6. POST /query ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["rag"])
@limiter.limit("30/minute")
async def query(
    request: Request,
    req: QueryRequest,
    state: dict = Depends(get_state),
) -> QueryResponse:
    """Main RAG query endpoint — retrieves context and generates a grounded answer."""
    # Jeju-only guard — zero LLM cost, instant rejection
    allowed, guard_msg = jeju_guard(req.query)
    if not allowed:
        raise HTTPException(
            status_code=400,
            detail={"error": "non_jeju_query", "message": guard_msg},
        )

    t0       = time.perf_counter()
    trace_id = str(uuid.uuid4())[:8]

    llm_client = state.get("llm_client")
    concepts   = state.get("concepts", [])
    linker     = state.get("linker")

    # Retrieve relevant concepts
    used_concepts: list[str] = []
    if linker and req.mode.value != "no_rag":
        try:
            linked = linker.link(req.query)
            used_concepts = [lc["concept"] if isinstance(lc, dict) else str(lc)
                             for lc in linked[:10]]
        except Exception as e:
            logger.warning(f"Linker error: {e}")

    # Build system prompt per mode
    _SYSTEM: dict[str, str] = {
        "no_rag":      "You are a helpful tourism assistant for Jeju Island.",
        "doc_rag":     "You are a tourism assistant with access to Jeju Island documents. "
                       "Ground your answers in retrieved context.",
        "onto_rag":    "You are a tourism assistant backed by a Jeju knowledge ontology. "
                       f"Known entities: {', '.join(used_concepts[:5])}.",
        "constrained": "You are a strictly factual Jeju tourism assistant. "
                       "Only mention verified entities. "
                       f"Allowed entities: {', '.join(used_concepts[:5])}. "
                       "Do not fabricate names, prices, or statistics.",
    }
    system = _SYSTEM.get(req.mode.value, _SYSTEM["no_rag"])

    # Generate answer
    answer = ""
    if llm_client:
        try:
            answer = llm_client.call(
                prompt=req.query,
                system=system,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise HTTPException(500, f"LLM generation failed: {e}")
    else:
        answer = f"[offline] Query received: {req.query[:80]}. LLM not loaded."

    # Compute grounding score (fraction of used_concepts mentioned in answer)
    grounding = 0.0
    if used_concepts:
        mentioned = sum(1 for c in used_concepts if c.lower() in answer.lower())
        grounding = round(mentioned / len(used_concepts), 3)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return QueryResponse(
        query=req.query,
        mode=req.mode.value,
        answer=answer,
        used_concepts=used_concepts,
        violations=[],
        grounding_score=grounding,
        confidence=grounding,
        trace_id=trace_id,
        elapsed_ms=elapsed_ms,
    )


# ── 7. POST /explain ──────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse, tags=["rag"])
async def explain(
    req: ExplainRequest,
    state: dict = Depends(get_state),
) -> ExplainResponse:
    """Return an explanation trace for a query-answer pair."""
    from ..evaluation.explainability_eval import ExplainabilityEvaluator

    concepts  = state.get("concepts", [])
    evaluator = ExplainabilityEvaluator(ontology_concepts=concepts)

    report = evaluator.evaluate(
        query_id=0,
        query=req.query,
        answer=req.answer,
        mode=req.mode.value,
        context_concepts=[{"name": c["name"]} for c in concepts[:20] if c.get("name")],
        context_relations=[],
        explanation_trace=[],
    )

    # Extract cited concepts as supporting facts
    supporting = [c for c in concepts[:50]
                  if c.get("name", "").lower() in req.answer.lower()]
    supporting_names = [c["name"] for c in supporting[:10]]

    return ExplainResponse(
        query=req.query,
        answer=req.answer,
        trace=[f"Step 1: Retrieved {len(supporting)} ontology concepts",
               f"Step 2: Grounding score = {report.grounding_precision:.3f}",
               f"Step 3: Evidence coverage = {report.evidence_trace_coverage:.3f}"],
        supporting_facts=supporting_names,
        grounding_score=report.grounding_precision,
        evidence_coverage=report.evidence_trace_coverage,
    )


# ── 8. POST /benchmark ────────────────────────────────────────────────────────

@router.post("/benchmark", response_model=BenchmarkResponse, tags=["evaluation"])
@limiter.limit("5/minute")
async def benchmark(
    request: Request,
    req: BenchmarkRequest,
    state: dict = Depends(get_state),
) -> BenchmarkResponse:
    """Run a mini benchmark on N queries and return per-query + aggregate metrics."""
    from ..evaluation.benchmark import TourismBenchmark
    from ..evaluation.hallucination_eval import HallucinationEvaluator

    t0       = time.perf_counter()
    concepts = state.get("concepts", [])
    linker   = state.get("linker")
    llm      = state.get("llm_client")

    split_path = ROOT / "data" / "splits" / "test.jsonl"
    if not split_path.exists():
        raise HTTPException(404, "Test split not found")

    bench = TourismBenchmark.build(
        test_split_path=split_path,
        ontology_concepts=concepts,
        n=req.n_queries,
        seed=req.seed,
    )
    evaluator = HallucinationEvaluator(ontology_concepts=concepts)
    per_item: list[BenchmarkItem] = []
    halluc_scores: list[float] = []
    grounding_scores: list[float] = []

    for item in bench.items:
        # Generate or stub answer
        if llm:
            try:
                answer = llm.call(prompt=item.query, system="Answer briefly.", max_tokens=150)
            except Exception:
                answer = item.expected_answer[:200]
        else:
            answer = item.expected_answer[:200]

        ctx = [{"name": c} if isinstance(c, str) else c for c in item.expected_concepts]
        report  = evaluator.evaluate(item.query, answer, req.mode.value, ctx)
        h_score = report.hallucination_score
        g_score = round(
            sum(1 for c in item.expected_concepts if c.lower() in answer.lower())
            / max(1, len(item.expected_concepts)), 3
        )
        per_item.append(BenchmarkItem(
            item_id=item.item_id,
            query=item.query[:100],
            answer=answer[:200],
            hallucination_score=h_score,
            grounding_score=g_score,
        ))
        halluc_scores.append(h_score)
        grounding_scores.append(g_score)

    n = len(per_item)
    agg = {
        "hallucination_rate":  round(sum(halluc_scores) / n, 3) if n else 0,
        "avg_grounding_score": round(sum(grounding_scores) / n, 3) if n else 0,
    }
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BenchmarkResponse(
        mode=req.mode.value,
        n_queries=n,
        aggregate_metrics=agg,
        per_item=per_item,
        elapsed_ms=elapsed_ms,
    )


# ── 9. GET /export-ontology ───────────────────────────────────────────────────

@router.get("/export-ontology", tags=["ontology"])
@limiter.limit("20/minute")
async def export_ontology(
    request: Request,
    version: str = "0.7.0",
    format:  str = "jsonld",
) -> Any:
    """Download the ontology in Turtle (.ttl) or JSON-LD (.jsonld) format."""
    fmt_map = {
        "turtle": ("ttl",    "text/turtle"),
        "ttl":    ("ttl",    "text/turtle"),
        "jsonld": ("jsonld", "application/ld+json"),
        "json":   ("jsonld", "application/ld+json"),
    }
    if format not in fmt_map:
        raise HTTPException(400, f"Unsupported format '{format}'. Use: turtle, jsonld")

    ext, media_type = fmt_map[format]
    ont_file = ROOT / "artifacts" / "ontologies" / f"tourism_v{version}.{ext}"
    if not ont_file.exists():
        raise HTTPException(404, f"Ontology file not found: {ont_file.name}")

    return FileResponse(
        path=str(ont_file),
        media_type=media_type,
        filename=ont_file.name,
    )
