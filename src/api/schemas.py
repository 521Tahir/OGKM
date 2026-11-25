"""
Pydantic request/response schemas for the Tourism Ontology RAG API.

All models use Pydantic v2 syntax.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class PipelineMode(str, Enum):
    no_rag      = "no_rag"
    doc_rag     = "doc_rag"
    onto_rag    = "onto_rag"
    constrained = "constrained"


class ExportFormat(str, Enum):
    turtle  = "turtle"
    jsonld  = "jsonld"
    json    = "json"


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:           str = "ok"
    ontology_version: str
    model:            str
    gpu_available:    bool
    n_concepts:       int
    n_relations:      int


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:            str  = Field(..., min_length=3, max_length=1000)
    mode:             PipelineMode = PipelineMode.onto_rag
    ontology_version: str  = "0.7.0"
    max_tokens:       int  = Field(default=512, ge=16, le=2048)
    temperature:      float = Field(default=0.0, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    query:           str
    mode:            str
    answer:          str
    used_concepts:   list[str]       = []
    violations:      list[str]       = []
    grounding_score: float           = 0.0
    confidence:      float           = 0.0
    trace_id:        str             = ""
    elapsed_ms:      float           = 0.0


# ── Explain ────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    query:  str = Field(..., min_length=3)
    answer: str = Field(..., min_length=1)
    mode:   PipelineMode = PipelineMode.onto_rag

class ExplainResponse(BaseModel):
    query:            str
    answer:           str
    trace:            list[str]   = []
    supporting_facts: list[str]   = []
    grounding_score:  float       = 0.0
    evidence_coverage: float      = 0.0


# ── Benchmark ─────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    n_queries: int = Field(default=50, ge=1, le=500)
    mode:      PipelineMode = PipelineMode.onto_rag
    seed:      int = 42

class BenchmarkItem(BaseModel):
    item_id:           int
    query:             str
    answer:            str
    hallucination_score: float
    grounding_score:   float

class BenchmarkResponse(BaseModel):
    mode:             str
    n_queries:        int
    aggregate_metrics: dict[str, float]
    per_item:         list[BenchmarkItem]
    elapsed_ms:       float


# ── Ontology management ────────────────────────────────────────────────────────

class OntologyUpdateRequest(BaseModel):
    records:          list[dict]    = Field(..., min_length=1)
    policy:           str           = "priority"    # priority | merge | flag
    ontology_version: str           = "0.7.0"

class OntologyUpdateResponse(BaseModel):
    new_version:       str
    concepts_added:    int
    concepts_merged:   int
    concepts_flagged:  int
    conflicts_detected: int

class OntologyExportRequest(BaseModel):
    version: str          = "0.7.0"
    format:  ExportFormat = ExportFormat.jsonld

class ValidationRequest(BaseModel):
    ontology_version: str = "0.7.0"

class ValidationResponse(BaseModel):
    ontology_version: str
    n_violations:     int
    violations:       list[dict]
    passed:           bool


# ── Data profiling ─────────────────────────────────────────────────────────────

class ProfileDataRequest(BaseModel):
    split: str = "test"   # train | val | test

class ProfileDataResponse(BaseModel):
    split:          str
    n_records:      int
    avg_input_len:  float
    avg_output_len: float
    top_tokens:     list[str]
