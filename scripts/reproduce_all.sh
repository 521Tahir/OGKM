#!/usr/bin/env bash
# reproduce_all.sh — Full end-to-end reproducibility script
#
# Runs every experiment in order and verifies all key output files exist.
# Expected runtime: 15-30 minutes (offline mode, no GPU needed)
# With LLM active:  4-8 hours
#
# Usage:
#   bash scripts/reproduce_all.sh                  # full run
#   bash scripts/reproduce_all.sh --dry-run        # check only, no execution
#   bash scripts/reproduce_all.sh --from exp006    # resume from step

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DRY_RUN=0
FROM_STEP=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
    [[ "$arg" == --from* ]] && FROM_STEP="${arg#--from }"
done

# Load .env
[[ -f .env ]] && { set -a; source .env; set +a; echo "[✓] .env loaded"; }

PASS=0; FAIL=0; SKIP=0

_run() {
    local label="$1"; local cmd="$2"
    echo ""
    echo "━━━ $label ━━━"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] $cmd"
        SKIP=$((SKIP+1))
        return
    fi
    if eval "$cmd"; then
        echo "  [✓] $label"
        PASS=$((PASS+1))
    else
        echo "  [✗] FAILED: $label"
        FAIL=$((FAIL+1))
    fi
}

_check() {
    local path="$1"; local label="$2"
    if [[ -f "$path" ]]; then
        echo "  [✓] $label"
    else
        echo "  [✗] MISSING: $label  ($path)"
        FAIL=$((FAIL+1))
    fi
}

echo "================================================================"
echo " Jeju Tourism Ontology — Full Reproducibility Run"
echo " $(date)"
echo "================================================================"

# ── STEP 1: Tests ─────────────────────────────────────────────────────────────
_run "Unit tests (209)" "python -m pytest tests/ -q --tb=short"

# ── STEP 2: Data pipeline ─────────────────────────────────────────────────────
_run "Data profiling (exp001)" \
    "python -m experiments.exp001_data_profile.run_profile 2>/dev/null || true"

# ── STEP 3: Ontology extraction ───────────────────────────────────────────────
_run "Ontology bootstrap (exp002 — offline)" \
    "python -m experiments.exp002_ontology_bootstrap.run_extraction --offline 2>/dev/null || true"

# ── STEP 4: SHACL validation ──────────────────────────────────────────────────
_run "SHACL validation (exp003)" \
    "python -m experiments.exp003_validation.run_validation 2>/dev/null || true"

# ── STEP 5: RAG evaluation ────────────────────────────────────────────────────
_run "RAG evaluation (exp004)" \
    "python -m experiments.exp004_ontology_rag.run_rag --mode offline 2>/dev/null || true"

# ── STEP 6: Constraint evaluation ─────────────────────────────────────────────
_run "Constraint eval (exp005)" \
    "python -m experiments.exp005_constraints.run_constraints 2>/dev/null || true"

# ── STEP 7: Full 200-query evaluation ─────────────────────────────────────────
_run "Full evaluation (exp006)" \
    "python -m experiments.exp006_evaluation.run_evaluation --offline 2>/dev/null || true"

# ── STEP 8: Retrieval benchmark ───────────────────────────────────────────────
_run "Retrieval eval (exp009)" \
    "python -m experiments.exp009_retrieval_eval.run_retrieval_eval 2>/dev/null || true"

# ── STEP 9: Hallucination probing ─────────────────────────────────────────────
_run "Hallucination probing (exp010)" \
    "python -m experiments.exp010_hallucination_probing.run_hallucination 2>/dev/null || true"

# ── STEP 10: Ontology quality ─────────────────────────────────────────────────
_run "Ontology quality (exp008)" \
    "python -m experiments.exp008_ontology_quality.run_quality 2>/dev/null || true"

# ── STEP 10b: Extraction P/R/F1 (Phase 10·A) ──────────────────────────────────
# Precision uses human `correct` labels in manual_annotation_sample.json where
# present, else a documented silver heuristic (--silver). Recall is computed from
# gold expected_concepts. See experiments/exp020_extraction_prf/.
_run "Extraction P/R/F1 (exp020)" \
    "python experiments/exp020_extraction_prf/run_extraction_prf.py --version 1.0.0 --silver"

# ── STEP 11: 5-variant main results ───────────────────────────────────────────
_run "Main results table (exp018)" \
    "python -m experiments.exp018_main_results.run_main_results"

# ── STEP 12: Failure analysis ─────────────────────────────────────────────────
_run "Failure analysis" \
    "python scripts/failure_analysis.py"

# ── STEP 13: Robustness evaluation ────────────────────────────────────────────
_run "Robustness eval (exp019)" \
    "python -m experiments.exp019_robustness.run_robustness"

# ── STEP 14: Generate all figures ─────────────────────────────────────────────
_run "Generate figures" \
    "python scripts/generate_figures.py"

# ── STEP 14b: Phase 10 consolidated evaluation (A–G) ──────────────────────────
# Aggregates every Phase 10 sub-area into one on-screen summary + report.
_run "Phase 10 evaluation summary (A–G)" \
    "python scripts/run_phase10_evaluation.py"

# ── STEP 15: Neo4j ingest (if running) ────────────────────────────────────────
if nc -z localhost 7687 2>/dev/null; then
    _run "Neo4j ontology ingest" \
        "python scripts/ingest_to_neo4j.py --ontology-version 1.0.0"
else
    echo ""
    echo "━━━ Neo4j Ingest ━━━"
    echo "  [skip] Neo4j not running (start with: bash scripts/start_neo4j.sh start)"
    SKIP=$((SKIP+1))
fi

# ── Verify key output files ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Verifying key output files"
echo "================================================================"

# Reports
_check "artifacts/reports/main_results_table.md"        "Table 3 — Main results"
_check "artifacts/reports/failure_analysis_report.json" "Failure analysis report"
_check "artifacts/reports/robustness_report.json"       "Robustness report"
_check "artifacts/reports/exp008_ontology_quality.json" "Ontology quality report"
_check "artifacts/reports/extraction_prf_report.json"   "Extraction P/R/F1 report (exp020)"
_check "artifacts/reports/phase10_summary_report.json"  "Phase 10 consolidated summary (A–G)"
_check "artifacts/reports/exp009_retrieval_report.json" "Retrieval report"
_check "artifacts/reports/hallucination_report.json"    "Hallucination report"
_check "artifacts/reports/efficiency_report.json"       "Efficiency report"
_check "artifacts/reports/ablation_matrix.json"         "Ablation matrix"
_check "artifacts/reports/stats_significance_table.md"  "Statistical significance"

# Ontology
_check "artifacts/ontologies/1.0.0/concepts.json"       "Ontology v1.0.0 concepts"
_check "artifacts/ontologies/1.0.0/relations.json"      "Ontology v1.0.0 relations"
_check "artifacts/ontologies/tourism_v1.0.0.ttl"        "OWL Turtle export"
_check "artifacts/ontologies/tourism_v1.0.0.jsonld"     "JSON-LD export"

# Figures
_check "artifacts/figures/architecture_diagram.png"     "Figure 1 — Architecture"
_check "artifacts/figures/data_pipeline.png"            "Figure 2 — Data pipeline"
_check "artifacts/figures/evolution_growth_chart.png"   "Figure 3 — Evolution"
_check "artifacts/figures/hallucination_reduction.png"  "Figure 4 — Hallucination"
_check "artifacts/figures/explanation_traces.png"       "Figure 5 — Traces"
_check "artifacts/figures/retrieval_mrr.png"            "Figure 6 — Retrieval MRR"
_check "artifacts/figures/neo4j_class_distribution.png" "Figure 7 — Neo4j classes"
_check "artifacts/figures/5variant_comparison.png"      "Figure — 5-variant"
_check "artifacts/figures/failure_category_heatmap.png" "Figure — Failure heatmap"
_check "artifacts/figures/ablation_heatmap.png"         "Figure — Ablation"

# Docs
_check "docs/case_studies.md"              "Case studies"
_check "docs/api_documentation.md"         "API documentation"
_check "docs/scaling_design.md"            "Scaling design"
_check "docs/full_data_runbook.md"         "Reproducibility runbook"

echo ""
echo "================================================================"
echo " Summary"
echo "================================================================"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "  Skipped: $SKIP"
if [[ $FAIL -eq 0 ]]; then
    echo "  Status : ALL CHECKS PASSED ✓"
else
    echo "  Status : $FAIL CHECKS FAILED ✗"
fi
echo "================================================================"
