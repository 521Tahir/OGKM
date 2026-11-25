"""
Main CLI entry point for the Tourism Ontology + LLM System.

Usage:
    python -m src.main --help
    python -m src.main interactive                         # real-time query prompt
    python -m src.main query "What are the best beaches?"  # single query
    python -m src.main profile
    python -m src.main build-ontology --records 100
    python -m src.main validate-ontology
    python -m src.main visualize Hallasan Jeju             # subgraph PNG
    python -m src.main stats                               # system status
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.seed import set_seed

app = typer.Typer(
    name="tourism-ontology",
    help="Jeju Island Tourism — Ontology RAG Knowledge Management System",
    add_completion=False,
)
logger = get_logger("main")

_ONT_VERSION = os.environ.get("ONTOLOGY_VERSION", "0.7.0")


# ── 1. Interactive real-time query prompt ──────────────────────────────────────

@app.command()
def interactive(
    mode: str = typer.Option("onto_rag", help="Pipeline mode: no_rag|doc_rag|onto_rag|constrained"),
    log_level: str = typer.Option("WARNING", help="Logging level"),
):
    """Launch the real-time interactive Jeju tourism query prompt."""
    setup_logging(log_level=log_level)
    from src.cli.interactive import run
    run()


# ── 2. Single query ────────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Tourism question to answer"),
    mode: str     = typer.Option("onto_rag", help="Pipeline mode: no_rag|doc_rag|onto_rag|constrained"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieved concepts and grounding"),
    log_level: str = typer.Option("WARNING", help="Logging level"),
):
    """Answer a single Jeju tourism question using the ontology RAG pipeline."""
    setup_logging(log_level=log_level)

    from src.graph.graph_store import TourismGraphStore
    from src.retrieval.concept_linker import ConceptLinker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.constrained_generator import ConstrainedGenerator
    from src.utils.io import load_json

    ont_dir = PROJECT_ROOT / "artifacts" / "ontologies" / _ONT_VERSION
    if not ont_dir.exists():
        typer.echo(f"ERROR: Ontology {_ONT_VERSION} not found at {ont_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nLoading ontology v{_ONT_VERSION}...")
    graph   = TourismGraphStore.from_version(_ONT_VERSION)
    linker  = ConceptLinker(graph=graph)
    retriever = HybridRetriever(graph=graph)
    generator = ConstrainedGenerator(client=None, graph=graph)

    typer.echo(f"Query [{mode}]: {question}\n")
    t0 = time.perf_counter()

    linked   = linker.link(question, top_k=10) if mode != "no_rag" else []
    concepts = [lc["concept"] if isinstance(lc, dict) else str(lc) for lc in linked]
    context  = retriever.retrieve(question, mode=mode) if mode != "no_rag" else None
    result   = generator.generate(question, context=context, mode=mode)
    elapsed  = round((time.perf_counter() - t0) * 1000, 1)

    typer.echo("=" * 60)
    typer.echo(f"Answer:\n{result.answer}")
    typer.echo("=" * 60)

    if verbose:
        typer.echo(f"\nConcepts used:  {', '.join(concepts[:8]) or 'none'}")
        typer.echo(f"Violations:     {len(result.violations)}")
        typer.echo(f"Grounded:       {result.is_grounded}")
        typer.echo(f"Elapsed:        {elapsed} ms")


# ── 3. Data profiling ─────────────────────────────────────────────────────────

@app.command()
def profile(
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run data profiling on the Jeju tourism dataset."""
    setup_logging(log_level=log_level)
    set_seed(42)
    from experiments.exp001_data_profile.run_profile import main as run_exp1
    run_exp1()


# ── 4. Build ontology ─────────────────────────────────────────────────────────

@app.command("build-ontology")
def build_ontology(
    records:  int = typer.Option(100, help="Records to process (0=all)"),
    strategy: str = typer.Option("cot", help="Prompting strategy: naive|cot|got|metacognitive|unified"),
    version:  str = typer.Option("0.1.0", help="Ontology version to create"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Build the tourism ontology from dataset records using LLM extraction.

    Strategies:
      naive        — basic extraction prompt
      cot          — chain-of-thought (step-by-step reasoning)
      got          — graph-of-thought (graph-structured reasoning, PDF §6.4)
      metacognitive— self-evaluation of extracted concepts (PDF §6.5-6.6)
      unified      — full pipeline: CoT → GoT coherence → Metacognitive filter (PDF §6.8)
    """
    setup_logging(log_level=log_level)
    set_seed(42)

    from src.data.loaders import TourismDataLoader
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager
    from src.ontology.ontology_builder import OntologyBuilder
    from src.utils.io import save_json

    logger.info(f"Building ontology: records={records}, strategy={strategy}, version={version}")
    typer.echo(f"Building ontology v{version} using strategy '{strategy}' on {records} records...")

    loader      = TourismDataLoader("data/processed/cleaned_records.jsonl")
    all_records = loader.load()
    if records > 0:
        all_records = all_records[:records]

    client = LLMClient()
    pm     = PromptManager(prompts_root=PROJECT_ROOT / "prompts")

    # GoT and metacognitive strategies use specialised wrappers
    if strategy == "got":
        typer.echo("  Using Graph-of-Thought prompting (PDF §6.4) — CoT extract + GoT coherence check")
        builder = OntologyBuilder(client=client, prompt_manager=pm,
                                   concept_strategy="got", version=version)
    elif strategy == "metacognitive":
        typer.echo("  Using Metacognitive prompting (PDF §6.5-6.6) — CoT extract + confidence calibration")
        builder = OntologyBuilder(client=client, prompt_manager=pm,
                                   concept_strategy="metacognitive", version=version)
    elif strategy == "unified":
        typer.echo("  Using Unified pipeline (PDF §6.8) — CoT → GoT coherence → Metacognitive filter")
        builder = OntologyBuilder(client=client, prompt_manager=pm,
                                   concept_strategy="unified", version=version)
    else:
        builder = OntologyBuilder(client=client, prompt_manager=pm,
                                   concept_strategy=strategy, version=version)

    summary = builder.process_records(all_records)
    paths   = builder.save_ontology()

    save_json(
        {"summary": summary, "paths": paths},
        f"artifacts/runs/ontology_build_{version}.json",
    )
    typer.echo(f"\nOntology v{version} built:")
    typer.echo(f"  Concepts : {summary.get('total_concepts', '?')}")
    typer.echo(f"  Relations: {summary.get('total_relations', '?')}")
    typer.echo(f"  Turtle   : {paths.get('turtle', '?')}")
    typer.echo(f"  JSON-LD  : {paths.get('jsonld', '?')}")


# ── 5. Validate ontology ──────────────────────────────────────────────────────

@app.command("validate-ontology")
def validate_ontology(
    version:   str = typer.Option(_ONT_VERSION, help="Ontology version to validate"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run SHACL validation on a stored ontology version."""
    setup_logging(log_level=log_level)

    from src.ontology.shacl_validator import SHACLValidator
    from src.utils.io import load_json

    ont_path = PROJECT_ROOT / "artifacts" / "ontologies" / version
    if not ont_path.exists():
        typer.echo(f"ERROR: Ontology {version} not found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Validating ontology v{version}...")
    concepts  = load_json(ont_path / "concepts.json")
    relations = load_json(ont_path / "relations.json")
    rules     = load_json(ont_path / "rules.json")

    validator = SHACLValidator()
    report    = validator.validate_ontology(concepts, relations)
    violations = report.violations

    typer.echo(f"\nResults:")
    typer.echo(f"  Concepts:   {len(concepts):,}")
    typer.echo(f"  Relations:  {len(relations):,}")
    typer.echo(f"  Violations: {len(violations)}")
    typer.echo(f"  Status:     {'PASS ✓' if report.conforms else 'FAIL ✗'}")

    if violations:
        typer.echo("\nFirst 5 violations:")
        for v in violations[:5]:
            typer.echo(f"  - {v.message}")


# ── 6. Visualize knowledge graph ──────────────────────────────────────────────

@app.command()
def visualize(
    concepts:  list[str] = typer.Argument(..., help="Seed concept names (e.g. Hallasan Jeju)"),
    max_hops:  int       = typer.Option(2, help="Graph traversal depth"),
    output:    str       = typer.Option("", help="Output PNG path (auto if empty)"),
    html:      bool      = typer.Option(False, "--html", help="Also generate interactive HTML"),
    log_level: str       = typer.Option("WARNING", help="Logging level"),
):
    """Visualize an ontology subgraph around seed concepts."""
    setup_logging(log_level=log_level)

    from src.graph.graph_store import TourismGraphStore
    from src.graph.visualizer import GraphVisualizer

    typer.echo(f"Loading ontology v{_ONT_VERSION}...")
    graph = TourismGraphStore.from_version(_ONT_VERSION)
    viz   = GraphVisualizer(graph)

    # ASCII to terminal first
    typer.echo(viz.ascii_subgraph(list(concepts), max_hops=max_hops))

    # PNG
    out = output or str(PROJECT_ROOT / "artifacts" / "figures" /
                         f"subgraph_{'_'.join(concepts[:2])}.png")
    path = viz.plot_subgraph(list(concepts), max_hops=max_hops, output=out)
    if path:
        typer.echo(f"PNG saved → {path}")

    # Optional interactive HTML
    if html:
        html_path = viz.interactive_html(list(concepts), max_hops=max_hops)
        if html_path:
            typer.echo(f"HTML saved → {html_path}")


# ── 7. System stats ───────────────────────────────────────────────────────────

@app.command()
def stats(
    version:   str  = typer.Option(_ONT_VERSION, help="Ontology version"),
    log_level: str  = typer.Option("WARNING", help="Logging level"),
):
    """Show ontology statistics and system status."""
    setup_logging(log_level=log_level)

    from src.graph.graph_store import TourismGraphStore

    ont_dir = PROJECT_ROOT / "artifacts" / "ontologies" / version
    if not ont_dir.exists():
        typer.echo(f"Ontology {version} not found", err=True)
        raise typer.Exit(1)

    graph = TourismGraphStore.from_version(version)
    s     = graph.stats()

    typer.echo(f"\nOntology v{version} Statistics")
    typer.echo("=" * 40)
    typer.echo(f"  Concepts    : {s['n_concepts']:,}")
    typer.echo(f"  Relations   : {s['n_relations']:,}")
    typer.echo(f"  Classes     : {s['n_classes']}")
    typer.echo(f"  Predicates  : {s['n_predicates']}")
    typer.echo("\n  Top concept classes:")
    for cls, cnt in sorted(s["class_counts"].items(), key=lambda x: x[1], reverse=True)[:8]:
        bar = "█" * min(30, cnt // max(1, s["n_concepts"] // 30))
        typer.echo(f"    {cls:<20} {cnt:>5}  {bar}")


if __name__ == "__main__":
    app()
