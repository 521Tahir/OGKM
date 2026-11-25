"""
Real-Time Interactive Query CLI — Jeju Island Tourism Ontology RAG System

Loads the ontology and (optionally) the LLM at startup, then enters a
prompt loop where the user can ask any tourism question and receive
a grounded, ontology-backed answer.

Features:
  - 4 pipeline modes: no_rag | doc_rag | onto_rag | constrained
  - Shows retrieved concepts, grounding score, violations
  - /explain   — show reasoning trace for last answer
  - /mode      — switch pipeline mode
  - /graph     — visualize subgraph for last query (ASCII + optional PNG)
  - /history   — show last N queries
  - /concepts  — search ontology for a term
  - /stats     — show system status
  - /help      — show all commands
  - /quit      — exit

Usage:
    # Without LLM (fast, offline responses):
    python -m src.cli.interactive

    # With real Llama 3.1-8B (requires LOAD_LLM=1 and HF_TOKEN):
    LOAD_LLM=1 CUDA_VISIBLE_DEVICES=5,6 python -m src.cli.interactive
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ..utils.logging_utils import get_logger, setup_logging
from ..utils.io import load_json

logger = get_logger("cli.interactive")
console = Console()

_ONTOLOGY_VERSION = os.environ.get("ONTOLOGY_VERSION", "0.7.0")
_LOAD_LLM         = os.environ.get("LOAD_LLM", "0") == "1"
_MODES            = ["no_rag", "doc_rag", "onto_rag", "constrained"]

_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       Jeju Island Tourism — Ontology RAG Query System           ║
║       Ontology v{version:<8}  |  {n_concepts:>6,} concepts              ║
║       Model: {model:<20}                              ║
╚══════════════════════════════════════════════════════════════════╝
  Type your question, or use a command:
    /mode [no_rag|doc_rag|onto_rag|constrained]
    /explain   /graph   /concepts <term>   /feedback approve|reject <concept>   /stats   /history   /help   /quit
"""

_HELP_TEXT = """
[bold cyan]Commands[/bold cyan]
  [yellow]/mode[/yellow] [no_rag|doc_rag|onto_rag|constrained]
         Switch the pipeline mode.
         • no_rag      — LLM only, no retrieval
         • doc_rag     — retrieve text passages
         • onto_rag    — retrieve ontology concepts (recommended)
         • constrained — onto_rag + strict entity constraints

  [yellow]/explain[/yellow]
         Show the reasoning trace for the last answer.

  [yellow]/graph[/yellow] [save]
         Print an ASCII subgraph for the last query.
         Add 'save' to also save a PNG to artifacts/figures/.

  [yellow]/concepts[/yellow] <term>
         Search the ontology for concepts matching <term>.

  [yellow]/stats[/yellow]
         Show system status (ontology size, model, GPU).

  [yellow]/history[/yellow] [N]
         Show the last N query-answer pairs (default 5).

  [yellow]/clear[/yellow]
         Clear the screen.

  [yellow]/quit[/yellow] (or /exit, Ctrl+C)
         Exit the system.

[bold cyan]Pipeline Modes Explained[/bold cyan]
  no_rag      → Raw LLM answer, no knowledge grounding. Fast but may hallucinate.
  doc_rag     → Retrieved document passages + LLM. Better factual grounding.
  onto_rag    → Ontology subgraph + LLM. Semantically precise, recommended.
  constrained → Strictest: only verified ontology entities allowed in answers.
"""


# ── Session state ─────────────────────────────────────────────────────────────

class Session:
    def __init__(self) -> None:
        self.mode:          str = "onto_rag"
        self.history:       list[dict] = []
        self.last_concepts: list[str]  = []
        self.last_answer:   str        = ""
        self.last_query:    str        = ""
        self.last_grounding: float     = 0.0

    def add(self, query: str, answer: str, concepts: list[str], grounding: float) -> None:
        self.last_query    = query
        self.last_answer   = answer
        self.last_concepts = concepts
        self.last_grounding = grounding
        self.history.append({
            "query":    query,
            "answer":   answer[:300],
            "mode":     self.mode,
            "concepts": concepts,
            "grounding": grounding,
        })


# ── System loader ─────────────────────────────────────────────────────────────

class TourismSystem:
    """Loads and holds all pipeline components."""

    def __init__(self) -> None:
        self.graph:      Optional[object] = None
        self.linker:     Optional[object] = None
        self.generator:  Optional[object] = None
        self.retriever:  Optional[object] = None
        self.llm:        Optional[object] = None
        self.concepts:   list[dict]       = []
        self.ont_version: str             = _ONTOLOGY_VERSION
        self.model_name:  str             = "none"
        self.n_concepts:  int             = 0
        self.n_relations: int             = 0
        self.gpu_info:    str             = "N/A"

    def load(self) -> None:
        """Load ontology, linker, retriever, and (optionally) LLM."""
        with console.status("[bold green]Loading ontology..."):
            self._load_ontology()

        with console.status("[bold green]Initialising concept linker..."):
            self._load_linker()

        with console.status("[bold green]Initialising retriever and generator..."):
            self._load_pipeline()

        if _LOAD_LLM:
            with console.status("[bold yellow]Loading Llama 3.1-8B (4-bit)... this takes ~30s"):
                self._load_llm()
        else:
            console.print("[dim]LLM not loaded (set LOAD_LLM=1 to enable). Using offline mode.[/dim]")

    def _load_ontology(self) -> None:
        from src.graph.graph_store import TourismGraphStore
        ont_dir  = ROOT / "artifacts" / "ontologies" / self.ont_version
        if not ont_dir.exists():
            console.print(f"[red]Ontology not found: {ont_dir}[/red]")
            return
        self.graph    = TourismGraphStore.from_version(self.ont_version)
        self.concepts = load_json(ont_dir / "concepts.json")
        rels          = load_json(ont_dir / "relations.json")
        self.n_concepts  = len(self.concepts)
        self.n_relations = len(rels)

    def _load_linker(self) -> None:
        if self.graph is None:
            return
        from src.retrieval.concept_linker import ConceptLinker
        self.linker = ConceptLinker(graph=self.graph)

    def _load_pipeline(self) -> None:
        if self.graph is None:
            return
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.generation.constrained_generator import ConstrainedGenerator
        self.retriever = HybridRetriever(graph=self.graph)
        self.generator = ConstrainedGenerator(client=None, graph=self.graph)

    def _load_llm(self) -> None:
        try:
            from src.llm.local_client import LocalLLMClient
            cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
            gpu_ids = [int(g) for g in cuda.split(",")]
            self.llm = LocalLLMClient(
                model_key="llama3.1-8b",
                gpu_ids=gpu_ids,
                load_in_4bit=True,
                hf_token=os.environ.get("HF_TOKEN"),
            )
            self.model_name = "llama3.1-8b"
            self.gpu_info   = f"GPU {cuda}"

            # Reconnect generator with LLM
            if self.generator:
                self.generator.client = self.llm
        except Exception as exc:
            console.print(f"[red]LLM load failed: {exc}[/red]")
            self.llm = None

    def query(self, q: str, mode: str) -> dict:
        """Run the full RAG pipeline and return a result dict."""
        t0 = time.perf_counter()

        # Step 1: Link concepts from query
        used_concepts: list[str] = []
        if self.linker and mode != "no_rag":
            try:
                linked = self.linker.link(q, top_k=10)
                used_concepts = []
                for lc in linked:
                    if isinstance(lc, dict):
                        c = lc.get("concept", lc.get("name", ""))
                        used_concepts.append(c.get("name", str(c)) if isinstance(c, dict) else str(c))
                    else:
                        used_concepts.append(str(lc))
            except Exception as exc:
                logger.warning(f"Linker error: {exc}")

        # Step 2: Retrieve context
        context = None
        if self.retriever and mode in ("doc_rag", "onto_rag", "constrained"):
            try:
                context = self.retriever.retrieve(q, mode=mode)
            except Exception as exc:
                logger.warning(f"Retriever error: {exc}")

        # Step 3: Generate answer
        answer   = ""
        violations: list[str] = []
        if self.generator:
            try:
                result    = self.generator.generate(q, context=context, mode=mode)
                answer    = result.answer
                violations = [v.to_dict().get("detail", "") for v in result.violations]
            except Exception as exc:
                logger.warning(f"Generator error: {exc}")

        # Offline fallback
        if not answer:
            answer = self._offline_answer(q, used_concepts, mode)

        # Step 4: Grounding score
        grounding = 0.0
        if used_concepts:
            mentioned = sum(1 for c in used_concepts if c.lower() in answer.lower())
            grounding = round(mentioned / len(used_concepts), 3)

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "query":       q,
            "answer":      answer,
            "mode":        mode,
            "concepts":    used_concepts,
            "violations":  violations,
            "grounding":   grounding,
            "elapsed_ms":  elapsed,
        }

    def _offline_answer(self, query: str, concepts: list[str], mode: str) -> str:
        """Structured offline fallback when LLM is not loaded."""
        q_lower = query.lower()
        prefix  = {
            "no_rag":      "[No RAG] ",
            "doc_rag":     "[Doc RAG] ",
            "onto_rag":    "[Onto RAG] ",
            "constrained": "[Constrained] ",
        }.get(mode, "")

        ctx = ""
        if concepts:
            ctx = f" Based on ontology concepts: {', '.join(concepts[:5])}."

        # Simple rule-based offline answers for Jeju tourism
        if any(w in q_lower for w in ["beach", "sea", "ocean", "swim"]):
            return (f"{prefix}Jeju Island has beautiful beaches including Hamdeok, Hyeopjae, "
                    f"and Jungmun Beach, known for clear waters and white sand.{ctx}")
        elif any(w in q_lower for w in ["hike", "mountain", "trail", "hallasan"]):
            return (f"{prefix}Hallasan National Park is Jeju's iconic volcano (1,950m). "
                    f"The Seongpanak and Gwaneumsa trails lead to the summit.{ctx}")
        elif any(w in q_lower for w in ["food", "eat", "restaurant", "pork", "seafood"]):
            return (f"{prefix}Must-try Jeju foods include Jeju Black Pork BBQ, haenyeo seafood, "
                    f"abalone porridge, and hallabong orange desserts.{ctx}")
        elif any(w in q_lower for w in ["hotel", "stay", "accommodation", "resort"]):
            return (f"{prefix}Jeju offers accommodation from luxury resorts at Jungmun to "
                    f"boutique guesthouses in Jeju City and Seogwipo.{ctx}")
        elif any(w in q_lower for w in ["transport", "bus", "taxi", "car", "rent"]):
            return (f"{prefix}Renting a car is the best way to explore Jeju. Public buses "
                    f"connect major attractions. Taxis and ride-sharing are available.{ctx}")
        elif any(w in q_lower for w in ["waterfall", "cheonjiyeon", "jeongbang", "cheonjeyeon"]):
            return (f"{prefix}Jeju has three famous waterfalls: Cheonjiyeon (천지연), "
                    f"Cheonjeyeon (천제연), and Jeongbang (정방) which falls directly into the sea.{ctx}")
        elif any(w in q_lower for w in ["cave", "manjang", "lava"]):
            return (f"{prefix}Manjanggul is one of the world's longest lava tube caves "
                    f"(7.4km), a UNESCO World Natural Heritage Site.{ctx}")
        elif any(w in q_lower for w in ["folk", "village", "culture", "tradition"]):
            return (f"{prefix}Jeju Folk Village Museum (성읍 민속마을) preserves traditional "
                    f"Jeju architecture, customs, and haenyeo (sea women) culture.{ctx}")
        elif any(w in q_lower for w in ["sunrise", "seongsan", "ilchulbong"]):
            return (f"{prefix}Seongsan Ilchulbong (Sunrise Peak) is a UNESCO site, famous "
                    f"for stunning sunrise views and the tuff cone crater.{ctx}")
        else:
            return (f"{prefix}Jeju Island (제주도) is South Korea's largest island, known for "
                    f"volcanic landscapes, beaches, seafood, and unique culture."
                    f"{ctx} [Set LOAD_LLM=1 for detailed LLM-powered answers.]")

    def search_concepts(self, term: str, top_k: int = 10) -> list[dict]:
        if self.graph is None:
            return []
        return self.graph.search_concepts(term, top_k=top_k)

    def ascii_graph(self, concepts: list[str]) -> str:
        if self.graph is None:
            return "(Graph not loaded)"
        from src.graph.visualizer import GraphVisualizer
        viz = GraphVisualizer(self.graph)
        return viz.ascii_subgraph(concepts, max_hops=1, max_nodes=20)

    def save_graph(self, concepts: list[str]) -> Optional[str]:
        if self.graph is None:
            return None
        from src.graph.visualizer import GraphVisualizer
        viz = GraphVisualizer(self.graph)
        out = ROOT / "artifacts" / "figures" / "query_subgraph.png"
        return viz.plot_subgraph(concepts, output=str(out))


# ── Display helpers ───────────────────────────────────────────────────────────

def _mode_colour(mode: str) -> str:
    return {
        "no_rag":      "dim",
        "doc_rag":     "blue",
        "onto_rag":    "green",
        "constrained": "magenta",
    }.get(mode, "white")


def _display_result(result: dict, session: Session) -> None:
    mode    = result["mode"]
    mc      = _mode_colour(mode)
    gs      = result["grounding"]
    gs_col  = "green" if gs >= 0.5 else "yellow" if gs >= 0.2 else "red"

    # Answer panel
    console.print(Panel(
        result["answer"],
        title=f"[bold {mc}][{mode.upper()}][/bold {mc}]  "
              f"Grounding: [{gs_col}]{gs:.1%}[/{gs_col}]  "
              f"Time: {result['elapsed_ms']:.0f}ms",
        border_style=mc,
        padding=(0, 1),
    ))

    # Concepts used
    if result["concepts"]:
        tags = "  ".join(
            f"[{mc}]#{c}[/{mc}]"
            for c in result["concepts"][:8]
        )
        console.print(f"  [dim]Concepts:[/dim] {tags}")

    # Violations
    if result["violations"]:
        console.print(
            f"  [red]⚠ {len(result['violations'])} constraint violation(s): "
            f"{result['violations'][0][:80]}[/red]"
        )


def _display_concepts(concepts: list[dict]) -> None:
    if not concepts:
        console.print("[yellow]No concepts found.[/yellow]")
        return
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Name",  style="white", min_width=20)
    table.add_column("Class", style="green", min_width=15)
    table.add_column("Conf",  style="yellow", justify="right", min_width=6)
    for c in concepts:
        prov = c.get("provenance", {})
        conf = prov.get("confidence", c.get("confidence", "?"))
        table.add_row(
            c.get("name", "?"),
            c.get("concept_class", "?"),
            f"{conf:.2f}" if isinstance(conf, float) else str(conf),
        )
    console.print(table)


def _display_history(history: list[dict], n: int = 5) -> None:
    for i, h in enumerate(history[-n:], 1):
        mc = _mode_colour(h["mode"])
        console.print(
            f"[dim]{i}.[/dim] [{mc}]{h['mode']}[/{mc}] "
            f"[bold]{h['query'][:60]}[/bold]"
        )
        console.print(f"   [dim]→ {h['answer'][:100]}...[/dim]")
        console.print(f"   [dim]Grounding: {h['grounding']:.1%} | "
                      f"Concepts: {len(h['concepts'])}[/dim]")
        console.print()


def _display_stats(sys: TourismSystem, session: Session) -> None:
    table = Table(box=box.ROUNDED, title="System Status", title_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value",    style="white")

    table.add_row("Ontology version",  sys.ont_version)
    table.add_row("Concepts",          f"{sys.n_concepts:,}")
    table.add_row("Relations",         f"{sys.n_relations:,}")
    table.add_row("LLM model",         sys.model_name)
    table.add_row("GPU",               sys.gpu_info)
    table.add_row("Current mode",      session.mode)
    table.add_row("Queries this session", str(len(session.history)))
    console.print(table)


# ── Expert feedback ───────────────────────────────────────────────────────────

def _record_expert_feedback(concept_name: str, action: str, sys_obj: "TourismSystem") -> None:
    """Write expert approve/reject to the review queue JSON (§8.4 expert feedback)."""
    import datetime
    queue_path = ROOT / "artifacts" / "reports" / "expert_review_queue.json"

    # Load existing queue
    try:
        import json
        raw = json.loads(queue_path.read_text()) if queue_path.exists() else []
        existing = raw if isinstance(raw, list) else raw.get("entries", [])
    except Exception:
        existing = []

    # Find if concept exists in ontology
    matched = sys_obj.graph.search_concepts(concept_name, top_k=1) if sys_obj.graph else []
    in_ontology = bool(matched)

    entry = {
        "concept":    concept_name,
        "action":     action,
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "in_ontology": in_ontology,
        "reviewer":   "interactive_user",
    }
    existing.append(entry)

    try:
        queue_path.write_text(__import__("json").dumps(existing, indent=2))
        status_color = "green" if action == "approve" else "red"
        console.print(
            f"[{status_color}]✓ Feedback recorded:[/{status_color}] "
            f"[bold]{action.upper()}[/bold] → '{concept_name}' "
            f"({'found in ontology' if in_ontology else 'not in ontology'})"
        )
    except Exception as e:
        console.print(f"[red]Failed to save feedback: {e}[/red]")


# ── Main REPL ─────────────────────────────────────────────────────────────────

def run() -> None:
    setup_logging("WARNING")  # suppress debug noise in interactive mode

    console.print("[bold green]Starting Tourism Ontology RAG System...[/bold green]")

    sys_obj = TourismSystem()
    sys_obj.load()
    session = Session()

    # Print banner
    banner = _BANNER.format(
        version=sys_obj.ont_version,
        n_concepts=sys_obj.n_concepts,
        model=sys_obj.model_name if _LOAD_LLM else "offline (set LOAD_LLM=1)",
    )
    console.print(banner, style="bold cyan")
    console.print(
        f"  Current mode: [green]{session.mode}[/green]  "
        f"(change with [yellow]/mode[/yellow])\n"
    )

    # ── REPL loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            raw = console.input("[bold cyan]You >[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                console.print(_HELP_TEXT)

            elif cmd == "/clear":
                console.clear()

            elif cmd == "/mode":
                if arg in _MODES:
                    session.mode = arg
                    console.print(f"[green]Mode switched to: {arg}[/green]")
                else:
                    console.print(
                        f"[yellow]Unknown mode '{arg}'. "
                        f"Choose: {', '.join(_MODES)}[/yellow]"
                    )

            elif cmd == "/explain":
                if not session.last_answer:
                    console.print("[yellow]No query yet. Ask something first.[/yellow]")
                else:
                    from src.evaluation.explainability_eval import ExplainabilityEvaluator
                    ev = ExplainabilityEvaluator(ontology_concepts=sys_obj.concepts)
                    ctx_c = [{"name": c} for c in session.last_concepts[:20]]
                    report = ev.evaluate(
                        query_id=0,
                        query=session.last_query,
                        answer=session.last_answer,
                        mode=session.mode,
                        context_concepts=ctx_c,
                        context_relations=[],
                        explanation_trace=[],
                    )
                    console.print(Panel(
                        f"[cyan]Query:[/cyan] {session.last_query}\n\n"
                        f"[cyan]Grounding precision:[/cyan]  {report.grounding_precision:.3f}\n"
                        f"[cyan]Evidence coverage:[/cyan]    {report.evidence_trace_coverage:.3f}\n"
                        f"[cyan]Supporting concepts:[/cyan]  "
                        f"{', '.join(session.last_concepts[:5]) or 'none'}\n\n"
                        f"[dim]Step 1: Retrieved {len(session.last_concepts)} ontology concepts\n"
                        f"Step 2: Grounding score = {session.last_grounding:.3f}\n"
                        f"Step 3: Answer generated with mode={session.mode}[/dim]",
                        title="[bold]Explanation Trace[/bold]",
                        border_style="cyan",
                    ))

            elif cmd == "/graph":
                if not session.last_concepts:
                    console.print("[yellow]No concepts from last query. Ask a question first.[/yellow]")
                else:
                    console.print(sys_obj.ascii_graph(session.last_concepts[:5]))
                    if "save" in arg:
                        path = sys_obj.save_graph(session.last_concepts[:5])
                        if path:
                            console.print(f"[green]PNG saved → {path}[/green]")

            elif cmd == "/concepts":
                if not arg:
                    console.print("[yellow]Usage: /concepts <search term>[/yellow]")
                else:
                    results = sys_obj.search_concepts(arg, top_k=10)
                    _display_concepts(results)

            elif cmd == "/stats":
                _display_stats(sys_obj, session)

            elif cmd == "/history":
                n = int(arg) if arg.isdigit() else 5
                if not session.history:
                    console.print("[yellow]No history yet.[/yellow]")
                else:
                    _display_history(session.history, n)

            elif cmd == "/feedback":
                # Expert feedback: /feedback approve <concept> | /feedback reject <concept>
                parts = arg.split(None, 1) if arg else []
                if len(parts) < 2 or parts[0] not in ("approve", "reject"):
                    console.print(
                        "[yellow]Usage: /feedback approve <concept name>\n"
                        "       /feedback reject  <concept name>[/yellow]"
                    )
                else:
                    action, concept_name = parts[0], parts[1].strip()
                    _record_expert_feedback(concept_name, action, sys_obj)

            else:
                console.print(f"[yellow]Unknown command '{cmd}'. Type /help.[/yellow]")

        else:
            # ── Query ──────────────────────────────────────────────────────────
            if len(raw) < 3:
                console.print("[yellow]Query too short. Please ask a complete question.[/yellow]")
                continue

            with console.status(f"[bold green]Querying [{session.mode}]..."):
                result = sys_obj.query(raw, session.mode)

            _display_result(result, session)
            session.add(raw, result["answer"], result["concepts"], result["grounding"])


def main() -> None:
    """Entry point for python -m src.cli.interactive"""
    run()


if __name__ == "__main__":
    main()
