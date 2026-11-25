"""
Knowledge Graph Visualizer for the Tourism Ontology.

Generates:
  1. Static PNG/SVG subgraph plots (matplotlib + networkx)
  2. Interactive HTML (pyvis)  — opens in browser
  3. ASCII art console view    — for terminal display without GUI

Usage:
    viz = GraphVisualizer(graph_store)
    viz.plot_subgraph(["Hallasan", "Jeju"], output="artifacts/figures/subgraph.png")
    viz.plot_class_distribution(output="artifacts/figures/class_dist.png")
    viz.ascii_subgraph(["Hallasan"], max_hops=1)
    html_path = viz.interactive_html(["Hallasan", "Jeju"])
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("graph.visualizer")

# Colour palette per concept class
_CLASS_COLOURS = {
    "Attraction":     "#2196F3",   # blue
    "Restaurant":     "#FF5722",   # deep orange
    "Hotel":          "#9C27B0",   # purple
    "Transportation": "#607D8B",   # blue-grey
    "Activity":       "#4CAF50",   # green
    "Culture":        "#FF9800",   # orange
    "Nature":         "#8BC34A",   # light green
    "Food":           "#F44336",   # red
    "Event":          "#E91E63",   # pink
    "Service":        "#00BCD4",   # cyan
    "Destination":    "#3F51B5",   # indigo
    "Accommodation":  "#795548",   # brown
}
_DEFAULT_COLOUR = "#9E9E9E"


class GraphVisualizer:
    """Visualize subgraphs of the tourism ontology."""

    def __init__(self, graph_store) -> None:
        """
        Args:
            graph_store: TourismGraphStore instance (or any object with
                         extract_subgraph(), search_concepts(), stats() methods)
        """
        self.graph = graph_store

    # ── Static plot ───────────────────────────────────────────────────────────

    def plot_subgraph(
        self,
        seed_names: list[str],
        max_hops:   int = 2,
        max_nodes:  int = 40,
        output:     Optional[str] = None,
        title:      str = "Tourism Knowledge Subgraph",
    ) -> Optional[str]:
        """
        Plot a subgraph around seed concepts as a static PNG.

        Returns the output path, or None if matplotlib/networkx are unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.warning("matplotlib/networkx not installed — skipping plot")
            return None

        sg    = self.graph.extract_subgraph(seed_names, max_hops=max_hops, max_nodes=max_nodes)
        nodes = sg.get("concepts", [])
        edges = sg.get("relations", [])

        if not nodes:
            logger.warning(f"No nodes found for seeds: {seed_names}")
            return None

        G = nx.DiGraph()
        for c in nodes:
            G.add_node(
                c.get("name", ""),
                concept_class=c.get("concept_class", "Other"),
                confidence=c.get("confidence", 0.7),
            )
        for r in edges:
            s, o, p = r.get("subject", ""), r.get("object", ""), r.get("predicate", "")
            if G.has_node(s) and G.has_node(o):
                G.add_edge(s, o, predicate=p)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2.0)
        node_colours = [
            _CLASS_COLOURS.get(G.nodes[n].get("concept_class", ""), _DEFAULT_COLOUR)
            for n in G.nodes
        ]
        seeds_set = {s.lower() for s in seed_names}
        node_sizes = [
            800 if n.lower() in seeds_set else 400
            for n in G.nodes
        ]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx(
            G, pos, ax=ax,
            node_color=node_colours,
            node_size=node_sizes,
            font_size=7,
            edge_color="#BBBBBB",
            arrows=True,
            arrowsize=10,
        )
        edge_labels = {(u, v): d.get("predicate", "")[:12] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, ax=ax)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=col, markersize=10, label=cls)
            for cls, col in _CLASS_COLOURS.items()
            if any(G.nodes[n].get("concept_class") == cls for n in G.nodes)
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2)
        ax.set_title(
            f"{title}\n{len(G.nodes)} concepts, {len(G.edges)} relations "
            f"| seeds: {', '.join(seed_names[:3])}",
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()

        out = output or f"artifacts/figures/subgraph_{'_'.join(seed_names[:2])}.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Subgraph saved → {out}")
        return out

    # ── Class distribution bar chart ──────────────────────────────────────────

    def plot_class_distribution(
        self,
        output: Optional[str] = None,
        title: str = "Ontology Concept Class Distribution",
    ) -> Optional[str]:
        """Bar chart of concepts per class."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping")
            return None

        stats  = self.graph.stats()
        counts = stats.get("class_counts", {})
        if not counts:
            logger.warning("No class counts available")
            return None

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        classes = [k for k, _ in sorted_items]
        values  = [v for _, v in sorted_items]
        colours = [_CLASS_COLOURS.get(c, _DEFAULT_COLOUR) for c in classes]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(classes, values, color=colours, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=8)
        ax.set_title(
            f"{title}\nTotal: {stats.get('n_concepts', 0):,} concepts | "
            f"{stats.get('n_relations', 0):,} relations",
            fontsize=11
        )
        ax.set_xlabel("Concept Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()

        out = output or "artifacts/figures/class_distribution.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Class distribution saved → {out}")
        return out

    # ── Interactive HTML (pyvis) ──────────────────────────────────────────────

    def interactive_html(
        self,
        seed_names: list[str],
        max_hops:   int = 2,
        max_nodes:  int = 60,
        output:     Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate an interactive HTML graph using pyvis.

        Returns the output path. Opens in default browser if pyvis is installed.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed — run: pip install pyvis")
            return None

        sg    = self.graph.extract_subgraph(seed_names, max_hops=max_hops, max_nodes=max_nodes)
        nodes = sg.get("concepts", [])
        edges = sg.get("relations", [])

        if not nodes:
            return None

        net = Network(height="750px", width="100%", directed=True,
                      bgcolor="#1a1a2e", font_color="white")
        net.set_options("""
        {
          "physics": {"stabilization": {"iterations": 100}},
          "nodes": {"borderWidth": 2},
          "edges": {"smooth": {"type": "continuous"}}
        }
        """)

        seeds_set = {s.lower() for s in seed_names}
        for c in nodes:
            name = c.get("name", "")
            cls  = c.get("concept_class", "Other")
            col  = _CLASS_COLOURS.get(cls, _DEFAULT_COLOUR)
            size = 30 if name.lower() in seeds_set else 15
            net.add_node(
                name,
                label=name[:20],
                title=f"{name}\nClass: {cls}\nConf: {c.get('confidence', '?')}",
                color=col,
                size=size,
            )

        for r in edges:
            s, o = r.get("subject", ""), r.get("object", "")
            if s and o:
                net.add_edge(s, o, title=r.get("predicate", ""), label=r.get("predicate", "")[:10])

        out = output or "artifacts/figures/knowledge_graph.html"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(out)
        logger.info(f"Interactive graph saved → {out}")
        return out

    # ── ASCII console view ────────────────────────────────────────────────────

    def ascii_subgraph(
        self,
        seed_names: list[str],
        max_hops: int = 1,
        max_nodes: int = 20,
    ) -> str:
        """
        Return a text-art representation of a subgraph for terminal display.

        Example:
            Hallasan [Nature/Attraction]
            ├─[locatedIn]──► Jeju Island [Destination]
            ├─[hasActivity]─► Hiking [Activity]
            └─[nearBy]──────► Eorimok Trail [Nature]
        """
        sg    = self.graph.extract_subgraph(seed_names, max_hops=max_hops, max_nodes=max_nodes)
        nodes = {c.get("name", ""): c for c in sg.get("concepts", [])}
        edges = sg.get("relations", [])

        # Group edges by subject
        from collections import defaultdict
        by_subject: dict[str, list[dict]] = defaultdict(list)
        for r in edges:
            by_subject[r.get("subject", "")].append(r)

        lines: list[str] = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  Knowledge Subgraph — seeds: {', '.join(seed_names)}")
        lines.append(f"  {len(nodes)} concepts  |  {len(edges)} relations")
        lines.append(f"{'='*60}")

        def _node_label(name: str) -> str:
            c = nodes.get(name, {})
            cls = c.get("concept_class", "?")
            return f"{name} [{cls}]"

        rendered: set[str] = set()

        def _render_node(name: str, prefix: str = "", is_seed: bool = False) -> None:
            if name in rendered:
                return
            rendered.add(name)
            marker = "★ " if is_seed else ""
            lines.append(f"{prefix}{marker}{_node_label(name)}")
            rels = by_subject.get(name, [])
            for i, r in enumerate(rels[:8]):
                obj = r.get("object", "")
                pred = r.get("predicate", "?")
                is_last = (i == len(rels[:8]) - 1)
                branch = "└─" if is_last else "├─"
                lines.append(f"{prefix}  {branch}[{pred}]──► {_node_label(obj)}")

        for seed in seed_names:
            _render_node(seed, prefix="", is_seed=True)

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
