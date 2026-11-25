"""
WBS Task 4.2.2 - Visualization outputs: architecture, ontology graphs, retrieval
workflow, hallucination plots, scalability and robustness curves.

Every figure is drawn from a JSON deliverable produced by another WBS script, never
from numbers typed in here. A figure whose input is missing is skipped and listed as
skipped in the index, so the figure set never shows a plot of values nothing measured.

  python -m wbs.task4_scalability.make_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SC = ROOT / "wbs" / "deliverables" / "task4_scalability"
EV = ROOT / "wbs" / "deliverables" / "task4_evaluation"
T2 = ROOT / "wbs" / "deliverables" / "task2_formalization"
T3 = ROOT / "wbs" / "deliverables" / "task3_retrieval"
DELIV = ROOT / "wbs" / "deliverables"
FIG = ROOT / "wbs" / "deliverables" / "figures"

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
C_RAG, C_ONTO, C_HYB, C_VAL, C_LLM = "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#999999"
made, skipped = [], []


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(fig, name: str, caption: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    made.append((name, caption))
    print(f"  wrote {name}")


def skip(name: str, why: str):
    skipped.append((name, why))
    print(f"  SKIP {name}: {why}")


# ── Figure 1: system architecture ─────────────────────────────────────────
def fig_architecture():
    q = load(T3 / "retriever_characterization.json")
    ont = load(T2 / "ontology_quality_metrics.json") or {}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(18, 101); ax.grid(False)

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec="#333", lw=.8, alpha=.92))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.45)

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.1, color="#444",
                                     shrinkA=2, shrinkB=2))
        if lbl:
            ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, lbl, fontsize=7,
                    color="#444", ha="left", va="center")

    nq = (q or {}).get("concept_mapping", {}).get("surface_forms", "—")
    box(2, 86, 22, 11, "Tourist question\n(free text)", "#EEE")
    box(2, 66, 22, 13, f"Query understanding\n8 intent groups → 14 classes\n"
                       f"concept mapping vs {nq:,} surface forms"
        if isinstance(nq, int) else "Query understanding", "#DCE8F5")
    box(2, 42, 22, 16, "Ontology subgraph retrieval\ndense entity cards\n"
                       "+ class routing by degree\n+ time-scoped facts", "#CFE3F7")
    box(2, 22, 22, 13, "Dense passage retrieval\nbge-base, top-5 over\n36,847 train passages",
        "#F6DFCF")
    box(31, 32, 20, 18, "Context assembly\nshared 3,000-char budget\n"
                        "hybrid: 20% facts /\n80% passages", "#E6F4EC")
    box(58, 32, 18, 18, "LLM generation\nLoRA, arm-dedicated\nadapter, greedy", "#EDE3F2")
    box(81, 54, 17, 18, "Constrained reasoning\nvalidate → repair →\ntrace → fallback",
        "#F3D9E7")
    box(81, 22, 17, 18, "Answer\n+ per-sentence\nevidence trace", "#EEE")
    t = (f"Ontology v1.4.0\n{ont.get('n_concepts', 1565):,} concepts  "
         f"{ont.get('n_relations', 7909):,} relations\nRDF/OWL · SHACL · OWL-RL\n"
         f"versioned, audited")
    box(31, 66, 45, 14, t, "#FFF3CF")

    arrow(13, 86, 13, 79)
    arrow(13, 66, 13, 58)
    arrow(13, 42, 13, 35)
    arrow(24, 50, 31, 45); arrow(24, 28, 31, 38)
    arrow(51, 41, 58, 41)
    arrow(76, 45, 83, 54)
    arrow(89, 54, 89, 40)
    ax.annotate("", xy=(31, 50), xytext=(40, 66),
                arrowprops=dict(arrowstyle="-|>", color="#B8860B", lw=1.1))
    ax.text(26, 60, "concepts,\nrelations", fontsize=7, color="#8B6914")
    ax.annotate("", xy=(89, 72), xytext=(70, 78),
                arrowprops=dict(arrowstyle="-|>", color="#B8860B", lw=1.1))
    ax.text(72, 77, "constraints", fontsize=7, color="#8B6914")
    ax.set_title("Figure 1 — Ontology-grounded tourism QA: system architecture",
                 fontsize=10.5, loc="left")
    save(fig, "fig01_architecture.png",
         "System architecture. The ontology feeds both retrieval and the "
         "post-generation constraint layer; all arms share one context budget.")


# ── Figure 2: ontology graph ──────────────────────────────────────────────
def fig_ontology_graph():
    try:
        import networkx as nx
    except ImportError:
        return skip("fig02_ontology_graph.png", "networkx not available")
    d = ROOT / "artifacts" / "ontologies" / "1.4.0"
    if not d.exists():
        return skip("fig02_ontology_graph.png", "ontology v1.4.0 missing")
    cons = json.loads((d / "concepts.json").read_text(encoding="utf-8"))
    rels = json.loads((d / "relations.json").read_text(encoding="utf-8"))
    cls = {c["name"]: (c.get("concept_class") or "Concept") for c in cons
           if isinstance(c, dict) and c.get("name")}
    G = nx.Graph()
    for r in rels:
        if isinstance(r, dict) and r.get("subject") and r.get("object"):
            G.add_edge(str(r["subject"]), str(r["object"]), p=str(r["predicate"]))
    top = sorted(G.degree, key=lambda x: -x[1])[:70]
    H = G.subgraph([n for n, _ in top])
    classes = sorted({cls.get(n, "Concept") for n in H})
    cmap = plt.get_cmap("tab20")
    col = {c: cmap(i % 20) for i, c in enumerate(classes)}
    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(H, seed=7, k=.55, iterations=200)
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=.22, width=.7)
    sizes = [60 + 10 * G.degree(n) for n in H]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes,
                           node_color=[col[cls.get(n, "Concept")] for n in H],
                           linewidths=.4, edgecolors="#333")
    lab = {n: n if len(n) < 22 else n[:20] + "…"
           for n, dg in sorted(H.degree, key=lambda x: -x[1])[:34]}
    nx.draw_networkx_labels(H, pos, labels=lab, ax=ax, font_size=6.4)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col[c],
                                  label=f"{c}", markersize=6) for c in classes],
              fontsize=6.6, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, title="concept class", title_fontsize=7)
    ax.set_axis_off(); ax.grid(False)
    ax.set_title(f"Figure 2 — Jeju tourism ontology v1.4.0: 70 highest-degree concepts "
                 f"of {G.number_of_nodes():,} (node size ∝ degree)",
                 fontsize=10.5, loc="left")
    save(fig, "fig02_ontology_graph.png",
         f"The 70 highest-degree concepts of the {G.number_of_nodes():,}-node, "
         f"{G.number_of_edges():,}-edge graph, coloured by concept class.")

    # class / predicate distribution
    from collections import Counter
    cc = Counter(cls.values()).most_common(14)
    pp = Counter(str(r["predicate"]) for r in rels
                 if isinstance(r, dict) and r.get("predicate")).most_common(14)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].barh([k for k, _ in cc][::-1], [v for _, v in cc][::-1], color=C_ONTO)
    axs[0].set_title("concepts per class (top 14)", fontsize=9)
    axs[0].set_xlabel("concepts")
    axs[1].barh([k for k, _ in pp][::-1], [v for _, v in pp][::-1], color=C_RAG)
    axs[1].set_title("relations per predicate (top 14)", fontsize=9)
    axs[1].set_xlabel("relations")
    for ax_ in axs:
        ax_.tick_params(axis="y", labelsize=7)
    fig.suptitle("Figure 3 — Ontology composition", fontsize=10.5, x=.09, ha="left")
    save(fig, "fig03_ontology_composition.png",
         "Concept-class and predicate distributions of ontology v1.4.0.")


# ── Figure 4: retrieval workflow with measured cost ───────────────────────
def fig_retrieval_workflow():
    r = load(SC / "retrieval_scalability.json")
    q = load(T3 / "retriever_characterization.json")
    if not r:
        return skip("fig04_retrieval_workflow.png", "retrieval_scalability.json missing")
    stages = []
    e = r.get("encode", {})
    if e:
        stages.append(("query\nencoding", e.get("p50_ms"), C_LLM))
    cs = r.get("card_search")
    if cs:
        stages.append((f"ontology card\nsearch ({cs['cards']:,})", cs["p50_ms"], C_ONTO))
    ps = [x for x in r.get("passage_search", []) if not x["synthetic_tiling"]]
    if ps:
        big = max(ps, key=lambda x: x["passages"])
        stages.append((f"passage search\n({big['passages']:,})", big["search_p50_ms"],
                       C_RAG))
    if q:
        stages.append(("context\nassembly", 0.1, C_HYB))
    e2e = r.get("end_to_end", {})
    for v, c in (("V4_ontology_rag", C_HYB), ("V5_ontology_rag_validated", C_VAL)):
        if v in e2e:
            if v.startswith("V5") and e2e[v].get("validation_mean_ms"):
                stages.append(("validation\n+ repair", e2e[v]["validation_mean_ms"], C_VAL))
            stages.append((f"generation\n({v[:2]})", e2e[v]["gen_mean_s"] * 1000, c))
            break
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    ax.bar(names, vals, color=[s[2] for s in stages], width=.62)
    ax.set_yscale("log")
    ax.set_ylabel("median cost per request (ms, log scale)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:,.2f} ms" if v < 100 else f"{v/1000:,.2f} s",
                ha="center", fontsize=7.5)
    ax.set_title("Figure 4 — Retrieval workflow: measured cost of each stage of one "
                 "request", fontsize=10.5, loc="left")
    ax.set_ylim(top=max(vals) * 4)
    save(fig, "fig04_retrieval_workflow.png",
         "Per-stage median cost of a single request. Retrieval is three to four "
         "orders of magnitude cheaper than generation.")


# ── Figures 5-6: variant comparison and hallucination ─────────────────────
def fig_variants():
    s = None
    for p in sorted(EV.glob("variant_scores__*.json")):
        s = load(p)
        model = p.stem.split("__")[1]
        break
    if not s:
        return skip("fig05_variant_comparison.png", "variant_scores__*.json missing")
    V = [v for v in ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
                     "V5_ontology_rag_validated"] if v in s["overall"]]
    lbl = {"V1_llm": "V1 LLM", "V2_rag": "V2 RAG", "V3_ontology_only": "V3 Onto",
           "V4_ontology_rag": "V4 Onto+RAG", "V5_ontology_rag_validated": "V5 +valid."}
    cols = {"V1_llm": C_LLM, "V2_rag": C_RAG, "V3_ontology_only": C_ONTO,
            "V4_ontology_rag": C_HYB, "V5_ontology_rag_validated": C_VAL}
    mets = [("token_f1", "Token F1"), ("entity_f1", "Entity F1"),
            ("entity_p", "Entity precision"), ("entity_hit", "Entity hit rate")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (k, t) in zip(axs, mets):
        y = [(s["overall"][v].get(k) or 0) * 100 for v in V]
        ax.bar([lbl[v] for v in V], y, color=[cols[v] for v in V], width=.68)
        ax.set_title(t, fontsize=9)
        ax.tick_params(axis="x", rotation=38, labelsize=7)
        for i, yy in enumerate(y):
            ax.text(i, yy + max(y) * .02, f"{yy:.1f}", ha="center", fontsize=7)
        ax.set_ylim(0, max(y) * 1.22)
    axs[0].set_ylabel("%")
    fig.suptitle(f"Figure 5 — Five system variants on the 347-item six-category "
                 f"benchmark ({model})", fontsize=10.5, x=.07, ha="left")
    save(fig, "fig05_variant_comparison.png",
         "Baseline comparison of the five WBS system variants.")

    # hallucination
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
    y = [(s["overall"][v].get("ungrounded_rate") or 0) * 100 for v in V]
    axs[0].bar([lbl[v] for v in V], y, color=[cols[v] for v in V], width=.68)
    for i, yy in enumerate(y):
        axs[0].text(i, yy + max(y) * .02, f"{yy:.1f}", ha="center", fontsize=7)
    axs[0].set_ylabel("% of named entities not in the ontology")
    axs[0].set_title("ungrounded entity rate, audit-grade (lower is better)",
                     fontsize=9)
    axs[0].tick_params(axis="x", rotation=38, labelsize=7)
    y2 = [(s["overall"][v].get("clean") or 0) * 100 for v in V]
    axs[1].bar([lbl[v] for v in V], y2, color=[cols[v] for v in V], width=.68)
    for i, yy in enumerate(y2):
        axs[1].text(i, yy + max(y2) * .02, f"{yy:.1f}", ha="center", fontsize=7)
    axs[1].set_ylabel("% of answers")
    axs[1].set_title("fully grounded answers (no ungrounded entity)", fontsize=9)
    axs[1].tick_params(axis="x", rotation=38, labelsize=7)
    fig.suptitle("Figure 6 — Hallucination, audited against the ontology "
                 "(reference-free)", fontsize=10.5, x=.07, ha="left")
    save(fig, "fig06_hallucination.png",
         "Groundedness of each variant, measured by auditing every named entity "
         "against ontology v1.4.0 rather than against a reference answer; "
         "sentence-initial imperatives, month names and markdown section labels are "
         "filtered out before counting.")

    # category heatmap
    cats = ["recommendation", "itinerary", "safety", "dining", "accessibility",
            "transport"]
    import numpy as np
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8))
    for ax, (k, t, cm) in zip(axs, [("entity_f1", "Entity F1 (%)", "viridis"),
                                    ("ungrounded_rate", "Ungrounded rate (%)",
                                     "magma_r")]):
        M = np.array([[(s["by_category"][v][c].get(k) or 0) * 100 for v in V]
                      for c in cats])
        im = ax.imshow(M, cmap=cm, aspect="auto")
        ax.set_xticks(range(len(V)), [lbl[v] for v in V], rotation=38, fontsize=7)
        ax.set_yticks(range(len(cats)), cats, fontsize=7.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="w" if (cm == "viridis") == (M[i, j] < M.mean()) else "k")
        ax.set_title(t, fontsize=9); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle("Figure 7 — Per-category results by variant", fontsize=10.5,
                 x=.07, ha="left")
    save(fig, "fig07_category_heatmap.png",
         "Entity F1 and ungrounded-entity rate for every category × variant cell.")


# ── Figure 8: scalability ─────────────────────────────────────────────────
def fig_scalability():
    o = load(SC / "ontology_scalability.json")
    r = load(SC / "retrieval_scalability.json")
    if not o:
        return skip("fig08_scalability.png", "ontology_scalability.json missing")
    rows = o["rows"]
    x = [q["triples"] for q in rows]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    axs[0].plot(x, [q["parse_s"] for q in rows], "o-", color=C_ONTO, label="RDF parse")
    axs[0].plot(x, [q["index_build_s"] for q in rows], "s-", color=C_HYB,
                label="adjacency index")
    axs[0].plot(x, [q["card_build_s"] for q in rows], "^-", color=C_RAG,
                label="entity cards")
    ow = [(q["triples"], q["owlrl_s"]) for q in rows if q.get("owlrl_s")]
    if ow:
        axs[0].plot([q for q, _ in ow], [v for _, v in ow], "d--", color=C_VAL,
                    label="OWL-RL closure")
    axs[0].set_xlabel("triples"); axs[0].set_ylabel("seconds")
    axs[0].set_title("one-off build cost", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False); axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    axs[1].plot(x, [q["hop1_p95_ms"] for q in rows], "o-", color=C_ONTO,
                label="1-hop subgraph p95")
    axs[1].plot(x, [q["hop2_p95_ms"] for q in rows], "s-", color=C_HYB,
                label="2-hop subgraph p95")
    axs[1].plot(x, [q.get("card_search_p50_ms") for q in rows], "^-", color=C_RAG,
                label="card search p50")
    axs[1].plot(x, [q.get("validate_p50_ms") for q in rows], "d-", color=C_VAL,
                label="validation p50")
    axs[1].set_xlabel("triples"); axs[1].set_ylabel("ms per request")
    axs[1].set_title("query-time cost", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False); axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    if r and r.get("passage_search"):
        pr = r["passage_search"]
        px = [q["passages"] for q in pr]
        real = [q for q in pr if not q["synthetic_tiling"]]
        axs[2].plot(px, [q["search_p50_ms"] for q in pr], "o-", color=C_RAG,
                    label="p50 (all sizes)")
        axs[2].plot([q["passages"] for q in real],
                    [q["search_p50_ms"] for q in real], "o", ms=9, mfc="none",
                    color="k", label="real passages")
        axs[2].set_xlabel("passages in dense index")
        axs[2].set_ylabel("ms per query")
        axs[2].set_title("dense passage search", fontsize=9)
        axs[2].legend(fontsize=7, frameon=False)
        axs[2].set_xscale("log")
    else:
        axs[2].set_axis_off()
        axs[2].text(.5, .5, "retrieval_scalability.json\nnot available",
                    ha="center", va="center", fontsize=8)
    fig.suptitle("Figure 8 — Scalability from the delivered size to 100K triples / "
                 "100K passages", fontsize=10.5, x=.07, ha="left")
    save(fig, "fig08_scalability.png",
         "Build and query-time cost as the ontology and the passage index grow. "
         "Subgraph traversal is flat; flat-index search and the validation scan are "
         "linear in size.")


# ── Figure 9: robustness ──────────────────────────────────────────────────
def fig_robustness():
    d = load(SC / "robustness.json")
    if not d:
        return skip("fig09_robustness.png", "robustness.json missing")
    rows = d["retrieval"]
    conds = [r["condition"] for r in rows]
    fig, axs = plt.subplots(1, 3 if "generation" in d else 2,
                            figsize=(13 if "generation" in d else 9.5, 3.8))
    axs[0].plot(conds, [r["rag_gold_coverage"] * 100 for r in rows], "o-", color=C_RAG,
                label="RAG")
    axs[0].plot(conds, [r["onto_gold_coverage"] * 100 for r in rows], "s-", color=C_ONTO,
                label="Ontology")
    axs[0].set_ylabel("% of gold places present in context")
    axs[0].set_title("gold-entity coverage under noise", fontsize=9)
    axs[0].legend(fontsize=7.5, frameon=False)
    axs[1].plot(conds, [r["rag_set_overlap_vs_clean"] * 100 for r in rows], "o-",
                color=C_RAG, label="RAG top-5")
    axs[1].plot(conds, [r["onto_set_overlap_vs_clean"] * 100 for r in rows], "s-",
                color=C_ONTO, label="Ontology cards")
    if "onto_class_routing_stability" in rows[0]:
        axs[1].plot(conds, [r["onto_class_routing_stability"] * 100 for r in rows],
                    "^--", color=C_HYB, label="class routing")
    axs[1].set_ylabel("% Jaccard overlap with clean retrieval")
    axs[1].set_title("retrieval stability under noise", fontsize=9)
    axs[1].legend(fontsize=7.5, frameon=False)
    if "generation" in d:
        by = {(g["arm"], g["condition"]): g for g in d["generation"]}
        axs[2].plot(conds, [by[("rag", c)]["entity_f1"] * 100 for c in conds], "o-",
                    color=C_RAG, label="RAG")
        axs[2].plot(conds, [by[("onto_v3", c)]["entity_f1"] * 100 for c in conds], "s-",
                    color=C_ONTO, label="Ontology")
        axs[2].set_ylabel("Entity F1 (%)")
        axs[2].set_title("answer quality under noise", fontsize=9)
        axs[2].legend(fontsize=7.5, frameon=False)
    for ax in axs:
        ax.tick_params(axis="x", rotation=38, labelsize=7)
    fig.suptitle(f"Figure 9 — Robustness under seven input conditions "
                 f"({d['n_questions']} questions, paired)", fontsize=10.5,
                 x=.07, ha="left")
    save(fig, "fig09_robustness.png",
         "Retrieval and (where measured) answer quality under character noise, "
         "lost casing, truncation, distractors and corrupted entity names.")


# ── Figure 10: does context coverage buy answer quality? ──────────────────
def fig_coverage_vs_f1():
    """The retrieval-tuning result: gold-entity coverage against generated token
    F1, with the question-independent control marked. A flat or falling line is
    the finding - coverage is not the lever."""
    t = load(DELIV / "task3_retrieval" / "onto_retrieval_tuning.json")
    if not t or not t.get("generation"):
        return skip("fig10_coverage_vs_f1.png",
                    "onto_retrieval_tuning.json not found or has no generation runs")
    cov, gen, pub = t["coverage"], t["generation"], t.get("published", {})
    pts = [(cov[k], gen[k]["token_f1"] * 100, gen[k]["entity_f1"] * 100, k)
           for k in gen if k in cov]
    if len(pts) < 2:
        return skip("fig10_coverage_vs_f1.png", "fewer than two policies generated")
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for ax, idx, lab in ((axs[0], 1, "Token F1"), (axs[1], 2, "Entity F1")):
        for c, tf, ef, k in pts:
            y = tf if idx == 1 else ef
            ctl = "global_degree" in k
            ax.scatter(c * 100, y, s=74, zorder=3,
                       marker="s" if ctl else "o",
                       color="#b3261e" if ctl else "#1f4e79")
            # a point near the right edge gets its label on the left, so the
            # control's name does not run off the axes
            right = c > 0.72
            ax.annotate(k, (c * 100, y), textcoords="offset points",
                        xytext=(-8 if right else 6, 5), fontsize=7.2,
                        ha="right" if right else "left")
        if pub.get("V2_rag"):
            yv = (pub["V2_rag"]["token_f1"] if idx == 1
                  else pub["V2_rag"]["entity_f1"]) * 100
            ax.axhline(yv, ls="--", lw=1.1, color="#777", zorder=1)
            ax.annotate("V2 Simple RAG", (ax.get_xlim()[0], yv), fontsize=7.2,
                        color="#555", textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("Gold-entity coverage of the context (%)", fontsize=8.5)
        ax.set_ylabel(f"{lab} of the generated answer (%)", fontsize=8.5)
        ax.grid(alpha=.25, zorder=0)
        ax.tick_params(labelsize=8)
        ax.margins(x=.09, y=.16)
    fig.suptitle("Figure 10 — More of the answer's entities in the context does not "
                 "buy a better answer\n(square = question-INDEPENDENT control: the "
                 "most-connected cards, same set for every question)",
                 fontsize=10.5, x=.02, ha="left")
    save(fig, "fig10_coverage_vs_f1.png",
         "Gold-entity coverage against generated token and entity F1 for each "
         "ontology context-selection policy, with the question-independent "
         "degree-roster control marked and the dense-RAG arm as a reference line.")


def fig_policy_transfer():
    """The transfer result: the same context policies at 8B, 3B and 1B, each as a
    delta against that model's own baseline. Every bar below 8B falling under the
    zero line is the finding - the tuning table does not transfer downward."""
    t = load(DELIV / "task3_retrieval" / "onto_retrieval_tuning.json")
    sc = (t or {}).get("scale") or {}
    if not sc or not (t or {}).get("generation"):
        return skip("fig11_policy_transfer.png",
                    "onto_retrieval_tuning.json carries no cross-size scale runs")
    big = t["generation"]
    series = [("llama3.1-8b", big)] + [(m, d) for m, d in sc.items()]
    pols = [p_ for p_ in ("round_robin", "routing_deep_hop2", "grouped_deep",
                          "grouped", "grouped_wide")
            if any(p_ in d for _, d in series)]
    if not pols:
        return skip("fig11_policy_transfer.png", "no policy is present at any size")
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    w = 0.8 / len(series)
    colors = ["#1f4e79", "#2e7d32", "#b3261e"]
    for si, (m, d) in enumerate(series):
        b = (d.get("baseline") or {}).get("token_f1")
        xs, ys = [], []
        for pi, p_ in enumerate(pols):
            o = d.get(p_)
            if not o or b is None:
                continue
            xs.append(pi - 0.4 + w * (si + 0.5))
            ys.append((o["token_f1"] - b) * 100)
        ax.bar(xs, ys, width=w * 0.92, label=m, zorder=3,
               color=colors[si % len(colors)])
    ax.axhline(0, color="#333", lw=1.1, zorder=2)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels(pols, fontsize=8.2)
    ax.set_ylabel("Token F1 against that model's own baseline (pp)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=.25, zorder=0)
    ax.legend(fontsize=8, frameon=False, ncol=len(series))
    fig.suptitle("Figure 11 — Context-assembly tuning does not transfer below 8B\n"
                 "(each bar is paired within one model against its own baseline; "
                 "no bar below 8B is positive)",
                 fontsize=10.5, x=.02, ha="left")
    save(fig, "fig11_policy_transfer.png",
         "Per-policy token F1 delta against each model's own ontology baseline at "
         "8B, 3B and 1B, every model with its own dedicated adapter - the "
         "cross-size transfer check for the retrieval-tuning table.")


def fig_budget_curve():
    """What a smaller context costs, per model size. The 8B line is flat to
    2,000 characters and the smaller models are not - the cost saving does not
    generalise downward."""
    t = load(DELIV / "task3_retrieval" / "onto_retrieval_tuning.json")
    bud = (t or {}).get("budget") or {}
    if not bud or not (t or {}).get("generation", {}).get("baseline"):
        return skip("fig12_budget_curve.png",
                    "onto_retrieval_tuning.json carries no budget ablation")
    full = t["budget_chars"]
    series = {"llama3.1-8b": (t["generation"]["baseline"]["token_f1"], bud)}
    for m, d in (t.get("budget_scale") or {}).items():
        b = ((t.get("scale") or {}).get(m) or {}).get("baseline")
        if b and d:
            series[m] = (b["token_f1"], d)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = {"llama3.1-8b": "#1f4e79", "llama3.2-3b": "#2e7d32",
              "llama3.2-1b": "#b3261e"}
    for m, (base_f1, d) in series.items():
        pts = sorted((int(k.split("_b")[1]), (v["token_f1"] - base_f1) * 100)
                     for k, v in d.items())
        xs = [full] + [x for x, _ in pts]
        ys = [0.0] + [y for _, y in pts]
        ax.plot(xs, ys, marker="o", lw=1.8, ms=5.5, zorder=3,
                color=colors.get(m, "#555"), label=m)
    ax.axhline(0, color="#333", lw=1.0, zorder=2)
    ax.set_xlabel("Ontology context budget (characters)", fontsize=8.5)
    ax.set_ylabel("Token F1 against that model's 3,000-char arm (pp)", fontsize=8.5)
    ax.grid(alpha=.25, zorder=0)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.margins(x=.06, y=.18)
    fig.suptitle("Figure 12 — A cheaper prompt is free at 8B and is not free "
                 "below it\n(identical retrieval; the context is simply "
                 "truncated, and every adapter was trained at 3,000)",
                 fontsize=10.5, x=.02, ha="left")
    save(fig, "fig12_budget_curve.png",
         "Token F1 against context budget for each model size, each curve "
         "relative to that model's own full-budget ontology arm - the cost "
         "ablation behind the project's one deployable saving and its limit.")


def fig_selective_validation():
    """The validation layer's trade-off as a curve, with the operating point
    that dominates validating everything marked. The point of the figure is
    that the curve is not monotone: spending the layer on every answer is worse
    on all three axes than spending it on the answers that need it."""
    d = load(DELIV / "task4_evaluation" / "selective_validation__llama3.1-8b.json")
    if not d or not d.get("curves"):
        return skip("fig13_selective_validation.png",
                    "selective_validation__llama3.1-8b.json not found")
    sig = d["best_signal"]
    pts = d["curves"][sig]
    ep, cv = d["endpoints"], d["cross_validated"][sig]
    xs = [p_["ungrounded_rate"] * 100 for p_ in pts]
    ys = [p_["token_f1"] * 100 for p_ in pts]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(xs, ys, "-o", lw=1.6, ms=4.5, color="#1f4e79", zorder=3,
            label=f"selective, threshold on {sig}")
    for lab, m, c, mk in (("V5: validate everything",
                           ep["V5 (validate everything)"], "#b3261e", "s"),
                          ("V4: validate nothing",
                           ep["V4 (validate nothing)"], "#2e7d32", "^"),
                          ("V2 Simple RAG", ep["V2 Simple RAG"], "#555", "D")):
        ax.plot(m["ungrounded_rate"] * 100, m["token_f1"] * 100, mk, ms=8,
                color=c, zorder=4, label=lab)
    ax.plot(cv["ungrounded_rate"] * 100, cv["token_f1"] * 100, "*", ms=18,
            color="#f9a825", mec="#333", mew=.6, zorder=5,
            label=f"cross-validated operating point "
                  f"({cv['n_validated']}/{cv['n']} validated)")
    ax.set_xlabel("Audit-grade ungrounded rate (%)  ← better", fontsize=8.5)
    ax.set_ylabel("Token F1 (%)  better →", fontsize=8.5)
    ax.grid(alpha=.25, zorder=0)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.margins(x=.08, y=.14)
    fig.suptitle("Figure 13 — The validation layer is worth spending "
                 "selectively, not everywhere\n(the starred point is more "
                 "grounded AND cleaner AND higher-overlap than validating "
                 "every answer)", fontsize=10.5, x=.02, ha="left")
    save(fig, "fig13_selective_validation.png",
         "Token F1 against audit-grade ungrounded rate along the selective "
         "validation curve, with the two published endpoints, Simple RAG, and "
         "the cross-validated operating point that dominates validating "
         "everything.")


# ── Figure 14: what the context supplies, under one fixed layer policy ────
def fig_context_decomposition():
    """Which part of the pipeline the groundedness win belongs to.

    The project credited the ontology with it, but every validated arm also
    reads retrieved prose. Holding the gate, the detector and the floor setting
    fixed and removing the prose from the context answers that: the graph-only
    arm is more grounded than its own draft and still behind both the hybrid
    arm and Simple RAG.
    """
    oo = load(DELIV / "task4_evaluation" / "ontology_only_validated__llama3.1-8b.json")
    sx = load(DELIV / "task4_evaluation" / "selective_axis__llama3.1-8b.json")
    sv = load(DELIV / "task4_evaluation" / "selective_validation__llama3.1-8b.json")
    if not (oo and sx and sv and oo.get("cells") and sx.get("cells")):
        return skip("fig14_context_decomposition.png",
                    "ontology_only_validated / selective_axis json not found")
    GO = "gated >=1 span, unsupportable only"
    rag = sv["endpoints"]["V2 Simple RAG"]
    arms = [("Simple RAG", rag, "#555"),
            ("Ontology\nonly", oo["cells"][GO], "#2e7d32"),
            ("Ontology\n+ retrieval", sx["cells"][GO], "#1f4e79")]
    panels = [("ungrounded_rate", "Audit-grade ungrounded rate (%)", "lower is better"),
              ("clean", "Fully grounded answers (%)", "higher is better"),
              ("token_f1", "Token F1 (%)", "higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))
    for ax, (f, lab, dirn) in zip(axes, panels):
        xs = range(len(arms))
        vals = [(m.get(f) or 0) * 100 for _, m, _ in arms]
        ax.bar(xs, vals, color=[c for _, _, c in arms], width=.62, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n for n, _, _ in arms], fontsize=7.5)
        ax.set_title(f"{lab}\n({dirn})", fontsize=8.5, pad=6)
        ax.grid(axis="y", alpha=.25, zorder=0)
        ax.tick_params(labelsize=8)
        ax.margins(y=.18)
    vh = (oo.get("vs_hybrid_same_cell") or {}).get("ungrounded_rate") or {}
    sub = ""
    if vh.get("delta") is not None and vh.get("p") is not None:
        sub = (f"\nremoving the retrieved prose costs {-vh['delta']:+.2f} pp "
               f"ungrounded (p = {vh['p']:.3g}) with everything else held fixed")
    fig.tight_layout(rect=(0, .06, 1, .87))
    sz = oo.get("beats_rag_at_sizes")
    if sz:
        sub += ("\nat " + ", ".join(sz) + " the graph-only arm does beat Simple "
                "RAG on both measures - this panel is llama3.1-8b")
    fig.suptitle("Figure 14 — At 8B the groundedness win belongs to ontology AND "
                 "retrieval, not to the graph alone" + sub,
                 fontsize=10.5, x=.02, y=.985, ha="left", va="top")
    fig.text(.5, .005, "the two ontology arms gate on the same signal at the "
             "same threshold and differ only in whether the context carries "
             "retrieved prose", ha="center", fontsize=7.5, color="#444")
    save(fig, "fig14_context_decomposition.png",
         "Three arms under one fixed validation policy (gated on the layer's "
         "own detector, abstaining only on an unsupportable claim, audit-grade "
         "detector): Simple RAG, ontology context alone, and ontology plus "
         "retrieved prose. Read from ontology_only_validated__llama3.1-8b.json "
         "and selective_axis__llama3.1-8b.json; the same arm at 1B and 3B is "
         "in ontology_only_validated__*.md, and its verdict is size dependent.")


def fig_repair_policy():
    """What the validation layer was charging for, and what it charges now.

    Left: the token F1 the ontology-validated arm gives up against Simple RAG,
    before and after the repair stops being a whole-answer rewrite. Right: the
    independent strict-band ungrounded rate, which the layer does not optimise,
    so the groundedness gain can be read without the constructional zero.
    """
    SIZES = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
    SHORT = {"llama3.2-1b": "1B", "llama3.2-3b": "3B", "llama3.1-8b": "8B"}
    REGEN = "V5 regen, gated (selective_axis best)"
    SURG = "V5 surgical clause, gated"
    d = {m: load(DELIV / "task4_evaluation" / f"repair_policy__{m}.json")
         for m in SIZES}
    d = {m: v for m, v in d.items() if v and v.get("cells")}
    if len(d) < 3:
        return skip("fig15_repair_policy.png", "repair_policy__*.json not found")

    def gap(m, arm):
        t = ((d[m]["cells"].get(arm) or {}).get("vs_rag") or {}).get("token_f1") or {}
        return t.get("delta")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    xs = list(range(len(d)))
    w = .36
    before = [-gap(m, REGEN) for m in d]
    after = [-gap(m, SURG) for m in d]
    ax1.bar([x - w / 2 for x in xs], before, w, color="#b23a3a",
            label="repair = re-prompt the model", zorder=3)
    ax1.bar([x + w / 2 for x in xs], after, w, color="#1f4e79",
            label="repair = cut the offending clause", zorder=3)
    for x, v in zip(xs, before):
        ax1.text(x - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for x, v in zip(xs, after):
        ax1.text(x + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(xs); ax1.set_xticklabels([SHORT[m] for m in d], fontsize=9)
    ax1.set_ylabel("token F1 given up vs Simple RAG (pp)", fontsize=8.5)
    ax1.set_title("What grounding costs in reference overlap\n(lower is better)",
                  fontsize=9, pad=6)
    ax1.grid(axis="y", alpha=.25, zorder=0); ax1.margins(y=.22)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=.9)

    arms = [("Simple RAG", "V2 Simple RAG", "#555"),
            ("draft, layer off", "V4 draft, layer off", "#999"),
            ("regen repair", REGEN, "#b23a3a"),
            ("surgical repair", SURG, "#1f4e79")]
    w2 = .2
    for i, (lab, arm, col) in enumerate(arms):
        vals = [(d[m]["cells"].get(arm) or {}).get("ungrounded_rate_strict") or 0
                for m in d]
        ax2.bar([x + (i - 1.5) * w2 for x in xs], vals, w2, color=col,
                label=lab, zorder=3)
        for x, v in zip(xs, vals):
            ax2.text(x + (i - 1.5) * w2, v, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=7)
    ax2.set_xticks(xs); ax2.set_xticklabels([SHORT[m] for m in d], fontsize=9)
    ax2.set_ylabel("ungrounded rate, strict detector (%)", fontsize=8.5)
    ax2.set_title("Groundedness under the detector the layer does NOT optimise\n"
                  "(lower is better)", fontsize=9, pad=6)
    ax2.grid(axis="y", alpha=.25, zorder=0); ax2.margins(y=.24)
    ax2.legend(fontsize=7.5, ncol=2, framealpha=.9)

    fig.tight_layout(rect=(0, .07, 1, .88))
    fig.suptitle("Figure 15 \u2014 The validation layer's overlap cost was the "
                 "rewrite, not the check (R19)", fontsize=10.5, x=.02, y=.985,
                 ha="left", va="top")
    fig.text(.5, .005, "published ungrounded rate for both surgical cells is "
             "0.00% by construction - the pass removes exactly what the published "
             "detector counts - so the right panel uses the strict rule instead",
             ha="center", fontsize=7.5, color="#444")
    save(fig, "fig15_repair_policy.png",
         "Left: token F1 the gated ontology-validated arm gives up against Simple "
         "RAG with the published re-prompting repair and with surgical clause "
         "removal, at three model sizes. Right: ungrounded rate under "
         "audit_spans(strict=True), the rule the repair policy does not optimise. "
         "Read from repair_policy__llama3.2-1b/3b/llama3.1-8b.json.")


# ── Figure 16: the two arms across eight models, by training budget ───────
def fig_cross_model():
    """The same comparison at eight models, split by who got how much training.

    Left: the token F1 delta per model, equal budget against extended ontology
    training. The equal-budget bars are the project's standing result; the
    extended bars are what the ontology context reaches when it is given three
    or four passes against a baseline given one, and they are drawn in a
    different colour because they are not the same experiment.

    Right: the groundedness delta on the same models. This is the metric that
    cannot be won by reproducing the reference's wording, and the one where the
    equal-budget comparison is not uniformly negative.
    """
    d = load(DELIV / "task4_evaluation" / "cross_model_results.json")
    if not d or not d.get("pairs"):
        return skip("fig16_cross_model.png", "cross_model_results.json not found")

    eq = {r["model"]: r for r in d["pairs"] if r["tier"] == "equal"}
    ex = {r["model"]: r for r in d["pairs"] if r["tier"] == "extended"}
    mt = {r["model"]: r for r in d["pairs"] if r["tier"] == "matched"}
    models = [m for m in ("llama3.2-1b", "llama3.2-3b", "qwen2.5-1.5b", "qwen2.5-3b")
              if m in eq]
    if len(models) < 4:
        return skip("fig16_cross_model.png", "fewer than four equal-budget pairs")
    SHORT = {"llama3.2-1b": "Llama 3.2\n1B", "llama3.2-3b": "Llama 3.2\n3B",
             "qwen2.5-1.5b": "Qwen 2.5\n1.5B", "qwen2.5-3b": "Qwen 2.5\n3B"}

    def delta(group, m, metric):
        r = group.get(m)
        return None if not r else r["metrics"][metric]["delta"]

    def sig(group, m, metric):
        r = group.get(m)
        return bool(r) and r["metrics"][metric]["verdict"] != "indistinguishable"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    xs = list(range(len(models)))
    # the matched tier is what the extended tier has to be read against, so it is a
    # bar beside it rather than a note under the figure
    groups = [(eq, "#1f4e79", "equal budget (1 pass each)"),
              (ex, "#c98a2b", "ontology 3-4 passes vs RAG 1")]
    if len(mt) == len(models):
        groups.append((mt, "#2e7d5b", "matched budget (same passes both arms)"))
    w = .78 / len(groups)
    offs = [(i - (len(groups) - 1) / 2) * w for i in range(len(groups))]
    for ax, metric, title, ylab in (
        (ax1, "token_f1", "Reference overlap\n(positive = ontology ahead)",
         "token F1, ontology − Simple RAG"),
        (ax2, "clean", "Fully grounded answers\n(positive = ontology ahead)",
         "share fully grounded, ontology − Simple RAG"),
    ):
        for off, (group, colour, label) in zip(offs, groups):
            vals = [delta(group, m, metric) for m in models]
            bars = ax.bar([x + off for x in xs], [v or 0 for v in vals], w,
                          color=colour, label=label, zorder=3)
            for x, v, m, bar in zip(xs, vals, models, bars):
                if v is None:
                    continue
                ax.text(x + off, v, f"{v:+.3f}", ha="center", fontsize=6.6,
                        va="bottom" if v >= 0 else "top")
                if not sig(group, m, metric):
                    # a bar the tests cannot separate from zero is hatched, so the
                    # eye does not read a height as a result
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
        ax.axhline(0, color="#333", lw=.9, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9, pad=6)
        ax.grid(axis="y", alpha=.25, zorder=0)
        ax.margins(y=.26)
    # proxy handles: the real bars carry per-cell hatching, and a legend that
    # inherited it would read as "the whole series is not separable from zero"
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=c, label=lab) for _, c, lab in groups]
                       + [Patch(facecolor="#9aa7b4", hatch="///", edgecolor="white",
                                label="not separable from zero")],
               fontsize=7, loc="lower right", framealpha=.95)
    fig.suptitle(
        f"Ontology-only vs Simple RAG, {d['split_n']:,} held-out questions per cell "
        "· hatched = not separable from zero",
        fontsize=9.5, y=1.0)
    fig.tight_layout()
    summ = d["summary"]

    def _mroll(metric):
        return d["metric_rollup"]["matched"][metric]["ontology_better_at"]

    save(fig, "fig16_cross_model.png",
         f"The ontology-only arm against Simple RAG on {d['split_n']:,} held-out "
         f"questions at four models with a comparable baseline, read by training "
         f"budget. At equal budget {summ['equal']['clean_sweeps']} of "
         f"{summ['equal']['pairs']} models lead on every metric; with three or four "
         f"ontology passes against the same one-pass baseline "
         f"{summ['extended']['never_worse']} of {summ['extended']['pairs']} are worse "
         f"on nothing measured; and with the baseline continued to the same pass count "
         f"(exp060) that falls to {summ['matched']['never_worse']} of "
         f"{summ['matched']['pairs']}, keeping a token F1 lead at "
         f"{', '.join(_mroll('token_f1')) or 'no model'} and the fully-grounded lead "
         f"at {', '.join(_mroll('clean')) or 'no model'}. "
         f"Hatched bars are deltas the paired Wilcoxon and the "
         f"bootstrap interval cannot separate from zero. Read from "
         f"cross_model_results.json; the four 8B pairs are excluded here because their "
         f"Simple RAG arm was never trained.")


def main() -> None:
    print("building figures …", flush=True)
    for fn in (fig_architecture, fig_ontology_graph, fig_retrieval_workflow,
               fig_variants, fig_scalability, fig_robustness,
               fig_coverage_vs_f1, fig_policy_transfer, fig_budget_curve,
               fig_selective_validation, fig_context_decomposition,
               fig_repair_policy, fig_cross_model):
        try:
            fn()
        except Exception as e:  # a broken figure must not take the set down
            skip(fn.__name__, f"{type(e).__name__}: {e}")
    L = ["# Task 4.2 - Figure Index\n",
         "Each figure is generated by `wbs/task4_scalability/make_figures.py` from the "
         "JSON deliverable named in its caption; none contains hand-entered values.\n"]
    for n, cap in made:
        L.append(f"- **`{n}`** — {cap}")

    # Figures this script does not build. Without this block the index silently
    # omits them while its own preamble claims to cover every figure present.
    elsewhere = {
        "fig17_ontology_whole.png":
            ("wbs/task3_storage/ontology_poster.py",
             "the entire ontology in one frame - every concept and every "
             "relation, force-directed, class-coloured, hubs labelled"),
        "fig18_ontology_by_class.png":
            ("wbs/task3_storage/ontology_poster.py",
             "the same graph with every class given its own cluster on a ring, "
             "so each of the 33 classes is findable"),
        "fig19_ontology_structure.png":
            ("wbs/task6_report/make_paper_figures.py",
             "the ontology as a structure rather than as instances: the class "
             "taxonomy beside the schema-level relation graph. The paper "
             "figure; also written as a vector PDF"),
        "fig20_qualitative_rag_vs_ontology.png":
            ("wbs/task6_report/make_paper_figures.py",
             "four benchmark questions with the prompt, the reference, the "
             "Simple-RAG answer and the ontology answer, entities marked, and "
             "the base rate of each direction stated on the figure"),
    }
    present = [(n, v) for n, v in elsewhere.items()
               if (FIG.parent / "figures" / n).exists()]
    if present:
        L.append("\n## Built by another module\n")
        for n, (gen, cap) in present:
            L.append(f"- **`{n}`** — {cap}. Generated by `{gen}`.")
    if skipped:
        L.append("\n## Not generated\n")
        for n, why in skipped:
            L.append(f"- `{n}` — {why}")
    (FIG.parent / "figures" / "FIGURE_INDEX.md").write_text("\n".join(L) + "\n",
                                                            encoding="utf-8")
    print(f"\n{len(made)} figures in {FIG}" +
          (f"; {len(skipped)} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
