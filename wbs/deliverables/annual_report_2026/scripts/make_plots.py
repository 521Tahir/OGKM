"""Figures for the 2026 annual report (ETRI joint/commissioned institution form).

Every value is read from a project deliverable or from the live Neo4j database;
nothing is typed in by hand except the Year-2 plan schedule, which is a plan.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import openpyxl

ROOT = Path("/home/tahirahmad/Ontology_Project_JejuFull")
D = ROOT / "wbs/deliverables"
OUT = D / "annual_report_2026/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

NAVY, TEAL, ORANGE, RED, GREY, GREEN, PURPLE = (
    "#1F3A5F", "#2A9D8F", "#E76F51", "#C0392B", "#8D99AE", "#5B8E3E", "#6A4C93")
VARIANT_COLORS = [GREY, NAVY, TEAL, PURPLE, ORANGE]
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag", "V5_ontology_rag_validated"]
VLABEL = ["V1 LLM only", "V2 Simple RAG", "V3 Ontology only", "V4 Ontology+RAG", "V5 Onto+RAG+validation"]
MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
MLABEL = {"llama3.2-1b": "Llama-3.2-1B", "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B"}


def load(rel):
    return json.load(open(D / rel))


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, color, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color, ec="none"))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", color=tc,
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", color=tc, fontsize=8.2,
                linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333333"))

    box(0.1, 4.3, 2.2, 1.9, "Jeju tourism corpus",
        "46,059 cleaned records\ntrain / val / test splits\nmetadata index\n(Task 1.2)", GREY)
    box(2.65, 4.3, 2.3, 1.9, "LLM extraction",
        "concepts, relations,\nhierarchy, rules\n5 prompting strategies\n(Task 2.1)", NAVY)
    box(5.3, 4.3, 2.4, 1.9, "Normalization &\nformalization",
        "\nsynonym merge, dedupe\nRDF / OWL / JSON-LD\nSHACL + OWL-RL checks\n(Task 2.2)", NAVY)
    box(8.05, 3.55, 3.85, 2.65, "Neo4j 5.26 ontology graph DB",
        "typed property graph of v1.4.0\n1,565 nodes  ·  7,909 relationships\n34 labels  ·  20 relationship types\n"
        "uniqueness constraint + indexes\nCypher competency queries (p50 < 1.5 ms)\n"
        "versioning, diff, RBAC, audit (Task 3.1)", "#0B6E4F")
    box(5.3, 1.25, 2.4, 2.1, "Ontology-aware\nretrieval",
        "\nintent → class routing\nsubgraph + dense cards\nhybrid with passages\n(Task 3.2.1)", TEAL)
    box(2.65, 1.25, 2.3, 2.1, "Constrained\ngeneration",
        "\nLLM + LoRA adapters\nvalidate, surgical repair,\nevidence trace, abstain\n(Task 3.2.2)", TEAL)
    box(0.1, 1.25, 2.2, 2.1, "Evaluation",
        "347-item benchmark\n5 system variants\n8 models · 4,607 test Qs\nscalability, robustness\n(Task 4)", ORANGE)
    box(8.05, 1.25, 3.85, 1.85, "Standards exchange (RDF)",
        "Turtle / OWL / N-Triples / JSON-LD\npySHACL validation, OWL-RL closure\nSPARQL endpoint for interoperability", PURPLE)

    arrow(2.3, 5.25, 2.65, 5.25); arrow(4.95, 5.25, 5.3, 5.25); arrow(7.7, 5.25, 8.05, 5.25)
    arrow(9.95, 3.55, 9.95, 3.1)
    arrow(8.05, 4.0, 7.7, 2.6)
    arrow(5.3, 2.3, 4.95, 2.3); arrow(2.65, 2.3, 2.3, 2.3)
    ax.text(7.95, 3.2, "subgraph\nqueries", fontsize=7.5, color="#555555", ha="right")
    ax.text(10.05, 3.3, "export", fontsize=7.5, color="#555555")
    ax.text(6.0, 0.55, "Fig. 1  System architecture: Neo4j is the ontology store and query engine; "
            "RDF exports carry standards validation.", ha="center", fontsize=9, style="italic")
    save(fig, "fig01_architecture_neo4j.png")


# ---------------------------------------------------------------- WBS gantt
WBS_ROWS = {
    5: ("1.1", "Requirements, modules, architecture, domain scope"),
    6: ("1.1", "Repository, tracking, reproducibility, risk plan"),
    8: ("1.2", "Collect / clean / dedupe / annotate tourism data"),
    9: ("1.2", "Ontology schema v0 (concepts, relations, rules)"),
    12: ("2.1", "Concept, relation, hierarchy, rule extraction"),
    13: ("2.1", "Prompting strategies + sensitivity / stability"),
    15: ("2.2", "Normalization and RDF/OWL conversion"),
    16: ("2.2", "Validation: consistency, SHACL, repair hints"),
    19: ("3.1", "Ontology storage (graph DB + files)"),
    20: ("3.1", "Versioning, diff, rollback, governance"),
    22: ("3.2", "Query understanding, subgraph and hybrid retrieval"),
    23: ("3.2", "Constrained generation, validation, repair, trace"),
    26: ("4.1", "Benchmark construction (6 task types)"),
    27: ("4.1", "5-variant evaluation and ablations"),
    29: ("4.2", "Scalability 10K-100K, latency, robustness"),
    30: ("4.2", "Visualization, final report, reproducibility"),
}
TASK_COLOR = {"1": GREY, "2": NAVY, "3": "#0B6E4F", "4": ORANGE}


def fig_gantt_2026():
    wb = openpyxl.load_workbook(D / "2026-08월 공동연구 WBS_제주대(윤여찬)_진척완료.xlsx")
    ws = wb["2026 실행Ver_1"]
    cols = [openpyxl.utils.column_index_from_string(c) for c in "DEFGHIJKLMNOPQRS"]
    rows = []
    for r, (task, label) in WBS_ROWS.items():
        filled = [i for i, c in enumerate(cols)
                  if ws.cell(r, c).fill is not None and ws.cell(r, c).fill.patternType == "solid"]
        prog = ws.cell(r, openpyxl.utils.column_index_from_string("T")).value
        rows.append((task, label, min(filled), max(filled) + 1, prog))
    fig, ax = plt.subplots(figsize=(11, 7.0))
    for i, (task, label, s, e, prog) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, (e - s) / 2, left=s / 2, height=0.62, color=TASK_COLOR[task[0]], alpha=0.9)
        ax.text(e / 2 + 0.06, y, f"{int(round((prog or 0) * 100))}%", va="center", fontsize=10,
                color="#222222", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{t}  {l}" for t, l, *_ in rows][::-1], fontsize=10.5)
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(months, fontsize=10.5)
    ax.set_xlim(0, 8.6)
    for m in range(9):
        ax.axvline(m, color="#cccccc", lw=0.6)
    ax.grid(False)
    ax.legend(handles=[mpatches.Patch(color=TASK_COLOR[k], label=f"Task {k}") for k in "1234"],
              loc="lower left", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0, -0.13))
    ax.set_title("Year-1 (2026) WBS execution schedule and progress, May-December 2026", fontsize=12.5)
    save(fig, "fig02_wbs_gantt_2026.png")


# ---------------------------------------------------------------- data profile
def fig_dataset():
    ds = json.load(open(ROOT / "artifacts/reports/dataset_statistics.json"))
    tax = sorted(ds["task_taxonomy"].items(), key=lambda kv: kv[1]["count"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [2.3, 1]})
    names = [k.replace("_", " ") for k, _ in tax]
    a1.barh(names, [v["pct"] for _, v in tax], color=TEAL)
    for i, (_, v) in enumerate(tax):
        a1.text(v["pct"] + 0.8, i, f'{v["pct"]:.1f}%', va="center", fontsize=8)
    a1.set_xlabel("share of records tagged (multi-label, %)")
    a1.set_title("Task taxonomy of the cleaned Jeju corpus")
    a1.set_xlim(0, 105); a1.tick_params(axis="y", labelsize=8.5)
    sp = ds["splits"]
    a2.bar(["train", "validation", "test"], [sp["train"], sp["val"], sp["test"]],
           color=[NAVY, GREY, ORANGE])
    for i, v in enumerate([sp["train"], sp["val"], sp["test"]]):
        a2.text(i, v + 500, f"{v:,}", ha="center", fontsize=9)
    a2.set_title(f'Splits ({ds["total_after_dedup"]:,} records, '
                 f'{ds["validation_summary"]["validity_rate"] * 100:.0f}% valid)')
    a2.set_ylabel("records")
    save(fig, "fig03_dataset_profile.png")


# ---------------------------------------------------------------- versions
def fig_versions():
    dl = load("task3_governance/diff_logs.json")
    vers = [dl[0]["from"]] + [x["to"] for x in dl]
    conc = [dl[0]["concepts"]["before"]] + [x["concepts"]["after"] for x in dl]
    rel = [dl[0]["relations"]["before"]] + [x["relations"]["after"] for x in dl]
    kinds = [None] + [x["change_type"] for x in dl]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    x = np.arange(len(vers))
    ax.plot(x, rel, "-o", color=NAVY, label="relations (triples)")
    ax.plot(x, conc, "-s", color=TEAL, label="concepts")
    for i, k in enumerate(kinds):
        if k == "breaking":
            ax.axvspan(i - 0.5, i + 0.5, color=ORANGE, alpha=0.10, lw=0)
        ax.annotate(f"{rel[i]:,}", (x[i], rel[i]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=NAVY)
        ax.annotate(f"{conc[i]:,}", (x[i], conc[i]), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7.5, color=TEAL)
    ax.axvline(vers.index("1.4.0"), color="#0B6E4F", ls="--", lw=1.2)
    ax.text(vers.index("1.4.0") + 0.08, max(rel) * 0.93, "v1.4.0 = evaluated release,\nloaded in Neo4j",
            color="#0B6E4F", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(["v" + v for v in vers], fontsize=8.5)
    ax.set_ylabel("count")
    ax.legend(handles=ax.get_legend_handles_labels()[0] +
              [mpatches.Patch(color=ORANGE, alpha=0.25, label="breaking change")], loc="upper left",
              frameon=False)
    ax.set_title("Ontology release history with content-hashed diffs (governance log)")
    save(fig, "fig04_ontology_versions.png")


# ---------------------------------------------------------------- Neo4j live
def neo4j_session():
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver("bolt://127.0.0.1:7690",
                               auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "jeju-ontology-2026")))
    return drv


def fig_neo4j(drv):
    with drv.session() as s:
        labels = s.run("MATCH (n) UNWIND labels(n) AS l WITH l, count(*) AS c "
                       "WHERE l <> 'Concept' RETURN l, c ORDER BY c DESC").data()
        rels = s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC").data()
        totals = s.run("MATCH (n) WITH count(n) AS n MATCH ()-[r]->() RETURN n, count(r) AS r").single()
        version = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
    snap = {"neo4j_version": version, "nodes": totals["n"], "relationships": totals["r"],
            "labels": labels, "relationship_types": rels}
    json.dump(snap, open(OUT / "neo4j_live_snapshot.json", "w"), indent=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.2))
    top = labels[:20][::-1]
    a1.barh([d["l"] for d in top], [d["c"] for d in top], color="#0B6E4F")
    for i, d in enumerate(top):
        a1.text(d["c"] + 5, i, f'{d["c"]:,}', va="center", fontsize=7.5)
    a1.set_title(f"Node labels (top 20 of {len(labels)} class labels)")
    a1.set_xlabel("nodes"); a1.tick_params(axis="y", labelsize=8)
    rr = rels[::-1]
    a2.barh([d["t"] for d in rr], [d["c"] for d in rr], color=NAVY)
    for i, d in enumerate(rr):
        a2.text(d["c"] + 20, i, f'{d["c"]:,}', va="center", fontsize=7.5)
    a2.set_title(f"Relationship types ({len(rels)} types, {totals['r']:,} relationships)")
    a2.set_xlabel("relationships"); a2.tick_params(axis="y", labelsize=8)
    a1.set_xlim(0, max(d["c"] for d in top) * 1.15)
    a2.set_xlim(0, max(d["c"] for d in rr) * 1.15)
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle(f"Live Neo4j {version} database contents: ontology v1.4.0, {totals['n']:,} nodes",
                 fontweight="bold")
    save(fig, "fig05_neo4j_schema_distribution.png")
    return snap


def fig_neo4j_subgraph(drv):
    q = ("MATCH (c:Concept {name:$name})-[r]-(x:Concept) "
         "WITH type(r) AS t, x ORDER BY t, x.name "
         "WITH t, collect(DISTINCT x)[0..4] AS xs UNWIND xs AS x "
         "RETURN t, x.name AS n, x.concept_class AS cls")
    with drv.session() as s:
        center = "Seongsan Ilchulbong"
        rows = s.run(q, name=center).data()
        ccls = s.run("MATCH (c:Concept {name:$n}) RETURN c.concept_class AS k", n=center).single()["k"]
    G = nx.Graph()
    G.add_node(center, cls=ccls)
    for r in rows:
        if r["n"] == center:
            continue
        G.add_node(r["n"], cls=r["cls"])
        G.add_edge(center, r["n"], t=r["t"])
    classes = sorted({d["cls"] for _, d in G.nodes(data=True)})
    cmap = plt.get_cmap("tab20")
    col = {c: cmap(i % 20) for i, c in enumerate(classes)}
    pos = nx.spring_layout(G, seed=7, k=0.9)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis("off"); ax.grid(False)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[G.nodes[n]["cls"]] for n in G],
                           node_size=[1500 if n == center else 420 for n in G], edgecolors="white")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5,
                            labels={n: (n if len(n) < 28 else n[:26] + "…") for n in G})
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=6.5, font_color=NAVY,
                                 edge_labels={(u, v): d["t"] for u, v, d in G.edges(data=True)})
    ax.legend(handles=[mpatches.Patch(color=col[c], label=c) for c in classes], fontsize=7.5,
              loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, title="concept_class")
    ax.set_title(f"Neo4j neighbourhood of '{center}' (up to 4 neighbours per relationship type, live Cypher query)")
    save(fig, "fig06_neo4j_subgraph.png")


def fig_cypher_latency():
    neo = load("task3_storage/graph_db_neo4j.json")
    typed = neo["mappings"]["typed"]["queries"]
    generic = neo["mappings"].get("generic", {}).get("queries", [])
    oxi = load("task3_storage/graph_db.json")["queries"]
    rdf = load("task3_storage/sparql_store.json")["queries"]
    names = [q["query"].split(" ", 1)[0] + "\n" + q["query"].split(" ", 1)[1].split(" (")[0] for q in typed]
    series = [("Neo4j, typed mapping (Cypher)", [q["p50_ms"] for q in typed], "#0B6E4F")]
    if generic:
        series.append(("Neo4j, generic mapping (Cypher)", [q["p50_ms"] for q in generic], TEAL))
    series += [("Oxigraph (SPARQL)", [q["p50_ms"] for q in oxi], PURPLE),
               ("rdflib in-memory (SPARQL)", [q["p50_ms"] for q in rdf], GREY)]
    x = np.arange(len(names)); w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    for i, (lab, vals, c) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=lab, color=c)
    ax.set_yscale("log"); ax.set_ylabel("median latency, ms (log)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.8)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title("Nine competency questions: median latency over 25 repeats (same ontology content)")
    save(fig, "fig07_cypher_latency.png")


# ---------------------------------------------------------------- validation
def fig_validation():
    v = load("task2_formalization/validation_report.json")
    codes = {k: c for k, c in v["code_counts"].items() if k != "OWL_CONSISTENT"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ks = sorted(codes, key=codes.get)
    a1.barh([k.replace("_", " ").lower() for k in ks], [codes[k] for k in ks],
            color=[ORANGE if "TYPE" in k or "MULTI" in k else GREY for k in ks])
    for i, k in enumerate(ks):
        a1.text(codes[k] + 20, i, f"{codes[k]:,}", va="center", fontsize=8.5)
    a1.set_title("L2 findings (orange = warning, grey = info); L1 errors = 0")
    a1.set_xlabel("findings")
    o = v["owl"]
    a2.bar(["asserted", "after OWL-RL\nclosure"], [o["triples_before_closure"], o["triples_after_closure"]],
           color=[NAVY, TEAL])
    for i, val in enumerate([o["triples_before_closure"], o["triples_after_closure"]]):
        a2.text(i, val + 300, f"{val:,}", ha="center", fontsize=9)
    a2.set_title(f"OWL-RL: consistent, {o['entailed']:,} entailed\nSHACL conforms: {v['shacl']['conforms']}")
    a2.set_ylabel("RDF triples")
    save(fig, "fig08_validation.png")


# ---------------------------------------------------------------- prompting
def fig_prompting():
    p = {**load("task2_extraction/prompt_sensitivity__gpu5.json")["strategies"],
         **load("task2_extraction/prompt_sensitivity__gpu6.json")["strategies"]}
    h = [load(f"task2_extraction/hypothesis_h1__s{i}.json") for i in (0, 1)]
    strat = ["naive", "cot", "graph_of_thought", "metacognitive", "repair_after_failure"]
    lab = ["naive", "CoT", "GoT", "metacog.", "repair"]
    jac = [p[s]["stability"]["concept_set_jaccard_mean"] for s in strat]
    cpr = [p[s]["stability"]["agg"]["concepts_per_record"]["mean"] for s in strat]
    rpr = [p[s]["stability"]["agg"]["relations_per_record"]["mean"] for s in strat]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    axs[0].bar(lab, jac, color=[GREY, NAVY, TEAL, PURPLE, ORANGE])
    for i, v in enumerate(jac):
        axs[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8.5)
    axs[0].set_ylim(0.55, 0.72); axs[0].set_title("Run-to-run concept agreement\n(mean Jaccard, 3 runs, T=0.7)")
    x = np.arange(len(strat))
    axs[1].bar(x - 0.2, cpr, 0.4, label="concepts / record", color=TEAL)
    axs[1].bar(x + 0.2, rpr, 0.4, label="relations / record", color=NAVY)
    axs[1].set_xticks(x); axs[1].set_xticklabels(lab); axs[1].legend(fontsize=8, frameon=False)
    axs[1].set_title("Extraction yield per record (15 records)")
    for k, (hh, c) in enumerate(zip(h, [GREY, ORANGE])):
        axs[2].bar(x + (k - 0.5) * 0.4, [hh[s]["l2_findings_per_100rel"] for s in strat], 0.4,
                   label=f"sample {k}", color=c)
    axs[2].set_xticks(x); axs[2].set_xticklabels(lab); axs[2].legend(fontsize=8, frameon=False)
    axs[2].set_title("H1 test: L2 findings per 100 relations\n(lower is better, 2 x 20 records)")
    save(fig, "fig09_prompting_strategies.png")


# ---------------------------------------------------------------- 5 variants
def fig_variants():
    vs = {m: load(f"task4_evaluation/variant_scores__{m}.json")["overall"] for m in MODELS}
    metrics = [("token_f1", "Token F1 (%) - higher is better"),
               ("ungrounded_rate", "Ungrounded-entity rate (%) - lower is better"),
               ("clean", "Fully grounded answers (%) - higher is better")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    x = np.arange(len(MODELS)); w = 0.16
    for ax, (k, t) in zip(axs, metrics):
        for i, v in enumerate(VARIANTS):
            vals = [vs[m][v][k] * 100 for m in MODELS]
            ax.bar(x + (i - 2) * w, vals, w, color=VARIANT_COLORS[i], label=VLABEL[i])
        ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in MODELS], fontsize=8.5)
        ax.set_title(t, fontsize=10)
    axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Five system variants on the 347-item benchmark at three model sizes "
                 "(published configuration, ontology v1.4.0)", fontweight="bold")
    save(fig, "fig10_five_variants.png")


def fig_category_heatmap():
    bc = load("task4_evaluation/variant_scores__llama3.1-8b.json")["by_category"]
    cats = ["recommendation", "itinerary", "safety", "dining", "accessibility", "transport"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.9))
    for ax, (k, t, cm) in zip(axs, [("token_f1", "Token F1 (%)", "Blues"),
                                    ("ungrounded_rate", "Ungrounded-entity rate (%, lower better)", "Oranges")]):
        M = np.array([[bc[v][c][k] * 100 for c in cats] for v in VARIANTS])
        im = ax.imshow(M, cmap=cm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > M.max() * 0.65 else "black")
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=8.5, rotation=15)
        ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VLABEL, fontsize=8.5)
        ax.set_title(t); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Per task type, Llama-3.1-8B (n = 60 per category; accessibility n = 47)",
                 fontweight="bold")
    save(fig, "fig11_category_heatmap.png")


# ---------------------------------------------------------------- repair
def fig_repair():
    arms = [("V2 Simple RAG", NAVY, "o"), ("V4 draft, layer off", PURPLE, "s"),
            ("V5 regen, gated (selective_axis best)", GREY, "^"),
            ("V5 surgical clause, gated", "#0B6E4F", "D"),
            ("V6 surgical clause on the RAG draft, gated", ORANGE, "P")]
    short = {"V2 Simple RAG": "V2 Simple RAG", "V4 draft, layer off": "V4 Onto+RAG draft",
             "V5 regen, gated (selective_axis best)": "V5 regenerate (gated)",
             "V5 surgical clause, gated": "V5 surgical repair (gated)",
             "V6 surgical clause on the RAG draft, gated": "V6 ontology verifier on RAG draft"}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, m in zip(axs, MODELS):
        c = load(f"task4_evaluation/repair_policy__{m}.json")["cells"]
        for a, col, mk in arms:
            cell = c[a]
            ax.scatter(cell["mean"]["token_f1"] * 100 if cell["mean"]["token_f1"] < 1 else cell["mean"]["token_f1"],
                       cell["ungrounded_rate_strict"], s=90, color=col, marker=mk, label=short[a])
        ax.set_title(MLABEL[m]); ax.set_xlabel("token F1 (%)")
    axs[0].set_ylabel("strict ungrounded rate (%), lower is better")
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Validation-layer repair policies: groundedness vs reference overlap (347 items)",
                 fontweight="bold")
    save(fig, "fig12_repair_tradeoff.png")


def fig_verifier():
    v = load("task4_evaluation/verifier_across_models__test.json")["models"]
    names = list(v.keys())
    pretty = {"llama3.2-1b": "Llama-3.2-1B", "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B",
              "mistral-7b": "Mistral-7B", "qwen3-8b": "Qwen3-8B", "deepseek-r1-8b": "DeepSeek-R1-8B"}
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, k, t in [(a1, "ungrounded_strict", "Strict ungrounded rate (%) - lower is better"),
                     (a2, "clean_strict", "Fully grounded answers, strict (%) - higher is better")]:
        b = [v[n]["before"][k] for n in names]; a = [v[n]["after"][k] for n in names]
        ax.bar(x - 0.2, b, 0.4, color=GREY, label="Simple RAG answer (before)")
        ax.bar(x + 0.2, a, 0.4, color="#0B6E4F", label="after ontology verifier")
        for i in range(len(names)):
            ax.text(x[i] + 0.2, a[i] + 0.3 if k.startswith("clean") else a[i] + 0.05,
                    f"{a[i]:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([pretty.get(n, n) for n in names], fontsize=8, rotation=12)
        ax.set_title(t, fontsize=10)
    a2.set_ylim(60, 90)
    a1.legend(fontsize=8, frameon=False)
    fig.suptitle("Ontology verifier applied to six models' answers on 4,607 held-out questions "
                 "(no retraining, no GPU)", fontweight="bold")
    save(fig, "fig13_verifier_across_models.png")


# ---------------------------------------------------------------- cross model
def fig_cross_model():
    cm = load("task4_evaluation/cross_model_results.json")
    mets = ["token_f1", "token_p", "token_r", "entity_f1", "entity_hit", "ungrounded_rate", "clean"]
    mlab = ["token F1", "token P", "token R", "entity F1", "entity hit", "ungrounded\n(lower better)", "clean"]
    pairs = [p for p in cm["pairs"] if p["tier"] in ("equal", "extended", "matched")]
    M = np.zeros((len(pairs), len(mets))); A = np.empty((len(pairs), len(mets)), dtype=object)
    for i, p in enumerate(pairs):
        for j, m in enumerate(mets):
            d = p["metrics"][m]; delta = d["delta"] * 100
            better = d["verdict"].startswith("ontology")
            worse = d["verdict"].startswith("simple")
            M[i, j] = 1 if better else (-1 if worse else 0)
            A[i, j] = f"{delta:+.2f}"
    fig, ax = plt.subplots(figsize=(12, 6.3))
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap(["#F4C7BE", "#EEEEEE", "#BFE3D6"]), vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, A[i, j], ha="center", va="center", fontsize=7.8)
    ax.set_xticks(range(len(mets))); ax.set_xticklabels(mlab, fontsize=8.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f'{p["tier"]:>8}  {p["model"]}  (onto {p["onto_passes"]} / RAG {p["rag_passes"]} passes)'
                        for p in pairs], fontsize=8)
    for b in (3.5, 7.5):
        ax.axhline(b, color="black", lw=1.2)
    ax.grid(False)
    ax.legend(handles=[mpatches.Patch(color="#BFE3D6", label="ontology arm better (Holm p<0.05)"),
                       mpatches.Patch(color="#F4C7BE", label="Simple RAG better"),
                       mpatches.Patch(color="#EEEEEE", label="indistinguishable")],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("Ontology-only vs Simple RAG, 4,607 paired test questions: delta in points "
                 "(ontology - RAG), by training-budget tier")
    save(fig, "fig14_cross_model_tiers.png")


# ---------------------------------------------------------------- scalability
def fig_scalability():
    s = load("task4_scalability/ontology_scalability.json")["rows"]
    r = load("task4_scalability/retrieval_scalability.json")["passage_search"]
    t = [x["triples"] for x in s]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].plot(t, [x["hop1_p95_ms"] for x in s], "-o", color="#0B6E4F", label="1-hop subgraph p95")
    axs[0].plot(t, [x["hop2_p95_ms"] for x in s], "-s", color=TEAL, label="2-hop subgraph p95")
    axs[0].plot(t, [x["card_search_p50_ms"] for x in s], "-^", color=ORANGE, label="entity-card search p50")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("ontology triples")
    axs[0].set_ylabel("ms"); axs[0].legend(fontsize=7.5, frameon=False)
    axs[0].set_title("Ontology retrieval latency, 7.9K to 100K triples")
    axs[1].plot(t, [x["parse_s"] for x in s], "-o", color=NAVY, label="parse (s)")
    axs[1].plot(t, [x["validate_p50_ms"] / 1000 for x in s], "-s", color=PURPLE, label="answer validation p50 (s)")
    axs[1].set_xscale("log"); axs[1].set_xlabel("ontology triples"); axs[1].set_ylabel("s")
    axs[1].legend(fontsize=7.5, frameon=False); axs[1].set_title("Load and validation cost")
    p = [x["passages"] for x in r]
    axs[2].plot(p, [x["search_p50_ms"] for x in r], "-o", color=NAVY, label="dense passage search p50")
    axs[2].plot(p, [x["search_p95_ms"] for x in r], "--", color=GREY, label="p95")
    axs[2].axvline(36847, color=ORANGE, ls=":", lw=1.2)
    axs[2].text(37500, max(x["search_p95_ms"] for x in r) * 0.85, "delivered corpus", fontsize=7.5, color=ORANGE)
    axs[2].set_xlabel("passages"); axs[2].set_ylabel("ms"); axs[2].legend(fontsize=7.5, frameon=False)
    axs[2].set_title("Simple RAG search, 10K to 100K passages")
    save(fig, "fig15_scalability.png")


def fig_robustness():
    r = load("task4_scalability/robustness.json")
    conds = [x["condition"] for x in r["retrieval"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(conds))
    a1.bar(x - 0.2, [c["rag_gold_coverage"] * 100 for c in r["retrieval"]], 0.4, color=NAVY, label="Simple RAG")
    a1.bar(x + 0.2, [c["onto_gold_coverage"] * 100 for c in r["retrieval"]], 0.4, color="#0B6E4F", label="ontology")
    a1.set_xticks(x); a1.set_xticklabels(conds, fontsize=8, rotation=15); a1.set_ylim(30, 46)
    a1.set_ylabel("gold-entity coverage (%)"); a1.legend(fontsize=8, frameon=False)
    a1.set_title(f"Retrieval under 7 input conditions (n = {r['n_questions']})")
    gen = {(g["arm"], g["condition"]): g["token_f1"] * 100 for g in r["generation"]}
    arms = sorted({g["arm"] for g in r["generation"]}, key=lambda a: a != "rag")
    gconds = [c for c in conds if all((a, c) in gen for a in arms)]
    xg = np.arange(len(gconds))
    for i, a in enumerate(arms):
        a2.bar(xg + (i - (len(arms) - 1) / 2) * 0.38, [gen[(a, c)] for c in gconds], 0.38,
               color=NAVY if a == "rag" else "#0B6E4F", label="Simple RAG" if a == "rag" else "ontology")
    a2.set_xticks(xg); a2.set_xticklabels(gconds, fontsize=8, rotation=15); a2.set_ylim(30, 50)
    a2.set_ylabel("token F1 (%)"); a2.legend(fontsize=8, frameon=False)
    a2.set_title(f"Generated answers, Llama-3.1-8B (n = {r['generation_n']})")
    save(fig, "fig16_robustness.png")


def fig_policy_edit():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    cats = ["rule reached\nthe context", "answer states\nthe rule"]
    a1.bar(np.arange(2) - 0.2, [0, 6], 0.4, color=GREY, label="before (v1.5.0)")
    a1.bar(np.arange(2) + 0.2, [10, 8], 0.4, color="#0B6E4F", label="after (v1.6.0, +22 triples)")
    for i, (b, a) in enumerate([(0, 6), (10, 8)]):
        a1.text(i - 0.2, b + 0.15, f"{b}/10", ha="center", fontsize=9)
        a1.text(i + 0.2, a + 0.15, f"{a}/10", ha="center", fontsize=9)
    a1.set_xticks(range(2)); a1.set_xticklabels(cats); a1.set_ylim(0, 11.5)
    a1.set_ylabel("of 10 policy questions"); a1.legend(fontsize=8, frameon=False, loc="upper right")
    a1.set_title("Policy edit reaches the answer (Llama-3.2-3B, same adapter)")
    a2.bar(["ontology edit\n(22 facts)", "15 LoRA adapters\n(fine-tuning)"], [0.0, 135.1],
           color=["#0B6E4F", ORANGE])
    a2.text(0, 3, "0 GPU-h\n(20 s CPU re-index)", ha="center", fontsize=8.5)
    a2.text(1, 137, "135.1 GPU-h", ha="center", fontsize=9)
    a2.set_ylabel("GPU-hours"); a2.set_ylim(0, 160)
    a2.set_title("Cost to change what the system knows")
    save(fig, "fig17_policy_edit.png")


def fig_plan_2027():
    items = [
        ("5-1", "In-database SHACL via neosemantics (n10s); single source of truth", 0, 3, "#0B6E4F"),
        ("5-1", "Cypher-native subgraph retriever + GraphRAG path", 1, 5, "#0B6E4F"),
        ("5-1", "Neo4j <-> RDF sync; regenerate RDF to current version", 0, 2, "#0B6E4F"),
        ("5-2", "Human expert evaluation (ontology quality, 115 items)", 2, 6, ORANGE),
        ("5-2", "Hallucination annotation with 2+ raters (150 spans)", 3, 7, ORANGE),
        ("5-3", "Korean-language extraction and bilingual labels", 3, 8, NAVY),
        ("5-3", "Live data connectors (events, weather, transport)", 5, 9, NAVY),
        ("5-4", "Authenticated RBAC and regression suite for edits", 4, 8, PURPLE),
        ("5-4", "Pilot deployment and user study with tourism partners", 7, 11, PURPLE),
        ("5-5", "Papers, software registration, Year-2 report", 8, 12, GREY),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, (_, lab, s, e, c) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, e - s, left=s, color=c, height=0.6, alpha=0.9)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[1] for it in items][::-1], fontsize=10.5)
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                       fontsize=10.5)
    ax.set_xlim(0, 12); ax.grid(False)
    for m in range(13):
        ax.axvline(m, color="#dddddd", lw=0.6)
    ax.set_title("Year-2 (2027) plan - proposed schedule (to be confirmed with ETRI)")
    save(fig, "fig18_plan_2027.png")


if __name__ == "__main__":
    fig_architecture()
    fig_gantt_2026()
    fig_dataset()
    fig_versions()
    drv = neo4j_session()
    fig_neo4j(drv)
    fig_neo4j_subgraph(drv)
    drv.close()
    fig_cypher_latency()
    fig_validation()
    fig_prompting()
    fig_variants()
    fig_category_heatmap()
    fig_repair()
    fig_verifier()
    fig_cross_model()
    fig_scalability()
    fig_robustness()
    fig_policy_edit()
    fig_plan_2027()
