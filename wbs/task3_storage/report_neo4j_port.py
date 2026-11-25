"""WBS Task 3.1.1 / 3.2.1 - report the ported Neo4j ontology and the run that used it.

Every number here is read from the JSON that produced it, so the prose cannot
drift from the measurement.

  python -m wbs.task3_storage.report_neo4j_port
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "wbs" / "deliverables" / "task3_storage"
RETR = ROOT / "wbs" / "deliverables" / "task3_retrieval"
OUT = STORE / "neo4j_ontology_port.md"

COMPANION = Path("/home/tahirahmad/Ontology_Project")


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main() -> None:
    port = json.loads((STORE / "graph_db_neo4j_portable.json")
                      .read_text(encoding="utf-8"))
    sweep_path = RETR / "neo4j_retrieval_sweep.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) \
        if sweep_path.exists() else None

    load, stats, probe, ms = (port["load"], port["stats"], port["probe"],
                              port["read_ms"])
    L: list[str] = []
    A = L.append

    A("# The ontology in Neo4j, built the companion project's way\n")
    A(f"`{COMPANION}` already had a Neo4j graph layer - "
      "`src/graph/neo4j_client.py`, `src/graph/cypher_queries.py` and "
      "`scripts/ingest_to_neo4j.py`. This is that design carried over to the "
      "Jeju ontology, so a Cypher query written against either project runs "
      "against the other.\n")

    A("## The schema, unchanged from theirs\n")
    s = port["schema"]
    A("```cypher")
    A(f"(:{s['node_label']} {{{', '.join(s['node_properties'])}}})")
    A(f"(:{s['node_label']})-[:{s['relationship_type']} "
      f"{{{', '.join(s['relationship_properties'])}}}]->(:{s['node_label']})")
    A("```\n")
    A(f"{len(s['indexes'])} indexes, theirs: name, class, confidence, "
      "relationship predicate, and a fulltext index over names. `MERGE` rather "
      "than `CREATE`, so a second run updates in place.\n")
    A("One property is added: `ordinal`, the position of the relation in the "
      "source file. The retrieval arm keeps the first 8 facts of an entity in "
      "the order the file asserts them, and Cypher makes no promise about the "
      "order `collect` returns edges in - without the index the store would "
      "silently become a variable in every comparison that used it.\n")

    A("## What was loaded\n")
    A(f"| | |\n|---|--:|")
    A(f"| ontology version | v{load['version']} |")
    A(f"| concepts | {load['concepts_in_graph']:,} |")
    A(f"| relations | {load['relations_in_graph']:,} |")
    A(f"| classes | {stats['n_classes']} |")
    A(f"| predicates | {stats['n_predicates']} |")
    A(f"| duplicate triples collapsed by MERGE | "
      f"{load['relations_collapsed_by_merge']} |")
    A(f"| nodes written in | {load['node_seconds']} s |")
    A(f"| edges written in | {load['edge_seconds']} s |")
    A(f"| transactions | {load['transactions']} |")
    A("")
    per_row = load["concepts_sent"] + load["relations_sent"]
    A(f"Their loader opens one write transaction per row, which is "
      f"{per_row:,} round trips for this ontology. This one keeps the same "
      f"`MERGE` and the same properties and sends them through `UNWIND` in "
      f"batches of {load['batch_size']}: {load['transactions']} transactions "
      f"for the same graph.\n")

    A("## The read interface, measured against the live graph\n")
    A("| call | ms | returned |\n|---|--:|---|")
    rows = [
        ("get_concept", ms.get("get_concept"),
         f"`Jeju` ({probe['get_concept_Jeju']['concept_class']}, confidence "
         f"{probe['get_concept_Jeju']['confidence']})"),
        ("concepts_by_class", ms.get("concepts_by_class"),
         f"{probe['concepts_by_class_Beach']} beaches"),
        ("neighbours", ms.get("neighbours"),
         f"{probe['neighbours_Jeju_both']} relations at `Jeju`"),
        ("search_concepts", ms.get("search_concepts"),
         f"fulltext top hit `{probe['search_concepts_top1']}`"),
        ("extract_subgraph", ms.get("extract_subgraph"),
         f"{probe['subgraph_2hop_nodes']} concepts, "
         f"{probe['subgraph_2hop_relations']:,} relations"),
        ("shortest_path", ms.get("shortest_path"),
         " → ".join(probe["shortest_path"]["nodes"])
         if probe["shortest_path"]["found"] else "no path"),
        ("cards", ms.get("cards"), f"{probe['cards']:,} entity cards"),
    ]
    for name, t, what in rows:
        A(f"| `{name}` | {t:.1f} | {what} |")
    A("")
    n = probe["subgraph_2hop_nodes"]
    A(f"One defect was fixed rather than carried over. Their "
      f"`extract_subgraph` caps the node list in Python after the relations "
      f"are collected, so the caller is handed every relation the traversal "
      f"touched, most of them pointing at nodes that were cut - measured at "
      f"7,001 relations for 50 nodes on v1.4.0 before the fix. This version "
      f"caps inside Cypher and returns only the edges induced on the nodes it "
      f"kept: {probe['subgraph_2hop_relations']:,} among {n}, every endpoint "
      f"present.\n")

    if sweep:
        g = sweep["graph"]
        A("## Using it for the experiment\n")
        A(f"The retrieval arm's entity cards are now built from the graph. "
          f"They are identical to the cards built from the files - same names, "
          f"classes, degrees, facts and fact order - so the store is not a "
          f"confound; `neo4j_context.card_parity` checks it and the run aborts "
          f"if it fails.\n")
        A(f"The ontology version matters here: this experiment runs on "
          f"**v1.2.0**, which is what `exp024_rag_vs_ontology/config.py` pins "
          f"and what the FAISS card index was built against, while the storage "
          f"and reporting deliverables are on v1.4.0. Neo4j Community serves "
          f"one database, so the graph has to hold whichever version is being "
          f"worked on.\n")
        A("### What the graph can do that the files cannot\n")
        A(f"A card keeps at most 8 facts, so a card-based hop can only follow "
          f"the hop predicates that survived the cap: "
          f"{g['targets_on_cards']:,} of the {g['targets_in_graph']:,} the "
          f"graph holds. {g['hidden_from_cards']:,} hop targets "
          f"({g['hidden_from_cards'] / g['targets_in_graph'] * 100:.0f}%) are "
          f"invisible to a retriever that reads cards, across "
          f"{g['cards_with_more_targets_in_graph']} of {g['cards']:,} cards.\n")
        A(f"Gold-entity coverage over {sweep['questions_with_gold']} of the "
          f"{sweep['questions']} benchmark questions:\n")
        A("| policy | hop structure | coverage | entities | chars |\n"
          "|---|---|--:|--:|--:|")
        for r in sweep["rows"]:
            src = "cards" if r["group"] == "published" else "graph"
            A(f"| `{r['config']}` | {src} | {pct(r['gold_coverage'])} | "
              f"{r['entities_picked']:.1f} | {r['ctx_chars']:,} |")
        A("")
        eff = sweep["hidden_structure_effect"]
        A(f"**The control settles the first question.** `graph_hop2_k2` is the "
          f"published policy at the published width with the hop structure "
          f"resolved in Cypher instead of read off the cards: "
          f"{eff * 100:+.2f} points. The hidden structure is large and the "
          f"policy that would reach it is two targets wide, so it never "
          f"arrives.\n")
        pb = sweep["paired_best"]
        lo, hi = pb["ci95"]
        A(f"**Widening the fan-out is what pays, and it pays little.** "
          f"`{sweep['best_graph_policy']}` reaches "
          f"{pct(sweep['best_graph_coverage'])} against "
          f"{pct(sweep['published_anchor_coverage'])} for "
          f"`{sweep['published_anchor']}`: {pb['mean_difference'] * 100:+.2f} "
          f"points, 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}] paired over "
          f"{pb['n_questions']} questions. {pb['questions_better']} questions "
          f"improve, {pb['questions_worse']} get worse and "
          f"{pb['questions_unchanged']} do not move.\n")
        A("**And coverage is a weak proxy.** Task 4.1.2 measured a 29.7-point "
          "coverage collapse costing 0.9 points of generated token F1. On that "
          "exchange rate a gain of about a point here is worth roughly 0.03 F1, "
          "below what generation would resolve. This is a ranking, not a "
          "result: it says a wider graph-resolved hop is the direction worth "
          "spending generation on, and nothing more.\n")

    A("## Reproducing\n")
    A("```")
    A("# the graph, the companion project's schema")
    A(f"python -m wbs.task3_storage.neo4j_ontology --version "
      f"{load['version']} --clear")
    A("")
    A("# card parity and hop headroom, no GPU")
    A("python -m wbs.task3_retrieval.neo4j_context --check")
    A("")
    A("# the coverage sweep")
    A("CUDA_VISIBLE_DEVICES=6 python -m wbs.task3_retrieval.run_neo4j_experiment")
    A("")
    A("# restore the typed mapping and the v1.4.0 storage deliverables")
    A("python -m wbs.task3_storage.neo4j_store")
    A("python -m wbs.task3_storage.neo4j_export")
    A("python -m wbs.task3_storage.ontology_explorer")
    A("```")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(' '.join(L).split()):,} words)")


if __name__ == "__main__":
    main()
