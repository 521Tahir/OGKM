"""
Maps every WBS line item to the artifacts that satisfy it, and checks that each named
artifact actually exists on disk. An item whose deliverable is missing is reported as
missing rather than as done.

  python -m wbs.task4_scalability.make_wbs_status
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
D = "wbs/deliverables"

# (task, WBS item text, WBS-named deliverables, files that satisfy it, code)
ITEMS = [
    ("1.1", "Repository structure, experiment tracking, reproducibility policy, risk plan",
     "Project plan", [f"{D}/REPRODUCIBILITY/RUN_ORDER.md",
                      f"{D}/REPRODUCIBILITY/MANIFEST.json"],
     "wbs/task4_scalability/make_repro_package.py"),
    ("1.2", "Data profiling, cleaning, metadata, ontology schema v0",
     "Cleaned dataset, profiling report, schema v0",
     ["data/jeju_full/splits/train.jsonl", "data/jeju_full/splits/test.jsonl",
      "artifacts/ontologies/0.1.0/concepts.json"], "(pre-existing)"),
    ("2.1", "Concept/relation extraction, hierarchy induction, prompt strategies",
     "Extraction modules, prompt library, prompt ablation",
     ["artifacts/ontologies/1.3.0/concepts.json",
      "artifacts/ontologies/1.3.0/relations.json"], "(pre-existing)"),
    ("2.2", "Synonym normalization, duplicate merging, canonical naming, RDF/OWL",
     "RDF/OWL ontology files, quality metrics",
     [f"{D}/task2_formalization/jeju_tourism_ontology.ttl",
      f"{D}/task2_formalization/jeju_tourism_ontology.owl",
      f"{D}/task2_formalization/jeju_tourism_ontology.jsonld",
      f"{D}/task2_formalization/normalization_report.json",
      f"{D}/task2_formalization/ontology_quality_metrics.json"],
     "wbs/task2_formalization/normalize_and_formalize.py"),
    ("2.2", "Logical consistency, structure validation, SHACL, validation reporting",
     "Validation engine, validation reports",
     [f"{D}/task2_formalization/validation_report.md",
      f"{D}/task2_formalization/validation_report.json"],
     "wbs/task2_formalization/validate_ontology.py"),
    ("3.1.1", "Ontology storage (graph + file based)",
     "Ontology storage system",
     ["artifacts/ontologies/current.json"], "(pre-existing)"),
    ("3.1.2", "Versioning, change detection, diff, rollback, governance, provenance",
     "Version control and diff logs, governance and audit trail",
     [f"{D}/task3_governance/version_governance_report.md",
      f"{D}/task3_governance/diff_logs.json",
      f"{D}/task3_governance/audit_trail.jsonl",
      f"{D}/task3_governance/proposals.json",
      f"{D}/task3_governance/current_version.json"],
     "wbs/task3_governance/version_governance.py"),
    ("3.2.1", "Query understanding, concept mapping, subgraph retrieval, hybrid "
              "retrieval, context assembly",
     "Ontology-aware retriever, context assembly module",
     [f"{D}/task3_retrieval/retriever_characterization.md",
      f"{D}/task3_retrieval/retriever_characterization.json"],
     "experiments/exp024_rag_vs_ontology/contexts.py + "
     "wbs/task3_retrieval/characterize_retriever.py"),
    ("3.2.2", "Constrained generation, post-generation validation, repair loop, "
              "explanation tracing, uncertainty fallback",
     "Constrained generation system, validation and repair module, explanation trace",
     [f"{D}/task4_evaluation/variant__llama3.1-8b__V5_ontology_rag_validated.jsonl"],
     "wbs/task3_reasoning/constrained_reasoner.py"),
    ("4.1", "Benchmark datasets for six tourism task types",
     "Benchmark dataset",
     [f"{D}/task4_evaluation/tourism_benchmark_categorized.jsonl",
      f"{D}/task4_evaluation/benchmark_manifest.json"],
     "wbs/task4_evaluation/build_benchmark.py"),
    ("4.1", "Evaluate five system variants (LLM, RAG, ontology-only, ontology-RAG, "
            "ontology-RAG+validation)",
     "Baseline comparison results",
     [f"{D}/task4_evaluation/baseline_comparison__llama3.1-8b.md",
      f"{D}/task4_evaluation/variant_scores__llama3.1-8b.json",
      f"{D}/task4_evaluation/variant_scores__llama3.1-8b.csv"],
     "wbs/task4_evaluation/run_variants.py + score_variants.py"),
    ("4.1", "The ontology-only and Simple RAG arms across eight models on the "
            "4,607-question held-out test set, re-scored on this project's "
            "metrics with the training budget of each arm on the record",
     "Cross-model comparison",
     [f"{D}/task4_evaluation/cross_model_results.md",
      f"{D}/task4_evaluation/cross_model_results.json",
      f"{D}/figures/fig16_cross_model.png"],
     "wbs/task4_evaluation/cross_model_results.py"),
    ("4.1", "Ablation studies on prompts, validation, retrieval and reasoning modules",
     "Ablation study results",
     [f"{D}/task4_evaluation/ablation_offline__llama3.1-8b.md",
      f"{D}/task4_evaluation/ablation_prompts__llama3.1-8b.json",
      f"{D}/task4_evaluation/ablation_validation__llama3.1-8b.json",
      f"{D}/task4_evaluation/ablation_retrieval__llama3.1-8b.json"],
     "wbs/task4_evaluation/ablate_offline.py + ablate_generation.py"),
    ("4.2", "Scalability (10K-100K), latency, throughput, robustness under noise",
     "Scalability report",
     [f"{D}/task4_scalability/ontology_scalability.md",
      f"{D}/task4_scalability/retrieval_scalability.md",
      f"{D}/task4_scalability/robustness.md"],
     "wbs/task4_scalability/{scale_ontology,bench_retrieval,robustness}.py"),
    ("4.2", "Hallucination evaluation",
     "Evaluation and hallucination report",
     [f"{D}/task4_evaluation/hallucination_report__llama3.1-8b.md"],
     "wbs/task4_evaluation/hallucination_report.py"),
    ("4.2", "Visualization (architecture, ontology graphs, retrieval workflow, "
            "hallucination plots)",
     "Visualization figures and tables",
     [f"{D}/figures/FIGURE_INDEX.md", f"{D}/figures/fig01_architecture.png",
      f"{D}/figures/fig02_ontology_graph.png",
      f"{D}/figures/fig05_variant_comparison.png",
      f"{D}/figures/fig06_hallucination.png", f"{D}/figures/fig08_scalability.png",
      f"{D}/figures/fig09_robustness.png"],
     "wbs/task4_scalability/make_figures.py"),
    ("4.2", "Final research outputs and reproducibility package",
     "Final research report, reproducibility package",
     [f"{D}/FINAL_REPORT.md", f"{D}/REPRODUCIBILITY/MANIFEST.json",
      f"{D}/REPRODUCIBILITY/RUN_ORDER.md", f"{D}/REPRODUCIBILITY/requirements.txt"],
     "wbs/task4_scalability/make_final_report.py + make_repro_package.py"),
]


def main() -> None:
    L = ["# WBS Completion Status\n",
         "Each row is a WBS line item, the deliverable the WBS names for it, and the "
         "files that satisfy it. Presence is checked on disk at generation time; a row "
         "with a missing file says so rather than claiming completion.\n",
         "| WBS | Item | WBS deliverable | Artifacts | Status |",
         "|---|---|---|---|---|"]
    done = part = 0
    for task, item, deliv, files, code in ITEMS:
        present = [f for f in files if (ROOT / f).exists()]
        missing = [f for f in files if not (ROOT / f).exists()]
        if not missing:
            status = "✅ complete"; done += 1
        elif present:
            status = f"◐ {len(present)}/{len(files)} present"; part += 1
        else:
            status = "⬜ missing"
        arts = "<br>".join(f"`{Path(f).name}`" for f in present[:6]) or "—"
        if missing:
            arts += "<br>_missing: " + ", ".join(f"`{Path(f).name}`"
                                                 for f in missing[:4]) + "_"
        L.append(f"| {task} | {item} | {deliv} | {arts} | {status} |")
    L.append(f"\n**{done} of {len(ITEMS)} line items complete**" +
             (f", {part} partial." if part else "."))
    L.append("\nCode for each item is listed in "
             "`wbs/deliverables/REPRODUCIBILITY/RUN_ORDER.md` together with the command "
             "that regenerates its deliverable.")
    p = ROOT / "wbs" / "deliverables" / "WBS_STATUS.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    print(f"  {done}/{len(ITEMS)} complete, {part} partial")
    for task, item, _, files, _ in ITEMS:
        miss = [f for f in files if not (ROOT / f).exists()]
        if miss:
            print(f"  [{task}] missing: {', '.join(Path(m).name for m in miss)}")


if __name__ == "__main__":
    main()
