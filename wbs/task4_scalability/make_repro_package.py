"""
WBS Task 4.2.2 - Reproducibility package.

Assembles everything needed to re-run the four WBS tasks and check that the inputs
are the ones the reported numbers came from:

  MANIFEST.json   sha256 of every input artifact (ontology versions, data splits,
                  benchmark, indices) and of every deliverable, plus the environment
                  the numbers were produced in (library versions, GPU, driver) and
                  every seed used.
  RUN_ORDER.md    the commands, in dependency order, with what each one writes and
                  roughly how long it takes.
  requirements.txt  the resolved versions of the libraries that matter here.

Checksums are of files, not of claims: a reader can verify that the ontology they
have is byte-identical to the one these numbers were computed on.

  python -m wbs.task4_scalability.make_repro_package
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DELIV = ROOT / "wbs" / "deliverables"
OUT = DELIV / "REPRODUCIBILITY"

INPUTS = [
    "artifacts/ontologies/1.3.0/concepts.json",
    "artifacts/ontologies/1.3.0/relations.json",
    "artifacts/ontologies/1.3.0-norm/concepts.json",
    "artifacts/ontologies/1.3.0-norm/relations.json",
    "artifacts/ontologies/1.4.0/concepts.json",
    "artifacts/ontologies/1.4.0/relations.json",
    "data/jeju_full/splits/train.jsonl",
    "data/jeju_full/splits/val.jsonl",
    "data/jeju_full/splits/test.jsonl",
    "data/jeju_full/jeju_gazetteer.json",
    "experiments/exp025_multihop/outputs/jeju_bench_200.jsonl",
]
CODE = [
    "wbs/task2_formalization/normalize_and_formalize.py",
    "wbs/task2_formalization/validate_ontology.py",
    "wbs/task3_governance/version_governance.py",
    "wbs/task3_governance/access_control.py",
    "wbs/task3_storage/sparql_store.py",
    "wbs/task3_storage/graph_db.py",
    "wbs/task2_extraction/prompt_sensitivity.py",
    "wbs/task2_extraction/report_sensitivity.py",
    "wbs/task3_retrieval/characterize_retriever.py",
    "wbs/task3_reasoning/constrained_reasoner.py",
    "wbs/task4_evaluation/build_benchmark.py",
    "wbs/task4_evaluation/run_variants.py",
    "wbs/task4_evaluation/score_variants.py",
    "wbs/task4_evaluation/ablate_offline.py",
    "wbs/task4_evaluation/ablate_generation.py",
    "wbs/task4_scalability/scale_ontology.py",
    "wbs/task4_scalability/bench_retrieval.py",
    "wbs/task4_scalability/robustness.py",
    "wbs/task4_scalability/make_figures.py",
    "wbs/task4_scalability/make_repro_package.py",
    "wbs/task4_scalability/make_final_report.py",
    "wbs/task3_storage/neo4j_store.py",
    "wbs/task3_storage/neo4j_schema.py",
    "wbs/task3_storage/neo4j_export.py",
    "wbs/task3_storage/neo4j_ontology.py",
    "wbs/task3_storage/report_neo4j_port.py",
    "wbs/task3_retrieval/neo4j_context.py",
    "wbs/task3_retrieval/run_neo4j_experiment.py",
    "wbs/task6_report/make_paper_figures.py",
    "wbs/task3_storage/ontology_explorer.py",
    "wbs/task3_storage/ontology_poster.py",
    "wbs/task3_storage/neo4j_server.sh",
    "wbs/task6_report/form_layout.py",
    "wbs/task6_report/make_annual_report.py",
    "wbs/task4_scalability/make_wbs_status.py",
    "wbs/task4_scalability/audit_wbs.py",
    "wbs/task4_scalability/audit_proposal.py",
    "wbs/task4_scalability/check_citations.py",
    "wbs/task2_formalization/rule_compliance.py",
    "wbs/task2_formalization/element_validation_status.py",
    "wbs/task2_extraction/hypothesis_h1.py",
    "wbs/task3_retrieval/tuned_context.py",
    "wbs/task4_evaluation/build_hybrid_train.py",
    "wbs/task3_retrieval/improve_onto_context.py",
    "wbs/task3_retrieval/generate_tuned_onto.py",
    "wbs/task3_retrieval/report_tuning.py",
    "wbs/task4_evaluation/hypotheses_h2_h3.py",
    "wbs/task4_evaluation/consistency.py",
    "wbs/task4_evaluation/report_hypotheses.py",
    "wbs/task4_evaluation/expert_review_pack.py",
    "wbs/task4_evaluation/score_human_review.py",
    "wbs/task4_evaluation/cross_model_results.py",
    "wbs/task4_evaluation/report_hybrid_adapter.py",
    "wbs/task3_reasoning/constraint_coverage.py",
    "wbs/task2_formalization/carry_rule_layer.py",
    "wbs/task4_evaluation/report_detector_sensitivity.py",
    "wbs/task4_scalability/ann_index.py",
    "wbs/task4_evaluation/selective_validation.py",
    "wbs/task4_evaluation/report_selective_axis.py",
    "wbs/task4_evaluation/report_ontology_only_validated.py",
    "wbs/task4_evaluation/coverage_routing.py",
    "wbs/task2_formalization/extract_weather_states.py",
    "wbs/task2_formalization/extract_policy_constraints.py",
    "wbs/task4_evaluation/reference_paraphrase.py",
    "wbs/task4_evaluation/repair_policy.py",
    "wbs/task4_evaluation/report_repair_policy.py",
    "wbs/task4_evaluation/verifier_across_models.py",
    "wbs/task4_scalability/make_goal_status.py",
    "wbs/task4_scalability/fill_wbs_workbook.py",
    "wbs/task5_demo/demo_policy_before_after.py",
    "wbs/task5_demo/check_deck.py",
    "wbs/task5_demo/text_metrics.py",
    "wbs/task5_demo/demo_live_edit.py",
    "wbs/task5_demo/build_demo_evidence.py",
    "wbs/task5_demo/make_deck_figures.py",
    "wbs/task5_demo/make_slides.py",
    "wbs/task5_demo/video_kit.py",
    "wbs/task5_demo/make_demo_video.py",
    "wbs/rates.py",
    "tests/test_rates.py",
    "experiments/exp024_rag_vs_ontology/contexts.py",
    "experiments/exp024_rag_vs_ontology/config.py",
    "experiments/exp024_rag_vs_ontology/chat.py",
]

RUN_ORDER = [
    ("Task 2.2", "ontology normalization + RDF/OWL serialization",
     "python -m wbs.task2_formalization.normalize_and_formalize", "~1 min", "CPU"),
    ("Task 2.2", "4-layer validation (structural, typing, SHACL, OWL-RL)",
     "python -m wbs.task2_formalization.validate_ontology", "~2 min", "CPU"),
    ("Task 3.1.2", "version snapshots, diffs, governance workflow, audit trail",
     "python -m wbs.task3_governance.version_governance --all", "~1 min", "CPU"),
    ("Task 2.1.2", "prompt-strategy sensitivity and stability sweep, shard 1 of 2",
     "python -m wbs.task2_extraction.prompt_sensitivity --gpu 5 "
     "--strategies naive cot graph_of_thought --n-records 15 "
     "--stability-runs 3 --sensitivity-samples 3 --tag gpu5", "~3.2 h", "1 GPU"),
    ("Task 2.1.2", "same sweep, shard 2 of 2 (runs concurrently on a second GPU)",
     "python -m wbs.task2_extraction.prompt_sensitivity --gpu 6 "
     "--strategies metacognitive repair_after_failure --n-records 15 "
     "--stability-runs 3 --sensitivity-samples 3 --tag gpu6", "~2.1 h", "1 GPU"),
    ("Task 2.1.2", "merge both shards into the stability/sensitivity report",
     "python -m wbs.task2_extraction.report_sensitivity", "~1 s", "CPU"),
    ("Task 3.1.1", "RDF store + SPARQL competency questions (in-process)",
     "python -m wbs.task3_storage.sparql_store", "~1 min", "CPU"),
    ("Task 3.1.1", "persistent Oxigraph graph database: load, bench, update, backup",
     "python -m wbs.task3_storage.graph_db --build --rebuild", "~1 min", "CPU"),
    ("Task 3.1.1", "SPARQL 1.1 Protocol endpoint, then the same queries over HTTP",
     "python -m wbs.task3_storage.graph_db --serve  # then: "
     "python -m wbs.task3_storage.sparql_store "
     "--sparql-endpoint http://127.0.0.1:8631/sparql", "~1 min", "CPU"),
    ("Proposal 11.4.1", "rule satisfaction, constraint adherence, redundancy, "
     "conflict and hierarchical balance",
     "python -m wbs.task2_formalization.rule_compliance", "~5 s", "CPU"),
    ("Proposal 9.4", "per-element validation verdicts, so retrieval can rank on them",
     "python -m wbs.task2_formalization.element_validation_status", "~20 s", "CPU"),
    ("Proposal 11.4.2 / 11.6b", "expert-review and blinded annotation instruments",
     "python -m wbs.task4_evaluation.expert_review_pack", "~10 s", "CPU"),
    ("Proposal 11.4.2 / 11.6b", "agreement and scoring for those instruments: "
     "Krippendorff alpha, Cohen kappa, the adjudication list, and the unblinded "
     "detector precision/recall - reported as undefined while fewer than two "
     "rater sheets exist, rather than as a mean of one opinion",
     "python -m wbs.task4_evaluation.score_human_review", "~5 s", "CPU"),
    ("Proposal 11.4.2 / 11.6b", "the scorer's own verification, including the "
     "alpha implementation against Krippendorff's published example",
     "python -m wbs.task4_evaluation.score_human_review --self-test",
     "~2 s", "CPU"),
    ("Task 4.1.2", "the ontology-only arm across eight models: reads each arm's "
     "training receipt from the companion project, re-scores all 20 stored arms "
     "over 4,607 held-out questions on this project's metric stack, and attaches "
     "the paired Wilcoxon and bootstrap tests the companion project's multi-pass "
     "results were published without. Reads "
     "/home/tahirahmad/Ontology_Project_Codex read-only; writes nothing there",
     "python -m wbs.task4_evaluation.cross_model_results --workers 8",
     "~3 min", "CPU"),
    ("Task 4.1.2", "the budget receipts and the router firing rate alone, without "
     "re-scoring",
     "python -m wbs.task4_evaluation.cross_model_results --harvest-only",
     "~5 s", "CPU"),
    ("Proposal 8.7", "role-based access control over knowledge changes: the "
     "policy, the principal registry, the protected-element set derived from "
     "the published ontology, and twelve enforcement attempts as five "
     "principals",
     "python -m wbs.task3_governance.access_control --report && "
     "python -m wbs.task3_governance.access_control --demo", "~10 s", "CPU"),
    ("Task 3.2.1", "ontology context selection policy sweep (offline, coverage)",
     "python -m wbs.task3_retrieval.improve_onto_context --device cpu", "~3 min", "CPU"),
    ("Task 3.2.1", "generation for the shortlisted policies, shard 1 of 2",
     "python -m wbs.task3_retrieval.generate_tuned_onto --gpu 5 "
     "--policies baseline routing_deep_hop2 --tag gpu5", "~1.5 h", "1 GPU"),
    ("Task 3.2.1", "generation for the shortlisted policies, shard 2 of 4",
     "python -m wbs.task3_retrieval.generate_tuned_onto --gpu 6 "
     "--policies round_robin global_degree80 --tag gpu6", "~1.5 h", "1 GPU"),
    ("Task 3.2.1", "generation, shard 3 of 4 - validation-status rank (9.4) and "
     "grouped encoding with deep routing",
     "python -m wbs.task3_retrieval.generate_tuned_onto --gpu 5 "
     "--policies routing_deep_hop2_valrank grouped_deep --tag gpu5b", "~1.5 h", "1 GPU"),
    ("Task 3.2.1", "generation, shard 4 of 4 - grouped encoding at equal and at "
     "doubled facts per card",
     "python -m wbs.task3_retrieval.generate_tuned_onto --gpu 6 "
     "--policies grouped_wide grouped --tag gpu6b", "~1.5 h", "1 GPU"),
    ("Task 3.2.1", "merge sweep and generation into the tuning report",
     "python -m wbs.task3_retrieval.report_tuning", "~10 s", "CPU"),
    ("Task 4.1", "training contexts for a dedicated V4/V5 hybrid adapter",
     "python -m wbs.task4_evaluation.build_hybrid_train --splits train val",
     "~1 min", "CPU"),
    ("Task 4.1", "train that adapter - removes the mixed-adapter confound from "
     "the two arms that combine ontology facts with retrieved prose",
     "EXP024_TAG=ded python -m experiments.exp024_rag_vs_ontology.train_lora "
     "--model llama3.1-8b --arm hybrid20d --gpu 6", "~11 h", "1 GPU"),
    ("Task 4.1", "re-run V4/V5 on the dedicated hybrid adapter, into tagged files "
     "so the published run is preserved",
     "python -m wbs.task4_evaluation.run_variants --gpu 6 --only V4_ontology_rag "
     "V5_ontology_rag_validated --hybrid-adapter hybrid20d --out-tag __hybded",
     "~1.5 h", "1 GPU"),
    ("Task 4.1 / R16", "the same adapter and re-run at 3B and 1B, so no hybrid "
     "claim below 8B carries the confound either",
     "EXP024_TAG=ded python -m experiments.exp024_rag_vs_ontology.train_lora "
     "--model llama3.2-3b --arm hybrid20d --gpu 6  # then --model llama3.2-1b; "
     "then run_variants --model llama3.2-{3b,1b} --hybrid-adapter hybrid20d "
     "--out-tag __hybded", "~6 h + ~1 h", "1 GPU"),
    ("Task 4.1 / R2", "the validation ablation re-run with the audit-grade span "
     "detector, which is the open action the risk register carried",
     "python -m wbs.task4_evaluation.ablate_generation --axis validation --gpu 5 "
     "--detector audit --limit 150 --seed 20260801", "~40 min", "1 GPU"),
    ("Task 4.1 / R2", "sensitivity of every groundedness comparison to the one "
     "generous rule in the audit filter",
     "python -m wbs.task4_evaluation.report_detector_sensitivity", "~4 min", "CPU"),
    ("Task 4.1", "spend the validation layer only where it pays: a "
     "cross-validated threshold on a draft-only signal, built from the stored "
     "V4/V5 runs with no new generation",
     "python -m wbs.task4_evaluation.selective_validation", "~3 min", "CPU"),
    ("Task 4.1", "the 2x2 that tests whether the two fixes to the validation "
     "layer compose: gate on the layer's own detector, crossed with abstaining on "
     "the coverage floor or only on an unsupportable claim, detector held at "
     "audit-grade in every cell",
     "python -m wbs.task4_evaluation.ablate_generation --axis selective --gpu 5 "
     "--model llama3.1-8b --limit 400 --detector audit "
     "--hybrid-adapter hybrid20d", "~1.5 h", "1 GPU"),
    ("Task 4.1 / R2", "the same 2x2 at 3B and 1B, so the gate - the project's "
     "deployable recommendation - is not quoted from one model size",
     "python -m wbs.task4_evaluation.ablate_generation --axis selective --gpu 5 "
     "--model llama3.2-3b --limit 400 --detector audit --hybrid-adapter "
     "hybrid20d  # then --model llama3.2-1b --gpu 6", "~2 h", "2 GPUs"),
    ("Task 4.1", "report that 2x2: cell means under the canonical scorer, each "
     "factor's marginal effect, the interaction, the strict-rule check and the "
     "cross-size transfer table",
     "python -m wbs.task4_evaluation.report_selective_axis --model llama3.2-1b "
     "# then --model llama3.2-3b, then --model llama3.1-8b, which builds the "
     "cross-size table from the other two", "~12 min", "CPU"),
    ("Task 4.1", "the same 2x2 on ontology context alone - no retrieved prose "
     "in the context at all - which is the arm the project's own comparison "
     "names and the one that had never been run with the layer in front of it",
     "python -m wbs.task4_evaluation.ablate_generation --axis selective --gpu 6 "
     "--model llama3.1-8b --limit 400 --detector audit "
     "--draft-arm V3_ontology_only", "~1 h", "1 GPU"),
    ("Task 4.1 / R17", "the same ontology-only 2x2 at 3B and 1B, so the "
     "decomposition is not quoted from one model size",
     "python -m wbs.task4_evaluation.ablate_generation --axis selective --gpu 5 "
     "--model llama3.2-3b --limit 400 --detector audit "
     "--draft-arm V3_ontology_only  # then --model llama3.2-1b --gpu 6",
     "~1 h", "2 GPUs"),
    ("Task 4.1", "report it against Simple RAG with Holm over the whole table, "
     "and against the same cell on the hybrid context, to separate what the "
     "ontology supplies from what retrieval supplies",
     "python -m wbs.task4_evaluation.report_ontology_only_validated --model "
     "llama3.2-1b  # then --model llama3.2-3b, then --model llama3.1-8b, which "
     "builds the cross-size table from the other two", "~12 min", "CPU"),
    ("Task 2.1 / R12", "extract the weather state two blocked rules need from "
     "the corpus, write it as an additive ontology version beside the "
     "published one, and measure what it unblocks",
     "python -m wbs.task2_formalization.extract_weather_states --write 1.5.0 && "
     "python -m wbs.task2_formalization.validate_ontology --version 1.5.0 "
     "--out-tag __1.5.0 && python -m wbs.task2_formalization.rule_compliance "
     "--versions 1.2.0 1.5.0 --out-tag __1.5.0 && "
     "python -m wbs.task2_formalization.extract_weather_states --write 1.5.0 "
     "--reuse-existing  # second pass writes the report with the delta",
     "~12 min", "CPU"),
    ("Task 4.1 / R7", "where the overlap deficit lives, and the bound on "
     "fixing it by choosing an arm per question: per-item gold-span coverage, "
     "a prose-density stratification, cross-validated question-side routers and "
     "an oracle selector. No generation; reads the stored runs",
     "python -m wbs.task4_evaluation.coverage_routing --model llama3.1-8b  "
     "# then --model llama3.2-3b, --model llama3.2-1b", "~9 min", "CPU"),
    ("Task 4.1 / R6", "test whether the overlap metric is measuring corpus "
     "copying: paraphrase every single-source reference with the base model, "
     "keep only paraphrases that preserve the gazetteer entities and the "
     "length band, and re-score every stored arm against them",
     "python -m wbs.task4_evaluation.reference_paraphrase --stage generate "
     "--gpu 5 && python -m wbs.task4_evaluation.reference_paraphrase "
     "--stage report", "~35 min", "1 GPU"),
    ("Task 4.2 / R8", "price the approximate index the scalability report named: "
     "recall on the real index, latency on resampled real vectors to 100K cards",
     "python -m wbs.task4_scalability.ann_index", "~8 min", "CPU"),
    ("Proposal H1", "per-strategy ontology consistency and hierarchy, sample 0",
     "python -m wbs.task2_extraction.hypothesis_h1 --gpu 5 --n-records 20 "
     "--sample 0 --tag s0", "~1.1 h", "1 GPU"),
    ("Proposal H1", "same, disjoint record sample 1, to test whether H1 survives",
     "python -m wbs.task2_extraction.hypothesis_h1 --gpu 6 --n-records 20 "
     "--sample 1 --tag s1", "~1.1 h", "1 GPU"),
    ("Proposal H2 / H3", "answer fluency under the base model, and attribution of "
     "each answer to its own evidence",
     "python -m wbs.task4_evaluation.hypotheses_h2_h3 --gpu 5", "~25 min", "1 GPU"),
    ("Proposal 11.5b", "paraphrase and repeat consistency, shard 1 of 2",
     "python -m wbs.task4_evaluation.consistency --gpu 5 --arms rag onto_v3 "
     "--tag gpu5", "~1.5 h", "1 GPU"),
    ("Proposal 11.5b", "paraphrase and repeat consistency, shard 2 of 2",
     "python -m wbs.task4_evaluation.consistency --gpu 6 --arms hybrid20 "
     "--tag gpu6", "~45 min", "1 GPU"),
    ("Proposal 3.3", "verdicts for H1, H2, H3 and the consistency table",
     "python -m wbs.task4_evaluation.report_hypotheses", "~5 s", "CPU"),
    ("Audit", "compliance against the research proposal (49 requirements)",
     "python -m wbs.task4_scalability.audit_proposal", "~5 s", "CPU"),
    ("Audit", "every artifact path cited in docs/ resolved against disk",
     "python -m wbs.task4_scalability.check_citations", "~20 s", "CPU"),
    ("Task 3.2.1", "retriever characterization",
     "python -m wbs.task3_retrieval.characterize_retriever --device cpu --n-test 1500",
     "~10 min", "CPU"),
    ("Task 3.2.2", "constrained reasoner self-test",
     "python -m wbs.task3_reasoning.constrained_reasoner --selftest", "seconds", "CPU"),
    ("Task 4.1", "build the six-category benchmark",
     "python -m wbs.task4_evaluation.build_benchmark --per-category 60", "~1 min", "CPU"),
    ("Task 4.1", "run variants V1-V3",
     "python -m wbs.task4_evaluation.run_variants --gpu 5 --model llama3.1-8b "
     "--only V1_llm V2_rag V3_ontology_only", "~2 h", "1 GPU"),
    ("Task 4.1", "run variants V4-V5",
     "python -m wbs.task4_evaluation.run_variants --gpu 6 --model llama3.1-8b "
     "--only V4_ontology_rag V5_ontology_rag_validated", "~1.5 h", "1 GPU"),
    ("Task 4.1", "score the variants and audit groundedness",
     "python -m wbs.task4_evaluation.score_variants --model llama3.1-8b", "~1 min", "CPU"),
    ("Task 4.1", "ablations that need no generation",
     "python -m wbs.task4_evaluation.ablate_offline --device cpu", "~2 min", "CPU"),
    ("Task 4.1", "ablations that need generation",
     "python -m wbs.task4_evaluation.ablate_generation --axis prompts --gpu 5 --limit 150",
     "~1 h per axis", "1 GPU"),
    ("Task 4.2", "ontology scalability 10K-100K triples",
     "python -m wbs.task4_scalability.scale_ontology --scales 7909 10000 25000 50000 "
     "100000 --owlrl-cap 25000", "~5 min", "CPU"),
    ("Task 4.2", "retrieval scalability, latency, throughput",
     "python -m wbs.task4_scalability.bench_retrieval --device cuda:0", "~10 min",
     "1 GPU (or --device cpu)"),
    ("Task 4.2", "robustness under seven noise conditions",
     "python -m wbs.task4_scalability.robustness --device cpu", "~15 min", "CPU"),
    ("Task 4.2", "all figures",
     "python -m wbs.task4_scalability.make_figures", "~1 min", "CPU"),
    ("Task 4.1 / R19", "the repair-policy comparison: surgical clause removal "
     "against the published re-prompting repair and against Simple RAG, at "
     "three sizes, with the independent strict-band rate beside every "
     "constructional zero. No GPU - every cell is a transformation of "
     "generations already on disk",
     "python -m wbs.task4_evaluation.repair_policy", "~3 min", "CPU"),
    ("Task 4.1 / R19", "report it",
     "python -m wbs.task4_evaluation.report_repair_policy", "~2 s", "CPU"),
    ("Task 4.1 / R19", "the same verifier run unchanged over six models' stored "
     "Simple RAG answers, including three families this pipeline was never tuned "
     "for, to separate grounding from a fit to one model",
     "python -m wbs.task4_evaluation.verifier_across_models --split test  "
     "# then --split bench", "~1 min", "CPU"),
    ("Task 4.2", "a decision-ready statement of what the standing goal asks, "
     "what was measured against it, and what each open item needs from outside "
     "this pipeline",
     "python -m wbs.task4_scalability.make_goal_status", "~2 s", "CPU"),
    ("Task 4.2", "the delivered WBS workbook handed back with 진척율 and "
     "산출물 filled from the on-disk audit, plus an evidence-path column",
     "python -m wbs.task4_scalability.fill_wbs_workbook", "~2 s", "CPU"),
    ("Task 2.1", "the policy layer the schema declared and nothing populated: "
     "requirements and restrictions extracted from the corpus with a proximity "
     "window, a prohibition cue and a coordinates gate, written as an additive "
     "version beside the published one and checked against the domain and "
     "range it declares",
     "python -m wbs.task2_formalization.extract_policy_constraints --write "
     "1.6.0 && python -m wbs.task2_formalization.validate_ontology --version "
     "1.6.0 --out-tag __1.6.0", "~6 min", "CPU"),
    ("Demo / R21 / R22", "the before-and-after the business brief asks for: "
     "ten questions the policy edit speaks to and six controls it does not, "
     "asked on either side of the edit with one adapter loaded once",
     "EXP024_TAG=ded python -m wbs.task5_demo.demo_policy_before_after --gpu 5 "
     "--model llama3.2-3b  # then --model llama3.2-1b", "~8 min", "1 GPU"),
    ("Demo", "the live ontology edit: one small model, one adapter, the same "
     "question asked against two ontology versions, with the adapter digests "
     "checked before and after",
     "python -m wbs.task5_demo.demo_live_edit --gpu 5", "~3 min", "1 GPU"),
    ("Demo", "every number and quoted answer the deck and the film may use",
     "python -m wbs.task5_demo.build_demo_evidence", "~20 s", "CPU"),
    ("Demo", "the two charts drawn for the business audience",
     "python -m wbs.task5_demo.make_deck_figures", "~10 s", "CPU"),
    ("Demo", "the business-plan deck",
     "python -m wbs.task5_demo.make_slides", "~5 s", "CPU"),
    ("Demo", "check the built deck: no shape off the slide, no text frame "
     "whose wrapped lines run into what sits under it",
     "python -m wbs.task5_demo.check_deck", "~5 s", "CPU"),
    ("Demo", "the narrated demo film (piper voice, PIL frames, ffmpeg mux)",
     "python -m wbs.task5_demo.make_demo_video", "~12 min", "CPU"),
    ("Task 4.2", "final report",
     "python -m wbs.task4_scalability.make_final_report", "~1 min", "CPU"),
    ("Task 3.1.1", "the property-graph half of \"RDF/graph database\": a Neo4j 5.26 server loaded with the same ontology, answering the same nine competency questions in Cypher under two mappings, and compared with the SPARQL path on row counts. Needs a JVM; `neo4j_server.sh start` brings the server up on Bolt 7687",
     "bash wbs/task3_storage/neo4j_server.sh start && python -m wbs.task3_storage.neo4j_store", "~1 min", "CPU"),
    ("Task 3.1.1", "the full ontology schema read back out of the live Neo4j server - labels and counts, relationship types, the domain and range the data actually has, properties, constraints, the class hierarchy and the governance-added elements the RDF paths cannot see",
     "python -m wbs.task3_storage.neo4j_schema", "~5 s", "CPU"),
    ("Task 3.1.1", "the complete ontology read back out of the live Neo4j "
     "server over Cypher, verified against the file store by name and by "
     "triple, and written as the snapshot the figures and the explorer are "
     "built from - plus `whole_ontology.cypher`, the ordered Browser script",
     "python -m wbs.task3_storage.neo4j_export", "~5 s", "CPU"),
    ("Task 3.1.1", "the ontology loaded under the schema the companion "
     "project at `Ontology_Project` uses - one :Concept label, one "
     "[:RELATES_TO] type carrying the predicate, five indexes including a "
     "fulltext one, MERGE rather than CREATE - with its whole read interface "
     "exercised against the live graph. Neo4j Community serves one database, "
     "so --clear chooses which of the two mappings is resident",
     "python -m wbs.task3_storage.neo4j_ontology --version 1.2.0 --clear",
     "~5 s", "CPU"),
    ("Task 3.2.1", "the retrieval arm sourced from that graph: entity cards "
     "built in Cypher and checked identical to the file-built ones, then a "
     "gold-coverage sweep asking whether the hop structure the 8-fact card cap "
     "hides is worth following. Needs the graph to hold v1.2.0, the version "
     "the experiment is pinned to",
     "CUDA_VISIBLE_DEVICES=6 python -m "
     "wbs.task3_retrieval.run_neo4j_experiment", "~2 min", "GPU"),
    ("Task 3.1.1", "the ported graph layer and the run that used it, reported "
     "from the JSON that measured them",
     "python -m wbs.task3_storage.report_neo4j_port", "~1 s", "CPU"),
    ("Task 4.2.2", "the entire ontology as a single image: every concept and "
     "every relation, once force-directed and once clustered by class, plus the "
     "coordinates the interactive explorer reuses so the two agree. Drawn "
     "from the Neo4j snapshot by default, so the printed figures and the "
     "server hold the same graph; `--source files` draws the file store",
     "python -m wbs.task3_storage.ontology_poster --dpi 150 --svg",
     "~5 min", "CPU"),
    ("Task 3.1.1", "the whole ontology as one self-contained HTML page - no "
     "server, no network, no CDN: class-level schema graph, per-class instance "
     "browser with neighbourhoods, search over names and aliases, and an "
     "Everything view drawn from the poster's own coordinates, and a Neo4j "
     "view carrying the server it was built from, the verification result and "
     "the Cypher behind each picture",
     "python -m wbs.task3_storage.ontology_explorer", "~2 s", "CPU"),
    ("Task 6", "the two paper figures: the ontology's complete structure "
     "(class taxonomy beside the schema-level relation graph, read from the "
     "Neo4j snapshot) and the qualitative Simple-RAG vs ontology comparison "
     "(four real benchmark cases, with the base rate of each direction stated "
     "on the figure). PNG and vector PDF",
     "python -m wbs.task6_report.make_paper_figures", "~20 s", "CPU"),
    ("Annual report", "the English annual report on the ETRI joint-institution form, with every quoted number read from the deliverable that owns it",
     "python -m wbs.task6_report.make_annual_report", "~5 s", "CPU"),
    ("Task 4.2", "this package",
     "python -m wbs.task4_scalability.make_repro_package", "~1 min", "CPU"),
]

SEEDS = {
    "benchmark construction": 20260801,
    "robustness perturbations": 20260801,
    "ablation subset sampling": 20260801,
    "retriever characterization sampling": 20260801,
    "ontology replication (scalability)": 20260801,
    "decoding": "greedy (do_sample=False) everywhere - no sampling seed is used",
}


def sha(p: Path) -> str | None:
    if not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    def entry(rel: str) -> dict:
        p = ROOT / rel
        return {"path": rel, "sha256": sha(p),
                "bytes": p.stat().st_size if p.exists() and p.is_file() else None,
                "missing": not p.exists()}

    inputs = [entry(r) for r in INPUTS]
    code = [entry(r) for r in CODE]

    # §5.1k reads stored answers from the companion project. They are outside this
    # repository and cannot be bundled, so the manifest names every one of them with
    # the sha256 recorded at the time they were scored. A reader without that project
    # can re-derive nothing here, and the note says so instead of implying otherwise.
    external = {"root": "/home/tahirahmad/Ontology_Project_Codex",
                "used_by": "wbs/task4_evaluation/cross_model_results.py",
                "access": "read-only; this project writes nothing there",
                "files": []}
    xm_path = DELIV / "task4_evaluation" / "cross_model_results.json"
    if xm_path.exists():
        xm = json.loads(xm_path.read_text(encoding="utf-8"))
        seen, own = {}, {}
        for pair in xm["pairs"]:
            for arm, src in pair["sources"].items():
                rec = {"path": src["path"], "sha256": src["sha256"],
                       "rows": src["rows"], "arm": arm,
                       "experiment": src["experiment"]}
                # The matched-budget arms are generated by this project, so listing
                # them under the companion project's root would misattribute them.
                (seen if src.get("project", "codex") == "codex" else own)[src["path"]] = rec
        external["files"] = sorted(seen.values(), key=lambda d: d["path"])
        external["count"] = len(external["files"])
        if own:
            external["generated_here"] = {
                "root": str(ROOT),
                "note": "Produced by experiments/exp060_matched_budget_rag to give the "
                        "extended-training comparison a baseline on the same budget; "
                        "reproduced by the RUN_ORDER steps, not downloaded.",
                "files": sorted(own.values(), key=lambda d: d["path"]),
                "count": len(own)}
    delivs = []
    for p in sorted(DELIV.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".json", ".csv", ".jsonl", ".ttl",
                                        ".owl", ".jsonld", ".png") \
                and "REPRODUCIBILITY" not in p.parts:
            delivs.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p),
                           "bytes": p.stat().st_size})

    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "sentence_transformers", "faiss",
                "rdflib", "pyshacl", "owlrl", "numpy", "scipy", "matplotlib",
                "networkx"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = "not installed"
    try:
        env["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.strip().splitlines()[:1]
    except Exception:
        env["gpu"] = "nvidia-smi unavailable"
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True,
                                           timeout=20).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                               capture_output=True, text=True,
                                               timeout=30).stdout.strip())
    except Exception:
        env["git_commit"] = "unavailable"

    man = {"generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
           "environment": env, "seeds": SEEDS,
           "inputs": inputs, "code": code, "deliverables": delivs,
           "external_inputs": external,
           "adapter_root": "/ceph_data/tahirahmad/ontology_exp024/adapters_ded "
                           "(arm-dedicated) and .../adapters (shared 'mixed')",
           "note": "Adapters are LoRA weights produced by the project's own training "
                   "runs and are not checksummed here because they live outside the "
                   "repository; the arm each variant uses is named in "
                   "wbs/task4_evaluation/run_variants.py."}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    L = ["# Reproducibility Package\n",
         f"Generated {man['generated']}. Every command below writes into "
         "`wbs/deliverables/`; nothing in this project reads a number from a file a "
         "human edited.\n",
         "## Environment the reported numbers were produced in\n",
         "| Component | Version |", "|---|---|"]
    for k, v in env.items():
        L.append(f"| {k} | {v if not isinstance(v, list) else v[0] if v else '—'} |")
    L.append("\n## Seeds\n")
    L.append("| Where | Seed |")
    L.append("|---|---|")
    for k, v in SEEDS.items():
        L.append(f"| {k} | `{v}` |")
    L.append("\n## Run order\n")
    L.append("| WBS | What it produces | Command | Time | Hardware |")
    L.append("|---|---|---|---|---|")
    for t, what, cmd, dur, hw in RUN_ORDER:
        L.append(f"| {t} | {what} | `{cmd}` | {dur} | {hw} |")
    missing = [e["path"] for e in inputs + code if e["missing"]]
    L.append(f"\n## Checksums\n\n`MANIFEST.json` carries sha256 for "
             f"{len([e for e in inputs if not e['missing']])} input artifacts, "
             f"{len([e for e in code if not e['missing']])} source files and "
             f"{len(delivs)} deliverables.")
    if missing:
        L.append("\nMissing at packaging time (listed rather than silently dropped): " +
                 ", ".join(f"`{m}`" for m in missing) + ".")
    L.append("\n## What is not in this package\n")
    L.append("- **LoRA adapters.** They are large binary artifacts on shared storage "
             "(`/ceph_data/tahirahmad/ontology_exp024/`). The variant-to-adapter "
             "mapping is in `wbs/task4_evaluation/run_variants.py`; the training code "
             "is in `experiments/exp024_rag_vs_ontology/`.\n"
             "- **The dense indices.** `train_bge.faiss` and the ontology card index "
             "are rebuilt by `RagContextBuilder.build()` / "
             "`OntoV3ContextBuilder.build()` from the checksummed splits and ontology, "
             "so they are derived rather than primary.\n"
             "- **Base model weights.** `meta-llama/Llama-3.1-8B-Instruct` and the "
             "other five models are pulled from the Hugging Face hub; the exact ids "
             "are in `experiments/exp024_rag_vs_ontology/config.py`.")
    (OUT / "RUN_ORDER.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    req = [f"{k}=={v}" for k, v in env.items()
           if k not in ("python", "platform", "gpu", "git_commit", "git_dirty")
           and v not in ("not installed", "unknown")]
    (OUT / "requirements.txt").write_text("\n".join(sorted(req)) + "\n",
                                          encoding="utf-8")
    print(f"wrote {OUT}/MANIFEST.json, RUN_ORDER.md, requirements.txt")
    print(f"  {len(inputs)} inputs, {len(code)} source files, {len(delivs)} deliverables")
    if missing:
        print(f"  MISSING: {', '.join(missing)}")


if __name__ == "__main__":
    main()
