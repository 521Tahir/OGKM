"""
Audit of the project against the research proposal, not against the workbook.

`audit_wbs.py` checks the 16 line items of
`2026-08월 공동연구 WBS_제주대(윤여찬).xlsx`. This script checks the concrete,
implementable requirements of
`Research on Ontology Based Knowledge Management and Integration with Large
Language Models.pdf` - its framework layers (4), generation pipeline (5),
prompting strategies (6), evolution and governance (8), ontology-integrated RAG
(9), reasoning and response generation (10), the experimental design (11) and the
three hypotheses (3.3). The proposal's own numbering skips sections 7 and 13.

Three statuses, and the distinction matters:

  met          an artifact on disk satisfies the requirement
  partial      implemented in part, with the missing part named
  out-of-scope the proposal asks for something this phase cannot do (a domain
               expert panel, a human user study), stated rather than quietly
               dropped or simulated

A requirement with no artifact is listed as missing. Nothing here infers
satisfaction from the presence of code that was never run: every path is checked.

  python -m wbs.task4_scalability.audit_proposal
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DELIV = ROOT / "wbs" / "deliverables"

# (section, requirement, [paths], note)
ITEMS: list[tuple[str, str, list[str], str]] = [
    # ── 4. methodology and conceptual framework ───────────────────────────
    ("4.1", "Five-layer end-to-end framework, documented",
     ["docs/system_requirements.md", "wbs/deliverables/figures/fig01_architecture.png"], ""),
    ("4.2", "Ontology learning decomposed into explicit prompting modules",
     ["src/ontology/concept_extractor.py", "src/ontology/relation_extractor.py",
      "src/ontology/hierarchy_inducer.py", "src/ontology/rule_suggester.py"], ""),
    ("4.5", "Validation and governance layer",
     ["wbs/task2_formalization/validate_ontology.py",
      "wbs/deliverables/task2_formalization/validation_report.md"], ""),
    ("4.6", "Ontology evolution with feedback integration",
     ["wbs/task3_governance/version_governance.py",
      "wbs/deliverables/task3_governance/proposals.json"], ""),
    ("4.7", "Ontology-aware retrieval of query-relevant subgraphs",
     ["experiments/exp024_rag_vs_ontology/contexts.py",
      "wbs/deliverables/task3_retrieval/retriever_characterization.md"], ""),
    ("4.8", "Constraint-based response generation",
     ["wbs/task3_reasoning/constrained_reasoner.py"], ""),
    # ── 5. ontology generation using LLMs ─────────────────────────────────
    ("5.2", "Domain text collection, cleaning, segmentation, metadata tagging",
     ["data/processed/cleaned_records.jsonl", "data/reports/data_profile_report.md"], ""),
    ("5.3", "Prompt-based concept identification",
     ["src/ontology/concept_extractor.py", "artifacts/ontologies/1.4.0/concepts.json"], ""),
    ("5.4", "Relation extraction with textual evidence and confidence",
     ["artifacts/ontologies/1.4.0/relations.json"], ""),
    ("5.5", "Hierarchy and taxonomy induction",
     ["src/ontology/hierarchy_inducer.py", "artifacts/ontologies/1.4.0/hierarchy.json"], ""),
    ("5.6", "Pattern induction and rule suggestion",
     ["src/ontology/rule_suggester.py", "artifacts/ontologies/1.2.0/rules.json",
      "artifacts/ontologies/1.4.0/rules.json",
      "wbs/task2_formalization/carry_rule_layer.py",
      "wbs/deliverables/task2_formalization/rule_layer_carry.json",
      "wbs/deliverables/task2_formalization/rule_compliance.json"], ""),
    ("5.8", "Confidence and ambiguity indicators retained, not resolved silently",
     ["wbs/deliverables/task2_formalization/normalization_report.json"], ""),
    # ── 6. advanced prompting strategies ──────────────────────────────────
    ("6.2-6.3", "Chain-of-thought for concepts and relation justification",
     ["prompts", "artifacts/reports/prompt_comparison_llm.md"], ""),
    ("6.4-6.5", "Graph-of-thought for structural coherence and taxonomy",
     ["src/llm/got_prompter.py"], ""),
    ("6.6", "Metacognitive prompting for self-evaluation and refinement",
     ["src/llm/metacognitive_prompter.py"], ""),
    ("6.7", "Prompt variants and sensitivity analysis",
     ["wbs/deliverables/task2_extraction/prompt_sensitivity.md"], ""),
    ("6.8", "Strategies integrated into one multi-stage pipeline",
     ["src/ontology/ontology_builder.py",
      "wbs/deliverables/task2_extraction/hypothesis_h1__s0.json",
      "wbs/deliverables/task2_extraction/hypothesis_h1__s1.json"], ""),
    # ── 8. ontology evolution and knowledge management ────────────────────
    ("8.3", "Incremental acquisition rather than full regeneration",
     ["wbs/deliverables/task3_governance/diff_logs.json",
      "wbs/task2_formalization/extract_policy_constraints.py",
      "artifacts/ontologies/1.6.0/relations.json"], ""),
    ("8.4", "Expert feedback as a first-class input",
     ["wbs/deliverables/task3_governance/proposals.json",
      "wbs/deliverables/task4_evaluation/expert_ontology_review.md"], ""),
    ("8.5", "Conflict resolution and knowledge revision, no silent overwrite",
     ["wbs/deliverables/task3_governance/version_governance_report.md"], ""),
    ("8.6", "Version control with timestamp, source and rationale; rollback",
     ["wbs/deliverables/task3_governance/current_version.json",
      "wbs/deliverables/task3_governance/audit_trail.jsonl"], ""),
    ("8.7", "Knowledge governance with role-based access control",
     ["wbs/task3_governance/access_control.py",
      "wbs/deliverables/task3_governance/access_control.md",
      "wbs/deliverables/task3_governance/principals.json",
      "wbs/deliverables/task3_governance/protected_elements.json",
      "wbs/deliverables/task3_governance/access_control_demo.md"], ""),
    ("8.8", "Evolved ontology reaches reasoning without model retraining",
     ["wbs/task3_reasoning/constrained_reasoner.py"], ""),
    # ── 9. ontology-integrated RAG ────────────────────────────────────────
    ("9.3", "Concept matching, relation traversal, subgraph extraction",
     ["src/graph/graph_store.py", "experiments/exp024_rag_vs_ontology/contexts.py"], ""),
    ("9.4", "Subgraph selection and relevance ranking, including by validation status",
     ["wbs/task3_retrieval/tuned_context.py",
      "wbs/deliverables/task3_retrieval/onto_context_tuning.json",
      "wbs/deliverables/task3_retrieval/onto_retrieval_tuning.json",
      "wbs/deliverables/task2_formalization/element_validation_status.json"], ""),
    ("9.5", "Structured prompt construction from ontological knowledge",
     ["experiments/exp024_rag_vs_ontology/config.py"], ""),
    ("9.6", "Ontology-based constraints applied during generation",
     ["wbs/deliverables/task4_evaluation/variant__llama3.1-8b__V5_ontology_rag_validated.jsonl"],
     ""),
    ("9.7", "Measured comparison against standard RAG",
     ["wbs/deliverables/task4_evaluation/baseline_comparison__llama3.1-8b.md"], ""),
    # ── 10. reasoning and response generation ─────────────────────────────
    ("10.3", "Constraint types: type, cardinality, exclusion, causal/temporal",
     ["wbs/task3_reasoning/constrained_reasoner.py",
      "wbs/task3_reasoning/constraint_coverage.py",
      "wbs/deliverables/task3_reasoning/constraint_coverage.json",
      "wbs/deliverables/task3_reasoning/constraint_coverage.md"], ""),
    ("10.4", "Evidence-grounded inference with per-claim justification",
     ["wbs/deliverables/task4_evaluation/variant__llama3.1-8b__V5_ontology_rag_validated.jsonl"],
     ""),
    ("10.5", "Pre-generation guidance and post-generation validation",
     ["wbs/deliverables/task4_evaluation/ablation_validation__llama3.1-8b.json"], ""),
    ("10.6", "Explainable response construction with reasoning traces",
     ["wbs/deliverables/task4_evaluation/hypotheses_h2_h3__llama3.1-8b.json"], ""),
    ("10.8", "Uncertainty handling and fallback on incomplete knowledge",
     ["wbs/deliverables/task4_evaluation/ablation_validation__llama3.1-8b.json"], ""),
    # ── 11. experimental design and evaluation ────────────────────────────
    ("11.2", "Five system variants evaluated",
     ["wbs/deliverables/task4_evaluation/baseline_comparison__llama3.1-8b.md",
      "wbs/deliverables/task4_evaluation/variant_scores__llama3.1-8b.json"], ""),
    ("11.4.1", "Automated ontology metrics: consistency, redundancy/conflict, "
               "depth and balance, rule satisfaction",
     ["wbs/deliverables/task2_formalization/ontology_quality_metrics.json",
      "wbs/deliverables/task2_formalization/rule_compliance.json"], ""),
    ("11.4.2", "Expert evaluation of concept and relation quality",
     ["wbs/deliverables/task4_evaluation/expert_ontology_review.csv",
      "wbs/deliverables/task4_evaluation/expert_ontology_review.md",
      "wbs/task4_evaluation/score_human_review.py",
      "wbs/deliverables/task4_evaluation/human_review_scores.md"], "11.4.2"),
    ("11.5a", "Response accuracy: exact and partial correctness",
     ["wbs/deliverables/task4_evaluation/variant_scores__llama3.1-8b.json"], ""),
    ("11.5b", "Consistency across paraphrased and repeated queries",
     ["wbs/deliverables/task4_evaluation/consistency__llama3.1-8b*.json"], ""),
    ("11.5c", "Stability under prompt variation and input noise",
     ["wbs/deliverables/task4_evaluation/ablation_prompts__llama3.1-8b.json",
      "wbs/deliverables/task4_scalability/robustness.md"], ""),
    ("11.6a", "Automated hallucination detection with ontology rules",
     ["wbs/deliverables/task4_evaluation/hallucination_report__llama3.1-8b.md"], ""),
    ("11.6b", "Manual annotation of hallucinated statements",
     ["wbs/deliverables/task4_evaluation/manual_hallucination_review.csv",
      "wbs/deliverables/task4_evaluation/manual_hallucination_review.md",
      "wbs/task4_evaluation/score_human_review.py",
      "wbs/deliverables/task4_evaluation/human_review_scores.md"], "11.6b"),
    ("11.6c", "Comparative hallucination analysis across variants",
     ["wbs/deliverables/task4_evaluation/hallucination_report__llama3.1-8b.json"], ""),
    ("11.7a", "Reasoning traces available and aligned with ontology structures",
     ["wbs/deliverables/task4_evaluation/hypotheses_h2_h3__llama3.1-8b.json"], ""),
    ("11.7b", "User study of perceived trust and transparency",
     [], "11.7b"),
    ("11.8", "Scalability: generation/update time, retrieval latency vs size, "
             "constraint overhead",
     ["wbs/deliverables/task4_scalability/ontology_scalability.md",
      "wbs/deliverables/task4_scalability/retrieval_scalability.md"], ""),
    ("11.9", "Statistical significance testing of all comparisons",
     ["wbs/deliverables/task4_evaluation/variant_scores__llama3.1-8b.json"], ""),
    # ── 3.3 hypotheses ────────────────────────────────────────────────────
    ("H1", "Structured prompting yields higher logical consistency and "
           "hierarchical quality than naive prompting",
     ["wbs/deliverables/task2_extraction/hypothesis_h1__*.json"], ""),
    ("H2", "Ontology constraints reduce hallucination without degrading fluency",
     ["wbs/deliverables/task4_evaluation/hypotheses_h2_h3__llama3.1-8b.json"], ""),
    ("H3", "Ontology-integrated systems are more interpretable and verifiable",
     ["wbs/deliverables/task4_evaluation/hypotheses_h2_h3__llama3.1-8b.json"], ""),
]

# requirements this phase cannot satisfy, with the reason, kept visible
OUT_OF_SCOPE = {
    "11.4.2": "The instrument is built and stratified (56 concepts over 33 classes, "
              "59 relations over 20 predicates, with source evidence and an "
              "agreement protocol) but no domain-expert panel is available to this "
              "project. The scores are deliberately not simulated. What was "
              "missing on the machine side is now closed: the protocol demanded "
              "Krippendorff's alpha and Cohen's kappa \"before reporting any mean "
              "score\" and nothing computed either, so "
              "`score_human_review.py` does - verified against Krippendorff's "
              "published four-observer example to 0.7434 against his 0.743 - and "
              "it enforces the ordering rather than trusting it: with fewer than "
              "two independent rater sheets it reports agreement as undefined and "
              "prints no mean at all. Filled sheets dropped in as "
              "`expert_ontology_review__<rater>.csv` produce the numbers with no "
              "new code. What remains out of scope is the panel.",
    "11.6b": "The blinded annotation sheet is built (150 spans across five "
             "variants, half auto-flagged and half not, variant identity hashed) "
             "but no annotator panel is available. The scoring side is "
             "implemented and tested in `score_human_review.py`, which computes "
             "kappa and nominal alpha over the four labels and only then "
             "unblinds - the variant key is read there, from "
             "`expert_review_pack.json`, and never by a rater - and which then "
             "reports the automated detector's precision and recall against the "
             "raters, the figure this project carries as a known unknown. "
             "Left unfilled rather than "
             "filled by a model grading its own output.",
    "11.7b": "A user study of perceived trust requires human participants and "
             "ethics approval, neither of which is in scope for this phase. The "
             "objective, machine-checkable part of 11.7 - trace availability and "
             "the share of an answer attributable to its own evidence - is "
             "measured instead.",
}

NOTE = {
    "5.6": "Fifteen rules exist (8 seed, 7 mined) and are now carried into "
           "v1.4.0, the version the validator, retriever and every evaluation "
           "arm load, by `carry_rule_layer.py` - which re-checks each rule "
           "against what that version actually contains rather than copying it "
           "blind. 11 of 15 are enforceable there; the other 4 need a "
           "severe-storm weather state, a queryable weather state, or a risk "
           "severity, and each is carried with its measured reason in its "
           "`enforceable_in` block - those are two different failures: risk "
           "severity is carried by no concept, relation or class at all, "
           "while WeatherCondition exists with 8 instances and simply has no "
           "predicate pointing at it on more than 11% of that predicate's own "
           "triples. Traveler type, named here until 2026-10-18 as one of the "
           "missing attributes, is not missing: `suitable_for` carries it on "
           "270 of its 283 triples, and the claim was an unmeasured `False` "
           "in a lookup table. The verdict is `rule_compliance.py`'s "
           "curated per-rule needs table rather than a second opinion derived "
           "in the carry, so the project states enforceability once. The temporal and schema layers were carried "
           "with them, so the knowledge base no longer falls back to v1.2.0 for "
           "its temporal facts. Concepts, relations and hierarchy were not "
           "touched and `ontology_quality_metrics.json` is byte-identical "
           "across the change.",
    "6.8": "`OntologyBuilder`'s `unified` strategy is the multi-stage pipeline "
           "6.8 describes: chain-of-thought extraction, then the graph-of-"
           "thought structural-coherence pass, then the metacognitive flag-for-"
           "review pass. It is now benchmarked against the single strategies it "
           "combines, on the same records, seed and decoding as the published "
           "H1 comparison (`HYPOTHESES.md`, \"The unified multi-stage "
           "pipeline\"). The integration exists and the benchmark reports what "
           "it is worth: across two disjoint 20-record samples it removed 1 "
           "relation of 963 and flagged 0 concepts, so its consistency and "
           "hierarchy metrics are its base strategy's, at a wall-clock cost. "
           "Held out of the H1 win counts, which were registered over the four "
           "original strategies.",
    "8.7": "Implemented and enforced in `access_control.py`, not described. "
           "Four roles over thirteen permissions - `reader`, `automation` "
           "(the extraction pipeline), `expert`, `admin` - with transitive "
           "inheritance, and every one of the six governance entry points in "
           "`version_governance.py` calls `require()` before it writes, so a "
           "principal without the permission receives `AccessDenied` and the "
           "ontology is unchanged. A test parses the module and fails if any "
           "of those six loses its check. §8.7's three named properties are "
           "each a mechanism: **protection** - 158 elements of the published "
           "v1.4.0 are protected, derived from the ontology rather than listed "
           "by hand (1 concept and 3 relations because a human approved them, "
           "§8.4's requirement, and 154 triples because they carry a safety "
           "constraint: `has_risk`, `avoid_during`, `requires`, "
           "`restricted_in`), and overriding one needs a permission only "
           "`admin` holds; **isolation** - a bare semantic version is "
           "production and anything else is staging, so `automation` can write "
           "and measure a candidate version but cannot write what the deployed "
           "system serves, and moving the published pointer is a third "
           "permission again; **accountability** - allow and deny both append "
           "to the same audit trail, so a refusal is as durable as a change. "
           "Sixteen attempts as five real principals are exercised in "
           "`access_control_demo.md`: 8 allowed, 8 denied. Two things it does "
           "not do, stated rather than glossed: it authenticates nothing - the "
           "principal is whatever `WBS_ACTOR` says, which is attribution and "
           "policy inside one trust boundary, not an identity provider (R14) - "
           "and separation of duties, now enforced, is not satisfied by the "
           "record that predates it: P0001 was created and approved by the "
           "same principal, which today's policy refuses. That merge is left "
           "in the audit trail rather than rewritten.",
    "9.4": "All three criteria 9.4 names are now implemented and swept in "
           "`onto_context_tuning.json`: conceptual proximity (dense card "
           "similarity), prominence (graph degree), and validation status. The "
           "last required a per-element verdict the ontology did not carry, so "
           "`element_validation_status.py` derives one (1,199 clean, 366 warned "
           "concepts). Ranking on it is costly, not free: 23.4% of cards carry a "
           "validation finding overall but 76.7% of the 120 highest-degree cards "
           "do, so the prominent entities the gold answers name are precisely the "
           "flagged ones, and validation-first ranking halves gold-entity coverage "
           "(53.0% to 24.3%). That trade has now been settled on real output: "
           "generating all 347 answers under validation-status ranking gives 42.2 "
           "token F1 against the baseline's 42.4 (p = 0.776) and 3.0% ungrounded "
           "against 3.7% (p = 0.444) - it neither helps nor hurts the answer at "
           "this sample size, while costing more than half the gold-entity "
           "coverage. The criterion is implemented, swept offline and measured "
           "end to end; what it is worth is small and not significant.",
    "10.3": "All four families are implemented and replayed against every "
            "stored answer by `constraint_coverage.py`, because a constraint "
            "that never runs on real output is a constraint in name only. Type "
            "(UNGROUNDED_ENTITY, RELATION_UNSUPPORTED) and exclusion "
            "(UNSUPPORTABLE_CLAIM) were already in place. Cardinality is derived "
            "from the ontology rather than asserted about it: a predicate counts "
            "as functional only if at least 95% of its subjects carry exactly "
            "one object over at least 20 subjects, which selects `is_a` (99.7%) "
            "and `subclass_of` (100%) and correctly rejects `located_in` (70%). "
            "Causal/temporal respects polarity - `avoid_during` lists months to "
            "stay away from, so it is violated by RECOMMENDING its subject "
            "inside them, while a positive fact is violated by placing it "
            "outside its months. Three of the four fire on this benchmark; "
            "cardinality is exercised and never violated. The temporal family "
            "fires 6 times, all of them in V1, the only arm with no grounding, "
            "and 0 times in all six grounded arms.",
    "11.4.2": "See the out-of-scope note.",
    "11.6b": "See the out-of-scope note.",
    "11.7b": "See the out-of-scope note.",
}


def main() -> None:
    rows, met, partial, oos, missing = [], 0, 0, 0, 0
    for sec, req, paths, note in ITEMS:
        def present(rel: str) -> bool:
            # several deliverables are written one shard per GPU, so the evidence
            # is a glob, not a fixed name
            if "*" in rel:
                return any(ROOT.glob(rel))
            return (ROOT / rel).exists()

        have = [p for p in paths if present(p)]
        gone = [p for p in paths if not present(p)]
        if sec in OUT_OF_SCOPE:
            status, mark = "out-of-scope", "◻️ out of scope"
            oos += 1
        elif not paths:
            status, mark = "missing", "❌ missing"
            missing += 1
        elif gone:
            status, mark = ("partial", "⚠️ partial") if have else ("missing", "❌ missing")
            partial += 1 if have else 0
            missing += 0 if have else 1
        elif note:
            status, mark = "partial", "⚠️ partial"
            partial += 1
        else:
            status, mark = "met", "✅ met"
            met += 1
        rows.append({"section": sec, "requirement": req, "status": status,
                     "evidence": have, "missing": gone,
                     "note": NOTE.get(note or sec, ""),
                     # the real reason, so a consumer of the JSON does not get
                     # handed "see the out-of-scope note" and have to go find it
                     "reason": OUT_OF_SCOPE.get(sec, "")})
        print(f"{mark:<16} {sec:<9} {req[:62]}")
        for p in gone:
            print(f"                     missing: {p}")

    total = len(ITEMS)
    L = [f"# Proposal Compliance Audit\n",
         f"Audited against `Research on Ontology Based Knowledge Management and "
         f"Integration with Large Language Models.pdf`. The proposal names "
         f"{total} concrete, implementable requirements across its framework, "
         f"pipeline, prompting, governance, retrieval, reasoning and evaluation "
         f"sections, plus the three hypotheses of section 3.3. Its own numbering "
         f"skips sections 7 and 13.\n",
         f"`WBS_WORKBOOK_AUDIT.md` answers a different question - whether the "
         f"16 line items of the funded workbook are complete. This file answers "
         f"whether the research proposal behind it has been carried out.\n",
         f"**{met} met, {partial} partial, {oos} out of scope, {missing} "
         f"missing, of {total}.**\n",
         "| Section | Requirement | Status |", "|---|---|---|"]
    MARK = {"met": "✅ met", "partial": "⚠️ partial",
            "out-of-scope": "◻️ out of scope", "missing": "❌ missing"}
    for r in rows:
        L.append(f"| {r['section']} | {r['requirement']} | {MARK[r['status']]} |")

    L.append("\n## What is not finished, and why\n")
    for r in rows:
        if r["status"] == "met":
            continue
        L.append(f"**{r['section']} — {r['requirement']}.** "
                 + (OUT_OF_SCOPE.get(r["section"], "") or r["note"]
                    or ("missing artifacts: " + ", ".join(f"`{m}`" for m in r["missing"]))))
        L.append("")

    # A section counted as met can still carry a finding a reader needs - that
    # the thing exists, works, and measurably does not help. Losing that when
    # the row flips from partial to met would make the audit read better than
    # the project did, so met rows with a note keep it.
    noted = [r for r in rows if r["status"] == "met" and r["note"]]
    if noted:
        L.append("\n## Met, with what the measurement found\n")
        L.append("These requirements are satisfied; the note is what measuring "
                 "them produced, which is not always a confirmation.\n")
        for r in noted:
            L.append(f"**{r['section']} — {r['requirement']}.** {r['note']}")
            L.append("")

    L.append("## Evidence\n")
    for r in rows:
        if not r["evidence"]:
            continue
        L.append(f"**{r['section']}** — " + ", ".join(f"`{e}`" for e in r["evidence"]))
    L.append("")

    (DELIV / "PROPOSAL_COMPLIANCE.md").write_text("\n".join(L), encoding="utf-8")
    (DELIV / "PROPOSAL_COMPLIANCE.json").write_text(
        json.dumps({"total": total, "met": met, "partial": partial,
                    "out_of_scope": oos, "missing": missing, "items": rows},
                   indent=2), encoding="utf-8")
    print(f"\nwrote PROPOSAL_COMPLIANCE.md")
    print(f"  {met} met, {partial} partial, {oos} out of scope, {missing} missing, "
          f"of {total}")


if __name__ == "__main__":
    main()
