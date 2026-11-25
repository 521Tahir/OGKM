"""Build the 2026 annual report (ETRI joint/commissioned institution form) in English.

Usage:  python build_annual_report.py [--pages pages.json]
The optional page map fills the cached table of contents; Word recomputes it on open.
"""
import argparse
import json
from pathlib import Path

from docx.shared import Cm

import docx_helpers as H
from docx_helpers import (box, bullets, caption_table, code_block, data_table, figure, form_table,
                          item, note, para, source, sub)

ROOT = Path("/home/tahirahmad/Ontology_Project_JejuFull")
D = ROOT / "wbs/deliverables"
OUT_DIR = D / "annual_report_2026"
FIG = OUT_DIR / "figures"
OUT = OUT_DIR / "Annual_Report_2026_EN_Neo4j.docx"

TBD = "[to be completed]"
INST = "Jeju National University (Dept. of Artificial Intelligence, NLP Lab)"
PI = "Prof. Yoon Yeo-Chan"
PROJECT = ("Research on Ontology-Based Knowledge Management and Integration with Large Language "
           "Models: a Neo4j-based Jeju Tourism Ontology and Ontology-Grounded LLM Services")
PERIOD_ALL = "2025. 4. 1. ~ 2027. 12. 31. (2 years 9 months)"
PERIOD_YEAR = "2026. 4. 1. ~ 2026. 12. 31. (9 months)"

NEO = json.load(open(FIG / "neo4j_live_snapshot.json"))
NEO_Q = json.load(open(D / "task3_storage/graph_db_neo4j.json"))

TOC_ENTRIES = [
    (1, "1. Overview of the R&D Project"),
    (1, "2. R&D Process and Content"),
    (1, "3. R&D Results and Degree of Goal Achievement"),
    (2, "3-1. Achievement Level of This Year's Objectives"),
    (2, "3-2. Qualitative R&D Outcomes"),
    (2, "3-3. Quantitative R&D Outcomes"),
    (2, "3-4. Detailed Quantitative R&D Outcomes"),
    (2, "3-5. Unplanned Outcomes and Contributions to Related Fields"),
    (1, "4. R&D Expenditure"),
    (2, "4-1. R&D Expenditure Statement of the R&D Institution"),
    (2, "4-2. R&D Expenditure Statement of the R&D Project"),
    (1, "5. R&D Plan for the Next Year"),
    (2, "5-1. R&D Objectives and Content"),
    (2, "5-2. Changes in the Domestic and International Environment"),
    (2, "5-3. R&D Strategy"),
    (2, "5-4. R&D Schedule and Expected Outcomes"),
    (2, "5-5. R&D Budget Plan"),
    (2, "5-6. Commercialisation Plan"),
    (2, "5-7. Utilisation of R&D Outcomes and Expected Effects"),
    (1, "[Attachments] Evidence and Supplementary Material for Detailed Quantitative Outcomes"),
]

CW = [2.6, 14.4]  # label / content widths of the form's boxed tables
IN_CELL_W = 13.6  # usable width inside the content column


# ====================================================================== cover
def cover(doc):
    t = doc.add_table(rows=1, cols=1)
    H.no_borders(t)
    H.set_widths(t, [H.TEXT_W])
    c = t.cell(0, 0)
    para(c, "Annual Report", 26, bold=True, align="c", before=6, after=2)
    para(c, "(Joint / Commissioned R&D Institution)", 12, align="c", after=8)
    para(c, "Security grade:  General [ ● ]   Secure [   ]", 11, align="r", after=6)

    rows = [
        ["R&D project title", "ETRI Jeju Regional Research Center Pilot Programme", None, None, None, None],
        ["Joint R&D project title", PROJECT, None, None, None, None],
        ["Joint R&D institution", "Institution name", "Jeju National University", "Business registration no.", TBD, None],
        [None, "Address", "102 Jejudaehak-ro, Jeju-si, Jeju Special Self-Governing Province, Republic of Korea",
         "Corporate registration no.", TBD, None],
        ["Principal investigator", "Name", "Yoon Yeo-Chan", "Position", "Assistant Professor, Dept. of Artificial Intelligence", None],
        [None, "Contact", "Office: " + TBD + "   Mobile: " + TBD, "E-mail", "ycyoon@jejunu.ac.kr", None],
        [None, "National researcher no.", TBD, None, None, None],
        ["R&D period", "Overall", PERIOD_ALL, None, None, None],
        [None, "This year", PERIOD_YEAR, None, None, None],
    ]
    w = [3.0, 2.6, 5.2, 2.6, 3.6]
    tb = doc.add_table(rows=len(rows), cols=5)
    H.borders(tb)
    H.cell_margins(tb)
    H.set_widths(tb, w)
    for r, row in enumerate(rows):
        for ci, txt in enumerate(row[:5]):
            if txt is None:
                continue
            lab = ci == 0 or (ci in (1, 3) and r >= 2)
            H.write_cell(tb.cell(r, ci), txt, 9, bold=lab, align="c" if lab else "l",
                         fill=H.LABEL_FILL if lab else None)
    for (a, b) in [((0, 1), (0, 4)), ((1, 1), (1, 4)), ((2, 0), (3, 0)), ((4, 0), (6, 0)),
                   ((6, 2), (6, 4)), ((7, 0), (8, 0)), ((7, 2), (7, 4)), ((8, 2), (8, 4))]:
        tb.cell(*a).merge(tb.cell(*b))
    H.spacer(doc, 4)

    para(doc, "R&D budget (unit: KRW thousand)", 9, bold=True, align="l", after=2)
    hdr = [["Stage", "Year", "ETRI R&D funds", "Institution contribution", None, "Other institutional support",
            None, "Local government", None, "Total R&D budget", None, None],
           [None, None, "Cash", "Cash", "In-kind", "Cash", "In-kind", "Cash", "In-kind", "Cash", "In-kind", "Sum"]]
    body = [["Stage 1", "Year 1", "", "", "", "", "", "", "", "", "", ""],
            [None, "Year 2", "", "", "", "", "", "", "", "", "", ""],
            [None, "Year 3", "", "", "", "", "", "", "", "", "", ""],
            ["Total", None, "", "", "", "", "", "", "", "", "", ""]]
    data_table(doc, hdr, body, [1.3, 1.3] + [1.44] * 10, size=7.5,
               merges=[((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 3), (0, 4)), ((0, 5), (0, 6)),
                       ((0, 7), (0, 8)), ((0, 9), (0, 11)), ((2, 0), (4, 0)), ((5, 0), (5, 1))])
    note(doc, "Funding amounts are entered by the Industry-Academic Cooperation Foundation from the agreement "
              "and settled accounts; they are intentionally left blank in this draft.")
    H.spacer(doc, 2)
    data_table(doc, None, [["R&D project practitioner", "Name", TBD, "Position", TBD],
                           [None, "Contact", "Office: " + TBD + "   Mobile: " + TBD, "E-mail", TBD],
                           [None, "National researcher no.", TBD, None, None]],
               [3.0, 2.6, 5.2, 2.6, 3.6], size=9, align=["c", "c", "l", "c", "l"],
               first_col_fill=H.LABEL_FILL, merges=[((0, 0), (2, 0)), ((2, 2), (2, 4))])
    H.spacer(doc, 8)
    para(doc, "I confirm that the contents of this annual report are true. If they are not, I will accept "
              "the consequences under the relevant laws and regulations, including suspension of the R&D "
              "project, termination of the agreement and sanctions.", 11, align="j", after=10)
    para(doc, "2026.      .      .", 12, align="c", after=10)
    para(doc, "Principal investigator:  Yoon Yeo-Chan            (seal)", 12, align="r", after=4)
    para(doc, "Head of the joint R&D institution:                         (official seal)", 12, align="r", after=14)
    para(doc, "To the President of the Electronics and Telecommunications Research Institute (ETRI)", 13,
         bold=True, align="c")
    H.page_break(doc)


# ====================================================================== summary
def summary(doc):
    para(doc, "< Summary >", 16, bold=True, align="c", after=2)
    note(doc, "The summary is limited to five pages.", 8)
    S = 9

    def lst(items):
        def f(cell):
            for it in items:
                if isinstance(it, tuple):
                    para(cell, "○ " + it[0], S, align="l", hanging=0.35, after=1)
                    for s in it[1]:
                        para(cell, "- " + s, S, align="l", left=0.45, hanging=0.3, after=1)
                else:
                    para(cell, "○ " + it, S, align="l", hanging=0.35, after=1)
        return f

    final_goal = ("Develop a governed, ontology-centred knowledge-management framework for large language models "
                  "(LLMs): an LLM-built and validated Jeju tourism ontology, stored, versioned and served from a "
                  "**Neo4j graph database**, integrated with retrieval-augmented generation (RAG) and a "
                  "post-generation ontology verifier. The system must reduce unsupported (hallucinated) content, "
                  "make every answer traceable to explicit knowledge, and allow domain knowledge to be corrected "
                  "without retraining the model.")
    overall = [
        "LLM-based ontology learning: concept, relation, hierarchy and rule extraction with structured prompting",
        "Ontology normalization, RDF/OWL formalization and multi-layer validation (structure, typing, SHACL, OWL-RL)",
        "Neo4j ontology graph database with versioning, diff, rollback, governance workflow and role-based access control",
        "Ontology-aware retrieval (intent-to-class routing, subgraph extraction, hybrid retrieval) and constrained generation "
        "(validation, surgical repair, evidence trace, abstention)",
        "Benchmark-based evaluation of five system variants, ablations, multi-model and scalability/robustness studies",
        "Year 2: in-database validation, Cypher-native GraphRAG retrieval, human expert evaluation, Korean-language support "
        "and a pilot service with tourism stakeholders",
    ]
    year_goal = ("Build and validate the Year-1 prototype end to end on the Jeju tourism domain: data, LLM ontology "
                 "extraction, normalization and validation, Neo4j storage with governance, ontology-aware retrieval "
                 "and constrained generation, and a benchmark evaluation of five system variants with scalability "
                 "and robustness analysis (all 16 WBS line items).")
    year_content = [
        "Task 1: requirements, architecture, reproducibility and risk plans; cleaning and profiling of 46,059 Jeju "
        "tourism records; ontology schema v0",
        "Task 2: LLM extraction pipeline with five prompting strategies and a staged pipeline; normalization, RDF/OWL "
        "serialization and four-layer validation",
        "Task 3: Neo4j 5.26 ontology database (typed property graph), content-hashed versioning, governance with "
        "RBAC and audit trail; ontology-aware retriever and constrained reasoner",
        "Task 4: 347-item benchmark over six task types; five variants at three model sizes; 4,607-question "
        "cross-model study on eight models; scalability to 100K triples; robustness under seven input conditions",
    ]
    outcomes = [
        ("Ontology v1.4.0 (evaluated release): 1,565 concepts and 7,909 relations over 20 relation types; 100% "
         "provenance coverage, 0 isolated concepts; SHACL conforms, OWL-RL consistent", []),
        ("Neo4j 5.26.0 holds the ontology as a typed property graph: 1,565 nodes, 7,909 relationships, 34 labels, "
         "20 relationship types, loaded in 0.51 s; nine competency questions answered in Cypher at a median of "
         "0.83-1.48 ms", []),
        ("Governance: 9 version transitions with content-hash diffs; 4 roles x 13 permissions; 158 protected "
         "elements; 78 audited decisions (36 allowed, 42 denied)", []),
        ("Groundedness: with gated surgical repair the strict ungrounded-entity rate is 1.58 / 1.82 / 2.15% at "
         "1B / 3B / 8B versus 4.50 / 4.93 / 5.66% for Simple RAG, at a token-F1 cost of 0.12-0.86 points", []),
        ("Model-agnostic verifier: applied to six models (three families) on 4,607 held-out questions it more than "
         "halves the strict ungrounded rate (about 5.1% to 2.1%) and raises fully grounded answers from about 70% "
         "to 83%, with no retraining and no GPU", []),
        ("Knowledge update without retraining: a 22-fact policy edit cost 20 s of CPU and 0 GPU-hours (vs 135.1 "
         "GPU-hours for the 15 LoRA adapters trained); the rule reached the model context in 10/10 probes", []),
        ("Limits stated: Simple RAG keeps the lead on reference-overlap metrics on clean input; H1 not supported, "
         "H2 and H3 partially supported; human expert ratings await raters (instruments built)", []),
    ]
    use = [
        "Year 2: Neo4j becomes the single source of truth (in-database SHACL via neosemantics), with a Cypher-native "
        "GraphRAG retriever and the ontology verifier as a plug-in for tourism chatbots",
        "Pilot with Jeju tourism stakeholders: trustworthy, source-traceable answers on safety, policy and "
        "accessibility; knowledge corrections in minutes instead of model retraining",
        "Transferable to other high-trust domains (public administration, healthcare information) that require "
        "verifiable LLM answers, in line with the transparency goals of Korea's AI Framework Act (in force 2026)",
    ]
    rows = [
        ["R&D project title", "ETRI Jeju Regional Research Center Pilot Programme"],
        ["Joint R&D project title", PROJECT],
        ["R&D period", f"Overall: {PERIOD_ALL}\nThis year: {PERIOD_YEAR}"],
        ["R&D budget", "Overall: total ______ KRW thousand (ETRI R&D funds: ______; institution contribution: ______; "
                       "local government: ______; other support: ______)\nThis year: total ______ KRW thousand "
                       "(ETRI R&D funds: ______; institution contribution: ______; local government: ______; "
                       "other support: ______)"],
        ["R&D stage", "Basic [  ]   Applied [ ● ]   Development [  ]   Other [  ]\n"
                      "Technology readiness level (TRL): at start [ TRL 3 ]   target at end [ TRL 5 ]   "
                      "not applicable [  ]"],
        ["Final objective", final_goal],
        ["Overall content", lst(overall)],
        ["This year's objective", year_goal],
        ["This year's content", lst(year_content)],
        ["R&D outcomes", lst(outcomes)],
        ["Utilisation plan and expected effects", lst(use)],
    ]
    form_table(doc, rows, [3.2, 13.8], size=S)
    H.spacer(doc, 2)
    data_table(doc, None, [
        ["Korean keywords (up to 5)", "온톨로지", "대규모 언어모델", "지식그래프(Neo4j)", "검색증강생성", "환각 억제"],
        ["English keywords (up to 5)", "Ontology", "Large Language Model", "Knowledge Graph (Neo4j)",
         "Retrieval-Augmented Generation", "Hallucination Mitigation"]],
        [3.2, 2.76, 2.76, 2.76, 2.76, 2.76], size=S, first_col_fill=H.LABEL_FILL,
        align=["c"] * 6)
    note(doc, "TRL values are the research team's assessment (Year 1 ends with a laboratory-validated prototype, "
              "TRL 4) and should be confirmed against the agreement.", 8)
    H.page_break(doc)


def contents(doc, pages):
    para(doc, "Table of Contents", 16, bold=True, align="c", after=10)
    H.toc(doc, TOC_ENTRIES, pages)
    H.spacer(doc, 10)
    note(doc, "Body text is set in Dotum 12 pt and tables in Dotum 8 pt, as the form requires. Page numbers "
              "refresh when the document is opened in Microsoft Word (accept the prompt to update fields).", 9)
    H.page_break(doc)


# ====================================================================== chapter 1
def chapter1(doc):
    H.chapter_banner(doc, 1, "Overview of the R&D Project")
    H.section_heading(doc, 1, "Purpose of the R&D", None)
    box(doc, "Final purpose")
    bullets(doc, [
        "Large language models answer fluently but keep their knowledge implicit: an answer cannot be traced to "
        "a fact, corrected at a single point, or checked against an explicit rule. This project makes the "
        "knowledge explicit and governable by building a domain ontology with LLMs, storing it in a **Neo4j "
        "graph database**, and using it to retrieve evidence for, constrain and verify LLM answers.",
        ("The Year-1 target domain is Jeju tourism, a high-trust setting where wrong answers about safety, "
         "access, transport or local rules have real consequences for visitors.",
         ["The ontology acts as a semantic control layer between the corpus and the model: it governs what is "
          "extracted, what is retrieved, what may be asserted, and how knowledge changes over time."]),
    ])
    box(doc, "Year-1 purpose")
    bullets(doc, [
        "Deliver a working, measured prototype of the whole pipeline (data, extraction, validation, Neo4j "
        "storage and governance, retrieval, constrained generation, evaluation) and answer the proposal's three "
        "hypotheses with pre-stated verdict rules.",
        "Quantify what ontology grounding buys and what it costs against a strong Simple RAG baseline over the "
        "same corpus, without inflating results (equal training budgets reported separately, held-out tests, "
        "paired significance tests with Holm correction).",
    ])

    H.section_heading(doc, 2, "Necessity of the R&D", None)
    box(doc, "Technical necessity")
    bullets(doc, [
        ("Hallucination is an inherent limitation of LLMs, so it must be managed by external checks rather than "
         "removed by training alone [10].",
         ["Passage-based RAG improves grounding but retrieves unstructured text: it cannot enforce a rule, "
          "detect an unsupported entity, or show which fact an answer used."]),
        ("Ontologies supply what RAG lacks: typed entities, explicit relations, hierarchy and constraints that "
         "can be queried, validated and versioned [13][19].",
         ["Graph databases such as Neo4j make that structure operational: millisecond multi-hop queries, "
          "schema constraints and indexes, and a standard query language (Cypher)."]),
        "Building ontologies by hand is slow; LLM-based ontology learning is promising but needs validation, "
        "human oversight and governance before its output can be trusted [16][17].",
    ])
    box(doc, "Economic and industrial necessity")
    bullets(doc, [
        "Tourism is a core industry of Jeju. Visitor-facing AI services (chatbots, itinerary planners) need "
        "answers that are correct, current and attributable, and operators need to correct them cheaply.",
        "Correcting an LLM by fine-tuning costs GPU time for every change; correcting an ontology costs a "
        "validated edit. Year 1 measured this directly (Chapter 3): 22 facts, 20 s of CPU and 0 GPU-hours, "
        "against 135.1 GPU-hours for the 15 adapters trained in this project.",
    ])
    box(doc, "Social and policy necessity")
    bullets(doc, [
        "Safety information (typhoons, rough seas, trail closures), policy constraints (dress codes, drone "
        "restrictions, permits) and accessibility information must not be invented by a model.",
        "Korea's Framework Act on the Development of Artificial Intelligence and Establishment of Trust "
        "(in force January 2026) emphasises transparency and trustworthiness of AI services; traceable, "
        "verifiable answers support that requirement.",
    ])

    H.section_heading(doc, 3, "Scope of the R&D", None)
    caption_table(doc, "1-1", "Scope of the Year-1 joint R&D")
    data_table(doc, ["Area", "Year-1 scope", "Out of scope in Year 1"], [
        ["Domain", "Jeju tourism: recommendation, itinerary, safety, dining, accessibility, transport",
         "Other regions and domains"],
        ["Data", "46,059 cleaned English Jeju tourism records (train 36,847 / validation 4,605 / test 4,607)",
         "Live feeds (events, weather, transport timetables); Korean-language source text"],
        ["Ontology", "LLM-extracted ontology with schema, hierarchy, rules, provenance and confidence; releases "
                     "v0.1.0 to v1.6.0 (v1.4.0 evaluated)", "Manual expert curation at scale"],
        ["Storage", "Neo4j 5.26 graph database (typed property graph) + versioned file store + RDF/OWL exports "
                    "(Turtle, OWL, JSON-LD, N-Triples)", "Production cluster deployment"],
        ["LLMs", "Llama-3.2-1B/3B, Llama-3.1-8B with LoRA adapters; Qwen 2.5/3, Mistral-7B, DeepSeek-R1-8B "
                 "for cross-model checks", "Commercial API models"],
        ["Evaluation", "347-item benchmark, 4,607-question held-out test, automatic metrics, paired tests, "
                       "ablations, scalability, robustness", "Human expert ratings (instruments built; raters "
                                                             "scheduled for Year 2)"],
    ], [2.4, 9.2, 5.4], size=8)
    source(doc, "Proposal §§1-6, 11; WBS workbook '2026 실행Ver_1'; artifacts/reports/dataset_statistics.json")
    figure(doc, FIG / "fig01_architecture_neo4j.png", "1-1",
           "Year-1 system architecture with Neo4j as the ontology graph database", "Research team; "
           "wbs/deliverables/annual_report_2026/scripts/make_plots.py", 16.5)
    caption_table(doc, "1-2", "Correspondence between proposal modules and WBS tasks")
    data_table(doc, ["Proposal module", "WBS task", "Period (2026)", "Main deliverable"], [
        ["Research alignment, corpus and schema", "1.1, 1.2", "May - Jul", "Architecture, cleaned corpus, schema v0"],
        ["Ontology generation pipeline, prompting", "2.1", "Aug - Sep", "Extraction modules, prompt library"],
        ["Validation and governance", "2.2, 3.1.2", "Sep - Oct", "RDF/OWL, validation engine, version control"],
        ["Ontology storage", "3.1.1", "Aug - Sep", "Neo4j graph database, file store, RDF store"],
        ["Ontology-aware RAG and constrained reasoning", "3.2", "Oct - Nov", "Retriever, constrained reasoner"],
        ["Pilot evaluation, final integration", "4.1, 4.2", "Nov - Dec", "Benchmark, evaluation, scalability, report"],
    ], [5.0, 2.0, 2.4, 7.6], size=8)
    source(doc, "WBS workbook sheets '제안서Ver' and '2026 실행Ver_1'")
    H.page_break(doc)


# ====================================================================== chapter 2
def subtask(doc, title, period, overview, process, outcomes):
    box(doc, f"{title}  ({period})")
    form_table(doc, [
        ["Performing institution / researchers", [f"{INST}; PI: {PI}; NLP Lab research team"]],
        ["Research overview", overview],
        ["Process", process],
        ["Main outcomes", outcomes],
    ], CW, size=10)
    H.spacer(doc, 6)


def chapter2(doc):
    H.chapter_banner(doc, 2, "R&D Process and Content")
    H.section_heading(doc, 1, "Objectives", None)
    form_table(doc, [
        ["Overall", "Objective", "**A governed ontology knowledge-management framework for LLMs, with the "
                                 "ontology stored and served from Neo4j, integrated with RAG and a verifier "
                                 "that suppresses unsupported content and keeps answers traceable.**"],
        [None, "Main content", [
            ("LLM ontology learning with structured prompting and rule induction", []),
            ("Validation, quality assurance and governance (versioning, diff, rollback, RBAC, audit)", []),
            ("Neo4j ontology graph database and standards exports (RDF/OWL)", []),
            ("Ontology-aware retrieval and constrained generation", []),
            ("Evaluation: benchmarks, ablations, multi-model, scalability, human evaluation (Year 2)", [])]],
        ["This year (2026)", "Objective", "**An end-to-end, measured Year-1 prototype on the Jeju tourism "
                                          "domain covering all 16 WBS line items.**"],
        [None, "Main content", [
            ("Task 1: planning, architecture, data profiling, schema v0", []),
            ("Task 2: LLM extraction pipeline and prompting study; normalization, formalization, validation", []),
            ("Task 3: Neo4j storage, versioning, governance; ontology-aware retrieval and constrained reasoning", []),
            ("Task 4: benchmark, five-variant evaluation, ablations; scalability, robustness, final outputs", [])]],
    ], [2.2, 2.4, 12.4], size=10, label_cols=(0, 1))
    t = doc.tables[-1]
    t.cell(0, 0).merge(t.cell(1, 0))
    t.cell(2, 0).merge(t.cell(3, 0))
    H.spacer(doc, 4)
    figure(doc, FIG / "fig02_wbs_gantt_2026.png", "2-1", "Year-1 WBS schedule and progress (all items 100%)",
           "wbs/deliverables/2026-08월 공동연구 WBS_제주대(윤여찬)_진척완료.xlsx (progress and evidence columns "
           "verified in WBS_WORKBOOK_AUDIT.md, 16/16 items)", 17.0)

    H.section_heading(doc, 2, "Process and Content", None)
    st_1_1(doc)
    st_1_2(doc)
    st_2_1(doc)
    st_2_2(doc)
    st_3_1(doc)
    st_3_2(doc)
    st_4_1(doc)
    st_4_2(doc)
    H.page_break(doc)


# ---------------------------------------------------------------- Task 1.1
def st_1_1(doc):
    def out(c):
        para(c, "○ All Task 1.1 deliverables exist and were verified against the workbook's evidence column "
                "(4/4 and 3/3 requirements).", 10, align="l", hanging=0.35, after=3)
        caption_table(c, "2-1", "Task 1.1 deliverables")
        data_table(c, ["Deliverable", "Content", "Evidence path"], [
            ["System architecture document", "Requirements, five core modules, interfaces, domain scope",
             "docs/system_requirements.md"],
            ["Implementation roadmap", "Module order, milestones, dependencies", "docs/implementation_roadmap.md"],
            ["Experiment tracking template", "Per-experiment audit record (config, seed, data, outputs)",
             "docs/experiment_audit_table.md"],
            ["Risk management plan", "Risk register with likelihood, impact and mitigations",
             "docs/risk_management_plan.md"],
            ["Reproducibility policy", "Run order, pinned requirements, manifest with sha256 of outputs",
             "wbs/deliverables/REPRODUCIBILITY/"],
        ], [3.6, 5.2, 4.8], size=8)
        source(c, "WBS_WORKBOOK_AUDIT.md, items 1.1.1 and 1.1.2")
    subtask(doc, "Task 1.1  Research planning and system architecture design", "May - Jun 2026",
            ["Define system requirements, the core modules (ontology generation, validation, retrieval, reasoning, "
             "evaluation), the overall architecture and the domain scope",
             "Design the repository structure, experiment tracking, reproducibility policy and risk plan"],
            [("Requirement analysis from the proposal (§§3-11) and the WBS", [
                "Mapped each proposal requirement to a module and an evaluation measure"]),
             ("Architecture: five modules around one ontology store (Figure 1-1)", [
                 "Ontology generation, validation and governance, storage (Neo4j), retrieval and constrained "
                 "generation, evaluation"]),
             ("Domain scope fixed to six Jeju tourism task types used throughout the benchmark", []),
             ("Engineering policy", [
                 "Fixed seeds, held-out test split never used for tuning, every reported number produced by a "
                 "named script and recorded in a run order",
                 "GPU use restricted to the shared server's allotted devices; frozen baselines never overwritten"])],
            out)


# ---------------------------------------------------------------- Task 1.2
def st_1_2(doc):
    def out(c):
        para(c, "○ A clean, validated corpus and a documented schema v0 that the extraction pipeline targets.",
             10, align="l", hanging=0.35, after=3)
        figure(c, FIG / "fig03_dataset_profile.png", "2-2", "Task taxonomy and splits of the cleaned corpus",
               "artifacts/reports/dataset_statistics.json; data/reports/data_profile_report.md", IN_CELL_W)
        caption_table(c, "2-2", "Ontology schema v0: core object properties (excerpt)")
        data_table(c, ["Property", "Domain", "Range", "Meaning"], [
            ["located_in", "Attraction, City, Island, Accommodation, Restaurant", "Destination, Region, City", "Geographic containment"],
            ["near", "Attraction, City", "Attraction, City", "Proximity"],
            ["served_at", "Cuisine", "Restaurant", "Dining"],
            ["reachable_by", "Destination, Attraction", "TransportMode, Route", "Access method"],
            ["has_risk", "Destination, Activity, Attraction", "SafetyAdvice", "Safety concern"],
            ["avoid_during", "Destination, Attraction, Activity", "Season, WeatherCondition", "Negative seasonality"],
            ["suitable_for", "Attraction, Activity, Destination", "TravelerType", "Traveller suitability"],
            ["requires", "Activity, Attraction", "Equipment, Permit, ...", "Prerequisite / policy"],
            ["has_accessibility_feature", "Attraction, Accommodation", "Accessibility vocabulary", "Disability access"],
            ["is_a / instance_of", "Any", "Class", "Taxonomy and class membership"],
        ], [3.0, 4.2, 3.4, 3.0], size=7.5)
        source(c, "docs/tourism_ontology_schema.md (24 object properties, data properties for label, alias, "
                  "confidence, coordinates, rating); docs/ONTOLOGY_SCHEMA_EXPLANATION.md")
    subtask(doc, "Task 1.2  Tourism data profiling and ontology schema design", "Jun - Jul 2026",
            ["Collect and pre-process tourism data: cleaning, deduplication, metadata annotation and validation",
             "Design ontology schema v0: concepts, relations, hierarchy, rules, provenance, confidence, taxonomy"],
            [("Corpus: 46,059 Jeju tourism records; schema validation 100% (0 invalid), 0 duplicates after "
              "deduplication", ["Splits: train 36,847, validation 4,605, test 4,607 (test held out from all tuning)"]),
             ("Metadata: multi-label task taxonomy (dining 90.8%, itinerary 83.3%, accommodation 79.3%, "
              "attractions 76.1%, transport 74.9%; safety 12.2%, accessibility 11.1%)", [
                  "Low accessibility coverage later limited the benchmark (47 accessibility items)"]),
             ("Schema v0: class hierarchy (Destination, NaturalAttraction, CulturalAttraction, Cuisine, "
              "TransportMode, TravelerType, SafetyAdvice, ...), object and data properties, provenance and "
              "confidence on every element", [])],
            out)


# ---------------------------------------------------------------- Task 2.1
def st_2_1(doc):
    def out(c):
        para(c, "○ Extraction modules and a prompt library with five strategies plus a staged (unified) "
                "pipeline; every output carries evidence and a confidence score "
                "(artifacts/ontologies/1.4.0/concepts.json).", 10, align="l", hanging=0.35, after=2)
        para(c, "○ Run-to-run agreement of the concept set is only 0.637-0.675 (mean Jaccard), so a single greedy "
                "run measures the draw as much as the strategy; the supported aggregate order is CoT > repair > "
                "metacognitive > naive > GoT.", 10, align="l", hanging=0.35, after=2)
        figure(c, FIG / "fig09_prompting_strategies.png", "2-3",
               "Prompting strategies: stability, yield and the H1 consistency test",
               "wbs/deliverables/task2_extraction/prompt_sensitivity__gpu5/6.json, hypothesis_h1__s0/1.json "
               "(Llama-3.1-8B-Instruct, bf16)", IN_CELL_W)
        caption_table(c, "2-3", "Hypothesis H1 (structured prompting beats naive prompting)")
        data_table(c, ["Strategy", "Sample 0: wins / 12 metrics", "Sample 1: wins / 12 metrics", "Beats naive overall"], [
            ["Chain-of-thought (CoT)", "3", "5", "No"],
            ["Graph-of-thought (GoT)", "3", "3", "No"],
            ["Metacognitive", "3", "3", "No"],
            ["Repair-after-failure", "3", "3", "No"],
            ["Unified staged pipeline", "= CoT (removed 1 of 963 relations)", "= CoT", "No (extra time per record)"],
        ], [3.8, 3.4, 3.4, 3.0], size=8)
        source(c, "wbs/deliverables/task4_evaluation/HYPOTHESES.md. Verdict: H1 not supported in both disjoint "
                  "20-record samples")
    subtask(doc, "Task 2.1  LLM-based tourism ontology extraction pipeline", "Aug - Sep 2026",
            ["Implement concept extraction, relation extraction, hierarchy induction and rule generation, and "
             "serialize outputs with evidence and confidence",
             "Develop and compare prompting strategies (naive, CoT, GoT, metacognitive, staged pipeline), "
             "including sensitivity and stability analysis"],
            [("Modules: src/ontology/concept_extractor.py, relation extractor, hierarchy induction, rule "
              "generator; prompt library src/llm/prompt_manager.py", []),
             ("Extractor LLM: Llama-3.1-8B-Instruct on local GPUs; JSON-schema-constrained outputs", []),
             ("Stability: 3 runs x 15 records at temperature 0.7; sensitivity: 3 disjoint 15-record samples", []),
             ("H1 test: each strategy's extraction assembled into an ontology and passed through the Task 2.2 "
              "validators; consistency counted per 100 relations so volume is not rewarded", []),
             ("Rule layer: 15 rules (8 written from domain knowledge, 7 mined from the graph)", [])],
            out)


# ---------------------------------------------------------------- Task 2.2
def st_2_2(doc):
    def out(c):
        caption_table(c, "2-4", "Normalization and four-layer validation")
        data_table(c, ["Step / layer", "Result"], [
            ["Normalization (v1.3.0 to v1.3.0-norm)", "1,585 to 1,564 concepts; 8,324 to 7,906 relations "
             "(418 duplicate relations removed, 7 alias merges, 14 duplicate collapses, 12 canonical recasings); "
             "92 near-duplicates flagged, not merged"],
            ["L1 structural", "0 errors"],
            ["L2 typing and symmetry", "920 warnings, 2,441 info (repair suggestion attached to each finding)"],
            ["L3 SHACL (pySHACL)", "Conforms = True, 0 results"],
            ["L4 OWL-RL closure", "Consistent; 17,706 to 27,521 triples (9,815 entailed)"],
            ["Serializations", "Turtle, OWL/XML, JSON-LD, N-Triples"],
        ], [4.6, 9.0], size=8)
        source(c, "wbs/deliverables/task2_formalization/validation_report.json, normalization_report.json")
        figure(c, FIG / "fig08_validation.png", "2-4", "Validation findings and OWL-RL closure",
               "wbs/deliverables/task2_formalization/validation_report.json", IN_CELL_W)
        caption_table(c, "2-5", "Ontology quality metrics")
        data_table(c, ["Metric", "Value", "Metric", "Value"], [
            ["Concepts / relations", "1,564 / 7,906", "Classes / distinct predicates", "31 / 19"],
            ["Average degree", "10.11", "Max degree", "1,598 (Jeju)"],
            ["Isolated concepts", "0", "Provenance coverage", "100%"],
            ["Mean / max hierarchy depth", "2.255 / 4", "Relationship richness", "0.613"],
            ["Directional conflicts", "328", "Branching Gini (balance)", "0.657"],
            ["Relation redundancy", "0.0", "Rules that can fire (of 15)", "11"],
        ], [3.8, 3.0, 3.8, 3.0], size=8, align=["l", "c", "l", "c"])
        source(c, "validation_report.json quality_metrics (v1.3.0-norm, the serialized release); conflicts, balance and rules measured on v1.4.0 (rule_compliance.md)")
        para(c, "○ Only 4 of 8 seed rules were enforceable on the data; a conservative weather-state extraction "
                "(53 avoid_during triples, v1.5.0) raised this to 6 of 8. The remaining two are blocked by the "
                "corpus (graded risk severity appears in 13 of 46,059 records).", 10, align="l", hanging=0.35)
    subtask(doc, "Task 2.2  Ontology normalization, formalization and validation", "Sep - Oct 2026",
            ["Synonym normalization, duplicate merging, canonical naming and conversion to RDF/OWL",
             "Validation: logical consistency, structure, SHACL/rule constraints, reporting with repair suggestions"],
            [("Normalization keeps evidence: uncertain near-duplicates are flagged for review rather than merged", []),
             ("Validation engine wbs/task2_formalization/validate_ontology.py runs four layers in order and "
              "emits a machine-readable report", []),
             ("Quality metrics include balance (Gini) and directional conflicts, which the evaluation plan "
              "names and the first report lacked", []),
             ("Governance gate: a change proposal cannot be approved while validation reports errors", [])],
            out)


# ---------------------------------------------------------------- Task 3.1
def st_3_1(doc):
    typed = NEO_Q["mappings"]["typed"]
    gen = NEO_Q["mappings"]["generic"]
    q_rows = []
    for tq, gq in zip(typed["queries"], gen["queries"]):
        q_rows.append([tq["query"], tq["question"], str(tq["rows"]), f'{tq["p50_ms"]:.2f}', f'{gq["p50_ms"]:.2f}'])

    def out(c):
        para(c, "○ **Neo4j is the project's ontology graph database.** It holds the evaluated release v1.4.0, "
                "including the governance-approved change P0001, as a typed property graph and answers the "
                "competency questions in Cypher. The versioned file store holds the same content; RDF exports "
                "carry standards validation and interoperability.", 10, align="l", hanging=0.35, after=3)
        caption_table(c, "2-6", "Neo4j deployment (read back from the live server)")
        data_table(c, ["Item", "Value"], [
            ["Engine", f'Neo4j {NEO["neo4j_version"]} Community; Bolt bolt://127.0.0.1:7690, HTTP 7475'],
            ["Content", f'Ontology v1.4.0: {NEO["nodes"]:,} nodes, {NEO["relationships"]:,} relationships, '
                        f'{len(NEO["labels"]) + 1} labels, {len(NEO["relationship_types"])} relationship types'],
            ["Load time", f'{typed["load"]["load_s"]:.3f} s typed mapping (schema {typed["load"]["schema_s"]} s, '
                          f'nodes {typed["load"]["nodes_s"]} s, relationships {typed["load"]["edges_s"]} s); '
                          f'0 relationships lost'],
            ["Schema", "Uniqueness constraint on Concept.name; range indexes on Concept.name and concept_class"],
            ["Node properties", "name, concept_class, aliases, confidence, source, validator_status (all nodes); "
                                "lat, lon (67 gazetteer places)"],
            ["Relationship properties", "predicate, confidence, source (all 7,909)"],
            ["Operations", "wbs/task3_storage/neo4j_server.sh (start/stop), neo4j_store.py (load + queries), "
                           "neo4j_schema.py (schema read-back), scripts/ingest_to_neo4j.py"],
            ["Data directory", "artifacts/graph_db/neo4j"],
        ], [3.2, 10.4], size=8)
        source(c, "wbs/deliverables/task3_storage/graph_db_neo4j.json, ontology_schema_neo4j.md; live queries "
                  "of 3 Nov 2026 (annual_report_2026/figures/neo4j_live_snapshot.json)")
        caption_table(c, "2-7", "Mapping of ontology elements to the Neo4j property graph")
        data_table(c, ["Ontology element", "Neo4j representation", "Example"], [
            ["Concept (individual)", "Node labelled :Concept plus one class label", "(:Concept:NaturalAttraction "
                                                                                   "{name:'Seongsan Ilchulbong'})"],
            ["Class", "Node label (33 class labels); class nodes carry :OntologyClass", ":Destination (537 nodes)"],
            ["Relation (triple)", "Relationship typed by the predicate (upper case)", "-[:SERVED_AT]-> (861)"],
            ["Hierarchy", "IS_A and SUBCLASS_OF relationships", "IS_A (1,541), SUBCLASS_OF (31)"],
            ["Provenance, confidence", "Properties on every node and relationship", "source: 'governance/P0001'"],
            ["Governance-approved change", "Ordinary nodes/relationships with governance provenance",
             "(Seongsan Ilchulbong)-[:HAS_DESIGNATION]->(UNESCO World Heritage)"],
        ], [3.0, 5.4, 5.2], size=7.5)
        source(c, "wbs/task3_storage/neo4j_store.py (typed mapping)")
        figure(c, FIG / "fig05_neo4j_schema_distribution.png", "2-5",
               "Node labels and relationship types in the live Neo4j database",
               "Live Cypher queries; annual_report_2026/figures/neo4j_live_snapshot.json", IN_CELL_W)
        figure(c, FIG / "fig06_neo4j_subgraph.png", "2-6",
               "A Neo4j neighbourhood: Seongsan Ilchulbong and its typed relationships",
               "Live Cypher query (up to four neighbours per relationship type)", IN_CELL_W)
        caption_table(c, "2-8", "Nine competency questions answered in Cypher (median of 25 runs)")
        data_table(c, ["Query", "Question", "Rows", "Typed p50 (ms)", "Generic p50 (ms)"], q_rows,
                   [2.6, 5.6, 1.2, 2.1, 2.1], size=7.5)
        source(c, "wbs/deliverables/task3_storage/graph_db_neo4j.json. 'Generic' is the :RELATES_TO mapping "
                  "used by src/graph/cypher_queries.py; the typed mapping loads 3.8x faster (0.51 s vs 1.97 s)")
        para(c, "Example Cypher (Q5 itinerary 2-hop path and Q6 safety constraints):", 9, bold=True, align="l",
             after=2)
        code_block(c, [
            "MATCH (p)-[:LOCATED_IN]->(r), (p)-[:NEAR]->(n)",
            "RETURN r.name AS region, p.name AS place, n.name AS neighbour LIMIT 20",
            "",
            "MATCH (s)-[r:HAS_RISK|AVOID_DURING|REQUIRES]->(o)",
            "RETURN s.name AS label, type(r) AS pred, o.name AS obj LIMIT 20",
        ], width=IN_CELL_W)
        H.spacer(c, 2)
        figure(c, FIG / "fig07_cypher_latency.png", "2-7",
               "Competency-question latency on Neo4j and on the SPARQL paths (same questions)",
               "graph_db_neo4j.json, graph_db.json (Oxigraph), sparql_store.json (rdflib)", IN_CELL_W)
        para(c, "○ All nine questions run in under 1.5 ms (median) on Neo4j. 8 of 9 return the same row count as "
                "SPARQL. The ninth difference exposed a real defect: the RDF exports were still at v1.3.0-norm "
                "and lacked P0001 (UNESCO World Heritage and three has_designation relations), while Neo4j and "
                "the file store carry v1.4.0. Regenerating the exports is scheduled (Chapter 5) rather than "
                "applied silently, because it changes published RDF figures.", 10, align="l", hanging=0.35, after=3)
        figure(c, FIG / "fig04_ontology_versions.png", "2-8", "Ontology releases with content-hashed diffs",
               "wbs/deliverables/task3_governance/diff_logs.json", IN_CELL_W)
        caption_table(c, "2-9", "Versioning and governance controls")
        data_table(c, ["Control", "Implementation and measured state"], [
            ["Versioning and diff", "Semantic versions 0.1.0 to 3.0.0; additive/breaking classification; content "
                                    "hash over canonical concept and triple sets (showed v1.1.0 = v1.2.0)"],
            ["Rollback", "Published pointer can be moved back to any stored version; staging vs production isolation"],
            ["Change workflow", "Proposal state machine; approval impossible while validation reports errors"],
            ["Role-based access", "4 roles (reader, automation, expert, admin) x 13 permissions, checked at every entry point"],
            ["Protection", "158 protected elements (4 human-approved, 154 safety-constrained); override needs admin"],
            ["Audit trail", "36 allowed and 42 denied decisions on record; end-to-end demo: 8 of 16 attempts allowed"],
            ["Separation of duties", "Enforced; it flagged that P0001 was created and approved by the same account"],
        ], [3.2, 10.4], size=8)
        source(c, "wbs/deliverables/task3_governance/ (diff_logs.json, audit_trail.jsonl, access_control.md, "
                  "access_control_demo.md, version_governance_report.md)")
    subtask(doc, "Task 3.1  Ontology storage (Neo4j), versioning and governance", "Aug - Sep 2026",
            ["Implement ontology storage in a graph database and file-based systems",
             "Implement versioning, change detection, diff analysis, rollback, governance workflow and incremental "
             "evolution with provenance tracking"],
            [("Neo4j 5.26.0 installed with its own JDK under the user's account; server script for Bolt 7690 "
              "and HTTP 7475", []),
             ("Two mappings measured: typed (label = class, relationship type = predicate) and generic "
              "(:RELATES_TO with the predicate as a property); the typed mapping is adopted", []),
             ("Batch ingestion with UNWIND; schema constraint and indexes created before data; counts read back "
              "and checked against the file store", []),
             ("Nine competency questions (one per task type plus governance and a transitive property path) "
              "answered in Cypher and cross-checked against SPARQL (rdflib, Oxigraph)", []),
             ("Governance: proposals, approvals, protection, isolation and audit enforced in code", [])],
            out)


# ---------------------------------------------------------------- Task 3.2
def st_3_2(doc):
    def out(c):
        caption_table(c, "2-10", "Ontology-aware retriever characterization (347 benchmark questions)")
        data_table(c, ["Measure", "Value"], [
            ["Intent groups", "8 keyword groups mapped to ontology classes; 2.9% of questions fall back to a "
                              "7-class default"],
            ["Concept mapping", "A specific (non-root) entity resolved in 18.7% of questions"],
            ["Subgraph size", "32.35 entities and 80.19 facts per question; 50.5% from intent-to-class routing"],
            ["Gold-entity coverage", "Ontology context 38.9% vs Simple RAG 42.3%; best question-dependent "
                                     "policy 53.1%; question-independent control (global_degree80) 96.9%"],
            ["Constraint types", "Ungrounded entity, unsupportable claim (prices, hours, timetables, "
                                 "accessibility guarantees), contradicted spatial relation"],
        ], [3.4, 10.2], size=8)
        source(c, "wbs/deliverables/task3_retrieval/retriever_characterization.md, onto_retrieval_tuning.md")
        caption_table(c, "2-11", "Retrieval ablation: what each component contributes (no generation)")
        data_table(c, ["Ontology context", "Gold-entity coverage (%)", "Change"], [
            ["Full (dense cards + class routing + temporal facts)", "38.9", "-"],
            ["Without temporal facts", "41.6", "+2.7 pp"],
            ["Without class routing (dense cards only)", "9.2", "-29.7 pp"],
            ["Without dense search (class routing only)", "41.9", "+3.0 pp"],
            ["Dense only, twice the entities", "8.8", "-30.2 pp"],
        ], [7.4, 3.4, 2.8], size=8)
        source(c, "FINAL_REPORT.md §6.1. Class routing carries the ontology retriever; dense card search does not")
        caption_table(c, "2-12", "Constrained generation: validation-layer operating points (Llama-3.1-8B, 347 items)")
        data_table(c, ["Policy", "Token F1", "Ungrounded (%)", "Clean (%)", "Validated", "Layer cost"], [
            ["V2 Simple RAG (reference)", "47.00", "3.75", "78.09", "-", "-"],
            ["V4 draft, layer off", "46.35", "3.58", "78.02", "0", "-"],
            ["V5 validate and regenerate every answer", "39.19", "2.20", "88.16", "347", "1 generation"],
            ["V5 selective (>= 1 flagged span)", "44.36", "1.91", "90.40", "71", "1 generation"],
            ["V5 surgical clause repair, gated", "46.14", "0.00*", "100.00*", "71", "1.5 ms"],
        ], [5.0, 1.6, 1.9, 1.6, 1.6, 1.9], size=7.5)
        source(c, "FINAL_REPORT.md §5.1c, §5.1h; repair_policy__llama3.1-8b.json. * Zero by construction under "
                  "the published detector; under the strict detector the surgical arm is 2.15% vs 5.66% for "
                  "Simple RAG")
    subtask(doc, "Task 3.2  Ontology-aware retrieval and constrained reasoning", "Oct - Nov 2026",
            ["Query understanding, concept mapping, ontology-based subgraph retrieval, hybrid retrieval and "
             "context assembly",
             "Constrained generation with ontology rules, post-generation validation, repair loop, explanation "
             "trace and uncertainty fallback"],
            [("Retriever: intent-to-class routing plus dense search over entity cards, temporal facts, and a "
              "hybrid mode that adds the top retrieved passages", []),
             ("Context policies swept (round-robin, grouped encoding, deeper routing, budgets 500-3,000 "
              "characters) and carried through generation at three model sizes", []),
             ("Reasoner (wbs/task3_reasoning/constrained_reasoner.py): validates the draft against the "
              "ontology, attaches a per-sentence evidence trace, abstains below a coverage floor", []),
             ("Repair redesigned from 'regenerate the whole answer' to surgical clause removal, which copies "
              "every other byte of the draft and needs no second generation", [])],
            out)


# ---------------------------------------------------------------- Task 4.1
def st_4_1(doc):
    def out(c):
        caption_table(c, "2-13", "Benchmark composition (347 items, six WBS task types)")
        data_table(c, ["Category", "Available", "Selected"], [
            ["Recommendation", "118", "60"], ["Itinerary", "2,600", "60"], ["Safety", "153", "60"],
            ["Dining", "274", "60"], ["Accessibility", "47", "47 (corpus-limited)"], ["Transport", "73", "60"],
        ], [5.0, 4.0, 4.6], size=8)
        source(c, "task4_evaluation/benchmark_manifest.json; ambiguous (764) and unmatched (578) items discarded")
        caption_table(c, "2-14", "Five system variants, Llama-3.1-8B (published configuration)")
        data_table(c, ["Variant", "Token F1", "Entity F1", "Entity hit", "Ungrounded (%) ↓", "Clean (%) ↑"], [
            ["V1 LLM only", "27.1", "17.5", "69.7", "11.7", "34.9"],
            ["V2 Simple RAG", "47.0", "23.0", "73.6", "3.8", "78.1"],
            ["V3 Ontology only", "42.4", "20.8", "69.1", "3.7", "73.9"],
            ["V4 Ontology + RAG", "45.8", "23.2", "73.4", "4.0", "75.5"],
            ["V5 Ontology + RAG + validation", "39.8", "20.1", "70.1", "4.5", "79.8"],
            ["V5 with dedicated adapter (hybrid20d)", "39.2", "-", "-", "2.2", "88.2"],
        ], [4.6, 1.7, 1.7, 1.7, 2.0, 1.9], size=8)
        source(c, "variant_scores__llama3.1-8b.json; hybrid_adapter_comparison.md (V5 vs V2: -1.44 pp ungrounded, "
                  "p = 0.0018; +10.06 pp clean, p = 1.9e-05)")
        figure(c, FIG / "fig10_five_variants.png", "2-9", "Five variants at three model sizes",
               "variant_scores__llama3.2-1b/3b, llama3.1-8b.json", IN_CELL_W)
        figure(c, FIG / "fig11_category_heatmap.png", "2-10", "Results per task type (Llama-3.1-8B)",
               "variant_scores__llama3.1-8b.json (by_category)", IN_CELL_W)
        figure(c, FIG / "fig12_repair_tradeoff.png", "2-11",
               "Repair policies: strict ungrounded rate vs token F1 at three sizes",
               "repair_policy__{model}.json; strict rate = audit_spans(strict=True)", IN_CELL_W)
        figure(c, FIG / "fig13_verifier_across_models.png", "2-12",
               "The ontology verifier applied to six models' Simple RAG answers (4,607 questions each)",
               "verifier_across_models__test.json; token-F1 cost 0.29-0.34 points", IN_CELL_W)
        figure(c, FIG / "fig14_cross_model_tiers.png", "2-13",
               "Ontology-only vs Simple RAG on 4,607 paired questions, by training-budget tier",
               "cross_model_results.json (2,000 bootstrap samples, Holm-corrected). The untuned-RAG tier "
               "(8B models, RAG arm not trained) is excluded from all claims", IN_CELL_W)
        caption_table(c, "2-15", "The proposal's hypotheses and their verdicts")
        data_table(c, ["Hypothesis", "Measurement", "Verdict"], [
            ["H1 structured prompting improves logical consistency and hierarchy",
             "12 validator metrics per strategy, 2 disjoint samples", "Not supported"],
            ["H2 ontology constraints reduce hallucination without hurting fluency",
             "Paired ungrounded rate vs V2; perplexity 4.49 (ontology) vs 4.70 (RAG)", "Partially supported"],
            ["H3 ontology integration improves interpretability and verifiability",
             "Attribution 50.3% vs 53.5%; per-sentence evidence trace; auditable store", "Partially supported"],
        ], [5.0, 6.0, 2.6], size=8)
        source(c, "task4_evaluation/HYPOTHESES.md, HYPOTHESES.json; consistency__llama3.1-8b__gpu5/6.json")
    subtask(doc, "Task 4.1  Benchmark construction and system evaluation", "Nov 2026",
            ["Construct benchmark datasets for the six tourism task types",
             "Evaluate the five system variants and run ablations on prompts, validation, retrieval and reasoning"],
            [("Metrics: token P/R/F1, entity P/R/F1 over the gazetteer, entity hit, ungrounded-entity rate "
              "(audit-grade and strict), fully grounded ('clean') share, fluency, attribution, consistency", []),
             ("Arm-dedicated LoRA adapters per variant and model size (a shared adapter was shown to bias the "
              "comparison)", []),
             ("Paired tests per item with bootstrap confidence intervals and Holm correction", []),
             ("Cross-model study: stored answers of 8 models (6 model lines) re-scored on 4,607 held-out "
              "questions; training budgets read from receipts and reported in tiers (equal, extended, matched)", [])],
            out)


# ---------------------------------------------------------------- Task 4.2
def st_4_2(doc):
    def out(c):
        caption_table(c, "2-16", "Scalability from the delivered 7,909 triples to 100,000")
        data_table(c, ["Operation", "7,909 triples", "100,000 triples", "Behaviour"], [
            ["1-hop subgraph p95", "0.118 ms", "0.094 ms", "Flat"],
            ["2-hop subgraph p95", "0.239 ms", "0.191 ms", "Flat"],
            ["Ontology parse", "0.35 s", "2.93 s", "Linear"],
            ["Entity-card search p50", "0.12 ms", "2.81 ms", "Linear (flat index; fixable)"],
            ["Answer validation p50", "1.06 ms", "13.3 ms", "Linear (substring fallback; fixable)"],
            ["Dense passage search p50", "7.7 ms at 36,847", "20.8 ms at 100,000 passages", "Linear"],
            ["Generation (V1, 8B)", "11.62 s per request", "-", "Dominates by three orders of magnitude"],
        ], [3.6, 2.8, 3.4, 3.8], size=8)
        source(c, "task4_scalability/ontology_scalability.json, retrieval_scalability.json (disjoint-namespace "
                  "clones of the real graph preserve shape)")
        figure(c, FIG / "fig15_scalability.png", "2-14", "Latency and cost as the ontology and corpus grow",
               "task4_scalability/*.json", IN_CELL_W)
        figure(c, FIG / "fig16_robustness.png", "2-15", "Robustness under seven input conditions",
               "task4_scalability/robustness.json", IN_CELL_W)
        para(c, "○ Under 15% character noise (typo15) Simple RAG's gold-entity coverage falls by 4.8 points while "
                "the ontology arm's does not fall under any condition; in generated answers the ontology arm "
                "scores 0.9 points above RAG under typo15, the one reference-overlap win in the project. Under "
                "truncation and distractors RAG keeps its lead.", 10, align="l", hanging=0.35, after=3)
        caption_table(c, "2-17", "Final research outputs")
        data_table(c, ["Output", "Path"], [
            ["Visualization figures and index", "wbs/deliverables/figures/ (16 figures, FIGURE_INDEX.md)"],
            ["Evaluation and hallucination report", "task4_evaluation/hallucination_report__llama3.1-8b.md"],
            ["Final research report", "wbs/deliverables/FINAL_REPORT.md"],
            ["Reproducibility package", "wbs/deliverables/REPRODUCIBILITY/ (MANIFEST.json, RUN_ORDER.md)"],
            ["Business-plan deck and demo film", "wbs/deliverables/presentation/"],
        ], [5.0, 8.6], size=8)
        source(c, "WBS_WORKBOOK_AUDIT.md items 4.2.1, 4.2.2")
    subtask(doc, "Task 4.2  Scalability verification and final research outputs", "Dec 2026 (completed early)",
            ["Evaluate scalability (10K-100K), latency, throughput and robustness under noisy input",
             "Produce visualizations, the final report and a reproducibility package"],
            [("Ontology operations measured on disjoint-namespace clones at 10K, 25K, 50K and 100K triples; "
              "OWL-RL capped at 25K and SHACL at 100K", []),
             ("Seven paired input perturbations: character noise at 5% and 15%, lowercase, 70% truncation, "
              "appended distractor, corrupted entity name", []),
             ("Figures generated from the result files by script, never by hand", [])],
            out)


# ====================================================================== chapter 3
ACHIEVE = [
    ["1.1.1 Requirements, core modules, architecture, domain scope",
     "Architecture document, roadmap, experiment template, risk plan delivered (4/4 verified)", "100"],
    ["1.1.2 Repository, tracking, reproducibility, risk management",
     "Repository layout, run order, sha256 manifest, risk register (3/3 verified)", "100"],
    ["1.2.1 Collect, clean, deduplicate, annotate, validate data",
     "46,059 records, 100% valid, 0 duplicates; profile report; splits 36,847 / 4,605 / 4,607", "100"],
    ["1.2.2 Ontology schema v0", "Classes, 24 object properties, data properties, provenance and confidence", "100"],
    ["2.1.1 Concept, relation, hierarchy, rule extraction with evidence and confidence",
     "Extraction modules; serialized ontology with evidence and confidence; 15 rules", "100"],
    ["2.1.2 Prompting strategies with sensitivity and stability analysis",
     "5 strategies + staged pipeline; Jaccard 0.637-0.675; H1 tested (not supported)", "100"],
    ["2.2.1 Normalization and RDF/OWL conversion",
     "1,585 to 1,564 concepts, 8,324 to 7,906 relations; Turtle, OWL, JSON-LD, N-Triples", "100"],
    ["2.2.2 Validation with SHACL, consistency checks, repair suggestions",
     "L1 0 errors; SHACL conforms; OWL-RL consistent; quality, balance, conflict metrics", "100"],
    ["3.1.1 Ontology storage in graph database and files",
     "Neo4j 5.26: 1,565 nodes, 7,909 relationships, 9 Cypher competency questions < 1.5 ms; file and RDF stores", "100"],
    ["3.1.2 Versioning, diff, rollback, governance, provenance",
     "9 content-hashed transitions; RBAC 4 roles x 13 permissions; 158 protected elements; audit trail", "100"],
    ["3.2.1 Query understanding, subgraph and hybrid retrieval, context assembly",
     "8 intent groups; 32.35 entities / 80.19 facts per question; policies swept at 3 sizes", "100"],
    ["3.2.2 Constrained generation, validation, repair, trace, fallback",
     "Validator, surgical repair (1.5 ms), evidence trace, abstention", "100"],
    ["4.1.1 Benchmark for six task types", "347 items (accessibility corpus-limited at 47)", "100"],
    ["4.1.2 Five-variant evaluation and ablations",
     "3 model sizes; 4 ablation axes; 8-model, 4,607-question cross-check; H1-H3 verdicts", "100"],
    ["4.2.1 Scalability 10K-100K, latency, throughput, robustness",
     "Flat subgraph latency to 100K; 7 robustness conditions", "100"],
    ["4.2.2 Visualizations, final report, reproducibility package",
     "16 figures, final report, reproducibility package, deck and demo film", "100"],
]


def chapter3(doc):
    H.chapter_banner(doc, 3, "R&D Results and Degree of Goal Achievement")
    H.section_heading(doc, 1, "Achievement Level of This Year's Objectives",
                      "3-1. Achievement Level of This Year's Objectives")
    caption_table(doc, "3-1", "Achievement of the 2026 objectives (WBS line items)")
    data_table(doc, ["Objective", "Achievement", "Achievement (%)"], ACHIEVE + [
        ["Overall (16 WBS line items)", "16 of 16 items complete with evidence verified on disk", "100"]],
        [6.2, 8.6, 2.2], size=8, align=["l", "l", "c"], bold_rows=(len(ACHIEVE),))
    source(doc, "2026-08월 공동연구 WBS_제주대(윤여찬)_진척완료.xlsx; WBS_WORKBOOK_AUDIT.md")
    note(doc, "Against the full research proposal, 46 of 49 requirements are met. The three open items need human "
              "participants (expert rating of ontology quality §11.4.2, manual hallucination annotation §11.6b, "
              "and the related agreement study). The instruments (115 and 150 items, blinded) and the scoring "
              "code are complete; ratings are scheduled for Year 2 (PROPOSAL_COMPLIANCE.md).", 9)

    H.section_heading(doc, 2, "Qualitative R&D Outcomes", "3-2. Qualitative R&D Outcomes")
    box(doc, "Scientific outcomes")
    bullets(doc, [
        ("A controlled answer to the project's central question: does grounding in an ontology beat passage "
         "retrieval over the same corpus?",
         ["On reference-overlap metrics, no: Simple RAG leads token F1 at every size on the benchmark, and at "
          "equal training budget on 4,607 questions no model's ontology arm leads on every metric while all "
          "four lose token precision.",
          "On groundedness, yes, in combination with retrieval: the hybrid arm with the verifier beats Simple "
          "RAG on both groundedness measures at all three sizes (Table 2-12, Figure 2-11)."]),
        ("Measurement validity was tested rather than assumed.",
         ["Rewording every reference shrinks Simple RAG's lead by 17-22%, showing that single-source references "
          "inflate but do not create it.",
          "Training-budget confounds were made explicit: with matched budgets, all apparent ontology wins at "
          "Llama 3.2 vanish, while groundedness wins at Qwen 2.5 sizes survive (Figure 2-13)."]),
        "Hypotheses were tested with verdict rules fixed in advance: H1 not supported, H2 and H3 partially "
        "supported. Negative results are reported as findings.",
    ])
    box(doc, "Technological outcomes")
    bullets(doc, [
        "Neo4j ontology graph database with a typed mapping, schema constraints, indexes and nine Cypher "
        "competency queries, all answered in under 1.5 ms (median).",
        "A model-agnostic ontology verifier: about 1.5 ms per answer, no GPU, transferable across six models "
        "from three families with no retraining (Figure 2-12).",
        "End-to-end governance in code: content-hashed versions, validation-gated approvals, RBAC, protection "
        "of safety-critical elements and an append-only audit trail.",
    ])
    box(doc, "Economic outcomes")
    bullets(doc, [
        ("The cost of changing what the system knows was measured (Figure 3-1).",
         ["A policy edit of 22 facts (dress codes, drone restrictions, permits, reef-safe sunscreen, cash "
          "payment, shoe removal) took 20 s of CPU re-indexing and 0 GPU-hours, with 0 validation errors.",
          "Teaching the same rules by fine-tuning would require retraining adapters that cost 135.1 GPU-hours "
          "in this project."]),
        "The ontology path adds no measurable serving cost: subgraph extraction stays at about 0.1 ms up to "
        "100K triples, while generation takes seconds.",
    ])
    H.figure(doc, FIG / "fig17_policy_edit.png", "3-1", "A knowledge edit reaches the answer without retraining",
             "demo_policy_before_after__llama3.2-3b.json; FINAL_REPORT.md §5.1j (at 1B the rule-stated count "
             "moves 7 to 8 of 10)", 15.5)
    box(doc, "Social outcomes")
    bullets(doc, [
        "Safer visitor information: unsupportable claims (prices, opening hours, timetables, accessibility "
        "guarantees no source supports) are detected and removed instead of being presented as fact.",
        "A data-gap finding of public value: accessibility is a stated target task, yet only 47 benchmark items "
        "could be built from the corpus; better accessibility data is a prerequisite for trustworthy services.",
        "Traceable answers (per-sentence evidence) support the transparency goals of the AI Framework Act.",
    ])
    box(doc, "Infrastructure outcomes")
    bullets(doc, [
        "A running Neo4j 5.26 ontology database with scripts for start-up, loading and schema read-back.",
        "Reusable research assets: the cleaned corpus and splits, the 347-item benchmark, ontology releases "
        "v0.1.0 to v1.6.0, evaluation code with paired statistics, and a reproducibility package.",
    ])

    H.section_heading(doc, 3, "Quantitative R&D Outcomes", "3-3. Quantitative R&D Outcomes")
    item(doc, "Outcomes registrable in 2026 are the technology summary and this report. No paper, conference "
              "presentation, intellectual-property filing or software registration is recorded in the project "
              "repository as of 3 November 2026; these are planned for 2027 (Section 5-4).")
    caption_table(doc, "3-2", "Quantitative R&D outcomes (unit: count, KRW thousand, USD, persons)")
    blank = ""
    rows = []
    reg = [("Papers (domestic / international)", "Count", "0"), ("Conference presentations", "Count", "0"),
           ("Technology summary information", "Count", "1"), ("Report full text", "Count", "1"),
           ("Bio-resources / compounds", "Count", "0"), ("IP (application)", "Count", "0"),
           ("IP (registration)", "Count", "0"), ("Copyright (software)", "Count", "0")]
    feat = [("Commercialisation", "Results", "0"), ("Technology transfer", "Count", "0"),
            ("Investment", "Amount", "0"), ("Sales", "Amount", "0"), ("Cost reduction", "Amount", "-"),
            ("Import substitution", "Amount", "0"), ("Job creation", "Persons", "0"),
            ("Promotion", "Count", "0"), ("Awards", "Count", "0")]
    for grp, lst_ in (("Registration / deposit indicators 1)", reg), ("Project-specific indicators 2)", feat)):
        for k, (name, unit, act) in enumerate(lst_):
            rows.append([grp if k == 0 else None, name, unit, "Target", blank, blank, blank, blank])
            rows.append([None, None, None, "Actual", "-", act, blank, act])
    n_reg = len(reg) * 2
    merges = [((2, 0), (1 + n_reg, 0)), ((2 + n_reg, 0), (1 + len(rows), 0))]
    for k in range(len(rows) // 2):
        r = 2 + 2 * k
        merges += [((r, 1), (r + 1, 1)), ((r, 2), (r + 1, 2))]
    data_table(doc, [["Indicator group", "Indicator", "Unit", "Stage / year", "Stage 1", None, None, "Total"],
                     [None, None, None, None, "2025", "2026", "2027", None]], rows,
               [3.0, 4.4, 1.5, 1.8, 1.5, 1.5, 1.5, 1.8], size=7.5,
               merges=[((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3)),
                       ((0, 4), (0, 6)), ((0, 7), (1, 7))] + merges)
    note(doc, "1) Registration/deposit indicators: papers (SCIE, non-SCIE, average impact factor), patents, report "
              "full text, research facilities and equipment, technology summary, copyright (software, books), "
              "bio-resources, standards, compounds, new varieties. For papers, presentations and patents the "
              "target-versus-actual comparison may be omitted.", 7.5)
    note(doc, "2) Project-specific indicators are chosen from technology transfer, commercialisation, cost "
              "reduction, certification, prototypes, human-resource development, policy use, promotion, awards "
              "and others. Targets come from the agreement and are left for the PI to enter; 2026 actuals are "
              "those recorded in the project repository on 3 November 2026 and should be updated before "
              "submission. Cost reduction is reported qualitatively (Section 3-2) rather than as an amount.", 7.5)

    H.section_heading(doc, 4, "Detailed Quantitative R&D Outcomes",
                      "3-4. Detailed Quantitative R&D Outcomes")
    note(doc, "Only applicable items are completed; supporting evidence is attached separately.", 9)
    box(doc, "Scientific outcomes")
    item(doc, "Papers in domestic and international journals")
    data_table(doc, ["No.", "Title", "Journal", "First author", "Vol./No.", "Country", "Publisher", "SCIE",
                     "Date", "ISSN", "Contribution"],
               [["-", "None in 2026 (manuscripts planned for 2027; see 5-4)", "", "", "", "", "", "", "", "", ""]],
               [0.8, 3.6, 1.8, 1.5, 1.1, 1.1, 1.5, 1.1, 1.3, 1.2, 2.0], size=7)
    item(doc, "Presentations at domestic and international conferences")
    data_table(doc, ["No.", "Conference", "Presenter", "Date", "Venue", "Country"],
               [["-", "None in 2026", "", "", "", ""]], [0.9, 6.0, 2.6, 2.4, 3.0, 2.1], size=7.5)
    item(doc, "Technology summary information")
    data_table(doc, ["Year", "Technology", "Summary", "Maturity", "Reg. no.", "Used", "Reason not used",
                     "External use", "Permission"],
               [["2026", "Neo4j-based governed tourism ontology with an ontology-grounded LLM answer verifier",
                 "LLM-built, validated ontology stored as a typed Neo4j property graph with versioning, RBAC and "
                 "audit; ontology-aware retrieval and a 1.5 ms verifier that removes ungrounded spans and halves "
                 "the strict ungrounded rate across six models without retraining",
                 "TRL 4 (laboratory-validated prototype)", TBD, "Yes (in-project evaluation)", "-", "No", TBD]],
               [1.0, 2.6, 5.2, 1.8, 1.3, 1.5, 1.2, 1.2, 1.2], size=7, align=["c", "l", "l", "c", "c", "c", "c",
                                                                             "c", "c"])
    item(doc, "Report full text")
    data_table(doc, ["Year", "Report type", "Issue date", "Registration no."],
               [["2026", "Annual report (Year 1)", TBD, TBD]], [2.0, 7.0, 4.0, 4.0], size=8)
    box(doc, "Technological outcomes")
    item(doc, "Intellectual property and copyright (software): none filed or registered in 2026. Candidate "
              "software for copyright registration in 2027: (1) the ontology construction, validation and "
              "governance toolkit; (2) the Neo4j ontology store and loader; (3) the ontology answer verifier.")
    box(doc, "Social outcomes")
    item(doc, "Training of research personnel")
    data_table(doc, [["No.", "Category", "Base year", "Status", None, None, None, None, None, None, None, None, None, None],
                     [None, None, None, "PhD", "MSc", "BSc", "Other", "Male", "Female", "Capital", "Chungcheong",
                      "Yeongnam", "Honam", "Other"]],
               [["1", "Participating researchers (NLP Lab)", "2026", "", "", "", "", "", "", "", "", "", "", ""]],
               [0.8, 3.4, 1.4] + [1.04] * 11, size=7,
               merges=[((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (0, 13))])
    note(doc, "Head counts by degree, gender and region are entered by the institution from the participation "
              "records.", 8)
    box(doc, "Infrastructure outcomes")
    item(doc, "Research facilities and equipment: none acquired with project funds is recorded in the project "
              "repository [to be confirmed by the institution]. The Neo4j database runs on the laboratory's "
              "existing GPU server.")
    box(doc, "Other outcomes")
    caption_table(doc, "3-3", "Research assets produced in 2026")
    data_table(doc, ["No.", "Asset", "Description", "Location"], [
        ["1", "Neo4j ontology database", "Ontology v1.4.0 as a typed property graph (1,565 nodes, 7,909 "
                                         "relationships)", "artifacts/graph_db/neo4j; wbs/task3_storage/"],
        ["2", "Ontology releases", "v0.1.0 to v1.6.0 (+ companion v3.0.0) with diffs and content hashes; RDF/OWL "
                                   "exports", "artifacts/ontologies/; wbs/deliverables/task2_formalization/"],
        ["3", "Cleaned Jeju tourism corpus", "46,059 records, splits and profile", "data/jeju_full; data/splits/"],
        ["4", "Jeju tourism benchmark", "347 categorized items over six task types", "task4_evaluation/"],
        ["5", "Evaluation and verifier code", "Metrics, paired tests, validation layer, surgical repair",
         "wbs/task4_evaluation/; wbs/task3_reasoning/"],
        ["6", "Reproducibility package", "Run order, requirements, manifest with sha256",
         "wbs/deliverables/REPRODUCIBILITY/"],
        ["7", "Demonstrators", "Interactive demo application, business-plan deck and narrated demo film",
         "Jeju_Tourism_Research_Demo/; wbs/deliverables/presentation/"],
    ], [0.9, 3.6, 7.2, 5.3], size=7.5)
    note(doc, "Categories not applicable in 2026: bio-resources/compounds, new-technology designation, "
              "certification, standardization, prototypes, technology transfer, commercialisation, sales, job "
              "creation, cost-reduction amounts, economic ripple effects, industrial support, technology trade, "
              "legislation, policy use, design standards, industrial training, use in other national R&D, "
              "international cooperation, promotion and awards.", 8)

    H.section_heading(doc, 5, "Unplanned Outcomes and Contributions to Related Fields",
                      "3-5. Unplanned Outcomes and Contributions to Related Fields")
    bullets(doc, [
        ("Cost-of-change evidence (requested in the 19 October business-plan comments)",
         ["The policy-edit experiment priced a knowledge change at 0 GPU-hours and showed that retrieval "
          "delivered the rule in 10/10 probes while the model stated it in 8/10, a failure mode now tracked "
          "for Year 2."]),
        ("A model-agnostic verifier",
         ["Planned as an internal validation step, the layer proved to transfer to Mistral, Qwen and DeepSeek "
          "models without adaptation, which makes it usable with third-party chatbots."]),
        ("Defects found by the project's own checks and disclosed",
         ["A shared training adapter biased the five-variant comparison; a span detector counted ordinary "
          "capitalised words as entities; the RDF exports lagged the Neo4j and file stores by one governed "
          "change. Each was measured, corrected or scheduled, and published numbers were not silently changed."]),
        ("Contribution to related fields",
         ["A reusable protocol for evaluating ontology-grounded LLMs: equal-budget tiers, held-out tests, "
          "strict and audit-grade groundedness, and paraphrased references to test the metric itself."]),
    ])
    caption_table(doc, "3-4", "Examples of evidence to attach by outcome type")
    data_table(doc, ["Outcome type", "Evidence to attach"], [
        ["Research paper", "Copy of the paper showing authors, abstract and acknowledgement (no duplicate "
                           "attachment across projects)"],
        ["Intellectual property", "Copy of the registration certificate or application (inventors, title, "
                                  "project source)"],
        ["Product development (prototype)", "Photographs and other evidence of prototype development"],
        ["Technology transfer", "Transfer or licensing contract, royalty receipt"],
        ["Commercialisation", "Product photographs; sales evidence (tax invoices, supply contracts)"],
        ["Product approval / clinical trial", "FDA / MFDS approval; IND approval"],
    ], [4.2, 12.8], size=8)
    H.page_break(doc)


# ====================================================================== chapter 4
def chapter4(doc):
    H.chapter_banner(doc, 4, "R&D Expenditure")
    H.section_heading(doc, 1, "R&D Expenditure Statement of the R&D Institution",
                      "4-1. R&D Expenditure Statement of the R&D Institution")
    expenditure_table(doc, "4-1", "Expenditure statement of Jeju National University (unit: KRW, %)")
    H.section_heading(doc, 2, "R&D Expenditure Statement of the R&D Project",
                      "4-2. R&D Expenditure Statement of the R&D Project")
    expenditure_table(doc, "4-2", "Expenditure statement of the R&D project (unit: KRW, %)")
    note(doc, "Prepared per R&D institution (lead and joint). All amounts are entered by the Industry-Academic "
              "Cooperation Foundation from the settled accounts; this draft leaves them blank on purpose. "
              "Commissioned R&D costs are entered by the lead institution only.", 9)
    H.page_break(doc)


def expenditure_table(doc, num, title):
    caption_table(doc, num, title)
    items = [("A. Direct costs", ["Cash", "In-kind", "Subtotal"]), ("1) Personnel", ["Cash", "In-kind"]),
             ("2) Student personnel", ["Cash"]), ("3) Research facilities and equipment", ["Cash", "In-kind"]),
             ("4) Research materials", ["Cash", "In-kind"]), ("5) Research activities", ["Cash", "In-kind"]),
             ("6) Research allowance", ["Cash"]), ("7) Security allowance", ["Cash"]),
             ("8) Commissioned R&D", ["Cash"]), ("9) International joint R&D", ["Cash"]),
             ("10) R&D burden charges", ["Cash"]), ("B. Indirect costs", ["Cash"]),
             ("Total", ["Cash", "In-kind", "Total"])]
    rows, merges = [], []
    r = 2
    for name, kinds in items:
        for k, kind in enumerate(kinds):
            rows.append([name if k == 0 else None, kind] + [""] * 9)
        if len(kinds) > 1:
            merges.append(((r, 0), (r + len(kinds) - 1, 0)))
        r += len(kinds)
    hdr = [["Item", None, "Prior-stage carry-over ①", "Original budget ②", "Revised budget ③",
            "Interest included ④", "Subtotal ⑤ (①+③+④)", "Used ⑥", "Balance (⑤-⑥)", "Use rate (⑥/⑤x100)",
            "Carry-over to next stage"],
           [None, None, None, None, None, None, None, None, None, None, None]]
    hdr_merges = [((0, 0), (1, 1))] + [((0, c), (1, c)) for c in range(2, 11)]
    bold = tuple(i for i, rw in enumerate(rows) if rw[0] in ("A. Direct costs", "B. Indirect costs", "Total"))
    data_table(doc, hdr, rows, [3.2, 1.2] + [1.4] * 9, size=7, merges=hdr_merges + merges, bold_rows=bold)


# ====================================================================== chapter 5
def chapter5(doc):
    H.chapter_banner(doc, 5, "R&D Plan for the Next Year")
    H.section_heading(doc, 1, "R&D Objectives and Content", "5-1. R&D Objectives and Content")
    form_table(doc, [
        ["Objective (2027)", "**Turn the Year-1 prototype into a Neo4j-centred, human-validated service: the "
                             "graph database becomes the single source of truth for storage, validation and "
                             "retrieval, and the verified answer pipeline is piloted with Jeju tourism "
                             "stakeholders.**"],
        ["Main content", [
            ("Neo4j as the single source of truth", [
                "In-database SHACL validation with the neosemantics (n10s) plugin; RDF/OWL exports regenerated "
                "from Neo4j on every approved change, closing the version drift found in Year 1",
                "Governance proposals applied as Cypher transactions with the audit trail stored alongside"]),
            ("Cypher-native GraphRAG retrieval", [
                "Move subgraph retrieval from the in-memory index to Cypher (intent-to-class routing, k-hop "
                "paths) and add Neo4j vector indexes for entity cards",
                "Detect unstated requirements: answers that omit a retrieved rule (Year-1 finding R22)"]),
            ("Human evaluation", [
                "Expert rating of ontology quality (115 items) and blinded hallucination annotation (150 spans) "
                "with at least two raters; agreement via Krippendorff's alpha and Cohen's kappa"]),
            ("Language and data", [
                "Korean-language extraction and bilingual labels; live connectors for events, weather and "
                "transport; targeted accessibility data"]),
            ("Deployment readiness", [
                "Authenticated RBAC (Year-1 attribution only, R14); regression suite for knowledge edits (R21); "
                "pilot and user study"])]],
    ], [3.0, 14.0], size=10)

    H.section_heading(doc, 2, "Changes in the Domestic and International Environment",
                      "5-2. Changes in the Domestic and International Environment")
    bullets(doc, [
        ("Technology", [
            "Graph-based RAG (GraphRAG) and ontology-grounded RAG have become active research and product "
            "directions; OG-RAG (EMNLP 2025) reports gains from ontology-grounded retrieval [19].",
            "Graph databases, including Neo4j, now provide vector indexes and GraphRAG tooling, which lowers the "
            "cost of combining symbolic queries with semantic search in one store.",
            "LLM-based ontology learning is maturing but still depends on validation and human review [16][17]."]),
        ("Regulation and policy", [
            "Korea's AI Framework Act took effect in January 2026, emphasising transparency and trust for AI "
            "services; the EU AI Act's obligations are being phased in. Verifiable, auditable answers are "
            "becoming a requirement rather than an option."]),
        ("Domain", [
            "Tourism services increasingly use conversational assistants; operators need control over what "
            "those assistants assert about safety, rules and access."]),
    ])

    H.section_heading(doc, 3, "R&D Strategy", "5-3. R&D Strategy")
    bullets(doc, [
        "Neo4j-first: one database for storage, validation, governance and retrieval, with RDF/OWL exports "
        "generated from it for interoperability.",
        "Keep the verifier model-agnostic so it can sit in front of any model a partner uses.",
        "Spend human effort where automation cannot answer: expert ratings and annotation, scheduled early.",
        "Keep the Year-1 measurement discipline: held-out tests, equal-budget comparisons reported separately, "
        "paired tests with correction, and published numbers never changed silently.",
        "Partner with Jeju tourism organisations for data access, domain review and the pilot.",
    ])

    H.section_heading(doc, 4, "R&D Schedule and Expected Outcomes", "5-4. R&D Schedule and Expected Outcomes")
    H.figure(doc, FIG / "fig18_plan_2027.png", "5-1", "Proposed Year-2 (2027) schedule",
             "Research team plan; to be confirmed with ETRI", 16.5)
    caption_table(doc, "5-1", "Expected outcomes for 2027 (proposed; targets to be agreed with ETRI)")
    data_table(doc, ["Outcome", "Proposed target", "Basis"], [
        ["International conference paper", "1", "Year-1 findings on groundedness, cost of change, verifier transfer"],
        ["Journal manuscript (submission)", "1", "Full evaluation incl. human ratings"],
        ["Software copyright registration", "1-2", "Ontology toolkit; Neo4j store and verifier"],
        ["Technology summary / report full text", "1 / 1", "Year-2 report"],
        ["Pilot deployment", "1", "Verified tourism Q&A service with a partner"],
        ["Measured targets", "Strict ungrounded rate below Simple RAG at every size; human-rater agreement "
                             "(alpha) >= 0.67", "Year-1 baselines (Table 2-12, Figure 2-12)"],
    ], [4.6, 5.4, 7.0], size=8)

    H.section_heading(doc, 5, "R&D Budget Plan", "5-5. R&D Budget Plan")
    box(doc, "Next year's R&D funding and contribution plan (unit: KRW thousand)")
    data_table(doc, [["Stage", "Year", "R&D institution", "Institution type", "ETRI R&D funds", "Institution "
                      "contribution", None, None, "Total", None, None],
                     [None, None, None, None, "Cash (A)", "Cash (B)", "In-kind (C)", "Subtotal (D)", "Cash",
                      "In-kind", "Sum (E)"]],
               [["1", "2", "Jeju National University", "University", "", "", "", "", "", "", ""],
                [None, None, "Subtotal", None, "", "", "", "", "", "", ""]],
               [1.0, 1.0, 3.2, 1.8] + [1.43] * 7, size=7,
               merges=[((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3)), ((0, 4), (0, 4)),
                       ((0, 5), (0, 7)), ((0, 8), (0, 10)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((3, 2), (3, 3))])
    box(doc, "Next year's R&D budget use plan")
    para(doc, "[ Joint R&D institution: Jeju National University ]  (unit: KRW thousand; the current year is shaded "
              "by the institution)", 10, align="l", after=3)
    budget_rows = [("Direct costs", "Personnel", "Internal personnel", "Cash"), (None, None, None, "In-kind"),
                   (None, None, "External personnel", "Cash"), (None, None, "Research-support staff", ""),
                   (None, None, "Subtotal", ""), (None, "Student personnel", "General", ""),
                   (None, None, "Integrated management", ""), (None, "Research facilities and equipment", "Cash", ""),
                   (None, None, "In-kind", ""), (None, None, "Integrated management (cash)", ""),
                   (None, "Research materials", "Cash", ""), (None, None, "In-kind", ""),
                   (None, "Research activities", "Cash", ""), (None, None, "In-kind", ""),
                   (None, "Research allowance", "", ""), (None, "Security allowance", "", ""),
                   (None, "Domestic commissioned R&D", "", ""), (None, "International joint R&D", "", ""),
                   (None, "Direct-cost subtotal", "", ""), ("Indirect costs", "", "", ""),
                   ("Total R&D budget", "", "", "")]
    rows = [list(b) + [""] * 7 for b in budget_rows]
    data_table(doc, [["Cost category", "Item", None, None, "Stage 1", None, None, "Stage 2", None, None, "Total"],
                     [None, None, None, None, "Year 1", "Year 2", "Year 3", "Year 1", "Year 2", "Year n", None]],
               rows, [1.8, 3.0, 2.8, 1.2] + [1.15] * 7, size=7,
               merges=[((0, 0), (1, 0)), ((0, 1), (1, 3)), ((0, 4), (0, 6)), ((0, 7), (0, 9)), ((0, 10), (1, 10)),
                       ((2, 0), (20, 0)), ((2, 1), (6, 1)), ((2, 2), (3, 2)), ((7, 1), (8, 1)), ((9, 1), (11, 1)),
                       ((12, 1), (13, 1)), ((14, 1), (15, 1))])
    note(doc, "Budget figures are entered by the institution from the agreement.", 8)

    H.section_heading(doc, 6, "Commercialisation Plan", "5-6. Commercialisation Plan")
    bullets(doc, [
        ("Verified tourism knowledge service", [
            "An API that answers visitor questions from the Neo4j ontology with evidence, offered to tourism "
            "agencies, platforms and local businesses."]),
        ("Ontology verifier as a plug-in", [
            "A model-agnostic post-processing component for existing chatbots, sold or licensed as software; "
            "Year-1 evidence shows transfer across six models without retraining."]),
        ("Knowledge-operations tooling", [
            "Governed editing (proposal, validation, approval, audit) for organisations that must correct AI "
            "answers quickly and prove who changed what."]),
        "Path: software copyright registration (2027), pilot with a partner, then technology transfer to a "
        "local SME. No revenue or transfer is claimed for 2026.",
    ])

    H.section_heading(doc, 7, "Utilisation of R&D Outcomes and Expected Effects",
                      "5-7. Utilisation of R&D Outcomes and Expected Effects")
    box(doc, "Utilisation")
    bullets(doc, [
        "ETRI Jeju Regional Research Center: a reference architecture and open evaluation protocol for "
        "trustworthy LLM services in the region.",
        "Tourism operators: controllable assistants whose safety, policy and access statements are traceable.",
        "Other high-trust domains: the same pipeline applies to public administration and health information.",
    ])
    box(doc, "Expected effects")
    bullets(doc, [
        "Technical: fewer fabricated entities and claims (strict ungrounded rate roughly halved in Year 1), "
        "millisecond knowledge queries, auditable knowledge change.",
        "Economic: knowledge corrections in seconds of CPU instead of GPU retraining; no added serving cost "
        "from the ontology path.",
        "Social: safer, more transparent visitor information aligned with national AI trust policy.",
    ])
    H.page_break(doc)


# ====================================================================== attachments
def attachments(doc):
    p = para(doc, "< Attachments >", 16, bold=True, align="c", after=8)
    H.tc_entry(p, TOC_ENTRIES[-1][1], 1)
    box(doc, "Attachment 1. Evidence for each WBS line item")
    data_table(doc, ["WBS item", "Evidence (path relative to /home/tahirahmad/Ontology_Project_JejuFull)"], [
        ["1.1.1", "docs/system_requirements.md; docs/implementation_roadmap.md; docs/experiment_audit_table.md; "
                  "docs/risk_management_plan.md"],
        ["1.1.2", "docs/PROJECT_IMPLEMENTATION_PLAN.md; wbs/deliverables/REPRODUCIBILITY/RUN_ORDER.md"],
        ["1.2.1", "data/jeju_full; data/reports/data_profile_report.md"],
        ["1.2.2", "docs/tourism_ontology_schema.md; docs/ONTOLOGY_SCHEMA_EXPLANATION.md"],
        ["2.1.1", "src/ontology/concept_extractor.py; artifacts/ontologies/1.4.0/concepts.json"],
        ["2.1.2", "src/llm/prompt_manager.py; artifacts/reports/prompt_comparison_llm.md; "
                  "wbs/deliverables/task2_extraction/prompt_sensitivity.md"],
        ["2.2.1", "wbs/deliverables/task2_formalization/jeju_tourism_ontology.ttl; normalization_report.json"],
        ["2.2.2", "wbs/task2_formalization/validate_ontology.py; task2_formalization/validation_report.md; "
                  "ontology_quality_metrics.json; rule_compliance.md"],
        ["3.1.1", "wbs/task3_storage/neo4j_store.py, neo4j_schema.py, neo4j_server.sh; "
                  "wbs/deliverables/task3_storage/graph_db_neo4j.md, ontology_schema_neo4j.md; "
                  "sparql_store.md; graph_db.md"],
        ["3.1.2", "wbs/deliverables/task3_governance/diff_logs.json, audit_trail.jsonl, access_control.md"],
        ["3.2.1", "experiments/exp024_rag_vs_ontology/contexts.py; wbs/deliverables/task3_retrieval/"
                  "retriever_characterization.md; onto_context_tuning.json"],
        ["3.2.2", "wbs/task3_reasoning/constrained_reasoner.py; task4_evaluation/"
                  "variant__llama3.1-8b__V5_ontology_rag_validated.jsonl"],
        ["4.1.1", "wbs/deliverables/task4_evaluation/tourism_benchmark_categorized.jsonl"],
        ["4.1.2", "task4_evaluation/baseline_comparison__*.md; ablation_offline__llama3.1-8b.md; HYPOTHESES.md; "
                  "cross_model_results.md"],
        ["4.2.1", "wbs/deliverables/task4_scalability/ontology_scalability.md; robustness.md"],
        ["4.2.2", "wbs/deliverables/figures/FIGURE_INDEX.md; FINAL_REPORT.md; REPRODUCIBILITY/MANIFEST.json"],
    ], [1.8, 15.2], size=7.5, align=["c", "l"])
    box(doc, "Attachment 2. Reproducing the Neo4j results")
    code_block(doc, [
        "# start the database (Bolt 7690, HTTP 7475; password from NEO4J_PASSWORD)",
        "bash wbs/task3_storage/neo4j_server.sh start",
        "# load ontology v1.4.0 (typed and generic mappings) and run the nine competency questions",
        "python wbs/task3_storage/neo4j_store.py",
        "# read the schema back from the live server",
        "python wbs/task3_storage/neo4j_schema.py",
        "# regenerate every figure in this report (queries Neo4j live)",
        "python wbs/deliverables/annual_report_2026/scripts/make_plots.py",
    ])
    H.spacer(doc, 4)
    box(doc, "Attachment 3. Open items carried into Year 2")
    data_table(doc, ["ID", "Item", "Reason open", "Year-2 action"], [
        ["R6", "Human adjudication of references and expert ratings", "Needs raters", "Two or more raters; alpha/kappa"],
        ["R12", "RAG baseline at matched training budget for all pairs", "GPU job", "Complete matched-budget runs"],
        ["R14", "Authentication of governance actors", "Attribution only in Year 1", "Authenticated RBAC"],
        ["R21", "Regression suite for knowledge edits", "Additive edits reword 4 of 6 control answers", "Edit regression set"],
        ["R22", "Detect unstated retrieved requirements", "Rule retrieved 10/10, stated 8/10", "Requirement-coverage check"],
        ["-", "RDF exports at v1.3.0-norm vs Neo4j v1.4.0", "Changes published RDF numbers", "Export from Neo4j (n10s)"],
    ], [1.2, 5.6, 4.6, 5.6], size=7.5)
    source(doc, "wbs/deliverables/GOAL_STATUS.md; FINAL_REPORT.md")
    box(doc, "Attachment 4. References cited")
    for ref in [
        "[10] Xu, Z., Jain, S., & Kankanhalli, M. (2024). Hallucination is inevitable: an innate limitation of "
        "large language models. arXiv:2401.11817.",
        "[13] Confalonieri, R., & Guizzardi, G. (2025). On the multiple roles of ontologies in explanations for "
        "neuro-symbolic AI. Neurosymbolic Artificial Intelligence, 1, NAI-240754.",
        "[16] Bakker, R. M., Di Scala, D. L., de Boer, M. H., & Raaijmakers, S. A. (2025). Ontology learning "
        "with LLMs: a benchmark study on axiom identification. arXiv:2512.05594.",
        "[17] Babaei Giglou, H., D'Souza, J., & Auer, S. (2023). LLMs4OL: large language models for ontology "
        "learning. International Semantic Web Conference, 408-427.",
        "[19] Sharma, K., Kumar, P., & Li, Y. (2025). OG-RAG: ontology-grounded retrieval-augmented generation "
        "for large language models. Proceedings of EMNLP 2025, 32950-32969.",
    ]:
        para(doc, ref, 9, align="l", left=0.0, hanging=0.6, after=2)
    note(doc, "Reference numbers follow the research proposal's bibliography.", 8)
    box(doc, "Attachment 5. Scope of registration and deposit of R&D outcomes (National R&D Innovation Act, "
             "Enforcement Decree Art. 33(4) and Annex 4)")
    data_table(doc, ["Type", "Target", "Scope of registration / deposit"], [
        ["Registration", "Papers", "Papers in journals or proceedings of domestic and international societies "
                                   "(including electronic full text)"],
        [None, "Patents", "Patents applied for or registered at home and abroad"],
        [None, "Report full text", "Full text of annual and final reports"],
        [None, "Facilities / equipment", "Facilities and equipment of KRW 30 million or more acquired through "
                                         "national R&D, or any that can be shared"],
        [None, "Technology summary", "Summary of the technology of outcomes whose annual or final report is complete"],
        [None, "Bio-information", "Genome, proteome and expression information and other bio-information"],
        [None, "Software", "Created software and the information needed to register it"],
        [None, "Standards", "National standards and official international standards (ISO, IEC, ITU and "
                            "recognised bodies)"],
        ["Deposit", "Biological resources", "Microbial, animal, plant and genomic resources"],
        [None, "Compounds", "Synthetic or natural organic compounds and related information"],
        [None, "New varieties", "Agricultural varieties applied for or registered at home and abroad"],
    ], [2.4, 3.4, 11.2], size=7.5, merges=[((1, 0), (8, 0)), ((9, 0), (11, 0))])
    H.page_break(doc)
    t = doc.add_table(rows=1, cols=1)
    H.borders(t, outer=8)
    H.set_widths(t, [H.TEXT_W])
    c = t.cell(0, 0)
    para(c, "[ Back cover ]", 10, align="l", after=10, color="606060")
    para(c, "NOTICE", 16, bold=True, align="c", after=12)
    for i, txt in enumerate([
        "This report is the annual report of an R&D project of the 'ETRI Jeju Regional Research Center Pilot "
        "Programme' and internal R&D programmes conducted by the Electronics and Telecommunications Research "
        "Institute (ETRI).",
        "When the content of this R&D is published externally, it must be stated that it is the result of an "
        "R&D programme conducted by ETRI.",
        "Content that must be kept confidential as national science and technology secrets must not be "
        "published or disclosed externally.",
    ], 1):
        para(c, f"{i}. {txt}", 12, align="j", left=0.0, hanging=0.6, after=10)


def build(pages=None):
    doc = H.new_document()
    H.header_footer(doc, "2026 Annual Report  |  Joint R&D Institution: Jeju National University")
    cover(doc)
    summary(doc)
    contents(doc, pages)
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4(doc)
    chapter5(doc)
    attachments(doc)
    H.update_fields_on_open(doc)
    doc.core_properties.title = "2026 Annual Report - " + PROJECT
    doc.core_properties.author = "Jeju National University NLP Lab"
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages")
    a = ap.parse_args()
    build(json.load(open(a.pages)) if a.pages else None)
