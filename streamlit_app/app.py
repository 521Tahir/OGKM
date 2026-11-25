"""
Ontology-Augmented LLM Benchmark — side-by-side Streamlit application.

Case A (no ontology) vs Case B (with ontology) across 4 vanilla LLMs, generated
in parallel across GPUs 5-8 (metrics on GPU 9). See application.txt.

Run:  bash streamlit_app/run_app.sh   (or)  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import os
# metrics (BERTScore/COMET) run in THIS process on the eval GPU; model workers
# spawn and pin to their own GPUs. Set before any torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app.config import (MODELS, MAX_NEW_TOKENS, ONTOLOGY_VERSION, TEMPORAL_VERSION,
                                   EXAMPLE_QUESTIONS, QUALITY_METRICS, METRIC_LABELS,
                                   COMPOSITE_WEIGHTS, LOGS)
from streamlit_app.ui.styles import CSS
from streamlit_app.ui.cards import render_card
from streamlit_app.ui import charts
from streamlit_app.evaluation import metrics as M

st.set_page_config(page_title="Ontology Augmented Models Comparison", layout="wide", page_icon="🧭")
st.markdown(CSS, unsafe_allow_html=True)

SYSTEM = ("You are a Jeju Island (South Korea) tourism assistant. Answer in 1-2 concise "
          "sentences, naming specific places, months, or seasons. Never mention that any "
          "context, facts, or ontology were provided.")
# Case A: vanilla model, no ontology.
CASE_A = "Question: {q}\nAnswer concisely in 1-2 sentences with specific names/months."
# Case B: SAME vanilla model, but grounded on the injected ontology facts.
CASE_B = ("Authoritative Jeju facts:\n{ctx}\n\nAnswer the question using ONLY the facts above. "
          "Use the exact names, places, months, and seasons from the facts. Be concise "
          "(1-2 sentences). Do not add unrelated information and do not mention these facts.\n\n"
          "Question: {q}\nAnswer:")
JUDGE = ("Grade how accurately the candidate answer matches the AUTHORITATIVE reference "
         "(ground truth) for a Jeju tourism question.\n"
         "Question: {q}\nReference (ground truth): {ref}\nCandidate: {ans}\n\n"
         "Give an integer score from 0 to 10 for how well the candidate matches the reference's "
         "key facts (months, seasons, place names): 10 = fully correct and complete, 7-9 = mostly "
         "correct, 4-6 = partially correct, 1-3 = mostly wrong, 0 = wrong or missing. "
         "Reply with ONLY the integer score. Score:")


@st.cache_resource(show_spinner="Loading 4 models across GPUs 5-8 (one-time, ~1-2 min)…")
def get_pool():
    from streamlit_app.utils.model_server import ModelPool
    return ModelPool(MODELS)


@st.cache_resource(show_spinner="Loading ontology + temporal layer…")
def get_ontology():
    from streamlit_app.ontology.retriever import OntologyContextProvider
    return OntologyContextProvider(ONTOLOGY_VERSION, TEMPORAL_VERSION)


def run_benchmark(question: str) -> dict:
    pool, onto = get_pool(), get_ontology()
    ev = onto.retrieve(question)
    ref = ev["reference"] or question

    # 1) Case A + Case B generation jobs for all models (parallel across GPUs)
    jobs = {}
    for m in MODELS:
        jobs[m["name"]] = [
            ("A", CASE_A.format(q=question), SYSTEM, MAX_NEW_TOKENS),
            ("B", CASE_B.format(ctx=ev["prompt_context"], q=question), SYSTEM, MAX_NEW_TOKENS),
        ]
    t0 = time.perf_counter()
    gen = pool.generate_all(jobs)
    wall_gen = round((time.perf_counter() - t0) * 1000, 1)

    # 2) LLM-judge — distribute the 8 answers across the 4 workers (parallel)
    judge_jobs = {m["name"]: [] for m in MODELS}
    judge_map = {}
    names = [m["name"] for m in MODELS]
    idx = 0
    for mname in names:
        for case in ("A", "B"):
            ans = gen[mname][case].get("answer", "")
            worker = names[idx % len(names)]; idx += 1
            label = f"{mname}|{case}"
            judge_jobs[worker].append((label, JUDGE.format(q=question, ref=ref[:400], ans=ans[:400]), "You are a strict grader.", 12))
            judge_map[label] = (mname, case)
    judged = pool.generate_all(judge_jobs)
    verdict = {}
    for worker, res in judged.items():
        for label, r in res.items():
            verdict[label] = M.parse_judge(r.get("answer", ""))

    # 3) text metrics (BERT-F1, METEOR, COMET) on the eval GPU + key-token
    key_tokens = M.derive_key_tokens(ref)
    cands, srcs, refs, order = [], [], [], []
    for mname in names:
        for case in ("A", "B"):
            cands.append(gen[mname][case].get("answer", "")); srcs.append(question); refs.append(ref)
            order.append((mname, case))
    bert = M.bert_f1_batch(cands, refs)
    comet = M.comet_batch(srcs, cands, refs)

    rows = []
    for (mname, case), bf, cm in zip(order, bert, comet):
        ans = gen[mname][case].get("answer", "")
        met = {"bertf1": bf, "comet": cm,
               "meteor": M.meteor_score_one(ans, ref),
               "keytok": M.key_token_score(ans, key_tokens),
               "judge": verdict.get(f"{mname}|{case}")}
        met["composite"] = M.composite(met, COMPOSITE_WEIGHTS)
        rows.append({"model": mname, "case": case, **met,
                     "ttft_ms": gen[mname][case].get("ttft_ms"),
                     "tpot_ms": gen[mname][case].get("tpot_ms"),
                     "answer": ans, "result": gen[mname][case]})
    return {"question": question, "evidence": ev, "reference": ref,
            "key_tokens": key_tokens, "rows": rows, "wall_gen_ms": wall_gen,
            "timestamp": datetime.now().isoformat()}


def log_run(data: dict) -> None:
    try:
        out = {k: v for k, v in data.items() if k != "rows"}
        out["rows"] = [{k: v for k, v in r.items() if k != "result"} for r in data["rows"]]
        (LOGS / f"run_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception:
        pass


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🧭 Ontology Augmented Models Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Side-by-side comparison of vanilla LLMs versus ontology-enhanced LLM reasoning.</div>',
            unsafe_allow_html=True)

if "question" not in st.session_state:
    st.session_state.question = ""

c_in, c_btn = st.columns([4, 1])
with c_in:
    q = st.text_input("Ask a tourism question...", value=st.session_state.question,
                      label_visibility="collapsed", placeholder="Ask a tourism question...")
with c_btn:
    gen_clicked = st.button("⚡ Generate", use_container_width=True, type="primary")

b1, b2, b3 = st.columns([1, 1, 4])
if b1.button("Clear", use_container_width=True):
    st.session_state.pop("last", None); st.session_state.question = ""; st.rerun()
if b2.button("Load Example Questions", use_container_width=True):
    st.session_state.question = EXAMPLE_QUESTIONS[0]; st.rerun()
with b3:
    pick = st.selectbox("Example questions", [""] + EXAMPLE_QUESTIONS, label_visibility="collapsed")
    if pick:
        q = pick

# model status
pool = get_pool()
ready = pool.ready_models()
st.caption(f"Models ready: {', '.join(ready) if ready else 'none'}  ·  "
           f"ontology v{ONTOLOGY_VERSION} + temporal v{TEMPORAL_VERSION}")

if gen_clicked and q.strip():
    st.session_state.question = q
    with st.spinner("Generating 8 answers in parallel across GPUs 5-8 + computing metrics…"):
        st.session_state.last = run_benchmark(q.strip())
        log_run(st.session_state.last)

data = st.session_state.get("last")
if data:
    st.markdown("---")
    st.markdown(f"### ❓ Question\n**{data['question']}**")
    if not data["evidence"].get("has_context"):
        st.warning("⚠️ This question isn't covered by the Jeju ontology, so Case B "
                   "receives no grounding and the quality metrics are not meaningful "
                   "(Case A ≈ Case B by design). Try an example question or ask about a "
                   "specific Jeju place, food, season, or event.")
    else:
        st.success(f"✅ Grounded gold reference (from ontology): **{data['reference']}**")
    by = {(r["model"], r["case"]): r for r in data["rows"]}

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="case-header case-a">❌ Case A — Without Ontology</div>', unsafe_allow_html=True)
        for m in MODELS:
            r = by.get((m["name"], "A"), {})
            render_card(m["name"], "A", r.get("result", {}), r)
    with right:
        st.markdown('<div class="case-header case-b">✅ Case B — With Ontology</div>', unsafe_allow_html=True)
        for m in MODELS:
            r = by.get((m["name"], "B"), {})
            render_card(m["name"], "B", r.get("result", {}), r, evidence=data["evidence"])

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Overall Comparison")
    df = pd.DataFrame([{
        "Model": r["model"], "Case": "A (no onto)" if r["case"] == "A" else "B (with onto)",
        **{METRIC_LABELS[k]: r.get(k) for k in QUALITY_METRICS},
    } for r in data["rows"]])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown("### 🏆 Leaderboard (composite score)")
    lb = pd.DataFrame([{
        "Model": r["model"], "Case": "A" if r["case"] == "A" else "B",
        "Composite": r.get("composite"),
    } for r in data["rows"]]).sort_values("Composite", ascending=False, na_position="last").reset_index(drop=True)
    lb.index = lb.index + 1
    lb.insert(0, "Rank", lb.index)
    st.dataframe(lb, use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Charts")
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(charts.quality_bar(data["rows"]), use_container_width=True)
    cc2.plotly_chart(charts.radar(data["rows"]), use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("### ⬇️ Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download Results CSV", df.to_csv(index=False).encode(),
                       file_name="benchmark_results.csv", mime="text/csv", use_container_width=True)
    full = {k: v for k, v in data.items() if k != "rows"}
    full["rows"] = [{k: v for k, v in r.items() if k != "result"} for r in data["rows"]]
    e2.download_button("Download JSON", json.dumps(full, indent=2, ensure_ascii=False).encode(),
                       file_name="benchmark_full.json", mime="application/json", use_container_width=True)
    report = (f"# Ontology-Augmented LLM Benchmark\n\n**Question:** {data['question']}\n\n"
              f"**Reference (ontology):** {data['reference']}\n\n## Results\n\n{df.to_markdown(index=False)}\n\n"
              f"## Leaderboard\n\n{lb.to_markdown(index=False)}\n")
    e3.download_button("Download Benchmark Report", report.encode(),
                       file_name="benchmark_report.md", mime="text/markdown", use_container_width=True)
else:
    st.info("Enter a tourism question and press **Generate** to compare all 4 models, Case A vs Case B.")
