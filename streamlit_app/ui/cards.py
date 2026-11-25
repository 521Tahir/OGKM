"""Model card rendering (answer + metrics + latency + tokens + evidence)."""
from __future__ import annotations

import streamlit as st

from ..config import METRIC_LABELS


def _fmt(v, latency=False):
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.0f}" if latency else f"{v:.3f}"


def render_card(model_name: str, case: str, result: dict, metrics: dict,
                evidence: dict | None = None) -> None:
    cls = "a" if case == "A" else "b"
    st.markdown(f'<div class="model-card {cls}">', unsafe_allow_html=True)
    st.markdown(
        f'<span class="model-name">{model_name}</span>'
        f'<span class="model-badge">{"Case A · vanilla" if case=="A" else "Case B · ontology"}</span>',
        unsafe_allow_html=True)

    ans = result.get("answer", "—")
    st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)

    # quality metrics only (no latency/token lines)
    cols = st.columns(5)
    for col, key in zip(cols, ["bertf1", "meteor", "comet", "keytok", "judge"]):
        col.metric(METRIC_LABELS[key], _fmt(metrics.get(key)))

    if evidence and evidence.get("has_context"):
        with st.expander("🔎 Ontology evidence", expanded=False):
            if evidence.get("ontology_facts"):
                st.markdown("**Ontology Facts**")
                for f in evidence["ontology_facts"]:
                    st.markdown(f"- `{f['subject']}` **{f['predicate']}** `{f['object']}`  ({f['when']})")
            if evidence.get("kg_paths"):
                st.markdown("**Knowledge-Graph Paths**")
                for p in evidence["kg_paths"][:6]:
                    st.markdown(f"- {p}")
            if evidence.get("temporal_evidence"):
                st.markdown("**Temporal Evidence**")
                st.markdown(f"> {evidence['temporal_evidence']}")
            if evidence.get("retrieved_context"):
                st.markdown("**Retrieved Context**")
                st.code(evidence["retrieved_context"][:900])
    st.markdown("</div>", unsafe_allow_html=True)
