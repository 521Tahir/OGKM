"""Plotly charts: Case A vs B bar, latency, and radar."""
from __future__ import annotations

import plotly.graph_objects as go

from ..config import QUALITY_METRICS, METRIC_LABELS

A_COLOR, B_COLOR = "#c0392b", "#27ae60"


def _mean(rows, case, key):
    vals = [r[key] for r in rows if r["case"] == case and isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def quality_bar(rows) -> go.Figure:
    labels = [METRIC_LABELS[k] for k in QUALITY_METRICS]
    a = [_mean(rows, "A", k) for k in QUALITY_METRICS]
    b = [_mean(rows, "B", k) for k in QUALITY_METRICS]
    fig = go.Figure()
    fig.add_bar(x=labels, y=a, name="Case A — no ontology", marker_color=A_COLOR,
                text=[f"{v:.2f}" for v in a], textposition="outside")
    fig.add_bar(x=labels, y=b, name="Case B — with ontology", marker_color=B_COLOR,
                text=[f"{v:.2f}" for v in b], textposition="outside")
    fig.update_layout(barmode="group", title="Answer quality — Case A vs Case B (avg over models)",
                      yaxis=dict(range=[0, 1.1], title="score"), height=380,
                      legend=dict(orientation="h", y=1.12))
    return fig


def latency_bar(rows) -> go.Figure:
    fig = go.Figure()
    for case, color in (("A", A_COLOR), ("B", B_COLOR)):
        fig.add_bar(x=["TTFT (ms)", "TPOT (ms)"],
                    y=[_mean(rows, case, "ttft_ms"), _mean(rows, case, "tpot_ms")],
                    name=f"Case {case}", marker_color=color)
    fig.update_layout(barmode="group", title="Latency — TTFT / TPOT (avg over models)",
                      yaxis=dict(title="ms"), height=380, legend=dict(orientation="h", y=1.12))
    return fig


def radar(rows) -> go.Figure:
    cats = [METRIC_LABELS[k] for k in QUALITY_METRICS]
    fig = go.Figure()
    for case, color in (("A", A_COLOR), ("B", B_COLOR)):
        vals = [_mean(rows, case, k) for k in QUALITY_METRICS]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                      name=f"Case {case}", line_color=color))
    fig.update_layout(title="Quality radar — Case A vs Case B",
                      polar=dict(radialaxis=dict(range=[0, 1], visible=True)), height=420,
                      legend=dict(orientation="h", y=1.12))
    return fig
