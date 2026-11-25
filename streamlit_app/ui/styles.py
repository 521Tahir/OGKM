"""Custom CSS so the app looks like a research demo, not default Streamlit."""

CSS = """
<style>
.block-container { padding-top: 3rem; max-width: 1700px; }
.app-title { font-size: 2.0rem; font-weight: 800; margin-bottom: 0.1rem;
            line-height: 1.35; padding: 0.15rem 0; display: inline-block; overflow: visible;
            background: linear-gradient(90deg,#2c3e50,#27ae60); -webkit-background-clip: text;
            background-clip: text; -webkit-text-fill-color: transparent; }
.app-sub { color:#5d6d7e; font-size:1.0rem; margin-bottom:0.8rem; }
.case-header { font-size:1.25rem; font-weight:800; padding:0.55rem 0.9rem; border-radius:12px;
              margin-bottom:0.7rem; text-align:center; }
.case-a { background:#fdecea; color:#b03a2e; border:1px solid #f1948a; }
.case-b { background:#eafaf1; color:#1e8449; border:1px solid #82e0aa; }
.model-card { border-radius:14px; padding:0.9rem 1.0rem; margin-bottom:0.9rem;
             box-shadow:0 2px 10px rgba(0,0,0,0.06); border:1px solid #e5e8ea; background:#ffffff; }
.model-card.a { border-left:5px solid #c0392b; }
.model-card.b { border-left:5px solid #27ae60; }
.model-name { font-weight:800; font-size:1.05rem; color:#2c3e50; }
.model-badge { display:inline-block; font-size:0.7rem; font-weight:700; padding:2px 8px;
              border-radius:20px; margin-left:8px; background:#eef2f3; color:#566573; vertical-align:middle; }
.answer-box { background:#f8f9f9; border-radius:10px; padding:0.6rem 0.8rem; margin:0.5rem 0;
             font-size:0.95rem; line-height:1.45; color:#212f3d; min-height:60px; }
.reason { color:#7f8c8d; font-size:0.82rem; font-style:italic; margin-top:0.2rem; }
.tok-line { color:#566573; font-size:0.8rem; margin-top:0.3rem; }
div[data-testid="stMetric"] { background:#f4f6f7; border-radius:8px; padding:6px 4px; }
div[data-testid="stMetricLabel"] p { font-size:0.72rem; }
.win-badge { color:#1e8449; font-weight:800; }
</style>
"""
