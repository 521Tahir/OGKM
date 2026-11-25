"""
LLM → Ontology → Neo4j  •  Knowledge Extraction / Editing / Graph Demo
=====================================================================

Three-stage interactive demo (mid-July presentation), on the FULL Jeju data:

  1. EXTRACT   — a vanilla LLM (Llama-3.1-8B, 4-bit) extracts concepts + relations
                 from a record drawn from the full 46,059-record Jeju dataset
                 (or from custom text) using the project's ConceptExtractor /
                 RelationExtractor (CoT prompting).
  2. EDIT      — the extracted knowledge is shown in editable tables; the user can
                 add / correct / delete concepts and relations before committing.
  3. NEO4J     — the (edited) knowledge is pushed into Neo4j (on top of the full
                 v1.1.0 ontology: 1,585 concepts / 9,208 relations) and rendered as
                 an interactive ontology graph, with a live link to Neo4j Browser.

Run:  bash streamlit_app/run_kg_demo.sh        # http://localhost:8502
"""
from __future__ import annotations

import os
# LLM extraction GPU — set before torch import. Metrics-free, one small 4-bit model.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("KG_DEMO_GPU", "7"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="LLM → Ontology → Neo4j", layout="wide", page_icon="🕸️")

DATASET = ROOT / "data" / "jeju_full" / "jeju_full_dataset.jsonl"
ONTO_VERSION = "1.1.0"
NEO4J_BROWSER = "http://localhost:7474"

CLASS_COLORS = {
    "Destination": "#2980b9", "Region": "#2980b9", "City": "#3498db", "Island": "#3498db",
    "Attraction": "#27ae60", "NaturalAttraction": "#16a085", "CulturalAttraction": "#8e44ad",
    "Beach": "#1abc9c", "Mountain": "#795548", "Trail": "#7f8c8d", "Museum": "#9b59b6",
    "Restaurant": "#e67e22", "Cuisine": "#d35400", "Accommodation": "#e84393",
    "TransportMode": "#34495e", "Activity": "#f39c12", "Season": "#c0392b",
    "Festival": "#e74c3c", "ReligiousSite": "#6c5ce7",
}


# ── cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Llama-3.1-8B (4-bit) for extraction…")
def get_extractors():
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager
    from src.ontology.concept_extractor import ConceptExtractor
    from src.ontology.relation_extractor import RelationExtractor
    key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "local-cache"
    llm = LLMClient(backend="local", model_key=key, load_in_4bit=True,
                    hf_token=token, max_tokens=800, temperature=0.0)
    pm = PromptManager(prompts_root=str(ROOT / "prompts"))
    return ConceptExtractor(llm, pm, strategy="cot"), RelationExtractor(llm, pm, strategy="cot")


@st.cache_resource(show_spinner="Connecting to Neo4j…")
def get_neo4j():
    from src.graph.neo4j_client import Neo4jClient
    return Neo4jClient.from_env()


@st.cache_data(show_spinner="Loading Jeju dataset (46K)…")
def load_records(n_sample: int = 400):
    recs = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    random.Random(7).shuffle(recs)
    return recs[:n_sample], len(recs)


def _color(cls: str) -> str:
    return CLASS_COLORS.get(cls, "#95a5a6")


def render_graph(concepts_df: pd.DataFrame, relations_df: pd.DataFrame,
                 neo4j_links: list[dict] | None = None, height: int = 560) -> str:
    """Build an interactive pyvis graph and return its HTML."""
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff",
                  font_color="#2c3e50", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-9000, central_gravity=0.3, spring_length=120)
    added = set()

    def add_node(name, cls, extracted=True):
        if not name or name in added:
            return
        added.add(name)
        net.add_node(name, label=name, title=f"{name}\nclass: {cls or '?'}",
                     color=_color(cls), size=26 if extracted else 16,
                     borderWidth=3 if extracted else 1,
                     shape="dot")

    for _, r in concepts_df.iterrows():
        add_node(str(r.get("name", "")).strip(), str(r.get("concept_class", "")).strip(), True)
    for _, r in relations_df.iterrows():
        s, p, o = (str(r.get(k, "")).strip() for k in ("subject", "predicate", "object"))
        if not (s and o):
            continue
        add_node(s, "", True); add_node(o, "", True)
        net.add_edge(s, o, label=p.replace("_", " "), title=p, color="#7f8c8d", arrows="to")
    # links into the existing full ontology (grey satellite nodes)
    for lk in (neo4j_links or []):
        s, p, o = lk.get("subject"), lk.get("predicate"), lk.get("object")
        if not (s and o):
            continue
        add_node(o if s in added else s, lk.get("object_class", ""), False)
        net.add_edge(s, o, label=str(p).replace("_", " "), color="#bdc3c7", arrows="to", dashes=True)
    net.set_options('{"physics":{"stabilization":{"iterations":150}},"interaction":{"hover":true}}')
    return net.generate_html(notebook=False)


# ── header ───────────────────────────────────────────────────────────────────
st.markdown("## 🕸️ LLM → Ontology → Neo4j  ·  Knowledge Extraction, Editing & Graph Demo")
st.caption(f"Vanilla Llama-3.1-8B (4-bit) · full Jeju ontology **v{ONTO_VERSION}** in Neo4j · "
           f"records from the full 46K dataset")

for k, v in {"concepts": None, "relations": None, "src_text": "", "pushed": False,
             "neo4j_links": []}.items():
    st.session_state.setdefault(k, v)

# Neo4j status line
try:
    neo = get_neo4j()
    ns = neo.stats()
    st.caption(f"🟢 Neo4j connected — **{ns.get('n_concepts', ns.get('nodes','?'))} nodes / "
               f"{ns.get('n_relations', ns.get('relationships','?'))} relations**  ·  "
               f"[open Neo4j Browser]({NEO4J_BROWSER})")
except Exception as e:
    neo = None
    st.warning(f"Neo4j not reachable: {str(e)[:120]}")

st.markdown("---")

# ── STAGE 1 — EXTRACT ────────────────────────────────────────────────────────
st.markdown("### 1️⃣ Extract knowledge from the LLM")
recs, total = load_records()
c1, c2 = st.columns([3, 2])
with c2:
    src = st.radio("Input source", ["Full Jeju dataset (46K)", "Custom text"], key="src_mode")
    if src.startswith("Full") and st.button("🎲 Load a random Jeju record", use_container_width=True):
        r = random.choice(recs)
        st.session_state.src_text = f"{r.get('Input','')}\n\n{r.get('Output','')}"[:1500]
        st.session_state.concepts = None; st.session_state.pushed = False
with c1:
    st.session_state.src_text = st.text_area(
        "Text to extract from", value=st.session_state.src_text, height=170,
        placeholder="Load a record from the full dataset, or paste any Jeju tourism text…")

if st.button("🔍 Extract Knowledge (LLM)", type="primary", disabled=not st.session_state.src_text.strip()):
    ce, re_ = get_extractors()
    with st.spinner("LLM extracting concepts and relations (CoT)…"):
        text = st.session_state.src_text.strip()
        concepts = ce.extract(text, record_id=-1)
        cdicts = [c.to_dict() for c in concepts]
        rels = re_.extract(text, cdicts, record_id=-1)
        rdicts = [r.to_dict() for r in rels]
    st.session_state.concepts = pd.DataFrame(
        [{"name": c.get("name", ""), "concept_class": c.get("concept_class", ""),
          "confidence": round(float(c.get("confidence", 0.8) or 0.8), 2)} for c in cdicts])
    st.session_state.relations = pd.DataFrame(
        [{"subject": r.get("subject", ""), "predicate": r.get("predicate", ""),
          "object": r.get("object", ""),
          "confidence": round(float(r.get("confidence", 0.8) or 0.8), 2)} for r in rdicts])
    st.session_state.pushed = False; st.session_state.neo4j_links = []
    st.success(f"Extracted **{len(cdicts)} concepts** and **{len(rdicts)} relations** from the LLM.")

# ── STAGE 2 — EDIT ───────────────────────────────────────────────────────────
if st.session_state.concepts is not None:
    st.markdown("---")
    st.markdown("### 2️⃣ Edit the extracted knowledge")
    st.caption("Add rows, correct names/classes/predicates, or delete wrong extractions before committing.")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Concepts**")
        st.session_state.concepts = st.data_editor(
            st.session_state.concepts, num_rows="dynamic", use_container_width=True,
            key="concept_editor",
            column_config={"confidence": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05)})
    with e2:
        st.markdown("**Relations**")
        st.session_state.relations = st.data_editor(
            st.session_state.relations, num_rows="dynamic", use_container_width=True,
            key="relation_editor",
            column_config={"confidence": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05)})

    # ── STAGE 3 — NEO4J ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3️⃣ Push to Neo4j & visualise the ontology graph")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("⬆️ Push to Neo4j", type="primary", disabled=neo is None):
        cdf, rdf = st.session_state.concepts.fillna(""), st.session_state.relations.fillna("")
        prov = {"source": "llm_demo", "confidence": 0.8,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extraction_prompt_version": "kg-demo/cot"}
        cpayload = [{"name": r["name"], "concept_class": r["concept_class"],
                     "confidence": float(r["confidence"] or 0.8),
                     "provenance": prov} for _, r in cdf.iterrows() if str(r["name"]).strip()]
        rpayload = [{"subject": r["subject"], "predicate": r["predicate"], "object": r["object"],
                     "confidence": float(r["confidence"] or 0.8), "provenance": prov}
                    for _, r in rdf.iterrows() if str(r["subject"]).strip() and str(r["object"]).strip()]
        nc = neo.bulk_load_concepts(cpayload)
        nr = neo.bulk_load_relations(rpayload)
        # find how the new concepts connect into the existing full ontology
        links = []
        for c in cpayload[:12]:
            try:
                for nb in neo.neighbours(c["name"], limit=4):
                    links.append(nb)
            except Exception:
                pass
        st.session_state.neo4j_links = links
        st.session_state.pushed = True
        st.success(f"Pushed **{nc} concepts** and **{nr} relations** into Neo4j "
                   f"(merged into the full v{ONTO_VERSION} graph). Found {len(links)} links to existing knowledge.")
    if b2.button("🔗 Connect to full ontology", disabled=neo is None):
        links = []
        for _, r in st.session_state.concepts.fillna("").iterrows():
            nm = str(r["name"]).strip()
            if nm:
                try:
                    links += neo.neighbours(nm, limit=4)
                except Exception:
                    pass
        st.session_state.neo4j_links = links
        st.info(f"Found {len(links)} connections from the extracted concepts into the full ontology.")

    # graph render
    html = render_graph(st.session_state.concepts.fillna(""),
                        st.session_state.relations.fillna(""),
                        st.session_state.neo4j_links)
    components.html(html, height=580, scrolling=False)
    st.caption("● bordered = newly extracted · ○ grey dashed = existing full-ontology neighbours")

    if st.session_state.pushed:
        names = [n for n in st.session_state.concepts["name"].dropna().tolist() if str(n).strip()][:8]
        namelist = ", ".join(f'"{n}"' for n in names)
        cypher = (f"MATCH (c:Concept)-[r]-(m)\nWHERE c.name IN [{namelist}]\nRETURN c, r, m LIMIT 100")
        st.markdown(f"**View this in Neo4j Browser** ([open]({NEO4J_BROWSER})) with:")
        st.code(cypher, language="cypher")
else:
    st.info("Load a record from the full dataset (or paste text) and press **Extract Knowledge**.")
