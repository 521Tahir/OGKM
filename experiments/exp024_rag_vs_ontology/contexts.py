"""
Context builders for the two arms.

RagContextBuilder  — textbook dense RAG: bge-base-en-v1.5 embeddings over the
                     36,847 train-split answers, FAISS inner-product top-k.
OntoContextBuilder — the project's ontology + temporal layer, with a fact budget
                     matched to the RAG arm and informative predicates preferred
                     over bare `is_a` typing triples.

Both emit a plain-text block capped at the SAME character budget so neither arm
gets more prompt real estate than the other.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from . import config as C

sys.path.insert(0, str(C.ROOT))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _truncate(text: str, limit: int) -> str:
    """Cut to `limit` chars on a sentence boundary where possible."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit * 0.6 else cut).strip()


# ══════════════════════════════════════════════════════════════════════════
#  Arm 1 — simple dense RAG over the train split
# ══════════════════════════════════════════════════════════════════════════
class RagContextBuilder:
    """Dense retrieval over train-split (Input, Output) pairs."""

    def __init__(self, device: str = "cuda"):
        self.index_path = C.INDEX_DIR / "train_bge.faiss"
        self.meta_path = C.INDEX_DIR / "train_passages.jsonl"
        self.device = device
        self._model = None
        self._index = None
        self._passages: list[dict] = []

    # ── corpus ────────────────────────────────────────────────────────────
    @staticmethod
    def _passage_text(rec: dict) -> str:
        """One retrievable passage: the train question plus its reference answer."""
        return f"{rec['Input'].strip()}\n{rec['Output'].strip()}"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
            self._model.max_seq_length = 512
        return self._model

    def build(self, force: bool = False) -> None:
        """Embed the train split and persist a FAISS inner-product index."""
        import faiss

        if self.index_path.exists() and self.meta_path.exists() and not force:
            return

        recs = [json.loads(l) for l in open(C.TRAIN_JSONL, encoding="utf-8")]
        texts = [self._passage_text(r) for r in recs]
        model = self._load_model()
        emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
        emb = emb.astype("float32")

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                f.write(json.dumps({"pos": i, "record_id": r.get("record_id"),
                                    "Input": r["Input"], "Output": r["Output"]},
                                   ensure_ascii=False) + "\n")

    def load(self):
        import faiss

        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
            self._passages = [json.loads(l) for l in open(self.meta_path, encoding="utf-8")]
        return self

    # ── retrieval ─────────────────────────────────────────────────────────
    def retrieve_batch(self, questions: list[str], exclude_ids: list | None = None,
                       top_k: int = None) -> list[dict]:
        """Return one context dict per question. `exclude_ids[i]` drops that record
        from question i's own results (self-retrieval when querying with train)."""
        self.load()
        k = top_k or C.RAG_TOP_K
        model = self._load_model()
        q = [C.EMBED_QUERY_PREFIX + x for x in questions]
        qe = model.encode(q, batch_size=128, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False).astype("float32")
        # over-fetch so self/duplicate hits can be dropped without losing depth
        scores, idxs = self._index.search(qe, k + 5)

        out = []
        for i, (srow, irow) in enumerate(zip(scores, idxs)):
            excl = exclude_ids[i] if exclude_ids else None
            qn = _norm(questions[i])
            hits, seen = [], set()
            for s, j in zip(srow, irow):
                if j < 0:
                    continue
                p = self._passages[j]
                if excl is not None and p.get("record_id") == excl:
                    continue
                if _norm(p["Input"]) == qn:      # exact duplicate question -> leakage
                    continue
                key = _norm(p["Output"])[:200]
                if key in seen:
                    continue
                seen.add(key)
                hits.append({"score": float(s), "Input": p["Input"], "Output": p["Output"],
                             "record_id": p.get("record_id")})
                if len(hits) == k:
                    break
            out.append({"context": self._format(hits), "n_hits": len(hits),
                        "top_score": hits[0]["score"] if hits else 0.0,
                        "hit_ids": [h["record_id"] for h in hits]})
        return out

    @staticmethod
    def _format(hits: list[dict]) -> str:
        if not hits:
            return "[no passages retrieved]"
        lines = []
        for n, h in enumerate(hits, 1):
            lines.append(f"[{n}] {_truncate(h['Output'], C.RAG_PASSAGE_CHARS)}")
        return _truncate("\n".join(lines), C.CTX_MAX_CHARS)


# ══════════════════════════════════════════════════════════════════════════
#  Arm 2 — proposed ontology grounding
# ══════════════════════════════════════════════════════════════════════════
# Bare typing triples carry little answer-bearing content; relational predicates
# do. Weight the relevance scan accordingly instead of letting `is_a` flood the
# top-k (which is what the 5-fact demo provider does on long itinerary queries).
_TYPE_PREDS = {"is_a", "subclass_of", "rdf:type", "instance_of"}
_PRED_WEIGHT = {
    "is_a": 0.35, "subclass_of": 0.35, "rdf:type": 0.35, "instance_of": 0.35,
    "located_in": 1.30, "near": 1.20, "has_activity": 1.35, "best_for": 1.40,
    "available_in": 1.30, "served_at": 1.35, "reachable_by": 1.25, "part_of": 1.00,
    "suitable_for": 1.35, "requires": 1.20, "connected_to": 1.05, "offers": 1.30,
}


class OntoContextBuilder:
    """Ontology + temporal facts, budget-matched to the RAG arm."""

    def __init__(self, onto_version: str = None, temporal_version: str = None):
        from streamlit_app.ontology.retriever import OntologyContextProvider
        self.p = OntologyContextProvider(onto_version or C.ONTO_VERSION,
                                         temporal_version or C.TEMPORAL_VERSION)
        self.n_concepts, self.n_relations = self.p.n_concepts, self.p.n_relations

    def _scan(self, question: str, limit: int) -> list[dict]:
        """Predicate-weighted relevance scan over the full relation set."""
        from streamlit_app.ontology.retriever import _content_tokens
        toks = _content_tokens(question)
        if not toks:
            return []
        scored, seen = [], set()
        for r in self.p._relations:
            subj, obj = str(r.get("subject", "")), str(r.get("object", ""))
            sl, ol = subj.lower(), obj.lower()
            s_hits = sum(1 for t in toks if t in sl)
            o_hits = sum(1 for t in toks if t in ol)
            if not (s_hits or o_hits):
                continue
            pred = str(r.get("predicate", ""))
            key = (sl, pred, ol)
            if key in seen:
                continue
            seen.add(key)
            conf = float(r.get("provenance", {}).get("confidence", 0.5) or 0.5)
            score = (3 * s_hits + 2 * o_hits + 0.2 * conf) * _PRED_WEIGHT.get(pred, 1.0)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: limit * 3]]

    def retrieve(self, question: str) -> dict:
        from streamlit_app.ontology.retriever import _pred
        # temporal layer, intent-aware re-ranked (same path as the demo provider,
        # called directly so the expensive relation scan runs once, not twice)
        ranked = self.p._rank_temporal(question, self.p.temporal.retrieve(question, top_k=12).facts)
        tfacts = [{"subject": f.subject, "predicate": f.predicate, "object": f.object,
                   "when": f.temporal.phrase()} for f in ranked][:6]
        rels = self.p._clean_rels(question, self._scan(question, C.ONTO_MAX_FACTS))

        # keep the block informative: cap pure typing triples at a third of budget
        n_type_max = max(2, C.ONTO_MAX_FACTS // 3)
        kept, n_type = [], 0
        for r in rels:
            if str(r.get("predicate", "")) in _TYPE_PREDS:
                if n_type >= n_type_max:
                    continue
                n_type += 1
            kept.append(r)
            if len(kept) >= C.ONTO_MAX_FACTS:
                break

        lines = []
        if tfacts:
            lines.append("Time-scoped facts:")
            lines += [f"- {f['subject']} {_pred(f['predicate'])} {f['object']} ({f['when']})"
                      for f in tfacts]
        if kept:
            if lines:
                lines.append("")
            lines.append("Knowledge-graph facts:")
            lines += [f"- {r.get('subject','')} {_pred(r.get('predicate',''))} {r.get('object','')}"
                      for r in kept]
        ctx = _truncate("\n".join(lines), C.CTX_MAX_CHARS) if lines else "[no ontology facts found]"
        return {"context": ctx, "n_hits": len(kept) + len(tfacts),
                "n_temporal": len(tfacts), "n_kg": len(kept)}


# ══════════════════════════════════════════════════════════════════════════
def build_prompt(question: str, arm: str, ctx: str | None) -> str:
    """User-turn text. Identical across arms except for the context block, and
    for arms in C.GROUNDED_ARMS, which carry an explicit grounding instruction."""
    if arm in ("none", "base") or not ctx:
        return C.USER_NO_CTX.format(q=question.strip())
    tpl = (C.USER_WITH_CTX_GROUNDED if arm in getattr(C, "GROUNDED_ARMS", set())
           else C.USER_WITH_CTX)
    return tpl.format(header=C.CTX_HEADER[arm], ctx=ctx.strip(), q=question.strip())


# ══════════════════════════════════════════════════════════════════════════
#  Arm 2b — onto_v2: dense retrieval over verbalised triples
# ══════════════════════════════════════════════════════════════════════════
class OntoV2ContextBuilder:
    """Ontology grounding with RAG's retrieval machinery.

    v1 scored the ontology by substring-matching the question's content tokens
    against 9,208 relations, capped at 15 facts. On generic questions ("plan a
    7-day itinerary") those tokens are generic, so it surfaced generic triples
    while the gold answer named specific places — 52% of test items retrieved
    ZERO gold entities even though the ontology contained 67.7% of them.

    v2 verbalises every relation and temporal fact into a sentence, embeds them
    with the SAME encoder the RAG arm uses, and retrieves by cosine. The two arms
    then differ only in what is indexed (triples vs passages), not in how it is
    retrieved — which is the comparison we actually want to make.
    """

    def __init__(self, onto_version: str = None, temporal_version: str = None,
                 device: str = "cuda"):
        self.onto_version = onto_version or C.ONTO_VERSION
        self.temporal_version = temporal_version or C.TEMPORAL_VERSION
        self.device = device
        self.index_path = C.INDEX_DIR / f"onto_{self.onto_version}_bge.faiss"
        self.meta_path = C.INDEX_DIR / f"onto_{self.onto_version}_facts.jsonl"
        self._model = None
        self._index = None
        self._facts: list[dict] = []
        self._by_subject: dict[str, list[int]] = {}

    # ── verbalisation ─────────────────────────────────────────────────────
    @staticmethod
    def _verbalise(subject: str, predicate: str, obj: str) -> str:
        return f"{subject} {predicate.replace('_', ' ').strip()} {obj}".strip()

    def _units(self) -> list[dict]:
        """One retrievable unit per relation, plus the temporal layer."""
        import json as _json
        base = C.ROOT / "artifacts" / "ontologies" / self.onto_version
        rels = _json.loads((base / "relations.json").read_text(encoding="utf-8"))
        units, seen = [], set()
        for r in rels:
            s, pred, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
            if not s or not o or s.lower() == o.lower():
                continue
            text = self._verbalise(s, pred, o)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            units.append({"text": text, "subject": s, "predicate": pred,
                          "object": o, "kind": "kg"})

        # temporal facts carry the time scoping the KG triples lack
        try:
            from src.retrieval.temporal_retriever import TemporalRetriever
            tr = TemporalRetriever.from_version(self.temporal_version)
            for f in getattr(tr, "facts", []) or []:
                text = f.to_sentence()
                if not text or text.lower() in seen:
                    continue
                seen.add(text.lower())
                units.append({"text": text, "subject": f.subject, "predicate": f.predicate,
                              "object": f.object, "kind": "temporal"})
        except Exception as e:                       # temporal layer is optional
            print(f"[onto_v2] temporal layer unavailable ({e})", flush=True)
        return units

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
            self._model.max_seq_length = 128        # triples are short
        return self._model

    def build(self, force: bool = False) -> None:
        import faiss
        import json as _json

        if self.index_path.exists() and self.meta_path.exists() and not force:
            return
        units = self._units()
        emb = self._load_model().encode([u["text"] for u in units], batch_size=512,
                                        convert_to_numpy=True, normalize_embeddings=True,
                                        show_progress_bar=True).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for u in units:
                f.write(_json.dumps(u, ensure_ascii=False) + "\n")
        print(f"[onto_v2] indexed {len(units):,} verbalised facts", flush=True)

    def load(self):
        import faiss
        import json as _json
        from collections import defaultdict

        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
            self._facts = [_json.loads(l) for l in open(self.meta_path, encoding="utf-8")]
            by = defaultdict(list)
            for i, u in enumerate(self._facts):
                by[u["subject"].lower()].append(i)
            self._by_subject = by
        return self

    # ── retrieval ─────────────────────────────────────────────────────────
    def retrieve_batch(self, questions: list[str]) -> list[dict]:
        self.load()
        k = C.ONTO_V2_MAX_FACTS
        qe = self._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in questions],
                                       batch_size=128, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        scores, idxs = self._index.search(qe, k)

        out = []
        for srow, irow in zip(scores, idxs):
            picked, seen = [], set()
            for s, j in zip(srow, irow):
                if j < 0 or j in seen:
                    continue
                seen.add(j)
                picked.append(self._facts[j])

            # top up the remaining budget with sibling relations of the best
            # subjects — a matched entity usually has more to say than one triple
            if C.ONTO_V2_EXPAND:
                for f in list(picked):
                    if len(picked) >= k:
                        break
                    added = 0
                    for j in self._by_subject.get(f["subject"].lower(), []):
                        if j in seen or added >= C.ONTO_V2_EXPAND_PER_SUBJ:
                            continue
                        seen.add(j)
                        picked.append(self._facts[j])
                        added += 1
                        if len(picked) >= k:
                            break

            temporal = [f for f in picked if f["kind"] == "temporal"]
            kg = [f for f in picked if f["kind"] != "temporal"]
            lines = []
            if temporal:
                lines.append("Time-scoped facts:")
                lines += [f"- {f['text']}" for f in temporal]
            if kg:
                if lines:
                    lines.append("")
                lines.append("Knowledge-graph facts:")
                lines += [f"- {f['text']}" for f in kg]
            ctx = _truncate("\n".join(lines), C.CTX_MAX_CHARS) if lines else "[no ontology facts found]"
            out.append({"context": ctx, "n_hits": len(picked),
                        "n_temporal": len(temporal), "n_kg": len(kg),
                        "top_score": float(srow[0]) if len(srow) else 0.0})
        return out


# ══════════════════════════════════════════════════════════════════════════
#  Arm 2c — onto_v3: entity cards + intent-driven class retrieval
# ══════════════════════════════════════════════════════════════════════════
# Why v2 failed (measured, 4,589 test items): dense retrieval over atomic
# triples returns the generic hub node. For "plan a 3-day trip, major natural
# attractions, traditional food" it retrieves `Jeju Island best for Sightseeing`
# while the gold answer names Hallim Park and Cheonjiyeon Waterfall — both IN the
# ontology, neither reachable from the question, because a generic question
# underdetermines which specific entities the answer should name. RAG escapes
# this by retrieving whole answers that already carry a plausible entity set.
#
# v3 supplies that entity set from the ontology's own structure instead:
#   1. entity CARDS (all of an entity's relations as one passage-like unit),
#      which match a question far better than a 5-word triple, plus
#   2. INTENT -> CLASS routing: "natural attractions" pulls the top-degree
#      NaturalAttraction instances, "traditional food" pulls Cuisine, and so on.
_INTENT_CLASSES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("natural", "nature", "scenery", "scenic", "waterfall", "beach", "cave", "volcan",
      "sunrise", "sunset", "oreum", "crater", "coast", "forest", "outdoor"),
     ("NaturalAttraction", "Beach", "Landmark", "Trail")),
    (("cultur", "heritage", "museum", "folk", "temple", "history", "historic",
      "tradition", "shrine", "art"),
     ("CulturalAttraction", "Museum", "ReligiousSite", "Landmark")),
    (("food", "eat", "cuisine", "dish", "meal", "restaurant", "seafood", "dining",
      "breakfast", "lunch", "dinner", "snack", "taste", "culinary", "halal",
      "vegetarian", "vegan"),
     ("Cuisine", "Restaurant", "Market")),
    (("stay", "hotel", "accommodat", "resort", "guesthouse", "lodging", "hostel"),
     ("Accommodation",)),
    (("activit", "hike", "hiking", "dive", "diving", "snorkel", "ride", "riding",
      "walk", "trek", "surf", "kayak", "cycl", "experience"),
     ("Activity", "Trail")),
    (("transport", "get around", "getting around", "rental", "car", "bus", "taxi",
      "airport", "ferry", "drive"),
     ("TransportMode",)),
    (("emergenc", "hospital", "safety", "medical", "clinic", "police", "insurance"),
     ("EmergencyService",)),
    (("shop", "market", "souvenir", "buy"), ("Market",)),
]
# a bare "plan me N days" question names no domain at all — give it the mix an
# itinerary actually needs
_DEFAULT_CLASSES = ("NaturalAttraction", "CulturalAttraction", "Cuisine", "Activity",
                    "Beach", "Trail", "Museum")


class OntoV3ContextBuilder:
    """Entity-card retrieval with intent-driven class routing."""

    # A card keeps at most 8 facts per entity, taken in the order the relation
    # file asserts them. For a well-connected place that order is "near X" six
    # times over, so a constraint asserted later - the `avoid_during` weather
    # states v1.5.0 adds, say - never reaches the card at all. `safety_first`
    # sorts the constraint predicates to the front of the entity's own facts
    # before the cap applies. It is off by default: the published arms were run
    # without it and must keep their exact context.
    CONSTRAINT_PREDICATES = ("avoid_during", "not_allowed", "requires",
                             "closed_during", "has_risk", "affects",
                             "prohibits", "restricted_to")

    def __init__(self, onto_version: str = None, temporal_version: str = None,
                 device: str = "cuda", safety_first: bool = False):
        self.onto_version = onto_version or C.ONTO_VERSION
        self.temporal_version = temporal_version or C.TEMPORAL_VERSION
        self.device = device
        self.safety_first = safety_first
        sfx = "_safety" if safety_first else ""
        self.index_path = C.INDEX_DIR / f"ontocards_{self.onto_version}{sfx}_bge.faiss"
        self.meta_path = C.INDEX_DIR / f"ontocards_{self.onto_version}{sfx}.jsonl"
        self._model = None
        self._index = None
        self._cards: list[dict] = []
        self._by_class: dict[str, list[int]] = {}
        self._temporal: list = []

    # ── card construction ─────────────────────────────────────────────────
    def _build_cards(self) -> list[dict]:
        import json as _json
        from collections import defaultdict

        base = C.ROOT / "artifacts" / "ontologies" / self.onto_version
        rels = _json.loads((base / "relations.json").read_text(encoding="utf-8"))
        cons = _json.loads((base / "concepts.json").read_text(encoding="utf-8"))
        cls_of = {c.get("name"): c.get("concept_class") for c in cons if c.get("name")}

        out_edges, degree = defaultdict(list), defaultdict(int)
        for r in rels:
            s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
            if not s or not o or s.lower() == o.lower():
                continue
            out_edges[s].append((p, o))
            degree[s] += 1
            degree[o] += 1

        cards = []
        for name, edges in out_edges.items():
            if len(name) < 3:
                continue
            klass = cls_of.get(name) or "Concept"
            # de-dup predicates so one entity's card isn't 20 near-identical lines
            seen, facts = set(), []
            if self.safety_first:
                rank = {p: i for i, p in enumerate(self.CONSTRAINT_PREDICATES)}
                edges = sorted(edges, key=lambda e: rank.get(e[0], len(rank)))
            for p, o in edges:
                key = (p.lower(), o.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(f"{name} {p.replace('_', ' ').strip()} {o}")
                if len(facts) >= 8:
                    break
            cards.append({"name": name, "klass": klass, "degree": degree[name],
                          "facts": facts,
                          "text": f"{name} ({klass}). " + "; ".join(facts)})
        return cards

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
            self._model.max_seq_length = 256
        return self._model

    def build(self, force: bool = False) -> None:
        import faiss
        import json as _json

        if self.index_path.exists() and self.meta_path.exists() and not force:
            return
        cards = self._build_cards()
        emb = self._load_model().encode([c["text"] for c in cards], batch_size=256,
                                        convert_to_numpy=True, normalize_embeddings=True,
                                        show_progress_bar=True).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for c in cards:
                f.write(_json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[onto_v3] indexed {len(cards):,} entity cards", flush=True)

    def load(self):
        import faiss
        import json as _json
        from collections import defaultdict

        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
            self._cards = [_json.loads(l) for l in open(self.meta_path, encoding="utf-8")]
            by = defaultdict(list)
            for i, c in enumerate(self._cards):
                by[c["klass"]].append(i)
            for k in by:                       # most-connected entity of a class first
                by[k].sort(key=lambda i: -self._cards[i]["degree"])
            self._by_class = by
            try:
                from src.retrieval.temporal_retriever import TemporalRetriever
                self._temporal = TemporalRetriever.from_version(self.temporal_version)
            except Exception:
                self._temporal = None
        return self

    # ── retrieval ─────────────────────────────────────────────────────────
    @staticmethod
    def _target_classes(question: str) -> list[str]:
        q = question.lower()
        hit = []
        for keys, classes in _INTENT_CLASSES:
            if any(k in q for k in keys):
                for c in classes:
                    if c not in hit:
                        hit.append(c)
        return hit or list(_DEFAULT_CLASSES)

    def retrieve_batch(self, questions: list[str], n_entities: int = 16) -> list[dict]:
        self.load()
        qe = self._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in questions],
                                       batch_size=128, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        scores, idxs = self._index.search(qe, n_entities)

        out = []
        for qi, (srow, irow) in enumerate(zip(scores, idxs)):
            question = questions[qi]
            picked, seen = [], set()
            # (a) entities the question actually resembles
            for s, j in zip(srow, irow):
                if j < 0 or j in seen:
                    continue
                seen.add(j)
                picked.append(self._cards[j])

            # (b) intent -> class: the specifics a generic question implies but
            #     never names. This is what v1/v2 could not reach.
            classes = self._target_classes(question)
            per_class = max(2, n_entities // max(len(classes), 1))
            for klass in classes:
                added = 0
                for j in self._by_class.get(klass, []):
                    if j in seen:
                        continue
                    seen.add(j)
                    picked.append(self._cards[j])
                    added += 1
                    if added >= per_class:
                        break

            # (c) time-scoped facts, if the layer has anything for this question
            tlines = []
            if self._temporal is not None:
                try:
                    tf = self._temporal.retrieve(question, top_k=6).facts
                    tlines = [f.to_sentence() for f in tf][:4]
                except Exception:
                    tlines = []

            lines, n_facts = [], 0
            if tlines:
                lines.append("Time-scoped facts:")
                lines += [f"- {t}" for t in tlines]
                n_facts += len(tlines)
            if picked:
                if lines:
                    lines.append("")
                lines.append("Knowledge-graph facts:")
                for c in picked:
                    for fact in c["facts"][:3]:
                        lines.append(f"- {fact}")
                        n_facts += 1
            ctx = _truncate("\n".join(lines), C.CTX_MAX_CHARS) if lines else "[no ontology facts found]"
            out.append({"context": ctx, "n_hits": n_facts, "n_entities": len(picked),
                        "classes": classes,
                        "entities": [c["name"] for c in picked],
                        "top_score": float(srow[0]) if len(srow) else 0.0})
        return out
