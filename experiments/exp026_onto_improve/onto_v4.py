"""
Goal 3, lever 2 — onto_v4: entity-linked multi-hop graph traversal.

Why v1-v3 could not do multi-hop:
  v1  scores every triple independently against the question, so the SECOND hop
      is retrieved only if the bridge entity happens to share tokens with the
      question. It does not traverse the graph at all.
  v2  dense retrieval over atomic triples returns the generic hub node
      ("Jeju Island best for Sightseeing"); coverage did not move (0.091 vs 0.093).
  v3  entity cards + intent->class routing doubled coverage (0.141 -> 0.275 on
      real entities) but still retrieves each entity independently.

v4 treats the question as a graph query:
  1. LINK    find ontology entities named in the question (word-boundary match on
             names and aliases, longest-match-wins).
  2. EXPAND  breadth-first from those seeds along edges, depth 2, so a two-hop
             question's bridge AND endpoint both enter the candidate set.
  3. RANK    score each candidate edge by hop distance, predicate
             informativeness, and dense similarity to the question.
  4. ROUTE   for questions naming no entity, fall back to v3's intent->class
             routing so generic questions still get specifics.
  5. FILL    emit grouped by entity until the character budget is spent.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import (_DEFAULT_CLASSES, _INTENT_CLASSES,
                                                         _truncate)

PRED_W = {"is_a": 0.35, "subclass_of": 0.35, "rdf:type": 0.35, "instance_of": 0.35,
          "located_in": 1.30, "near": 1.25, "has_activity": 1.35, "best_for": 1.40,
          "available_in": 1.30, "served_at": 1.35, "reachable_by": 1.25,
          "part_of": 1.05, "suitable_for": 1.35, "requires": 1.20,
          "connected_to": 1.05, "offers": 1.30, "has_seasonality": 1.30}
STOP_ENT = {"jeju", "jeju island", "island", "korea", "south korea", "destination",
            "attraction", "activity", "cuisine", "trail", "region", "city", "day",
            "morning", "afternoon", "evening", "tourist", "traveler", "itinerary"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class OntoV4ContextBuilder:
    def __init__(self, onto_version: str = "1.3.0", temporal_version: str = None,
                 device: str = "cuda", max_facts: int = 45, depth: int = 2,
                 route_always: bool = False, route_per_class: int = 4):
        self.ver = onto_version
        self.temporal_version = temporal_version or C.TEMPORAL_VERSION
        self.device = device
        self.max_facts = max_facts
        self.depth = depth
        # v5 keeps class routing switched on even when traversal already filled the
        # budget: traversal gives precise connected facts, routing gives breadth.
        self.route_always = route_always
        self.route_per_class = route_per_class
        self.index_path = C.INDEX_DIR / f"ontov4_{onto_version}_edges.npy"
        self._model = None
        self._load_graph()

    # ── graph ─────────────────────────────────────────────────────────────
    def _load_graph(self) -> None:
        base = ROOT / "artifacts" / "ontologies" / self.ver
        rels = json.loads((base / "relations.json").read_text(encoding="utf-8"))
        cons = json.loads((base / "concepts.json").read_text(encoding="utf-8"))
        self.cls = {c["name"]: c.get("concept_class") for c in cons if c.get("name")}

        self.edges: list[tuple[str, str, str]] = []
        self.adj: dict[str, list[int]] = defaultdict(list)
        seen = set()
        for r in rels:
            s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
            if not s or not o or _norm(s) == _norm(o):
                continue
            k = (_norm(s), p, _norm(o))
            if k in seen:
                continue
            seen.add(k)
            i = len(self.edges)
            self.edges.append((s, p, o))
            self.adj[_norm(s)].append(i)
            self.adj[_norm(o)].append(i)          # undirected traversal
        self.edge_text = [f"{s} {p.replace('_',' ')} {o}" for s, p, o in self.edges]

        by_class = defaultdict(list)
        deg = defaultdict(int)
        for s, p, o in self.edges:
            deg[s] += 1; deg[o] += 1
        self.deg = deg
        # Hub nodes ("Jeju Island", "Destination") connect to everything, so seeding
        # or expanding through them floods the block with generic facts.
        self.hubs = {_norm(n) for n, d in deg.items() if d >= 120}
        # Class names are types, not instances: linking "Museum" as a seed is what
        # made a museum-proximity question retrieve "Museum located in Jeju".
        self.class_names = {_norm(k) for k in self.cls.values() if k}

        # entity surface forms for linking (longest first so "Jeju City" beats "Jeju")
        names = set()
        for c in cons:
            if c.get("name"):
                names.add(c["name"])
            for a in (c.get("aliases") or []):
                if isinstance(a, str) and 3 < len(a) < 60:
                    names.add(a)
        self.surface = sorted(
            {n for n in names
             if _norm(n) not in STOP_ENT and len(n) >= 4
             and _norm(n) not in self.class_names and _norm(n) not in self.hubs},
            key=len, reverse=True)
        self.surface_norm = [(_norm(n), n) for n in self.surface]
        for name, k in self.cls.items():
            if k:
                by_class[k].append(name)
        for k in by_class:
            by_class[k].sort(key=lambda nm: -deg[nm])
        self.by_class = by_class

        try:
            from src.retrieval.temporal_retriever import TemporalRetriever
            self.temporal = TemporalRetriever.from_version(self.temporal_version)
        except Exception:
            self.temporal = None

    # ── dense reranker over edges ─────────────────────────────────────────
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
            self._model.max_seq_length = 64
        return self._model

    def _cards(self) -> list[tuple[str, str]]:
        """(entity, card text) for every non-hub entity with outgoing edges."""
        facts = defaultdict(list)
        for s, p, o in self.edges:
            if _norm(s) in self.hubs:
                continue
            if len(facts[s]) < 8:
                facts[s].append(f"{p.replace('_', ' ')} {o}")
        return [(e, f"{e} ({self.cls.get(e) or 'Concept'}). " + "; ".join(f))
                for e, f in facts.items() if len(e) >= 4]

    def build(self, force: bool = False) -> None:
        card_path = self.index_path.with_name(self.index_path.stem + "_cards.npy")
        meta_path = self.index_path.with_name(self.index_path.stem + "_cards.json")
        if self.index_path.exists() and card_path.exists() and not force:
            self.emb = np.load(self.index_path)
            self.card_emb = np.load(card_path)
            self.card_names = json.loads(meta_path.read_text(encoding="utf-8"))
            return
        emb = self._load_model().encode(self.edge_text, batch_size=512,
                                        convert_to_numpy=True, normalize_embeddings=True,
                                        show_progress_bar=True).astype("float32")
        np.save(self.index_path, emb)
        self.emb = emb
        cards = self._cards()
        self.card_names = [c[0] for c in cards]
        self.card_emb = self._load_model().encode([c[1] for c in cards], batch_size=256,
                                                  convert_to_numpy=True,
                                                  normalize_embeddings=True,
                                                  show_progress_bar=True).astype("float32")
        np.save(card_path, self.card_emb)
        meta_path.write_text(json.dumps(self.card_names, ensure_ascii=False), encoding="utf-8")
        print(f"[onto_v4] embedded {len(self.edges):,} edges, {len(cards):,} entity cards",
              flush=True)

    def load(self):
        if not hasattr(self, "emb"):
            self.build()
        return self

    def dense_seeds(self, qvec: np.ndarray, k: int = 3, floor: float = 0.55) -> list[str]:
        """Resolve a DESCRIPTION to entities when literal linking finds nothing.
        “Jeju's UNESCO-listed lava tube” never matches a name, but it matches
        the Manjanggul card. This is what lets T3-style questions seed at all."""
        sim = self.card_emb @ qvec
        top = np.argpartition(-sim, min(k, len(sim) - 1))[:k]
        return [self.card_names[int(i)] for i in top[np.argsort(-sim[top])]
                if float(sim[int(i)]) >= floor]

    # ── 1. entity linking ─────────────────────────────────────────────────
    def link(self, question: str) -> list[str]:
        q = " " + _norm(question) + " "
        hits, covered = [], []
        for nname, orig in self.surface_norm:
            if len(nname) < 4:
                continue
            if re.search(rf"(?<![a-z]){re.escape(nname)}(?![a-z])", q):
                if any(nname in c for c in covered):      # subsumed by a longer match
                    continue
                covered.append(nname)
                hits.append(orig)
                if len(hits) >= 6:
                    break
        return hits

    # ── 2. traversal ──────────────────────────────────────────────────────
    def expand(self, seeds: list[str]) -> dict[int, int]:
        """edge index -> hop distance from the nearest seed."""
        found: dict[int, int] = {}
        visited = {_norm(s) for s in seeds}
        frontier = deque((_norm(s), 0) for s in seeds)
        while frontier:
            node, d = frontier.popleft()
            if d >= self.depth:
                continue
            for ei in self.adj.get(node, []):
                found.setdefault(ei, d + 1)
                s, p, o = self.edges[ei]
                nxt = _norm(o) if _norm(s) == node else _norm(s)
                if nxt in self.hubs:            # never expand through a hub node
                    continue
                if nxt not in visited and d + 1 < self.depth:
                    visited.add(nxt)
                    frontier.append((nxt, d + 1))
        return found

    # ── 3-5. rank, route, fill ────────────────────────────────────────────
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

    def retrieve_batch(self, questions: list[str]) -> list[dict]:
        self.load()
        qe = self._load_model().encode([C.EMBED_QUERY_PREFIX + q for q in questions],
                                       batch_size=128, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
        out = []
        for qi, question in enumerate(questions):
            seeds = self.link(question)
            n_literal = len(seeds)
            if len(seeds) < 2:
                for e in self.dense_seeds(qe[qi], k=3):
                    if _norm(e) not in {_norm(x) for x in seeds}:
                        seeds.append(e)
            cand = self.expand(seeds) if seeds else {}

            # questions naming no entity (generic itinerary asks) get v3-style
            # class routing so they still receive concrete specifics
            routed = []
            if self.route_always or len(cand) < self.max_facts:
                for klass in self._target_classes(question):
                    for name in self.by_class.get(klass, [])[:self.route_per_class]:
                        if _norm(name) in {_norm(s) for s in seeds}:
                            continue
                        routed.append(name)
                        for ei in self.adj.get(_norm(name), [])[:4]:
                            cand.setdefault(ei, 3)      # hop 3 = routed, lowest rank
                    if len(cand) >= self.max_facts * 3:
                        break

            if not cand:
                out.append({"context": "[no ontology facts found]", "n_hits": 0,
                            "n_seeds": 0, "seeds": [], "n_hop1": 0, "n_hop2": 0})
                continue

            idx = np.fromiter(cand.keys(), dtype=np.int64)
            sim = self.emb[idx] @ qe[qi]
            scored = []
            for j, ei in enumerate(idx):
                hop = cand[int(ei)]
                s, p, o = self.edges[int(ei)]
                hop_w = {1: 1.00, 2: 0.85, 3: 0.45}.get(hop, 0.4)
                hub_pen = 0.25 if (_norm(s) in self.hubs or _norm(o) in self.hubs) else 1.0
                scored.append((hop_w * PRED_W.get(p, 1.0) * hub_pen * (0.55 + float(sim[j])),
                               hop, int(ei)))
            scored.sort(key=lambda x: -x[0])
            # Reserve budget for the SECOND hop. Without this, hop-1 edges fill the
            # block and the retriever is multi-hop in name only.
            quota2 = int(self.max_facts * 0.40)
            hop2 = [x for x in scored if x[1] == 2][:quota2]
            rest = [x for x in scored if x not in hop2]
            picked = (hop2 + rest)[: self.max_facts]
            picked.sort(key=lambda x: (x[1], -x[0]))

            # group by subject so the block reads as connected knowledge, and keep
            # hop-1 groups first (they concern what the question actually named)
            groups: dict[str, list[str]] = defaultdict(list)
            order: list[str] = []
            for _, hop, ei in picked:
                s, p, o = self.edges[ei]
                if s not in groups:
                    order.append(s)
                groups[s].append(f"{s} {p.replace('_', ' ')} {o}")

            tlines = []
            if self.temporal is not None:
                try:
                    tf = self.temporal.retrieve(question, top_k=6).facts
                    tlines = [f.to_sentence() for f in tf][:4]
                except Exception:
                    tlines = []

            lines, n = [], 0
            if tlines:
                lines.append("Time-scoped facts:")
                lines += [f"- {t}" for t in tlines]
                n += len(tlines)
            if order:
                if lines:
                    lines.append("")
                lines.append("Knowledge-graph facts:")
                for subj in order:
                    for f in groups[subj]:
                        lines.append(f"- {f}")
                        n += 1
            ctx = _truncate("\n".join(lines), C.CTX_MAX_CHARS)
            hops = [h for _, h, _ in picked]
            out.append({"context": ctx, "n_hits": n, "n_seeds": len(seeds),
                        "n_literal_seeds": n_literal, "seeds": seeds,
                        "n_hop1": hops.count(1), "n_hop2": hops.count(2),
                        "n_routed": hops.count(3)})
        return out
