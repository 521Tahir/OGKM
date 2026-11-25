"""
WBS Task 3.2.1 - Parameterised ontology context assembly.

One class, one `build` method, and a named policy table. `improve_onto_context`
sweeps the policies offline against gold-entity coverage; `generate_tuned_onto`
runs the shortlisted ones through real generation. Both import this module, so
the context a policy name produces is identical in the two measurements.

Every policy walks the ontology only - cards, classes, graph degree, out-edges.
None of them reads the corpus, the train split or the gold answer, so the arm
stays ontology-only and remains comparable with the published V3 numbers.

  n_entities    cards contributed by dense card search
  per_class     cards contributed per routed class (0 = n_entities // len(classes))
  facts_per     facts emitted per card
  round_robin   emit fact 1 of every card before fact 2 of any, so the final
                truncation thins all cards instead of deleting the tail cards
  routing_first routed cards before dense ones in emission order
  routing       'degree' (most-connected first) or 'similarity' (closest to query)
  hop2          follow located_in / near / part_of / connected_to out of the first
                N dense cards
  hub_demote    drop the top-0.1% degree mega-hubs, whose facts are generic
  global_degree question-INDEPENDENT control: the N most-connected cards overall
  val_drop      drop cards whose concept carries any validation finding
  val_rank      order routed cards by validation findings ascending, then degree
                descending, which is proposal 9.4's "validation status" ranking
  group         emit one line per card - "Name (Class): pred obj; pred obj" -
                instead of one line per triple, so the subject name is written
                once rather than once per fact. Same facts, fewer characters,
                and the line reads as a statement about an entity rather than
                as a bullet list of fragments.

Validation status comes from `task2_formalization/element_validation_status.json`,
which is generated once by `element_validation_status.py`. It is NOT free: 23.4% of
cards carry a finding overall but 76.7% of the 120 highest-degree cards do, so
degree-ordered routing systematically prefers the most-flagged entities and
ranking on validation status pulls against prominence. That trade is the point of
measuring it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HOP_PREDS = ("located in", "near", "part of", "connected to", "reachable by")

POLICIES: dict[str, dict] = {
    # the published V3 arm, reproduced here for a like-for-like comparison
    "baseline": dict(n_entities=16, facts_per=3),
    # the opposite direction from every other policy here: the same context,
    # truncated harder. If the published budget is already past the point where
    # more context pays, these should not cost much.
    "baseline_b2000": dict(n_entities=16, facts_per=3, budget=2000),
    "baseline_b1500": dict(n_entities=16, facts_per=3, budget=1500),
    "baseline_b1000": dict(n_entities=16, facts_per=3, budget=1000),
    "baseline_b500": dict(n_entities=16, facts_per=3, budget=500),
    "round_robin": dict(n_entities=16, facts_per=3, round_robin=True),
    "rr_hop2": dict(n_entities=32, facts_per=1, round_robin=True, hop2=8),
    "routing_deep": dict(n_entities=16, facts_per=2, round_robin=True,
                         routing_first=True, per_class=12),
    "routing_deep_hop2": dict(n_entities=16, facts_per=2, round_robin=True,
                              routing_first=True, per_class=12, hop2=6),
    "routing_deep_hop2_rr3": dict(n_entities=16, facts_per=3, round_robin=True,
                                  routing_first=True, per_class=12, hop2=6),
    # proposal 9.4: rank by validation status, not only by proximity and prominence
    "routing_deep_hop2_valrank": dict(n_entities=16, facts_per=2, round_robin=True,
                                      routing_first=True, per_class=12, hop2=6,
                                      val_rank=True),
    "routing_deep_hop2_clean": dict(n_entities=16, facts_per=2, round_robin=True,
                                    routing_first=True, per_class=12, hop2=6,
                                    val_drop=True),
    # encoding, not selection: identical card set to `baseline`, one line per
    # card instead of one line per triple
    "grouped": dict(n_entities=16, facts_per=3, group=True),
    "grouped_wide": dict(n_entities=16, facts_per=6, group=True),
    "grouped_deep": dict(n_entities=16, facts_per=3, round_robin=False,
                         routing_first=True, per_class=12, hop2=6, group=True),
    # control: no question-dependent retrieval at all
    "global_degree40": dict(global_degree=40, facts_per=2, round_robin=True,
                            routing_first=True),
    "global_degree80": dict(global_degree=80, facts_per=1, round_robin=True,
                            routing_first=True),
}


class TunedOntoContext:
    """OntoV3 cards with a selectable selection policy."""

    DEEP = 64          # dense search depth, enough for every policy

    # How many hop targets to follow out of each hopped-from card. 2 is what the
    # published arms ran, so it stays the default; `neo4j_context` raises it,
    # because a card only exposes the targets that survived its 8-fact cap and
    # the graph exposes all of them.
    hop_fanout = 2

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._onto = None
        self._ready = False

    def load(self):
        if self._ready:
            return self
        from experiments.exp024_rag_vs_ontology.contexts import OntoV3ContextBuilder
        self._onto = OntoV3ContextBuilder(device=self.device)
        self._onto.load()
        self.cards = self._onto._cards
        import numpy as np
        self._np = np
        self.deg = np.asarray([c["degree"] for c in self.cards])
        self.hub_cut = float(np.quantile(self.deg, 0.999))
        self.cls_idx = {k: np.asarray(v, dtype="int64")
                        for k, v in self._onto._by_class.items()}
        self.by_degree = np.argsort(-self.deg)
        # per-element validation verdicts, for proposal 9.4's validation-status rank
        vp = (ROOT / "wbs" / "deliverables" / "task2_formalization"
              / "element_validation_status.json")
        self.val_findings = [0] * len(self.cards)
        if vp.exists():
            import re as _re
            st_ = json.loads(vp.read_text(encoding="utf-8"))["status"]
            norm = lambda x: _re.sub(r"\s+", " ", str(x or "").strip()).lower()
            for i, c in enumerate(self.cards):
                self.val_findings[i] = (st_.get(norm(c["name"])) or {}).get("findings", 0)
        by_name = {c["name"].lower(): i for i, c in enumerate(self.cards)}
        self.hop_targets = []
        for c in self.cards:
            tgt = []
            for f in c["facts"]:
                low = f.lower()
                for p in HOP_PREDS:
                    k = low.find(f" {p} ")
                    if k >= 0:
                        j = by_name.get(f[k + len(p) + 2:].strip().lower())
                        if j is not None:
                            tgt.append(j)
                        break
            self.hop_targets.append(tgt)
        self._ready = True
        return self

    # ── per-question precomputation, shared by every policy ───────────────
    def prepare(self, questions: list[str]):
        from experiments.exp024_rag_vs_ontology import config as C
        self.load()
        self.questions = questions
        self.qe = self._onto._load_model().encode(
            [C.EMBED_QUERY_PREFIX + q for q in questions], batch_size=128,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False).astype("float32")
        _, self.didx = self._onto._index.search(self.qe, self.DEEP)
        self.cvec = self._onto._index.reconstruct_n(0, len(self.cards))
        self.temporal_lines = []
        for q in questions:
            t = []
            if self._onto._temporal is not None:
                try:
                    t = [f.to_sentence()
                         for f in self._onto._temporal.retrieve(q, top_k=6).facts][:4]
                except Exception:
                    t = []
            self.temporal_lines.append(t)
        return self

    # ── assembly ──────────────────────────────────────────────────────────
    def build(self, qi: int, n_entities=16, facts_per=3, round_robin=False,
              routing="degree", hop2=0, hub_demote=False, temporal=True,
              routing_first=False, per_class=0, global_degree=0,
              val_drop=False, val_rank=False, group=False,
              budget=0) -> dict:
        from experiments.exp024_rag_vs_ontology import config as C
        from experiments.exp024_rag_vs_ontology.contexts import _truncate
        np = self._np
        picked, seen = [], set()

        def take(j) -> None:
            j = int(j)
            if j in seen or (hub_demote and self.deg[j] >= self.hub_cut):
                return
            if val_drop and self.val_findings[j]:
                return
            seen.add(j); picked.append(j)

        def dense() -> None:
            for j in self.didx[qi][:n_entities]:
                if j >= 0:
                    take(j)
            if hop2:
                for j in list(picked)[:hop2]:
                    for t in self.hop_targets[j][:self.hop_fanout]:
                        take(t)

        def routed() -> None:
            if global_degree:
                for j in self.by_degree[:global_degree]:
                    take(j)
                return
            classes = self._onto._target_classes(self.questions[qi])
            per = per_class or max(2, n_entities // max(len(classes), 1))
            for klass in classes:
                pool = self.cls_idx.get(klass)
                if pool is None or not len(pool):
                    continue
                if routing == "similarity":
                    pool = pool[np.argsort(-(self.cvec[pool] @ self.qe[qi]))]
                elif val_rank:
                    # fewest validation findings first, prominence breaking ties
                    vf = np.asarray([self.val_findings[int(j)] for j in pool])
                    pool = pool[np.lexsort((-self.deg[pool], vf))]
                added = 0
                for j in pool:
                    if int(j) in seen:
                        continue
                    before = len(picked)
                    take(j)
                    added += len(picked) - before
                    if added >= per:
                        break

        if routing_first:
            routed(); dense()
        else:
            dense(); routed()

        lines, n_facts = [], 0
        if temporal and self.temporal_lines[qi]:
            lines.append("Time-scoped facts:")
            lines += [f"- {t}" for t in self.temporal_lines[qi]]
            n_facts += len(self.temporal_lines[qi])
        if picked:
            if lines:
                lines.append("")
            lines.append("Knowledge-graph facts:")
            if group:
                # one line per entity: the subject name is written once, which
                # is where the saving comes from
                for j in picked:
                    c = self.cards[j]
                    nm, fs = c["name"], c["facts"][:facts_per]
                    rest = [f[len(nm) + 1:] if f.startswith(nm + " ") else f
                            for f in fs]
                    if not rest:
                        continue
                    lines.append(f"- {nm} ({c['klass']}): " + "; ".join(rest))
                    n_facts += len(rest)
            elif round_robin:
                for depth in range(facts_per):
                    for j in picked:
                        fs = self.cards[j]["facts"]
                        if depth < len(fs):
                            lines.append(f"- {fs[depth]}")
                            n_facts += 1
            else:
                for j in picked:
                    for f in self.cards[j]["facts"][:facts_per]:
                        lines.append(f"- {f}")
                        n_facts += 1
        # every policy above spends the shared budget differently; `budget`
        # instead asks what a SMALLER budget costs, which is the one direction
        # the sweep never tested
        cap = budget or C.CTX_MAX_CHARS
        ctx = (_truncate("\n".join(lines), cap) if lines
               else "[no ontology facts found]")
        return {"context": ctx, "n_entities": len(picked), "n_facts": n_facts,
                "budget_chars": cap,
                "entities": [self.cards[j]["name"] for j in picked]}

    def build_all(self, policy: str) -> list[dict]:
        if policy not in POLICIES:
            raise KeyError(f"unknown policy {policy!r}; have {sorted(POLICIES)}")
        return [self.build(i, **POLICIES[policy]) for i in range(len(self.questions))]
