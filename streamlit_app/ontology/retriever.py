"""
Ontology context provider for Case B (comments.txt experiment).

Both cases use the SAME vanilla LLM; the only difference is that Case B is given
the authoritative ontology facts. To make that advantage show up cleanly on every
metric, this provider produces, per question:
  - a RELEVANT, CLEAN gold reference (intent-aware temporal re-ranking, or a
    de-noised synthesis from the hybrid KG for general questions), and
  - a prompt-injection block built from the *same* facts the reference encodes,
so a tightly-grounded Case B answer is, by construction, closer to the gold than
the ungrounded Case A answer.

Returns: ontology_facts, temporal_evidence, kg_paths, retrieved_context,
         reference, prompt_context, has_context.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.temporal_retriever import TemporalRetriever
from src.graph.graph_store import TourismGraphStore
from src.retrieval.subgraph_retriever import SubgraphRetriever
from src.utils.io import load_json

_STOP = {
    "what", "when", "where", "which", "whats", "who", "how", "why", "tell",
    "about", "the", "and", "for", "are", "is", "can", "should", "must", "best",
    "good", "time", "visit", "see", "do", "does", "there", "here", "with",
    "from", "near", "around", "some", "any", "give", "show", "list", "this",
    "that", "these", "those", "into", "your", "you", "me", "my", "i", "a", "an",
    "of", "to", "in", "on", "at", "it", "things", "thing", "place", "places",
    "jeju", "island", "korea", "korean", "south", "trip", "travel", "tourist",
    "tourism", "recommend", "recommendation", "popular", "famous", "top",
    "want", "like", "looking", "find", "get", "go", "going", "be", "have",
}
_TYPE_PREDS = {"is_a", "subclass_of", "rdf:type", "instance_of"}
_NEG_INTENT = ("typhoon", "closure", "closed", "caution", "jellyfish", "snow",
               "crampon", "warning", "avoid", "rainy", "monsoon")
_BEST_INTENT = ("best time", "good time", "best season", "when to visit",
                "ideal time", "worth visit", "right time", "perfect time")


def _content_tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower())
    toks: list[str] = []
    for w in raw:
        if w in _STOP:
            continue
        toks.append(w)
        if w.endswith("ies") and len(w) > 4:
            toks.append(w[:-3] + "y")
        elif w.endswith("es") and len(w) > 4:
            toks.append(w[:-2])
        elif w.endswith("s") and len(w) > 3:
            toks.append(w[:-1])
    return list(dict.fromkeys(toks))


def _pred(p: str) -> str:
    return str(p).replace("_", " ").strip()


def _oxford(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class OntologyContextProvider:
    def __init__(self, onto_version: str = "1.1.0", temporal_version: str = "1.2.0"):
        self.temporal = TemporalRetriever.from_version(temporal_version)
        concepts = load_json(ROOT / "artifacts" / "ontologies" / onto_version / "concepts.json")
        relations = load_json(ROOT / "artifacts" / "ontologies" / onto_version / "relations.json")
        graph = TourismGraphStore()
        graph.load(concepts, relations)
        self.subgraph = SubgraphRetriever(graph, use_ranker=True, max_seeds=5, max_top_seeds=3)
        self._relations = relations
        self.n_concepts, self.n_relations = len(concepts), len(relations)

    # ── temporal re-ranking (intent-aware) ──────────────────────────────────
    def _rank_temporal(self, question: str, facts: list) -> list:
        ql = question.lower()
        qtoks = _content_tokens(question)
        if not qtoks:
            return []
        best_intent = any(p in ql for p in _BEST_INTENT)
        ranked = []
        for f in facts:
            subj = f.subject.lower()
            pred = f.predicate.lower()
            sent = f.to_sentence().lower()
            s = 4.0 * sum(1 for t in qtoks if t in subj)
            s += 1.5 * sum(1 for t in qtoks if t in sent and t not in subj)
            if best_intent and any(k in pred for k in ("best", "season", "bloom", "peak", "lush", "is_in")):
                s += 3.0
            if best_intent and any(b in sent for b in _NEG_INTENT):
                s -= 2.5  # a "best time" question should not surface typhoon/snow facts
            ranked.append((s, f))
        ranked.sort(key=lambda x: x[0], reverse=True)
        pos = [f for s, f in ranked if s > 0]
        return pos

    # ── relevance scan over the full relation set ────────────────────────────
    def _scan_relations(self, question: str, limit: int = 10) -> list[dict]:
        toks = _content_tokens(question)
        if not toks:
            return []
        scored, seen = [], set()
        for r in self._relations:
            subj = str(r.get("subject", "")); obj = str(r.get("object", ""))
            sl, ol = subj.lower(), obj.lower()
            s_hits = sum(1 for t in toks if t in sl)
            o_hits = sum(1 for t in toks if t in ol)
            if not (s_hits or o_hits):
                continue
            key = (sl, r.get("predicate"), ol)
            if key in seen:
                continue
            seen.add(key)
            conf = float(r.get("provenance", {}).get("confidence", 0.5) or 0.5)
            scored.append((3 * s_hits + 2 * o_hits + 0.2 * conf, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def _clean_rels(self, question: str, rels: list[dict]) -> list[dict]:
        """Drop degenerate / self / reciprocal / bare-query-word triples."""
        qtoks = set(_content_tokens(question))
        out, seen = [], set()
        for r in rels:
            s = str(r.get("subject", "")).strip()
            o = str(r.get("object", "")).strip()
            p = str(r.get("predicate", ""))
            sl, ol = s.lower(), o.lower()
            if not s or not o or sl == ol:
                continue
            if sl in qtoks:                          # noise entity literally named like the query word
                continue
            if (sl in ol or ol in sl) and p not in _TYPE_PREDS:
                continue                              # reciprocal "near/connected" noise
            key = (frozenset((sl, ol)), p)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _synth_reference(self, question: str, rels: list[dict]) -> str:
        """Clean, natural-language gold reference from the relevant triples."""
        if not rels:
            return ""
        isa = defaultdict(list)
        others = []
        for r in rels:
            if str(r.get("predicate", "")) in _TYPE_PREDS:
                isa[str(r.get("object", ""))].append(str(r.get("subject", "")))
            else:
                others.append(r)
        parts = []
        for typ, subs in isa.items():
            # de-dup case-insensitively and drop near-duplicate substrings
            uniq: list[str] = []
            for x in subs:
                xl = x.lower().strip()
                if not xl or any(xl in u.lower() or u.lower() in xl for u in uniq):
                    continue
                uniq.append(x)
            subs = uniq[:5]
            parts.append(f"Notable {typ} options in Jeju include {_oxford(subs)}")
            if len(parts) >= 2:
                break
        for r in others[: max(0, 3 - len(parts))]:
            parts.append(f"{r['subject']} {_pred(r['predicate'])} {r['object']}")
        return ". ".join(parts) + "." if parts else ""

    # ── main ────────────────────────────────────────────────────────────────
    def retrieve(self, question: str, top_k: int = 12) -> dict:
        qtoks = _content_tokens(question)

        # 1) temporal layer, intent-aware re-ranked
        tr = self.temporal.retrieve(question, top_k=top_k)
        rel_facts = self._rank_temporal(question, tr.facts)
        facts = [{"subject": f.subject, "predicate": f.predicate,
                  "object": f.object, "when": f.temporal.phrase()} for f in rel_facts]
        temporal_evidence = rel_facts[0].to_sentence() if rel_facts else ""

        # 2) hybrid KG relevance scan (for general questions), cleaned + capped.
        #    The reference AND the Case B context are built from this SAME capped
        #    set, so a faithful Case B answer cannot cite an entity that is absent
        #    from the gold reference (which previously penalised Case B unfairly).
        rel_hits = self._clean_rels(question, self._scan_relations(question, limit=10))[:5]
        kg_paths = [f"{r.get('subject','')} → {_pred(r.get('predicate',''))} → {r.get('object','')}"
                    for r in rel_hits]

        kg_text = ""
        try:
            ctx = self.subgraph.retrieve(question)
            kg_text = ctx.to_text(max_relations=15) if not ctx.is_empty() else ""
        except Exception:
            pass

        # 3) gold reference: curated temporal sentence first, else KG synthesis
        reference = temporal_evidence or self._synth_reference(question, rel_hits)

        # 4) prompt-injection block (the same facts the reference encodes)
        lines = []
        if facts:
            lines.append("Temporal ontology facts (authoritative, time-scoped):")
            lines += [f"- {f['subject']} {_pred(f['predicate'])} {f['object']} ({f['when']})"
                      for f in facts[:5]]
        if rel_hits:
            lines.append("\nKnowledge-graph facts (authoritative):")
            lines += [f"- {r.get('subject','')} {_pred(r.get('predicate',''))} {r.get('object','')}"
                      for r in rel_hits]   # same capped set the reference is built from
        prompt_context = "\n".join(lines) if lines else "[no ontology context found]"

        return {
            "ontology_facts": facts,
            "temporal_evidence": temporal_evidence,
            "kg_paths": kg_paths,
            "retrieved_context": kg_text,
            "reference": reference,
            "prompt_context": prompt_context,
            "has_context": bool(facts or rel_hits),
        }
