"""
Temporal ontology demo — interactive side-by-side CLI.

Ask any Jeju tourism question and see the vanilla LLM's answer WITHOUT the
ontology (Case A) next to its answer WITH the injected temporal ontology facts
(Case B), plus the temporal evidence used. Built for the live mid-July demo.

Usage:
  python scripts/temporal_demo_cli.py                       # interactive loop
  python scripts/temporal_demo_cli.py -q "When do cherry blossoms bloom in Jeju?"
  python scripts/temporal_demo_cli.py -q "What can I see now?" --month 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient
from src.retrieval.temporal_retriever import TemporalRetriever

G, R, B, RST, C, Y, DIM = "\033[92m", "\033[91m", "\033[1m", "\033[0m", "\033[96m", "\033[93m", "\033[2m"

SYSTEM = "You are a concise tourism assistant for Jeju Island, South Korea. Answer in 1-2 sentences."
CASE_A = "Question: {q}\nAnswer concisely."
CASE_B = ("{context}\n\nUsing ONLY the authoritative temporal facts above when relevant, "
          "answer the question. Be specific about months or seasons.\n\nQuestion: {q}\nAnswer concisely.")


def answer_once(llm: LLMClient, retriever: TemporalRetriever, q: str, month=None) -> None:
    tr = retriever.retrieve(q, top_k=5, reference_month=month)
    ans_a = llm.call(CASE_A.format(q=q), system=SYSTEM).strip()
    ans_b = llm.call(CASE_B.format(context=tr.context, q=q), system=SYSTEM).strip()

    print(f"\n{B}Q:{RST} {q}" + (f"   {DIM}(reference month: {month}){RST}" if month else ""))
    print(f"\n{R}{B}── Case A — WITHOUT ontology ──{RST}")
    print(f"  {ans_a}")
    print(f"\n{G}{B}── Case B — WITH temporal ontology ──{RST}")
    print(f"  {ans_b}")
    print(f"\n{C}Temporal evidence injected (Case B):{RST}")
    if tr.facts:
        for f in tr.facts:
            print(f"  {DIM}• {f.to_sentence()}{RST}")
    else:
        print(f"  {DIM}(no matching temporal facts){RST}")
    print("─" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal demo CLI (Case A vs B)")
    ap.add_argument("-q", "--question", default=None)
    ap.add_argument("--month", type=int, default=None, help="reference month 1-12 (optional)")
    ap.add_argument("--version", default="1.2.0")
    args = ap.parse_args()

    retriever = TemporalRetriever.from_version(args.version)
    llm = LLMClient(backend="local", max_tokens=160, temperature=0.0)
    if not llm.is_available:
        print(f"{R}No local LLM backend. Set HF_TOKEN + LOCAL_MODEL_KEY in .env.{RST}")
        sys.exit(1)

    print(f"{B}Temporal Ontology Demo{RST}  {DIM}(LLM: {llm.backend}, ontology v{args.version}){RST}")

    if args.question:
        answer_once(llm, retriever, args.question, month=args.month)
        return

    print(f"{DIM}Type a question (or 'quit'). Optionally prefix 'month=4: <question>'.{RST}")
    while True:
        try:
            line = input(f"\n{B}>{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        month = args.month
        if line.lower().startswith("month="):
            head, _, rest = line.partition(":")
            try:
                month = int(head.split("=")[1])
            except (ValueError, IndexError):
                month = args.month
            line = rest.strip() or line
        answer_once(llm, retriever, line, month=month)


if __name__ == "__main__":
    main()
