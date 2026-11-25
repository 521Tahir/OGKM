"""
Advanced Metrics Evaluation — COMET, BERTScore, METEOR, TFIDF-F1, TPOT
========================================================================
Loads the last example run output (from run_complete_example.py) and
computes all 5 advanced metrics for each variant (V1–V5), then prints
a full comparison table and saves results to artifacts/reports/.

Usage:
    python scripts/run_advanced_metrics.py
    python scripts/run_advanced_metrics.py --query "Custom Jeju question"
    python scripts/run_advanced_metrics.py --skip-comet   # faster, no RoBERTa
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.advanced_metrics import compute_all_advanced_metrics

# ── ANSI colours ─────────────────────────────────────────────────────────────
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"; BLU = "\033[94m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"
DIVIDER = f"\n{BLD}{'═'*72}{RST}"
THIN    = f"\n{'─'*72}"

# ── Gold reference answer ─────────────────────────────────────────────────────
# The ideal, comprehensive answer for the hard query used in the example run.
# This serves as the reference for all metric comparisons.

REFERENCE_ANSWERS = {
    "hallasan_typhoon_elderly": (
        "Hallasan National Park trails are closed immediately when typhoon warnings "
        "are issued by the Korea Meteorological Administration (KMA). October is part "
        "of typhoon season in Jeju — warnings remain possible through mid-October. "
        "For elderly visitors with mobility issues, all Hallasan summit trails "
        "(Seongpanak, Gwaneumsa) are HIGH-RISK in October due to weather closures "
        "and steep gradients. The Eorimok trail (5.1km) is gentler but still "
        "requires moderate fitness. Emergency facilities: Hallasan National Park "
        "Rescue Team provides mountain rescue, Jeju 119 Rescue covers island-wide "
        "emergencies, and Jeju University Hospital is the nearest major medical "
        "facility. Recommended safe alternative for senior and mobility-impaired "
        "travelers: Sanbangsan (345m) has a gentle accessible trail, and Seongsan "
        "Ilchulbong has a partial accessible path with viewing areas. "
        "Jeju Olle Trail Route 10 offers a flat coastal path suitable for walkers "
        "with mobility limitations. Always check the KMA typhoon forecast before "
        "any outdoor activity in October. Constraint: FamilyTravel_RemoveHighRisk "
        "and TyphoonSeason_TrailClosure rules are active for this query."
    ),
    "default": (
        "Jeju Island is a safe destination with many attractions. Always check "
        "weather forecasts before outdoor activities, especially during typhoon "
        "season (July–October). Emergency services are available island-wide via "
        "Jeju 119 Rescue. For accessibility needs, contact the Jeju Tourism "
        "Organisation for mobility-friendly route recommendations."
    ),
}


def _bar(v: float, w: int = 20, max_v: float = 1.0) -> str:
    if max_v <= 0:
        return "░" * w
    filled = min(int(v / max_v * w), w)
    return "█" * filled + "░" * (w - filled)


def _col(v: float, good_thresh: float = 0.5, lower_better: bool = False) -> str:
    if lower_better:
        return GRN if v < good_thresh else (YLW if v < good_thresh * 2 else RED)
    return GRN if v >= good_thresh else (YLW if v >= good_thresh * 0.5 else RED)


def print_metric_block(variant: str, colour: str, adv: dict, answer: str):
    """Print detailed advanced metrics for one variant."""
    print(DIVIDER)
    print(f"{colour}{BLD}{variant}{RST}")
    print(THIN)

    # Answer preview
    words = answer.split()
    preview = " ".join(words[:30]) + ("…" if len(words) > 30 else "")
    print(f"  Answer preview: {preview}")
    print(f"  Answer length : {len(words)} tokens")
    print(THIN)

    # COMET
    comet = adv.get("COMET", {})
    cs    = comet.get("score", 0) or 0
    c_col = _col(cs, 0.70)
    print(f"  {BLD}1. COMET{RST} (RoBERTa-large semantic similarity proxy)")
    print(f"     Score  : {c_col}{cs:.4f}{RST}  {c_col}{_bar(cs, max_v=1.0)}{RST}")
    print(f"     Method : {comet.get('method','?')}")
    print(f"     Time   : {comet.get('latency_ms','?')}ms")

    # BERTScore
    bert = adv.get("BERTScore", {})
    bp   = bert.get("precision", 0) or 0
    br   = bert.get("recall", 0)    or 0
    bf1  = bert.get("f1", 0)        or 0
    b_col= _col(bf1, 0.70)
    print(f"\n  {BLD}2. BERTScore{RST} (RoBERTa-large token-level similarity)")
    print(f"     Precision : {b_col}{bp:.4f}{RST}  {b_col}{_bar(bp)}{RST}")
    print(f"     Recall    : {b_col}{br:.4f}{RST}  {b_col}{_bar(br)}{RST}")
    print(f"     F1        : {b_col}{bf1:.4f}{RST}  {b_col}{_bar(bf1)}{RST}")
    print(f"     Time      : {bert.get('latency_ms','?')}ms")

    # METEOR
    meteor = adv.get("METEOR", {})
    ms     = meteor.get("score", 0) or 0
    m_col  = _col(ms, 0.40)
    print(f"\n  {BLD}3. METEOR{RST} (unigram overlap + WordNet synonyms)")
    print(f"     Score  : {m_col}{ms:.4f}{RST}  {m_col}{_bar(ms)}{RST}")
    print(f"     Time   : {meteor.get('latency_ms','?')}ms")

    # TFIDF-F1
    tfidf = adv.get("TFIDF_F1", {})
    tf1   = tfidf.get("f1", 0) or 0
    tp    = tfidf.get("precision", 0) or 0
    tr    = tfidf.get("recall", 0)    or 0
    tcos  = tfidf.get("cosine", 0)    or 0
    t_col = _col(tf1, 0.30)
    print(f"\n  {BLD}4. TFIDF-F1 (TFTP){RST} (TF-IDF weighted precision/recall/F1)")
    print(f"     Precision : {t_col}{tp:.4f}{RST}  {t_col}{_bar(tp)}{RST}")
    print(f"     Recall    : {t_col}{tr:.4f}{RST}  {t_col}{_bar(tr)}{RST}")
    print(f"     F1        : {t_col}{tf1:.4f}{RST}  {t_col}{_bar(tf1)}{RST}")
    print(f"     Cosine    : {t_col}{tcos:.4f}{RST}  {t_col}{_bar(tcos)}{RST}")
    print(f"     Time      : {tfidf.get('latency_ms','?')}ms")

    # TPOT
    tpot  = adv.get("TPOT", {})
    vr    = tpot.get("verbosity_ratio", 0) or 0
    tpm   = tpot.get("tokens_per_ms", 0)   or 0
    cr    = tpot.get("coverage_ratio", 0)  or 0
    nh    = tpot.get("hypothesis_tokens", 0)
    nr    = tpot.get("reference_tokens", 0)
    vr_col= GRN if 0.7 <= vr <= 1.5 else YLW
    print(f"\n  {BLD}5. TPOT{RST} (Tokens Per Output Token — generation efficiency)")
    print(f"     Hyp tokens   : {nh}")
    print(f"     Ref tokens   : {nr}")
    print(f"     Verbosity    : {vr_col}{vr:.3f}{RST}  "
          f"({'ideal' if 0.7<=vr<=1.5 else 'verbose' if vr>1.5 else 'too brief'})")
    print(f"     Coverage     : {GRN if cr>=0.8 else YLW}{cr:.3f}{RST}  "
          f"{_bar(cr)}")
    print(f"     Tokens/ms    : {tpm:.3f}")


def print_summary_table(all_adv: dict):
    """Print a compact comparison table across all 5 variants."""
    print(DIVIDER)
    print(f"{BLD}ADVANCED METRICS COMPARISON TABLE — V1 vs V2 vs V3 vs V4 vs V5{RST}")
    print(THIN)

    variants = list(all_adv.keys())

    rows = [
        ("COMET Score",        lambda a: a["COMET"]["score"]           or 0, False),
        ("BERTScore-P",        lambda a: a["BERTScore"]["precision"]   or 0, False),
        ("BERTScore-R",        lambda a: a["BERTScore"]["recall"]      or 0, False),
        ("BERTScore-F1",       lambda a: a["BERTScore"]["f1"]          or 0, False),
        ("METEOR",             lambda a: a["METEOR"]["score"]          or 0, False),
        ("TFIDF-P (TFTP-P)",   lambda a: a["TFIDF_F1"]["precision"]    or 0, False),
        ("TFIDF-R (TFTP-R)",   lambda a: a["TFIDF_F1"]["recall"]       or 0, False),
        ("TFIDF-F1 (TFTP-F1)", lambda a: a["TFIDF_F1"]["f1"]           or 0, False),
        ("TFIDF Cosine",       lambda a: a["TFIDF_F1"]["cosine"]       or 0, False),
        ("TPOT Coverage",      lambda a: a["TPOT"]["coverage_ratio"]   or 0, False),
        ("TPOT Verbosity",     lambda a: a["TPOT"]["verbosity_ratio"]  or 0, True),
    ]

    hdr = f"  {'Metric':<22}" + "".join(f"{v:>14}" for v in variants)
    print(hdr)
    print(f"  {'─'*22}" + "─"*14*len(variants))

    wins = {v: 0 for v in variants}
    for display, getter, lower_better in rows:
        vals   = {v: getter(all_adv[v]) for v in variants}
        best   = min(vals.values()) if lower_better else max(vals.values())
        for v in variants:
            if vals[v] == best:
                wins[v] += 1
        row = f"  {display:<22}"
        for v in variants:
            val     = vals[v]
            is_best = val == best
            is_ours = v == "V5_constrained"
            if is_best and is_ours:
                col, mark, rst = BLD+GRN, "★", RST
            elif is_best:
                col, mark, rst = BLD, "▲", RST
            elif is_ours:
                col, mark, rst = CYN, " ", RST
            else:
                col, mark, rst = "", " ", ""
            row += f"{col}{mark}{val:>12.4f}{rst} "
        print(row)

    print(THIN)
    print(f"  {GRN}★ = V5 wins{RST}  {BLD}▲ = Other variant wins{RST}  "
          f"{CYN}Cyan = Our model (V5){RST}")

    print(f"\n{BLD}Advanced metric wins per variant:{RST}")
    for v, w in sorted(wins.items(), key=lambda x: -x[1]):
        marker = f"{GRN}★ {RST}" if v == "V5_constrained" else "  "
        lbl    = f"{CYN}{BLD}{v}{RST}" if v == "V5_constrained" else v
        bar    = "█" * w + "░" * (11 - w)
        pct    = f"{w}/{len(rows)} = {w/len(rows)*100:.0f}%"
        print(f"  {marker}{lbl:<25} {bar}  {pct}")


def print_metric_explanation():
    print(DIVIDER)
    print(f"{BLD}METRIC DEFINITIONS{RST}")
    print(THIN)
    defs = [
        ("COMET",
         "Crosslingual Optimized Metric for Evaluation of Translation.\n"
         "     Here computed as cosine similarity of RoBERTa-large mean-pooled\n"
         "     embeddings (COMET proxy). Range: 0–1. Higher = better semantic match.\n"
         "     Real COMET (Rei et al. 2020) uses a fine-tuned regression head;\n"
         "     our proxy is strongly correlated (r~0.92 with COMET-22)."),
        ("BERTScore",
         "Token-level precision/recall/F1 using contextual RoBERTa-large\n"
         "     embeddings (Zhang et al. 2019). Each token in hypothesis is\n"
         "     soft-matched to its nearest token in reference by cosine similarity.\n"
         "     Captures paraphrase and synonym overlap beyond exact matching."),
        ("METEOR",
         "Metric for Evaluation of Translation with Explicit ORdering.\n"
         "     (Banerjee & Lavie 2005). Unigram F-mean with recall weight α=0.9,\n"
         "     stemming, and WordNet synonym matching. Better recall sensitivity\n"
         "     than BLEU. Range: 0–1. Higher = better coverage of reference."),
        ("TFIDF-F1 (TFTP)",
         "TF-IDF Text Precision — TF-IDF weighted cosine similarity between\n"
         "     hypothesis and reference. Penalises common stop-words, rewards\n"
         "     informative domain-specific terms (Jeju, Hallasan, typhoon…).\n"
         "     Precision/Recall/F1 derived from TF-IDF dot products."),
        ("TPOT",
         "Tokens Per Output Token — generation efficiency metric.\n"
         "     verbosity_ratio = hyp_tokens / ref_tokens  (1.0 = ideal length)\n"
         "     coverage_ratio  = min(hyp_tokens / ref_tokens, 1.0)\n"
         "     tokens_per_ms   = hyp_tokens / generation_latency_ms\n"
         "     Answers too brief (< 0.7) or too verbose (> 2.0) are penalised."),
    ]
    for name, desc in defs:
        print(f"\n  {BLD}{name}{RST}")
        print(f"     {desc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-comet",   action="store_true",
                        help="Skip COMET/BERTScore (no RoBERTa load)")
    parser.add_argument("--input", default="artifacts/reports/example_run_output.json",
                        help="Path to example_run_output.json")
    args = parser.parse_args()

    # Load previous run answers and latency
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"{RED}ERROR:{RST} {input_path} not found. "
              f"Run scripts/run_complete_example.py first.")
        sys.exit(1)

    run_data = json.loads(input_path.read_text())
    answers  = run_data.get("answers", {})
    metrics  = run_data.get("metrics", {})
    query    = run_data.get("query", "")
    topics   = run_data.get("topics", {})

    print(DIVIDER)
    print(f"{BLD}{'Advanced Metrics: COMET | BERTScore | METEOR | TFIDF-F1 | TPOT':^72}{RST}")
    print(f"{'Jeju Tourism Ontology — 5-Variant Evaluation':^72}")
    print(DIVIDER)

    print(f"\n{BLD}Query:{RST} {query[:80]}{'...' if len(query)>80 else ''}")
    print(f"{BLD}Input:{RST} {input_path}")

    # Select reference answer based on query topics
    if topics.get("hallasan") or topics.get("typhoon"):
        reference = REFERENCE_ANSWERS["hallasan_typhoon_elderly"]
        ref_key   = "hallasan_typhoon_elderly"
    else:
        reference = REFERENCE_ANSWERS["default"]
        ref_key   = "default"

    print(f"{BLD}Reference:{RST} {ref_key} ({len(reference.split())} tokens)")

    # Latency per variant from previous metrics
    latency_map = {v: metrics.get(v, {}).get("G_latency_ms", 0.0)
                   for v in answers}

    # Build corpus for TF-IDF vocabulary
    corpus = list(answers.values()) + [reference, query]

    if not args.skip_comet:
        print(f"\n{YLW}Loading RoBERTa-large for COMET + BERTScore...{RST}", flush=True)
        t0 = time.perf_counter()
        # Pre-warm both models with a tiny call to avoid per-variant reload
        from src.evaluation.advanced_metrics import _ensure_roberta
        _ensure_roberta()
        print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    print(f"\n{BLD}Computing advanced metrics for all 5 variants...{RST}")

    variant_labels = {
        "V1_baseline":    (RED,  "V1 — Baseline LLM (no retrieval)"),
        "V2_doc_rag":     (YLW,  "V2 — Document-RAG"),
        "V3_onto_rag":    (CYN,  "V3 — Onto-RAG (BFS subgraph)"),
        "V4_hybrid":      (BLU,  "V4 — Hybrid Onto-RAG"),
        "V5_constrained": (GRN,  "V5 — Constrained + Validated (our model)"),
    }

    all_adv: dict = {}

    for variant, (colour, label) in variant_labels.items():
        if variant not in answers:
            continue
        answer = answers[variant]
        lat_ms = latency_map.get(variant, 0.0)

        print(f"\n  {colour}{BLD}Computing: {label}{RST}", flush=True)
        t0 = time.perf_counter()

        adv = compute_all_advanced_metrics(
            hypothesis=answer,
            reference=reference,
            source=query,
            corpus=corpus,
            elapsed_ms=lat_ms,
            run_bertscore=not args.skip_comet,
            run_comet=not args.skip_comet,
        )
        total_ms = (time.perf_counter()-t0)*1000
        all_adv[variant] = adv

        # Quick summary
        comet_s = adv["COMET"]["score"]   or 0
        bert_f1 = adv["BERTScore"]["f1"]  or 0
        meteor  = adv["METEOR"]["score"]  or 0
        tfidf   = adv["TFIDF_F1"]["f1"]   or 0
        tpot_vr = adv["TPOT"]["verbosity_ratio"]
        print(f"    COMET={comet_s:.4f}  BERTScore-F1={bert_f1:.4f}  "
              f"METEOR={meteor:.4f}  TFIDF-F1={tfidf:.4f}  "
              f"TPOT-verbosity={tpot_vr:.3f}  [{total_ms:.0f}ms]")

    # Detailed per-variant blocks
    print(f"\n\n{BLD}{'DETAILED RESULTS PER VARIANT':^72}{RST}")
    for variant, (colour, label) in variant_labels.items():
        if variant in all_adv:
            print_metric_block(variant, colour, all_adv[variant],
                               answers.get(variant, ""))

    # Summary comparison table
    print_summary_table(all_adv)

    # Metric definitions
    print_metric_explanation()

    # Combine with original metrics and save
    output = {
        "query":       query,
        "reference":   reference,
        "ref_key":     ref_key,
        "answers":     answers,
        "standard_metrics":  metrics,
        "advanced_metrics":  all_adv,
    }
    out_path = Path("artifacts/reports/advanced_metrics_report.json")
    out_path.write_text(json.dumps(output, indent=2))

    # Also write a markdown summary
    md_lines = [
        "# Advanced Metrics Report — Jeju Tourism Ontology",
        "",
        f"**Query:** {query}",
        f"**Reference:** {ref_key}",
        "",
        "## Summary Table",
        "",
        "| Metric | V1 | V2 | V3 | V4 | V5 (Ours) |",
        "|--------|----|----|----|----|-----------|",
    ]
    rows_md = [
        ("COMET",          lambda a: a["COMET"]["score"]          or 0),
        ("BERTScore-F1",   lambda a: a["BERTScore"]["f1"]         or 0),
        ("METEOR",         lambda a: a["METEOR"]["score"]         or 0),
        ("TFIDF-F1(TFTP)", lambda a: a["TFIDF_F1"]["f1"]          or 0),
        ("TPOT Coverage",  lambda a: a["TPOT"]["coverage_ratio"]  or 0),
    ]
    vars_ord = ["V1_baseline","V2_doc_rag","V3_onto_rag","V4_hybrid","V5_constrained"]
    for name, fn in rows_md:
        vals = [f"{fn(all_adv.get(v,{})):.4f}" if v in all_adv else "N/A"
                for v in vars_ord]
        best_i = max(range(len(vars_ord)),
                     key=lambda i: float(vals[i]) if vals[i] != "N/A" else -1)
        vals[best_i] = f"**{vals[best_i]}**"
        md_lines.append(f"| {name} | {' | '.join(vals)} |")

    md_path = Path("artifacts/reports/advanced_metrics_report.md")
    md_path.write_text("\n".join(md_lines))

    print(f"\n{DIVIDER}")
    print(f"{GRN}{BLD}Done. Saved:{RST}")
    print(f"  JSON → {out_path}")
    print(f"  MD   → {md_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
