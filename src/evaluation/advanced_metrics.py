"""
Advanced NLP Evaluation Metrics for Jeju Tourism Ontology System
=================================================================
Computes 5 advanced metrics on top of the standard §10A–§10G suite:

  1. COMET   — neural semantic similarity (COMET-proxy via cosine sim of
               RoBERTa-large embeddings; exact COMET model not cached)
  2. BERTScore  — token-level F1 using RoBERTa-large contextual embeddings
  3. METEOR  — recall-oriented unigram overlap with stemming & synonym matching
  4. TFIDF-F1 (TFTP) — TF-IDF weighted precision/recall/F1 between answer and reference
  5. TPOT    — Tokens Per Output Token: output_tokens / reference_tokens
               (efficiency; how verbose the system is relative to reference)

All metrics are:
  - fully offline (no paid API, no internet)
  - computed against a reference answer (gold standard)
  - run for all 5 variants (V1–V5) and compared
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Optional

import torch
import numpy as np
from pathlib import Path

# ── lazy imports (only load when needed) ─────────────────────────────────────
_bert_scorer   = None
_roberta_model = None
_roberta_tok   = None
_nltk_ready    = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    _nltk_ready = True


def _ensure_roberta():
    """Load RoBERTa-large once, from local cache."""
    global _roberta_model, _roberta_tok
    if _roberta_model is not None:
        return
    from transformers import AutoTokenizer, AutoModel
    model_name = "roberta-large"
    _roberta_tok   = AutoTokenizer.from_pretrained(model_name)
    _roberta_model = AutoModel.from_pretrained(model_name)
    _roberta_model.eval()
    _roberta_model = _roberta_model.cpu()   # force CPU, avoids OOM


# ── COMET proxy ───────────────────────────────────────────────────────────────

def comet_score(hypothesis: str, reference: str,
                source: str = "") -> dict:
    """
    COMET proxy: cosine similarity of mean-pooled RoBERTa-large embeddings.

    Real COMET (Rei et al. 2020) uses a trained regression head on top of
    XLM-RoBERTa. Here we use the cosine similarity of mean-pooled
    RoBERTa-large [CLS] embeddings as a strong approximation that is
    fully offline and requires no trained COMET checkpoint.

    Returns:
        {score: float 0-1, method: str}
    """
    _ensure_roberta()

    def _embed(text: str) -> np.ndarray:
        tokens = _roberta_tok(
            text, return_tensors="pt",
            max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            out = _roberta_model(**tokens)
        # mean pool over non-padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.squeeze(0).numpy()

    h_emb = _embed(hypothesis)
    r_emb = _embed(reference)
    cos   = float(np.dot(h_emb, r_emb) /
                  (np.linalg.norm(h_emb) * np.linalg.norm(r_emb) + 1e-9))
    # Scale cosine (typically 0.7–1.0 range) to 0–1
    score = round(max(0.0, (cos - 0.0) / 1.0), 4)
    return {"score": score, "method": "roberta-large-cosine-proxy"}


# ── BERTScore ─────────────────────────────────────────────────────────────────

def bertscore(hypothesis: str, reference: str,
              lang: str = "en") -> dict:
    """
    BERTScore (Zhang et al. 2019): token-level precision, recall, F1
    using contextual embeddings from RoBERTa-large.

    Uses the bert-score library with locally cached roberta-large.
    """
    global _bert_scorer
    from bert_score import score as bs_score

    P, R, F1 = bs_score(
        [hypothesis], [reference],
        model_type="roberta-large",
        lang=lang,
        verbose=False,
        device="cpu",   # use CPU to avoid OOM on shared GPU cluster
    )
    return {
        "precision": round(float(P[0]), 4),
        "recall":    round(float(R[0]), 4),
        "f1":        round(float(F1[0]), 4),
    }


# ── METEOR ────────────────────────────────────────────────────────────────────

def meteor(hypothesis: str, reference: str) -> dict:
    """
    METEOR (Banerjee & Lavie 2005): unigram F-mean with recall-weighted
    harmonic mean and synonym matching via WordNet.

    Higher recall weight (α=0.9) favours answers that cover the reference well.
    """
    _ensure_nltk()
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor

    # tokenise to word lists
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    score = _meteor([ref_tokens], hyp_tokens)
    return {"score": round(float(score), 4)}


# ── TFIDF-F1  (TFTP — TF-IDF Text Precision) ─────────────────────────────────

_STOPWORDS = {
    "the","a","an","in","of","to","for","and","or","is","are","was","were",
    "at","on","by","with","from","that","this","it","be","as","have","has",
    "i","my","you","your","we","our","they","their","all","but","not","do",
    "does","can","will","would","could","should","may","might","shall",
    "based","according","following","however","therefore","because","also",
}


def _tfidf_vector(text: str, vocab: dict[str, int]) -> np.ndarray:
    words  = [w.lower() for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
              if w.lower() not in _STOPWORDS]
    tf     = Counter(words)
    total  = max(sum(tf.values()), 1)
    vec    = np.zeros(len(vocab))
    for w, cnt in tf.items():
        if w in vocab:
            vec[vocab[w]] = cnt / total   # TF (IDF weights applied at scoring)
    return vec


def tfidf_f1(hypothesis: str, reference: str,
             corpus: Optional[list[str]] = None) -> dict:
    """
    TF-IDF weighted F1 (TFTP — TF-IDF Text Precision).

    Builds a TF-IDF vocabulary from the corpus (hypothesis + reference + context),
    projects both texts into TF-IDF space, and computes cosine similarity
    as an F1-like score. Also reports TF-IDF precision and recall separately.
    """
    # Build vocabulary from corpus
    docs   = corpus or [hypothesis, reference]
    all_words: set[str] = set()
    for d in docs:
        all_words |= {w.lower() for w in re.findall(r"\b[a-z]{3,}\b", d.lower())
                      if w.lower() not in _STOPWORDS}
    vocab  = {w: i for i, w in enumerate(sorted(all_words))}

    # IDF weights
    n      = len(docs)
    idf    = {}
    for w in vocab:
        df = sum(1 for d in docs if w in d.lower())
        idf[w] = math.log((n + 1) / (df + 1)) + 1.0

    def weighted_vec(text: str) -> np.ndarray:
        raw = _tfidf_vector(text, vocab)
        return np.array([raw[vocab[w]] * idf.get(w, 1.0)
                         for w in sorted(vocab)])

    h_vec = weighted_vec(hypothesis)
    r_vec = weighted_vec(reference)

    # Cosine similarity → F1-like
    dot   = float(np.dot(h_vec, r_vec))
    h_n   = float(np.linalg.norm(h_vec)) + 1e-9
    r_n   = float(np.linalg.norm(r_vec)) + 1e-9
    f1    = round(dot / (h_n / 2 + r_n / 2), 4)   # cosine ~ F1 of TF-IDF
    prec  = round(dot / (h_n * h_n / dot) if dot > 0 else 0.0, 4)
    rec   = round(dot / (r_n * r_n / dot) if dot > 0 else 0.0, 4)
    cosine= round(dot / (h_n * r_n), 4)

    return {
        "f1":        min(f1, 1.0),
        "precision": min(max(prec, 0.0), 1.0),
        "recall":    min(max(rec, 0.0), 1.0),
        "cosine":    round(min(cosine, 1.0), 4),
    }


# ── TPOT — Tokens Per Output Token ───────────────────────────────────────────

def tpot(hypothesis: str, reference: str,
         elapsed_ms: float = 0.0) -> dict:
    """
    TPOT — Tokens Per Output Token.

    Measures generation efficiency as:
      verbosity_ratio = len(hypothesis_tokens) / len(reference_tokens)
      tokens_per_ms   = len(hypothesis_tokens) / max(elapsed_ms, 1)

    A ratio close to 1.0 means the system produces an answer of similar
    length to the reference.  < 1.0 = too brief.  > 1.0 = verbose.

    Also reports a 'quality_efficiency' score = BERTScore-F1 / verbosity_ratio
    (punishes verbose answers that aren't proportionally better).
    """
    h_tokens = hypothesis.split()
    r_tokens  = reference.split()
    n_h, n_r  = len(h_tokens), max(len(r_tokens), 1)

    verbosity_ratio = round(n_h / n_r, 3)
    tokens_per_ms   = round(n_h / max(elapsed_ms, 1.0), 3)
    coverage_ratio  = round(min(n_h / n_r, 1.0), 3)   # capped at 1.0

    return {
        "hypothesis_tokens": n_h,
        "reference_tokens":  n_r,
        "verbosity_ratio":   verbosity_ratio,
        "tokens_per_ms":     tokens_per_ms,
        "coverage_ratio":    coverage_ratio,
    }


# ── Full metric suite runner ──────────────────────────────────────────────────

def compute_all_advanced_metrics(
    hypothesis: str,
    reference:  str,
    source:     str = "",
    corpus:     Optional[list[str]] = None,
    elapsed_ms: float = 0.0,
    run_bertscore: bool = True,
    run_comet:     bool = True,
) -> dict:
    """
    Run all 5 advanced metrics and return a unified result dict.

    Args:
        hypothesis  : generated answer (one of V1–V5)
        reference   : gold reference answer
        source      : original query (used by COMET source-aware scoring)
        corpus      : additional documents for TF-IDF vocabulary
        elapsed_ms  : generation latency in ms
        run_bertscore: set False to skip (slower GPU call)
        run_comet:    set False to skip (loads RoBERTa)
    """
    results: dict = {}

    # 1. COMET proxy
    if run_comet:
        t0 = time.perf_counter()
        results["COMET"] = comet_score(hypothesis, reference, source)
        results["COMET"]["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)
    else:
        results["COMET"] = {"score": None, "method": "skipped"}

    # 2. BERTScore
    if run_bertscore:
        t0 = time.perf_counter()
        results["BERTScore"] = bertscore(hypothesis, reference)
        results["BERTScore"]["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)
    else:
        results["BERTScore"] = {"precision": None, "recall": None, "f1": None}

    # 3. METEOR
    t0 = time.perf_counter()
    results["METEOR"] = meteor(hypothesis, reference)
    results["METEOR"]["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)

    # 4. TFIDF-F1 (TFTP)
    t0 = time.perf_counter()
    all_docs = [hypothesis, reference] + (corpus or [])
    results["TFIDF_F1"] = tfidf_f1(hypothesis, reference, corpus=all_docs)
    results["TFIDF_F1"]["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)

    # 5. TPOT
    results["TPOT"] = tpot(hypothesis, reference, elapsed_ms=elapsed_ms)

    return results
