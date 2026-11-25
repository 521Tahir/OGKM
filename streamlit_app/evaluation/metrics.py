"""
Metric computation (runs in the main process pinned to the eval GPU).

Reference = the retrieved ontology fact (authoritative gold). Computes:
  BERT-F1 (BERTScore), METEOR, COMET, Key-token.
LLM-Judge is computed separately by routing judge prompts through the warm
model pool (see app.py), so no extra model is loaded here.
"""
from __future__ import annotations

import re

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]

_bert = {"loaded": False}
_comet = {"model": None, "tried": False}


def _ensure_meteor():
    try:
        import nltk
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False


_KT_STOP = {"jeju", "island", "korea", "korean", "the", "and", "near", "season"}


def derive_key_tokens(reference: str) -> list[str]:
    """Salient gold tokens: month names + named entities (multi-word capitalised
    phrases) from the reference, so non-temporal answers are scored too."""
    ref = reference or ""
    tokens: list[str] = [m for m in _MONTHS if re.search(rf"\b{m}\b", ref.lower())]
    # multi-word capitalised entity phrases, e.g. "Jeju Haenyeo Museum"
    for phrase in re.findall(r"\b[A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*\b", ref):
        if phrase.lower() in _KT_STOP or len(phrase) < 4:
            continue
        # drop a leading bare "Jeju " qualifier so the distinctive name remains
        cleaned = re.sub(r"^Jeju\s+", "", phrase).strip()
        if len(cleaned) >= 4 and cleaned.lower() not in _KT_STOP:
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))[:8]


def key_token_score(candidate: str, key_tokens: list[str]) -> float | None:
    if not key_tokens:
        return None
    a = candidate.lower()
    return round(sum(1 for t in key_tokens if t.lower() in a) / len(key_tokens), 4)


def meteor_score_one(candidate: str, reference: str) -> float | None:
    if not _ensure_meteor():
        return None
    try:
        from nltk.translate.meteor_score import meteor_score
        return round(meteor_score([reference.split()], candidate.split()), 4)
    except Exception:
        return None


def bert_f1_batch(cands: list[str], refs: list[str]) -> list[float | None]:
    try:
        from bert_score import score
        _, _, f = score(cands, refs, lang="en", verbose=False)
        return [round(x, 4) for x in f.tolist()]
    except Exception:
        return [None] * len(cands)


def comet_batch(srcs: list[str], cands: list[str], refs: list[str]) -> list[float | None]:
    if not _comet["tried"]:
        _comet["tried"] = True
        try:
            from comet import download_model, load_from_checkpoint
            _comet["model"] = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        except Exception:
            _comet["model"] = None
    if _comet["model"] is None:
        return [None] * len(cands)
    try:
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, cands, refs)]
        out = _comet["model"].predict(data, batch_size=16, gpus=1, progress_bar=False)
        return [round(x, 4) for x in out.scores]
    except Exception:
        return [None] * len(cands)


def parse_judge(verdict: str) -> float:
    """Continuous response-accuracy in [0,1] from a graded 0-10 judge score
    (falls back to the older CORRECT/PARTIAL/INCORRECT wording)."""
    v = (verdict or "").strip()
    m = re.search(r"\b(10|[0-9])(?:\s*/\s*10)?\b", v)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 10:
            return round(n / 10.0, 3)
    u = v.upper()
    if "CORRECT" in u and "INCORRECT" not in u:
        return 1.0
    if "PARTIAL" in u:
        return 0.5
    return 0.0


def composite(metrics: dict, weights: dict) -> float | None:
    num, wsum = 0.0, 0.0
    for k, w in weights.items():
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            num += w * v
            wsum += w
    return round(num / wsum, 4) if wsum else None
