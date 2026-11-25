"""
One place that turns (question, context) into model-ready token ids.

Training and generation both import from here, so the prompt a model is tuned on
is byte-identical to the prompt it is served at test time.
"""
from __future__ import annotations

import re

from . import config as C
from .contexts import build_prompt

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by DeepSeek-R1 / Qwen3."""
    return _THINK_RE.sub("", text or "").replace("<think>", "").replace("</think>", "").strip()


def prepare_tokenizer(tok, for_training: bool):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right" if for_training else "left"
    return tok


def _messages(question: str, arm: str, ctx: str | None, supports_system: bool) -> list[dict]:
    user = build_prompt(question, arm, ctx)
    if supports_system:
        return [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": user}]
    return [{"role": "user", "content": f"{C.SYSTEM}\n\n{user}"}]


def supports_system_role(tok) -> bool:
    """Mistral's template raises on a system turn; Llama/Qwen/DeepSeek accept one."""
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"},
                                 {"role": "user", "content": "y"}], tokenize=False)
        return True
    except Exception:
        return False


def render_prompt(tok, question: str, arm: str, ctx: str | None, supports_system: bool) -> str:
    """Chat-templated prompt string ending at the assistant turn."""
    msgs = _messages(question, arm, ctx, supports_system)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:                                   # Qwen3: keep the model out of thinking mode
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def build_training_example(tok, question: str, answer: str, arm: str, ctx: str | None,
                           supports_system: bool, max_len: int) -> dict | None:
    """Prompt+answer ids with the prompt masked out of the loss.

    Long prompts are truncated from the LEFT of the context block (keeping the
    question, which sits at the end) so the answer is never cut.
    """
    prompt = render_prompt(tok, question, arm, ctx, supports_system)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer.strip() + tok.eos_token, add_special_tokens=False)["input_ids"]

    if len(a_ids) >= max_len:                       # pathological answer, skip
        return None
    if len(p_ids) + len(a_ids) > max_len:
        keep = max_len - len(a_ids)
        p_ids = p_ids[-keep:]                       # drop the head of the context block
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + list(a_ids)
    return {"input_ids": ids, "labels": labels}
