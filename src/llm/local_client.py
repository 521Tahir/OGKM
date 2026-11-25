"""
Local LLM client using HuggingFace Transformers.

Supports any instruction-tuned model loadable via the HF hub.
Default: meta-llama/Meta-Llama-3-8B-Instruct (requires HF token + Meta license)
Fallback: Qwen/Qwen2.5-7B-Instruct (fully open, no token required)

Features:
- Multi-GPU tensor parallel via device_map="auto"
- 4-bit quantization (bitsandbytes) to save VRAM
- Model-specific chat template formatting (Llama3, Qwen, Mistral)
- Configurable generation parameters
- First-call lazy loading (model loaded only when needed)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.local_client")

# ── Supported model chat templates ───────────────────────────────────────────

LLAMA3_SYSTEM_TPL = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

LLAMA3_NO_SYSTEM_TPL = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

QWEN_TPL = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
QWEN_NO_SYSTEM_TPL = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

MISTRAL_TPL = "[INST] {user} [/INST]"
MISTRAL_SYSTEM_TPL = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def _get_template_family(model_name: str) -> str:
    """Detect template family from model name."""
    lower = model_name.lower()
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    if "qwen" in lower:
        return "qwen"
    if "mistral" in lower:
        return "mistral"
    # Default: try to use tokenizer's built-in apply_chat_template
    return "auto"


def _format_prompt(
    user: str, system: str, family: str, tokenizer=None
) -> str:
    """Format prompt according to model family's chat template."""
    if family == "auto" and tokenizer is not None:
        # Use HF tokenizer's built-in chat template
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # fall through to llama3 default

    if family == "llama3":
        if system:
            return LLAMA3_SYSTEM_TPL.format(system=system, user=user)
        return LLAMA3_NO_SYSTEM_TPL.format(user=user)
    if family == "qwen":
        if system:
            return QWEN_TPL.format(system=system, user=user)
        return QWEN_NO_SYSTEM_TPL.format(user=user)
    if family == "mistral":
        if system:
            return MISTRAL_SYSTEM_TPL.format(system=system, user=user)
        return MISTRAL_TPL.format(user=user)

    # Bare fallback
    return f"{system}\n\n{user}" if system else user


# ── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "llama3",
        "requires_token": True,
        "recommended_gpus": 1,
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama3",
        "requires_token": True,
        "recommended_gpus": 1,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "requires_token": False,
        "recommended_gpus": 1,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "requires_token": False,
        "recommended_gpus": 1,
    },
    # DeepSeek-R1 distill (8B, Llama-based). family="auto" → use the tokenizer's
    # own chat template, which is correct for R1-distill models. Used to generate
    # the Jeju-specific dataset (matches the model that produced the original data).
    "deepseek-r1-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "family": "auto",
        "requires_token": False,
        "recommended_gpus": 1,
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "family": "auto",
        "requires_token": False,
        "recommended_gpus": 1,
    },
}


class LocalLLMClient:
    """
    Local HuggingFace model inference client.

    Usage:
        client = LocalLLMClient(model_key="llama3-8b")
        # or with full HF model ID:
        client = LocalLLMClient(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    Args:
        model_key:     shorthand key from MODEL_REGISTRY (e.g. "llama3-8b")
        model_id:      full HuggingFace model ID (overrides model_key)
        gpu_ids:       list of GPU device indices to use, e.g. [5, 6, 7, 8, 9]
        load_in_4bit:  quantise to 4-bit with bitsandbytes (saves ~50% VRAM)
        load_in_8bit:  quantise to 8-bit
        max_new_tokens: max tokens to generate
        temperature:   sampling temperature (0 = greedy)
        hf_token:      HuggingFace access token (needed for gated models)
    """

    def __init__(
        self,
        model_key: str = "llama3-8b",
        model_id: Optional[str] = None,
        gpu_ids: list[int] | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        # Resolve model ID
        if model_id:
            self.model_id = model_id
            self.family = _get_template_family(model_id)
        else:
            entry = MODEL_REGISTRY.get(model_key)
            if entry is None:
                raise ValueError(
                    f"Unknown model_key '{model_key}'. "
                    f"Available: {list(MODEL_REGISTRY.keys())}"
                )
            self.model_id = entry["hf_id"]
            self.family = entry["family"]

        # When CUDA_VISIBLE_DEVICES=2,3 is set, PyTorch remaps physical GPUs to
        # local indices 0,1,...  — we must always use local (0-based) indices in
        # max_memory, NOT physical GPU IDs.
        _env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        _n_visible = len([g for g in _env_gpus.split(",") if g.strip().isdigit()]) if _env_gpus else 0
        if gpu_ids is not None:
            # Caller-provided IDs: remap to local indices if CUDA_VISIBLE_DEVICES is set
            if _n_visible > 0:
                _visible_list = [int(g) for g in _env_gpus.split(",") if g.strip().isdigit()]
                self.gpu_ids = [
                    _visible_list.index(g) if g in _visible_list else i
                    for i, g in enumerate(gpu_ids)
                ]
            else:
                self.gpu_ids = gpu_ids
        else:
            # Default: use all visible GPUs as local indices [0, 1, ...]
            self.gpu_ids = list(range(_n_visible)) if _n_visible > 0 else [0]
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")
        self.cache_dir = cache_dir

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_failed = False  # set True on permanent failure (OOM, etc.)

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._last_metrics: dict = {}   # populated after each stream_call()

        logger.info(
            f"LocalLLMClient ready: model={self.model_id}, "
            f"gpus={self.gpu_ids}, 4bit={load_in_4bit}, 8bit={load_in_8bit}"
        )

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and tokenizer into GPU memory (called lazily on first call)."""
        if self._loaded:
            return
        if self._load_failed:
            raise RuntimeError(f"Model {self.model_id} failed to load previously — not retrying.")

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading model: {self.model_id} ...")
        start = time.perf_counter()

        token = self.hf_token or None
        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.model_id,
            "token": token,
            "trust_remote_code": True,
        }
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(**kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # GPU device map — use local (remapped) indices in max_memory
        # When CUDA_VISIBLE_DEVICES=2,3, local indices are 0,1 — not 2,3
        if self.gpu_ids:
            kwargs["max_memory"] = {local_idx: "44GiB" for local_idx in self.gpu_ids}
            kwargs["device_map"] = "auto"
        # Only set torch_dtype when NOT using quantization (bitsandbytes manages its own dtype)
        if not (self.load_in_4bit or self.load_in_8bit):
            kwargs["torch_dtype"] = torch.bfloat16

        try:
            self._model = AutoModelForCausalLM.from_pretrained(**kwargs)
        except Exception as e:
            self._load_failed = True
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
        self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info(f"Model loaded in {elapsed:.1f}s.")
        self._loaded = True

    # ── Availability ─────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True once the model is loaded (or will be lazy-loaded on first call)."""
        return not self._load_failed

    # ── Inference ─────────────────────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Run inference on a single prompt.

        Args:
            prompt:      user message content
            system:      optional system message
            max_tokens:  override max_new_tokens
            temperature: override temperature

        Returns:
            Generated text (stripped of prompt echo).
        """
        self.load()  # lazy load

        import torch

        _max = max_tokens or self.max_new_tokens
        _temp = temperature if temperature is not None else self.temperature

        formatted = _format_prompt(prompt, system, self.family, self._tokenizer)

        inputs = self._tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Move to first GPU
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)

        n_input = input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": _max,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if _temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = _temp
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode only new tokens (exclude prompt echo)
        new_tokens = output_ids[0][n_input:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self._total_input_tokens += n_input
        self._total_output_tokens += len(new_tokens)
        self._call_count += 1

        return text

    def call_batch(
        self,
        prompts: list[str],
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        assistant_prefix: str = "",
    ) -> list[str]:
        """
        Batched inference — generate for many prompts in one forward pass.

        Much higher throughput than looping call() (essential for large-scale
        data generation). Uses left padding so generation is correct for batches.
        Returns one decoded string per prompt (prompt echo stripped).

        assistant_prefix: text appended after the generation prompt to seed the
            assistant turn. For DeepSeek-R1 distills, pass "<think>\\n\\n</think>\\n\\n"
            to SKIP the long reasoning trace (much faster, avoids truncation).
        """
        self.load()
        import torch

        _max = max_tokens or self.max_new_tokens
        _temp = temperature if temperature is not None else self.temperature

        formatted = [_format_prompt(p, system, self.family, self._tokenizer) + assistant_prefix
                     for p in prompts]

        tok = self._tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        enc = tok(formatted, return_tensors="pt", padding=True)
        tok.padding_side = prev_side

        device = next(self._model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        n_input = input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": _max,
            "pad_token_id": tok.eos_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if _temp > 0:
            gen_kwargs.update(do_sample=True, temperature=_temp, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self._model.generate(input_ids, attention_mask=attn, **gen_kwargs)

        results = []
        for i in range(out.shape[0]):
            new_tokens = out[i][n_input:]
            results.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
        self._call_count += len(prompts)
        return results

    def stream_call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        assistant_prefix: str = "",
    ) -> str:
        """
        Streaming inference — prints tokens to stdout as they are generated.
        Returns the complete generated text when done.

        assistant_prefix: text seeding the assistant turn. For DeepSeek-R1 / Qwen3
            reasoning models pass "<think>\\n\\n</think>\\n\\n" to skip the long
            reasoning trace and emit a concise answer (still measures TTFT/TPOT).
        """
        from threading import Thread
        from transformers import TextIteratorStreamer

        self.load()

        import torch

        _max  = max_tokens or self.max_new_tokens
        _temp = temperature if temperature is not None else self.temperature

        formatted = _format_prompt(prompt, system, self.family, self._tokenizer) + assistant_prefix
        inputs    = self._tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        device    = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        n_input   = input_ids.shape[1]

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs: dict[str, Any] = {
            "input_ids":       input_ids,
            "attention_mask":  attn_mask,
            "max_new_tokens":  _max,
            "pad_token_id":    self._tokenizer.eos_token_id,
            "eos_token_id":    self._tokenizer.eos_token_id,
            "streamer":        streamer,
        }
        if _temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = _temp
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        t_request = time.perf_counter()
        thread.start()

        result   = ""
        t_first: Optional[float] = None
        for chunk in streamer:
            if chunk:
                if t_first is None:
                    t_first = time.perf_counter()   # first non-empty token
                print(chunk, end="", flush=True)
                result += chunk
        t_end = time.perf_counter()
        thread.join()

        n_new = len(self._tokenizer.encode(result, add_special_tokens=False))
        self._total_input_tokens  += n_input
        self._total_output_tokens += n_new
        self._call_count          += 1

        # Store last call metrics for retrieval by caller
        ttft_ms  = round((t_first - t_request) * 1000, 1) if t_first else None
        total_ms = round((t_end   - t_request) * 1000, 1)
        tpot_ms  = (
            round(((t_end - t_first) * 1000) / max(n_new - 1, 1), 2)
            if t_first and n_new > 1 else None
        )
        self._last_metrics = {
            "ttft_ms":      ttft_ms,
            "tpot_ms":      tpot_ms,
            "total_ms":     total_ms,
            "output_tokens": n_new,
            "input_tokens":  n_input,
            "throughput_tps": round(n_new / ((t_end - t_request) or 1e-9), 1),
        }
        return result.strip()

    def get_usage_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "model_id": self.model_id,
        }

    def unload(self) -> None:
        """Release model from GPU memory."""
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded from GPU memory.")
