"""
Unified LLM client factory.

Priority order for backend selection:
  1. LOCAL  — if LOCAL_MODEL_KEY or LOCAL_MODEL_ID is set in env or passed explicitly
  2. ANTHROPIC — if ANTHROPIC_API_KEY is set
  3. OFFLINE — no LLM (rule-based fallback inside OntologyBuilder)

Usage:
    # Auto-detect best backend
    client = LLMClient()

    # Force local Llama 3 8B
    client = LLMClient(backend="local", model_key="llama3-8b")

    # Force Anthropic Claude
    client = LLMClient(backend="anthropic")

    # Direct call interface (same for all backends)
    response = client.call("What are the best beaches in Bali?")
"""
from __future__ import annotations

import os
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.client")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_LOCAL_MODEL_KEY = "llama3.1-8b"


class LLMClient:
    """
    Provider-agnostic LLM client.

    Wraps either:
    - LocalLLMClient  (HuggingFace local model — Llama 3, Qwen, Mistral, …)
    - AnthropicClient (Claude API)
    """

    def __init__(
        self,
        backend: str = "auto",       # "auto" | "local" | "anthropic"
        # Local model settings
        model_key: str = DEFAULT_LOCAL_MODEL_KEY,
        model_id: Optional[str] = None,
        gpu_ids: list[int] | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        hf_token: Optional[str] = None,
        # Anthropic settings
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
        anthropic_api_key: Optional[str] = None,
        # Shared settings
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._backend_name = backend
        self._client = None

        # Resolve backend
        resolved = self._resolve_backend(backend, model_id, model_key, anthropic_api_key, hf_token)

        if resolved == "local":
            # Prefer env var over default if caller didn't explicitly pass a non-default model_key
            env_model_key = os.getenv("LOCAL_MODEL_KEY")
            if env_model_key and model_key == DEFAULT_LOCAL_MODEL_KEY:
                model_key = env_model_key
            from .local_client import LocalLLMClient
            # Parse CUDA_VISIBLE_DEVICES from env for default GPU list
            import os as _os
            _env_gpus = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _default_gpus = (
                [int(g) for g in _env_gpus.split(",") if g.strip().isdigit()]
                if _env_gpus else [2, 3]
            )
            self._client = LocalLLMClient(
                model_key=model_key,
                model_id=model_id,
                gpu_ids=gpu_ids or _default_gpus,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                max_new_tokens=max_tokens,
                temperature=temperature,
                hf_token=hf_token,
            )
            self._backend_name = "local"
            logger.info(
                f"LLMClient → LOCAL backend "
                f"({self._client.model_id})"
            )

        elif resolved == "anthropic":
            from .anthropic_client import AnthropicClient
            self._client = AnthropicClient(
                model=anthropic_model,
                api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", ""),
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )
            self._backend_name = "anthropic"
            logger.info(f"LLMClient → ANTHROPIC backend ({anthropic_model})")

        else:
            self._backend_name = "none"
            logger.warning(
                "LLMClient: no backend available. "
                "Set ANTHROPIC_API_KEY or HF_TOKEN + LOCAL_MODEL_KEY."
            )

    def _resolve_backend(
        self,
        backend: str,
        model_id: Optional[str],
        model_key: str,
        anthropic_key: Optional[str],
        hf_token: Optional[str],
    ) -> str:
        if backend == "local":
            return "local"
        if backend == "anthropic":
            return "anthropic"

        # Auto-detect
        has_local = bool(
            model_id
            or os.getenv("LOCAL_MODEL_KEY")
            or os.getenv("LOCAL_MODEL_ID")
        )
        has_anthropic = bool(anthropic_key or os.getenv("ANTHROPIC_API_KEY"))

        if has_local:
            return "local"
        if has_anthropic:
            return "anthropic"
        return "none"

    def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run a single inference call. Interface is identical across all backends."""
        if self._client is None:
            raise RuntimeError(
                "No LLM backend configured. "
                "Set ANTHROPIC_API_KEY for Claude, or HF_TOKEN + LOCAL_MODEL_KEY for local Llama."
            )
        return self._client.call(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )

    def stream_call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Streaming inference — prints tokens to stdout as they arrive.
        Falls back to regular call() if the backend does not support streaming.
        """
        if self._client is None:
            raise RuntimeError("No LLM backend configured.")
        if hasattr(self._client, "stream_call"):
            return self._client.stream_call(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            )
        # Fallback: non-streaming (Anthropic, etc.)
        result = self.call(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        print(result)
        return result

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def get_last_metrics(self) -> dict:
        """Return TTFT / TPOT / throughput from the most recent stream_call()."""
        if self._client is None:
            return {}
        return getattr(self._client, "_last_metrics", {})

    def get_usage_stats(self) -> dict:
        if self._client is None:
            return {}
        return self._client.get_usage_stats()
