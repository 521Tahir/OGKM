"""Anthropic Claude API client (extracted from original client.py)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.anthropic_client")


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API with retry logic."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

        try:
            import anthropic as _anthropic
            self._sdk = _anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install the 'anthropic' package: pip install anthropic")

    def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        _max = max_tokens or self.max_tokens
        _temp = temperature if temperature is not None else self.temperature
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": _max,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                if _temp is not None:
                    kwargs["temperature"] = _temp

                resp = self._sdk.messages.create(**kwargs)
                text = resp.content[0].text if resp.content else ""
                if hasattr(resp, "usage"):
                    self._total_input_tokens += resp.usage.input_tokens
                    self._total_output_tokens += resp.usage.output_tokens
                self._call_count += 1
                return text

            except Exception as e:
                last_error = e
                logger.warning(f"Anthropic attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"Anthropic API failed after {self.max_retries} retries: {last_error}")

    def get_usage_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "model": self.model,
        }
