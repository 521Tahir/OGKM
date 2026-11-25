"""
Parse structured JSON from LLM text outputs.

Handles common LLM output formats:
- Pure JSON
- JSON inside markdown code blocks
- Malformed JSON requiring repair
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..utils.logging_utils import get_logger

logger = get_logger("llm.parsers")

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
# Match individual JSON objects in a (possibly truncated) array
_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class JSONOutputParser:
    """Parse JSON from LLM text output with fallback repair strategies."""

    def parse(self, text: str) -> tuple[Any, bool, str]:
        """
        Attempt to parse JSON from text.

        Returns:
            (parsed_object, success, error_message)
        """
        # Strategy 1: direct JSON parse
        stripped = text.strip()
        try:
            return json.loads(stripped), True, ""
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from code block
        match = _CODE_BLOCK_RE.search(stripped)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate), True, ""
            except json.JSONDecodeError:
                pass

        # Strategy 3: find first { or [ and try from there
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = stripped.find(start_char)
            if idx != -1:
                candidate = stripped[idx:]
                try:
                    return json.loads(candidate), True, ""
                except json.JSONDecodeError:
                    # Strategy 4: repair trailing commas
                    repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
                    try:
                        return json.loads(repaired), True, ""
                    except json.JSONDecodeError:
                        pass

        # Strategy 5: extract individual objects from a truncated array
        objects = _OBJECT_RE.findall(stripped)
        if objects:
            parsed_objs = []
            for o in objects:
                try:
                    parsed_objs.append(json.loads(o))
                except json.JSONDecodeError:
                    pass
            if parsed_objs:
                return parsed_objs, True, ""

        error_msg = f"Failed to parse JSON from output (length={len(text)})"
        logger.warning(error_msg)
        return None, False, error_msg

    def parse_list(self, text: str) -> tuple[list, bool, str]:
        """Parse and ensure the result is a list."""
        obj, ok, err = self.parse(text)
        if not ok:
            return [], False, err
        if isinstance(obj, list):
            return obj, True, ""
        if isinstance(obj, dict):
            # Common LLM pattern: {"concepts": [...]}
            for key in ("concepts", "relations", "rules", "items", "results", "data"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key], True, ""
            return [obj], True, ""
        return [], False, f"Expected list, got {type(obj).__name__}"
