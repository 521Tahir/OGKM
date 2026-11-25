"""
Prompt template management.

Loads prompt templates from the prompts/ directory and renders them
with variable substitution.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..utils.logging_utils import get_logger

logger = get_logger("llm.prompt_manager")

_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


class PromptManager:
    """Load and render prompt templates from disk."""

    def __init__(self, prompts_root: str | Path = "prompts"):
        self.prompts_root = Path(prompts_root)

    def load_template(self, category: str, name: str) -> str:
        """Load a prompt template file. E.g. load_template('concept_extraction', 'cot')."""
        path = self.prompts_root / category / f"{name}.txt"
        if not path.exists():
            # Try .md extension
            path = self.prompts_root / category / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Replace {{variable}} placeholders in template with values."""
        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in variables:
                logger.warning(f"Prompt variable '{key}' not provided; left as-is.")
                return match.group(0)
            return str(variables[key])
        return _VAR_RE.sub(replace, template)

    def render_template(
        self, category: str, name: str, variables: dict[str, Any]
    ) -> str:
        """Load and render a template in one call."""
        template = self.load_template(category, name)
        return self.render(template, variables)

    def list_templates(self) -> dict[str, list[str]]:
        """Return all available templates grouped by category."""
        result: dict[str, list[str]] = {}
        for cat_dir in sorted(self.prompts_root.iterdir()):
            if cat_dir.is_dir():
                files = [f.stem for f in sorted(cat_dir.glob("*.txt")) + sorted(cat_dir.glob("*.md"))]
                if files:
                    result[cat_dir.name] = files
        return result
