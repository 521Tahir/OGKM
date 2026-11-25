"""
Batch LLM call runner with checkpointing, retry, and progress tracking.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .client import LLMClient
from ..utils.logging_utils import get_logger
from ..utils.io import save_json, load_json, ensure_dir
from ..utils.misc import chunk_list

logger = get_logger("llm.batch_runner")


class BatchRunner:
    """
    Run a prompt-generating function over a list of items with:
    - batching
    - checkpointing (resume after crash)
    - progress logging
    - error collection
    """

    def __init__(
        self,
        client: LLMClient,
        checkpoint_dir: str | Path = "artifacts/runs/checkpoints",
        batch_size: int = 10,
        delay_between_batches: float = 1.0,
    ):
        self.client = client
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        self.delay = delay_between_batches

    def run(
        self,
        items: list[Any],
        prompt_fn: Callable[[Any], str],
        job_name: str = "batch_job",
        system_prompt: str = "",
        resume: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """
        Process items with LLM calls.

        Args:
            items: list of items to process
            prompt_fn: function that converts an item to a prompt string
            job_name: unique identifier for checkpointing
            system_prompt: optional system message
            resume: if True, skip already-processed items

        Returns:
            (results, errors) where each is a list of dicts
        """
        ensure_dir(self.checkpoint_dir)
        checkpoint_path = self.checkpoint_dir / f"{job_name}_checkpoint.json"
        results_path = self.checkpoint_dir / f"{job_name}_results.json"
        errors_path = self.checkpoint_dir / f"{job_name}_errors.json"

        # Load checkpoint
        completed_ids: set = set()
        results: list[dict] = []
        errors: list[dict] = []

        if resume and checkpoint_path.exists():
            checkpoint = load_json(checkpoint_path)
            completed_ids = set(checkpoint.get("completed_ids", []))
            if results_path.exists():
                results = load_json(results_path)
            if errors_path.exists():
                errors = load_json(errors_path)
            logger.info(f"Resuming {job_name}: {len(completed_ids)} already completed.")

        pending = [item for i, item in enumerate(items) if i not in completed_ids]
        logger.info(f"Starting {job_name}: {len(pending)} pending of {len(items)} total.")

        for batch_idx, batch in enumerate(chunk_list(pending, self.batch_size)):
            for item in batch:
                item_id = getattr(item, "record_id", id(item))
                try:
                    prompt = prompt_fn(item)
                    response = self.client.call(prompt, system=system_prompt)
                    results.append({"item_id": item_id, "response": response, "success": True})
                    completed_ids.add(item_id)
                except Exception as e:
                    logger.error(f"Error processing item {item_id}: {e}")
                    errors.append({"item_id": item_id, "error": str(e)})

            # Save checkpoint after each batch
            save_json({"completed_ids": list(completed_ids)}, checkpoint_path)
            save_json(results, results_path)
            save_json(errors, errors_path)
            logger.info(
                f"Batch {batch_idx + 1} done. "
                f"Total: {len(results)} success, {len(errors)} errors."
            )
            if self.delay > 0:
                time.sleep(self.delay)

        logger.info(f"BatchRunner {job_name} complete: {len(results)} results, {len(errors)} errors.")
        return results, errors
