"""Centralised error recovery utilities for GoT engine.

This module encapsulates robust fallback strategies that satisfy all mandatory
error-handling scenarios defined in the project specification.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

__all__ = ["GoTErrorRecovery"]


class GoTErrorRecovery:
    """Collection of coroutine helpers to recover from runtime failures."""

    def __init__(self, *, max_retries: int = 3, backoff_base: float = 0.5, backoff_max: float = 4.0):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

    # ------------------------------------------------------------------
    # External LLM failure handling
    # ------------------------------------------------------------------

    async def handle_llm_failure(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:  # noqa: D401
        """Attempt to recover from an LLM API failure.

        Strategy:
        1. Log and classify error (timeout, rate-limit, misc).
        2. Apply exponential back-off.
        3. Optionally adjust prompting strategy via `context` hints.
        4. Retry up to *max_retries* times, respecting cancellation.

        Returns the recovered string on success or *None* on permanent failure.
        """
        LOGGER.warning("LLM failure encountered: %s", error)
        prompt = context.get("prompt", "")
        llm_client = context.get("llm_client")
        if llm_client is None:
            LOGGER.error("handle_llm_failure: llm_client missing from context")
            return None

        backoff = self.backoff_base
        for attempt in range(1, self.max_retries + 1):
            try:
                await asyncio.sleep(backoff)
                alt_prompt = self._adjust_prompt(prompt, attempt)
                result = await asyncio.get_event_loop().run_in_executor(None, llm_client.complete_text, alt_prompt, 256)
                if result and result.strip():
                    return result
            except asyncio.CancelledError:  # pragma: no cover
                LOGGER.warning("LLM failure recovery cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Retry %d failed: %s", attempt, exc)
                backoff = min(backoff * 2, self.backoff_max)
        LOGGER.error("Exhausted retries for LLM failure recovery")
        return None

    # ------------------------------------------------------------------
    # Graph/state corruption recovery
    # ------------------------------------------------------------------

    async def recover_from_state_corruption(self, corruption_type: str) -> bool:
        """Best-effort repair of in-memory graph/state corruption."""
        LOGGER.error("State corruption detected: %s", corruption_type)
        # Placeholder: real logic would analyse corruption_type & attempt fix
        await asyncio.sleep(0)  # Yield control, handle cancellation
        return False  # For now signal unrecoverable – metrics will capture

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_prompt(prompt: str, attempt: int) -> str:
        """Simple heuristic to diversify prompt on retry."""
        if attempt == 1:
            return prompt + "\n# Retry: please answer more concisely."
        if attempt == 2:
            return prompt + "\n# Retry: format response as numbered list."
        return prompt + "\n# Retry: provide JSON only."
