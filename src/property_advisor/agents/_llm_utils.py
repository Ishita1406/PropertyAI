"""Shared helpers for the two LLM agents."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from property_advisor.config import MAX_LLM_RETRIES

logger = logging.getLogger("property_advisor")
T = TypeVar("T")


def invoke_with_retries(fn: Callable[[], T], label: str, attempts: int | None = None, backoff_s: float = 1.0) -> T:
    """Call ``fn`` with a bounded number of retries (1 + MAX_LLM_RETRIES).

    Missing-API-key errors are not retried (they cannot succeed).
    """
    total = 1 + (MAX_LLM_RETRIES if attempts is None else attempts)
    last_exc: Exception | None = None
    for i in range(total):
        try:
            return fn()
        except RuntimeError as exc:
            if "GROQ_API_KEY" in str(exc):
                raise
            last_exc = exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        logger.warning("%s attempt %d/%d failed: %s", label, i + 1, total, last_exc)
        if i < total - 1 and backoff_s:
            time.sleep(backoff_s * (i + 1))
    assert last_exc is not None
    raise last_exc
