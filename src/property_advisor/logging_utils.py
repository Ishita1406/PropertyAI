"""Structured per-agent logging + LangSmith tracing.

Every agent must log agent_name, an input summary, an output summary, and
execution_time. Logs are
written as JSON lines to ./logs/project.log and mirrored to the console.

Each agent is also wrapped with LangSmith's @traceable decorator so it shows
up as its own named span when tracing is enabled (in addition to whatever
LangGraph's own runnable tracing captures for the graph as a whole). If
LANGSMITH_TRACING isn't enabled (no key configured), @traceable is a no-op —
tracing degrades gracefully rather than failing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from langsmith import traceable

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "project.log")

logger = logging.getLogger("property_advisor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s - %(message)s"))
    logger.addHandler(handler)


def _write_record(record: dict[str, Any]) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _summarize(value: Any, max_str_len: int = 160) -> Any:
    """Lightweight one-level-deep summary so logs stay readable instead of
    dumping entire nested states (rag_context, property_data, ...) every call.
    """
    if isinstance(value, dict):
        return {k: _summarize_value(v, max_str_len) for k, v in value.items()}
    return _summarize_value(value, max_str_len)


def _summarize_value(value: Any, max_str_len: int) -> Any:
    if isinstance(value, dict):
        return f"{{...{len(value)} keys}}" if value else "{}"
    if isinstance(value, list):
        return f"[...{len(value)} items]" if value else "[]"
    if isinstance(value, str) and len(value) > max_str_len:
        return value[:max_str_len] + "..."
    return value


@contextmanager
def log_agent_call(agent_name: str, input_state: dict[str, Any]):
    """Context manager that times an agent node and logs a structured record.

    Usage:
        with log_agent_call("property_agent", state.model_dump()) as finish:
            ...
            finish(output_state)
    """
    start = time.perf_counter()
    captured: dict[str, Any] = {}

    def finish(output_state: dict[str, Any]) -> None:
        captured["output_state"] = output_state

    yield finish

    execution_time = time.perf_counter() - start
    record = {
        "agent_name": agent_name,
        "input_summary": _summarize(input_state),
        "output_summary": _summarize(captured.get("output_state", {})),
        "execution_time_seconds": round(execution_time, 4),
    }
    logger.info("%s completed in %.3fs", agent_name, execution_time)
    _write_record(record)


def traced_router(router_name: str) -> Callable:
    """Decorator for conditional-edge routing functions: logs the routing
    decision and traces it as its own named LangSmith span, so "why did the
    graph go this way" is answerable from the trace alone, not just from
    which node ran next.
    """

    def decorator(fn: Callable) -> Callable:
        traced_fn = traceable(name=router_name, run_type="chain")(fn)

        def wrapped(state):
            decision = traced_fn(state)
            logger.info("routing[%s] -> %s", router_name, decision)
            _write_record({"router_name": router_name, "decision": decision})
            return decision

        wrapped.__name__ = fn.__name__
        return wrapped

    return decorator


def timed_node(agent_name: str) -> Callable:
    """Decorator that wraps a LangGraph node function with structured
    logging AND LangSmith tracing (named span = agent_name)."""

    def decorator(fn: Callable) -> Callable:
        traced_fn = traceable(name=agent_name, run_type="chain")(fn)

        def wrapped(state):
            input_state = state.model_dump()
            with log_agent_call(agent_name, input_state) as finish:
                update = traced_fn(state)
                finish(update if isinstance(update, dict) else {})
            return update

        wrapped.__name__ = fn.__name__
        return wrapped

    return decorator
