"""Environment-driven configuration. Groq is the only supported LLM provider."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# LangSmith tracing. Supports both the modern LANGSMITH_* env vars and the
# legacy LANGCHAIN_* aliases (LANGSMITH_* wins if both are set). Tracing is
# enabled only if explicitly turned on AND an API key is present — otherwise
# it's a no-op, never a crash.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "property-investment-advisor")
_tracing_requested = (
    os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "false"
).lower() == "true"
LANGSMITH_TRACING_ENABLED = bool(_tracing_requested and LANGSMITH_API_KEY)

# kept for any code/tools that still read the legacy names directly
LANGCHAIN_TRACING_V2 = "true" if LANGSMITH_TRACING_ENABLED else "false"
LANGCHAIN_API_KEY = LANGSMITH_API_KEY
LANGCHAIN_PROJECT = LANGSMITH_PROJECT

RISK_HUMAN_REVIEW_THRESHOLD = float(os.getenv("RISK_HUMAN_REVIEW_THRESHOLD", "75"))
GUARDRAIL_CONFIDENCE_THRESHOLD = float(os.getenv("GUARDRAIL_CONFIDENCE_THRESHOLD", "0.70"))

MAX_DATA_RETRIES = int(os.getenv("MAX_DATA_RETRIES", "2"))
MAX_REANALYSIS_RETRIES = int(os.getenv("MAX_REANALYSIS_RETRIES", "2"))

AUTO_APPROVE_HUMAN_REVIEW = os.getenv("AUTO_APPROVE_HUMAN_REVIEW", "false").lower() == "true"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Propagate the resolved tracing config back into the environment so both
# LangGraph/LangChain's internal tracer and the langsmith SDK's @traceable
# decorator (used directly on each agent — see logging_utils.py) pick it up
# consistently, regardless of which env var naming convention was used.
if LANGSMITH_TRACING_ENABLED:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
else:
    # Explicitly disable rather than leaving stale/partial env vars from a
    # previous run that could otherwise half-enable tracing.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
# Only applied to reasoning models (openai/gpt-oss-*).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_REASONING_TOKEN_HEADROOM = int(os.getenv("LLM_REASONING_TOKEN_HEADROOM", "3000"))

# Bounded retries for each LLM agent call (transient Groq/API/parse errors).
MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "2"))
# Max tool-calling rounds the Research & Due Diligence Agent may use.
RESEARCH_MAX_TOOL_ROUNDS = int(os.getenv("RESEARCH_MAX_TOOL_ROUNDS", "4"))

# --- RAG paths -----------------------------------------------------------
# Source documents live in data/ (read-only). Everything generated from them
# (ChromaDB files, extracted/OCR text, ingestion manifest) lives elsewhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_SOURCE_DIR = os.getenv("KB_SOURCE_DIR", os.path.join(PROJECT_ROOT, "data"))
KB_PROCESSED_DIR = os.getenv("KB_PROCESSED_DIR", os.path.join(PROJECT_ROOT, "data_processed"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJECT_ROOT, "chroma_db"))
KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "india_property_knowledge_base")
KB_ENABLE_OCR = os.getenv("KB_ENABLE_OCR", "true").lower() == "true"


def get_llm(temperature: float = 0.2, max_tokens: int = DEFAULT_MAX_TOKENS):
    """Return a Groq chat model. Raises clearly if no API key is configured.

    Defaults are deliberately tight (low temperature, capped output tokens):
    every call here produces short structured JSON, not prose, so there's no
    reason to pay for long completions.
    """
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )
    extra: dict = {}
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        # Reasoning model: hidden reasoning tokens count against max_tokens, so
        # without headroom the structured JSON gets truncated mid-output
        # ("Failed to parse tool call arguments as JSON").
        extra["reasoning_effort"] = LLM_REASONING_EFFORT
        max_tokens = max_tokens + LLM_REASONING_TOKEN_HEADROOM
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY,
        **extra,
    )
