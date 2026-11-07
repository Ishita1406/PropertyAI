"""Shared fixtures: an offline fake LLM, a fake retrieval index and a
deterministic embedding function, so agent/graph/ingestion tests need neither
a GROQ_API_KEY nor the downloaded embedding model."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chromadb.api.types import EmbeddingFunction  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class _Bound:
    def __init__(self, parent: "FakeLLM"):
        self.parent = parent

    def invoke(self, messages):
        self.parent.tool_invocations.append(list(messages))
        if self.parent.fail:
            raise RuntimeError("simulated Groq outage")
        if self.parent.tool_turns:
            return self.parent.tool_turns.pop(0)
        return AIMessage(content="DONE")


class _Structured:
    def __init__(self, parent: "FakeLLM", schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        self.parent.structured_invocations.append((self.schema.__name__, list(messages)))
        if self.parent.fail:
            raise RuntimeError("simulated Groq outage")
        queue = self.parent.structured.get(self.schema.__name__, [])
        if not queue:
            raise AssertionError(f"FakeLLM has no scripted output for {self.schema.__name__}")
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        return out(messages) if callable(out) else out


class FakeLLM:
    """Scripted stand-in for ChatGroq: ``bind_tools`` + ``with_structured_output``."""

    def __init__(self, tool_turns=None, structured=None, fail=False):
        self.tool_turns = list(tool_turns or [])
        self.structured = {k: list(v) for k, v in (structured or {}).items()}
        self.fail = fail
        self.tool_invocations: list = []
        self.structured_invocations: list = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return _Bound(self)

    def with_structured_output(self, schema):
        return _Structured(self, schema)


def tool_call_turn(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(calls)],
    )


# ---------------------------------------------------------------------------
# Fake retrieval index (stands in for ChromaDB in agent/graph tests)
# ---------------------------------------------------------------------------

FAKE_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "rera_act_2016::abc::00010",
        "source": "reraact.pdf",
        "text": "3. (1) No promoter shall advertise, market, book, sell or offer for sale any plot, apartment or building "
        "in any real estate project without registering the real estate project with the Real Estate Regulatory Authority.",
        "document_id": "rera_act_2016",
        "document_title": "The Real Estate (Regulation and Development) Act, 2016 (Act No. 16 of 2016)",
        "publisher": "Ministry of Law and Justice",
        "source_url": "https://www.indiacode.nic.in/",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act)",
        "section_number": "3",
        "page_number": 7,
        "page_end": 7,
        "effective_date": "2016-05-01",
        "version": "As enacted",
        "extraction_method": "text_layer",
    },
    {
        "chunk_id": "registration_act_1908::def::00040",
        "source": "registerationact.pdf",
        "text": "17. Documents of which registration is compulsory.—(1) ... instruments of gift of immovable property; "
        "leases of immovable property from year to year, or for any term exceeding one year.",
        "document_id": "registration_act_1908",
        "document_title": "The Registration Act, 1908 (Act No. 16 of 1908)",
        "publisher": "India Code",
        "source_url": "https://www.indiacode.nic.in/",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act)",
        "section_number": "17",
        "page_number": 12,
        "page_end": 13,
        "effective_date": "1909-01-01",
        "version": "India Code consolidated",
        "extraction_method": "text_layer",
    },
    {
        "chunk_id": "model_tenancy_act_2021::ghi::00005",
        "source": "mohua.pdf",
        "text": "11. Security deposit.—(1) The security deposit to be paid by the tenant shall not exceed two months' rent "
        "in case of residential premises.",
        "document_id": "model_tenancy_act_2021",
        "document_title": "The Model Tenancy Act, 2021",
        "publisher": "Ministry of Housing and Urban Affairs",
        "source_url": "https://mohua.gov.in/",
        "document_type": "model_legislation",
        "source_type": "official_government",
        "binding_status": "model_law_not_binding_unless_adopted_by_state",
        "jurisdiction": "Model law",
        "section_number": "11",
        "page_number": 9,
        "page_end": 9,
        "effective_date": "2021-06-02",
        "version": "2021",
        "extraction_method": "text_layer",
    },
    {
        "chunk_id": "sim_hinjewadi_bullish_analyst_note::x::0",
        "source": "hinjewadi_bullish_analyst_note.txt",
        "text": "SYNTHETIC: Hinjewadi Phase 2 demand is strong, appreciation expected to stay high.",
        "document_id": "sim_hinjewadi_bullish_analyst_note",
        "document_title": "[SIMULATED] Hinjewadi Bullish Analyst Note",
        "publisher": "project",
        "source_url": "",
        "document_type": "simulated_market_commentary",
        "source_type": "simulated",
        "binding_status": "simulated_not_evidence",
        "jurisdiction": "Pune — simulated",
        "section_number": "not_identified",
        "page_number": 1,
        "page_end": 1,
        "effective_date": "not_applicable",
        "version": "mock",
        "extraction_method": "text_layer",
    },
    {
        "chunk_id": "sim_hinjewadi_bearish_analyst_note::y::0",
        "source": "hinjewadi_bearish_analyst_note.txt",
        "text": "SYNTHETIC: Hinjewadi Phase 2 faces oversupply; prices may stagnate.",
        "document_id": "sim_hinjewadi_bearish_analyst_note",
        "document_title": "[SIMULATED] Hinjewadi Bearish Analyst Note",
        "publisher": "project",
        "source_url": "",
        "document_type": "simulated_market_commentary",
        "source_type": "simulated",
        "binding_status": "simulated_not_evidence",
        "jurisdiction": "Pune — simulated",
        "section_number": "not_identified",
        "page_number": 1,
        "page_end": 1,
        "effective_date": "not_applicable",
        "version": "mock",
        "extraction_method": "text_layer",
    },
]


def fake_query_rag(query, k=4, source_type=None, document_type=None, document_ids=None, where=None):
    def ok(value, flt):
        if flt is None:
            return True
        return value in flt if isinstance(flt, (list, tuple, set)) else value == flt

    words = set(query.lower().split())
    hits = [
        c for c in FAKE_CHUNKS
        if ok(c["source_type"], source_type) and ok(c["document_type"], document_type)
        and words & set(c["text"].lower().split())
    ]
    hits.sort(key=lambda c: -len(words & set(c["text"].lower().split())))
    return [dict(h) for h in hits[:k]]


@pytest.fixture
def fake_rag(monkeypatch):
    import property_advisor.agents.research_agent as ra

    monkeypatch.setattr(ra, "query_rag", fake_query_rag)
    return fake_query_rag


@pytest.fixture
def patch_llm(monkeypatch):
    """Install a FakeLLM (or a factory raising an error) for both agents."""
    import property_advisor.agents.investment_analyst_agent as ia
    import property_advisor.agents.research_agent as ra
    import property_advisor.agents._llm_utils as utils

    def install(research_llm=None, analyst_llm=None, fail_with: Exception | None = None):
        def factory_for(llm):
            def get_llm(*args, **kwargs):
                if fail_with is not None:
                    raise fail_with
                return llm

            return get_llm

        monkeypatch.setattr(ra, "get_llm", factory_for(research_llm))
        monkeypatch.setattr(ia, "get_llm", factory_for(analyst_llm))
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)

    return install


# ---------------------------------------------------------------------------
# Deterministic embedding function for ChromaDB ingestion tests
# ---------------------------------------------------------------------------


class HashEmbedding(EmbeddingFunction):
    """Bag-of-words hashing embedding — deterministic and offline."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, input):  # noqa: A002 (chroma's protocol name)
        out = []
        for text in input:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in text.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out

    @staticmethod
    def name() -> str:
        return "test-hash-embedding"

    def get_config(self):
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding(config.get("dim", 128))
