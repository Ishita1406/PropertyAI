"""API contract test: /analyze -> /reject -> /approve keeps the response shape
the frontend relies on (offline, fake LLM + fake retrieval)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
from tests.test_graph import _rec, _research_llm  # noqa: E402
from tests.conftest import FakeLLM  # noqa: E402


def test_analyze_reject_approve_contract(fake_rag, patch_llm, tmp_path, monkeypatch):
    patch_llm(research_llm=_research_llm(), analyst_llm=FakeLLM(structured={"RecommendationOutput": [_rec(), _rec("HOLD", "Reviewer feedback applied.")]}))
    monkeypatch.setattr(app_module, "REPORTS_DIR", tmp_path)
    client = TestClient(app_module.app)

    r = client.post("/analyze", json={"address": "Whitefield, Bangalore", "budget": 9500000, "horizon": 5, "strategy": "rental"})
    assert r.status_code == 200
    data = r.json()
    assert data["approval_required"] is True
    for key in ("thread_id", "recommendation", "investment_metrics", "risk_assessment", "guardrail_result",
                "research_findings", "data_disclaimer"):
        assert key in data
    tid = data["thread_id"]

    r = client.post("/reject", json={"thread_id": tid, "feedback": "be more cautious"})
    assert r.status_code == 200 and r.json()["approval_required"] is True
    assert r.json()["recommendation"]["decision"] == "HOLD"

    r = client.post("/approve", json={"thread_id": tid})
    assert r.status_code == 200
    body = r.json()
    assert body["final_report"]["status"] == "approved"
    assert set(body["report_paths"]) == {"json", "markdown", "pdf"}

    assert client.post("/analyze", json={"address": "x", "budget": 1, "strategy": "bogus"}).status_code == 400
