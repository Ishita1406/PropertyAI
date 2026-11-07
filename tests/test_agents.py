"""Tests for the two LLM agents with a scripted fake LLM (no API key needed)."""

from property_advisor.agents.investment_analyst_agent import investment_analyst_agent
from property_advisor.agents.research_agent import research_agent
from property_advisor.nodes import compute_financials_node, fetch_property_data_node
from property_advisor.schemas import (
    EvidenceGap,
    PropertyConcern,
    RecommendationOutput,
    RegulatoryRequirement,
    ResearchFindingsOutput,
)
from property_advisor.state import PropertyState

from tests.conftest import FakeLLM, tool_call_turn


def _prepared_state(address="Whitefield, Bangalore, 560066", budget=9_500_000, strategy="rental") -> PropertyState:
    s = PropertyState(property_address=address, budget=budget, investment_strategy=strategy)
    s = s.model_copy(update=fetch_property_data_node(s))
    s = s.model_copy(update=compute_financials_node(s))
    return s


def _findings(**kw) -> ResearchFindingsOutput:
    base = dict(
        summary="Simulated Whitefield flat; RERA registration and sale-deed registration must be verified.",
        regulatory_requirements=[
            RegulatoryRequirement(requirement="Project must be RERA-registered before sale.", relevance="Buyer protection.", citation_ids=["E2"]),
            RegulatoryRequirement(requirement="Invented requirement.", relevance="x", citation_ids=["E99"]),
        ],
        concerns=[PropertyConcern(concern="Leases over one year need registration.", severity="medium", basis="official_document", citation_ids=["E3"])],
        evidence_gaps=[EvidenceGap(gap="Karnataka RERA rules absent.", why_it_matters="State rules govern.")],
        conflicting_evidence=[],
        due_diligence_checklist=["Verify RERA number on the State portal"],
    )
    base.update(kw)
    return ResearchFindingsOutput(**base)


# ---------------------------------------------------------------------------
# Agent 1
# ---------------------------------------------------------------------------


def test_research_agent_plans_searches_and_returns_structured_cited_findings(fake_rag, patch_llm):
    llm = FakeLLM(
        tool_turns=[
            tool_call_turn(
                ("search_official_documents", {"query": "promoter registration real estate project", "document_type": "primary_legislation"}),
                ("search_official_documents", {"query": "registration compulsory lease immovable property"}),
            ),
            tool_call_turn(("search_official_documents", {"query": "security deposit tenant rent", "document_type": "model_legislation"})),
        ],
        structured={"ResearchFindingsOutput": [_findings()]},
    )
    patch_llm(research_llm=llm)
    state = _prepared_state()
    out = research_agent(state)
    rf = out["research_findings"]

    assert rf["status"] == "complete"
    # the LLM's own searches were executed and logged
    queries = [s["query"] for s in rf["retrieval_log"] if s["tool"] == "search_official_documents"]
    assert len(queries) == 3
    # evidence carries full metadata and citations
    ev = {e["evidence_id"]: e for e in rf["evidence"]}
    for e in ev.values():
        for key in ("document_title", "publisher", "source_url", "document_type", "jurisdiction", "section_number",
                    "page_number", "effective_date", "version", "source_type", "document_id"):
            assert key in e
    # requirement with a fabricated citation is removed; the valid one keeps a resolved citation
    reqs = rf["regulatory_requirements"]
    assert len(reqs) == 1 and reqs[0]["citation_ids"] == ["E2"]
    assert "Registration" in reqs[0]["citations"][0] or "Real Estate" in reqs[0]["citations"][0]
    assert rf["unsupported_statements"][0]["statement"] == "Invented requirement."
    assert "E99" in rf["unverifiable_citations"]
    # simulated vs official distinguished
    assert {e["source_type"] for e in rf["evidence"]} >= {"official_government"}
    # structural gaps always reported (missing TPA / Stamp Act)
    gaps = " ".join(g["gap"] for g in rf["evidence_gaps"])
    assert "Transfer of Property Act" in gaps and "Stamp Act" in gaps
    assert out["rag_context"] and all("evidence_id" in c for c in out["rag_context"])


def test_research_agent_binding_status_comes_from_metadata_not_llm(fake_rag, patch_llm):
    llm = FakeLLM(
        tool_turns=[tool_call_turn(("search_official_documents", {"query": "security deposit tenant rent", "document_type": "model_legislation"}))],
        structured={"ResearchFindingsOutput": [lambda msgs: _findings(regulatory_requirements=[
            RegulatoryRequirement(requirement="Security deposit capped at two months' rent.", relevance="Rental strategy.", citation_ids=["E1"])
        ])]},
    )
    patch_llm(research_llm=llm)
    rf = research_agent(_prepared_state())["research_findings"]
    req = rf["regulatory_requirements"][0]
    assert req["binding_status"] == ["model_law_not_binding_unless_adopted_by_state"]


def test_research_agent_falls_back_to_baseline_when_llm_skips_official_search(fake_rag, patch_llm):
    llm = FakeLLM(tool_turns=[], structured={"ResearchFindingsOutput": [_findings(regulatory_requirements=[])]})
    patch_llm(research_llm=llm)
    rf = research_agent(_prepared_state())["research_findings"]
    assert any(s["tool"] == "baseline_fallback" for s in rf["retrieval_log"])
    assert any(e["source_type"] == "official_government" for e in rf["evidence"])


def test_research_agent_degrades_gracefully_when_llm_unavailable(fake_rag, patch_llm):
    patch_llm(research_llm=FakeLLM(fail=True))
    out = research_agent(_prepared_state())
    rf = out["research_findings"]
    assert rf["status"] == "degraded_llm_unavailable"
    assert rf["regulatory_requirements"] == []
    assert rf["evidence"]  # baseline retrieval still ran
    assert any("unavailable" in e for e in out["errors"])


def test_research_agent_reports_missing_data_and_simulated_conflicts(fake_rag, patch_llm):
    patch_llm(research_llm=FakeLLM(fail=True))
    # Hinjewadi: both simulated analyst notes are retrieved by the locality baseline search
    rf = research_agent(_prepared_state("Hinjewadi Phase 2, Pune", 7_200_000))["research_findings"]
    assert any("hinjewadi_bullish" in c and "hinjewadi_bearish" in c for c in rf["conflicting_evidence"])
    # Muzaffarpur: no simulated market data
    rf2 = research_agent(_prepared_state("Plot near Saraiyaganj, Muzaffarpur, Bihar", 3_500_000, "long_term_appreciation"))["research_findings"]
    assert any("market trend data" in g["gap"] for g in rf2["evidence_gaps"])


# ---------------------------------------------------------------------------
# Agent 2
# ---------------------------------------------------------------------------


def _with_research(state: PropertyState, fake_rag, patch_llm) -> PropertyState:
    llm = FakeLLM(
        tool_turns=[tool_call_turn(("search_official_documents", {"query": "promoter registration real estate project"}))],
        structured={"ResearchFindingsOutput": [_findings()]},
    )
    patch_llm(research_llm=llm)
    return state.model_copy(update=research_agent(state))


def test_investment_analyst_produces_structured_recommendation(fake_rag, patch_llm):
    state = _with_research(_prepared_state(), fake_rag, patch_llm)
    rec_out = RecommendationOutput(
        decision="BUY",
        justification="Strong simulated appreciation and positive operating cash flow.",
        supporting_evidence=["roi_pct=%s" % state.investment_metrics["roi_pct"]],
        confidence_score=0.8,
        key_risks=["Leveraged cash flow is negative"],
        assumptions=["Simulated rent holds"],
        cited_evidence_ids=["E2"],
    )
    analyst = FakeLLM(structured={"RecommendationOutput": [rec_out]})
    patch_llm(research_llm=FakeLLM(), analyst_llm=analyst)
    rec = investment_analyst_agent(state)["recommendation"]
    assert rec["decision"] == "BUY"
    assert rec["citations"][0]["evidence_id"] == "E2"
    assert rec["financial_assumptions"] and "SIMULATED" in rec["financial_assumptions"][0]
    # The prompt fed to the LLM contains the deterministic metrics and Agent 1's findings
    prompt = analyst.structured_invocations[0][1][1].content
    assert str(state.investment_metrics["roi_pct"]) in prompt
    assert "RERA-registered" in prompt
    assert "Stress scenarios" in prompt


def test_investment_analyst_safety_net_overrides_buy_on_significantly_negative_operating_cash_flow(patch_llm):
    state = PropertyState(
        property_address="X", budget=1e7, investment_strategy="rental",
        investment_metrics={"cash_flow_severity": "significantly_negative", "strong_appreciation_evidence": False,
                            "negative_cash_flow": True, "assumptions": []},
        research_findings={"status": "complete", "evidence": []},
    )
    rec_out = RecommendationOutput(decision="BUY", justification="Looks fine.", confidence_score=0.9)
    patch_llm(analyst_llm=FakeLLM(structured={"RecommendationOutput": [rec_out]}))
    rec = investment_analyst_agent(state)["recommendation"]
    assert rec["decision"] == "HOLD"
    assert rec["confidence_score"] <= 0.55
    assert "Guardrail override" in rec["justification"]


def test_investment_analyst_caps_confidence_when_research_degraded(patch_llm):
    state = PropertyState(property_address="X", budget=1e7, research_findings={"status": "degraded_llm_unavailable", "evidence": []})
    patch_llm(analyst_llm=FakeLLM(structured={"RecommendationOutput": [RecommendationOutput(decision="HOLD", justification="ok", confidence_score=0.95)]}))
    rec = investment_analyst_agent(state)["recommendation"]
    assert rec["confidence_score"] == 0.6


def test_investment_analyst_failure_returns_error_without_decision(patch_llm):
    patch_llm(analyst_llm=FakeLLM(fail=True))
    out = investment_analyst_agent(PropertyState(property_address="X", budget=1e7))
    assert "decision" not in out["recommendation"]
    assert "error" in out["recommendation"]
