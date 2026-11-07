"""End-to-end LangGraph tests with a fake LLM and fake retrieval (offline):
routing, retries, human approval/rejection with checkpoint resume, refusal,
invalid input and report generation."""

from __future__ import annotations

from langgraph.types import Command

from property_advisor.config import MAX_DATA_RETRIES
from property_advisor.graph import build_graph
from property_advisor.report_generator import generate_reports
from property_advisor.schemas import RecommendationOutput, RegulatoryRequirement, ResearchFindingsOutput
from tests.conftest import FakeLLM, tool_call_turn

WHITEFIELD = {
    "property_address": "Whitefield, Bangalore, 560066",
    "budget": 9_500_000.0,
    "investment_horizon_years": 5,
    "investment_strategy": "rental",
}


def _research_llm():
    return FakeLLM(
        tool_turns=[tool_call_turn(("search_official_documents", {"query": "promoter registration real estate project"}))],
        structured={"ResearchFindingsOutput": [ResearchFindingsOutput(
            summary="RERA registration must be verified for this simulated property.",
            regulatory_requirements=[RegulatoryRequirement(requirement="Project must be RERA-registered.", relevance="New flat.", citation_ids=["E1"])],
            due_diligence_checklist=["Check State RERA portal"],
        )]},
    )


def _rec(decision="BUY", justification="Strong simulated appreciation with acceptable risk.", **kw):
    return RecommendationOutput(decision=decision, justification=justification, confidence_score=0.85,
                                supporting_evidence=["E1: RERA registration required"], cited_evidence_ids=["E1"], **kw)


def _run_to_interrupt(graph, state, thread):
    cfg = {"configurable": {"thread_id": thread}}
    return graph.invoke(state, cfg), cfg


def test_full_workflow_approval_produces_cited_report(fake_rag, patch_llm, tmp_path):
    analyst = FakeLLM(structured={"RecommendationOutput": [_rec()]})
    patch_llm(research_llm=_research_llm(), analyst_llm=analyst)
    graph = build_graph()
    result, cfg = _run_to_interrupt(graph, WHITEFIELD, "t-approve")

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    for key in ("recommendation", "investment_metrics", "risk_assessment", "guardrail_result", "research_findings", "data_disclaimer"):
        assert key in payload
    assert payload["guardrail_result"]["status"] == "human_review_required"
    assert payload["research_findings"]["status"] == "complete"

    final = graph.invoke(Command(resume={"approved": True, "feedback": ""}), cfg)["final_report"]
    assert final["status"] == "approved"
    assert final["recommendation"]["decision"] == "BUY"
    assert final["evidence_sources"] and final["evidence_sources"][0].startswith("E1: ")
    assert "SIMULATED" in final["data_disclaimer"]
    assert final["data_provenance"]["property_data"].startswith("SIMULATED")
    assert final["investment_metrics"]["sensitivity"]

    paths = generate_reports(final, output_dir=str(tmp_path))
    md = open(paths["md"]).read()
    assert "Due Diligence Findings" in md and "SIMULATED" in md and "E1" in md and "Stress Scenarios" in md
    assert open(paths["pdf"], "rb").read(4) == b"%PDF"


def test_rejection_routes_back_to_analyst_with_feedback_then_resumes(fake_rag, patch_llm):
    analyst = FakeLLM(structured={"RecommendationOutput": [_rec(), _rec("HOLD", "Reviewer concerns noted; waiting for RERA verification.")]})
    patch_llm(research_llm=_research_llm(), analyst_llm=analyst)
    graph = build_graph()
    result, cfg = _run_to_interrupt(graph, WHITEFIELD, "t-reject")
    assert "__interrupt__" in result

    result = graph.invoke(Command(resume={"approved": False, "feedback": "Verify RERA first"}), cfg)
    assert "__interrupt__" in result  # back at human review with a new recommendation
    assert result["__interrupt__"][0].value["recommendation"]["decision"] == "HOLD"
    second_prompt = analyst.structured_invocations[1][1][1].content
    assert "Verify RERA first" in second_prompt

    final = graph.invoke(Command(resume={"approved": True, "feedback": ""}), cfg)["final_report"]
    assert final["status"] == "approved" and final["recommendation"]["decision"] == "HOLD"


def test_invalid_input_short_circuits_without_llm(patch_llm):
    research, analyst = FakeLLM(), FakeLLM()
    patch_llm(research_llm=research, analyst_llm=analyst)
    graph = build_graph()
    out = graph.invoke({**WHITEFIELD, "budget": -5.0, "investment_horizon_years": 0}, {"configurable": {"thread_id": "t-invalid"}})
    assert "__interrupt__" not in out
    assert out["final_report"]["status"] == "invalid_input"
    assert research.tool_invocations == [] and analyst.structured_invocations == []


def test_missing_market_data_triggers_bounded_retry(fake_rag, patch_llm):
    patch_llm(research_llm=_research_llm(), analyst_llm=FakeLLM(structured={"RecommendationOutput": [_rec("HOLD", "Simulated market data is missing.")]}))
    graph = build_graph()
    result, _ = _run_to_interrupt(graph, {
        "property_address": "Plot near Saraiyaganj, Muzaffarpur, Bihar", "budget": 3_500_000.0,
        "investment_horizon_years": 5, "investment_strategy": "long_term_appreciation",
    }, "t-missing")
    state = graph.get_state({"configurable": {"thread_id": "t-missing"}}).values
    assert state["data_retry_count"] == MAX_DATA_RETRIES
    assert state["market_data"] == {}
    assert state["guardrail_result"]["missing_market_data"] is True
    assert state["requires_human_review"] is True


def test_fabricated_number_triggers_bounded_reanalysis(fake_rag, patch_llm):
    bad = _rec(justification="Expect a guaranteed 47% ROI.")
    analyst = FakeLLM(structured={"RecommendationOutput": [bad, _rec()]})
    patch_llm(research_llm=_research_llm(), analyst_llm=analyst)
    graph = build_graph()
    result, cfg = _run_to_interrupt(graph, WHITEFIELD, "t-reanalyse")
    state = graph.get_state(cfg).values
    assert state["reanalysis_retry_count"] == 1
    assert len(analyst.structured_invocations) == 2
    assert "47%" in analyst.structured_invocations[1][1][1].content  # told what to fix
    assert state["guardrail_result"]["has_unsupported_claims"] is False


def test_total_llm_outage_degrades_then_refuses(fake_rag, patch_llm):
    patch_llm(fail_with=RuntimeError("GROQ_API_KEY is not set."))
    graph = build_graph()
    out = graph.invoke(WHITEFIELD, {"configurable": {"thread_id": "t-outage"}})
    assert "__interrupt__" not in out
    assert out["final_report"]["status"] == "refused"
    assert out["research_findings"]["status"] == "degraded_llm_unavailable"
    assert out["research_findings"]["evidence"]  # deterministic baseline retrieval still ran
