"""Deterministic input validation and recommendation guardrail tests."""

from property_advisor.validation import (
    build_allowed_numbers,
    find_invalid_citations,
    find_unsupported_numbers,
    validate_inputs,
    validate_recommendation,
)


def _state(**overrides):
    base = {
        "property_address": "Whitefield, Bangalore",
        "budget": 9_500_000.0,
        "investment_horizon_years": 5,
        "property_data": {"city": "Bangalore", "price_inr": 9_500_000, "area_sqft": 1250},
        "market_data": {"appreciation_rate_5yr_pct": 9.2, "rental_yield_avg_pct": 4.3},
        "investment_metrics": {"roi_pct": 28.68, "rental_yield_pct": 4.42, "negative_cash_flow": False,
                               "annual_cash_flow_inr": -543_000.5, "unlevered_annual_cash_flow_inr": 315_000.0},
        "risk_assessment": {"risk_score": 18.5},
        "research_findings": {"status": "complete", "concerns": [], "conflicting_evidence": [],
                              "evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}]},
        "rag_context": [],
        "recommendation": {"decision": "BUY", "justification": "Simulated ROI of 28.7% and yield of 4.42%.",
                           "supporting_evidence": ["roi_pct=28.68", "E1: RERA registration"],
                           "confidence_score": 0.8, "cited_evidence_ids": ["E1"]},
        "reanalysis_retry_count": 0,
        "requires_human_review": False,
    }
    base.update(overrides)
    return base


def test_validate_inputs_accepts_good_request():
    assert validate_inputs("Whitefield, Bangalore", 9_500_000, 5, "rental")["valid"]


def test_validate_inputs_rejects_bad_values():
    r = validate_inputs("", -1, 0, "crypto")
    assert not r["valid"]
    assert len(r["errors"]) == 4


def test_validate_inputs_warns_on_tiny_budget():
    r = validate_inputs("Whitefield", 50_000, 5, "rental")
    assert r["valid"] and r["warnings"]


def test_numbers_matching_metrics_pass():
    allowed = build_allowed_numbers(_state())
    texts = ["ROI about 28.7%, budget ₹95 lakh, INR 9,500,000, 80% LTV, under RERA s. 3 of the Act, 2016."]
    # 80 comes from the LTV ratio 0.8 once metrics include loan assumptions; add it explicitly here.
    allowed.add(80.0)
    assert find_unsupported_numbers(texts, allowed) == []


def test_fabricated_numbers_are_flagged():
    allowed = build_allowed_numbers(_state())
    bad = find_unsupported_numbers(["Expect a 45% ROI and rent of INR 72,000 per month."], allowed)
    assert "45%" in bad
    assert any("72,000" in b for b in bad)


def test_numbers_from_retrieved_legal_text_are_allowed():
    state = _state(rag_context=[{"text": "deposit seventy per cent ... not exceed two months' rent; 70% of amounts"}])
    allowed = build_allowed_numbers(state)
    assert find_unsupported_numbers(["RERA requires 70% of buyer funds in escrow."], allowed) == []


def test_invalid_citations_detected():
    rec = {"justification": "per E9", "supporting_evidence": ["E1"], "cited_evidence_ids": ["E1", "E7"]}
    assert find_invalid_citations(rec, {"E1", "E2"}) == ["E7", "E9"]


def test_guardrail_clean_recommendation_goes_to_human_review():
    out = validate_recommendation(_state())
    g = out["guardrail_result"]
    assert g["status"] == "human_review_required"
    assert not g["has_unsupported_claims"]
    assert out["requires_human_review"] is False


def test_guardrail_requests_reanalysis_for_fabricated_number_then_caps():
    rec = dict(_state()["recommendation"], justification="ROI will be 55% next year.")
    g = validate_recommendation(_state(recommendation=rec))["guardrail_result"]
    assert g["status"] == "request_reanalysis"
    assert g["unsupported_numbers"] == ["55%"]
    g2 = validate_recommendation(_state(recommendation=rec, reanalysis_retry_count=99))["guardrail_result"]
    assert g2["status"] == "human_review_required"


def test_guardrail_refuses_missing_decision():
    g = validate_recommendation(_state(recommendation={"error": "LLM unavailable"}))["guardrail_result"]
    assert g["status"] == "refuse"


def test_guardrail_flags_conflicts_missing_data_and_degraded_research():
    state = _state(
        market_data={},
        rag_context=[{"source": "hinjewadi_bullish_analyst_note.txt"}, {"source": "hinjewadi_bearish_analyst_note.txt"}],
        research_findings={"status": "degraded_llm_unavailable", "concerns": [{"concern": "Unregistered project", "severity": "high"}],
                           "conflicting_evidence": [], "evidence": [{"evidence_id": "E1"}]},
    )
    out = validate_recommendation(state)
    g = out["guardrail_result"]
    assert g["conflicting_evidence"] is True
    assert g["missing_market_data"] is True
    assert g["research_status"] == "degraded_llm_unavailable"
    assert g["high_severity_concerns"] == ["Unregistered project"]
    assert out["requires_human_review"] is True
