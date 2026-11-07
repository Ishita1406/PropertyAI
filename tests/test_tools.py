"""Unit tests for the deterministic, non-LLM tool layer.

These run without a GROQ_API_KEY since none of the tools call an LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from property_advisor.tools.financial_calculator import (  # noqa: E402
    calculate_emi,
    compute_investment_metrics,
)
from property_advisor.tools.market_data_tool import get_market_data  # noqa: E402
from property_advisor.tools.property_data_tool import get_property_data, parse_address  # noqa: E402
from property_advisor.tools.risk_data_tool import compute_risk_score, get_risk_data  # noqa: E402


def test_property_data_tool_finds_known_locality():
    data = get_property_data("Flat 12B, Whitefield, Bangalore, 560066")
    assert data["city"] == "Bangalore"
    assert data["locality_slug"] == "whitefield_bangalore"
    assert data["price_inr"] > 0


def test_property_data_tool_returns_empty_for_unknown_locality():
    assert get_property_data("Unknown Town, Nowhere State") == {}


def test_parse_address_extracts_pin_code():
    parsed = parse_address("Flat 12B, Whitefield, Bangalore, 560066")
    assert parsed["pin_code"] == "560066"


def test_market_data_tool_missing_for_tier3_city():
    assert get_market_data("Muzaffarpur", "Saraiyaganj", slug="muzaffarpur_bihar") == {}


def test_market_data_tool_found_for_known_locality():
    data = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    assert data["appreciation_rate_5yr_pct"] > 0


def test_risk_score_unknown_locality_is_moderate_not_zero():
    score = compute_risk_score(get_risk_data(None))
    assert score["risk_score"] == 50.0
    assert score["data_quality_confidence"] == 0.0


def test_high_flood_risk_forces_human_review_threshold():
    risk_data = get_risk_data("dadar_kurla_mumbai")
    score = compute_risk_score(risk_data)
    assert risk_data["flood_risk"] == "high"
    assert score["risk_score"] > 75


def test_low_risk_locality_stays_below_threshold():
    risk_data = get_risk_data("whitefield_bangalore")
    score = compute_risk_score(risk_data)
    assert score["risk_score"] < 75


def test_emi_is_deterministic_and_positive():
    emi1 = calculate_emi(1_000_000, 9.0, 20)
    emi2 = calculate_emi(1_000_000, 9.0, 20)
    assert emi1 == emi2
    assert emi1 > 0


def test_emi_zero_principal_is_zero():
    assert calculate_emi(0, 9.0, 20) == 0.0


def test_investment_metrics_negative_roi_for_low_yield_high_price_property():
    property_data = get_property_data("Worli, Mumbai")
    market_data = get_market_data("Mumbai", "Worli", slug="worli_mumbai")
    metrics = compute_investment_metrics(property_data, market_data, budget=45_000_000, horizon_years=5)
    assert metrics["roi_pct"] < 0
    # Operating cash flow is positive (the asset earns money before financing); the
    # leveraged EMI shortfall is reported separately and is the routine Indian-rental case.
    assert metrics["negative_cash_flow"] is False
    assert metrics["cash_flow_severity"] == "positive"
    assert metrics["levered_cash_flow_negative"] is True
    assert metrics["strong_appreciation_evidence"] is False


def test_strong_appreciation_evidence_true_for_high_growth_corridor_despite_negative_cash_flow():
    property_data = get_property_data("Whitefield, Bangalore")
    market_data = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    metrics = compute_investment_metrics(property_data, market_data, budget=9_500_000, horizon_years=5)
    # Operating cash flow is positive; only the leveraged figure is negative (normal here).
    assert metrics["negative_cash_flow"] is False
    assert metrics["cash_flow_severity"] == "positive"
    assert metrics["levered_cash_flow_negative"] is True
    assert metrics["strong_appreciation_evidence"] is True


def test_strong_appreciation_evidence_false_for_flood_prone_dadar():
    property_data = get_property_data("Dadar, Mumbai")
    market_data = get_market_data("Mumbai", "Dadar-Kurla Belt", slug="dadar_kurla_mumbai")
    metrics = compute_investment_metrics(property_data, market_data, budget=18_000_000, horizon_years=5)
    assert metrics["strong_appreciation_evidence"] is False


def test_investment_metrics_positive_roi_for_high_growth_corridor():
    property_data = get_property_data("Whitefield, Bangalore")
    market_data = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    metrics = compute_investment_metrics(property_data, market_data, budget=9_500_000, horizon_years=5)
    assert metrics["roi_pct"] > 0


def test_investment_metrics_handles_missing_market_data_without_crashing():
    property_data = get_property_data("Muzaffarpur, Bihar")
    metrics = compute_investment_metrics(property_data, {}, budget=3_500_000, horizon_years=5)
    assert metrics["rental_yield_pct"] == 0.0
    assert metrics["data_quality_confidence"] < 1.0
