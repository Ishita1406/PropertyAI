"""Independent checks of the deterministic financial engine (no LLM)."""

import pytest

from property_advisor.tools.financial_calculator import (
    calculate_break_even_years,
    calculate_cap_rate,
    calculate_cash_flow,
    calculate_emi,
    calculate_rental_yield,
    calculate_roi,
    compute_investment_metrics,
    compute_sensitivity,
    financial_assumptions,
)
from property_advisor.tools.market_data_tool import get_market_data
from property_advisor.tools.property_data_tool import get_property_data


def test_emi_matches_textbook_value():
    # INR 10 lakh, 9% p.a., 20 years -> INR 8,997.26 (standard amortisation tables)
    assert calculate_emi(1_000_000, 9.0, 20) == pytest.approx(8997.26, abs=0.01)


def test_emi_zero_interest_is_straight_line():
    assert calculate_emi(1_200_000, 0.0, 10) == pytest.approx(10_000.0)


def test_basic_ratios():
    assert calculate_rental_yield(360_000, 9_000_000) == 4.0
    assert calculate_cap_rate(270_000, 9_000_000) == 3.0
    assert calculate_cash_flow(360_000, 90_000, 200_000) == 70_000
    assert calculate_roi(1_000_000, 4_000_000) == 25.0
    assert calculate_break_even_years(1_000_000, 100_000) == 10.0
    assert calculate_break_even_years(1_000_000, -5) is None
    assert calculate_rental_yield(1, 0) == 0.0


def test_whitefield_metrics_by_hand():
    p = get_property_data("Whitefield, Bangalore")
    m = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    metrics = compute_investment_metrics(p, m, budget=9_500_000, horizon_years=5)
    monthly_rent = m["avg_monthly_rent_per_sqft_inr"] * p["area_sqft"]
    annual_rent = monthly_rent * 12
    assert metrics["annual_rent_inr"] == pytest.approx(annual_rent)
    assert metrics["annual_operating_expenses_inr"] == pytest.approx(annual_rent * 0.25)
    assert metrics["rental_yield_pct"] == pytest.approx(round(annual_rent / p["price_inr"] * 100, 2))
    assert metrics["cap_rate_pct"] == pytest.approx(round(annual_rent * 0.75 / p["price_inr"] * 100, 2))
    expected_value = p["price_inr"] * (1 + m["appreciation_rate_5yr_pct"] / 100) ** 5
    assert metrics["projected_value_inr"] == pytest.approx(expected_value, rel=1e-6)


def test_sensitivity_scenarios_move_in_expected_direction():
    p = get_property_data("Whitefield, Bangalore")
    m = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    base = compute_investment_metrics(p, m, 9_500_000, 5)
    sens = compute_sensitivity(p, m, 9_500_000, 5)
    assert set(sens) == {"interest_rate_plus_2pp", "appreciation_halved", "zero_appreciation", "higher_vacancy_and_costs"}
    assert sens["interest_rate_plus_2pp"]["annual_cash_flow_inr"] < base["annual_cash_flow_inr"]
    assert sens["appreciation_halved"]["roi_pct"] < base["roi_pct"]
    assert sens["zero_appreciation"]["roi_pct"] < sens["appreciation_halved"]["roi_pct"]
    assert sens["higher_vacancy_and_costs"]["unlevered_annual_cash_flow_inr"] < base["unlevered_annual_cash_flow_inr"]
    # deterministic
    assert compute_sensitivity(p, m, 9_500_000, 5) == sens


def test_assumptions_disclose_simulated_inputs_and_exclusions():
    p = get_property_data("Whitefield, Bangalore")
    m = get_market_data("Bangalore", "Whitefield", slug="whitefield_bangalore")
    text = " ".join(financial_assumptions(compute_investment_metrics(p, m, 9_500_000, 5)))
    assert "SIMULATED" in text
    assert "stamp duty" in text
    assert "80%" in text
