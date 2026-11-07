"""Financial Calculator Tool.

All investment-metric calculations are deterministic and INR-based,financial calculations be tool-based, never
LLM-based. No function here calls an LLM.
"""

from __future__ import annotations

from typing import Any

MONTHS_PER_YEAR = 12

# Typical Indian residential buy-to-let financing assumptions, used to model
# EMI-driven cash flow (Indian rental yields are well known to rarely cover
# full-leverage EMI; appreciation, not rental cash flow, drives most returns).
DEFAULT_LOAN_TO_VALUE_RATIO = 0.80
DEFAULT_INTEREST_RATE_PCT = 9.0
DEFAULT_LOAN_TENURE_YEARS = 20
DEFAULT_OPERATING_EXPENSE_RATIO = 0.25  # property tax, maintenance, vacancy loss


def calculate_emi(principal_inr: float, annual_interest_rate_pct: float, tenure_years: int) -> float:
    """Standard reducing-balance EMI formula. Returns 0 if principal is 0."""
    if principal_inr <= 0:
        return 0.0
    monthly_rate = (annual_interest_rate_pct / 100) / MONTHS_PER_YEAR
    n_months = tenure_years * MONTHS_PER_YEAR
    if monthly_rate == 0:
        return round(principal_inr / n_months, 2)
    factor = (1 + monthly_rate) ** n_months
    emi = principal_inr * monthly_rate * factor / (factor - 1)
    return round(emi, 2)


def calculate_rental_yield(annual_rent_inr: float, price_inr: float) -> float:
    """Gross rental yield as a percentage."""
    if price_inr <= 0:
        return 0.0
    return round((annual_rent_inr / price_inr) * 100, 2)


def calculate_cap_rate(annual_net_operating_income_inr: float, price_inr: float) -> float:
    """Net operating income / price, as a percentage."""
    if price_inr <= 0:
        return 0.0
    return round((annual_net_operating_income_inr / price_inr) * 100, 2)


def calculate_cash_flow(
    annual_rent_inr: float,
    annual_operating_expenses_inr: float,
    annual_debt_service_inr: float = 0.0,
) -> float:
    """Annual cash flow in INR after operating expenses and any debt service."""
    return round(annual_rent_inr - annual_operating_expenses_inr - annual_debt_service_inr, 2)


def calculate_roi(total_gain_inr: float, total_investment_inr: float) -> float:
    """Simple ROI as a percentage over the holding period."""
    if total_investment_inr <= 0:
        return 0.0
    return round((total_gain_inr / total_investment_inr) * 100, 2)


def calculate_break_even_years(total_investment_inr: float, annual_cash_flow_inr: float) -> float | None:
    """Years to recover the initial investment from cash flow alone.

    Returns None if cash flow is non-positive (break-even is never reached
    on cash flow alone).
    """
    if annual_cash_flow_inr <= 0:
        return None
    return round(total_investment_inr / annual_cash_flow_inr, 1)


def _cash_flow_severity(annual_cash_flow_inr: float, price_inr: float) -> str:
    """Classify cash-flow health.

    IMPORTANT: this must be fed the *unlevered* (operating) cash flow, not the
    fully-leveraged figure. Indian gross rental yields (~2-6%) never cover an
    80%-LTV EMI, so a leveraged cash flow is deeply negative for essentially
    every property — using it here mislabels the entire market as
    "significantly_negative" and is what previously biased recommendations
    toward AVOID. The operating cash flow reflects whether the asset itself is
    income-positive (financing-independent), which is the meaningful signal.
    """
    if price_inr <= 0:
        return "unknown"
    ratio_pct = (annual_cash_flow_inr / price_inr) * 100
    if ratio_pct >= 0:
        return "positive"
    if ratio_pct >= -3.0:
        return "mildly_negative"
    return "significantly_negative"


# Thresholds for what counts as "extremely strong appreciation evidence" — the
# one thing that can justify a BUY despite significantly negative cash flow.
# Both must hold: the underlying market appreciation rate itself must be high
# (not just a high ROI driven by leverage math), AND the resulting 5-year ROI
# must clear a wide margin, not just barely positive.
STRONG_APPRECIATION_RATE_THRESHOLD_PCT = 8.0
STRONG_APPRECIATION_ROI_THRESHOLD_PCT = 15.0


def _is_strong_appreciation_evidence(appreciation_pct: float, roi_pct: float) -> bool:
    return (
        appreciation_pct >= STRONG_APPRECIATION_RATE_THRESHOLD_PCT
        and roi_pct >= STRONG_APPRECIATION_ROI_THRESHOLD_PCT
    )


def compute_investment_metrics(
    property_data: dict[str, Any],
    market_data: dict[str, Any],
    budget: float,
    horizon_years: int,
    operating_expense_ratio: float = DEFAULT_OPERATING_EXPENSE_RATIO,
    loan_to_value_ratio: float = DEFAULT_LOAN_TO_VALUE_RATIO,
    interest_rate_pct: float = DEFAULT_INTEREST_RATE_PCT,
    loan_tenure_years: int = DEFAULT_LOAN_TENURE_YEARS,
) -> dict[str, Any]:
    """Aggregate all deterministic investment metrics for the Investment Metrics Agent."""
    price_inr = float(property_data.get("price_inr", budget) or budget)
    area_sqft = float(property_data.get("area_sqft", 0) or 0)
    rent_per_sqft = float(market_data.get("avg_monthly_rent_per_sqft_inr", 0) or 0)
    appreciation_pct = float(market_data.get("appreciation_rate_5yr_pct", 0) or 0)

    monthly_rent_inr = rent_per_sqft * area_sqft
    annual_rent_inr = monthly_rent_inr * MONTHS_PER_YEAR
    annual_operating_expenses_inr = annual_rent_inr * operating_expense_ratio
    annual_net_operating_income_inr = annual_rent_inr - annual_operating_expenses_inr

    loan_principal_inr = price_inr * loan_to_value_ratio
    monthly_emi_inr = calculate_emi(loan_principal_inr, interest_rate_pct, loan_tenure_years)
    annual_debt_service_inr = round(monthly_emi_inr * MONTHS_PER_YEAR, 2)

    unlevered_cash_flow_inr = calculate_cash_flow(annual_rent_inr, annual_operating_expenses_inr)
    annual_cash_flow_inr = calculate_cash_flow(
        annual_rent_inr, annual_operating_expenses_inr, annual_debt_service_inr
    )

    rental_yield_pct = calculate_rental_yield(annual_rent_inr, price_inr)
    cap_rate_pct = calculate_cap_rate(annual_net_operating_income_inr, price_inr)

    projected_value_inr = price_inr * ((1 + appreciation_pct / 100) ** horizon_years)
    cumulative_cash_flow_inr = round(annual_cash_flow_inr * horizon_years, 2)
    total_gain_inr = (projected_value_inr - price_inr) + cumulative_cash_flow_inr
    roi_pct = calculate_roi(total_gain_inr, price_inr)
    break_even_years = calculate_break_even_years(price_inr, unlevered_cash_flow_inr)
    strong_appreciation_evidence = _is_strong_appreciation_evidence(appreciation_pct, roi_pct)

    return {
        "price_inr": price_inr,
        "monthly_rent_inr": round(monthly_rent_inr, 2),
        "annual_rent_inr": round(annual_rent_inr, 2),
        "annual_operating_expenses_inr": round(annual_operating_expenses_inr, 2),
        "loan_assumptions": {
            "loan_to_value_ratio": loan_to_value_ratio,
            "interest_rate_pct": interest_rate_pct,
            "loan_tenure_years": loan_tenure_years,
            "monthly_emi_inr": monthly_emi_inr,
            "annual_debt_service_inr": annual_debt_service_inr,
        },
        "unlevered_annual_cash_flow_inr": unlevered_cash_flow_inr,
        "annual_cash_flow_inr": annual_cash_flow_inr,
        # Severity / negative_cash_flow describe the property's OPERATING
        # (unlevered) economics — the financing-independent measure of whether
        # the asset itself earns money. The leveraged shortfall is reported
        # separately as levered_cash_flow_negative so it can be weighed as a
        # financing caveat without disqualifying every leveraged Indian rental.
        "cash_flow_severity": _cash_flow_severity(unlevered_cash_flow_inr, price_inr),
        "rental_yield_pct": rental_yield_pct,
        "cap_rate_pct": cap_rate_pct,
        "projected_value_inr": round(projected_value_inr, 2),
        "cumulative_cash_flow_inr": cumulative_cash_flow_inr,
        "roi_pct": roi_pct,
        "break_even_years": break_even_years,
        "horizon_years": horizon_years,
        "negative_cash_flow": unlevered_cash_flow_inr < 0,
        "levered_cash_flow_negative": annual_cash_flow_inr < 0,
        "strong_appreciation_evidence": strong_appreciation_evidence,
        "data_quality_confidence": 1.0 if (price_inr and rent_per_sqft) else 0.3,
    }


# --------------------------------------------------------------------------
# Deterministic stress scenarios and stated assumptions (consumed by the
# Investment Analyst Agent; the LLM never computes these itself).
# --------------------------------------------------------------------------

SENSITIVITY_SCENARIOS: dict[str, dict[str, Any]] = {
    "interest_rate_plus_2pp": {
        "description": "Home-loan rate 2 percentage points higher",
        "interest_rate_pct": DEFAULT_INTEREST_RATE_PCT + 2.0,
    },
    "appreciation_halved": {
        "description": "Annual appreciation half of the simulated locality rate",
        "appreciation_multiplier": 0.5,
    },
    "zero_appreciation": {
        "description": "No price appreciation over the horizon",
        "appreciation_multiplier": 0.0,
    },
    "higher_vacancy_and_costs": {
        "description": "Operating expense + vacancy ratio raised from 25% to 40% of rent",
        "operating_expense_ratio": 0.40,
    },
}


def compute_sensitivity(
    property_data: dict[str, Any],
    market_data: dict[str, Any],
    budget: float,
    horizon_years: int,
) -> dict[str, dict[str, Any]]:
    """Re-run compute_investment_metrics under fixed stress scenarios."""
    results: dict[str, dict[str, Any]] = {}
    for name, sc in SENSITIVITY_SCENARIOS.items():
        md = dict(market_data)
        if "appreciation_multiplier" in sc:
            md["appreciation_rate_5yr_pct"] = float(md.get("appreciation_rate_5yr_pct", 0) or 0) * sc["appreciation_multiplier"]
        m = compute_investment_metrics(
            property_data,
            md,
            budget,
            horizon_years,
            operating_expense_ratio=sc.get("operating_expense_ratio", DEFAULT_OPERATING_EXPENSE_RATIO),
            interest_rate_pct=sc.get("interest_rate_pct", DEFAULT_INTEREST_RATE_PCT),
        )
        results[name] = {
            "description": sc["description"],
            "roi_pct": m["roi_pct"],
            "annual_cash_flow_inr": m["annual_cash_flow_inr"],
            "unlevered_annual_cash_flow_inr": m["unlevered_annual_cash_flow_inr"],
            "cap_rate_pct": m["cap_rate_pct"],
        }
    return results


def financial_assumptions(metrics: dict[str, Any]) -> list[str]:
    """Plain-language statement of every assumption behind the metrics."""
    loan = metrics.get("loan_assumptions", {})
    return [
        "All property, rent, appreciation and risk inputs come from SIMULATED datasets, not market observations.",
        f"Financing: {loan.get('loan_to_value_ratio', DEFAULT_LOAN_TO_VALUE_RATIO) * 100:.0f}% loan-to-value at "
        f"{loan.get('interest_rate_pct', DEFAULT_INTEREST_RATE_PCT)}% p.a. over "
        f"{loan.get('loan_tenure_years', DEFAULT_LOAN_TENURE_YEARS)} years (reducing-balance EMI).",
        f"Operating expenses, property tax and vacancy = {DEFAULT_OPERATING_EXPENSE_RATIO * 100:.0f}% of gross rent.",
        "Rent = simulated locality rent per sq ft x property area; appreciation compounds annually at the simulated locality rate.",
        "ROI = (projected value gain + cumulative levered cash flow) / purchase price; no taxes, stamp duty, "
        "registration charges, brokerage or loan fees are included.",
    ]
