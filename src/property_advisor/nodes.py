"""Deterministic LangGraph nodes — supporting components, NOT agents.

These were previously labelled "Property Agent", "Market Agent", "Investment
Metrics Agent", "Risk Assessment Agent" and "Guardrail Agent". None of them
reasons or makes choices, so they are ordinary nodes over the tool/service
layer. The only LLM agents live in ``property_advisor.agents``.
"""

from __future__ import annotations

from langgraph.types import interrupt

from property_advisor.config import RISK_HUMAN_REVIEW_THRESHOLD
from property_advisor.logging_utils import timed_node
from property_advisor.state import PropertyState
from property_advisor.tools.financial_calculator import (
    compute_investment_metrics,
    compute_sensitivity,
    financial_assumptions,
)
from property_advisor.tools.market_data_tool import get_market_data
from property_advisor.tools.property_data_tool import get_property_data
from property_advisor.tools.risk_data_tool import compute_risk_score, get_risk_data
from property_advisor.validation import validate_inputs, validate_recommendation

DATA_PROVENANCE = {
    "property_data": "SIMULATED — mock dataset (src/property_advisor/data/mock_properties.json)",
    "market_data": "SIMULATED — mock dataset (src/property_advisor/data/mock_market_trends.json)",
    "risk_indicators": "SIMULATED — mock dataset (src/property_advisor/data/mock_risk_data.json)",
    "investment_metrics": "DERIVED — deterministic Python calculations on simulated inputs",
    "legal_and_policy_evidence": "OFFICIAL — Indian legislation / government publications in data/ (see citations)",
    "market_commentary": "SIMULATED — synthetic analyst notes in data/rag_corpus (not real reports)",
}

DATA_DISCLAIMER = (
    "Property details, prices, rents, appreciation rates, risk indicators and market commentary in this "
    "analysis are SIMULATED for demonstration and are not observations of the real market. Legal and policy "
    "references come from official government documents but are not legal advice; model laws and guidelines "
    "apply only where adopted, and State/local rules may differ."
)


@timed_node("validate_input")
def validate_input_node(state: PropertyState) -> dict:
    result = validate_inputs(
        state.property_address, state.budget, state.investment_horizon_years, state.investment_strategy
    )
    update: dict = {"input_validation": result, "data_provenance": DATA_PROVENANCE}
    if not result["valid"]:
        update["errors"] = [*state.errors, *result["errors"]]
    return update


def route_after_validation(state: PropertyState) -> str:
    return "valid" if state.input_validation.get("valid", False) else "invalid"


@timed_node("fetch_property_data")
def fetch_property_data_node(state: PropertyState) -> dict:
    """Simulated property + market lookup (formerly property_agent + market_agent)."""
    errors = list(state.errors)
    property_data = get_property_data(state.property_address)
    if not property_data:
        errors.append(f"No property data found for address: {state.property_address!r}")
    market_data = get_market_data(
        property_data.get("city", ""), property_data.get("locality", ""), slug=property_data.get("locality_slug")
    )
    if property_data and not market_data:
        errors.append(
            f"No market trend data found for {property_data.get('locality', '')}, {property_data.get('city', '')}"
        )
    warnings = list(state.input_validation.get("warnings", []))
    price = property_data.get("price_inr")
    if price and state.budget < price:
        warnings.append(
            f"Budget INR {state.budget:,.0f} is below the simulated listing price INR {price:,.0f}."
        )
    return {
        "property_data": property_data,
        "market_data": market_data,
        "errors": errors,
        "input_validation": {**state.input_validation, "warnings": list(dict.fromkeys(warnings))},
    }


def data_retry_increment_node(state: PropertyState) -> dict:
    return {"data_retry_count": state.data_retry_count + 1}


@timed_node("compute_financials")
def compute_financials_node(state: PropertyState) -> dict:
    """Deterministic metrics, stress scenarios and risk score."""
    metrics = compute_investment_metrics(
        property_data=state.property_data,
        market_data=state.market_data,
        budget=state.budget,
        horizon_years=state.investment_horizon_years,
    )
    metrics["sensitivity"] = compute_sensitivity(
        state.property_data, state.market_data, state.budget, state.investment_horizon_years
    )
    metrics["assumptions"] = financial_assumptions(metrics)
    metrics["data_source"] = "derived_from_simulated_inputs"

    risk_data = get_risk_data(state.property_data.get("locality_slug"))
    risk_assessment = compute_risk_score(risk_data)
    risk_assessment["raw_indicators"] = risk_data
    risk_assessment["data_source"] = "simulated" if risk_data else "unavailable"

    requires_human_review = state.requires_human_review or (
        risk_assessment["risk_score"] > RISK_HUMAN_REVIEW_THRESHOLD
    )
    return {
        "investment_metrics": metrics,
        "risk_assessment": risk_assessment,
        "requires_human_review": requires_human_review,
    }


@timed_node("validate_recommendation")
def validate_recommendation_node(state: PropertyState) -> dict:
    update = validate_recommendation(state.model_dump())
    if update["guardrail_result"]["status"] == "request_reanalysis":
        update["reanalysis_retry_count"] = state.reanalysis_retry_count + 1
    return update


def _public_risk(state: PropertyState) -> dict:
    return {k: v for k, v in state.risk_assessment.items() if k != "raw_indicators"}


@timed_node("human_review")
def human_review_node(state: PropertyState) -> dict:
    payload = {
        "property_address": state.property_address,
        "recommendation": state.recommendation,
        "investment_metrics": state.investment_metrics,
        "risk_assessment": _public_risk(state),
        "guardrail_result": state.guardrail_result,
        "research_findings": state.research_findings,
        "data_provenance": state.data_provenance,
        "data_disclaimer": DATA_DISCLAIMER,
    }
    decision = interrupt(payload)
    return {"human_decision": decision}


@timed_node("final_report")
def final_report_node(state: PropertyState) -> dict:
    if not state.input_validation.get("valid", True):
        report = {
            "status": "invalid_input",
            "property_address": state.property_address,
            "reason": "Invalid request: " + " ".join(state.input_validation.get("errors", [])),
            "input_validation": state.input_validation,
        }
        return {"final_report": report, "workflow_status": "invalid_input"}

    if state.guardrail_result.get("status") == "refuse":
        report = {
            "status": "refused",
            "property_address": state.property_address,
            "reason": "Guardrail refused this recommendation: no valid decision/justification was produced.",
            "guardrail_result": state.guardrail_result,
            "research_findings": state.research_findings,
            "errors": state.errors,
        }
        return {"final_report": report, "workflow_status": "refused"}

    evidence = state.research_findings.get("evidence", [])
    report = {
        "status": "approved",
        "property_address": state.property_address,
        "budget_inr": state.budget,
        "investment_horizon_years": state.investment_horizon_years,
        "investment_strategy": state.investment_strategy,
        "data_disclaimer": DATA_DISCLAIMER,
        "data_provenance": state.data_provenance,
        "input_warnings": state.input_validation.get("warnings", []),
        "property_data": state.property_data,
        "market_data": state.market_data,
        "recommendation": state.recommendation,
        "investment_metrics": state.investment_metrics,
        "risk_assessment": _public_risk(state),
        "research_findings": state.research_findings,
        "guardrail_result": state.guardrail_result,
        "human_decision": state.human_decision,
        # Kept for backward compatibility: list of human-readable source citations.
        "evidence_sources": [f"{e['evidence_id']}: {e['citation']}" for e in evidence],
    }
    return {"final_report": report, "workflow_status": "completed"}
