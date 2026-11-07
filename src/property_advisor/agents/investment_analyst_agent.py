"""AGENT 2 — Investment Analyst Agent (LLM, Groq).

Consolidates the former Recommendation Agent. It weighs:

* deterministic investment metrics and stress scenarios
  (``tools/financial_calculator.py``) — the LLM never computes figures;
* the deterministic risk score;
* the structured findings of the Property Research & Due Diligence Agent
  (regulatory requirements, concerns, gaps, conflicts, evidence IDs);

and produces a BUY / HOLD / AVOID call with justification, evidence,
key risks, assumptions and cited evidence IDs.

Deterministic post-processing (not left to LLM discretion):
* the original operating-cash-flow safety net (BUY -> HOLD when operating cash
  flow is significantly negative without strong appreciation evidence);
* confidence is capped at 0.6 when the research step was degraded;
* the full list of deterministic financial assumptions is attached.

Every number the model writes is later checked against deterministic values by
the guardrail (``validation.find_unsupported_numbers``).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from property_advisor.agents._llm_utils import invoke_with_retries
from property_advisor.config import get_llm
from property_advisor.logging_utils import timed_node
from property_advisor.schemas import RecommendationOutput
from property_advisor.state import PropertyState

DEGRADED_RESEARCH_CONFIDENCE_CAP = 0.6

SYSTEM_PROMPT = """You are the Investment Analyst Agent for Indian residential real estate. Output BUY, HOLD, or AVOID.

Decide from the WHOLE picture, never a single metric. Use ONLY numbers given below; never invent, \
estimate or recompute figures (all calculations are done deterministically for you). Weigh together:
- Total return: roi_pct (already includes appreciation AND levered cash flow over the horizon) and the \
simulated appreciation rate; check the stress scenarios for fragility.
- Income: rental_yield_pct and cash_flow_severity (OPERATING cash flow, before any loan).
- Risk: risk_score (0-100, higher = worse) and data_quality_confidence.
- Due-diligence findings: regulatory requirements, concerns (with severity), evidence gaps and conflicts. \
Serious legal/regulatory concerns or missing evidence should lower confidence or the decision.

Guidance (lean, not rigid thresholds):
- Strong total return + solid yield + acceptable risk + no serious due-diligence concerns -> lean BUY.
- Mixed or borderline signals, or unresolved due-diligence gaps -> HOLD.
- Reserve AVOID for genuinely poor deals: negative/very weak roi_pct, very low yield, high risk, or a \
serious regulatory red flag -- NOT merely because a leveraged buyer sees negative monthly cash flow.

Financing caveat: levered_cash_flow_negative=true means an 80%-LTV EMI exceeds net rent. This is normal \
for Indian rentals; treat it as a confidence caveat, not by itself a reason to AVOID.

Operating cash-flow rule (mandatory): cash_flow_severity="significantly_negative" -> HOLD or AVOID \
UNLESS strong_appreciation_evidence=true.

Evidence: market/property figures are SIMULATED — say "simulated" when relying on them. Put evidence IDs \
(E1, E2, ...) you relied on in cited_evidence_ids; use only IDs listed. Be concise."""


def _compact_metrics(metrics: dict) -> dict:
    keys = (
        "roi_pct",
        "rental_yield_pct",
        "cap_rate_pct",
        "annual_cash_flow_inr",
        "unlevered_annual_cash_flow_inr",
        "cash_flow_severity",
        "negative_cash_flow",
        "levered_cash_flow_negative",
        "strong_appreciation_evidence",
        "break_even_years",
        "data_quality_confidence",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def _compact_sensitivity(metrics: dict) -> dict:
    return {
        name: {k: v for k, v in sc.items() if k in ("roi_pct", "annual_cash_flow_inr")}
        for name, sc in (metrics.get("sensitivity") or {}).items()
    }


def _compact_risk(risk: dict) -> dict:
    keys = ("risk_score", "data_quality_confidence", "factors")
    return {k: risk[k] for k in keys if k in risk}


def _compact_research(findings: dict) -> dict:
    if not findings:
        return {"status": "missing"}
    return {
        "status": findings.get("status"),
        "summary": findings.get("summary"),
        "regulatory_requirements": [
            {"requirement": r["requirement"], "evidence": r["citation_ids"], "binding": r["binding_status"]}
            for r in findings.get("regulatory_requirements", [])
        ],
        "concerns": [
            {"concern": c["concern"], "severity": c["severity"], "basis": c["basis"], "evidence": c["citation_ids"]}
            for c in findings.get("concerns", [])
        ],
        "evidence_gaps": [g["gap"] for g in findings.get("evidence_gaps", []) if g.get("origin") != "deterministic"][:4]
        + [g["gap"] for g in findings.get("evidence_gaps", []) if g.get("origin") == "deterministic"][:3],
        "conflicting_evidence": findings.get("conflicting_evidence", []),
        "evidence_ids_available": [e["evidence_id"] for e in findings.get("evidence", [])],
    }


def _enforce_cashflow_safety_net(result: RecommendationOutput, state: PropertyState) -> RecommendationOutput:
    """Deterministic safety net carried over unchanged from the original
    Recommendation Agent: financial decision rules are not left to the LLM."""
    metrics = state.investment_metrics
    if not metrics or state.investment_strategy != "rental":
        return result

    significantly_negative = metrics.get("cash_flow_severity") == "significantly_negative"
    strong_appreciation = bool(metrics.get("strong_appreciation_evidence"))
    levered_negative = bool(metrics.get("levered_cash_flow_negative"))

    decision = result.decision
    confidence = result.confidence_score
    justification = result.justification
    overridden = False

    if significantly_negative and not strong_appreciation and decision == "BUY":
        decision = "HOLD"
        confidence = round(min(confidence, 0.55), 2)
        overridden = True
        justification = (
            f"{justification} [Guardrail override: the Investment Analyst Agent's original BUY call "
            f"was downgraded to HOLD because the property's operating (pre-financing) cash flow is "
            f"significantly negative and strong_appreciation_evidence was not met.]"
        )

    if not overridden and levered_negative and "cash flow" not in justification.lower():
        confidence = round(confidence * 0.95, 2)
        justification = (
            f"{justification} [Note: at 80% loan-to-value the EMI exceeds net rent (levered annual "
            f"cash flow INR {metrics.get('annual_cash_flow_inr')}); Indian rentals typically rely on "
            f"appreciation rather than monthly cash flow, so this is a confidence caveat, not a "
            f"disqualifier.]"
        )

    return result.model_copy(
        update={
            "decision": decision,
            "justification": justification,
            "confidence_score": max(0.0, min(1.0, confidence)),
        }
    )


def build_user_prompt(state: PropertyState) -> str:
    feedback_note = ""
    if state.human_decision.get("approved") is False:
        feedback_note = f"\nRejected by human reviewer. Feedback: {state.human_decision.get('feedback', '')}"
    reanalysis_note = ""
    if state.reanalysis_retry_count > 0:
        reasons = state.guardrail_result.get("unsupported_claims", []) or state.guardrail_result.get("reasons", [])
        reanalysis_note = (
            f"\nRe-analysis #{state.reanalysis_retry_count}. Fix these validation problems "
            f"(use only provided numbers and evidence IDs): {json.dumps(reasons)}"
        )
    prop = {k: state.property_data.get(k) for k in ("city", "locality", "property_type", "price_inr", "area_sqft") if k in state.property_data}
    return f"""Address: {state.property_address}
Budget: INR {state.budget} | Horizon: {state.investment_horizon_years}yr | Strategy: {state.investment_strategy}

Property (SIMULATED): {json.dumps(prop) if prop else "not found"}
Market (SIMULATED): {json.dumps(state.market_data) if state.market_data else "none"}
Metrics (deterministic): {json.dumps(_compact_metrics(state.investment_metrics))}
Stress scenarios (deterministic): {json.dumps(_compact_sensitivity(state.investment_metrics))}
Risk (deterministic on simulated indicators): {json.dumps(_compact_risk(state.risk_assessment))}
Due-diligence findings (Agent 1): {json.dumps(_compact_research(state.research_findings))}
{feedback_note}{reanalysis_note}
"""


@timed_node("investment_analyst_agent")
def investment_analyst_agent(state: PropertyState) -> dict:
    try:
        llm = get_llm(temperature=0.2, max_tokens=700)
        structured_llm = llm.with_structured_output(RecommendationOutput)
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_user_prompt(state))]
        result: RecommendationOutput = invoke_with_retries(
            lambda: structured_llm.invoke(messages), "investment_analyst_agent"
        )
    except Exception as exc:  # noqa: BLE001
        # No decision -> the guardrail refuses; nothing invalid reaches a human.
        return {
            "recommendation": {"error": f"LLM unavailable: {exc}"},
            "errors": [*state.errors, f"Investment Analyst Agent failed: {exc}"],
        }

    result = _enforce_cashflow_safety_net(result, state)
    rec: dict[str, Any] = result.model_dump()

    if state.research_findings.get("status") != "complete":
        rec["confidence_score"] = round(min(rec["confidence_score"], DEGRADED_RESEARCH_CONFIDENCE_CAP), 2)
        rec["confidence_capped_reason"] = "due-diligence research was degraded"

    rec["financial_assumptions"] = state.investment_metrics.get("assumptions", [])
    evidence = {e["evidence_id"]: e["citation"] for e in state.research_findings.get("evidence", [])}
    rec["citations"] = [
        {"evidence_id": i, "citation": evidence[i]} for i in rec.get("cited_evidence_ids", []) if i in evidence
    ]
    return {"recommendation": rec}
