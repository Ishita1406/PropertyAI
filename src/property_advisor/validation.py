"""Deterministic validation and guardrails (no LLM calls).

* ``validate_inputs``           — request sanity checks before any work is done.
* ``find_unsupported_numbers``  — every number in the analyst's text must match
                                  a deterministic value (metrics, simulated data,
                                  thresholds) or a number in retrieved evidence.
* ``find_invalid_citations``    — evidence IDs cited by the analyst must exist in
                                  the research findings.
* ``validate_recommendation``   — the guardrail: aggregates all checks into a
                                  routing status + reasons for the human reviewer.

This replaces the former LLM-based claim audit, so the system has exactly two
LLM agents and its guardrails are reproducible and testable offline.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from property_advisor.config import (
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    MAX_REANALYSIS_RETRIES,
    RISK_HUMAN_REVIEW_THRESHOLD,
)

VALID_STRATEGIES = ("rental", "flip", "long_term_appreciation")
MAX_BUDGET_INR = 1e11  # INR 10,000 crore — anything above is almost certainly a typo

# Known pairs of SIMULATED notes that intentionally disagree (see
# data/rag_corpus/hinjewadi_*_analyst_note.txt). Kept from the original guardrail.
CONFLICT_PAIRS = [
    ("hinjewadi_bullish_analyst_note.txt", "hinjewadi_bearish_analyst_note.txt"),
]

_MULTIPLIERS = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "k": 1e3, "thousand": 1e3,
    "bn": 1e9, "billion": 1e9,
}
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z_\d.])(?P<cur>INR|Rs\.?|₹)?\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suf>%|percent|lakhs?|lacs?|crores?|cr|million|mn|thousand|billion|bn|k|l|m)?(?![A-Za-z\d])",
    re.I,
)
_LEGAL_REF_RE = re.compile(
    r"\b(?:sections?|ss?\.|clauses?|rules?|chapters?|pp?\.|pages?|no\.|act no\.?|schedule)\s*"
    r"\d+[A-Z]?(?:\(\w+\))*(?:\s*(?:-|to|and|,)\s*\d+[A-Z]?(?:\(\w+\))*)*",
    re.I,
)
_EVIDENCE_ID_RE = re.compile(r"\bE\d{1,3}\b")
_BRACKETED_NOTE_RE = re.compile(r"\[(?:Guardrail override|Note)[^\]]*\]")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_inputs(address: str, budget: float, horizon: int, strategy: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not address or not address.strip():
        errors.append("Property address is required.")
    elif len(address.strip()) < 3:
        errors.append("Property address is too short to identify a locality.")
    elif len(address) > 300:
        errors.append("Property address is too long (max 300 characters).")
    if budget is None or budget <= 0:
        errors.append("Budget must be greater than 0 INR.")
    elif budget > MAX_BUDGET_INR:
        errors.append("Budget exceeds INR 10,000 crore; please check the amount.")
    elif budget < 100_000:
        warnings.append("Budget below INR 1 lakh is unusually low for Indian residential property.")
    if horizon is None or not (1 <= int(horizon) <= 50):
        errors.append("Investment horizon must be between 1 and 50 years.")
    if strategy not in VALID_STRATEGIES:
        errors.append(f"Strategy must be one of {', '.join(VALID_STRATEGIES)}.")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Numeric-claim audit
# ---------------------------------------------------------------------------


def _collect_numbers(obj: Any, out: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        if 0 < abs(obj) <= 1:  # ratios may be quoted as percentages
            out.add(float(obj) * 100)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUMBER_RE.finditer(obj):
            val = _parse_match(m)
            if val is not None:
                out.add(val)


def _parse_match(m: re.Match) -> float | None:
    try:
        val = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    suf = (m.group("suf") or "").lower()
    if suf in _MULTIPLIERS:
        val *= _MULTIPLIERS[suf]
    return val


def build_allowed_numbers(state_like: dict[str, Any]) -> set[float]:
    """Numbers the analyst may legitimately quote."""
    allowed: set[float] = set()
    for key in (
        "investment_metrics",
        "market_data",
        "property_data",
        "risk_assessment",
        "research_findings",
    ):
        _collect_numbers(state_like.get(key, {}), allowed)
    for key in ("budget", "investment_horizon_years"):
        _collect_numbers(state_like.get(key), allowed)
    for ctx in state_like.get("rag_context", []) or []:
        _collect_numbers(ctx.get("text", ""), allowed)
    _collect_numbers([RISK_HUMAN_REVIEW_THRESHOLD, GUARDRAIL_CONFIDENCE_THRESHOLD, 100], allowed)
    return allowed


def _matches(value: float, allowed: Iterable[float]) -> bool:
    v = abs(value)
    for a in allowed:
        a = abs(a)
        if abs(v - a) <= max(0.051, 0.015 * a):
            return True
    return False


def find_unsupported_numbers(texts: Iterable[str], allowed: set[float]) -> list[str]:
    """Return number strings in ``texts`` that match no allowed value."""
    unsupported: list[str] = []
    for text in texts:
        if not text:
            continue
        cleaned = _BRACKETED_NOTE_RE.sub(" ", text)
        cleaned = _LEGAL_REF_RE.sub(" ", cleaned)
        cleaned = _EVIDENCE_ID_RE.sub(" ", cleaned)
        for m in _NUMBER_RE.finditer(cleaned):
            val = _parse_match(m)
            if val is None:
                continue
            suf = (m.group("suf") or "").lower()
            has_unit = bool(suf or m.group("cur"))
            raw = m.group("num").replace(",", "")
            is_int = "." not in raw
            if not has_unit and is_int and 0 <= abs(val) <= 10:
                continue  # small counts ("2 risks", "5-year")
            if not has_unit and is_int and 1800 <= val <= 2100:
                continue  # years ("Act, 2016")
            if not _matches(val, allowed):
                unsupported.append(m.group(0).strip())
    return list(dict.fromkeys(unsupported))


def find_invalid_citations(recommendation: dict[str, Any], valid_ids: set[str]) -> list[str]:
    cited = set(recommendation.get("cited_evidence_ids", []) or [])
    for text in [recommendation.get("justification", ""), *recommendation.get("supporting_evidence", [])]:
        cited.update(_EVIDENCE_ID_RE.findall(text or ""))
    return sorted(c for c in cited if c not in valid_ids)


def detect_corpus_conflicts(sources: Iterable[str]) -> list[str]:
    present = set(sources)
    return [f"{a} vs {b}" for a, b in CONFLICT_PAIRS if a in present and b in present]


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


def validate_recommendation(state_like: dict[str, Any]) -> dict[str, Any]:
    """Deterministic guardrail over the analyst's recommendation.

    Returns ``{"guardrail_result": ..., "requires_human_review": bool}``.
    Routing: ``refuse`` (nothing valid to approve), ``request_reanalysis``
    (fixable: unsupported numbers / invalid citations, while retries remain),
    else ``human_review_required``.
    """
    rec = state_like.get("recommendation", {}) or {}
    metrics = state_like.get("investment_metrics", {}) or {}
    risk = state_like.get("risk_assessment", {}) or {}
    research = state_like.get("research_findings", {}) or {}
    reanalysis_count = int(state_like.get("reanalysis_retry_count", 0))
    reasons: list[str] = []

    missing_property_data = not state_like.get("property_data")
    missing_market_data = not state_like.get("market_data")
    if missing_property_data:
        reasons.append("Property data is missing from the simulated dataset for this address.")
    if missing_market_data:
        reasons.append("Simulated market trend data is missing for this locality (data-quality risk).")

    risk_score = risk.get("risk_score", 0)
    high_risk = risk_score > RISK_HUMAN_REVIEW_THRESHOLD
    if high_risk:
        reasons.append(f"Risk score {risk_score} exceeds the human-review threshold of {RISK_HUMAN_REVIEW_THRESHOLD}.")

    negative_cash_flow = bool(metrics.get("negative_cash_flow"))
    if negative_cash_flow:
        reasons.append(f"Negative operating cash flow (INR {metrics.get('unlevered_annual_cash_flow_inr')}).")

    confidence = float(rec.get("confidence_score", 0.0) or 0.0)
    confidence_below_threshold = confidence < GUARDRAIL_CONFIDENCE_THRESHOLD
    if rec.get("decision") and confidence_below_threshold:
        reasons.append(f"Recommendation confidence {confidence:.2f} is below threshold {GUARDRAIL_CONFIDENCE_THRESHOLD}.")

    # Conflicting evidence: known simulated conflict pairs + conflicts the research agent reported.
    corpus_conflicts = detect_corpus_conflicts(c.get("source", "") for c in state_like.get("rag_context", []) or [])
    reported_conflicts = list(research.get("conflicting_evidence", []) or [])
    conflicts = list(reported_conflicts)
    for c in corpus_conflicts:
        if not any(c in r for r in reported_conflicts):
            conflicts.append(c)
    conflicts = list(dict.fromkeys(conflicts))
    conflicting_evidence = bool(conflicts)
    if conflicting_evidence:
        reasons.append("Conflicting evidence: " + "; ".join(conflicts))

    research_status = research.get("status", "missing")
    research_degraded = research_status != "complete"
    if research_degraded:
        reasons.append(
            f"Due-diligence research status is '{research_status}': regulatory findings were not fully analysed."
        )
    high_concerns = [c.get("concern", "") for c in research.get("concerns", []) if c.get("severity") == "high"]
    if high_concerns:
        reasons.append("High-severity due-diligence concerns: " + "; ".join(high_concerns[:3]))

    # Grounding checks on the analyst's own words.
    allowed = build_allowed_numbers(state_like)
    texts = [rec.get("justification", ""), *rec.get("supporting_evidence", []), *rec.get("key_risks", []), *rec.get("assumptions", [])]
    unsupported_numbers = find_unsupported_numbers(texts, allowed)
    valid_ids = {c.get("evidence_id") for c in research.get("evidence", [])}
    invalid_citations = find_invalid_citations(rec, valid_ids)
    unsupported_claims = [f"number not in deterministic data: {n}" for n in unsupported_numbers] + [
        f"unknown evidence id: {c}" for c in invalid_citations
    ]
    has_unsupported_claims = bool(unsupported_claims)
    if has_unsupported_claims:
        reasons.append("Unsupported claims found: " + "; ".join(unsupported_claims[:5]))

    if rec.get("error"):
        reasons.append(f"Investment Analyst Agent error: {rec['error']}")

    is_refusable = not rec.get("decision") or not rec.get("justification")
    if is_refusable:
        status = "refuse"
    elif has_unsupported_claims and reanalysis_count < MAX_REANALYSIS_RETRIES:
        status = "request_reanalysis"
    else:
        status = "human_review_required"

    requires_human_review = bool(
        state_like.get("requires_human_review")
        or high_risk
        or negative_cash_flow
        or confidence_below_threshold
        or missing_property_data
        or missing_market_data
        or conflicting_evidence
        or research_degraded
        or bool(high_concerns)
        or has_unsupported_claims
    )

    return {
        "guardrail_result": {
            "status": status,
            "reasons": reasons,
            "missing_property_data": missing_property_data,
            "missing_market_data": missing_market_data,
            "high_risk": high_risk,
            "negative_cash_flow": negative_cash_flow,
            "confidence_below_threshold": confidence_below_threshold,
            "conflicting_evidence": conflicting_evidence,
            "conflicts": conflicts,
            "research_status": research_status,
            "high_severity_concerns": high_concerns,
            "has_unsupported_claims": has_unsupported_claims,
            "unsupported_claims": unsupported_claims,
            "unsupported_numbers": unsupported_numbers,
            "invalid_citations": invalid_citations,
        },
        "requires_human_review": requires_human_review,
    }
