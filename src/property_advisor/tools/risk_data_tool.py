"""Risk Data Tool.

Retrieves crime, flood, vacancy, and regulatory risk indicators for a locality
and deterministically aggregates them into a single risk_score (0-100, higher
is riskier). This calculation is NOT LLM-based, risk/financial calculations be deterministic and tool-based.
"""

from __future__ import annotations

import json
import os
from typing import Any

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mock_risk_data.json")

with open(_DATA_PATH) as f:
    _RISK_DATA: dict[str, dict[str, Any]] = json.load(f)

_QUALITATIVE_WEIGHT = {"low": 10, "medium": 25, "high": 45}

# Flood risk is weighted most heavily: a single monsoon flood event can make a
# property uninhabitable or illiquid, unlike the other softer risk factors.
_FLOOD_WEIGHT_MULTIPLIER = 1.4


def get_risk_data(slug: str | None) -> dict[str, Any]:
    """Look up qualitative risk indicators by locality_slug.

    Returns {} if unavailable, signaling low data quality to the caller.
    """
    if slug and slug in _RISK_DATA:
        return dict(_RISK_DATA[slug])
    return {}


def compute_risk_score(risk_data: dict[str, Any]) -> dict[str, Any]:
    """Deterministically combine qualitative risk indicators into a 0-100 score."""
    if not risk_data:
        return {
            "risk_score": 50.0,  # unknown risk is treated as moderate-high, not zero
            "data_quality_confidence": 0.0,
            "factors": {},
        }

    factors = {
        "crime_risk": min(float(risk_data.get("crime_score", 50)), 100.0),
        "flood_risk": _QUALITATIVE_WEIGHT.get(risk_data.get("flood_risk", "medium"), 25)
        * _FLOOD_WEIGHT_MULTIPLIER,
        "vacancy_risk": _QUALITATIVE_WEIGHT.get(risk_data.get("vacancy_risk", "medium"), 25),
        "regulatory_risk": _QUALITATIVE_WEIGHT.get(risk_data.get("regulatory_risk", "medium"), 25),
        "market_volatility": _QUALITATIVE_WEIGHT.get(risk_data.get("market_volatility", "medium"), 25),
    }
    weights = {
        "crime_risk": 0.15,
        "flood_risk": 0.35,
        "vacancy_risk": 0.20,
        "regulatory_risk": 0.15,
        "market_volatility": 0.15,
    }
    risk_score = sum(factors[k] * weights[k] for k in factors)

    # A "high" flood-risk classification (e.g. known monsoon waterlogging
    # zones) is treated as a hard red flag that overrides the weighted
    # average, rather than just nudging it — a single severe physical risk
    # shouldn't be diluted away by otherwise-average factors.
    if risk_data.get("flood_risk") == "high":
        risk_score = max(risk_score, 80.0)

    risk_score = max(0.0, min(100.0, risk_score))

    return {
        "risk_score": round(risk_score, 1),
        "data_quality_confidence": 1.0,
        "factors": {k: round(v, 1) for k, v in factors.items()},
    }
