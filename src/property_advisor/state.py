"""Shared, typed LangGraph state for the Property Investment Advisor.

Nodes read/write only the fields relevant to their responsibility. Routing/
control fields are kept separate from domain data. Field names that existed
before the two-agent refactor are unchanged (API, frontend, evals and reports
depend on them).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PropertyState(BaseModel):
    # --- Inputs ---
    property_address: str
    budget: float  # INR
    investment_horizon_years: int = 5
    investment_strategy: Literal["rental", "flip", "long_term_appreciation"] = "rental"

    # --- Input validation ---
    input_validation: dict[str, Any] = Field(default_factory=dict)

    # --- Simulated data + deterministic calculations (supporting nodes) ---
    property_data: dict[str, Any] = Field(default_factory=dict)
    market_data: dict[str, Any] = Field(default_factory=dict)
    investment_metrics: dict[str, Any] = Field(default_factory=dict)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    data_provenance: dict[str, str] = Field(default_factory=dict)

    # --- Agent 1: Property Research & Due Diligence Agent ---
    research_findings: dict[str, Any] = Field(default_factory=dict)
    # Retrieved chunks (with metadata) the research agent saw — kept under the
    # original field name for backward compatibility.
    rag_context: list[dict[str, Any]] = Field(default_factory=list)

    # --- Agent 2: Investment Analyst Agent ---
    recommendation: dict[str, Any] = Field(default_factory=dict)

    # --- Deterministic validation / guardrails ---
    guardrail_result: dict[str, Any] = Field(default_factory=dict)

    # --- Human-in-the-loop ---
    requires_human_review: bool = False
    human_decision: dict[str, Any] = Field(default_factory=dict)
    final_report: dict[str, Any] = Field(default_factory=dict)

    # --- Routing / control ---
    data_retry_count: int = 0
    reanalysis_retry_count: int = 0
    workflow_status: str = "in_progress"
    errors: list[str] = Field(default_factory=list)
