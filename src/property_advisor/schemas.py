"""Structured-output schemas for the two LLM agents.

Field descriptions ask for brevity — these are JSON outputs, and every word is
an output token. The LLM never produces financial figures of its own: numbers
it mentions are checked against deterministic values by
``validation.find_unsupported_numbers``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Agent 1 — Property Research & Due Diligence Agent
# ---------------------------------------------------------------------------


class RegulatoryRequirement(BaseModel):
    requirement: str = Field(description="One sentence: what the law/guideline requires or provides.")
    relevance: str = Field(description="One short sentence: why it matters for THIS property/strategy.")
    citation_ids: list[str] = Field(default_factory=list, description="Evidence IDs like 'E3' from the retrieved list ONLY.")


class PropertyConcern(BaseModel):
    concern: str = Field(description="One sentence.")
    severity: Literal["low", "medium", "high"]
    basis: Literal["simulated_data", "official_document", "both", "missing_evidence"]
    citation_ids: list[str] = Field(default_factory=list, description="Evidence IDs like 'E3' (may be empty for simulated_data).")


class EvidenceGap(BaseModel):
    gap: str = Field(description="What evidence is missing or outdated.")
    why_it_matters: str = Field(description="One short sentence.")


class ResearchFindingsOutput(BaseModel):
    summary: str = Field(description="Max 3 sentences.")
    regulatory_requirements: list[RegulatoryRequirement] = Field(default_factory=list, description="Up to 6.")
    concerns: list[PropertyConcern] = Field(default_factory=list, description="Up to 6.")
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list, description="Up to 5.")
    conflicting_evidence: list[str] = Field(default_factory=list, description="Up to 3; name the evidence IDs that conflict.")
    due_diligence_checklist: list[str] = Field(default_factory=list, description="Up to 6 short buyer actions.")


# ---------------------------------------------------------------------------
# Agent 2 — Investment Analyst Agent
# ---------------------------------------------------------------------------


class RecommendationOutput(BaseModel):
    decision: Literal["BUY", "HOLD", "AVOID"]
    justification: str = Field(description="Max 3 short sentences. Use only numbers given to you.")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Up to 4 short items, e.g. 'roi_pct=28.7' or 'E2: RERA registration required'.",
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_risks: list[str] = Field(default_factory=list, description="Up to 4 short items.")
    assumptions: list[str] = Field(default_factory=list, description="Up to 3 qualitative assumptions you relied on.")
    cited_evidence_ids: list[str] = Field(default_factory=list, description="Evidence IDs (e.g. 'E2') from the research findings that you relied on.")
