"""LangGraph wiring: nodes, conditional routing, human-in-the-loop.

Workflow (exactly two LLM agents, marked ★):

    validate_input ──invalid──────────────────────────────────────────┐
        │ valid                                                       │
    fetch_property_data ⇄ data_retry_increment (bounded)              │
        │                                                             │
    compute_financials (deterministic metrics, stress tests, risk)    │
        │                                                             │
    ★ research_agent  (Property Research & Due Diligence Agent)      │
        │                                                             │
    ★ investment_analyst_agent  ◄──────────────┐                      │
        │                                      │ reanalyse (bounded)  │
    validate_recommendation (deterministic) ───┤                      │
        │ refuse ──────────────────────────────┼──────────────────────┤
        │ human_review_required                │                      │
    human_review (interrupt) ── rejected ──────┘                      │
        │ approved                                                    │
    final_report ◄────────────────────────────────────────────────────┘
        │
       END

Human approval is required before every approved report (the only bypasses
are invalid input and an outright guardrail refusal, where there is nothing
valid to approve). State is checkpointed with MemorySaver so the interrupt can
be resumed by thread_id.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from property_advisor.agents.investment_analyst_agent import investment_analyst_agent
from property_advisor.agents.research_agent import research_agent
from property_advisor.config import MAX_DATA_RETRIES
from property_advisor.logging_utils import traced_router
from property_advisor.nodes import (
    compute_financials_node,
    data_retry_increment_node,
    fetch_property_data_node,
    final_report_node,
    human_review_node,
    route_after_validation,
    validate_input_node,
    validate_recommendation_node,
)
from property_advisor.state import PropertyState

# Re-exported for backward compatibility (previously defined here).
__all__ = [
    "build_graph",
    "route_after_data",
    "route_after_guardrail",
    "route_after_human",
    "human_review_node",
    "final_report_node",
]


@traced_router("route_after_validation")
def _route_after_validation(state: PropertyState) -> Literal["valid", "invalid"]:
    return route_after_validation(state)


@traced_router("route_after_data")
def route_after_data(state: PropertyState) -> Literal["retry", "continue"]:
    data_missing = not state.property_data or not state.market_data
    if data_missing and state.data_retry_count < MAX_DATA_RETRIES:
        return "retry"
    return "continue"


# Old name kept as an alias.
route_after_market = route_after_data


@traced_router("route_after_guardrail")
def route_after_guardrail(state: PropertyState) -> Literal["reanalyze", "refuse", "human_review"]:
    status = state.guardrail_result.get("status", "human_review_required")
    if status == "request_reanalysis":
        return "reanalyze"
    if status == "refuse":
        return "refuse"
    return "human_review"


@traced_router("route_after_human")
def route_after_human(state: PropertyState) -> Literal["approved", "rejected"]:
    return "approved" if state.human_decision.get("approved") else "rejected"


def build_graph(checkpointer=None):
    graph = StateGraph(PropertyState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("fetch_property_data", fetch_property_data_node)
    graph.add_node("data_retry_increment", data_retry_increment_node)
    graph.add_node("compute_financials", compute_financials_node)
    graph.add_node("research_agent", research_agent)
    graph.add_node("investment_analyst_agent", investment_analyst_agent)
    graph.add_node("validate_recommendation", validate_recommendation_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input", _route_after_validation, {"valid": "fetch_property_data", "invalid": "final_report"}
    )
    graph.add_conditional_edges(
        "fetch_property_data", route_after_data, {"retry": "data_retry_increment", "continue": "compute_financials"}
    )
    graph.add_edge("data_retry_increment", "fetch_property_data")
    graph.add_edge("compute_financials", "research_agent")
    graph.add_edge("research_agent", "investment_analyst_agent")
    graph.add_edge("investment_analyst_agent", "validate_recommendation")
    graph.add_conditional_edges(
        "validate_recommendation",
        route_after_guardrail,
        {"reanalyze": "investment_analyst_agent", "refuse": "final_report", "human_review": "human_review"},
    )
    graph.add_conditional_edges(
        "human_review", route_after_human, {"approved": "final_report", "rejected": "investment_analyst_agent"}
    )
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
