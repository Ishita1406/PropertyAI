"""CLI entrypoint for the Property Investment Advisor.

Runs the LangGraph workflow end-to-end for a single property (or, with
--demo, for three fixed scenarios back-to-back), pausing for human approval
via LangGraph's interrupt mechanism before any final report is produced. After approval, JSON/Markdown/PDF
reports are generated automatically (see report_generator.py).
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from langgraph.types import Command  # noqa: E402

from property_advisor.config import AUTO_APPROVE_HUMAN_REVIEW  # noqa: E402
from property_advisor.graph import build_graph  # noqa: E402
from property_advisor.report_generator import generate_reports  # noqa: E402

DEMO_CASES = [
    {
        "name": "Test Case 1: High-growth Bangalore property",
        "input": {
            "property_address": "Whitefield, Bangalore, 560066",
            "budget": 9_500_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
    },
    {
        "name": "Test Case 2: Negative cash flow property",
        "input": {
            "property_address": "Worli, Mumbai",
            "budget": 45_000_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
    },
    {
        "name": "Test Case 3: High risk (flood-prone) property",
        "input": {
            "property_address": "Dadar, Mumbai",
            "budget": 18_000_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Property Investment Advisor")
    parser.add_argument("--address", help="Property address (India-based)")
    parser.add_argument("--budget", type=float, help="Budget in INR")
    parser.add_argument("--horizon", type=int, default=5, help="Investment horizon in years")
    parser.add_argument(
        "--strategy",
        choices=["rental", "flip", "long_term_appreciation"],
        default="rental",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=AUTO_APPROVE_HUMAN_REVIEW,
        help="Auto-approve human review steps (non-interactive mode for testing)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip JSON/Markdown/PDF report generation after approval",
    )
    parser.add_argument("--reports-dir", default="reports", help="Directory to write generated reports into")
    parser.add_argument("--thread-id", default="cli-session", help="LangGraph checkpoint thread id")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run 3 fixed demo scenarios (high-growth, negative cash flow, high risk) and generate reports for each",
    )
    return parser.parse_args()


def prompt_human_decision(payload: dict, auto_approve: bool) -> dict:
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 70)
    print(json.dumps(payload, indent=2, default=str))
    print("=" * 70)

    if auto_approve:
        print("[auto-approve mode] Approving without prompting.")
        return {"approved": True, "feedback": ""}

    while True:
        choice = input("Approve this recommendation? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return {"approved": True, "feedback": ""}
        if choice in ("n", "no"):
            feedback = input("Feedback for re-analysis: ").strip()
            return {"approved": False, "feedback": feedback}
        print("Please answer y or n.")


def run_pipeline(graph, initial_state: dict, thread_id: str, auto_approve: bool) -> dict:
    """Runs the graph to completion, handling the human-approval interrupt
    loop, and returns the final_report dict."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config)

    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        decision = prompt_human_decision(interrupt_obj.value, auto_approve)
        result = graph.invoke(Command(resume=decision), config)

    return result.get("final_report", {})


def generate_and_print_reports(final_report: dict, reports_dir: str) -> None:
    if final_report.get("status") != "approved":
        print("\nSkipping report generation: recommendation was not approved.")
        return

    paths = generate_reports(final_report, output_dir=reports_dir)
    print("\n✓ Recommendation approved")
    print("✓ JSON report generated")
    print("✓ Markdown report generated")
    print("✓ PDF report generated")
    print(f"\nReport saved to:\n  {paths['pdf']}")


def run_single(args: argparse.Namespace) -> None:
    address = args.address or input("Property address (India): ").strip()
    budget = args.budget if args.budget is not None else float(input("Budget (INR): ").strip())

    graph = build_graph()
    initial_state = {
        "property_address": address,
        "budget": budget,
        "investment_horizon_years": args.horizon,
        "investment_strategy": args.strategy,
    }

    final_report = run_pipeline(graph, initial_state, args.thread_id, args.auto_approve)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(json.dumps(final_report, indent=2, default=str))

    if not args.no_reports:
        generate_and_print_reports(final_report, args.reports_dir)


def run_demo(args: argparse.Namespace) -> None:
    graph = build_graph()
    print("=" * 70)
    print(f"DEMO MODE — running {len(DEMO_CASES)} fixed scenarios with auto-approval")
    print("=" * 70)

    for i, case in enumerate(DEMO_CASES, start=1):
        print(f"\n{'-' * 70}\n{case['name']}\n{'-' * 70}")
        final_report = run_pipeline(graph, case["input"], thread_id=f"demo-{i}", auto_approve=True)
        decision = final_report.get("recommendation", {}).get("decision", "N/A")
        print(f"Decision: {decision} | status: {final_report.get('status')}")
        generate_and_print_reports(final_report, args.reports_dir)

    print(f"\n{'=' * 70}\nDemo complete — see ./{args.reports_dir}/ for all generated reports\n{'=' * 70}")


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
