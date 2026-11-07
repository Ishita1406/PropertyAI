"""Runs the 5 required evaluation scenarios end-to-end through the LangGraph
workflow (auto-approving human review so it can run non-interactively) and
checks each scenario's expected outcome.

Requires GROQ_API_KEY (the Recommendation and Guardrail agents call Groq).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.types import Command  # noqa: E402

from property_advisor.config import GROQ_API_KEY  # noqa: E402
from property_advisor.graph import build_graph  # noqa: E402

from scenarios import SCENARIOS  # noqa: E402


def run_scenario(graph, scenario: dict) -> dict:
    config = {"configurable": {"thread_id": f"eval-{scenario['name']}"}}
    result = graph.invoke(scenario["input"], config)

    while "__interrupt__" in result:
        result = graph.invoke(Command(resume={"approved": True, "feedback": ""}), config)

    return result


def check_expectations(result: dict, expected: dict) -> list[str]:
    failures = []

    if "decision" in expected:
        actual = result.get("recommendation", {}).get("decision")
        if actual != expected["decision"]:
            failures.append(f"expected decision={expected['decision']!r}, got {actual!r}")

    if "requires_human_review" in expected:
        actual = result.get("requires_human_review")
        if actual != expected["requires_human_review"]:
            failures.append(f"expected requires_human_review={expected['requires_human_review']}, got {actual}")

    if "risk_score_above" in expected:
        score = result.get("risk_assessment", {}).get("risk_score", 0)
        if not score > expected["risk_score_above"]:
            failures.append(f"expected risk_score > {expected['risk_score_above']}, got {score}")

    if "data_retry_count_above" in expected:
        count = result.get("data_retry_count", 0)
        if not count > expected["data_retry_count_above"]:
            failures.append(f"expected data_retry_count > {expected['data_retry_count_above']}, got {count}")

    if "market_data_empty" in expected:
        is_empty = not result.get("market_data")
        if is_empty != expected["market_data_empty"]:
            failures.append(f"expected market_data_empty={expected['market_data_empty']}, got {not is_empty}")

    if "conflicting_evidence" in expected:
        actual = result.get("guardrail_result", {}).get("conflicting_evidence")
        if actual != expected["conflicting_evidence"]:
            failures.append(f"expected conflicting_evidence={expected['conflicting_evidence']}, got {actual}")

    return failures


def main() -> None:
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key, then re-run.")
        sys.exit(1)

    graph = build_graph()
    passed, failed = 0, 0

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']}: {scenario['description']} ---")
        try:
            result = run_scenario(graph, scenario)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR running scenario: {exc}")
            failed += 1
            continue

        failures = check_expectations(result, scenario["expected"])
        if failures:
            print("FAIL")
            for f in failures:
                print(f"  - {f}")
            failed += 1
        else:
            print("PASS")
            passed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(SCENARIOS)} scenarios")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
