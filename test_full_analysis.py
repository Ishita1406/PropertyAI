#!/usr/bin/env python
"""Test the full analysis pipeline for different properties."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from property_advisor.graph import build_graph
from property_advisor.state import PropertyState

def analyze_property(address, budget=9500000, horizon=5, strategy="rental", thread_id="test"):
    """Run the full analysis and print results."""
    print(f"\n{'='*70}")
    print(f"📊 Analyzing: {address}")
    print(f"   Budget: ₹{budget:,.0f}")
    print(f"   Horizon: {horizon} years")
    print(f"   Strategy: {strategy}")
    print('='*70)
    
    # Build graph
    graph = build_graph()
    
    # Create initial state
    initial_state = {
        "property_address": address,
        "budget": budget,
        "investment_horizon_years": horizon,
        "investment_strategy": strategy,
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph
    result = graph.invoke(initial_state, config)
    
    # Get the recommendation
    rec = result.get("recommendation", {})
    guardrail = result.get("guardrail_result", {})
    metrics = result.get("investment_metrics", {})
    risk = result.get("risk_assessment", {})
    rag_context = result.get("rag_context", [])
    
    # Print results
    print(f"\n💡 RECOMMENDATION: {rec.get('decision', 'N/A')}")
    print(f"   Confidence: {rec.get('confidence_score', 'N/A')}")
    print(f"   Justification: {rec.get('justification', 'N/A')[:200]}...")
    
    print(f"\n🛡️ GUARDRAIL STATUS: {guardrail.get('status', 'N/A')}")
    reasons = guardrail.get('reasons', [])
    if reasons:
        print("   Reasons:")
        for r in reasons:
            print(f"     - {r}")
    else:
        print("   No guardrail issues flagged")
    
    print(f"\n📈 METRICS:")
    print(f"   ROI: {metrics.get('roi_pct', 'N/A')}%")
    print(f"   Rental Yield: {metrics.get('rental_yield_pct', 'N/A')}%")
    print(f"   Annual Cash Flow: ₹{metrics.get('annual_cash_flow_inr', 'N/A'):,.0f}")
    print(f"   Cap Rate: {metrics.get('cap_rate_pct', 'N/A')}%")
    print(f"   Break-even: {metrics.get('break_even_years', 'N/A')} years")
    
    print(f"\n⚠️ RISK SCORE: {risk.get('risk_score', 'N/A')}/100")
    factors = risk.get('factors', {})
    if factors:
        print("   Factors:")
        for k, v in list(factors.items())[:5]:
            print(f"     - {k}: {v}")
    
    print(f"\n📚 RAG CONTEXT ({len(rag_context)} documents):")
    for doc in rag_context[:5]:
        print(f"   - {doc.get('source', 'Unknown')}")
    
    # Check if there was an interrupt (human approval required)
    if "__interrupt__" in result:
        print("\n⏸️ HUMAN APPROVAL REQUIRED")
        print(f"   Interrupt: {result['__interrupt__'][0].value}")
    
    return result

if __name__ == "__main__":
    # Test different properties
    test_cases = [
        ("Whitefield, Bangalore, 560066", 9500000, 5, "rental"),
        ("Worli, Mumbai", 45000000, 5, "rental"),
        ("Dadar, Mumbai", 18000000, 5, "rental"),
        ("Hinjewadi, Pune", 15000000, 5, "rental"),
    ]
    
    for address, budget, horizon, strategy in test_cases:
        try:
            result = analyze_property(address, budget, horizon, strategy)
        except Exception as e:
            print(f"\n❌ ERROR analyzing {address}: {e}")
            import traceback
            traceback.print_exc()
        print("\n" + "="*70 + "\n")