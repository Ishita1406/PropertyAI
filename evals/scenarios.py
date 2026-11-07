

SCENARIOS = [
    {
        "name": "high_growth_metro_buy",
        "description": "High-growth Indian metro (Bangalore IT corridor) -> expect BUY",
        "input": {
            "property_address": "Flat 4B, Whitefield, Bangalore, 560066",
            "budget": 9_500_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
        "expected": {"decision": "BUY"},
    },
    {
        "name": "negative_cash_flow_avoid",
        "description": "Negative cash flow property in Tier-1 city (Mumbai Worli) -> expect AVOID",
        "input": {
            "property_address": "Sea view flat, Worli, Mumbai",
            "budget": 45_000_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
        "expected": {"decision": "AVOID"},
    },
    {
        "name": "flood_prone_human_review",
        "description": "Property in flood-prone area (Mumbai low-lying zone) -> expect human review triggered",
        "input": {
            "property_address": "Flat near Dadar station, Dadar, Mumbai",
            "budget": 18_000_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
        "expected": {"requires_human_review": True, "risk_score_above": 75},
    },
    {
        "name": "missing_market_data_retry",
        "description": "Missing market data for Tier-3 city (Muzaffarpur) -> expect retry workflow",
        "input": {
            "property_address": "Plot near Saraiyaganj, Muzaffarpur, Bihar",
            "budget": 3_500_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "long_term_appreciation",
        },
        "expected": {"data_retry_count_above": 0, "market_data_empty": True},
    },
    {
        "name": "conflicting_reports_guardrail",
        "description": "Conflicting Indian market reports (Hinjewadi analyst notes) -> expect guardrail intervention",
        "input": {
            "property_address": "Hinjewadi Phase 2, Pune",
            "budget": 7_200_000.0,
            "investment_horizon_years": 5,
            "investment_strategy": "rental",
        },
        "expected": {"conflicting_evidence": True},
    },
]
