"""Market Data Tool.

Retrieves historical prices, appreciation rates, and demand/supply trends for
Indian cities/localities. Returns an empty dict when no data is available for
a locality (e.g. Tier-3 cities), which drives the missing-data retry routing.
"""

from __future__ import annotations

import json
import os
from typing import Any

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mock_market_trends.json")

with open(_DATA_PATH) as f:
    _MARKET_TRENDS: dict[str, dict[str, Any]] = json.load(f)


def get_market_data(city: str, locality: str | None = None, slug: str | None = None) -> dict[str, Any]:
    """Look up market trend data, preferring an exact locality_slug match
    (carried over from the Property Agent) and falling back to city/locality
    text matching. Returns {} when no data is available, e.g. for Tier-3
    cities not covered by the mock dataset — this signals missing data to
    the calling agent and drives the missing-data retry routing.
    """
    if slug and slug in _MARKET_TRENDS:
        return dict(_MARKET_TRENDS[slug])
    if not city:
        return {}
    city_norm = city.lower()
    locality_norm = (locality or "").lower()
    for record in _MARKET_TRENDS.values():
        if record["city"].lower() == city_norm and (
            not locality_norm or locality_norm in record["locality"].lower()
        ):
            return dict(record)
    return {}
