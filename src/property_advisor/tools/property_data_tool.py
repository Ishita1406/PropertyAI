"""Property Data Tool.

Retrieves Indian property details and location information. This
may be backed by MagicBricks/99acres/Housing.com/RapidAPI, or a mock dataset.
This implementation uses a curated mock dataset (no network access required),
matched against the free-text property address.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mock_properties.json")

with open(_DATA_PATH) as f:
    _PROPERTIES: dict[str, dict[str, Any]] = json.load(f)

INDIAN_PIN_CODE_RE = re.compile(r"\b(\d{6})\b")


def parse_address(address: str) -> dict[str, Any]:
    """Lightweight, deterministic parse of an Indian address string."""
    pin_match = INDIAN_PIN_CODE_RE.search(address)
    parts = [p.strip() for p in address.split(",") if p.strip()]
    return {
        "raw_address": address,
        "pin_code": pin_match.group(1) if pin_match else None,
        "address_parts": parts,
    }


def get_property_data(address: str) -> dict[str, Any]:
    """Look up property details by matching keywords in the address.

    Returns an empty dict if no match is found, signaling missing data to
    the calling agent (used to drive the missing-data retry routing).
    """
    normalized = address.lower()
    for slug, record in _PROPERTIES.items():
        if any(keyword in normalized for keyword in record["match_keywords"]):
            data = {k: v for k, v in record.items() if k != "match_keywords"}
            data["parsed_address"] = parse_address(address)
            data["locality_slug"] = slug
            return data
    return {}
