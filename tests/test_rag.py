"""Test RAG retrieval for property addresses (integration test against the
real ChromaDB knowledge base built from data/).

Originally this exercised the deterministic ``rag_agent`` node. That node was
folded into the Property Research & Due Diligence Agent, whose retrieval goes
through ``query_rag``; the same Whitefield scenario is exercised here with
the retrieval layer directly, now with assertions.

The first run builds the knowledge base if it does not exist yet (no OCR,
~1-2 min). Run ``python scripts/ingest_documents.py`` beforehand for the full
OCR build.
"""

import sys
from pathlib import Path

# Get the project root (where src/ is located)
project_root = Path(__file__).parent.parent  # Go up from tests/ to root
src_path = project_root / "src"

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from property_advisor.tools.property_data_tool import get_property_data  # noqa: E402
from property_advisor.tools.rag_tool import format_citation, query_rag  # noqa: E402


def test_rag():
    """Test RAG retrieval."""
    print("Testing RAG...")

    property_data = get_property_data("Whitefield, Bangalore, 560066")
    query = (
        f"{property_data['locality']} {property_data['city']} real estate appreciation trends, zoning and RERA "
        "regulations, infrastructure/metro development, flood and regulatory risk"
    )
    results = query_rag(query, k=6)

    print(f"Found {len(results)} documents")
    for doc in results[:3]:
        print(f"  - {doc.get('source', 'Unknown')}")

    assert results, "knowledge base returned no results"
    for doc in results:
        assert doc["text"]
        assert doc["source"]
        assert doc["document_title"]
        assert doc["source_type"] in ("official_government", "secondary", "simulated")


def test_rag_official_filter_returns_cited_legislation():
    hits = query_rag(
        "registration of real estate project with the Real Estate Regulatory Authority before sale",
        k=4,
        source_type="official_government",
        document_type="primary_legislation",
    )
    assert hits
    assert all(h["source_type"] == "official_government" for h in hits)
    assert any(h["document_id"] == "rera_act_2016" for h in hits)
    citation = format_citation(hits[0])
    assert "p." in citation or "pp." in citation


if __name__ == "__main__":
    test_rag()
