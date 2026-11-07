#!/usr/bin/env python
"""Build or refresh the ChromaDB knowledge base from data/ (+ simulated corpus).

    python scripts/ingest_documents.py            # incremental, OCR where needed
    python scripts/ingest_documents.py --force    # re-process every document
    python scripts/ingest_documents.py --no-ocr   # skip OCR (garbled pages are skipped, not indexed)

Source files in data/ are opened read-only. Generated artifacts go to
chroma_db/ (vectors) and data_processed/ (extracted/OCR text cache, manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from property_advisor.rag.ingest import ingest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-process all documents even if unchanged")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for garbled/scanned pages")
    parser.add_argument("--no-simulated", action="store_true", help="Do not index the simulated market notes")
    args = parser.parse_args()

    report = ingest(force=args.force, enable_ocr=not args.no_ocr, include_simulated=not args.no_simulated)

    print("\n=== Ingestion report ===")
    for row in report["ingested"]:
        print(f"  ingested   {row['file']}: {row['pages']} pages -> {row['chunks']} chunks")
    for name in report["skipped_unchanged"]:
        print(f"  unchanged  {name}")
    for row in report["excluded"]:
        print(f"  EXCLUDED   {row['file']}: {row['reason']}")
    for row in report["duplicates"]:
        print(f"  DUPLICATE  {row['file']} (same content as {row['duplicate_of']})")
    for name, pages in report["ocr_pages"].items():
        print(f"  OCR used   {name}: {len(pages)} pages")
    for row in report["needs_ocr"]:
        print(f"  NEEDS OCR  {row['file']}: pages {row['pages'][:20]}{'...' if len(row['pages']) > 20 else ''}")
    for name in report["removed"]:
        print(f"  removed    {name}")
    for row in report["errors"]:
        print(f"  ERROR      {row['file']}: {row['error']}")
    print("\nMissing documents (not in data/):")
    for doc in report["missing_documents"]:
        print(f"  - {doc['document_title']}\n      official source: {doc['official_source_url']}")
    print(f"\nOCR available: {report['ocr_available']}   total chunks in collection: {report['total_chunks']}")
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
