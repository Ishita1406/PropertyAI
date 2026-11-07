"""RAG retrieval tool.

Vector search over the knowledge base built by ``property_advisor.rag.ingest``:
official Indian legislation / government publications from ``data/`` plus the
project's SIMULATED market notes (``data/rag_corpus``), every chunk tagged
with ``source_type`` so the two are never confused.

Embeddings use ChromaDB's built-in local ONNX model (all-MiniLM-L6-v2), so no
embedding API key is needed — Groq (the only LLM provider) has no embeddings
endpoint.

``query_rag(query, k)`` keeps its original signature and still returns
``{"text", "source"}`` per hit; hits now also carry full document metadata
(title, publisher, section_number, page_number, ...) and ``chunk_id``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from property_advisor import config

logger = logging.getLogger("property_advisor.rag")

_collection = None
_lock = threading.Lock()


def set_collection(collection) -> None:
    """Inject a collection (used by tests and by the ingestion script)."""
    global _collection
    _collection = collection


def _get_collection():
    """Open the persistent collection, building it on first use.

    The first-use build skips OCR (it would take minutes inside a request);
    run ``python scripts/ingest_documents.py`` once to OCR scanned pages.
    """
    global _collection
    if _collection is not None:
        return _collection
    with _lock:
        if _collection is not None:
            return _collection
        from property_advisor.rag.ingest import get_collection, ingest

        collection = get_collection()
        if collection.count() == 0:
            logger.warning(
                "Knowledge base is empty - building it now without OCR. "
                "Run scripts/ingest_documents.py for a full (OCR) build."
            )
            ingest(collection=collection, enable_ocr=False)
        _collection = collection
        return _collection


def build_where(
    source_type: str | list[str] | None = None,
    document_type: str | list[str] | None = None,
    document_ids: list[str] | None = None,
    where: dict | None = None,
) -> dict | None:
    clauses: list[dict] = []
    for field, value in (("source_type", source_type), ("document_type", document_type), ("document_id", document_ids)):
        if not value:
            continue
        if isinstance(value, (list, tuple, set)):
            clauses.append({field: {"$in": list(value)}})
        else:
            clauses.append({field: value})
    if where:
        clauses.append(where)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def query_rag(
    query: str,
    k: int = 4,
    source_type: str | list[str] | None = None,
    document_type: str | list[str] | None = None,
    document_ids: list[str] | None = None,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Query the knowledge base with optional metadata filters.

    Returns a list of dicts: ``text``, ``source`` (file name), ``chunk_id``,
    ``distance`` and all stored metadata fields.
    """
    collection = _get_collection()
    flt = build_where(source_type, document_type, document_ids, where)
    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
    if flt:
        kwargs["where"] = flt
    results = collection.query(**kwargs)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[None] * len(documents)])[0]
    hits = []
    for doc, meta, cid, dist in zip(documents, metadatas, ids, distances):
        meta = dict(meta or {})
        hits.append(
            {
                **meta,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_id": cid,
                "distance": round(float(dist), 4) if dist is not None else None,
            }
        )
    return hits


def format_citation(hit: dict[str, Any]) -> str:
    """Human-readable citation, e.g.
    'The Registration Act, 1908 — s. 17, p. 12 [primary_legislation; official_government]'."""
    title = hit.get("document_title") or hit.get("source", "unknown source")
    parts = [title]
    section = hit.get("section_number")
    loc = []
    if section and section != "not_identified":
        loc.append(f"s. {section}" if hit.get("document_type") in ("primary_legislation", "model_legislation") else f"§ {section}")
    page, page_end = hit.get("page_number"), hit.get("page_end")
    if page:
        loc.append(f"p. {page}" if not page_end or page_end == page else f"pp. {page}-{page_end}")
    if loc:
        parts.append(", ".join(loc))
    tag = f"[{hit.get('document_type', 'unknown')}; {hit.get('source_type', 'unknown')}]"
    return " — ".join(parts) + " " + tag
