"""Knowledge-base ingestion into ChromaDB.

Run with:  python scripts/ingest_documents.py  [--force] [--no-ocr]

Pipeline per document:
  discover (data/ + simulated corpus) -> SHA-256 -> duplicate check
  -> catalogue metadata -> extract (text layer, OCR only where needed)
  -> section-aware chunking with overlap -> embed (Chroma's built-in local
  all-MiniLM-L6-v2 ONNX model, same as before) -> upsert.

Re-ingestion is idempotent: a document whose SHA-256, catalogue version and
chunking parameters are unchanged (and whose chunks are present) is skipped.
A changed document has its old chunks deleted before new ones are written, so
no duplicate chunks accumulate. Identical files under different names are
detected by hash and ingested once. Documents removed from data/ have their
chunks removed from the collection.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from property_advisor import config
from property_advisor.rag.catalog import (
    CATALOG_VERSION,
    MISSING_DOCUMENTS,
    OFFICIAL_DOCUMENTS,
    simulated_metadata,
)
from property_advisor.rag.chunking import CHUNKER_VERSION, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from property_advisor.rag.extraction import extract_document, ocr_available, sha256_file

logger = logging.getLogger("property_advisor.rag")

SIMULATED_CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rag_corpus")
SUPPORTED_EXTENSIONS = (".pdf", ".txt")
CHUNKING_SIGNATURE = f"v{CHUNKER_VERSION}:{DEFAULT_CHUNK_SIZE}/{DEFAULT_CHUNK_OVERLAP}"
METADATA_FIELDS = (
    "document_id",
    "document_title",
    "publisher",
    "source_url",
    "document_type",
    "jurisdiction",
    "effective_date",
    "version",
    "source_type",
    "binding_status",
)


def _manifest_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, "ingestion_manifest.json")


def load_manifest(processed_dir: str) -> dict[str, Any]:
    path = _manifest_path(processed_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"documents": {}}


def _save_manifest(processed_dir: str, manifest: dict[str, Any]) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    with open(_manifest_path(processed_dir), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def discover_documents(source_dir: str, simulated_dir: str | None) -> list[dict[str, Any]]:
    """List candidate files with their catalogue metadata (top level only)."""
    found: list[dict[str, Any]] = []
    if os.path.isdir(source_dir):
        for name in sorted(os.listdir(source_dir)):
            path = os.path.join(source_dir, name)
            if not os.path.isfile(path) or not name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            meta = OFFICIAL_DOCUMENTS.get(name)
            if meta is None:
                meta = {
                    "document_id": "uncatalogued_" + os.path.splitext(name)[0].lower().replace(" ", "_"),
                    "document_title": name,
                    "include_in_rag": False,
                    "exclusion_reason": "Not in rag/catalog.py — add verified metadata before ingesting.",
                }
            found.append({"file": name, "path": path, "meta": dict(meta)})
    if simulated_dir and os.path.isdir(simulated_dir):
        for name in sorted(os.listdir(simulated_dir)):
            if name.endswith(".txt"):
                found.append(
                    {"file": name, "path": os.path.join(simulated_dir, name), "meta": simulated_metadata(name)}
                )
    return found


def _chunk_metadata(meta: dict[str, Any], filename: str, chunk, index: int) -> dict[str, Any]:
    md = {k: str(meta.get(k, "") or "") for k in METADATA_FIELDS}
    md.update(
        {
            "source": filename,  # kept for backward compatibility with older consumers
            "section_number": chunk.section_number,
            "sections_in_chunk": ",".join(chunk.sections_in_chunk)[:500],
            "page_number": int(chunk.page_number),
            "page_end": int(chunk.page_end),
            "chunk_index": index,
            "extraction_method": chunk.extraction_method,
            "content_sha": chunk.content_sha,
        }
    )
    return md


def get_collection(chroma_dir: str | None = None, collection_name: str | None = None, embedding_function=None):
    import chromadb
    from chromadb.utils import embedding_functions

    chroma_dir = chroma_dir or config.CHROMA_DIR
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    ef = embedding_function or embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name or config.KB_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _existing_chunk_count(collection, document_id: str) -> int:
    got = collection.get(where={"document_id": document_id}, include=[])
    return len(got.get("ids", []))


def ingest(
    source_dir: str | None = None,
    processed_dir: str | None = None,
    collection=None,
    include_simulated: bool = True,
    simulated_dir: str | None = SIMULATED_CORPUS_DIR,
    enable_ocr: bool | None = None,
    force: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Ingest/refresh the knowledge base. Returns a report dict."""
    source_dir = source_dir or config.KB_SOURCE_DIR
    processed_dir = processed_dir or config.KB_PROCESSED_DIR
    enable_ocr = config.KB_ENABLE_OCR if enable_ocr is None else enable_ocr
    collection = collection if collection is not None else get_collection()
    manifest = load_manifest(processed_dir)
    docs_state: dict[str, Any] = manifest.setdefault("documents", {})
    extracted_dir = os.path.join(processed_dir, "extracted")

    report: dict[str, Any] = {
        "ingested": [],
        "skipped_unchanged": [],
        "excluded": [],
        "duplicates": [],
        "needs_ocr": [],
        "ocr_pages": {},
        "removed": [],
        "errors": [],
        "missing_documents": MISSING_DOCUMENTS,
        "ocr_available": ocr_available(),
    }

    seen_hashes: dict[str, str] = {}
    present_ids: set[str] = set()
    candidates = discover_documents(source_dir, simulated_dir if include_simulated else None)

    for cand in candidates:
        name, path, meta = cand["file"], cand["path"], cand["meta"]
        doc_id = meta["document_id"]
        sha = sha256_file(path)
        if sha in seen_hashes:
            report["duplicates"].append({"file": name, "duplicate_of": seen_hashes[sha]})
            docs_state[doc_id] = {"file": name, "sha256": sha, "status": "duplicate", "duplicate_of": seen_hashes[sha]}
            continue
        seen_hashes[sha] = name

        if not meta.get("include_in_rag", False):
            report["excluded"].append({"file": name, "reason": meta.get("exclusion_reason", "")})
            docs_state[doc_id] = {"file": name, "sha256": sha, "status": "excluded", "reason": meta.get("exclusion_reason", "")}
            # Ensure no stale chunks of an excluded document remain.
            if _existing_chunk_count(collection, doc_id):
                collection.delete(where={"document_id": doc_id})
            continue

        present_ids.add(doc_id)
        prev = docs_state.get(doc_id, {})
        unchanged = (
            not force
            and prev.get("sha256") == sha
            and prev.get("catalog_version") == CATALOG_VERSION
            and prev.get("chunking") == CHUNKING_SIGNATURE
            and prev.get("status") == "ingested"
            and not (prev.get("needs_ocr_pages") and enable_ocr and ocr_available())
            and _existing_chunk_count(collection, doc_id) == prev.get("chunks")
        )
        if unchanged:
            report["skipped_unchanged"].append(name)
            if prev.get("needs_ocr_pages"):
                report["needs_ocr"].append({"file": name, "pages": prev["needs_ocr_pages"]})
            if prev.get("ocr_pages"):
                report["ocr_pages"][name] = prev["ocr_pages"]
            continue

        try:
            pages = extract_document(path, cache_dir=extracted_dir, sha256=sha, enable_ocr=enable_ocr)
            chunks = chunk_pages(pages, section_style=meta.get("section_style", "none"))
            # Drop exact-duplicate chunk texts within the document (e.g. repeated headers).
            unique, seen_content = [], set()
            for c in chunks:
                if c.content_sha in seen_content:
                    continue
                seen_content.add(c.content_sha)
                unique.append(c)

            collection.delete(where={"document_id": doc_id})
            ids = [f"{doc_id}::{sha[:12]}::{i:05d}" for i in range(len(unique))]
            for b in range(0, len(unique), batch_size):
                part = unique[b : b + batch_size]
                collection.add(
                    ids=ids[b : b + batch_size],
                    documents=[c.text for c in part],
                    metadatas=[_chunk_metadata(meta, name, c, b + j) for j, c in enumerate(part)],
                )

            ocr_pages = [p.page_number for p in pages if p.method == "ocr"]
            pending = [p.page_number for p in pages if p.method == "needs_ocr"]
            docs_state[doc_id] = {
                "file": name,
                "sha256": sha,
                "status": "ingested",
                "catalog_version": CATALOG_VERSION,
                "chunking": CHUNKING_SIGNATURE,
                "pages": len(pages),
                "chunks": len(unique),
                "ocr_pages": ocr_pages,
                "needs_ocr_pages": pending,
                "source_type": meta.get("source_type"),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            report["ingested"].append({"file": name, "pages": len(pages), "chunks": len(unique)})
            if ocr_pages:
                report["ocr_pages"][name] = ocr_pages
            if pending:
                report["needs_ocr"].append({"file": name, "pages": pending})
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingestion failed for %s", name)
            report["errors"].append({"file": name, "error": str(exc)})
            docs_state[doc_id] = {"file": name, "sha256": sha, "status": "error", "error": str(exc)}

    # Remove documents that disappeared from the source folders.
    for doc_id, st in list(docs_state.items()):
        if st.get("status") == "ingested" and doc_id not in present_ids:
            collection.delete(where={"document_id": doc_id})
            report["removed"].append(st.get("file", doc_id))
            del docs_state[doc_id]

    manifest["catalog_version"] = CATALOG_VERSION
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["missing_documents"] = MISSING_DOCUMENTS
    _save_manifest(processed_dir, manifest)
    report["total_chunks"] = collection.count()
    return report
