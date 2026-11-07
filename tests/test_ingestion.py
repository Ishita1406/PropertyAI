"""Document ingestion, metadata preservation, duplicate handling, re-ingestion
and retrieval/citation tests. Uses a generated PDF in a temp directory, an
in-test catalogue and a deterministic hashing embedding (fully offline)."""

from __future__ import annotations

import hashlib

import chromadb
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import property_advisor.rag.ingest as ingest_mod
from property_advisor.rag.chunking import chunk_pages
from property_advisor.rag.extraction import PageText, extract_document, page_needs_ocr, text_quality
from property_advisor.tools import rag_tool
from tests.conftest import HashEmbedding

FILLER = (
    "The promoter shall disclose the sanctioned plans, layout plans and specifications of the project "
    "and the stage-wise time schedule of completion including the provisions for civic infrastructure. "
)


def _make_pdf(path, pages: list[list[str]]):
    c = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()


def _meta(doc_id, title, **kw):
    base = {
        "document_id": doc_id,
        "document_title": title,
        "publisher": "Test Publisher",
        "source_url": "https://example.invalid/" + doc_id,
        "source_url_precision": "document_page",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (test)",
        "effective_date": "2020-01-01",
        "version": "v1",
        "section_style": "legislation",
        "include_in_rag": True,
    }
    base.update(kw)
    return base


@pytest.fixture
def kb(tmp_path, monkeypatch):
    src = tmp_path / "data"
    src.mkdir()
    processed = tmp_path / "data_processed"
    long_section = [(FILLER * 6)[i : i + 90] for i in range(0, len(FILLER) * 6, 90)]
    _make_pdf(
        src / "test_act.pdf",
        [
            ["THE TEST ACT, 2020", "ARRANGEMENT OF SECTIONS", "1. Short title.", "2. Definitions.", "3. Registration of projects."],
            ["THE TEST ACT, 2020", "BE it enacted by Parliament as follows:-", "1. (1) This Act may be called the Test Act, 2020.",
             "2. In this Act, unless the context otherwise requires, promoter means a person who builds."],
            ["3. Registration of projects.-(1) No promoter shall advertise or sell any apartment without registering",
             "the real estate project with the Authority."] + long_section,
        ],
    )
    (src / "copy_of_test_act.pdf").write_bytes((src / "test_act.pdf").read_bytes())
    (src / "commentary.txt").write_text("A blog explaining that tenants should register leases. " * 10)
    (src / "old_tax.pdf").write_bytes((src / "test_act.pdf").read_bytes()[:-10] + b"%%EOF-old\n")
    (src / "notes.md").write_text("ignored: unsupported extension")

    catalogue = {
        "test_act.pdf": _meta("test_act_2020", "The Test Act, 2020"),
        "copy_of_test_act.pdf": _meta("test_act_copy", "The Test Act, 2020 (copy)"),
        "commentary.txt": _meta("blog", "Blog commentary", document_type="secondary_commentary", source_type="secondary",
                                binding_status="secondary_commentary_not_authoritative", section_style="none"),
        "old_tax.pdf": _meta("old_tax", "Old Tax Act", include_in_rag=False, exclusion_reason="outdated"),
    }
    monkeypatch.setattr(ingest_mod, "OFFICIAL_DOCUMENTS", catalogue)

    client = chromadb.EphemeralClient()
    name = "test_kb_" + hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]
    collection = client.get_or_create_collection(name=name, embedding_function=HashEmbedding())

    def run(**kw):
        return ingest_mod.ingest(
            source_dir=str(src), processed_dir=str(processed), collection=collection,
            include_simulated=False, enable_ocr=False, **kw,
        )

    yield {"src": src, "processed": processed, "collection": collection, "run": run}
    client.delete_collection(name)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_ingestion_metadata_duplicates_exclusions_and_source_preservation(kb):
    before = {p.name: _sha(p) for p in kb["src"].iterdir()}
    report = kb["run"]()
    after = {p.name: _sha(p) for p in kb["src"].iterdir()}
    assert before == after, "source documents must not be modified"

    ingested = {r["file"] for r in report["ingested"]}
    # Identical files: exactly one copy is ingested (discovery is alphabetical), the other is reported.
    assert ingested == {"copy_of_test_act.pdf", "commentary.txt"}
    assert report["duplicates"] == [{"file": "test_act.pdf", "duplicate_of": "copy_of_test_act.pdf"}]
    assert [r["file"] for r in report["excluded"]] == ["old_tax.pdf"]

    got = kb["collection"].get(include=["metadatas", "documents"])
    assert got["ids"]
    required = ("document_id", "document_title", "publisher", "source_url", "document_type", "jurisdiction",
                "section_number", "page_number", "effective_date", "version", "source_type")
    for md in got["metadatas"]:
        for key in required:
            assert key in md and md[key] not in (None,), key
    assert not any(md["document_id"] == "old_tax" for md in got["metadatas"])

    # Section tracking: body text of s.3 is on page 3; TOC page 1 is not used for section numbers.
    act_chunks = [(d, m) for d, m in zip(got["documents"], got["metadatas"]) if m["source"] in ("test_act.pdf", "copy_of_test_act.pdf")]
    s3 = [m for d, m in act_chunks if "No promoter shall advertise" in d]
    assert s3 and s3[0]["page_end"] == 3 and "3" in s3[0]["sections_in_chunk"].split(",")
    later = [m for d, m in act_chunks if m["page_number"] == 3]
    assert later and all(m["section_number"] == "3" for m in later)

    # Manifest written outside data/
    assert (kb["processed"] / "ingestion_manifest.json").exists()
    assert not any(p.suffix == ".json" for p in kb["src"].iterdir())


def test_reingestion_is_idempotent_and_updates_changed_documents(kb):
    kb["run"]()
    count1 = kb["collection"].count()
    report2 = kb["run"]()
    assert report2["ingested"] == []
    assert set(report2["skipped_unchanged"]) == {"copy_of_test_act.pdf", "commentary.txt"}
    assert len(report2["duplicates"]) == 1
    assert kb["collection"].count() == count1

    # Change one document -> only it is re-processed, old chunks replaced (no duplicates).
    (kb["src"] / "commentary.txt").write_text("Updated commentary about stamp duty. " * 5)
    report3 = kb["run"]()
    assert [r["file"] for r in report3["ingested"]] == ["commentary.txt"]
    blog = kb["collection"].get(where={"document_id": "blog"}, include=["documents"])
    assert all("Updated commentary" in d for d in blog["documents"])
    assert len(set(blog["ids"])) == len(blog["ids"])

    # Remove a document -> its chunks are removed.
    (kb["src"] / "commentary.txt").unlink()
    report4 = kb["run"]()
    assert "commentary.txt" in report4["removed"]
    assert kb["collection"].get(where={"document_id": "blog"})["ids"] == []


def test_retrieval_filters_and_citations(kb):
    kb["run"]()
    rag_tool.set_collection(kb["collection"])
    try:
        hits = rag_tool.query_rag("promoter registering real estate project", k=3, source_type="official_government")
        assert hits and all(h["source_type"] == "official_government" for h in hits)
        top = hits[0]
        for key in ("chunk_id", "document_title", "section_number", "page_number", "publisher", "source_url"):
            assert key in top
        cit = rag_tool.format_citation(top)
        assert "The Test Act, 2020" in cit and "p." in cit and "[primary_legislation; official_government]" in cit

        sec = rag_tool.query_rag("register leases", k=3, document_type="secondary_commentary")
        assert sec and all(h["document_type"] == "secondary_commentary" for h in sec)
    finally:
        rag_tool.set_collection(None)


def test_text_quality_and_ocr_detection():
    good = "17. Documents of which registration is compulsory.—(1) The following documents shall be registered"
    garbled = "ǡʹͲͳͶ Ǥ Ȁ ǣȀȀǤǤ ǡ ǡ Ǥ ǯ ȋ Ȍ ǡ Ǯ ǯ ǡ ǡ Ǥ ǡ ǡ Ǥ ǡ ǡ Ǥ"
    assert text_quality(good) > 0.8
    assert text_quality(garbled) < 0.2
    assert page_needs_ocr(garbled)
    assert page_needs_ocr("Figure 3")
    assert not page_needs_ocr(good * 2)


def test_extraction_cache_and_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world " * 20)
    pages = extract_document(f, cache_dir=tmp_path / "cache")
    assert pages[0].method == "text_layer"
    assert list((tmp_path / "cache").glob("*.json"))
    assert extract_document(f, cache_dir=tmp_path / "cache")[0].text == pages[0].text


def test_chunking_ignores_table_of_contents_for_sections():
    pages = [
        PageText(1, "ARRANGEMENT OF SECTIONS\n1. Short title.\n17. Registration compulsory.\n" + "x " * 300, "text_layer", 1.0),
        PageText(2, "BE it enacted by Parliament\n1. (1) This Act may be called the Test Act.\n" + "y " * 300, "text_layer", 1.0),
    ]
    chunks = chunk_pages(pages, section_style="legislation", chunk_size=300, overlap=50)
    page1 = [c for c in chunks if c.page_number == 1 and c.page_end == 1]
    assert page1 and all(c.section_number == "not_identified" for c in page1)
    assert all("17" not in c.sections_in_chunk for c in chunks)
    assert any(c.page_number == 2 and c.section_number == "1" for c in chunks)


def test_chunking_overlap_and_page_ranges():
    pages = [PageText(1, "A " * 500, "text_layer", 1.0), PageText(2, "B " * 500, "text_layer", 1.0)]
    chunks = chunk_pages(pages, chunk_size=400, overlap=100)
    assert len(chunks) > 3
    assert any(c.page_number == 1 and c.page_end == 2 for c in chunks)  # chunk spanning pages
    for a, b in zip(chunks, chunks[1:]):
        assert a.text[-40:].strip()[-10:] in b.text  # consecutive chunks overlap
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, overlap=100)
