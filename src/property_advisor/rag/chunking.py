"""Section-aware, page-tracking text chunking.

The whole document is treated as one stream so chunks overlap across page
boundaries, while each chunk still records the page it starts on
(``page_number``) and ends on (``page_end``). Section numbers are tracked from
headings in the text:

* ``legislation`` style — Indian statute headings such as
  ``17. Documents of which registration is compulsory.—`` or
  ``1. (1) This Act may be called ...``. Pages that are part of an
  "ARRANGEMENT OF SECTIONS/CLAUSES" table of contents are not used for section
  tracking (they would otherwise assign every section number to page 1).
* ``numbered_headings`` style — guideline headings such as ``3.2.1 Zoning``.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field

from property_advisor.rag.extraction import PageText

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

_LEGIS_HEADING = re.compile(r"^\s*\[?(\d{1,3}[A-Z]{0,3})\.\s*(?:\(1\)|[A-Z\[“\"‘'])")
_NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+[A-Z]")
_TOC_MARKER = re.compile(r"ARRANGEMENT OF (SECTIONS|CLAUSES)|^\s*CONTENTS\s*$", re.I | re.M)
# Statute body starts at the enacting formula / long title, or (for model
# laws without one, e.g. the Model Tenancy Act) at "This Act may be called".
_BODY_MARKER = re.compile(r"BE it enacted|An Act to|WHEREAS it is expedient|This Act may be called", re.I)
CHUNKER_VERSION = "2"


@dataclass
class Chunk:
    text: str
    page_number: int
    page_end: int
    section_number: str
    sections_in_chunk: list[str] = field(default_factory=list)
    extraction_method: str = "text_layer"
    content_sha: str = ""


def _toc_pages(pages: list[PageText]) -> set[int]:
    """Pages from the first TOC marker until the statute body begins."""
    toc: set[int] = set()
    in_toc = False
    for p in pages:
        if not in_toc and _TOC_MARKER.search(p.text) and not _BODY_MARKER.search(p.text):
            in_toc = True
        if in_toc:
            if _BODY_MARKER.search(p.text):
                break
            toc.add(p.page_number)
    return toc


def _heading_regex(style: str):
    if style == "legislation":
        return _LEGIS_HEADING
    if style == "numbered_headings":
        return _NUMBERED_HEADING
    return None


def chunk_pages(
    pages: list[PageText],
    section_style: str = "none",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    heading_re = _heading_regex(section_style)
    skip_pages = _toc_pages(pages) if section_style == "legislation" else set()

    # Build a single stream with page offsets and heading offsets.
    parts: list[str] = []
    page_starts: list[int] = []
    page_numbers: list[int] = []
    page_methods: list[str] = []
    headings: list[tuple[int, str]] = []
    offset = 0
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue
        page_starts.append(offset)
        page_numbers.append(p.page_number)
        page_methods.append(p.method)
        if heading_re and p.page_number not in skip_pages:
            line_off = 0
            for line in text.split("\n"):
                m = heading_re.match(line)
                if m:
                    headings.append((offset + line_off, m.group(1)))
                line_off += len(line) + 1
        parts.append(text)
        offset += len(text) + 2  # "\n\n" separator
    stream = "\n\n".join(parts)
    if not stream.strip():
        return []

    heading_offsets = [h[0] for h in headings]

    def page_at(pos: int) -> int:
        idx = bisect.bisect_right(page_starts, pos) - 1
        return max(idx, 0)

    chunks: list[Chunk] = []
    start = 0
    n = len(stream)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer to break at a paragraph/sentence/line boundary.
            window = stream[start + chunk_size // 2 : end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut != -1:
                    end = start + chunk_size // 2 + cut + len(sep)
                    break
        text = stream[start:end].strip()
        if text:
            first_page_idx = page_at(start)
            last_page_idx = page_at(max(end - 1, start))
            h_before = bisect.bisect_right(heading_offsets, start) - 1
            inside = [h[1] for h in headings if start <= h[0] < end]
            if h_before >= 0:
                section = headings[h_before][1]
            elif inside:
                section = inside[0]
            else:
                section = "not_identified"
            methods = set(page_methods[first_page_idx : last_page_idx + 1])
            chunks.append(
                Chunk(
                    text=text,
                    page_number=page_numbers[first_page_idx],
                    page_end=page_numbers[last_page_idx],
                    section_number=section,
                    sections_in_chunk=list(dict.fromkeys(inside)),
                    extraction_method="ocr" if "ocr" in methods else "text_layer",
                    content_sha=hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16],
                )
            )
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
