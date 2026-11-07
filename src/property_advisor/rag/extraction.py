"""Document text extraction with OCR only where needed.

* Text PDFs are extracted page-by-page with ``pypdf``.
* Each page is scored for extraction quality. A page is OCR'd only when its
  text layer is (a) garbled — fonts without a Unicode map, as in the URDPFI
  Guidelines — or (b) nearly empty (scanned page / figure).
* OCR uses RapidOCR (ONNX runtime, no system binaries) on pages rendered with
  pypdfium2. Both are optional: if they are not installed, affected pages are
  recorded as ``needs_ocr`` and skipped rather than indexing garbage.
* Results are cached as JSON under ``data_processed/extracted/`` keyed by the
  file's SHA-256, so OCR runs once per document version. Source files are only
  ever opened read-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("property_advisor.rag")

MIN_WORDS_PER_PAGE = 15
MIN_CLEAN_TOKEN_RATIO = 0.5
_CLEAN_TOKEN_RE = re.compile(r"^[\(\[\"'“‘]?[A-Za-z0-9][A-Za-z0-9,.;:'’\-\)\]\"”/%—–]*$")
EXTRACTOR_VERSION = "1"


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str
    method: str  # "text_layer" | "ocr" | "needs_ocr" | "empty"
    quality: float


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_quality(text: str) -> float:
    """Share of whitespace tokens that look like real words/numbers.

    Garbled extractions (missing ToUnicode maps) produce tokens such as
    'ǡʹͲͳͶ' and score close to 0; normal English legal text scores > 0.8.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    clean = sum(1 for t in tokens if _CLEAN_TOKEN_RE.match(t))
    return clean / len(tokens)


def page_needs_ocr(text: str) -> bool:
    words = len(text.split())
    if words < MIN_WORDS_PER_PAGE:
        return True
    return text_quality(text) < MIN_CLEAN_TOKEN_RATIO


def ocr_available() -> bool:
    try:
        import pypdfium2  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class _OcrEngine:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from rapidocr_onnxruntime import RapidOCR

            cls._instance = RapidOCR()
        return cls._instance


def _ocr_pdf_page(path: str, page_index: int, scale: float = 2.0) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        image = pdf[page_index].render(scale=scale).to_numpy()
    finally:
        pdf.close()
    result, _ = _OcrEngine.get()(image)
    return "\n".join(line[1] for line in (result or []))


def _extract_pdf(path: str, enable_ocr: bool) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    can_ocr = enable_ocr and ocr_available()
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("text extraction failed for %s p%d: %s", path, i + 1, exc)
            text = ""
        quality = text_quality(text)
        if not page_needs_ocr(text):
            pages.append(PageText(i + 1, text, "text_layer", round(quality, 3)))
            continue
        if can_ocr:
            try:
                ocr_text = _ocr_pdf_page(path, i)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s p%d: %s", path, i + 1, exc)
                ocr_text = ""
            if len(ocr_text.split()) >= MIN_WORDS_PER_PAGE:
                pages.append(PageText(i + 1, ocr_text, "ocr", round(text_quality(ocr_text), 3)))
            else:
                pages.append(PageText(i + 1, "", "empty", 0.0))
        else:
            # Do not index garbage; keep any short-but-clean text (e.g. a heading).
            keep = text if quality >= MIN_CLEAN_TOKEN_RATIO else ""
            pages.append(PageText(i + 1, keep, "needs_ocr", round(quality, 3)))
    return pages


def _extract_txt(path: str) -> list[PageText]:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [PageText(1, text, "text_layer", round(text_quality(text), 3))]


def extract_document(
    path: str | Path,
    cache_dir: str | Path | None = None,
    sha256: str | None = None,
    enable_ocr: bool = True,
) -> list[PageText]:
    """Extract page texts, using/refreshing the on-disk cache when given.

    A cached extraction made without OCR is reused only if OCR is still
    unavailable/disabled; otherwise the document is re-extracted so that
    previously skipped pages get OCR'd.
    """
    path = str(path)
    sha256 = sha256 or sha256_file(path)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{sha256[:16]}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            pages = [PageText(**p) for p in cached["pages"]]
            has_pending_ocr = any(p.method == "needs_ocr" for p in pages)
            if cached.get("extractor_version") == EXTRACTOR_VERSION and not (
                has_pending_ocr and enable_ocr and ocr_available()
            ):
                return pages

    if path.lower().endswith(".pdf"):
        pages = _extract_pdf(path, enable_ocr)
    else:
        pages = _extract_txt(path)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "source_file": os.path.basename(path),
                    "sha256": sha256,
                    "extractor_version": EXTRACTOR_VERSION,
                    "pages": [asdict(p) for p in pages],
                },
                f,
            )
    return pages
