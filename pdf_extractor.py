"""
pdf_extractor.py
----------------
Step 1 of the pipeline: validate an uploaded PDF and extract its text.

* Works entirely in memory (bytes) - the resume is NEVER written to disk.
* Primary backend: PyMuPDF (fitz). Fallback backend: pdfplumber.
* Raises PDFValidationError with a human-readable message for any problem,
  so the Streamlit UI can show a friendly error instead of a stack trace.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

MAX_FILE_MB = 5          # reject very large files
MAX_PAGES = 10           # resumes are short; protects the server
MIN_TEXT_CHARS = 150     # fewer characters => probably a scanned image


class PDFValidationError(Exception):
    """Raised when the uploaded file is not a usable text-based PDF."""


@dataclass
class ExtractionResult:
    text: str
    page_count: int
    word_count: int
    backend: str


# ---------------------------------------------------------------- validation
def validate_pdf_bytes(data: bytes) -> None:
    """Cheap checks done before any parsing library touches the file."""
    if not data:
        raise PDFValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise PDFValidationError(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller PDF.")
    # Every real PDF contains the '%PDF-' signature near the start of the file.
    if b"%PDF-" not in data[:1024]:
        raise PDFValidationError("This does not look like a valid PDF file (missing PDF header).")


# ---------------------------------------------------------------- backends
def _with_pymupdf(data: bytes) -> tuple[str, int]:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise PDFValidationError("The PDF is password-protected. Please upload an unlocked copy.")
        if doc.page_count > MAX_PAGES:
            raise PDFValidationError(f"The PDF has {doc.page_count} pages; the limit is {MAX_PAGES}.")
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages), doc.page_count
    finally:
        doc.close()


def _with_pdfplumber(data: bytes) -> tuple[str, int]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        if len(pdf.pages) > MAX_PAGES:
            raise PDFValidationError(f"The PDF has {len(pdf.pages)} pages; the limit is {MAX_PAGES}.")
        pages = [(p.extract_text() or "") for p in pdf.pages]
        return "\n".join(pages), len(pdf.pages)


# ---------------------------------------------------------------- cleaning
def clean_extracted_text(text: str) -> str:
    """Tidy raw PDF text but KEEP line breaks (needed later for section detection)."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[\u2022\u25CF\u25AA\u25E6\u2023\u2043\u25A0]", "- ", text)   # bullet glyphs -> '- '
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                                # re-join hyphenated words
    text = re.sub(r"[ \t\u00A0]+", " ", text)                                   # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                                      # collapse blank lines
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ---------------------------------------------------------------- public API
def extract_text_from_pdf(data: bytes) -> ExtractionResult:
    """Validate + extract. Tries PyMuPDF first, then pdfplumber."""
    validate_pdf_bytes(data)

    raw, pages, backend, last_error = None, 0, "", None
    for name, fn in (("PyMuPDF", _with_pymupdf), ("pdfplumber", _with_pdfplumber)):
        try:
            raw, pages = fn(data)
            backend = name
            break
        except PDFValidationError:
            raise                      # our own, already user-friendly
        except ImportError as exc:     # library not installed -> try next backend
            last_error = exc
        except Exception as exc:       # corrupted file, encrypted, etc.
            last_error = exc
    if raw is None:
        raise PDFValidationError(f"Could not read this PDF ({type(last_error).__name__}). It may be corrupted or encrypted.")

    text = clean_extracted_text(raw)
    if len(text) < MIN_TEXT_CHARS:
        raise PDFValidationError(
            "Very little text could be extracted. The PDF is probably a scanned image; "
            "please upload a text-based PDF (exported from Word/Google Docs/LaTeX)."
        )
    return ExtractionResult(text=text, page_count=pages, word_count=len(text.split()), backend=backend)
