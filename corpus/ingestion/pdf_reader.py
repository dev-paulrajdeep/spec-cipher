"""
PDF reader for the SpecCipher corpus ingestion pipeline.

This module is responsible only for:
    - Opening PDF documents (from disk or in-memory bytes)
    - Reading page-level text
    - Preserving page boundaries
    - Returning basic document information, including signals needed
      to decide whether OCR is required downstream

Text cleaning, metadata extraction, section detection, and chunking
belong to downstream corpus modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)

# Below this average character count per page, a page is treated as
# text-sparse and likely to need OCR (e.g. a scanned image page).
OCR_CHAR_THRESHOLD = 20

# PyMuPDF's get_text() modes that return a plain string. Modes like
# "words", "blocks", "dict", and "rawdict" return lists/dicts instead
# and are not compatible with PDFPage.text: str.
_STRING_EXTRACTION_MODES = frozenset({"text", "html", "xml", "xhtml"})


@dataclass(frozen=True)
class PDFPage:
    """Raw text extracted from a single PDF page."""

    page_number: int
    text: str
    # Internal: plain-text extraction used to judge text density, kept
    # separate from `text` so a non-default extraction_mode (e.g. "html")
    # can never mask a blank/scanned page behind markup boilerplate.
    # None means "extraction_mode was already 'text'", so `text` itself
    # is the reference; use field(compare=False) so it doesn't affect
    # equality between otherwise-identical pages.
    _sparsity_text: str | None = field(default=None, repr=False, compare=False)

    @property
    def char_count(self) -> int:
        """
        Number of characters used to judge text density.

        Always measured from plain-text extraction regardless of
        `extraction_mode`, so markup added by "html"/"xml"/"xhtml"
        modes never masks a page that is actually blank or scanned.
        """
        reference = self.text if self._sparsity_text is None else self._sparsity_text
        return len(reference.strip())

    @property
    def is_text_sparse(self) -> bool:
        """
        Whether this page yielded little to no extractable text.

        A sparse page is a strong signal that the page is a scanned
        image (or otherwise non-text) and should be routed to OCR by
        the ingestion pipeline. This flags the condition only; it does
        not attempt OCR itself, which is out of scope for this module.
        """
        return self.char_count < OCR_CHAR_THRESHOLD


@dataclass(frozen=True)
class PDFDocument:
    """Page-preserving representation of an extracted PDF."""

    source_file: str
    page_count: int
    pages: list[PDFPage] = field(default_factory=list)

    @property
    def sparse_page_numbers(self) -> list[int]:
        """Page numbers (1-indexed) that likely need OCR."""
        return [p.page_number for p in self.pages if p.is_text_sparse]

    @property
    def likely_scanned(self) -> bool:
        """
        Whether the document as a whole looks scanned/image-based.

        True when the majority of pages are text-sparse. Downstream
        ingestion can use this to route the whole document to an OCR
        step rather than checking pages one by one.
        """
        if not self.pages:
            return False
        return len(self.sparse_page_numbers) > self.page_count / 2


class PDFReadError(RuntimeError):
    """Raised when a PDF cannot be opened or read."""


class EncryptedPDFError(PDFReadError):
    """Raised when a PDF is password-protected and cannot be opened."""


class PDFReader:
    """
    Read PDF documents and extract page-level raw text.

    The reader deliberately does not modify or clean the extracted text.
    Downstream processing is responsible for normalization and cleaning.
    """

    def __init__(self, extraction_mode: str = "text") -> None:
        """
        Args:
            extraction_mode: PyMuPDF text extraction mode passed to
                `page.get_text()`. Must be one of "text" (default),
                "html", "xml", or "xhtml" — the modes that return a
                plain string. Layout-structured modes ("words",
                "blocks", "dict", "rawdict") return lists/dicts and
                are out of scope for this module's str-based PDFPage.

        Raises:
            ValueError: If `extraction_mode` is not a string-returning mode.
        """
        if extraction_mode not in _STRING_EXTRACTION_MODES:
            raise ValueError(
                f"Unsupported extraction_mode '{extraction_mode}'. "
                f"Must be one of {sorted(_STRING_EXTRACTION_MODES)}."
            )
        self.extraction_mode = extraction_mode

    def read(self, file_path: str | Path) -> PDFDocument:
        """
        Read a PDF from disk and return its page-preserving raw text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFDocument containing raw text for every page.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the supplied file is not a PDF.
            EncryptedPDFError: If the PDF is password-protected.
            PDFReadError: If PyMuPDF cannot open or read the document.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        document = self._open(path, stream=None)
        return self._extract(document, source_name=path.name)

    def read_bytes(self, data: bytes, source_name: str = "<upload>") -> PDFDocument:
        """
        Read a PDF from an in-memory byte string (e.g. a Flask file
        upload) without requiring it to be written to disk first.

        Args:
            data: Raw PDF file bytes.
            source_name: Label to store as `PDFDocument.source_file`,
                e.g. the original uploaded filename.

        Returns:
            PDFDocument containing raw text for every page.

        Raises:
            ValueError: If `data` is empty.
            EncryptedPDFError: If the PDF is password-protected.
            PDFReadError: If PyMuPDF cannot open or read the document.
        """
        if not data:
            raise ValueError("No PDF data provided.")

        document = self._open(path=None, stream=data)
        return self._extract(document, source_name=source_name)

    def _open(
        self, path: Path | None, stream: bytes | None
    ) -> pymupdf.Document:
        """Open a PDF via PyMuPDF, raising specific errors for common failure modes."""
        try:
            document = (
                pymupdf.open(stream=stream, filetype="pdf")
                if stream is not None
                else pymupdf.open(path)
            )
        except Exception as exc:
            label = path if path is not None else "<in-memory PDF>"
            raise PDFReadError(f"Failed to open PDF '{label}': {exc}") from exc

        if not document.is_pdf:
            document.close()
            label = path if path is not None else "<in-memory PDF>"
            raise ValueError(f"File is not a valid PDF: {label}")

        if document.needs_pass:
            document.close()
            label = path if path is not None else "<in-memory PDF>"
            raise EncryptedPDFError(
                f"PDF '{label}' is password-protected and cannot be read."
            )

        return document

    def _extract(self, document: pymupdf.Document, source_name: str) -> PDFDocument:
        """Extract page-level text from an already-opened, validated document."""
        try:
            pages = [
                PDFPage(
                    page_number=page_number,
                    text=page.get_text(self.extraction_mode),
                    _sparsity_text=(
                        page.get_text("text")
                        if self.extraction_mode != "text"
                        else None
                    ),
                )
                for page_number, page in enumerate(document, start=1)
            ]
            page_count = len(document)
        except Exception as exc:
            raise PDFReadError(
                f"Failed to extract text from PDF '{source_name}': {exc}"
            ) from exc
        finally:
            document.close()

        pdf_document = PDFDocument(
            source_file=source_name,
            page_count=page_count,
            pages=pages,
        )

        if pdf_document.likely_scanned:
            logger.warning(
                "PDF '%s' looks scanned (%d/%d pages text-sparse); "
                "consider routing to OCR.",
                source_name,
                len(pdf_document.sparse_page_numbers),
                page_count,
            )

        return pdf_document