"""
Document extractor — completes the ingestion stage of the SpecCipher
corpus pipeline.

This module is responsible only for:
    - Detecting the format of a raw source document
    - Dispatching to the correct format-specific reader (e.g. PDFReader)
    - Producing a single, format-agnostic ExtractedDocument with optional layout/assets
    - Passing through OCR-need signals so downstream processing knows
      which pages/documents require an OCR pass

Text cleaning, metadata extraction (standard number, title, etc.),
section detection, and chunking are downstream corpus modules
(processing/, metadata/, chunking/) and are deliberately out of scope
here. This module's only job is: "however the source arrived, hand
downstream a single consistent shape."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from corpus.models import BoundingBox, Figure, Table, TextBlock

from .pdf_reader import EncryptedPDFError, PDFDocument, PDFReadError, PDFReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedPage:
    """A single page of raw, unmodified text from any source format."""

    page_number: int
    text: str
    width: float | None = None
    height: float | None = None
    bbox: BoundingBox | None = None
    asset_path: str | None = None
    text_blocks: tuple[TextBlock, ...] = ()
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    """
    Format-agnostic, page-preserving raw extraction result.

    Every reader in this package (present and future) normalizes its
    output into this shape, so downstream corpus modules (cleaning,
    metadata, chunking) never need to know or care whether a document
    originally came from a PDF, a plain text file, or something else.
    """

    source_file: str
    source_format: str  # e.g. "pdf", "txt"
    page_count: int
    pages: list[ExtractedPage] = field(default_factory=list)

    # OCR-routing signals, passed through from format-specific readers
    # where applicable (currently only PDFReader produces these).
    sparse_page_numbers: list[int] = field(default_factory=list)
    likely_scanned: bool = False
    asset_directory: str | None = None

    @property
    def full_text(self) -> str:
        """
        Concatenated raw text across all pages.

        Pages are joined with a form-feed character (\\f) so downstream
        modules can still recover page boundaries from the flat string
        if they need them, without carrying the page list around.
        """
        return "\f".join(page.text for page in self.pages)


class ExtractionError(RuntimeError):
    """Raised when a document cannot be extracted, regardless of format."""


class UnsupportedFormatError(ExtractionError):
    """Raised when the source file's format has no registered extractor."""


class DocumentExtractor:
    """
    Format-agnostic entry point for corpus ingestion.

    Detects a source file's format from its extension and dispatches to
    the appropriate format-specific reader, always returning a single
    unified ExtractedDocument. New formats (e.g. DOCX, HTML) can be
    supported later by adding an entry to `_extractors` without
    changing any calling code.
    """

    def __init__(self, pdf_reader: PDFReader | None = None) -> None:
        """
        Args:
            pdf_reader: Optional PDFReader instance to use for PDF
                sources. Defaults to a plain `PDFReader()`; inject a
                custom one (e.g. non-default extraction_mode) if needed.
        """
        self._pdf_reader = pdf_reader or PDFReader()
        self._extractors: dict[str, Callable[[Path], ExtractedDocument]] = {
            ".pdf": self._extract_pdf,
            ".txt": self._extract_txt,
        }

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this extractor currently knows how to handle."""
        return sorted(self._extractors)

    def extract(
        self, file_path: str | Path, *, asset_directory: str | Path | None = None
    ) -> ExtractedDocument:
        """
        Extract a document from disk, regardless of format.

        Args:
            file_path: Path to the source file.

        Returns:
            ExtractedDocument with page-preserving raw text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a file.
            UnsupportedFormatError: If no extractor is registered for
                this file's extension.
            ExtractionError: If the file exists and is a supported
                format, but reading/extraction fails (corrupt file,
                encrypted PDF, unreadable encoding, etc.).
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        extension = path.suffix.lower()
        extractor_fn = self._extractors.get(extension)
        if extractor_fn is None:
            raise UnsupportedFormatError(
                f"No extractor registered for '{extension}' files. "
                f"Supported formats: {self.supported_extensions}"
            )

        try:
            document = (
                self._extract_pdf(path, asset_directory)
                if extension == ".pdf"
                else extractor_fn(path)
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to extract '{path}': {exc}") from exc

        logger.info(
            "Extracted '%s' (%s, %d page(s))",
            document.source_file,
            document.source_format,
            document.page_count,
        )
        return document

    def extract_bytes(
        self,
        data: bytes,
        source_name: str,
        source_format: str,
        *,
        asset_directory: str | Path | None = None,
    ) -> ExtractedDocument:
        """
        Extract a document from in-memory bytes (e.g. a Flask upload),
        where the format can't be inferred from a filesystem path.

        Args:
            data: Raw file bytes.
            source_name: Label to store as `source_file`, e.g. the
                original uploaded filename.
            source_format: Format hint, with or without a leading dot
                (e.g. "pdf" or ".pdf").

        Returns:
            ExtractedDocument with page-preserving raw text.

        Raises:
            ValueError: If `data` is empty.
            UnsupportedFormatError: If `source_format` isn't registered.
            ExtractionError: If reading/extraction fails.
        """
        if not data:
            raise ValueError("No document data provided.")

        extension = "." + source_format.lower().lstrip(".")
        if extension not in self._extractors:
            raise UnsupportedFormatError(
                f"No extractor registered for '{extension}' format. "
                f"Supported formats: {self.supported_extensions}"
            )

        try:
            if extension == ".pdf":
                pdf_document = self._pdf_reader.read_bytes(
                    data, source_name=source_name, asset_directory=asset_directory
                )
                document = self._from_pdf_document(pdf_document)
            else:  # ".txt"
                text = data.decode("utf-8", errors="replace")
                document = self._build_txt_document(source_name, text)
        except ExtractionError:
            raise
        except (PDFReadError, EncryptedPDFError) as exc:
            raise ExtractionError(str(exc)) from exc
        except Exception as exc:
            raise ExtractionError(f"Failed to extract '{source_name}': {exc}") from exc

        logger.info(
            "Extracted '%s' from bytes (%s, %d page(s))",
            document.source_file,
            document.source_format,
            document.page_count,
        )
        return document

    # -- format-specific extraction, one method per registered format ---

    def _extract_pdf(self, path: Path, asset_directory: str | Path | None = None) -> ExtractedDocument:
        try:
            pdf_document = self._pdf_reader.read(path, asset_directory=asset_directory)
        except (PDFReadError, EncryptedPDFError) as exc:
            raise ExtractionError(str(exc)) from exc
        return self._from_pdf_document(pdf_document)

    def _from_pdf_document(self, pdf_document: PDFDocument) -> ExtractedDocument:
        """Normalize a PDFReader result into the format-agnostic shape."""
        return ExtractedDocument(
            source_file=pdf_document.source_file,
            source_format="pdf",
            page_count=pdf_document.page_count,
            pages=[
                ExtractedPage(
                    page_number=p.page_number,
                    text=p.text,
                    width=p.width,
                    height=p.height,
                    bbox=p.bbox,
                    asset_path=p.asset_path,
                    text_blocks=p.text_blocks,
                    tables=p.tables,
                    figures=p.figures,
                )
                for p in pdf_document.pages
            ],
            sparse_page_numbers=pdf_document.sparse_page_numbers,
            likely_scanned=pdf_document.likely_scanned,
            asset_directory=pdf_document.asset_directory,
        )

    def _extract_txt(self, path: Path) -> ExtractedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ExtractionError(f"Failed to read text file '{path}': {exc}") from exc
        return self._build_txt_document(path.name, text)

    def _build_txt_document(self, source_name: str, text: str) -> ExtractedDocument:
        # Plain text has no native page concept, so the whole file is
        # treated as a single page. Downstream chunking (not this
        # module) is responsible for splitting it further.
        return ExtractedDocument(
            source_file=source_name,
            source_format="txt",
            page_count=1,
            pages=[ExtractedPage(page_number=1, text=text)],
            sparse_page_numbers=[],
            likely_scanned=False,
        )
