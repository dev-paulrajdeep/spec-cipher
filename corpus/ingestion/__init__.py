"""
SpecCipher corpus ingestion package.

Format-agnostic ingestion for the standards corpus: reads raw source
documents (currently PDF and plain text) and hands downstream corpus
modules (processing, metadata, chunking) a single consistent shape,
regardless of how the source document arrived.

PDF extraction preserves deterministic layout, tables, embedded images,
vector-drawing regions, and optional raw visual assets alongside raw text.

Text cleaning, metadata extraction, section detection, and chunking
are out of scope here — see the sibling `processing`, `metadata`, and
`chunking` packages.
"""

from .extractor import (
    DocumentExtractor,
    ExtractedDocument,
    ExtractedPage,
    ExtractionError,
    UnsupportedFormatError,
)
from .pdf_reader import (
    EncryptedPDFError,
    PDFDocument,
    PDFPage,
    PDFReadError,
    PDFReader,
)
from .ocr import OCRPageProvider, OCRRouter, OCRRoutingResult

__all__ = [
    "DocumentExtractor",
    "ExtractedDocument",
    "ExtractedPage",
    "ExtractionError",
    "UnsupportedFormatError",
    "PDFReader",
    "PDFDocument",
    "PDFPage",
    "PDFReadError",
    "EncryptedPDFError",
    "OCRPageProvider",
    "OCRRouter",
    "OCRRoutingResult",
]
