"""Optional OCR routing boundary; no OCR engine is required for text PDFs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Protocol

from .extractor import ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)


class OCRPageProvider(Protocol):
    """Adapter contract for a project-specific OCR implementation."""

    def extract_page(self, document: ExtractedDocument, page_number: int) -> str: ...


@dataclass(frozen=True)
class OCRRoutingResult:
    document: ExtractedDocument
    ocr_page_numbers: tuple[int, ...]
    warnings: tuple[str, ...]


class OCRRouter:
    """Route only sparse PDF pages to an explicitly supplied OCR provider."""

    def route(self, document: ExtractedDocument, provider: OCRPageProvider | None = None) -> OCRRoutingResult:
        requested = tuple(document.sparse_page_numbers) if document.source_format == "pdf" else ()
        if not requested:
            return OCRRoutingResult(document, (), ())
        if provider is None:
            warning = f"OCR recommended for page(s) {list(requested)} but no OCR provider is configured."
            logger.warning("%s Source: %s", warning, document.source_file)
            return OCRRoutingResult(document, (), (warning,))

        replacements: dict[int, str] = {}
        warnings: list[str] = []
        for page_number in requested:
            try:
                text = provider.extract_page(document, page_number)
            except Exception as exc:  # Provider failures must remain visible to callers.
                warnings.append(f"OCR failed for page {page_number}: {exc}")
                continue
            if text.strip():
                # Keep even sparse native extraction: OCR augments it rather than
                # silently replacing potentially meaningful source characters.
                original = next(page.text for page in document.pages if page.page_number == page_number)
                replacements[page_number] = "\n".join(item for item in (original, text) if item.strip())
            else:
                warnings.append(f"OCR returned no text for page {page_number}.")

        merged = ExtractedDocument(
            source_file=document.source_file,
            source_format=document.source_format,
            page_count=document.page_count,
            pages=[
                replace(page, text=replacements.get(page.page_number, page.text))
                for page in document.pages
            ],
            sparse_page_numbers=[number for number in requested if number not in replacements],
            likely_scanned=document.likely_scanned and not bool(replacements),
            asset_directory=document.asset_directory,
        )
        logger.info("OCR applied to %d page(s) for '%s'.", len(replacements), document.source_file)
        return OCRRoutingResult(merged, tuple(sorted(replacements)), tuple(warnings))
