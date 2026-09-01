"""Orchestrate the deterministic corpus preparation flow."""

from __future__ import annotations

import logging
from pathlib import Path

from corpus.chunking import ChunkingConfig, DocumentChunker
from corpus.ingestion import DocumentExtractor, OCRPageProvider, OCRRouter
from corpus.metadata import MetadataExtractor
from corpus.models import CorpusPage, StandardDocument, ValidationIssue
from corpus.processing import SectionDetector, TextCleaner
from corpus.relationships import RelationshipExtractor
from corpus.serialization import write_chunks_jsonl, write_document_json
from corpus.validation import CorpusValidator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class CorpusBuilder:
    """Build a canonical standard record; no retrieval or reasoning is performed here."""

    def __init__(
        self,
        *,
        extractor: DocumentExtractor | None = None,
        cleaner: TextCleaner | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        section_detector: SectionDetector | None = None,
        relationship_extractor: RelationshipExtractor | None = None,
        chunker: DocumentChunker | None = None,
        validator: CorpusValidator | None = None,
        ocr_router: OCRRouter | None = None,
        ocr_provider: OCRPageProvider | None = None,
    ) -> None:
        self.extractor = extractor or DocumentExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.section_detector = section_detector or SectionDetector()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()
        self.chunker = chunker or DocumentChunker(ChunkingConfig())
        self.validator = validator or CorpusValidator()
        self.ocr_router = ocr_router or OCRRouter()
        self.ocr_provider = ocr_provider

    def build(self, source_path: str | Path) -> StandardDocument:
        """Build from a source file, including optional per-page OCR routing."""
        extracted = self.extractor.extract(source_path)
        return self._build_extracted(extracted)

    def build_bytes(self, data: bytes, source_name: str, source_format: str) -> StandardDocument:
        """Build from an upload without writing it to disk first."""
        extracted = self.extractor.extract_bytes(data, source_name, source_format)
        return self._build_extracted(extracted)

    def build_and_write(
        self, source_path: str | Path, canonical_json_path: str | Path, chunks_jsonl_path: str | Path
    ) -> StandardDocument:
        """Build then write canonical JSON and the retrieval-oriented JSONL projection."""
        document = self.build(source_path)
        write_document_json(document, canonical_json_path)
        write_chunks_jsonl(document.chunks, chunks_jsonl_path)
        return document

    def _build_extracted(self, extracted) -> StandardDocument:
        ocr_result = self.ocr_router.route(extracted, self.ocr_provider)
        pages = self.cleaner.clean_document(ocr_result.document)
        if ocr_result.ocr_page_numbers:
            pages = tuple(
                CorpusPage(
                    page_number=page.page_number,
                    raw_text=page.raw_text,
                    cleaned_text=page.cleaned_text,
                    ocr_applied=page.page_number in ocr_result.ocr_page_numbers,
                )
                for page in pages
            )
        metadata = self.metadata_extractor.extract(pages, ocr_result.document.source_file)
        sections = self.section_detector.detect(pages)
        relationships = self.relationship_extractor.extract(pages, metadata.standard_number, sections)
        chunks = self.chunker.chunk(sections, metadata.standard_number, metadata.source_file)
        preliminary = StandardDocument(
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
            pages=pages,
            sections=sections,
            relationships=relationships,
            chunks=chunks,
        )
        issues = (*self.validator.validate(preliminary), *(
            # OCR absence/failures are uncertain extraction, never silently hidden.
            ValidationIssue("uncertain", "ocr_unavailable", warning) for warning in ocr_result.warnings
        ))
        document = StandardDocument(
            schema_version=preliminary.schema_version,
            metadata=preliminary.metadata,
            pages=preliminary.pages,
            sections=preliminary.sections,
            relationships=preliminary.relationships,
            chunks=preliminary.chunks,
            validation=tuple(issues),
        )
        logger.info(
            "Processed '%s': pages=%d ocr_pages=%d sections=%d relationships=%d chunks=%d issues=%d",
            metadata.source_file, len(pages), len(ocr_result.ocr_page_numbers), len(sections),
            len(relationships), len(chunks), len(issues),
        )
        return document
