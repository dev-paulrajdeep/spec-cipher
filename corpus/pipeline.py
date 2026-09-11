"""Orchestrate the deterministic corpus preparation flow."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from corpus.chunking import ChunkingConfig, DocumentChunker
from corpus.ingestion import DocumentExtractor, OCRPageProvider, OCRRouter
from corpus.metadata import MetadataExtractor
from corpus.models import CorpusPage, ElementProvenance, StandardDocument, ValidationIssue
from corpus.processing import SectionDetector, TextCleaner
from corpus.relationships import RelationshipExtractor
from corpus.serialization import write_chunks_jsonl, write_document_json
from corpus.validation import CorpusValidator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"


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
        raw_asset_directory: str | Path | None = None,
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
        self.raw_asset_directory = Path(raw_asset_directory) if raw_asset_directory is not None else None

    def build(self, source_path: str | Path) -> StandardDocument:
        """Build from a source file, including optional per-page OCR routing."""
        source = Path(source_path)
        asset_directory = self.raw_asset_directory or source.parent / f"{source.stem}.assets"
        extracted = self.extractor.extract(source, asset_directory=asset_directory)
        return self._build_extracted(extracted)

    def build_bytes(
        self,
        data: bytes,
        source_name: str,
        source_format: str,
        *,
        raw_asset_directory: str | Path | None = None,
    ) -> StandardDocument:
        """Build from an upload without writing it to disk first."""
        asset_directory = Path(raw_asset_directory) if raw_asset_directory is not None else self.raw_asset_directory
        if asset_directory is None and source_format.lower().lstrip(".") == "pdf":
            asset_directory = Path.cwd() / "corpus_assets" / Path(source_name).stem
        extracted = self.extractor.extract_bytes(
            data, source_name, source_format, asset_directory=asset_directory
        )
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
                replace(page, ocr_applied=page.page_number in ocr_result.ocr_page_numbers)
                for page in pages
            )
        metadata = self.metadata_extractor.extract(pages, ocr_result.document.source_file)
        sections = self.section_detector.detect(pages)
        relationships = self.relationship_extractor.extract(pages, metadata.standard_number, sections)
        chunks = self.chunker.chunk(
            sections,
            metadata.standard_number,
            metadata.source_file,
            document_id=metadata.document_id,
            document_fingerprint=metadata.document_fingerprint,
        )
        chunks, pages = self._link_multimodal_provenance(
            pages, sections, chunks, metadata.source_file, metadata.standard_number
        )
        preliminary = StandardDocument(
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
            pages=pages,
            sections=sections,
            relationships=relationships,
            chunks=chunks,
            raw_asset_directory=ocr_result.document.asset_directory,
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
            raw_asset_directory=preliminary.raw_asset_directory,
        )
        logger.info(
            "Processed '%s': pages=%d ocr_pages=%d sections=%d relationships=%d chunks=%d issues=%d",
            metadata.source_file, len(pages), len(ocr_result.ocr_page_numbers), len(sections),
            len(relationships), len(chunks), len(issues),
        )
        return document

    @staticmethod
    def _link_multimodal_provenance(pages, sections, chunks, source_file, standard_id):
        """Attach conservative page-to-section/chunk links without guessing exact clause geometry."""
        chunk_ids_by_page = {
            page.page_number: tuple(
                chunk.chunk_id for chunk in chunks if chunk.page_start <= page.page_number <= chunk.page_end
            )
            for page in pages
        }
        section_numbers_by_page = {
            page.page_number: tuple(
                section.section_number for section in sections
                if section.section_number is not None
                and any(fragment.page_number == page.page_number for fragment in section.fragments)
            )
            for page in pages
        }
        linked_pages = []
        for page in pages:
            provenance = ElementProvenance(
                source_file,
                standard_id,
                page.page_number,
                section_numbers_by_page[page.page_number],
                chunk_ids_by_page[page.page_number],
            )

            def link(element):
                return replace(element, standard_id=standard_id, provenance=provenance)

            linked_pages.append(
                replace(
                    page,
                    id=page.id or f"page_{page.page_number:04d}",
                    standard_id=standard_id,
                    provenance=provenance,
                    text_blocks=tuple(link(item) for item in page.text_blocks),
                    tables=tuple(link(item) for item in page.tables),
                    figures=tuple(link(item) for item in page.figures),
                )
            )
        linked_chunks = tuple(
            replace(
                chunk,
                element_ids=tuple(
                    element.id
                    for page in linked_pages
                    if chunk.page_start <= page.page_number <= chunk.page_end
                    for element in (*page.text_blocks, *page.tables, *page.figures)
                ),
            )
            for chunk in chunks
        )
        return linked_chunks, tuple(linked_pages)
