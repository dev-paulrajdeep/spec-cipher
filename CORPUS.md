# Corpus layer

The corpus layer turns a source standard into deterministic, retrieval-ready
data. It deliberately stops before embeddings, vector search, ranking,
applicability decisions, serving, or graph storage.

## Flow

```text
PDF/TXT -> page-preserving ingestion -> optional sparse-page OCR
        -> conservative cleaning -> metadata + sections + relationships
        -> section-aware chunks -> canonical JSON and chunks JSONL
```

`corpus.ingestion` owns source reading and preserves raw page boundaries.
`corpus.processing` retains raw and cleaned page text separately, removes only
high-confidence extraction artifacts, and detects common numbered clauses,
annexes, and appendices. `metadata` and `relationships` only emit values that
are explicitly evidenced in the source. `chunking` never crosses a detected
section boundary.

## Canonical record

`StandardDocument` is the single downstream representation. Its JSON contains
`schema_version`, `metadata`, `pages`, `sections`, `relationships`, `chunks`,
and `validation`. The internal section fragments preserve page provenance for
chunking while the public section record exposes its page range.

Every retrieval chunk has `chunk_id`, `standard_number`, `source_file`,
`page_start`, `page_end`, `section_number`, `section_title`, and `text`.
Unknown metadata stays `null`; it is never filled with a guess. Validation
reports errors, warnings, and uncertain extractions without deleting content.

## OCR

PyMuPDF is always the normal PDF path. Pages with fewer than 20 extracted text
characters are routed only when a caller supplies an `OCRPageProvider`. This
supports mixed PDFs. If no provider is configured, the original page is kept
and an `uncertain` validation issue records the OCR requirement.

An OCR adapter implements:

```python
def extract_page(self, document: ExtractedDocument, page_number: int) -> str: ...
```

The adapter can wrap Tesseract, a cloud OCR service, or a project-specific
engine without coupling those dependencies to normal text extraction.

## Usage

```python
from corpus import CorpusBuilder

builder = CorpusBuilder()
document = builder.build_and_write(
    "standards/IS-2645.pdf",
    "output/IS-2645.json",
    "output/IS-2645.chunks.jsonl",
)
```

Configure chunking with `DocumentChunker(ChunkingConfig(max_chars=2000,
overlap_chars=200))` and inject it into `CorpusBuilder` when a different
retrieval budget is needed.

## Tests

Run `python -m pytest`. The tests generate PDFs in memory with PyMuPDF and do
not require a copyrighted standard. To inspect a real standard, use the usage
example above and review both JSON outputs and the validation findings.
