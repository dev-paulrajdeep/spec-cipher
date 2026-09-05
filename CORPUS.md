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

For PDFs, ingestion is multimodal but deterministic: PyMuPDF emits text blocks
with bounding boxes, detected tables with headers/rows plus a retrieval-text
projection, embedded raster images, vector-drawing regions, page dimensions,
and a 144-DPI full-page PNG. The PNG is a fallback for every page, so visual
content not recognized as an image or figure remains reviewable without a VLM.

`corpus.ingestion` owns source reading and preserves raw page boundaries.
`corpus.processing` retains raw and cleaned page text separately, removes only
high-confidence extraction artifacts, and detects common numbered clauses,
annexes, and appendices. `metadata` and `relationships` only emit values that
are explicitly evidenced in the source. `chunking` never crosses a detected
section boundary.

## Canonical record

`StandardDocument` is the single downstream representation. Its JSON contains
`schema_version`, `metadata`, `pages`, `sections`, `relationships`, `chunks`,
`validation`, and (when applicable) `raw_asset_directory`. Schema `1.1` adds
spatial data without changing the retrieval JSONL contract.

Each canonical PDF page records its `id`, `standard_id`, `page_number`, page
`bbox`, width, height, page-render `asset_path`, and provenance. It owns
`text_blocks`, `tables`, and `figures`. Every element has an id, page bounding
box, type, content or asset path, and provenance. Table `content` is a
labelled, retrieval-friendly text rendering; headers and rows remain separate
structured fields. Element provenance carries the source, standard, page, and
all conservatively applicable detected clause and chunk ids. Exact geometric
clause assignment is intentionally not guessed when a page has multiple
clauses.

Every retrieval chunk has `chunk_id`, `standard_number`, `source_file`,
`page_start`, `page_end`, `section_number`, `section_title`, and `text`.
Unknown metadata stays `null`; it is never filled with a guess. Validation
reports errors, warnings, and uncertain extractions without deleting content.
Canonical chunks additionally retain `element_ids`; this is deliberately not
added to JSONL so existing retrieval loaders and indexes stay compatible.

## Raw visual assets

`CorpusBuilder.build("standard.pdf")` writes raw visual assets to a sibling
`standard.assets/` directory by default. Pass `raw_asset_directory=` to the
builder (or to `build_bytes`) to choose a durable asset store. Byte uploads
without an explicit directory use `corpus_assets/<source-stem>/` under the
current working directory. These files are raw evidence; normalized text and
the retrieval JSONL remain separate.

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
retrieval budget is needed. Retrieval can continue to embed the unchanged
chunk JSONL. Explainability can dereference a chunk's canonical `element_ids`
and then use element provenance, coordinates, tables, or page image to show
the underlying source evidence.

## Tests

Run `python -m pytest`. The tests generate PDFs in memory with PyMuPDF and do
not require a copyrighted standard. To inspect a real standard, use the usage
example above and review both JSON outputs and the validation findings.
