# Corpus layer

The corpus layer turns **arbitrary technical/standards PDFs** into deterministic,
retrieval-ready data. It deliberately stops before embeddings, vector search,
ranking, applicability decisions, serving, or graph storage.

This pipeline is **general-purpose**. Indian Standards (IS) are one current use
case, not an architectural special case. The same flow accepts ISO, IEC, ASTM,
company specifications, and engineering manuals without Python changes per
document.

## Source vs generated vs runtime

| Layer | Location | Role |
| --- | --- | --- |
| SOURCE DATA | `officialBIS/` (or `SPEC_CIPHER_BIS_SOURCE_DIR`) | User-supplied PDFs only. Never generated, never committed. |
| GENERATED ARTIFACTS | `corpus/data/documents/`, `corpus.chunks.jsonl`, `manifest.json`, `retrieval-index/` | Produced only by `scripts/update_corpus.py`. |
| LEGACY DEMO DATA | `corpus/data/IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json` | Historical checklist for demo serving when no authentic source-derived corpus exists. Not an official PDF extraction. |
| RUNTIME STATE | Flask process | Loads the persisted FAISS bundle. **Does not parse PDFs.** |

## Adding a new PDF

1. Place the PDF in `officialBIS/` (nested folders are fine).
2. Run:

```powershell
.\.venv\Scripts\python.exe scripts\update_corpus.py
```

3. Verify the update summary (discovered / processed / duplicates / unresolved /
   failed / chunks / vectors).
4. Start the application.

Optional check mode:

```powershell
.\.venv\Scripts\python.exe scripts\update_corpus.py --check
```

Useful flags: `--source`, `--output`, `--device`, `--allow-cpu-fallback`, `--force`.

## End-to-end ingestion flow

```text
LOCAL PDF DIRECTORY
        ↓
PDF discovery (recursive, PDF-only, path-safe, deterministic)
        ↓
document validation (size/pages/encryption)
        ↓
PDF text extraction (PyMuPDF)
        ↓
conservative metadata + document identity
        ↓
structure-aware chunking (section → paragraph → controlled split)
        ↓
canonical document JSON + JSONL chunks
        ↓
BGE-M3 embeddings (normalized) → FAISS IndexFlatIP
        ↓
manifest / provenance / corpus fingerprint
```

## Identity resolution (conservative)

`corpus.update.identity.DocumentIdentity` resolves **document-level** identity:

- filename identity
- PDF metadata title
- single clear header identifier

Supported number families include IS/SP, ISO, IEC, EN, BS, ASTM, and
conservative structured ids. Unknown fields stay `null`.

Rules:

- Mentions of other document numbers inside a PDF are **references**, not new
  corpus documents.
- Catalog/list PDFs that only mention many numbers are rejected.
- Unresolved but ingestible PDFs (for example an engineering manual with no
  number) may be stored with `document_number=null` and a fingerprint-based
  `document_id`. They cannot become recommendations.
- Different content sharing the same document number is a conflict, never a
  silent merge.
- Identical bytes are treated as duplicates.

## Chunk JSONL schema

One line = one retrieval chunk. Field ordering is sorted for determinism.

```json
{
  "chunk_id": "...",
  "chunking_version": "1.1",
  "document_fingerprint": "...",
  "document_id": "...",
  "page_end": 1,
  "page_start": 1,
  "section_number": "... or null",
  "section_title": "... or null",
  "source_file": "...",
  "source_path": "relative/path.pdf",
  "standard_number": "... or null",
  "text": "...",
  "title": "... or null"
}
```

`standard_number` is the generic document/standard number field (IS, ISO, etc.,
or `null`). Absolute developer machine paths are never persisted.

## Chunking strategy

`DocumentChunker` is structure-aware:

1. Detect numbered sections / annexes / appendices.
2. Chunk within a section.
3. Split oversized sections at paragraph, then sentence boundaries.
4. Apply small controlled overlap only when packing units.

Same inputs + same chunking version ⇒ same JSONL.

## OCR

PyMuPDF is the normal PDF path. Pages with fewer than 20 extracted text
characters are routed only when a caller supplies an `OCRPageProvider`.
If no provider is configured, the page is kept and an `uncertain` validation
issue records that OCR is required. Image-only PDFs are reported as failures
with remediation; the pipeline does not pretend OCR ran.

## Canonical record

`StandardDocument` is the downstream representation: `schema_version`,
`metadata`, `pages`, `sections`, `relationships`, `chunks`, `validation`, and
optional `raw_asset_directory`.

## Manifest

`corpus/data/manifest.json` records schema/ingestion/chunking versions, source
directory identity, document and chunk counts, fingerprints, unresolved and
failed documents, and embedding/index identity after rebuild. Startup can detect
stale indexes when fingerprints diverge.

## Historical demo checklist

The checked-in checklist JSON is **legacy/demo data**. It is never overwritten
by `update_corpus.py`. When authentic source-derived `corpus.chunks.jsonl` is
absent, runtime may fall back to that checklist for demonstration only.

## Tests

Corpus update tests use **synthetic fixture PDFs only** (generic identities such
as TECH-1001 / STD-A / DOC-2026-001). Production code does not hard-code those
values. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests --disable-warnings --maxfail=1 -p no:cacheprovider
```
