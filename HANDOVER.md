# Retrieval Layer Handover

## Status

The retrieval layer is complete as a corpus-to-candidate boundary. It loads
corpus chunks, creates normalized embeddings, searches an exact FAISS index,
and returns provenance-preserving semantic candidates. It does not perform
applicability assessment, legal/compliance reasoning, reranking, relationship
traversal, or Flask response formatting.

Verified in this checkout:

- stable serving-facing `RetrievalService` interface;
- explicit FAISS-position -> chunk-ID -> `DocumentChunk` integrity;
- save/load of the complete index bundle with identical loaded search output;
- model, dimension, normalization, index-type, metadata, mapping, and chunk
  fingerprint validation;
- malformed inputs and persistence artifacts fail predictably;
- optional corpus element-level provenance is preserved and validated;
- current `IS 5120` corpus/index identity and provenance checks.

The current serving corpus is intentionally smaller: one historical/demo
canonical JSON artifact with one independently indexed `IS 5120` chunk. That
checklist is **not** an official BIS PDF extraction; the original official
source PDF is absent from this repository. Provenance for the demo path ends
at the checked-in canonical JSON and its recorded page/chunk fields.

Authentic official BIS publications must be supplied by the user under
`officialBIS/` and ingested only through `scripts/update_corpus.py`. SpecCipher
does not generate, synthesize, reconstruct, or download PDFs. Catalog/list PDFs
that merely mention many IS numbers are rejected and those mentions are not
promoted to independently indexed standards. If no authentic IS 5120 PDF is
present, the updater reports IS 5120 as unavailable rather than inventing one.

## Architecture

```text
corpus JSON / JSONL
        |
        v
CorpusChunkLoader -> validated DocumentChunk records
        |
        v
SentenceTransformerEncoder (unit-normalized vectors)
        |
        v
FAISS IndexFlatIP + explicit position-to-chunk-ID mapping
        |
        v
SemanticRetriever -> RetrievalResult[] -> RetrievalService.search(...)
```

`intelligence/` consumes the resulting `RetrievalResult` objects. `backend/`
should consume only `RetrievalService` and the JSON-ready `to_dict()` result
view.

## Integration Contract

Use this import and this interface at Flask application startup:

```python
from retrieval import RetrievalService

retriever = RetrievalService.from_persisted("path/to/retrieval-index")
results = retriever.search(query, top_k=5)
payload = [result.to_dict() for result in results]
```

`RetrievalService.from_persisted(directory)` loads a complete, compatible
index bundle. `search(query: str, top_k: int = 5)` returns
`tuple[RetrievalResult, ...]` and exposes no FAISS or embedding object.

Input rules:

- `query` must be a non-empty, non-whitespace `str`.
- `top_k` must be a positive `int`; booleans are invalid.
- Results contain at most `top_k` items. If fewer chunks are indexed, all
  available chunks are returned.

`RetrievalResult` fields are:

| Field | Type | Meaning |
| --- | --- | --- |
| `chunk` | `DocumentChunk` | The original corpus retrieval unit, including provenance. |
| `similarity_score` | `float` | Cosine similarity of normalized query and chunk embeddings. Not applicability or confidence. |
| `rank` | `int` | One-based position in descending similarity order. |

For Flask JSON, call `result.to_dict()`. It returns:

```text
{
  standard_number: str | null,  # corpus-provided; no separate standard_id exists
  title: null,                  # title is not represented by DocumentChunk
  section_number: str | null,
  section_title: str | null,
  text: str,
  score: float,                 # equal to similarity_score
  rank: int,
  source: {
    chunk_id: str,
    source_file: str,
    page_start: int,
    page_end: int,
    element_ids: list[str]
  }
}
```

All source and clause fields are retained from the corpus. `standard_number`,
`section_number`, and `section_title` can be `null` when not evidenced by the
source. `title` is always `null` at retrieval level because it is not part of
the existing `DocumentChunk` schema. Results are ordered by descending cosine
similarity; ties retain FAISS result order.

Error behavior:

- invalid `query` or `top_k`: `ValueError`;
- `search()` before loading: `RetrievalUnavailableError`;
- unavailable FAISS: `FaissUnavailableError`;
- absent index bundle artifacts: `FileNotFoundError`;
- corrupt/malformed metadata, chunks, mapping, index, or a model/dimension/
  normalization/index-type mismatch: `IndexCompatibilityError`;
- unavailable embedding runtime/model: `RuntimeError` with the underlying
  cause.

## Example Request/Response

The current serving bundle contains one independently indexed `IS 5120` chunk.
Use the live API or the generated `retrieval-index/chunks.json` for the exact
chunk ID, extracted text, and model-dependent score; this handover does not
copy a fabricated response or imply coverage of the related codes mentioned
inside the source text.

The exact score is meaningful only for this model and index configuration; do
not turn it into an applicability/confidence percentage in Flask.

## Persistence

`RetrievalPipeline.build_and_persist(corpus_artifact, index_directory)` creates
one required bundle:

- `index.faiss` — FAISS `IndexFlatIP` vectors;
- `chunk_ids.json` — explicit FAISS-position-to-chunk-ID mapping;
- `chunks.json` — `DocumentChunk` provenance and text;
- `metadata.json` — schema, model identity, dimension, normalization,
  corpus-schema version, corpus fingerprint, chunk count, index type, and
  creation time.

All four files must stay together. At app startup, load the directory with
`RetrievalService.from_persisted(index_directory)`. Never construct a FAISS
index in a request handler.

## Retrieval Index Bundle

The Flask integration path is the repository-root directory
`retrieval-index/`. It must contain these four files together:

```text
retrieval-index/
├── index.faiss
├── chunk_ids.json
├── chunks.json
└── metadata.json
```

The files are generated and intentionally gitignored. They are not committed
binary artifacts. The checked-in build entry point is
`scripts/build_retrieval_index.py`; it uses the existing
`RetrievalPipeline.build_and_persist()` path and does not create placeholder
files.

From the repository root, build the current serving bundle with:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py --corpus "corpus\data\IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json" --output retrieval-index --device cpu --allow-cpu-fallback
```

The corpus argument must point to the actual retrieval-ready `.json` or
`.jsonl` corpus produced by the corpus pipeline. Do not substitute a different
standard or model without rebuilding and revalidating the fingerprinted
bundle.

The application loads the bundle once at startup through the public interface:

```python
from retrieval import RetrievalService

retriever = RetrievalService.from_persisted(RETRIEVAL_INDEX_DIR)
results = retriever.search(user_input, top_k=5)
payload = [result.to_dict() for result in results]
```

Required prerequisites are the Python dependencies in `requirements.txt`, the
`BAAI/bge-m3` model, FAISS, and a complete corpus artifact. The generated
metadata records the backend, model revision, reported dimension (normally
1024 for BGE-M3), normalized `IndexFlatIP` configuration, and corpus
fingerprint. A mismatch fails during load instead of silently serving
incompatible results.

Before running Flask, make sure the retrieval index has been generated at
`retrieval-index/`. If it does not exist, run
`.\.venv\Scripts\python.exe scripts\build_retrieval_index.py --corpus "corpus\data\IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json" --output retrieval-index --device cpu --allow-cpu-fallback`.

## Configuration

The current serving configuration is:

| Setting | Value |
| --- | --- |
| Embedding model | `BAAI/bge-m3` |
| Embedding dimension | 1024 when reported by the loaded model |
| Index | exact FAISS `IndexFlatIP` |
| Score | inner product of unit-normalized vectors = cosine similarity |
| Default query limit | 5 |
| Default build batch size | 16 |
| Corpus inputs | canonical corpus `.json` or chunk `.jsonl` |

The production encoder is `BAAI/bge-m3`, normally used with CUDA when
available. Batch size defaults to 16 to keep VRAM use conservative. CPU use
is explicit (`embedding_device="cpu"`) or requires
`allow_cpu_fallback=True`; importing retrieval remains lazy and does not
allocate VRAM.

## Validation

- The maintained non-model suite was re-run in a compatible workspace-local
  Python 3.12 runtime: **105 passed**; the focused API/Qwen/frontend set after
  the current hardening changes: **32 passed**.
- The optional GPU/model smoke tests and real BGE-M3 demo tests require the
  project Python 3.11 environment plus cached model runtime and are reported
  separately when unavailable.
- `python -m compileall -q corpus retrieval intelligence backend tests` and
  `git diff --check` are required final checks.
- Retrieval unit tests cover the public service contract, top-k cap/order,
  ID-to-chunk mapping, persistence/reload equivalence, output provenance,
  invalid query/top-k values, missing/corrupt bundle artifacts, metadata
  mismatch, and wrong FAISS metric/index type.
- The current MVP has one independently indexed standard and one chunk. Unit
  tests use deterministic injected encoders for repeatable integrity checks;
  real BGE-M3 startup and live HTTP acceptance require cached model artifacts
  and a compatible local runtime.

These checks verify execution and provenance, not Recall@K or broad standards
quality. A larger independently sourced corpus and corpus-wide gold set are
required before any BIS-wide coverage claim.

## Daksh Integration Notes

In `backend/app.py`, create one `RetrievalService` when the Flask app starts,
using a configured persisted-index directory. For each validated request:

```python
results = retriever.search(user_input, top_k=5)
return jsonify({"status": "success", "recommendations": [r.to_dict() for r in results]})
```

Translate `ValueError` into HTTP 400, `FileNotFoundError`,
`IndexCompatibilityError`, and `RetrievalUnavailableError` into an explicit
service-unavailable HTTP 503/500 response as appropriate for deployment.

Daksh should call only `RetrievalService.from_persisted(...)` and
`retriever.search(...)`, then serialize with `result.to_dict()`. Do not import
or depend on `RetrievalPipeline`, `SemanticRetriever`, `FaissIndex`, encoder
classes, index files individually, or private attributes. Do not use the
SQLite `LIKE` search in `backend/app.py` as the semantic retrieval path.

## Known Issues

- The generated bundle is an operational handoff artifact, not a relevance
  benchmark. Evaluate it against a corpus-wide gold set before making quality
  claims.
- Running bare `pytest` at repository root fails during collection because
  `corpus/database/test_pdf.py` reads a hard-coded, absent
  `national building construction standards 2026 v1.pdf` at import time. The
  maintained suite is `pytest tests`; this retrieval handoff does not modify
  that unrelated database script.
