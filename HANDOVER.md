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
- real SP 7:2026 source-text smoke queries through build, persist, reload, and
  search.

The bundled real corpus has 13,077 SP 7:2026 chunks but no evidenced standard
number for that Special Publication, so `standard_number` is correctly `null`
for those results. The smoke index used 32 actual source chunks selected for
three domain queries; it is a qualitative integration check, not a full-corpus
relevance benchmark.

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

The following is an actual rank-1 result from the reloaded 32-chunk SP 7:2026
source-text smoke index:

```text
query: "structural concrete design requirements"

{
  "standard_number": null,
  "title": null,
  "section_number": "5.2.1",
  "section_title": "Concrete",
  "text": "5.2.1 Concrete\nThe concrete used for construction shall be in accordance with Part C ‘Structural Design/ Section 5 Structural Concrete’ of this NBCS.",
  "score": 0.666084,
  "rank": 1,
  "source": {
    "chunk_id": "chunk_26f243acbab738462082",
    "source_file": "SP 7_2026.pdf",
    "page_start": 412,
    "page_end": 412,
    "element_ids": []
  }
}
```

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

## Configuration

The verified configuration is:

| Setting | Value |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 (verified from the loaded model) |
| Index | exact FAISS `IndexFlatIP` |
| Score | inner product of unit-normalized vectors = cosine similarity |
| Default query limit | 5 |
| Default build batch size | 16 |
| Corpus inputs | canonical corpus `.json` or chunk `.jsonl` |

The default batch size was reduced from 32 to 16 after the 32-text batch did
not complete on the repository's large SP 7 chunks in this runtime. Increase
it only after measuring memory headroom on the deployment host.

## Validation

- `.venv\\Scripts\\python.exe -m pytest tests --disable-warnings --maxfail=1
  -p no:cacheprovider`: **46 passed**.
- `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py
  --disable-warnings --maxfail=1 -p no:cacheprovider`: **16 passed**.
- `python -m compileall -q corpus retrieval intelligence backend tests`:
  passed (no compilation errors).
- `git diff --check`: passed (no whitespace errors).
- Retrieval unit tests cover the public service contract, top-k cap/order,
  ID-to-chunk mapping, persistence/reload equivalence, output provenance,
  invalid query/top-k values, missing/corrupt bundle artifacts, metadata
  mismatch, and wrong FAISS metric/index type.
- Actual-source qualitative smoke validation used a persisted/reloaded
  32-chunk index from `SP 7_2026.chunks.jsonl`; a fresh build was persisted,
  reloaded through `RetrievalService.from_persisted`, and original/loaded
  top-three search output compared equal for all three queries. Representative
  rank-1 results were:

  - query `fire and life safety requirements`: `chunk_eee3e84fbff6b6180d48`,
    score `0.675408`, page 600, clause `21.2 Fire and Life Safety`;
  - query `structural concrete design requirements`:
    `chunk_26f243acbab738462082`, score `0.666084`, page 412, clause
    `5.2.1 Concrete`;
  - query `water supply drainage and sanitation`:
    `chunk_2b11f832f1e7c1359a16`, score `0.576256`, pages 2506–2511,
    `Annex C`.

These smoke results verify execution and provenance, not Recall@K or overall
quality: the full 13,077-chunk production index and a corpus-wide gold set
remain to be built/evaluated on a suitably provisioned host.

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

- A full FAISS build/query evaluation across all 13,077 available SP 7:2026
  chunks was not completed in this finalization session. The 32-text default
  batch failed to complete on this runtime; batch size 16 was verified on real
  source chunks and is now the default. Build the complete deployment bundle
  and evaluate it against a corpus-wide gold set before making quality claims.
- Running bare `pytest` at repository root fails during collection because
  `corpus/database/test_pdf.py` reads a hard-coded, absent
  `national building construction standards 2026 v1.pdf` at import time. The
  maintained suite is `pytest tests`; this retrieval handoff does not modify
  that unrelated database script.
