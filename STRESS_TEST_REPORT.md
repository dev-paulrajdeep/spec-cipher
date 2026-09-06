# SpecCipher Retrieval Stress Test

## Executive Verdict

**PASS WITH ISSUES**

The corpus-to-retrieval boundary passed the adversarial checks after two
reproducible defects were fixed. The retrieval layer is safe for Daksh to
integrate through its documented public service, subject to the remaining
operational risks below. The existing Flask endpoint is not yet wired to this
service and still performs SQLite `LIKE` search; that is integration work for
Daksh, not a retrieval implementation failure.

## Scope Tested

Inspected and exercised the corpus loader, `DocumentChunk` model, retrieval
pipeline, embedding adapter, normalized FAISS index, persistence bundle,
public `RetrievalService`, tests, `backend/app.py`, `HANDOVER.md`,
`RETRIEVAL.md`, dependencies, and the available SP 7 smoke artifacts.

The tests used isolated temporary directories for generated indexes and never
modified the canonical corpus or checked-in artifacts.

## Test Matrix

| Category | Test performed | Result | Failures / severity |
| --- | --- | --- | --- |
| Input fuzzing | Empty, whitespace, `None`, wrong types, bytes, punctuation, Unicode, long repeated text, casing, duplicate calls, and invalid `top_k` values | Passed; deterministic `ValueError` for invalid public inputs | One bug found and fixed: `top_k=None` leaked through the public service, **MEDIUM** |
| Retrieval correctness | Top-k cap/order, finite scores, repeatability, unrelated and domain-style queries | Passed on deterministic fixtures and real 15-query batch | No correctness failure; real subset relevance remains qualitative |
| Provenance attack | Special-character/Unicode IDs, missing optional metadata, long metadata, `element_ids`, mapping and chunk identity checks | Passed | One loader omission fixed before this report; **MEDIUM** |
| Persistence attack | Build/save/load, repeated load, missing files, malformed metadata, changed chunks, wrong mapping, wrong metric, invalid vectors | Passed after fixes | Invalid persisted vectors were accepted initially; **HIGH** |
| Model/index integrity | Model identity, dimension, normalization, index type, vector count, fingerprint | Passed | No remaining reproduced mismatch |
| Corpus edge cases | Missing fields, empty text, invalid pages/types, malformed element provenance, duplicate IDs, invalid JSONL | Passed | No remaining reproduced parser failure |
| Real corpus stress | 15 varied queries against persisted SP 7 bundle, top-5 invariants and provenance | 15/15 success; 0 failures | No correctness failure; 32-chunk subset is not a full-corpus benchmark |
| Repeated/concurrent use | 10 repeated identical searches and 32 concurrent reads on one service | Passed; all results consistent | No reproduced state leakage or nondeterminism |
| Public API contract | Import, `from_persisted`, `search`, `RetrievalResult.to_dict`, exceptions | Matches `HANDOVER.md` after fix | No remaining contract mismatch |
| Flask assumptions | Inspected endpoint and response construction | Retrieval service is JSON-serializable, but endpoint currently uses SQLite `LIKE` | Integration not complete; Daksh must wire service |

## Adversarial Cases

- A persisted `IndexFlatIP` containing `[2.0, 0.0, 0.0]` originally loaded and
  returned score `2.0`, violating cosine-score semantics. `from_native` now
  reconstructs and validates all stored vectors for finite values and unit
  norm. Regression coverage is in `tests/test_retrieval.py`.
- A persisted NaN vector originally loaded and silently produced no hit.
  The same persisted-vector validation now rejects it as incompatible.
- `RetrievalService.search("query", top_k=None)` originally used the internal
  retriever default despite the public contract requiring a positive integer.
  The service now validates query and `top_k` before delegation.
- Changed persisted chunk text is rejected by the corpus fingerprint check.
- Mapping IDs that do not match persisted chunks, duplicate IDs, wrong FAISS
  metric, missing files, malformed metadata, and dimension/normalization
  mismatches all fail loudly with `IndexCompatibilityError` or
  `FileNotFoundError`.

## Real Corpus Results

The available real artifact is `SP 7_2026.chunks.jsonl`; the smoke index
contains 32 actual chunks selected from the available SP 7 source corpus.
The embedding model loaded successfully as
`sentence-transformers/all-MiniLM-L6-v2` with dimension 384.

The batch contained 15 queries covering natural language, technical terms,
standard/section references, partial phrases, vague language, and an
unrelated query. All 15 returned five results, all scores were finite and in
non-increasing order, and all returned chunks had non-empty text and source
files. Representative rank-1 results:

| Query | Chunk ID | Score | Section / pages |
| --- | --- | ---: | --- |
| `fire and life safety requirements` | `chunk_eee3e84fbff6b6180d48` | 0.675408 | 21.2 Fire and Life Safety / 600 |
| `structural concrete design requirements` | `chunk_26f243acbab738462082` | 0.666084 | 5.2.1 Concrete / 412 |
| `water supply drainage and sanitation` | `chunk_2b11f832f1e7c1359a16` | 0.576256 | Annex C / 2506-2511 |
| `section 5 structural concrete` | `chunk_26f243acbab738462082` | 0.773963 | 5.2.1 Concrete / 412 |
| `random procurement software unrelated` | `chunk_679948ef4e4ad6106eb1` | 0.152139 | Annex C / 2506-2511 |

These are retrieval outputs, not fabricated relevance labels. The unrelated
query demonstrates that the semantic index returns nearest neighbors; it does
not provide a confidence threshold or abstention decision.

Measured query latency after service/model initialization for the 15-query
sequential batch: minimum 7.13 ms, median 8.72 ms, maximum 121.38 ms. No
resource benchmark or full 13,077-chunk index benchmark was claimed.

## Bugs Found

### 1. Invalid persisted FAISS vectors

- **Severity:** HIGH
- **Reproduction:** Replace `index.faiss` with a valid `IndexFlatIP` containing
  a non-unit vector or NaN vector, then call `RetrievalPipeline.load()`.
- **Root cause:** Load checked FAISS type, dimension, count, and mapping but
  not reconstructed vector values.
- **Fix:** `FaissIndex.from_native` now validates reconstructed vectors for
  finite values and unit norm before accepting the index.
- **Regression test:** `test_load_rejects_invalid_persisted_vectors` in
  `tests/test_retrieval.py`.

### 2. Public `top_k=None` contract bypass

- **Severity:** MEDIUM
- **Reproduction:** Call `RetrievalService.search("query", top_k=None)`.
- **Root cause:** The internal `SemanticRetriever` intentionally accepts
  `None` for its optional default, and the public facade delegated without
  enforcing its stricter contract.
- **Fix:** `RetrievalService.search` validates query type/content and requires
  a positive integer `top_k` before checking loaded state or delegating.
- **Regression test:** Public-service assertion in `tests/test_retrieval.py`.

## API/Handover Verification

`HANDOVER.md` matches the implementation for the public import,
`RetrievalService.from_persisted(directory)`, `search(query, top_k)`, result
tuple type, `to_dict()` JSON shape, score semantics, ordering, persistence
files, and exception classes. Its test counts were updated to the final
46-test maintained suite.

The JSON-ready result contains primitive values only: standard/section fields,
text, float score, rank, and a source object containing chunk ID, source file,
page range, and element IDs. `standard_number` and title remain null where the
existing corpus schema does not evidence them.

## Performance

For the real 15-query sequential batch after initialization: 7.13 ms minimum,
8.72 ms median, and 121.38 ms maximum per query. Thirty-two concurrent reads
returned consistent results. These observations are smoke measurements, not a
capacity or production SLA benchmark. The full 13,077-chunk corpus was not
indexed in this run.

## Remaining Risks

- The available full SP 7 corpus has 13,077 chunks, but the verified smoke
  index contains 32 selected chunks. A full production build and corpus-wide
  relevance evaluation still require a suitably provisioned host and gold
  data.
- Bare repository-root pytest fails during collection because
  `corpus/database/test_pdf.py` reads an absent hard-coded PDF at import time.
  The maintained `tests` suite is clean; this is unrelated to retrieval.
- `backend/app.py` still uses SQLite `LIKE` and does not initialize or call
  `RetrievalService`. Daksh must perform that application wiring and map
  retrieval exceptions to HTTP responses.
- Persistence writes four files sequentially and is intended as an offline
  build artifact. Deployments should publish a completed bundle atomically and
  keep all four files from the same build together.

## Final Handoff Status

The retrieval layer is **safe for Daksh to integrate** through
`from retrieval import RetrievalService` and `retriever.search(...)`, with
startup loading, documented exception handling, and `result.to_dict()` for
Flask JSON. Daksh should not depend on FAISS, encoder classes, pipeline
internals, private attributes, or the current SQLite search path.

Final automated verification:

- `.venv\\Scripts\\python.exe -m pytest tests --disable-warnings --maxfail=1 -p no:cacheprovider`: **46 passed**.
- `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval_stress.py --disable-warnings --maxfail=1 -p no:cacheprovider`: **10 passed**.
- `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py --disable-warnings --maxfail=1 -p no:cacheprovider`: **16 passed**.
- `.venv\\Scripts\\python.exe -m compileall -q corpus retrieval intelligence backend tests`: passed.
- `git diff --check`: passed.