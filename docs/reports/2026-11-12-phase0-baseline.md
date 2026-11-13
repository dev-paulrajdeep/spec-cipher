# Phase 0 Repository Baseline — 2026-11-12

## Repository state

- Branch: `Architectural_Overhaul_Expansion`, tracking `origin/main`.
- HEAD: `1d381468c7d8114c8975d440692963820c148e88` (`feat: Implement CatalogRepository and confidence calculation logic; enhance ApplicationContainer with release management`).
- The working tree was already dirty before implementation: changes covered the API, catalog, settings, release manifest, and E2E tests, including deletion of `tests/e2e/fixtures/adapters.py`. No application changes from that work were overwritten.
- Reconnaissance indexing updated `.codebase-memory/artifact.json` and `.codebase-memory/graph.db.zst`; these are also present in the current diff.

## Environment and project layout

- Python: 3.11.9.
- Installed tools: pytest 9.2.0, Ruff 0.17.4, Import Linter 2.16.
- Installed numeric/runtime packages include NumPy 2.4.6, FAISS CPU 1.16.0, and Torch 2.15.0.
- `uv` is not installed; no lockfile is present.
- The project already has `pyproject.toml`, Ruff and pytest configuration, Import Linter contracts, a pre-commit 5 MiB guard, and GitHub Actions CI.
- `officialBIS/` contains no source PDFs in this checkout.

## Corpus and tracked artifacts

- `corpus/data/manifest.json` records 12 documents and 13,151 chunks, of which 1 document and 1,177 chunks are eligible.
- The manifest identifies 11 unresolved documents, including the remaining SP 7:2026 volumes. Without authentic source PDFs, no operator override or full-publication eligibility claim can be validated.
- Fifteen generated files under `corpus/data/` are tracked and occupy 180.67 MiB in the working tree. `retrieval-index/` is ignored and has no tracked files.
- The current `.gitignore` re-includes generated corpus files despite its comment that they should be ignored.

## Baseline checks

| Check | Result |
| --- | --- |
| Default `pytest tests --disable-warnings --maxfail=1 -p no:cacheprovider` | **Failed during E2E collection**: `test_tier3_combinations.py` imports the deleted `tests.e2e.fixtures.adapters`; FAISS also emitted a Windows native DLL initialization exception from E2E `conftest.py`. |
| Non-E2E suite without exclusions | **Failed** at `test_faiss_vector_count_matches_eligible_chunks`: FAISS `_swigfaiss` DLL initialization failed. |
| Bounded non-E2E suite | **180 passed, 1 skipped, 6 deselected in 34.53s**; another 6 `test_demo_queries.py` cases were excluded because the configured BGE model is absent from the offline Hugging Face cache. Command: `pytest tests --ignore=tests/e2e --ignore=tests/test_demo_queries.py --deselect=tests/test_e2e_smoke.py::test_smoke_defaults_and_recommendation_projection --deselect=tests/test_corpus_update.py::test_faiss_vector_count_matches_eligible_chunks --deselect=tests/test_corpus_update.py::test_empty_eligible_index_is_allowed --deselect=tests/test_corpus_update.py::test_preflight_rejects_persisted_zero_vector_index --deselect=tests/test_ingestion.py::test_pdf_multimodal_extraction_preserves_geometry_assets_and_provenance --deselect=tests/test_retrieval_metadata_validation.py::test_metadata_validation_reports_actual_production_coverage --disable-warnings --maxfail=1 -p no:cacheprovider`. |
| `python -m compileall -q corpus retrieval intelligence backend src tests` | Passed. |
| `lint-imports --config pyproject.toml` | Passed: 4 contracts kept. |
| `ruff check src` | Passed. |
| `ruff check src tests` | Failed: 59 findings, concentrated in the in-progress E2E changes plus one existing test warning. |

The bounded run is a diagnostic signal, not a passing full-suite baseline. The E2E suite, real FAISS operations, model-backed queries, and one PyMuPDF table extraction assertion remain unverified or blocked in this environment.

## Immediate architectural gaps confirmed on disk

- `src/speccipher/release/` has manifest verification but no production publisher or rollback implementation.
- `verify_release()` does not require hashes for every payload, does not validate catalog/chunk/vector identity comprehensively, and accepts `CURRENT` targets that escape the release root.
- `ApplicationContainer` suppresses errors while mounting a configured release and still constructs legacy corpus/retrieval components instead of using the release as the sole serving contract.
- The new `/v1/analyze` route duplicates lexical search as dense search, runs RRF over the same list twice, and emits fixed applicability/reranker values. The legacy catalog branch similarly treats BM25 as applicability.

