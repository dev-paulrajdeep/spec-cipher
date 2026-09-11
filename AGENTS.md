# SpecCipherML Agent Guide

## Working Rules

- Treat repository contents, documentation, tests, command output, and runtime behavior as evidence. Distinguish facts from inferences and state uncertainty when evidence is missing.
- Inspect the relevant entry point, execution path, configuration, and nearby tests before changing behavior. Prefer the smallest change that preserves existing public interfaces.
- Reproduce failures before fixing them, identify the first point where behavior diverges, then rerun the original check and relevant regression tests.
- Preserve user-owned working-tree changes. Do not reset, clean, checkout over, or commit changes unless explicitly requested.
- Do not invent corpus metadata, standard numbers, titles, clauses, statuses, relationships, scores, or confidence values. Preserve unknown values as `null`.
- Avoid adding dependencies or abstractions unless the existing code and measured behavior justify them. Keep comments limited to non-obvious rationale.

## Project Shape

- `backend/app.py` is the Flask entry point. It serves the static frontend, loads the persisted retrieval bundle at startup, runs intelligence recommendations, and exposes `POST /get_result`.
- `retrieval/service.py` is the serving boundary. Application code should use `RetrievalService.from_persisted()` and `.search()` rather than accessing FAISS or embedding objects directly.
- `intelligence/pipeline/engine.py` owns constraints, reranking, applicability, relationships, evidence, and recommendation construction.
- `corpus/pipeline.py` and `corpus/models.py` define ingestion, validation, canonical documents, and retrieval-ready chunks.
- `corpus/update/` owns the general-purpose officialBIS PDF → JSONL → FAISS update workflow (`scripts/update_corpus.py`).
- `backend/qwen_reasoning.py` adapts to Ollama. Explanations are optional and must fail open without changing finalized recommendations.
- `frontend/` is a static HTML/CSS/JavaScript client with no frontend build system.
- Public package exports live in `corpus/__init__.py`, `retrieval/__init__.py`, and `intelligence/__init__.py`.

## Commands

Run commands from the repository root with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest tests --disable-warnings --maxfail=1 -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
.\.venv\Scripts\python.exe -m compileall -q corpus retrieval intelligence backend tests
.\.venv\Scripts\python.exe scripts\update_corpus.py
.\.venv\Scripts\python.exe scripts\e2e_smoke.py
.\.venv\Scripts\python.exe scripts\validate_full_chain.py
.\.venv\Scripts\python.exe backend\app.py
```

- Prefer a focused test first, then the full `tests/` suite when the change crosses module boundaries.
- Use `pytest tests`, not bare `pytest`; repository-root collection includes an unrelated database test that expects a missing hard-coded PDF.
- E2E checks require Ollama at `http://localhost:11434`, the configured Qwen model, and Playwright Chromium. Relevant environment variables include `SPEC_CIPHER_OLLAMA_MODEL`, `SPEC_CIPHER_OLLAMA_TIMEOUT_SECONDS`, `SPEC_CIPHER_BIS_SOURCE_DIR`, `SPEC_CIPHER_HF_OFFLINE`, `QWEN_OLLAMA_HOST`, `QWEN_MODEL_NAME`, `QWEN_TIMEOUT_SECONDS`, `SPEC_CIPHER_HOST`, and `SPEC_CIPHER_PORT`.

## Retrieval Constraints

- `retrieval-index/` is generated and must contain a matching `index.faiss`, `chunk_ids.json`, `chunks.json`, and `metadata.json` bundle. Do not hand-edit or commit placeholder artifacts.
- Rebuild the bundle with:

  ```powershell
  .\.venv\Scripts\python.exe scripts\update_corpus.py
  ```

  Or manually from an existing artifact:

  ```powershell
  .\.venv\Scripts\python.exe scripts\build_retrieval_index.py --corpus "corpus\data\corpus.chunks.jsonl" --output retrieval-index
  ```

- Real retrieval requires the BGE-M3 model, Sentence Transformers, FAISS, and model-compatible device resources. Real reranking uses BGE reranker v2 M3; sequential lifecycle and conservative batches protect limited GPU memory.
- Retrieval similarity is cosine similarity, not applicability or confidence. Do not convert it into a percentage without an explicit, tested domain rule.
- Keep provenance intact. Unknown source metadata remains `null`; do not infer it from filenames or neighboring records.

## Documentation

Link to the authoritative document instead of copying its contract:

- [README.md](README.md): product overview and workflow.
- [CORPUS.md](CORPUS.md): corpus schema, ingestion, OCR, and chunking.
- [RETRIEVAL.md](RETRIEVAL.md): retrieval semantics, persistence, and model requirements.
- [INTELLIGENCE.md](INTELLIGENCE.md): recommendation pipeline and score semantics.
- [API.md](API.md): frontend/backend request and response contract.
- [HANDOVER.md](HANDOVER.md): retrieval integration contract and verified operational commands.
- [STRESS_TEST_REPORT.md](STRESS_TEST_REPORT.md): adversarial findings and historical validation; verify claims against current code.

Generated validation assets belong under `.tmp/` and are workspace state, not source prerequisites.