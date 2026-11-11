# ADR-0004: Application Composition Root and Decoupled Flask Factory

- **Status**: Accepted
- **Date**: 2026-11-11
- **Deciders**: SpecCipher Architecture Team
- **Consulted**: Milestone 1 Specialists

## Context

In the legacy design (`backend/app.py`), application bootstrapping was executed at module import time:
```python
# Legacy backend/app.py:
app = Flask(__name__)
# ...
RETRIEVER, INTELLIGENCE, QWEN_REASONER = _load_runtime()
```
When `_load_runtime()` executed during `import backend.app`:
1. It invoked preflight validation over hard-coded file paths on disk.
2. It initialized `RetrievalService.from_persisted()`, immediately loading the FAISS vector index and embedding models into GPU/CPU memory.
3. It loaded `CorpusKnowledgeBase.from_canonical_json()`, synchronously parsing approximately 170 MB of JSON documents from disk.

This created critical operational defects:
- **Untestable Delivery Layer**: Unit testing HTTP routes or error handling required loading the full ML runtime, making tests slow (>60s) and brittle.
- **Hardware Coupling**: Simply importing the Flask module failed on machines lacking model caches, GPU memory, or access to disk assets.
- **Inflexible Configuration**: Runtime settings (ports, devices, paths) were frozen at import time rather than configured per environment or test runner.

## Decision

We decouple application configuration and dependency wiring into two distinct constructs:
1. **Application Composition Root (`speccipher.services.container.ApplicationContainer`)**:
   - Acts as the single authoritative composition root for the entire system.
   - Accepts typed settings (`speccipher.config.Settings`) and an optional release path (`release_dir`).
   - Implements lazy property instantiation for long-lived services (`catalog`, `retrieval_service`, `decision_engine`, `explanation_service`).
   - Manages graceful shutdown and resource cleanup via `.close()`.

2. **Decoupled Flask Application Factory (`speccipher.api.app.create_app`)**:
   - Replaces the global module-level `app = Flask(__name__)` singleton with a pure factory function:
     ```python
     def create_app(container: ApplicationContainer | None = None) -> Flask:
         ...
     ```
   - If no container is passed, a lightweight default container is resolved.
   - Binds versioned API routes (`/v1/*`), the legacy compatibility adapter (`/get_result`), error handlers (RFC 7807), and observability middleware without touching filesystem indexes or ML weights.

3. **Zero Heavyweight Initialization on Import**:
   - **MANDATORY INVARIANT**: Importing any module under `speccipher.api` or `speccipher.services` must execute in **less than 1.0 second**, perform **zero disk I/O**, and initialize **no heavy ML models**.
   - Model weights and vector indexes are loaded strictly on first query demand or via explicit warmup commands.

## Consequences

### Positive
- **Instantaneous Test Fixtures**: API contract tests can instantiate fresh Flask app instances backed by mock or fake containers in <10ms.
- **Explicit Lifecycle Management**: Server startup, preflight verification, and shutdown are deterministic and observable.
- **Testability**: Different service implementations (e.g. fake embedding encoder vs. real BGE-M3) can be injected cleanly without patching global module variables.

### Negative & Trade-offs
- **Slight Indirection**: Code cannot access global service instances directly; components must access services through the container or request context (`g.container`).

### Compliance
Enforced by:
- `tests/test_architecture_contracts.py::TestArchitectureContracts::test_zero_heavy_init_on_module_import`
- API contract test suite using lightweight fake containers.
