# ADR-0001: Modular Monolith under `src/speccipher` with 7-Layer Architecture

- **Status**: Accepted
- **Date**: 2026-11-11
- **Deciders**: SpecCipher Architecture Team
- **Consulted**: Milestone 1 Specialists

## Context

The legacy SpecCipher implementation exhibited several architectural boundary defects:
1. **Cyclic Dependencies**: The `corpus` package imported `retrieval` to rebuild indexes, while `retrieval` imported data models from `corpus.models`.
2. **Boundary Leaks**: The `backend/app.py` serving module bypassed facade boundaries to directly call private methods in retrieval persistence (`IndexPersistence._chunks_from_rows`, `FaissIndex._import_faiss`).
3. **Implicit Layering**: Module responsibilities across ingestion, retrieval, reasoning, and presentation were intermingled without machine-enforced boundaries.

While microservices were considered, SpecCipher operates over a bounded corpus (~13,151 chunks, 54MB index) and requires low latency, single-binary deployment simplicity, and synchronous ranking pipelines. A distributed architecture would introduce network latency, RPC serialization overhead, and operational complexity without delivering tangible benefits.

## Decision

We adopt a **Modular Monolith** architecture rooted under `src/speccipher/`, structured into **seven strictly ordered architectural layers**:

```
Layer 1: Presentation & Delivery (speccipher.api, speccipher.cli)
   ↓
Layer 2: Application Services & Composition Root (speccipher.services)
   ↓
Layer 3: Corpus Release & Lifecycle Management (speccipher.release)
   ↓
Layer 4: Core Domain Engines (speccipher.retrieval, speccipher.decision,
         speccipher.ingestion, speccipher.understanding, speccipher.explanation)
   ↓
Layer 5: Infrastructure & Persistence (speccipher.catalog, speccipher.pdfio)
   ↓
Layer 6: Configuration (speccipher.config)
   ↓
Layer 7: Pure Domain Kernel (speccipher.domain)
```

### Architectural Invariants:
1. **Strict Downward Dependencies**: A layer may import from layers below it, but **NEVER** from layers above it.
2. **Layer 4 Sibling Independence**: The core engines in Layer 4 (`retrieval`, `decision`, `ingestion`, `understanding`, `explanation`) are independent siblings. They **MUST NEVER** import each other directly. They interact solely by exchanging domain entities defined in Layer 7 or coordinated by Layer 2 application services.
3. **Offline/Serving Plane Isolation**: Modules in the online serving plane (`api`, `services`, `retrieval`, `decision`, `understanding`, `explanation`) **MUST NEVER** import `speccipher.ingestion`.
4. **Machine Enforcement**: Boundaries are validated automatically on every commit via `import-linter` contracts and pytest AST checks.

## Consequences

### Positive
- **Elimination of Cycles**: Eliminates cyclic dependencies between corpus processing and retrieval.
- **Clear Seams for Testing**: Components can be tested in complete isolation using lightweight doubles and test fixtures.
- **Cognitive Clarity**: Developers know exactly where new logic belongs and what dependencies are permitted.
- **Fast Local Feedback**: The modular monolith runs as a single unified Python package with no external network service prerequisites for unit and integration testing.

### Negative & Trade-offs
- **Discipline Required**: Cross-cutting changes require coordination through domain models and service interfaces rather than quick direct imports.
- **Initial Migration Overhead**: Legacy code paths and imports must be systematically refactored to conform to the 7-layer hierarchy.

### Compliance
Enforced by:
- `.importlinter` configuration
- `tests/test_architecture_contracts.py`
- GitHub Actions CI (`.github/workflows/ci.yml`)
