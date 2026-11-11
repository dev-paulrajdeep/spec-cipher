# ADR-0003: Pure Domain Kernel in Standard Library Only

- **Status**: Accepted
- **Date**: 2026-11-11
- **Deciders**: SpecCipher Architecture Team
- **Consulted**: Milestone 1 Specialists

## Context

In the legacy architecture, domain concepts (`DocumentChunk`, `StandardDocument`, `RetrievalResult`, `RequirementFeatures`, etc.) were scattered across `corpus.models`, `retrieval.models`, and `intelligence.models`. These data classes imported third-party libraries including:
- `numpy` (for arrays and vector buffers)
- `torch` (for tensor operations)
- `faiss` (for index representations)
- `pymupdf` (for PDF structures)
- `flask` (for HTTP request parsing)

This coupling produced severe negative consequences:
1. **Transitive Dependency Bloat**: Any module or test requiring a simple domain concept (e.g. `StandardId` or `DocumentChunk`) transitively imported heavy native C/C++ libraries (PyTorch, FAISS).
2. **Circular Packaging**: `retrieval.models` imported from `corpus.models`, which tangled the offline data pipeline with serving retrieval code.
3. **Slow Tests**: Unit tests of core business logic suffered significant initialization latency due to heavy imports.

## Decision

We establish `src/speccipher/domain` as a **Pure Domain Kernel** governed by strict dependency invariants:

1. **Python Standard Library Only**: Modules under `speccipher.domain` may **ONLY** import from the Python Standard Library (`dataclasses`, `typing`, `enum`, `abc`, `uuid`, `math`, `re`, `datetime`, `collections`, etc.) and from sibling modules within `speccipher.domain`.
2. **Zero Third-Party Dependencies**: No `numpy`, no `torch`, no `faiss`, no `flask`, no `requests`, no `pydantic`.
3. **Zero Framework/Infrastructure Dependencies**: The domain kernel contains no I/O, no database connections, no file system manipulation, and no HTTP abstractions.
4. **Immutable Value Objects**: Fundamental domain entities are implemented as immutable dataclasses (`@dataclass(frozen=True)`):
   - `StandardId`: Canonical standard identifier (family, number, part, year, code).
   - `LogicalDocument`: Single publication or edition abstraction.
   - `SourceFile`: Physical PDF file provenance with absolute page mappings.
   - `Section` & `DocumentChunk`: Structured chunk representation with absolute page citations.
   - `StandardStub`: Representation for uningested referenced standards.
   - `Relationship`: Typed standard-to-standard relationships.
   - `Candidate`: Universal candidate entity representing retrieval matches.
   - `Recommendation`: Finalized recommendation object.
   - `RequirementFeatures`: Extracted requirement features and domain keywords.
   - `Confidence`: Categorical confidence (`High`, `Medium`, `Low`) with machine-readable reasons.

## Consequences

### Positive
- **Universal Portability**: The domain kernel can be safely imported by any layer (Layers 1 through 6) without fear of dependency contagion or circular imports.
- **Ultra-Fast Unit Testing**: Domain entities and business rules can be instantiated and tested in milliseconds with zero mock setup and zero external dependencies.
- **Stable Core**: Domain business concepts remain insulated from changes in external ML libraries, database engines, or web frameworks.

### Negative & Trade-offs
- **Primitive Conversion**: Numerical arrays and embedding vectors produced by FAISS or NumPy must be adapted to standard Python primitive types (`float`, `tuple[float, ...]`) when encapsulated within domain entities.

### Compliance
Enforced by:
- Automated AST analysis in `tests/test_architecture_contracts.py::TestArchitectureContracts::test_domain_purity_stdlib_only` using `sys.stdlib_module_names`.
- `.importlinter` contract `Domain Purity Contract`.
