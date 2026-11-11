# SpecCipher: Target Architecture and Repository Design

*Prepared 11 Nov 2026 from a static inspection of dev-paulrajdeep/spec-cipher at commit b5f6e79 (94 commits). Basis: the full file tree and sizes, every Markdown document, the core modules of all four layers, an AST-computed import graph, the corpus manifest, and git metadata. Not done: the test suite was not run and no model was loaded, so runtime behaviour is inferred from code and docs.*

## 1. Executive summary

SpecCipher is stronger than a typical hackathon repository. The corpus build is deterministic and fingerprinted, every chunk carries provenance, unknown metadata stays null, explanations fail open without touching the ranking, and there are 141 test functions. The weaknesses sit in the seams between those parts, plus one data problem that outweighs everything else.

- **Data reality (the biggest issue).** The manifest lists 12 PDFs and 13,151 chunks. Eight of those PDFs are volumes of one publication, SP 7:2026 (pages 1 to 2,999 split into files such as SP 7\_2026-401-800.pdf). Only the first volume resolved an identity, so only 1,177 chunks (about 9%) are recommendation-eligible. The other 11,923 SP 7 chunks are embedded and searched but can never be recommended.
- **Structure.** Four clean layers on paper, but corpus and retrieval import each other, the backend reaches into retrieval internals, and the Flask module builds the whole runtime at import time.
- **Operations.** About 180 MiB of generated corpus JSON is committed to git, and the server parses all canonical documents (about 170 MB of JSON) at startup only to read titles, scope text and relationships.

**Target in one sentence:** a modular monolith with two planes, an offline build plane that produces immutable, versioned Corpus Releases and an online serve plane that only ever loads one release, arranged around a pure domain kernel and enforced by automated import contracts.

This is not a microservice split and not a rewrite. Almost every move below is a rename, an extraction or a seam, and the existing tests act as the safety net.

## 2. What exists today

| Area | Files | Lines | Role today |
| --- | --- | --- | --- |
| corpus/ | 26 py | 3,861 | PDF to canonical JSON to chunks, plus the update workflow |
| retrieval/ | 26 py | 1,811 | BGE-M3 embeddings, exact FAISS IndexFlatIP, optional BGE reranker |
| intelligence/ | 31 py | 1,278 | constraints, rerank, applicability, evidence, relationship graph |
| backend/ | 3 py | 1,048 | Flask app, preflight, Ollama explanation adapter |
| scripts/ | 6 py | 772 | index build, corpus update, preflight, validation, browser e2e |
| tests/ | 20 py | 3,613 | 141 test functions, flat layout |
| frontend/ | 11 | 4,562 | static HTML, CSS, JS; 3 views; no build step |

Corpus today: 12 source PDFs, 13,151 chunks, 1 resolved and eligible document (SP 7:2026, first volume), 5 catalog or list PDFs, embedding model BAAI/bge-m3 (1024 dimensions). A flat index of 13,151 vectors is only about 54 MB, so exact search is the right choice at this scale.

Measured import graph between top-level packages (production code only): backend depends on corpus, intelligence and retrieval; corpus depends on retrieval; retrieval depends on corpus; intelligence depends on corpus and retrieval. The corpus and retrieval pair is a cycle.

## 3. Findings, ranked

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| F1 | Critical | 91% of indexed text can never be recommended | 8 SP 7 volume PDFs hold 13,100 of 13,151 chunks; only SP 7\_2026-1-400.pdf resolved identity. Each volume reports its own page\_count (400), so page numbers are probably per file and later volumes would be cited with wrong pages (inference, verify). The product can currently output exactly one standard, as the README admits. |
| F2 | High | Startup parses about 170 MB of JSON; generated data is in git | CorpusKnowledgeBase.from\_canonical\_json reads every canonical document in full to extract metadata, 'scope' sections and relationships. Those 12 files (0.15 to 24 MB each) plus corpus.chunks.jsonl (15.7 MB) are tracked, though the README says generated artifacts are never committed. |
| F3 | High | corpus and retrieval import each other | corpus.update.workflow imports retrieval to rebuild the index; retrieval and intelligence import corpus.models. Shared types live in the wrong package. |
| F4 | High | Boundary leaks | backend imports retrieval.corpus\_loader and retrieval.indexing.persistence and calls the private IndexPersistence.\_chunks\_from\_rows and FaissIndex.\_import\_faiss. HANDOVER.md forbids exactly this. Corpus-path policy lives in retrieval/corpus\_config.py. |
| F5 | High | Flask module builds the world at import | retriever, intelligence and qwen\_reasoner are created at module level by \_load\_runtime(). Routes, middleware, error handlers, serialization and bootstrap share one 14 KB file. No app factory. |
| F6 | High | Recommendation latency is coupled to LLM latency | The rank-1 explanation is generated inside the request. The Ollama timeout defaults to 180 s and the frontend timeout is 240 s, so a cold model can hold the whole result back. |
| F7 | Medium | Promised features are not in code | Confidence is missing (only final\_score exists). Hybrid retrieval is protocols only. The README relationship taxonomy (Test, Safety, Allied, Installation, ...) is not implemented; the graph holds only mechanical types such as references and supersedes. No tender PDF endpoint. The SQL store is dead: standards.db has 13,077 chunk rows, 0 standards rows, and nothing reads it. |
| F8 | Medium | Family-specific logic in general layers | intelligence/standards.py hard-codes an IS regex and \_promote\_explicit\_matches only matches IS or SP codes, while the corpus docs promise a family-agnostic pipeline. |
| F9 | Medium | Two reranking stages | retrieval/reranking/bge.py (RetrievalConfig) and intelligence/reranking (IntelligenceConfig) have separate flags, configs and model lifecycles. |
| F10 | Medium | Config sprawl | 29 os.getenv reads across 6 modules, legacy QWEN\_\* aliases, no single validated settings object. |
| F11 | Medium | Documentation drift | STRESS\_TEST\_REPORT.md says the app uses SQLite LIKE and a MiniLM model (neither is true now). HANDOVER.md describes a one-chunk IS 5120 corpus and addresses individuals. The README opens with an operational 'judge-ready' block above the product text. The demo needs corpus.eligible.chunks.jsonl, which is absent from this checkout, and resolve\_retrieval\_corpus\_path() silently falls back to the full 13,151-chunk file. |
| F12 | Medium | Packaging and delivery | No pyproject or lock file. requirements.txt is unpinned and ships pytest and playwright as runtime dependencies. Every command uses Windows-style .venv paths. sys.path.insert hacks in app.py and scripts. No CI, Dockerfile or LICENSE. Flat tests/ with no gpu, e2e or slow markers. |
| F13 | Low | Dead and oversized code | corpus/database/\*, frontend/js/api\_s.js (commented code plus a stray function), the scripts/preflight.py wrapper. update\_corpus\_from\_source is 381 lines, resolve\_document\_identity 215, \_process\_one\_pdf 208. |

## 4. What to keep exactly as is

- Deterministic chunking and corpus fingerprints; the fingerprint-checked index bundle idea (it becomes the release manifest).
- Provenance on every chunk, and the rule that unknown metadata stays null.
- Conservative identity resolution: references are not documents, catalogs are rejected, conflicts are never merged silently.
- The RetrievalService facade with its RLock and lazy model loading.
- Fail-open explanations that can never mutate ranking fields, plus the evidence guard that rejects unsupported standards in LLM output.
- Decomposed applicability scores and the dependency-injected IntelligenceEngine.
- Security headers and CSP, bounded request size, and JSON-only error responses.
- Synthetic-fixture tests and fake encoders, so unit tests never download models.

## 5. Target architecture

### 5.1 Principles

1. **Two planes, one contract.** Build offline, serve read-only. The Corpus Release is the only thing that crosses between them.
2. **Pure kernel.** The domain package has no I/O, no numpy, no Flask. This removes the cycle in F3.
3. **Ports and adapters at every heavy dependency** (FAISS, BGE, Ollama, SQLite, the PDF engine), so decisions never import infrastructure.
4. **One composition root.** Only services/container.py constructs models and wires components. Nothing does work at import time.
5. **Evidence discipline is a type, not a habit.** Provenance and null-for-unknown are encoded in domain types.
6. **Explanation never alters a decision** (existing guarantee, kept).
7. **Enforce with tools.** Import contracts, CODEOWNERS and size-limited pre-commit hooks, because rules that are only written down rot.

### 5.2 The two planes

```text
BUILD PLANE (offline, on demand)              SERVE PLANE (online, read-only)
data/sources/*.pdf                            HTTP -> api -> services.AnalyzeRequirement
  -> pdfio -> ingestion                         -> understanding (text/tender -> features)
     (identity, assembly, structure,            -> retrieval (dense + lexical -> RRF -> rerank)
      chunking) -> data/workspace/              -> decision (constraints, applicability,
  -> release.build                                  confidence, evidence, relations)
     (catalog + chunks + vectors)               -> explanation (optional, fail-open)
  -> release.verify -> release.publish        loads exactly ONE release at startup
              \______ data/releases/<release_id>/ ______/
```

### 5.3 Dependency rules (enforced, top may import below, never upward)

| Layer | Packages | Rule |
| --- | --- | --- |
| 1 | cli, api | Delivery. api talks only to services and DTOs. |
| 2 | services | Application layer and the only composition root. |
| 3 | release | Build and load of Corpus Releases. |
| 4 | retrieval, decision, understanding, explanation, ingestion | Independent siblings: none imports another. They exchange data only through domain types. |
| 5 | catalog, pdfio | Storage and file-format infrastructure. |
| 6 | config | One validated settings object. |
| 7 | domain | Pure kernel, standard library only. |

Hard rules: the serve plane never imports ingestion (no PDF parsing at request time; tender upload uses pdfio, which is separate); nothing imports api; only services/container.py instantiates models.

```toml
[tool.importlinter]
root_package = 'speccipher'

[[tool.importlinter.contracts]]
name = 'Layered architecture'
type = 'layers'
layers = [
  'speccipher.cli | speccipher.api',
  'speccipher.services',
  'speccipher.release',
  'speccipher.retrieval | speccipher.decision | speccipher.understanding | speccipher.explanation | speccipher.ingestion',
  'speccipher.catalog | speccipher.pdfio',
  'speccipher.config',
  'speccipher.domain',
]
```

## 6. Domain model (the shared kernel)

| Concept | Purpose and key fields |
| --- | --- |
| StandardId | Value object: family (IS, SP, ISO, IEC, EN, ASTM, ...), number, optional part and year. Parsing comes from a registry of per-family patterns, replacing the hard-coded IS regex. |
| LogicalDocument | One publication or edition: id, standard\_id or None, title, type, status, edition, identity {status, confidence, source}, parts, eligible. Identity source is one of filename, header, pdf\_metadata, multipart\_inherit, operator\_override. |
| SourceFile | One physical PDF: relative path, sha256, page\_count, part\_index, absolute page offset and range, extraction status. This is the fix for F1. |
| Section and Chunk | Chunk provenance uses absolute page ranges plus source\_file and part\_index, so citations stay correct across volumes. |
| StandardStub | A standard that is referenced by an indexed document but not itself ingested: identifier, title if evidenced, relationships, ingested=false. It may appear as a related standard with a warning, never as the primary recommendation. |
| Relationship | Typed edge. Two families: mechanical (references, supersedes, amendment\_of, revision\_of) and the README roles (normative\_reference, test, safety, allied, application, installation). |
| Candidate | Chunk plus dense score, lexical score and fused rank. Replaces RetrievalResult as the cross-layer type. |
| Recommendation | standard\_id, title, role (primary or supporting), relevance, confidence, score\_breakdown, evidence\[\], related\[\], status, version, warnings\[\]. |

**Relevance versus Confidence** (a README promise that is not yet implemented): relevance stays a bounded ordering signal, not a probability, as AGENTS.md and API.md require. Confidence is categorical (high, medium, low) with machine-readable reasons, computed from evidence sufficiency: eligible and resolved identity, number of independent evidence chunks, presence of a Scope clause, requirement completeness (product, application, material, conditions), score margin over the next candidate, and status currency. Percentages appear only after calibration on the evaluation set.

## 7. Target repository structure

```text
spec-cipher/
├── pyproject.toml              # deps, extras [serve,build,gpu,dev,e2e], tool config, console script
├── uv.lock                     # pinned and reproducible
├── Makefile                    # cross-platform: serve | test | lint | contracts | release
├── README.md                   # one page: what it is, quickstart, links
├── AGENTS.md  LICENSE  CONTRIBUTING.md  CHANGELOG.md  SECURITY.md
├── .pre-commit-config.yaml     # ruff, secret scan, block files larger than 5 MB
├── .github/
│   ├── workflows/ci.yml        # lint, types, import contracts, unit, integration, contract
│   ├── workflows/nightly.yml   # gpu, e2e, eval (manual or self-hosted)
│   ├── CODEOWNERS              # folder to team-role mapping
│   └── pull_request_template.md
├── configs/                    # default.toml, demo.toml, prod.toml (non-secret, one schema)
├── docs/
│   ├── architecture/           # overview.md, dependency-rules.md, adr/0001-*.md
│   ├── contracts/              # openapi.yaml, release-manifest.schema.json
│   ├── guides/                 # quickstart, update-corpus, demo-runbook, deployment
│   ├── reference/              # corpus.md, retrieval.md, decision.md
│   └── reports/                # dated reports, e.g. 2026-09-retrieval-stress-test.md
├── src/speccipher/
│   ├── domain/                 # PURE kernel (stdlib only)
│   │   ├── identity.py  documents.py  chunks.py  relations.py
│   │   ├── requirement.py  candidates.py  recommendation.py  errors.py
│   ├── config/settings.py      # ONE settings object; SPEC_CIPHER_* plus legacy QWEN_* aliases
│   ├── pdfio/                  # PDF reader and OCR port, shared by ingestion and tender upload
│   ├── ingestion/              # BUILD: PDFs to canonical documents
│   │   ├── discovery.py  fingerprint.py  validation.py  serialization.py
│   │   ├── identity/{resolver.py, overrides.py, families/{is_code,sp,iso,iec,astm,en}.py}
│   │   ├── assembly/multipart.py     # NEW: volumes become one logical document
│   │   ├── structure/{cleaner,normalizer,section_detector}.py
│   │   ├── extraction/{metadata,relationships}.py
│   │   ├── chunking/chunker.py
│   │   └── pipeline.py               # staged: discover, inspect, process, dedupe, manifest
│   ├── catalog/                # NEW: structured standards store (the SQL layer)
│   │   ├── schema.sql  migrations/
│   │   ├── repository.py             # port; SQLite adapter now, Postgres later
│   │   └── queries.py                # context_for, related, scope_text, indexed_standards
│   ├── release/                # THE CONTRACT BETWEEN THE PLANES
│   │   ├── manifest.py  verifier.py  publisher.py  loader.py
│   │   └── build/{embed.py, faiss_build.py, fts_build.py, catalog_build.py}
│   ├── retrieval/              # SERVE: query to candidates
│   │   ├── ports.py                  # Encoder, VectorIndex, LexicalIndex, Reranker protocols
│   │   ├── dense/{faiss_index.py, bge_m3.py}
│   │   ├── lexical/fts5.py           # NEW: BM25 via SQLite FTS5
│   │   ├── fusion/rrf.py             # NEW: reciprocal rank fusion
│   │   ├── rerank/{bge_reranker.py, token_overlap.py}   # the ONLY rerank stage
│   │   └── service.py                # RetrievalService facade (RLock, lifecycle)
│   ├── understanding/          # SERVE: text or tender to RequirementFeatures
│   │   ├── interpreter.py  llm_interpreter.py (optional, injected)
│   │   ├── explicit_refs.py          # explicit standard mentions, family-agnostic
│   │   └── tender.py                 # NEW: uploaded PDF to requirement text
│   ├── decision/               # SERVE: was intelligence/
│   │   ├── constraints/{admissibility,scope,status,version}.py
│   │   ├── applicability/{assessor,rules}.py   # adds metadata_compatibility sub-score
│   │   ├── confidence/               # NEW: evidence sufficiency to Confidence + reasons
│   │   ├── evidence/{locator,formatter}.py
│   │   ├── relationships/graph.py    # reads catalog, not 170 MB of JSON
│   │   ├── scoring.py  pipeline.py   # pipeline orchestrates; stages never import each other
│   │   └── instrumentation.py
│   ├── explanation/            # SERVE: optional and fail-open
│   │   ├── ports.py  deterministic.py  guard.py  ollama.py
│   ├── services/               # APPLICATION: the only place that composes
│   │   ├── analyze.py  standards.py  health.py
│   │   └── container.py              # settings to release to services; lazy, nothing at import
│   ├── api/                    # DELIVERY: Flask
│   │   ├── app.py                    # create_app(container) factory
│   │   ├── routes/{analyze,standards,health,static}.py
│   │   ├── schemas.py  serializers.py  errors.py
│   │   └── middleware.py             # security headers, CORS, request id, timing
│   ├── observability/{logging,metrics,request_context}.py
│   └── cli/main.py             # speccipher ingest | release build/verify/publish/rollback | serve | preflight | eval
├── web/                        # frontend: still no build step, but modular
│   ├── index.html
│   ├── src/{main.js, config.js, api/client.js, state/store.js, views/*.js, components/*.js}
│   ├── styles/{tokens.css, base.css, components/*.css, responsive.css}
│   └── tests/contract.test.mjs
├── eval/                       # NEW: roadmap Phase 6 made real
│   ├── datasets/{gold_queries.jsonl, no_match.jsonl, ambiguous.jsonl}
│   ├── run.py  metrics.py            # Recall@K, MRR, nDCG, abstention precision
│   └── reports/
├── tests/{unit/, integration/, contract/, e2e/, gpu/, fixtures/}
├── deploy/{Dockerfile, compose.yml, env.example}   # compose has an optional ollama profile
└── data/                       # gitignored; shared via release assets, LFS or DVC
    ├── sources/                # was officialBIS/ (env alias kept)
    ├── workspace/              # was corpus/data/documents and extraction cache
    └── releases/{<release_id>/, CURRENT}
```

## 8. Current to target move map

| Current | Target |
| --- | --- |
| corpus/models.py | domain/\* (split by concept) |
| corpus/ingestion/{pdf\_reader,ocr,extractor}.py | pdfio/ |
| corpus/processing, metadata, relationships, chunking | ingestion/structure, extraction, chunking |
| corpus/update/identity.py | ingestion/identity/resolver.py plus families/ |
| corpus/update/workflow.py, corpus/pipeline.py | ingestion/pipeline.py (staged) and release/build/ |
| corpus/database/\* | deleted; replaced by catalog/ |
| corpus/data/\*\* | data/workspace and data/releases (untracked) |
| retrieval/corpus\_config.py | config/ paths plus release/loader.py |
| retrieval/corpus\_loader.py, indexing/\* | release/ (reader, build) and retrieval/dense/faiss\_index.py |
| retrieval/embeddings, reranking, search, hybrid | retrieval/dense, rerank, lexical, fusion, service |
| retrieval/models/\* | domain/candidates.py |
| retrieval/evaluation/\* | eval/ |
| intelligence/{constraints,applicability,evidence,relationships,pipeline} | decision/\* |
| intelligence/reranking/\* | retrieval/rerank/ (single stage) |
| intelligence/applicability/feature\_extraction.py | understanding/interpreter.py |
| intelligence/standards.py, models/\* | domain/identity.py, recommendation.py |
| backend/app.py | api/ (split) plus services/analyze.py and container.py |
| backend/preflight.py | release/verifier.py (public APIs only) |
| backend/qwen\_reasoning.py | explanation/{ollama,guard,deterministic}.py |
| scripts/\* | cli/ commands; e2e\_smoke moves to tests/e2e |
| frontend/ | web/ |
| officialBIS/ | data/sources/ |
| tests/\* | tests/{unit,integration,contract,e2e,gpu} |
| STRESS\_TEST\_REPORT.md | docs/reports/ (dated, historical) |
| CORPUS, RETRIEVAL, INTELLIGENCE, API, HANDOVER .md | docs/reference and docs/contracts (rewritten to current behaviour) |

Use git mv so history follows the files, and rename packages in one coordinated window to avoid breaking open pull requests.

## 9. The Corpus Release contract

```text
data/releases/20261111T093000Z-8f3a1c2/
├── release.json     # schema_version, release_id, created_at, git_sha, chunking_version,
│                    # embedding {model, revision, dim, normalized}, index_type,
│                    # counts {documents, eligible_documents, chunks, eligible_chunks, vectors},
│                    # sources [{path, sha256, pages, disposition}], files {name: sha256, bytes}
├── catalog.sqlite   # logical_documents, source_files, relationships, scope_text,
│                    # chunk metadata, FTS5 table for lexical search
├── chunks.jsonl.zst # chunk text with provenance
├── vectors.faiss    # plus chunk_ids.json
└── QUALITY.json     # unresolved docs, OCR-required pages, identity conflicts, eligible ratio
data/releases/CURRENT    # one line: the active release_id, swapped by atomic rename
```

Lifecycle: build into a staging folder, run release verify (checksums, counts, unit-norm vectors, catalog, chunks and vectors agree, embedding identity), publish by atomic rename and a CURRENT swap, serve loads once at startup, and rollback is rewriting CURRENT. This replaces the four-file bundle, the manual fingerprint dance and backend/preflight.py, and it removes the 170 MB startup parse because the decision layer reads the small catalog instead. Invariants: a loaded release is read-only, the request path never writes, and a build never touches the serving release.

## 10. API v1

- Keep POST /get\_result as a deprecated alias so the current frontend and its contract tests keep working.
- **POST /v1/analyze**: input {requirement, options}. Output adds request\_id, release\_id, a requirement\_understanding block (product, application, material, conditions, completeness, missing\[\]), and result\_state in recommendations, no\_match or insufficient\_input. This delivers the README promise of 'Specification incomplete' guidance. Each recommendation carries standard\_id, title, role, relevance, confidence {level, reasons\[\]}, score\_breakdown, evidence\[\], related\[\], status and version.
- **POST /v1/explain** for the generative summary, so ranking returns in milliseconds and the UI fills the explanation in progressively (fixes F6). Until then, expose include\_explanation=false on analyze.
- **POST /v1/analyze/document** for tender PDF upload (Phase 6): size, type and page caps, sandboxed parse, asynchronous job id for slow files.
- **GET /v1/standards/{id}** and **GET /v1/standards?q=** over the catalog.
- **GET /health/live**, **GET /health/ready** (release\_id, model states, Ollama as optional) and **GET /version**.
- One error mapper turning domain errors into problem+json; OpenAPI generated from api/schemas.py; additive-only changes inside v1.

## 11. Quality system

| Tier | Scope | Runs |
| --- | --- | --- |
| unit | pure logic, fake encoders, no models (default) | every push |
| integration | build a tiny fixture release, then serve it end to end | every push |
| contract | API against OpenAPI, release manifest schema, frontend DTO shape | every push |
| e2e | Playwright against a running app (marker e2e) | nightly or manual |
| gpu | real BGE-M3 and reranker (marker gpu) | nightly or manual |
| eval | gold queries, no-match and ambiguous sets with regression thresholds | per release build |

Release gates (a release is not publishable unless these pass): no duplicate chunk IDs; embedding identity matches the manifest; eligible ratio reported and above an agreed floor; no orphan chunks without a document; eval Recall@5 not below the previous baseline minus a tolerance; no-match precision not degraded. Pre-commit blocks any file over 5 MB, which would have stopped the 24 MB JSON commits. Stage timings that PipelineMetrics already records should be emitted as structured log fields with the request id. Before any public deployment add rate limiting and strict upload validation.

## 12. Migration roadmap

Tag the current main as demo-freeze before starting. Phases 0 to 2 are mechanical and covered by the existing tests; later phases change behaviour and should be merged before a demo only if eval shows parity.

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| 0 Hygiene (about 1 day) | Delete dead code (corpus/database, api\_s.js, scripts/preflight.py). Fix doc drift and move the stress report to docs/reports. Add pyproject and pinned deps; move pytest and playwright to extras. Stop tracking generated data (git rm --cached plus .gitignore) and add the pre-commit size guard. Choose a LICENSE. Add CODEOWNERS. | Fresh clone runs the full unit suite with one command; repo tracks no generated data. |
| 1 Structure (1 to 2 days) | src/speccipher layout with git mv. Extract domain and break the corpus and retrieval cycle. Add the settings module and import-linter in CI. | Import contracts pass; zero cross-layer private-attribute access; all existing tests green. |
| 2 Serving seam (1 to 2 days) | create\_app factory, services/analyze, container, DTOs, error mapper, request id. Lazy runtime construction. Contract tests using a fake retriever. | Importing the app loads no model; API tests run in under a few seconds. |
| 3 Release and catalog | Manifest, catalog.sqlite, release build, verify, publish, loader. decision/relationships reads the catalog. Replace backend/preflight with release verify. | Startup no longer parses canonical JSON; rollback by CURRENT works; serving output identical to before on the fixture set. |
| 4 Corpus correctness | ingestion/assembly/multipart with absolute pages and operator\_override identity for SP 7 volumes. StandardStub for referenced codes. Gold query set started. | Eligible chunks rise from about 9% of SP 7 to its full volume set; citations carry correct absolute pages. |
| 5 Retrieval and decision maturity | FTS5 lexical plus RRF fusion; single rerank stage; Confidence model; Relevance and Confidence shown separately; family registry beyond IS. | Eval beats the dense-only baseline on Recall@5 and no-match precision. |
| 6 Product expansion | /v1 API with split explain, tender upload, standards detail pages, Docker deploy, auth and rate limits, optional Postgres and pgvector. | Demo runbook reproducible from a clean machine via compose. |

Ownership by existing team roles (via CODEOWNERS): AI/ML owns ingestion, retrieval, decision, understanding, explanation and eval; backend owns api, services, release and deploy; frontend owns web; SQL support owns catalog.

## 13. If you only do three things before the demo

1. **Lift eligibility (F1).** Declare the SP 7 volumes as one logical document through an operator override, so identity is recorded as human-asserted rather than invented, and fix page numbering. This is the highest value change in the repository.
2. **Remove the startup parse (F2).** Even before a full catalog, write a small sidecar JSON of metadata, scope text and relationships during update\_corpus and load that instead.
3. **Install the guardrails.** src layout, domain extraction and import-linter prevent the cycle from returning.

## 14. Decisions, risks and ADRs

**Decisions to confirm**

1. Product scope: should referenced IS codes (for example IS 2645 inside SP 7) be surfaced as supporting stubs, with the primary recommendation reserved for ingested documents? The current code does this implicitly and the README goal suggests it.
2. How to share data: LFS, DVC, cloud bucket or release assets.
3. History rewrite: removing the 180 MiB from past commits needs git filter-repo, a force-push and a team re-clone. The cheaper alternative is to stop tracking going forward and accept the history size.
4. Keep Confidence categorical until the eval set can calibrate it.
5. Postgres only when there are concurrent writers or millions of chunks; SQLite plus flat FAISS is correct at today's scale.

**Risks**

- Rename churn can strand open branches; do Phase 1 in a single announced window.
- An operator override is only as good as the operator; record who asserted it and when in the manifest.
- Multi-volume assembly must not merge genuinely different documents; require contiguous page ranges and a shared filename stem, and keep conflicts as unresolved.

**ADRs to write first** (one page each, in docs/architecture/adr): 0001 modular monolith; 0002 src layout and namespace; 0003 Corpus Release as the plane contract; 0004 SQLite catalog; 0005 domain kernel; 0006 hybrid retrieval with RRF; 0007 relevance versus confidence; 0008 optional fail-open explanation; 0009 versioned API.
