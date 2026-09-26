# SpecCipher

### AI-Powered Standards Recommendation Engine

> **Judge-ready MVP scope (September 2026).** This workspace currently serves
> the source-derived SP 7:2026 corpus from `corpus/data/corpus.eligible.chunks.jsonl`.
> Its manifest records one resolved, recommendation-eligible standard. Verify
> the active corpus with preflight and `GET /health` before a demo. The historical
> checklist fallback is not the source for the judge walkthrough.
> Authentic publications belong only in `officialBIS/` and are ingested by the
> general-purpose `scripts\update_corpus.py` pipeline.

## Source-of-truth distinction

| Artifact | Role | Origin |
| --- | --- | --- |
| `officialBIS/` | User-supplied technical/standards PDFs only | Local; never generated or committed |
| `scripts/update_corpus.py` | Offline general PDF → JSONL → FAISS rebuild | Corpus update time |
| `corpus/data/documents/`, `corpus.chunks.jsonl`, `manifest.json` | Generated corpus projection | Produced only by the updater |
| `corpus/data/IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json` | Historical/demo serving corpus | Checked-in demo data; **not** an official PDF extraction |
| Flask runtime | Prefers eligible source-derived chunks, then the canonical source-derived corpus, then the historical checklist | Runtime request time; never parses PDFs |

SpecCipher never downloads source documents, never fabricates PDFs, and never
treats mentioned document numbers inside a catalog PDF as independently indexed
documents.

## Updating the corpus (general-purpose PDF ingestion)

**Corpus update time** is separate from **runtime request time**. The server
never parses `officialBIS/` on startup or per request.

### Adding a new PDF

1. Place any suitable technical/standards PDF in:

```text
officialBIS/
```

2. Run one command:

```powershell
.\.venv\Scripts\python.exe scripts\update_corpus.py
```

3. Verify the update summary.
4. Start SpecCipher:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

No manual JSON editing. No manual FAISS commands. No Python changes per new PDF.

Pipeline:

```text
PDF → extraction → metadata/identity → structure-aware chunking → JSONL
  → BGE-M3 embeddings → FAISS IndexFlatIP → manifest/provenance
```

What the updater does:

- discovers PDFs recursively under `officialBIS/` (or `SPEC_CIPHER_BIS_SOURCE_DIR`)
- extracts text with the existing PyMuPDF corpus pipeline
- resolves document identity conservatively (filename / title / header)
- rejects catalog/list PDFs that only mention many document numbers
- preserves unresolved ingestible documents without inventing numbers
- writes canonical documents under `corpus/data/documents/`
- writes retrieval chunks to `corpus/data/corpus.chunks.jsonl`
- rebuilds FAISS under `retrieval-index/`
- leaves the historical demo checklist JSON untouched

If no PDF produces a usable corpus, the updater reports failures, clears
source-derived generated artifacts, and does **not** manufacture a substitute
PDF or corpus document. Runtime can still use the historical demo checklist.

Dry-run:

```powershell
.\.venv\Scripts\python.exe scripts\update_corpus.py --check
```

## Run the current MVP

Use Python 3.11+ and a project-local virtual environment, then install
`requirements.txt`. Obtain/cache the exact `BAAI/bge-m3` model before offline
operation; normal serving does not download models.

From the repository root, validate that the persisted retrieval bundle matches
the configured source-derived corpus, then launch Flask:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py
.\.venv\Scripts\python.exe backend\app.py
```

The preflight command validates corpus identity, chunk mapping, vector count,
and embedding dimensions without loading the embedding model. Flask loads
`BAAI/bge-m3` from the local cache (or the configured model source) at startup.
For disconnected startup, set `SPEC_CIPHER_HF_OFFLINE=1` after confirming the
model is cached. If the generated source-derived corpus or its matching index
is missing, use the documented corpus updater with the authentic PDFs in
`officialBIS/`; do not silently substitute the historical checklist for this demo.

When authentic officialBIS PDFs need to be ingested or refreshed:

```powershell
.\.venv\Scripts\python.exe scripts\update_corpus.py
.\.venv\Scripts\python.exe scripts\preflight.py
.\.venv\Scripts\python.exe backend\app.py
```

For disconnected startup, set `SPEC_CIPHER_HF_OFFLINE=1`. `GET /health` reports
`corpus_origin` as `historical_demo_checklist` or `officialbis_derived`.
Qwen/Ollama remains optional explanation-only.

### Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `SPEC_CIPHER_BIS_SOURCE_DIR` | PDF source directory | `officialBIS/` |
| `SPEC_CIPHER_OLLAMA_MODEL` | Optional explanation model name | `qwen3-4b-thinking:latest` |
| `SPEC_CIPHER_OLLAMA_TIMEOUT_SECONDS` | Bounded Ollama timeout | `180` |
| `SPEC_CIPHER_OLLAMA_THINK` | Enable model thinking tokens (`1`/`0`) | `0` |
| `SPEC_CIPHER_OLLAMA_KEEP_ALIVE` | Ollama keep-alive after explanation | `0` |
| `SPEC_CIPHER_HF_OFFLINE` | Force Hugging Face offline mode | unset |
| `SPEC_CIPHER_EMBEDDING_DEVICE` | `auto` / `cuda` / `cpu` | `auto` |
| `SPEC_CIPHER_ALLOW_CPU_FALLBACK` | Allow CPU if CUDA fails | `1` |
| `SPEC_CIPHER_HOST` / `SPEC_CIPHER_PORT` | Flask bind address | `127.0.0.1` / `5000` |

Legacy aliases `QWEN_OLLAMA_HOST`, `QWEN_MODEL_NAME`, `QWEN_TIMEOUT_SECONDS`,
`QWEN_THINK`, and `QWEN_KEEP_ALIVE` remain supported.

### Current capability and boundary

The current manifest marks SP 7:2026 as the single recommendation-eligible
document (1,177 eligible chunks). Other source files remain unresolved or are
catalog/list material; mentions of other codes do not make them independently
eligible recommendations. Unknown source metadata remains unknown.

### Judge demo procedure

1. Run `scripts\preflight.py` and confirm it reports a valid bundle.
2. Start `backend\app.py`; verify `GET /health` reports `"status":"ready"`,
    `"corpus_origin":"officialbis_derived"`, and
    `"corpus":"corpus.eligible.chunks.jsonl"`.
3. Submit `building design considering wind effects on occupant comfort inside and outside`.
4. Verify the SP 7:2026 recommendation includes section 4.1.11 evidence from
    pages 185–186 of `SP 7_2026-1-400.pdf`.
5. Submit `general-purpose centrifugal pumps for industrial water service` and
    verify the no-match state (no recommendations).
6. Qwen/Ollama is optional; when unavailable the API and UI retain the finalized
    recommendation and show a deterministic evidence-grounded explanation.

The browser acceptance command is
`.\.venv\Scripts\python.exe scripts\e2e_smoke.py`. It starts and stops only the
backend process it owns, checks `/health`, exercises the recommendation, the
optional-Qwen fallback, and the no-match state, and writes evidence under
`.tmp/e2e_smoke/`. Playwright Chromium must be installed in the active project
environment.

**Smart India Hackathon 2026 --- SIH26108**

SpecCipher is an AI-assisted decision-support system for identifying
potentially applicable **standards** from procurement requirements.

It helps procurement users move from an unstructured requirement---such
as a product description, technical specification, application,
operating condition, or tender document---to a structured set of
potentially relevant standards, with applicability reasoning, supporting
evidence, and related-standard context.

------------------------------------------------------------------------

## The Problem

Identifying the right standard for a procurement requirement can
be difficult.

A procurement specification may contain product information, technical
parameters, intended applications, performance requirements, or
operating conditions without explicitly stating the applicable standard.
At the same time, standards may have related scopes, references to other
standards, revisions, amendments, and supporting requirements.

A simple keyword search is therefore often insufficient.

The real question is not:

> **"Which standard contains these words?"**

It is:

> **"Which Indian Standard is potentially applicable to this procurement
> requirement, why does it match, and what related standards should also
> be considered?"**

------------------------------------------------------------------------

## Our Approach

SpecCipher combines requirement understanding, standards retrieval,
applicability assessment, and explainable results into a single
workflow.

``` text
┌──────────────────────────────┐
│   Procurement Requirement    │
│  Text / Technical Spec / PDF │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Understand the Requirement │
│ Product • Application •      │
│ Parameters • Conditions      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Retrieve Candidates      │
│     Relevant IS Standards    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Rank & Assess Applicability  │
│ Relevance • Confidence       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Identify Standard Relations  │
│ Normative • Test • Safety •  │
│ Allied • Application         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Explain the Recommendation   │
│ Evidence • Reasoning • Status│
└──────────────────────────────┘
```

------------------------------------------------------------------------

## Key Features

### 🔎 Requirement Understanding

Accept procurement information in natural language rather than requiring
users to know the relevant IS number beforehand.

Inputs can include:

-   Product descriptions
-   Technical requirements
-   Intended applications
-   Operating conditions
-   Tender or specification documents

### 📄 Tender / PDF Analysis

Allow procurement documents to be supplied for analysis so that relevant
technical information can be extracted from the document.

### 🧠 AI-Assisted Standards Retrieval

Identify potentially relevant standards based on the meaning and
technical context of a requirement rather than relying only on exact
keyword matches.

### 📊 Applicability Assessment

Rank retrieved standards according to their relevance to the submitted
requirement and distinguish recommendation confidence from semantic
relevance.

### 🔗 Related Standards

Represent standards in context rather than treating every returned IS
number as an equivalent recommendation.

Potential relationships include:

  Relationship              Purpose
  ------------------------- --------------------------------------------------
  **Primary**               Main potentially applicable standard
  **Normative Reference**   Standard referenced by another standard
  **Test Standard**         Relevant testing or measurement standard
  **Safety Standard**       Relevant safety requirements
  **Allied Standard**       Closely related technical standard
  **Application**           Standard relevant to the intended application
  **Installation**          Relevant installation or implementation guidance

### 💡 Explainable Recommendations

Each recommendation can be accompanied by information such as:

-   Why it was selected
-   Matching requirements
-   Relevant scope information
-   Supporting evidence
-   Standard relationships
-   Status or version information

The goal is to make the recommendation understandable rather than
presenting an unexplained score.

### ⚠️ Uncertainty Handling

If the specification does not contain enough information for a reliable
recommendation, SpecCipher should communicate that limitation instead of
manufacturing false precision.

Example:

``` text
Specification incomplete

Additional information may improve matching:

• Product type
• Intended application
• Technical requirements
• Operating conditions
```

------------------------------------------------------------------------

## Example

### Input

``` text
We need a waterproofing compound for use with
cement concrete in an underground water reservoir.
```

### Requirement Understanding

``` text
Product:
Waterproofing compound

Application:
Underground water reservoir

Material / context:
Cement concrete

Requirement:
Waterproofing
```

### Recommendation

``` text
PRIMARY APPLICABLE STANDARD

IS XXXX
Indian Standard Title

Relevance: 94%
Confidence: High
Status: Current
```

### Supporting Context

``` text
RELATED / SUPPORTING STANDARDS

IS YYYY
Normative Reference

IS ZZZZ
Test Standard

IS AAAA
Application / Allied Standard
```

The interface can then expose the reasoning and supporting evidence
behind the recommendation.

------------------------------------------------------------------------

## Why SpecCipher?

Traditional search can answer:

> **"Which documents look similar?"**

SpecCipher is designed to answer a more useful procurement question:

> **"Which Indian Standard appears applicable to this requirement, what
> role does it play, what related standards matter, and what evidence
> supports the recommendation?"**

This makes the system a **procurement decision-support workflow**,
rather than simply a standards search box.

------------------------------------------------------------------------

## System Architecture

``` text
                         ┌─────────────────┐
                         │     Frontend    │
                         │   HTML / CSS /  │
                         │    JavaScript   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Flask API    │
                         │     Backend     │
                         └────────┬────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
        ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Requirement    │ │ Standards    │ │   Database   │
        │ Processing     │ │ Retrieval    │ │ / Metadata   │
        └───────┬────────┘ └──────┬───────┘ └──────┬───────┘
                │                 │                │
                └─────────────────┼────────────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Applicability   │
                         │ & Recommendation│
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Explainable     │
                         │ Results         │
                         └─────────────────┘
```

## Corpus preparation

The repository now includes a deterministic multimodal corpus layer for
preparing source standards for later retrieval work. It preserves source
pages, text-block/table/image/vector-figure geometry, raw visual assets and
full-page render fallbacks; routes sparse PDF pages to optional OCR; retains
raw and cleaned text; extracts only evidenced metadata and relationships;
detects standard clauses; and emits canonical JSON plus backward-compatible
provenance-complete chunk JSONL. See [CORPUS.md](CORPUS.md) for the schema,
raw-asset boundary, OCR adapter boundary, and usage.

## Retrieval preparation

The retrieval layer consumes corpus JSON/JSONL chunks, encodes them with a
pretrained sentence-embedding model, and performs exact normalized-vector
FAISS Top-K search. It preserves corpus provenance and exposes semantic
candidate chunks only; it does not make applicability decisions. See
[RETRIEVAL.md](RETRIEVAL.md) for persistence, score semantics, usage, and
evaluation.

## Intelligence decision engine

The Intelligence layer consumes retrieval candidates and explicit corpus facts
to apply binary constraints, rerank, assess decomposed applicability, select
provenance-complete evidence, and traverse stated relationships. It produces
explainable recommendations, not API responses or opaque LLM decisions. See
[INTELLIGENCE.md](INTELLIGENCE.md) for the contract and scoring semantics.

## Browser E2E acceptance test

The reproducible browser acceptance script is `scripts/e2e_smoke.py`. It
starts the real Flask entrypoint (`backend/app.py`), checks `/health`, serves
the existing frontend, launches installed Chromium through Playwright, and
drives the actual query form with:

```text
general-purpose centrifugal pumps for industrial water service
```

Prerequisites are the project Python environment, the persisted
`retrieval-index/` bundle, cached BGE-M3 artifacts, and Playwright's installed
Chromium. Ollama and Qwen are optional; an unavailable or invalid Ollama
response is expected to exercise the deterministic explanation fallback.
Run it from the repository root:

```powershell
python scripts/e2e_smoke.py
```

The script polls `/health` and the served HTML before opening Chromium, verifies the real
`/get_result` response against the rendered recommendation page, captures a
successful screenshot, then repeats the browser flow with an invalid Ollama
endpoint to verify fail-open explanation behavior and with a no-result query
to verify that Qwen is not invoked. It writes `summary.json`, backend logs,
and `successful-result.png` under `.tmp/e2e_smoke/`.

PASS requires the successful explanation flow plus the browser-visible
failure-isolation and no-result flows. Missing prerequisites, startup or HTTP
errors, absent rendered fields, recommendation mutation, or cleanup failures
exit non-zero. The script terminates only backend processes it starts;
Ollama is not started or stopped by the script, and Qwen requests use the
application's existing `keep_alive=0` contract.

Expected runtime depends primarily on BGE model loading and Ollama response
latency and is commonly several minutes on a cold local environment.

------------------------------------------------------------------------

## AI / Retrieval Workflow

The intended AI pipeline follows a retrieval-first approach:

``` text
User Input
    │
    ▼
Text / PDF Processing
    │
    ▼
Requirement Extraction
    │
    ▼
Semantic Representation
    │
    ▼
Candidate Retrieval
    │
    ▼
Relevance Ranking
    │
    ▼
Applicability Assessment
    │
    ▼
Relationship Resolution
    │
    ▼
Evidence Collection
    │
    ▼
Recommendation
```

The exact models, retrieval methods, and supporting libraries may evolve
during development as the system is evaluated.

------------------------------------------------------------------------

## Relevance vs Confidence

SpecCipher treats **relevance** and **confidence** as separate concepts.

### Relevance

How closely a standard matches the submitted procurement requirement.

### Confidence

How confident the system is that the recommendation is justified by the
available information and evidence.

For example:

``` text
Relevance: 94%
Confidence: Medium
```

A high semantic match does not automatically imply that the available
evidence is sufficient for a high-confidence recommendation.

------------------------------------------------------------------------

## Technology Stack

### Frontend

-   HTML
-   CSS
-   JavaScript

### Backend

-   Python
-   Flask

### AI / Machine Learning

The AI layer is intended to support:

-   Natural-language processing
-   Semantic representations
-   Embedding-based retrieval
-   Candidate ranking
-   Applicability assessment
-   Evidence-based recommendations

### Database

-   SQL-based storage for structured standards metadata and application
    data

------------------------------------------------------------------------

## Design Principles

### Retrieval Before Generation

Recommendations should be grounded in retrieved standards information
rather than relying solely on model-generated knowledge.

### Evidence Over Assertions

The system should provide supporting evidence wherever available.

### Primary vs Supporting Standards

A primary recommendation should be distinguishable from normative
references, test standards, safety standards, and other related
standards.

### Uncertainty Is Explicit

When evidence or input information is insufficient, the system should
communicate uncertainty rather than fabricate precision.

### Human-in-the-Loop

SpecCipher is designed to assist procurement professionals, not replace
professional judgment.

### Source Authority

The official BIS publication remains the authoritative source for
standards information.

------------------------------------------------------------------------

## Current Status

🚧 **Active Development**

SpecCipher is being developed as a solution for:

**Smart India Hackathon 2026 --- SIH26108**

The frontend workflow and system architecture are being developed
alongside the AI, backend, database, and standards-data components.

------------------------------------------------------------------------

## Roadmap

### Phase 1 --- Foundation

-   [x] Define problem and product concept
-   [x] Define procurement workflow
-   [x] Build initial frontend workflow
-   [ ] Finalize API contract
-   [ ] Establish repository architecture

### Phase 2 --- Standards Data

-   [ ] Collect available standards data
-   [ ] Normalize standards metadata
-   [ ] Build searchable standards representation
-   [ ] Model standard relationships
-   [ ] Handle revisions and status information

### Phase 3 --- AI / Retrieval

-   [ ] Requirement extraction
-   [ ] Text preprocessing
-   [ ] Semantic representation
-   [ ] Candidate retrieval
-   [ ] Ranking / reranking
-   [ ] Applicability assessment
-   [ ] Evidence retrieval

### Phase 4 --- Backend

-   [ ] Flask API
-   [ ] Analysis endpoint
-   [ ] Standards search endpoint
-   [ ] Standard details endpoint
-   [ ] Validation and error handling

### Phase 5 --- Integration

-   [ ] Connect frontend to backend
-   [ ] Render recommendations
-   [ ] Render confidence and relevance
-   [ ] Render evidence
-   [ ] Render related standards
-   [ ] Implement low-confidence handling

### Phase 6 --- Evaluation

-   [ ] Build representative procurement queries
-   [ ] Evaluate retrieval quality
-   [ ] Evaluate ranking quality
-   [ ] Test ambiguous specifications
-   [ ] Test incomplete specifications
-   [ ] Test related-standard retrieval
-   [ ] Reduce false recommendations

### Phase 7 --- SIH Demo

-   [ ] End-to-end testing
-   [ ] Demo scenarios
-   [ ] Performance testing
-   [ ] UI refinement
-   [ ] Final documentation
-   [ ] Presentation preparation

------------------------------------------------------------------------

## Team

### SpecCipher --- SIH26108

  Member           Responsibility                           Department
  ---------------- ---------------------------------------- ------------
  **Rajdeep**      AI / ML                                  ECE
  **Saptak**       Backend / Flask                          CSE
  **Daksh**        Backend / Flask                          MnC
  **Siddhartha**   Frontend / HTML, CSS, JS                 CSE
  **Kush**         Frontend / HTML, CSS, JS                 EE
  **Asmita**       SQL backend support + Frontend support   MnC

------------------------------------------------------------------------

## Disclaimer

> **SpecCipher is an AI-assisted decision-support tool.**

Recommendations are **indicative** and should be verified against the
**current official BIS publication** before being used for:

-   Procurement specifications
-   Tender documents
-   Contracts
-   Compliance decisions
-   Certification decisions
-   Other official or regulatory purposes

**SpecCipher is not affiliated with, operated by, or an official product
of the Bureau of Indian Standards (BIS).**

------------------------------------------------------------------------

## License

No open-source license has been selected at this stage.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<strong>`{=html}SpecCipher`</strong>`{=html}`<br>`{=html} AI-assisted
standards intelligence for better procurement decisions.
```{=html}
</p>
```
