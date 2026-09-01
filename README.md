# SpecCipher

### AI-Powered Indian Standards Recommendation Engine

**Smart India Hackathon 2026 --- SIH26108**

SpecCipher is an AI-assisted decision-support system for identifying
potentially applicable **Indian Standards (IS)** from procurement
requirements.

It helps procurement users move from an unstructured requirement---such
as a product description, technical specification, application,
operating condition, or tender document---to a structured set of
potentially relevant standards, with applicability reasoning, supporting
evidence, and related-standard context.

------------------------------------------------------------------------

## The Problem

Identifying the right Indian Standard for a procurement requirement can
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

The repository now includes a deterministic corpus layer for preparing source
standards for later retrieval work. It preserves source pages, routes sparse
PDF pages to optional OCR, retains raw and cleaned text, extracts only
evidenced metadata and relationships, detects standard clauses, and emits
canonical JSON plus provenance-complete chunk JSONL. See [CORPUS.md](CORPUS.md)
for the schema, OCR adapter boundary, and usage.

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
