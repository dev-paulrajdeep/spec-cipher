# Intelligence layer

Intelligence converts retrieval candidates into explainable, evidence-backed
standard recommendations. It consumes `retrieval.models.RetrievalResult` and
read-only corpus metadata/relationship facts. It does not parse PDFs, generate
embeddings, access FAISS, implement an API, or make unsupported claims.

## Flow

```text
RetrievalResult[] -> binary constraints -> reranking -> applicability fit
                  -> relationship traversal -> chunk evidence -> Recommendation[]
```

Hard constraints run first and retain rejected candidates with explicit
reasons. Statuses such as withdrawn/superseded/cancelled are rejected, as are
explicit requested-standard mismatches and clear corpus scope exclusions.
Unknown metadata and scope never cause a speculative rejection.

Reranking is query-passage relevance, not applicability. The default is a
deterministic token-overlap reranker. A lazy `CrossEncoderReranker` adapter can
be selected through `IntelligenceConfig` and uses a configured Sentence
Transformers CrossEncoder. The final priority score is explicitly a configured
blend of normalized rerank relevance and decomposed applicability; it is not a
confidence score.

## Applicability

`ApplicabilityScore` contains product, material, application, scope, and
requirement matches plus a weighted overall fit. There is deliberately no
duplicated metadata-alignment score: metadata establishes hard admissibility,
while each fit score is grounded in candidate/scope text. The default
interpreter is deterministic. `RequirementInterpreter` is the only boundary
where an optional LLM/Nemotron structured interpreter may later be injected;
no LLM is configured or required.

## Evidence and relationships

`EvidenceLocator` selects only already retrieved chunks and retains standard,
chunk ID, source file, section, page range, text, and score. `CorpusKnowledgeBase`
loads canonical corpus JSON or in-memory `StandardDocument` objects. Its
`RelationshipGraph` is query-only over explicit corpus facts; reverse views
such as `referenced_by` are mechanically derived from explicit facts, never
semantic similarity.

## Usage

```python
from intelligence import IntelligenceEngine

run = IntelligenceEngine(knowledge_base=knowledge_base).recommend(
    "waterproofing compound for concrete reservoir",
    retrieval_results,
)
for recommendation in run.recommendations:
    print(recommendation.standard_number, recommendation.final_score)
```

`IntelligenceRun.candidates` retains rejected and intermediate candidates for
diagnostics. `IntelligenceRun.metrics` records stage durations/counts/errors
without storing document text.
