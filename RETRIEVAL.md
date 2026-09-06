# Retrieval layer

The retrieval layer turns the corpus layer's retrieval-ready chunks into
semantic candidate chunks. It deliberately does **not** parse PDFs, judge
applicability, rerank, recommend standards, provide evidence reasoning, or
serve an API.

## Corpus contract

`CorpusChunkLoader` reads either the canonical corpus JSON object (its
`chunks` array) or the JSONL emitted by `write_chunks_jsonl`. Every input record
must contain the corpus `DocumentChunk` fields: `chunk_id`, `standard_number`,
`source_file`, `page_start`, `page_end`, `section_number`, `section_title`, and
`text`. It rejects malformed records, invalid ranges, empty provenance, and
duplicate IDs before any embedding is computed. `standard_number` may be
`null` when the corpus has no evidenced identifier (for example, a Special
Publication); retrieval preserves that uncertainty rather than inventing one.

## Flow

```text
corpus JSON / JSONL -> validated DocumentChunk records -> embeddings
                   -> normalized IndexFlatIP -> RetrievalResult[]
```

The default encoder is the pretrained
`sentence-transformers/all-MiniLM-L6-v2` model. It batches calls through
Sentence Transformers and normalizes all vectors. Therefore `IndexFlatIP`
scores are cosine similarities: they measure semantic proximity only, never
applicability or confidence.

`RetrievalConfig.batch_size` defaults to 16, a conservative setting verified
against the repository's large SP 7 chunks. Deployments with measured memory
headroom may set a larger value explicitly when building an index.

## Persistence and reproducibility

`RetrievalPipeline.build_and_persist()` writes one directory containing:

- `index.faiss` — exact FAISS `IndexFlatIP` vectors;
- `chunk_ids.json` — explicit FAISS-position-to-chunk-ID mapping;
- `chunks.json` — retrieval chunks with all corpus provenance;
- `metadata.json` — model name, vector dimension, normalization strategy,
  corpus schema/version fingerprint, index type, count, and creation time.

Reloading checks model identity, dimension, normalization, mapping alignment,
fingerprint, and vector count. A mismatch fails instead of querying an
incompatible index.

## Usage

```python
from retrieval import RetrievalPipeline

pipeline = RetrievalPipeline()
retriever = pipeline.build_and_persist(
    "output/IS-2645.chunks.jsonl",
    "output/retrieval-index",
)

for result in retriever.search("waterproofing compound for concrete", top_k=5):
    print(result.rank, result.similarity_score, result.chunk.standard_number)
```

The result includes the original `DocumentChunk`, so source file, standard,
section, page range, and chunk ID remain available to `intelligence/`.
`search_candidates()` offers a non-destructive grouping by standard; every
returned chunk remains in each candidate's `results` collection.

## Future hybrid boundary

`retrieval.search.Retriever` is the common candidate interface.
`retrieval.hybrid.LexicalRetriever` and `HybridFusion` are protocols for a
future lexical implementation and explicit fusion policy. There is no lexical
engine or reranker in this layer.

## Evaluation

`retrieval.evaluation` provides Recall@K, Precision@K, and MRR@K. The small
hand-curated `DEMO_GOLD_CASES` measures whether expected standards appear in
the candidate set; it is a smoke benchmark, not a quality claim for a complete
standards corpus.

Run all tests with `python -m pytest`.
