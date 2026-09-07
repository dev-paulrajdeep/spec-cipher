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

The production encoder is the pretrained `BAAI/bge-m3` model through the
Sentence Transformers runtime. BGE-M3 is used only to create semantic
representations. FAISS performs nearest-vector retrieval; reranking,
applicability, recommendations, and explanation remain outside this layer.
The model is loaded lazily, so importing `retrieval` does not allocate GPU
memory or download model weights.

All vectors are normalized before insertion and query. Consequently,
`IndexFlatIP` inner products are cosine similarities: they measure semantic
proximity only, never applicability or confidence. The default BGE-M3
dimension is 1024 when reported by the loaded model; the adapter validates
the configured dimension when one is supplied.

`RetrievalConfig` controls `embedding_model`, `embedding_model_version`,
`embedding_device` (`auto`, `cuda`, or `cpu`), `allow_cpu_fallback`,
`batch_size`, `max_seq_length`, `embedding_dimension`, and normalization.
`auto` selects CUDA when available. CPU fallback is disabled unless
`allow_cpu_fallback=True`; `embedding_device="cpu"` is always explicit.
Batch size defaults to 16, a conservative setting for an 8 GB GPU. Increase
it only after measuring memory headroom.

Optional second-stage reranking is configured in the same `RetrievalConfig`
with `reranker_enabled=True`. It uses `BAAI/bge-reranker-v2-m3` only on the
FAISS-returned candidate pool, with independent model/version, device, CPU
fallback, batch size, sequence length, candidate-pool size, and `use_fp16`
settings. It never searches the corpus or changes the FAISS index.

## Persistence and reproducibility

`RetrievalPipeline.build_and_persist()` writes one directory containing:

- `index.faiss` — exact FAISS `IndexFlatIP` vectors;
- `chunk_ids.json` — explicit FAISS-position-to-chunk-ID mapping;
- `chunks.json` — retrieval chunks with all corpus provenance;
- `metadata.json` — embedding backend, model name and revision, vector
    dimension, normalization strategy, corpus schema/version fingerprint, index
    type, count, and creation time.

The metadata schema is versioned (`1.1`). Reloading checks backend, model and
revision identity, dimension, normalization, mapping alignment, fingerprint,
and vector count. A mismatch, including an older pre-BGE-M3 bundle, fails
clearly instead of querying an incompatible index.

## Usage

Build the generated application bundle from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/build_retrieval_index.py --corpus "corpus\data\SP 7_2026.chunks.jsonl" --output retrieval-index --device cuda
```

For a CPU-only build, explicitly use `--device cpu`. To permit automatic CPU
fallback when CUDA is unavailable, add `--allow-cpu-fallback`. A normal
production build should use CUDA on the RTX host and keep fallback disabled.

The output directory is `retrieval-index/` and contains all four persistence
files. It is generated and gitignored; keep the complete directory together
for Flask startup.

```python
from retrieval import RetrievalService

retriever = RetrievalService.from_persisted("retrieval-index")

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

Run all tests with `.venv\Scripts\python.exe -m pytest`. The default tests
use deterministic fake encoders and do not download BGE-M3. Optional real
model smoke testing requires the model to be available locally or a working
Hugging Face connection:

```powershell
.\.venv\Scripts\python.exe scripts/build_retrieval_index.py --corpus path\to\chunks.jsonl --output retrieval-index --device cuda
.\.venv\Scripts\python.exe -m pytest tests -q
```

The end-to-end contract is: canonical chunks -> BGE-M3 batch embeddings ->
unit-normalized vectors -> FAISS `IndexFlatIP` -> persisted index and explicit
ID mapping -> reload compatibility checks -> semantic query -> optional
BGE-reranker-v2-m3 scoring of those candidates -> ranked `RetrievalResult`
objects containing the original chunk and provenance. Reranked results retain
both `semantic_score` (the original FAISS score) and `reranker_score`.

Limitations: BGE-M3 quality depends on the indexed corpus and text language;
scores are not calibrated probabilities; CPU inference can be slow; model
weights must be downloaded or pre-cached for a real build; and changing the
model revision, dimension, normalization, or backend requires rebuilding the
index. Reranking is runtime configuration and is not persisted into the FAISS
bundle. On an 8 GB GPU, load the two large models sequentially or use
half-precision and conservative sequence/batch settings; both full models may
not fit concurrently.
