# ADR-0006: Hybrid Retrieval and Score Separation

- **Status:** Accepted
- **Context:** Dense cosine similarity and SQLite FTS5 BM25 expose different score scales, so adding their raw scores is not meaningful.
- **Decision:** Query the release-backed vector and lexical indexes independently, fuse their ranked lists with Reciprocal Rank Fusion (`k=60`), and preserve dense, lexical, RRF, rerank, and applicability values as separate fields.
- **Alternatives rejected:** Raw-score addition depends on incomparable scales; dense-only search loses exact identifier and terminology matches.
- **Consequences:** API consumers can inspect each stage. One configured BGE cross-encoder pass is available after fusion; deterministic token overlap is the fail-open fallback.
