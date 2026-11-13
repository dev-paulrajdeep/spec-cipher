# ADR-0007: Release-Backed Decision Policy

- **Status:** Accepted
- **Context:** The legacy decision engine reads canonical JSON files and cannot serve the immutable release as its read-only source of truth.
- **Decision:** The versioned serving path evaluates catalog eligibility, status, explicit scope exclusions, evidence, deterministic token-overlap reranking, and decomposed applicability from the mounted release catalog. Confidence stays categorical and uses reason codes; cosine, BM25, RRF, rerank, and applicability remain distinct.
- **Alternatives rejected:** Loading canonical JSON at request time violates plane and startup boundaries; treating retrieval relevance as applicability or confidence overstates the available evidence.
- **Consequences:** v1 decisions are catalog-backed. Term features and fallback reranking are deterministic; optional BGE reranking loads lazily and fails open. These policies need evaluation against gold queries before production confidence thresholds are considered calibrated.
