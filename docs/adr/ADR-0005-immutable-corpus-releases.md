# ADR-0005: Immutable Corpus Releases

- **Context:** Serving needs a verifiable snapshot and a rollback path without rebuilding artifacts during requests.
- **Decision:** Verify staged payload digests, catalog/chunk/vector counts, provenance, FTS integrity, and FAISS metadata before promotion; select a release through an atomically replaced `CURRENT` file.
- **Alternatives rejected:** Mutating a live corpus in place risks mixed snapshots; copying files during promotion is not atomic; silently falling back after a corrupt configured release hides deployment errors.
- **Consequences:** A promoted release ID is immutable, rollback changes only `CURRENT`, and serving opens the catalog read-only. Publication requires staging and releases on one filesystem.
