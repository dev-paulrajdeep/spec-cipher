# Corpus release operations

SpecCipher treats a release directory as immutable after publication. Staged releases live under `data/staging/<release_id>/`; published releases live under `data/releases/<release_id>/`. `data/releases/CURRENT` contains the release ID and SHA-256 digest of `release.json`, and is replaced atomically after verification. Legacy one-line pointers remain readable during migration; the publisher writes the digest-bearing format.

Use the project virtual environment from the repository root:

```powershell
.\.venv\Scripts\speccipher.exe verify
.\.venv\Scripts\speccipher.exe verify data\releases\<release_id>
.\.venv\Scripts\speccipher.exe publish <release_id>
.\.venv\Scripts\speccipher.exe rollback <previous_release_id>
```

`publish` expects a fully built release in `data/staging/<release_id>/`. It verifies the schema, payload SHA-256 digests, catalog and FTS integrity, chunk provenance, and FAISS dimension/count before atomically moving the directory and updating `CURRENT`. Staging and releases must be on the same filesystem. A release ID cannot be published over an existing release.

`rollback` verifies the selected release and changes only `CURRENT`; it does not modify release contents. Keep older release directories available until rollback is no longer needed.

The repository currently has no complete command that builds all release artifacts from source PDFs. `publish` and `verify` do not ingest PDFs, create embeddings, or build FAISS/SQLite artifacts. The configured release must contain every payload listed in the release manifest, including `QUALITY.json`.

## Building the SQLite catalog

The offline `speccipher.offline.indexing.fts_build.build_catalog_payloads` function and `speccipher catalog` command build `catalog.sqlite` and ordered `chunks.jsonl` from already-extracted records. They do not discover or parse PDFs. Pass document, source-file, and chunk records with explicit IDs and page metadata; records with missing document/source links or inconsistent physical-to-absolute page mapping are rejected. The destinations must not exist.

```python
from speccipher.offline.indexing.fts_build import build_catalog_payloads

build_catalog_payloads(
    "data/staging/<release_id>/catalog.sqlite",
    "data/staging/<release_id>/chunks.jsonl",
    documents=documents,
    source_files=source_files,
    chunks=chunks,
    relationships=relationships,
    scope_texts=scope_texts,
)
```

The CLI accepts a JSON object with `documents`, `source_files`, and `chunks` arrays, plus optional `relationships`, `scope_texts`, `operator_overrides`, and `missing_volumes` arrays:

```powershell
.\.venv\Scripts\speccipher.exe catalog <release_id> catalog-records.json
```

The builder creates the catalog and FTS5 schema consumed by the read-only serving repository, checks SQLite and FTS integrity, and atomically creates each destination. Unresolved document and chunk identities remain SQL `NULL`; the schema also preserves identity status, operator audit evidence, and declared missing-volume page ranges.

The offline projection helper converts already extracted canonical document records without reading PDFs. Its `volume_layouts`, `operator_overrides`, and `missing_volumes` arguments must be explicit evidence keyed by source path or publication identity:

```python
from speccipher.offline.assembly.records import records_from_canonical_documents

catalog_records = records_from_canonical_documents(
    canonical_documents,
    # Every member of a multipart group declares the same logical ID and
    # its absolute page range; matching standard numbers alone are insufficient.
    volume_layouts=volume_layouts,
    operator_overrides=operator_overrides,
    missing_volumes=missing_volumes,
)
```

The returned mapping has the arrays accepted by `speccipher catalog`. Multipart groups require declared page ranges plus one shared `logical_document_id`; no order is inferred from filenames, and declared gaps do not create synthetic chunks. This projection is not yet a complete source-PDF build command: PDF discovery/extraction orchestration, release `QUALITY.json` generation, and end-to-end release creation remain unimplemented.

## Writing the staged manifest

Build the dense payloads first from the staged `chunks.jsonl`. The command uses the configured embedding model and keeps model loading inside the offline CLI process:

```powershell
.\.venv\Scripts\speccipher.exe index <release_id>
```

After all five payloads exist, write the checksummed manifest with the CLI or `speccipher.offline.publishing.manifest.write_release_manifest`. It reads document and chunk counts from the catalog, checks vector count against `chunk_ids.json`, records hashes and byte sizes, and refuses to replace an existing manifest. The release verifier still checks the actual FAISS index dimension, type, and vector count before publication.

```powershell
.\.venv\Scripts\speccipher.exe manifest <release_id> `
  --embedding-model BAAI/bge-m3 `
  --embedding-revision main `
  --chunking-version 2.0 `
  --git-sha <source-commit>
```

```python
from speccipher.offline.publishing.manifest import write_release_manifest

write_release_manifest(
    "data/staging/<release_id>",
    embedding_model="BAAI/bge-m3",
    embedding_revision="main",
    git_sha="<source-commit>",
    chunking_version="2.0",
)
```
