"""Idempotent PDF ingestion into canonical corpus + FAISS index.

The workflow is standards-family agnostic: any technical PDF collection can be
placed under the configured source directory. Identity is resolved
conservatively. Catalog/list PDFs may be retained as reference-only documents
but never enter the recommendation FAISS index unless explicitly identity-
resolved as a single publication.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from time import perf_counter
from typing import Any

import pymupdf

from corpus.chunking.chunker import CHUNKING_VERSION
from corpus.models import DocumentChunk, StandardDocument
from corpus.pipeline import CorpusBuilder
from corpus.relationships import RelationshipExtractor
from corpus.serialization import write_chunks_jsonl, write_document_json
from corpus.update.identity import resolve_document_identity
from retrieval import RetrievalConfig, RetrievalPipeline

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 80 * 1024 * 1024
MAX_PDF_PAGES = 400
SCHEMA_VERSION = "1.2"
INGESTION_VERSION = "1.2"
DEFAULT_BIS_SOURCE_DIRNAME = "officialBIS"
ELIGIBLE_CHUNKS_FILENAME = "corpus.eligible.chunks.jsonl"


@dataclass(frozen=True)
class DocumentFailure:
    filename: str
    source_path: str
    error_class: str
    message: str
    remediation: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class ProcessedDocument:
    document_id: str
    standard_number: str | None
    title: str | None
    source_filename: str
    source_path: str
    document_fingerprint: str
    chunk_count: int
    page_count: int
    identity_status: str
    document_type: str
    recommendation_eligible: bool
    document_path: Path
    document: StandardDocument


@dataclass(frozen=True)
class CorpusUpdateSummary:
    source_pdfs_discovered: int
    successfully_processed: int
    standards_indexed: int
    documents_total: int
    documents_eligible: int
    documents_catalog_list: int
    documents_unresolved: int
    chunks_generated: int
    chunks_eligible: int
    vectors_generated: int
    duplicates_skipped: int
    unresolved_identity: int
    rejected_documents: int
    failed_documents: int
    extraction_failures: int
    index_rebuilt: bool
    check_only: bool
    failures: tuple[DocumentFailure, ...] = ()
    processed: tuple[ProcessedDocument, ...] = ()
    would_change: bool = False
    timings_ms: dict[str, float] | None = None

    def print_report(self) -> None:
        mode = "CHECK" if self.check_only else "update complete"
        print(f"Corpus {mode}")
        print(f"  Source PDFs discovered: {self.source_pdfs_discovered}")
        print(f"  Processed: {self.successfully_processed}")
        print(f"  Rejected: {self.rejected_documents}")
        print(f"  Duplicates: {self.duplicates_skipped}")
        print(f"  Unresolved identity: {self.unresolved_identity}")
        print(f"  Catalog/list: {self.documents_catalog_list}")
        print(f"  Extraction failures: {self.extraction_failures}")
        print(f"  Failed documents: {self.failed_documents}")
        print()
        print(f"  Documents total: {self.documents_total}")
        print(f"  Documents eligible: {self.documents_eligible}")
        print(f"  Chunks total: {self.chunks_generated}")
        print(f"  Chunks eligible: {self.chunks_eligible}")
        print(f"  Indexed vectors: {self.vectors_generated}")
        print()
        print(f"  Index rebuilt: {'yes' if self.index_rebuilt else 'no'}")
        if self.check_only:
            print(f"  Would change corpus/index: {'yes' if self.would_change else 'no'}")
        if self.timings_ms:
            print("  Timings (ms):")
            for key, value in self.timings_ms.items():
                print(f"    {key}: {value:.1f}")
        for failure in self.failures:
            page = f", page {failure.page}" if failure.page is not None else ""
            print(
                f"  FAIL {failure.filename}{page} [{failure.error_class}]: {failure.message}"
            )
            if failure.remediation:
                print(f"       Remediation: {failure.remediation}")


def default_source_dir(project_root: Path) -> Path:
    configured = os.getenv("SPEC_CIPHER_BIS_SOURCE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root / DEFAULT_BIS_SOURCE_DIRNAME).resolve()


def discover_pdfs(source_dir: Path) -> tuple[Path, ...]:
    """Discover PDF files recursively; ignore unrelated files and reject escapes."""
    if not source_dir.exists():
        return ()
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    root = source_dir.resolve()
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Refusing path outside source directory: {path}") from exc
        found.append(resolved)
    return tuple(found)


def document_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_source_path(path: Path, source_dir: Path) -> str:
    return path.resolve().relative_to(source_dir.resolve()).as_posix()


def safe_document_stem(document_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", document_id).strip("_") or "document"


def update_corpus_from_source(
    *,
    project_root: Path,
    source_dir: Path | None = None,
    corpus_data_dir: Path | None = None,
    index_dir: Path | None = None,
    check_only: bool = False,
    force: bool = False,
    rebuild_index: bool = True,
    embedding_device: str | None = None,
    allow_cpu_fallback: bool | None = None,
) -> CorpusUpdateSummary:
    """Ingest source PDFs into canonical corpus artifacts and rebuild FAISS."""
    timings: dict[str, float] = {}
    total_started = perf_counter()
    root = project_root.resolve()
    source = (source_dir or default_source_dir(root)).resolve()
    data_dir = (corpus_data_dir or (root / "corpus" / "data")).resolve()
    documents_dir = data_dir / "documents"
    chunks_path = data_dir / "corpus.chunks.jsonl"
    eligible_chunks_path = data_dir / ELIGIBLE_CHUNKS_FILENAME
    manifest_path = data_dir / "manifest.json"
    index_path = (index_dir or (root / "retrieval-index")).resolve()

    discovery_started = perf_counter()
    pdfs = discover_pdfs(source)
    timings["discovery"] = (perf_counter() - discovery_started) * 1000

    failures: list[DocumentFailure] = []
    processed: list[ProcessedDocument] = []
    fingerprints_seen: dict[str, str] = {}
    duplicates_skipped = 0
    unresolved_identity = 0
    rejected_documents = 0
    extraction_failures = 0
    builder = CorpusBuilder(raw_asset_directory=data_dir / "_assets")

    extract_started = perf_counter()
    for pdf_path in pdfs:
        rel = relative_source_path(pdf_path, source)
        try:
            item = _process_one_pdf(
                pdf_path=pdf_path,
                relative_path=rel,
                builder=builder,
                asset_root=data_dir / "_assets",
                fingerprints_seen=fingerprints_seen,
            )
            processed.append(item)
            if item.identity_status != "resolved":
                unresolved_identity += 1
        except _SkipDuplicate as skip:
            duplicates_skipped += 1
            logger.info(
                "Skipping duplicate PDF fingerprint=%s path=%s",
                skip.fingerprint[:12],
                rel,
            )
        except _DocumentProcessingError as exc:
            if exc.error_class in {"unresolved_identity", "ambiguous_identity", "identity_conflict"}:
                unresolved_identity += 1
                rejected_documents += 1
            else:
                extraction_failures += 1
            failures.append(
                DocumentFailure(
                    filename=pdf_path.name,
                    source_path=rel,
                    error_class=exc.error_class,
                    message=exc.message,
                    remediation=exc.remediation,
                    page=exc.page,
                )
            )
    timings["extraction_chunking"] = (perf_counter() - extract_started) * 1000

    # One canonical document per non-null document number; fingerprint keys unresolved docs.
    processed = _dedupe_documents(processed, failures)

    existing_manifest = _read_manifest(manifest_path)
    planned_docs = _manifest_documents(processed)
    planned_fingerprint = _manifest_fingerprint(planned_docs)
    corpus_changed = (
        force
        or planned_fingerprint != existing_manifest.get("corpus_fingerprint")
        or not chunks_path.is_file()
        or not eligible_chunks_path.is_file()
        or not manifest_path.is_file()
    )
    index_missing = rebuild_index and not (index_path / "metadata.json").is_file()
    would_change = corpus_changed or index_missing

    catalog_list_count = sum(
        1 for item in processed if item.document_type in {"catalog", "list", "index"}
    )
    eligible_docs = [item for item in processed if item.recommendation_eligible]
    eligible_chunk_total = sum(item.chunk_count for item in eligible_docs)
    chunk_total = sum(item.chunk_count for item in processed)

    if check_only:
        timings["total"] = (perf_counter() - total_started) * 1000
        return CorpusUpdateSummary(
            source_pdfs_discovered=len(pdfs),
            successfully_processed=len(processed),
            standards_indexed=len(eligible_docs),
            documents_total=len(processed),
            documents_eligible=len(eligible_docs),
            documents_catalog_list=catalog_list_count,
            documents_unresolved=sum(1 for item in processed if item.identity_status != "resolved"),
            chunks_generated=chunk_total,
            chunks_eligible=eligible_chunk_total,
            vectors_generated=0,
            duplicates_skipped=duplicates_skipped,
            unresolved_identity=unresolved_identity,
            rejected_documents=rejected_documents,
            failed_documents=len(failures),
            extraction_failures=extraction_failures,
            index_rebuilt=False,
            check_only=True,
            failures=tuple(failures),
            processed=tuple(processed),
            would_change=would_change,
            timings_ms=timings,
        )

    if not processed:
        _clear_generated_corpus_artifacts(
            documents_dir, chunks_path, eligible_chunks_path, manifest_path, index_path if rebuild_index else None
        )
        raise RuntimeError(
            "No authentic PDFs produced a usable corpus. Place technical/standards "
            "PDFs under the configured source directory with clear document identity "
            "where possible. SpecCipher never generates or synthesizes source PDFs. "
            "Generated source-derived corpus artifacts were cleared; the historical "
            "demo checklist JSON was left unchanged."
        )

    chunk_started = perf_counter()
    if documents_dir.exists():
        shutil.rmtree(documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    written: list[ProcessedDocument] = []
    all_chunks: list[DocumentChunk] = []
    for item in sorted(
        processed,
        key=lambda row: (
            0 if row.recommendation_eligible else 1,
            row.standard_number or "",
            row.document_id,
            row.source_path,
        ),
    ):
        target = documents_dir / f"{safe_document_stem(item.document_id)}.json"
        write_document_json(item.document, target)
        all_chunks.extend(item.document.chunks)
        written.append(replace(item, document_path=target))

    all_chunks_sorted = tuple(
        sorted(
            all_chunks,
            key=lambda chunk: (
                0 if chunk.recommendation_eligible else 1,
                chunk.standard_number or "",
                chunk.document_id or "",
                chunk.chunk_id,
                chunk.source_file,
            ),
        )
    )
    eligible_chunks_sorted = tuple(
        chunk for chunk in all_chunks_sorted if chunk.recommendation_eligible
    )
    if not all_chunks_sorted:
        raise RuntimeError(
            "Corpus update produced zero retrieval chunks. Refusing to write an empty ready corpus."
        )
    write_chunks_jsonl(all_chunks_sorted, chunks_path)
    write_chunks_jsonl(eligible_chunks_sorted, eligible_chunks_path)
    timings["write_artifacts"] = (perf_counter() - chunk_started) * 1000

    source_identity = _source_dir_identity(source, root)
    eligible_written = [item for item in written if item.recommendation_eligible]
    catalog_list_written = [
        item for item in written if item.document_type in {"catalog", "list", "index"}
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "ingestion_version": INGESTION_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "source_dir": source_identity,
        "source_policy": "user_supplied_pdfs_only",
        "documents": _manifest_documents(written),
        "document_count": len(written),
        "document_count_eligible": len(eligible_written),
        "document_count_catalog_list": len(catalog_list_written),
        "chunk_count": len(all_chunks_sorted),
        "chunk_count_eligible": len(eligible_chunks_sorted),
        "chunks_fingerprint": _chunks_fingerprint(all_chunks_sorted),
        "eligible_chunks_fingerprint": _chunks_fingerprint(eligible_chunks_sorted),
        "corpus_fingerprint": _manifest_fingerprint(_manifest_documents(written)),
        "retrieval_corpus": "corpus.chunks.jsonl",
        "recommendation_corpus": ELIGIBLE_CHUNKS_FILENAME,
        "unresolved_documents": [
            {
                "document_id": item.document_id,
                "source_path": item.source_path,
                "title": item.title,
                "document_type": item.document_type,
                "recommendation_eligible": item.recommendation_eligible,
            }
            for item in written
            if item.identity_status != "resolved"
        ],
        "failed_documents": [
            {
                "filename": item.filename,
                "source_path": item.source_path,
                "error_class": item.error_class,
                "message": item.message,
                "remediation": item.remediation,
            }
            for item in failures
        ],
        "historical_checklist_untouched": True,
    }

    index_rebuilt = False
    vectors_generated = 0
    embedding_meta: dict[str, Any] = {}
    if rebuild_index:
        embed_started = perf_counter()
        device = (embedding_device or os.getenv("SPEC_CIPHER_EMBEDDING_DEVICE", "auto")).strip().lower()
        if allow_cpu_fallback is None:
            cpu_fallback = os.getenv("SPEC_CIPHER_ALLOW_CPU_FALLBACK", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            cpu_fallback = allow_cpu_fallback
        config = RetrievalConfig(embedding_device=device, allow_cpu_fallback=cpu_fallback)
        RetrievalPipeline(config).build_and_persist(eligible_chunks_path, index_path)
        vectors_generated = len(eligible_chunks_sorted)
        index_rebuilt = True
        timings["embedding_faiss"] = (perf_counter() - embed_started) * 1000
        index_meta = _read_manifest(index_path / "metadata.json")
        embedding_meta = {
            "embedding_model": index_meta.get("embedding_model"),
            "embedding_dimension": index_meta.get("embedding_dimension"),
            "index_fingerprint": index_meta.get("corpus_fingerprint"),
            "indexed_chunk_count": index_meta.get("chunk_count"),
        }
        if vectors_generated != index_meta.get("chunk_count"):
            raise RuntimeError("FAISS vector count does not match eligible chunk count.")
        logger.info("Rebuilt retrieval index duration_ms=%.2f", timings["embedding_faiss"])

    manifest.update(embedding_meta)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Intentionally do not overwrite IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json.
    # That file is historical/demo data and is not a source-derived artifact.

    timings["total"] = (perf_counter() - total_started) * 1000
    return CorpusUpdateSummary(
        source_pdfs_discovered=len(pdfs),
        successfully_processed=len(written),
        standards_indexed=len(eligible_written),
        documents_total=len(written),
        documents_eligible=len(eligible_written),
        documents_catalog_list=len(catalog_list_written),
        documents_unresolved=sum(1 for item in written if item.identity_status != "resolved"),
        chunks_generated=len(all_chunks_sorted),
        chunks_eligible=len(eligible_chunks_sorted),
        vectors_generated=vectors_generated,
        duplicates_skipped=duplicates_skipped,
        unresolved_identity=unresolved_identity,
        rejected_documents=rejected_documents,
        failed_documents=len(failures),
        extraction_failures=extraction_failures,
        index_rebuilt=index_rebuilt,
        check_only=False,
        failures=tuple(failures),
        processed=tuple(written),
        would_change=True,
        timings_ms=timings,
    )


class _DocumentProcessingError(Exception):
    def __init__(
        self,
        error_class: str,
        message: str,
        page: int | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.message = message
        self.page = page
        self.remediation = remediation


class _SkipDuplicate(Exception):
    def __init__(self, fingerprint: str) -> None:
        super().__init__(fingerprint)
        self.fingerprint = fingerprint


def _process_one_pdf(
    *,
    pdf_path: Path,
    relative_path: str,
    builder: CorpusBuilder,
    asset_root: Path,
    fingerprints_seen: dict[str, str],
) -> ProcessedDocument:
    try:
        size = pdf_path.stat().st_size
    except OSError as exc:
        raise _DocumentProcessingError(
            "io_error",
            f"Could not stat PDF: {exc}",
            remediation="Verify file permissions and that the path is readable.",
        ) from exc
    if size <= 0:
        raise _DocumentProcessingError(
            "empty_pdf",
            "PDF file is empty.",
            remediation="Replace the empty file with a valid source PDF.",
        )
    if size > MAX_PDF_BYTES:
        raise _DocumentProcessingError(
            "pdf_too_large",
            f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MiB ingestion limit.",
            remediation="Split or replace oversized PDFs before ingestion.",
        )
    try:
        data = pdf_path.read_bytes()
    except OSError as exc:
        raise _DocumentProcessingError(
            "io_error",
            f"Could not read PDF: {exc}",
            remediation="Verify file permissions and that the path is readable.",
        ) from exc

    fingerprint = document_fingerprint(data)
    if fingerprint in fingerprints_seen:
        raise _SkipDuplicate(fingerprint)
    fingerprints_seen[fingerprint] = relative_path

    leading_text, page_count, pdf_title, image_only = _inspect_pdf(data)
    if page_count > MAX_PDF_PAGES:
        raise _DocumentProcessingError(
            "too_many_pages",
            f"PDF has {page_count} pages; limit is {MAX_PDF_PAGES}.",
            remediation="Provide a smaller publication or split volumes.",
        )
    if image_only:
        raise _DocumentProcessingError(
            "image_only_pdf",
            "PDF appears image-only / scanned; text extraction produced no usable text.",
            remediation="OCR is not enabled in this pipeline. Provide a text-extractable PDF or add an OCR provider.",
        )

    identity = resolve_document_identity(
        source_filename=pdf_path.name,
        leading_text=leading_text,
        pdf_title=pdf_title,
        document_fingerprint=fingerprint,
    )
    if not identity.ingestible:
        error_class = {
            "conflict": "identity_conflict",
            "catalog": "unresolved_identity",
            "ambiguous": "ambiguous_identity",
        }.get(identity.source, "unresolved_identity")
        raise _DocumentProcessingError(
            error_class,
            identity.reason,
            remediation=(
                "Use one publication per PDF and a clear filename identity "
                "(for example STD-1001.pdf or ISO_9001.pdf)."
            ),
        )

    document_id = identity.document_id or f"doc_{fingerprint[:16]}"
    asset_dir = asset_root / fingerprint[:16]
    try:
        document = builder.build_bytes(
            data,
            pdf_path.name,
            "pdf",
            raw_asset_directory=asset_dir,
        )
    except Exception as exc:
        raise _DocumentProcessingError(
            exc.__class__.__name__,
            f"Extraction/build failed: {exc}",
            remediation="Inspect the PDF for encryption, corruption, or unsupported structure.",
        ) from exc

    title = identity.title or document.metadata.title
    metadata = replace(
        document.metadata,
        standard_number=identity.document_number,
        title=title,
        source_file=pdf_path.name,
        source_path=relative_path,
        document_fingerprint=fingerprint,
        document_id=document_id,
        document_type=identity.document_type,
        identity_status=identity.status,
        recommendation_eligible=identity.recommendation_eligible,
    )
    # Re-chunk with authoritative document identity so IDs never depend on
    # synthetic/filename-invented standard numbers.
    chunks = builder.chunker.chunk(
        document.sections,
        identity.document_number,
        pdf_path.name,
        document_id=document_id,
        document_fingerprint=fingerprint,
    )
    chunks = tuple(
        replace(
            chunk,
            standard_number=identity.document_number,
            source_file=pdf_path.name,
            source_path=relative_path,
            document_fingerprint=fingerprint,
            document_id=document_id,
            title=title,
            chunking_version=chunk.chunking_version or CHUNKING_VERSION,
            document_type=identity.document_type,
            recommendation_eligible=identity.recommendation_eligible,
            identity_status=identity.status,
        )
        for chunk in chunks
    )
    relationships = RelationshipExtractor().extract(
        document.pages, identity.document_number, document.sections
    )
    pages = []
    for page in document.pages:
        provenance = page.provenance
        if provenance is not None:
            provenance = replace(
                provenance,
                standard_id=identity.document_number,
                source_file=pdf_path.name,
                chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in chunks
                    if chunk.page_start <= page.page_number <= chunk.page_end
                ),
            )
        pages.append(
            replace(page, standard_id=identity.document_number, provenance=provenance)
        )
    document = replace(
        document,
        metadata=metadata,
        chunks=chunks,
        pages=tuple(pages),
        relationships=relationships,
    )
    if not document.chunks:
        raise _DocumentProcessingError(
            "no_chunks",
            "Extraction produced no retrieval chunks; check whether the PDF has extractable text.",
            remediation="Provide a text-extractable PDF. Image-only PDFs require OCR, which is not enabled.",
        )

    return ProcessedDocument(
        document_id=document_id,
        standard_number=identity.document_number,
        title=title,
        source_filename=pdf_path.name,
        source_path=relative_path,
        document_fingerprint=fingerprint,
        chunk_count=len(document.chunks),
        page_count=document.metadata.page_count,
        identity_status=identity.status,
        document_type=identity.document_type,
        recommendation_eligible=identity.recommendation_eligible,
        document_path=Path(""),
        document=document,
    )


def _inspect_pdf(data: bytes) -> tuple[str, int, str | None, bool]:
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise _DocumentProcessingError(
                "encrypted_pdf",
                "PDF is encrypted or password-protected.",
                remediation="Provide an unencrypted source PDF.",
            ) from exc
        raise _DocumentProcessingError(
            "pdf_open_error",
            f"Could not open PDF: {exc}",
            remediation="Replace the malformed PDF with a valid source file.",
        ) from exc
    try:
        if getattr(pdf, "needs_pass", False):
            raise _DocumentProcessingError(
                "encrypted_pdf",
                "PDF is encrypted or password-protected.",
                remediation="Provide an unencrypted source PDF.",
            )
        page_count = pdf.page_count
        if page_count < 1:
            raise _DocumentProcessingError(
                "empty_pdf",
                "PDF contains no pages.",
                remediation="Replace the empty file with a valid source PDF.",
            )
        texts: list[str] = []
        extractable = False
        for index in range(page_count):
            try:
                page_text = pdf[index].get_text() or ""
            except Exception as exc:
                raise _DocumentProcessingError(
                    "page_extract_error",
                    f"Text extraction failed: {exc}",
                    page=index + 1,
                    remediation="Inspect the page for corruption or provide a cleaner export.",
                ) from exc
            if page_text.strip():
                extractable = True
            if index < 3:
                texts.append(page_text)
        leading = "\n".join(texts)
        image_only = not extractable
        meta = pdf.metadata or {}
        title = meta.get("title") if isinstance(meta, dict) else None
        if isinstance(title, str):
            title = title.strip() or None
        else:
            title = None
        return leading, page_count, title, image_only
    finally:
        pdf.close()


def _dedupe_documents(
    processed: list[ProcessedDocument],
    failures: list[DocumentFailure],
) -> list[ProcessedDocument]:
    chosen: dict[str, ProcessedDocument] = {}
    for item in sorted(
        processed,
        key=lambda row: (
            0 if row.recommendation_eligible else 1,
            row.standard_number or "",
            row.document_id,
            row.source_path,
        ),
    ):
        key = item.standard_number or f"fingerprint:{item.document_fingerprint}"
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = item
            continue
        if existing.document_fingerprint == item.document_fingerprint:
            continue
        if item.standard_number:
            failures.append(
                DocumentFailure(
                    filename=item.source_filename,
                    source_path=item.source_path,
                    error_class="document_number_conflict",
                    message=(
                        f"Document number {item.standard_number} already resolved from "
                        f"{existing.source_path} with a different document fingerprint. "
                        "Keep one authoritative PDF per document number."
                    ),
                    remediation="Remove or rename the conflicting PDF; do not silently merge distinct sources.",
                )
            )
            continue
        # Distinct unresolved documents keyed by fingerprint remain independent.
        chosen[f"fingerprint:{item.document_fingerprint}"] = item
    return list(chosen.values())


def _manifest_documents(
    processed: list[ProcessedDocument] | tuple[ProcessedDocument, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "document_id": item.document_id,
            "standard_number": item.standard_number,
            "title": item.title,
            "source_filename": item.source_filename,
            "source_path": item.source_path,
            "document_fingerprint": item.document_fingerprint,
            "chunk_count": item.chunk_count,
            "page_count": item.page_count,
            "identity_status": item.identity_status,
            "document_type": item.document_type,
            "recommendation_eligible": item.recommendation_eligible,
            "document_path": f"documents/{safe_document_stem(item.document_id)}.json",
        }
        for item in sorted(
            processed,
            key=lambda row: (
                0 if row.recommendation_eligible else 1,
                row.standard_number or "",
                row.document_id,
                row.source_path,
            ),
        )
    ]


def _manifest_fingerprint(documents: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _chunks_fingerprint(chunks: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": getattr(chunk, "document_id", None),
            "standard_number": chunk.standard_number,
            "source_file": chunk.source_file,
            "source_path": getattr(chunk, "source_path", None),
            "document_fingerprint": getattr(chunk, "document_fingerprint", None),
            "document_type": getattr(chunk, "document_type", None),
            "recommendation_eligible": getattr(chunk, "recommendation_eligible", None),
            "identity_status": getattr(chunk, "identity_status", None),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_number": chunk.section_number,
            "section_title": chunk.section_title,
            "title": getattr(chunk, "title", None),
            "chunking_version": getattr(chunk, "chunking_version", None),
            "text": chunk.text,
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _source_dir_identity(source: Path, project_root: Path) -> str:
    try:
        return source.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        # Outside the project: keep only the directory name, never a developer home path.
        return source.name


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _clear_generated_corpus_artifacts(
    documents_dir: Path,
    chunks_path: Path,
    eligible_chunks_path: Path,
    manifest_path: Path,
    index_dir: Path | None = None,
) -> None:
    """Remove source-derived generated artifacts without touching historical demo JSON."""
    if documents_dir.exists():
        shutil.rmtree(documents_dir)
    for path in (chunks_path, eligible_chunks_path, manifest_path):
        if path.is_file():
            path.unlink()
    if index_dir is not None and index_dir.exists():
        for name in ("index.faiss", "chunk_ids.json", "chunks.json", "metadata.json"):
            candidate = index_dir / name
            if candidate.is_file():
                candidate.unlink()
