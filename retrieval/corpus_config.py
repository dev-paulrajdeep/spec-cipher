"""Single source of truth for corpus artifacts used by serving and indexing."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DATA_DIR = PROJECT_ROOT / "corpus" / "data"
CORPUS_DOCUMENTS_DIR = CORPUS_DATA_DIR / "documents"
CORPUS_MANIFEST_PATH = CORPUS_DATA_DIR / "manifest.json"
# Produced only by scripts/update_corpus.py from user-supplied officialBIS PDFs.
CANONICAL_CORPUS_PATH = CORPUS_DATA_DIR / "corpus.chunks.jsonl"
# Historical/demo checklist JSON. Not an official BIS PDF extraction and must
# not be overwritten by the officialBIS update workflow.
LEGACY_CANONICAL_CORPUS_PATH = (
    CORPUS_DATA_DIR / "IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json"
)
DEFAULT_BIS_SOURCE_DIR = "officialBIS"
RETRIEVAL_INDEX_DIR = PROJECT_ROOT / "retrieval-index"


def resolve_knowledge_base_paths() -> tuple[Path, ...]:
    """Return canonical document JSON paths for intelligence metadata/relationships.

    Prefer authentic officialBIS-derived documents when present; otherwise fall
    back to the historical demo checklist JSON.
    """
    if CORPUS_MANIFEST_PATH.is_file():
        try:
            import json

            payload = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
            documents = payload.get("documents") if isinstance(payload, dict) else None
            if isinstance(documents, list) and documents:
                paths: list[Path] = []
                for row in documents:
                    if not isinstance(row, dict):
                        continue
                    relative = row.get("document_path")
                    if isinstance(relative, str) and relative.strip():
                        candidate = CORPUS_DATA_DIR / relative
                        if candidate.is_file():
                            paths.append(candidate)
                if paths:
                    return tuple(paths)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if CORPUS_DOCUMENTS_DIR.is_dir():
        documents = tuple(sorted(CORPUS_DOCUMENTS_DIR.glob("*.json")))
        if documents:
            return documents
    if LEGACY_CANONICAL_CORPUS_PATH.is_file():
        return (LEGACY_CANONICAL_CORPUS_PATH,)
    return ()


def resolve_retrieval_corpus_path() -> Path:
    """Prefer officialBIS-derived chunks; otherwise the historical demo checklist."""
    if CANONICAL_CORPUS_PATH.is_file():
        return CANONICAL_CORPUS_PATH
    if LEGACY_CANONICAL_CORPUS_PATH.is_file():
        return LEGACY_CANONICAL_CORPUS_PATH
    return CANONICAL_CORPUS_PATH


def serving_corpus_origin() -> str:
    """Describe which corpus artifact runtime serving will use."""
    if CANONICAL_CORPUS_PATH.is_file():
        return "officialbis_derived"
    if LEGACY_CANONICAL_CORPUS_PATH.is_file():
        return "historical_demo_checklist"
    return "missing"
