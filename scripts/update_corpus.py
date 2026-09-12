"""Build/update the SpecCipher standards corpus from local source PDFs.

Place authentic, user-supplied technical/standards PDFs under officialBIS/ (or
SPEC_CIPHER_BIS_SOURCE_DIR), then run this command. SpecCipher never downloads,
generates, synthesizes, or reconstructs PDFs.

The pipeline is general-purpose: IS, ISO, IEC, ASTM, and unnamed technical PDFs
are handled by the same discovery → extraction → identity → chunking → JSONL →
FAISS flow. Catalog/list PDFs may be retained as searchable/reference-only
records but are not recommendation-eligible and are excluded from the
recommendation FAISS index.

The historical demo checklist JSON under corpus/data/ is separate and is not
overwritten by this workflow.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\update_corpus.py
  .\\.venv\\Scripts\\python.exe scripts\\update_corpus.py --check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_huggingface_offline() -> None:
    """Prevent Hub resolution during corpus updates unless explicitly enabled."""
    if os.getenv("SPEC_CIPHER_ALLOW_MODEL_DOWNLOAD", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_configure_huggingface_offline()

from corpus.update import update_corpus_from_source
from corpus.update.workflow import default_source_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="PDF source directory (default: officialBIS/ or SPEC_CIPHER_BIS_SOURCE_DIR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Corpus data directory (default: corpus/data)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Inspect PDFs and report what would change without writing corpus/index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild corpus/index even when fingerprints are unchanged",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Write canonical corpus artifacts without rebuilding FAISS",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true", default=None)
    args = parser.parse_args()

    source = args.source.resolve() if args.source else default_source_dir(PROJECT_ROOT)
    print(f"PDF source directory: {source}")
    try:
        summary = update_corpus_from_source(
            project_root=PROJECT_ROOT,
            source_dir=source,
            corpus_data_dir=args.output.resolve() if args.output else None,
            check_only=args.check,
            force=args.force,
            rebuild_index=not args.skip_index and not args.check,
            embedding_device=args.device,
            allow_cpu_fallback=True if args.allow_cpu_fallback else None,
        )
    except Exception as exc:
        print(f"Corpus update failed: {exc}")
        return 1
    summary.print_report()
    if summary.successfully_processed == 0 and summary.failed_documents:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
