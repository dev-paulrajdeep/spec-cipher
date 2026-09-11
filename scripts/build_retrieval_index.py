"""Build the generated retrieval bundle used by the Flask application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import RetrievalConfig, RetrievalPipeline
from retrieval.corpus_config import CANONICAL_CORPUS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CANONICAL_CORPUS_PATH,
        help="canonical corpus JSON or retrieval-ready JSONL artifact",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "retrieval-index",
        help="directory for the complete generated retrieval bundle",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    config = RetrievalConfig(
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        embedding_device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )
    RetrievalPipeline(config).build_and_persist(args.corpus, args.output)
    print(f"Built retrieval index at {args.output.resolve()}")


if __name__ == "__main__":
    main()