"""Validate SpecCipher serving prerequisites without starting Flask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.preflight import run_preflight
from retrieval.corpus_config import resolve_retrieval_corpus_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "retrieval-index")
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args()
    run_preflight(args.index, args.corpus or resolve_retrieval_corpus_path())
