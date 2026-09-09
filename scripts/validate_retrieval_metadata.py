"""Report parent-standard identity coverage in a retrieval corpus or index bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.processing.normalizer import normalize_standard_number
from intelligence.standards import canonical_is_code
from retrieval.corpus_loader import CorpusChunkLoader


def _rows(path: Path) -> list[dict[str, object]]:
    bundle_chunks = path.is_dir()
    if path.is_dir():
        path = path / "chunks.json"
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bundle_chunks:
            if not isinstance(payload, list):
                raise ValueError("Retrieval bundle chunks.json must contain a list")
            return payload
        loaded = CorpusChunkLoader().load(path)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "standard_number": chunk.standard_number,
            }
            for chunk in loaded.chunks
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected a chunk JSONL, canonical JSON, or retrieval index directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="canonical corpus, chunks JSONL, or retrieval-index directory")
    args = parser.parse_args()
    rows = _rows(args.path)
    standard_numbers = {normalize_standard_number(str(row.get("standard_number", ""))) for row in rows}
    standard_numbers.discard(None)
    codes = {canonical_is_code(row.get("standard_number")) for row in rows}
    codes.discard(None)
    identifiable = sum(
        normalize_standard_number(str(row.get("standard_number", ""))) is not None
        for row in rows
    )
    print(f"Total chunks: {len(rows)}")
    print(f"Chunks with standard_number: {identifiable}")
    print(f"Chunks without standard_number: {len(rows) - identifiable}")
    print(f"Unique IS standards: {len(codes)}")


if __name__ == "__main__":
    main()