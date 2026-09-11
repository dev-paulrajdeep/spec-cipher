"""Single source of truth for the corpus used by serving and indexing."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CORPUS_PATH = PROJECT_ROOT / "corpus" / "data" / "temp_2131689472.json"
