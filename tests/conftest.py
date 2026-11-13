from __future__ import annotations

import sys
from pathlib import Path

# Pre-load faiss on Windows before other C-extensions to avoid OpenMP handle conflict
try:
    import faiss  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
