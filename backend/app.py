"""SpecCipher serving entry point and backward-compatibility facade."""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speccipher.api.app import _highlight_terms, _serialize_recommendations, create_app  # noqa: E402
from speccipher.config.settings import Settings  # noqa: E402
from speccipher.services.container import ApplicationContainer  # noqa: E402

__all__ = [
    "app",
    "create_app",
    "container",
    "settings",
    "FRONTEND_DIR",
    "RETRIEVAL_INDEX_DIR",
    "MAX_INPUT_CHARACTERS",
    "FINAL_TOP_K",
    "RETRIEVAL_CANDIDATE_K",
    "_serialize_recommendations",
    "_highlight_terms",
]

logger = logging.getLogger("speccipher")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("SPEC_CIPHER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# Instantiate container and app factory with ZERO heavy imports
settings = Settings.from_env()
container = ApplicationContainer(settings=settings)
app = create_app(container)

FRONTEND_DIR = settings.paths.frontend_dir
RETRIEVAL_INDEX_DIR = settings.paths.release_path
MAX_INPUT_CHARACTERS = settings.server.max_input_characters
FINAL_TOP_K = settings.retrieval.final_top_k
RETRIEVAL_CANDIDATE_K = settings.retrieval.top_k

atexit.register(container.close)


def __getattr__(name: str) -> Any:
    """Dynamically resolve legacy module variables without triggering eager model loads."""
    if name == "retriever":
        return container.retrieval_service
    if name == "intelligence":
        return container.intelligence_engine
    if name == "qwen_reasoner":
        return container.explanation_service
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    app.run(
        host=settings.server.host,
        port=settings.server.port,
        debug=False,
        use_reloader=False,
    )
