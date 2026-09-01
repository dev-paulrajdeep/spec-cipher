"""FAISS index abstraction and durable, validated persistence."""

from .faiss_index import FaissIndex, IndexSearchHit
from .persistence import IndexCompatibilityError, IndexMetadata, IndexPersistence, PersistedIndex

__all__ = [
    "FaissIndex", "IndexSearchHit", "IndexCompatibilityError", "IndexMetadata",
    "IndexPersistence", "PersistedIndex",
]
