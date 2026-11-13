"""Dense online retrieval adapters."""

from .faiss_index import Encoder, FaissVectorIndex

__all__ = ["Encoder", "FaissVectorIndex"]
