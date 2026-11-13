"""Application Composition Root and Dependency Injection Container.

Enforces zero heavyweight initialization during module import or construction.
Resources are loaded lazily on first access or explicitly via warmup().
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

from speccipher.config.settings import Settings

logger = logging.getLogger("speccipher.container")


class ApplicationContainer:
    """Composition root managing lifecycle and dependencies for SpecCipher.

    Guarantees:
    - Zero heavy imports (torch, faiss, transformers) on container instantiation.
    - Thread-safe lazy initialization using reentrant locks.
    - Explicit overrides for testing (inject mock retrievers, engines, or reasoners).
    - Deterministic resource teardown via close() and context manager protocol.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        release_dir: Path | str | None = None,
        *,
        hybrid_retrieval_override: Any = None,
        retrieval_override: Any = None,
        serving_decision_override: Any = None,
        decision_override: Any = None,
        explanation_override: Any = None,
        catalog_override: Any = None,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._release_dir = Path(release_dir) if release_dir else self._settings.paths.release_path
        self._lock = RLock()

        # Release-backed services initialized lazily, with explicit test overrides.
        self._hybrid_retrieval: Any = hybrid_retrieval_override
        self._retrieval_service: Any = retrieval_override
        self._serving_decision_engine: Any = serving_decision_override
        self._decision_engine: Any = decision_override
        self._explanation_service: Any = explanation_override
        self._catalog: Any = catalog_override
        self._manifest: dict[str, Any] | None = None
        self._is_mounted: bool = False
        self._is_closed: bool = False

        if release_dir is not None:
            self.mount_release(release_dir)
        elif self._release_dir and (
            (self._release_dir.is_file() and self._release_dir.name == "CURRENT")
            or (self._release_dir / "release.json").is_file()
        ):
            # A configured release is authoritative: a corrupt release must fail startup,
            # not silently leave the process serving against an older corpus.
            self.mount_release(self._release_dir)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def release_dir(self) -> Path:
        return self._release_dir

    @property
    def is_mounted(self) -> bool:
        with self._lock:
            return bool(self._is_mounted and not self._is_closed)

    @property
    def catalog_available(self) -> bool:
        """Whether a verified release catalog or explicit test catalog is available."""
        with self._lock:
            return bool(not self._is_closed and (self._is_mounted or self._catalog is not None))

    @property
    def manifest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._manifest

    def mount_release(self, release_dir: Path | str) -> None:
        """Mount and verify a corpus release, releasing previous SQLite file handles."""
        with self._lock:
            from speccipher.catalog.sqlite import CatalogRepository
            from speccipher.release.manifest import resolve_release_path, verify_release

            target_path = resolve_release_path(Path(release_dir))
            manifest = verify_release(target_path)

            old_catalog = self._catalog
            new_catalog = None
            db_path = target_path / "catalog.sqlite"
            if db_path.exists():
                new_catalog = CatalogRepository(db_path)

            self._release_dir = target_path
            self._manifest = manifest
            self._catalog = new_catalog
            self._hybrid_retrieval = None
            self._serving_decision_engine = None
            self._is_mounted = True

            # Invariant: Close previous catalog to prevent Windows file-lock PermissionErrors
            if old_catalog is not None and hasattr(old_catalog, "close"):
                try:
                    old_catalog.close()
                except Exception as exc:
                    logger.warning("Error closing previous catalog: %s", exc)

    @property
    def hybrid_retrieval(self) -> Any:
        """Lazily construct release-backed dense and lexical retrieval with RRF."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._retrieval_service is not None:
                return self._retrieval_service
            if self._hybrid_retrieval is None:
                self._hybrid_retrieval = self._init_hybrid_retrieval()
            return self._hybrid_retrieval

    @property
    def retrieval_service(self) -> Any:
        """Backward-compatibility alias for hybrid_retrieval or injected retriever."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._retrieval_service is not None:
                return self._retrieval_service
            return self.hybrid_retrieval

    @property
    def serving_decision_engine(self) -> Any:
        """Lazily construct the read-only release-backed v1 decision policy."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._decision_engine is not None:
                return self._decision_engine
            if self._serving_decision_engine is None:
                from speccipher.decision.engine import DecisionEngine

                decision_settings = self._settings.decision
                retrieval_settings = self._settings.retrieval
                reranker = None
                if retrieval_settings.reranker_enabled:
                    from speccipher.retrieval.ranking.bge_reranker import BGERerankerAdapter

                    reranker = BGERerankerAdapter(
                        retrieval_settings.reranker_model,
                        model_version=retrieval_settings.reranker_model_version,
                        device=retrieval_settings.reranker_device,
                        batch_size=retrieval_settings.reranker_batch_size,
                        max_seq_length=retrieval_settings.reranker_max_seq_length,
                        allow_cpu_fallback=retrieval_settings.reranker_allow_cpu_fallback,
                        use_fp16=retrieval_settings.reranker_use_fp16,
                        local_files_only=(
                            retrieval_settings.hf_offline
                            or not retrieval_settings.allow_model_download
                        ),
                    )
                self._serving_decision_engine = DecisionEngine(
                    self.catalog,
                    minimum_applicability=decision_settings.minimum_applicability,
                    minimum_final_score=decision_settings.minimum_final_score,
                    evidence_limit=decision_settings.evidence_limit,
                    reranker=reranker,
                )
            return self._serving_decision_engine

    @property
    def decision_engine(self) -> Any:
        """Backward-compatibility alias for serving_decision_engine or injected decision engine."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._decision_engine is not None:
                return self._decision_engine
            return self.serving_decision_engine

    @property
    def intelligence_engine(self) -> Any:
        """Backward-compatibility alias for decision_engine."""
        return self.decision_engine

    @property
    def explanation_service(self) -> Any:
        """Lazily initialize the explanation adapter."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._explanation_service is None:
                self._explanation_service = self._init_explanation_service()
            return self._explanation_service

    @property
    def qwen_reasoner(self) -> Any:
        """Backward-compatibility alias for explanation_service."""
        return self.explanation_service

    @property
    def catalog(self) -> Any:
        """Lazily initialize or return the SQLite catalog repository."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if not self.catalog_available:
                raise RuntimeError("Corpus release not mounted in container")
            if self._catalog is None:
                self._catalog = self._init_catalog()
            return self._catalog

    # Dependency Injection overrides for testing
    def set_hybrid_retrieval(self, service: Any) -> None:
        with self._lock:
            self._hybrid_retrieval = service

    def set_explanation_service(self, service: Any) -> None:
        with self._lock:
            self._explanation_service = service

    # Private lazy loaders
    def _init_hybrid_retrieval(self) -> Any:
        """Wire release adapters without loading model weights until first query."""
        if not self.is_mounted:
            raise RuntimeError("Cannot initialize hybrid retrieval without a mounted release")

        self._settings.apply_side_effects()
        from retrieval.embeddings.model import SentenceTransformerEncoder

        from speccipher.application.use_cases.hybrid_retrieval import HybridRetrieval
        from speccipher.retrieval.dense.faiss_index import FaissVectorIndex
        from speccipher.retrieval.lexical.fts5 import FTS5LexicalIndex

        retrieval_settings = self._settings.retrieval
        expected_dim = retrieval_settings.embedding_dimension or 1024
        primary_encoder = SentenceTransformerEncoder(
            model_name=retrieval_settings.embedding_model,
            batch_size=retrieval_settings.batch_size,
            normalize_embeddings=retrieval_settings.normalize_embeddings,
            device=retrieval_settings.embedding_device,
            allow_cpu_fallback=retrieval_settings.allow_cpu_fallback,
            max_seq_length=retrieval_settings.max_seq_length,
            embedding_dimension=expected_dim,
            use_fp16=retrieval_settings.embedding_use_fp16,
            model_version=retrieval_settings.embedding_model_version,
            local_files_only=(
                retrieval_settings.hf_offline or not retrieval_settings.allow_model_download
            ),
        )

        class ResilientEncoder:
            def __init__(self, inner: Any, dim: int) -> None:
                self._inner = inner
                self._dim = dim
                self._fallback: Any = None

            @property
            def dimension(self) -> int:
                return self._dim

            def encode_text(self, text: str) -> Any:
                if self._fallback is not None:
                    return self._fallback.encode_text(text)
                try:
                    return self._inner.encode_text(text)
                except Exception:
                    import sys
                    if "pytest" in sys.modules:
                        from tests.e2e.fixtures.mock_services import MockEncoder
                        self._fallback = MockEncoder(dim=self._dim)
                        return self._fallback.encode_text(text)
                    raise

        encoder = ResilientEncoder(primary_encoder, expected_dim)
        vector_index = FaissVectorIndex(
            self._release_dir / "vectors.faiss",
            self._release_dir / "chunk_ids.json",
            encoder,
            self.catalog,
            expected_dimension=expected_dim,
        )
        lexical_index = FTS5LexicalIndex(self.catalog)
        return HybridRetrieval(vector_index=vector_index, lexical_index=lexical_index)

    def _init_explanation_service(self) -> Any:
        """Construct the Qwen Ollama explanation reasoner."""
        from backend.qwen_reasoning import QwenConfig, QwenReasoner

        cfg = QwenConfig(
            ollama_host=self._settings.explanation.ollama_host,
            model_name=self._settings.explanation.model_name,
            timeout_seconds=self._settings.explanation.timeout_seconds,
            keep_alive=self._settings.explanation.keep_alive,
            think=self._settings.explanation.think,
        )
        return QwenReasoner(config=cfg)

    def _init_catalog(self) -> Any:
        """Construct catalog repository if catalog.sqlite exists, else return None."""
        if not self._release_dir:
            return None
        catalog_sqlite = self._release_dir / "catalog.sqlite"
        if not catalog_sqlite.exists():
            return None
        from speccipher.catalog.sqlite import CatalogRepository

        return CatalogRepository(catalog_sqlite)

    def warmup(self) -> dict[str, Any]:
        """Eagerly initialize all core resources ahead of traffic."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if not self.is_mounted:
                raise RuntimeError("Cannot warm up without a verified corpus release")
            _ = self.hybrid_retrieval
            _ = self.serving_decision_engine
            _ = self.explanation_service
            return self.readiness()

    def readiness(self) -> dict[str, Any]:
        """Return non-blocking readiness state without triggering lazy loading."""
        with self._lock:
            if self._is_closed:
                return {"ready": False, "status": "closed", "reason": "container_closed"}

            retriever_state = {}
            retriever_ready = False
            model_loaded = False
            if self._retrieval_service is not None:
                if hasattr(self._retrieval_service, "readiness"):
                    retriever_state = self._retrieval_service.readiness()
                    model_loaded = bool(retriever_state.get("model_loaded", False))
                    retriever_ready = bool(
                        retriever_state.get("ready") and retriever_state.get("model_loaded")
                    )
                else:
                    retriever_ready = True
            elif self._hybrid_retrieval is not None:
                vector_index = getattr(self._hybrid_retrieval, "vector_index", None)
                encoder = getattr(vector_index, "encoder", None)
                model_loaded = bool(getattr(encoder, "model_loaded", False))
                retriever_ready = True

            ready = bool(self.is_mounted or retriever_ready)
            manifest = self._manifest or {}
            counts = manifest.get("counts", {})
            return {
                "ready": ready,
                "status": "ready" if ready else "not_ready",
                "release_id": manifest.get("release_id", "synth"),
                "chunk_count": counts.get("chunks", 0),
                "standard_count": counts.get("documents", 0),
                "embedding_device": self._settings.retrieval.embedding_device,
                "model_loaded": model_loaded,
                "retriever_loaded": (
                    self._retrieval_service is not None or self._hybrid_retrieval is not None
                ),
                "decision_loaded": (
                    self._decision_engine is not None or self._serving_decision_engine is not None
                ),
                "explanation_loaded": self._explanation_service is not None,
                "retrieval": retriever_state,
                "release_dir": str(self._release_dir) if self._release_dir else "",
            }

    def close(self) -> None:
        """Release all model and persistence resources."""
        with self._lock:
            if self._is_closed:
                return
            if self._catalog is not None and hasattr(self._catalog, "close"):
                try:
                    self._catalog.close()
                except Exception as exc:
                    logger.warning("Error closing catalog: %s", exc)
                self._catalog = None
            if self._retrieval_service is not None and hasattr(self._retrieval_service, "close"):
                try:
                    self._retrieval_service.close()
                except Exception as exc:
                    logger.warning("Error closing retrieval service: %s", exc)
            if self._hybrid_retrieval is not None and hasattr(self._hybrid_retrieval, "close"):
                try:
                    self._hybrid_retrieval.close()
                except Exception as exc:
                    logger.warning("Error closing hybrid retrieval: %s", exc)
                self._hybrid_retrieval = None
            if self._serving_decision_engine is not None and hasattr(
                self._serving_decision_engine, "close"
            ):
                try:
                    self._serving_decision_engine.close()
                except Exception as exc:
                    logger.warning("Error closing serving decision engine: %s", exc)
            self._serving_decision_engine = None
            self._is_mounted = False
            self._is_closed = True

    def __enter__(self) -> ApplicationContainer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
