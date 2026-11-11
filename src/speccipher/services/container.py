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
        retrieval_override: Any = None,
        decision_override: Any = None,
        explanation_override: Any = None,
        catalog_override: Any = None,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._release_dir = Path(release_dir) if release_dir else self._settings.paths.release_path
        self._lock = RLock()

        # Service instances initialized to None (or injected overrides)
        self._retrieval_service: Any = retrieval_override
        self._decision_engine: Any = decision_override
        self._explanation_service: Any = explanation_override
        self._catalog: Any = catalog_override
        self._manifest: dict[str, Any] | None = None
        self._is_mounted: bool = False
        self._is_closed: bool = False

        if release_dir is not None:
            self.mount_release(release_dir)
        elif self._release_dir and (self._release_dir / "release.json").exists():
            try:
                self.mount_release(self._release_dir)
            except Exception:
                pass

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
            self._is_mounted = True

            # Invariant: Close previous catalog to prevent Windows file-lock PermissionErrors
            if old_catalog is not None and hasattr(old_catalog, "close"):
                try:
                    old_catalog.close()
                except Exception as exc:
                    logger.warning("Error closing previous catalog: %s", exc)

    @property
    def retrieval_service(self) -> Any:
        """Lazily initialize the retrieval service facade."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._retrieval_service is None:
                self._retrieval_service = self._init_retrieval_service()
            return self._retrieval_service

    @property
    def decision_engine(self) -> Any:
        """Lazily initialize the decision/intelligence engine."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ApplicationContainer is closed")
            if self._decision_engine is None:
                self._decision_engine = self._init_decision_engine()
            return self._decision_engine

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
            if not self.is_mounted and self._catalog is None:
                raise RuntimeError("Corpus release not mounted in container")
            if self._catalog is None:
                self._catalog = self._init_catalog()
            return self._catalog

    # Dependency Injection overrides for testing
    def set_retrieval_service(self, service: Any) -> None:
        with self._lock:
            self._retrieval_service = service

    def set_decision_engine(self, engine: Any) -> None:
        with self._lock:
            self._decision_engine = engine

    def set_explanation_service(self, service: Any) -> None:
        with self._lock:
            self._explanation_service = service

    # Private lazy loaders
    def _init_retrieval_service(self) -> Any:
        """Construct the retrieval service facade."""
        self._settings.apply_side_effects()
        from retrieval.config import RetrievalConfig
        from retrieval.corpus_config import resolve_retrieval_corpus_path
        from retrieval.service import RetrievalService

        corpus_file = resolve_retrieval_corpus_path()
        config = RetrievalConfig(
            embedding_model=self._settings.retrieval.embedding_model,
            embedding_device=self._settings.retrieval.embedding_device,
            allow_cpu_fallback=self._settings.retrieval.allow_cpu_fallback,
            reranker_enabled=self._settings.retrieval.reranker_enabled,
            reranker_model=self._settings.retrieval.reranker_model,
            reranker_device=self._settings.retrieval.reranker_device,
            reranker_allow_cpu_fallback=self._settings.retrieval.reranker_allow_cpu_fallback,
            batch_size=self._settings.retrieval.batch_size,
            sequential_model_lifecycle=self._settings.retrieval.sequential_model_lifecycle,
            top_k=self._settings.retrieval.top_k,
            corpus_path=corpus_file,
        )
        logger.info("Initializing RetrievalService from %s", self._release_dir)
        return RetrievalService.from_persisted(self._release_dir, config=config)

    def _init_decision_engine(self) -> Any:
        """Construct the decision / intelligence engine."""
        from intelligence.pipeline import IntelligenceEngine
        from intelligence.relationships import CorpusKnowledgeBase
        from retrieval.corpus_config import resolve_knowledge_base_paths

        knowledge_paths = resolve_knowledge_base_paths()
        if not knowledge_paths:
            raise RuntimeError("No canonical corpus documents were found under corpus/data")
        knowledge_base = CorpusKnowledgeBase.from_canonical_json(knowledge_paths)
        return IntelligenceEngine(knowledge_base=knowledge_base)

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
            _ = self.retrieval_service
            _ = self.decision_engine
            _ = self.explanation_service
            return self.readiness()

    def readiness(self) -> dict[str, Any]:
        """Return non-blocking readiness state without triggering lazy loading."""
        with self._lock:
            if self._is_closed:
                return {"ready": False, "status": "closed", "reason": "container_closed"}

            retriever_state = {}
            retriever_ready = False
            if self._retrieval_service is not None:
                if hasattr(self._retrieval_service, "readiness"):
                    retriever_state = self._retrieval_service.readiness()
                    retriever_ready = bool(
                        retriever_state.get("ready") and retriever_state.get("model_loaded")
                    )
                else:
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
                "model_loaded": bool(ready),
                "retriever_loaded": self._retrieval_service is not None,
                "decision_loaded": self._decision_engine is not None,
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
            self._is_mounted = False
            self._is_closed = True

    def __enter__(self) -> ApplicationContainer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
