"""Type-safe application settings and environment variable bindings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_str(env: Mapping[str, str], key: str, default: str, *aliases: str) -> str:
    for candidate in (key, *aliases):
        value = env.get(candidate)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_int(env: Mapping[str, str], key: str, default: int, *aliases: str) -> int:
    raw = _env_str(env, key, "", *aliases)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(env: Mapping[str, str], key: str, default: float, *aliases: str) -> float:
    raw = _env_str(env, key, "", *aliases)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(env: Mapping[str, str], key: str, default: bool, *aliases: str) -> bool:
    raw = _env_str(env, key, "", *aliases)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_tuple(
    env: Mapping[str, str], key: str, default: tuple[str, ...], *aliases: str
) -> tuple[str, ...]:
    raw = _env_str(env, key, "", *aliases)
    if not raw:
        return default
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return items if items else default


@dataclass(frozen=True)
class ServerSettings:
    """HTTP delivery and server networking settings."""

    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ()
    max_content_length: int = 16 * 1024  # 16 KB request body limit
    max_input_characters: int = 4_000

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Server port must be between 1 and 65535, got {self.port}")
        if self.max_content_length < 1024:
            raise ValueError(
                f"max_content_length must be at least 1024 bytes, got {self.max_content_length}"
            )
        if self.max_input_characters < 1:
            raise ValueError("max_input_characters must be positive")


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations for artifacts, releases, and frontend."""

    project_root: Path
    release_path: Path
    frontend_dir: Path
    sources_dir: Path
    workspace_dir: Path

    @classmethod
    def resolve(
        cls,
        root: Path | None = None,
        release_path: Path | None = None,
        sources_dir: Path | None = None,
    ) -> PathSettings:
        resolved_root = (root or Path.cwd()).resolve()

        # Priority for release path:
        # 1. Explicit release_path argument
        # 2. data/releases/CURRENT pointer
        # 3. Legacy retrieval-index directory
        if release_path is not None:
            resolved_release = release_path.resolve()
        else:
            current_pointer = resolved_root / "data" / "releases" / "CURRENT"
            if current_pointer.exists():
                resolved_release = current_pointer.resolve()
            else:
                resolved_release = (resolved_root / "retrieval-index").resolve()

        resolved_sources = (sources_dir or (resolved_root / "officialBIS")).resolve()
        return cls(
            project_root=resolved_root,
            release_path=resolved_release,
            frontend_dir=(resolved_root / "frontend").resolve(),
            sources_dir=resolved_sources,
            workspace_dir=(resolved_root / "data" / "workspace").resolve(),
        )


@dataclass(frozen=True)
class RetrievalSettings:
    """Dense, lexical, and reranking parameters."""

    embedding_model: str = "BAAI/bge-m3"
    embedding_model_version: str = "main"
    embedding_dimension: int | None = 1024
    embedding_use_fp16: bool = True
    embedding_device: str = "auto"
    allow_cpu_fallback: bool = True
    batch_size: int = 16
    max_seq_length: int | None = 8192
    normalize_embeddings: bool = True
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_version: str = "main"
    reranker_device: str = "auto"
    reranker_allow_cpu_fallback: bool = True
    reranker_batch_size: int = 16
    sequential_model_lifecycle: bool = False
    top_k: int = 500
    final_top_k: int = 5
    hf_offline: bool = False
    allow_model_download: bool = False

    def __post_init__(self) -> None:
        if self.embedding_device not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                f"embedding_device must be 'auto', 'cuda', or 'cpu', got '{self.embedding_device}'"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.final_top_k < 1:
            raise ValueError("final_top_k must be positive")


@dataclass(frozen=True)
class ExplanationSettings:
    """Ollama client settings for generative summaries."""

    enabled: bool = True
    ollama_host: str = "http://localhost:11434"
    model_name: str = "qwen3-4b-thinking:latest"
    timeout_seconds: float = 180.0
    keep_alive: int | str = 0
    think: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class DecisionSettings:
    """Decision, constraint, and applicability parameters."""

    minimum_applicability: float = 0.20
    minimum_final_score: float = 0.20
    evidence_limit: int = 3
    relationship_depth: int = 1
    final_top_k: int = 5

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum_applicability <= 1.0):
            raise ValueError("minimum_applicability must be between 0.0 and 1.0")
        if not (0.0 <= self.minimum_final_score <= 1.0):
            raise ValueError("minimum_final_score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class Settings:
    """Unified application settings object for SpecCipher."""

    server: ServerSettings = field(default_factory=ServerSettings)
    paths: PathSettings = field(default_factory=lambda: PathSettings.resolve())
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    explanation: ExplanationSettings = field(default_factory=ExplanationSettings)
    decision: DecisionSettings = field(default_factory=DecisionSettings)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
        env = os.environ if environ is None else environ

        server = overrides.get("server") or ServerSettings(
            host=_env_str(env, "SPEC_CIPHER_HOST", "127.0.0.1"),
            port=_env_int(env, "SPEC_CIPHER_PORT", 5000),
            log_level=_env_str(env, "SPEC_CIPHER_LOG_LEVEL", "INFO").upper(),
            cors_origins=_env_tuple(env, "SPEC_CIPHER_CORS_ORIGINS", ()),
            max_content_length=_env_int(env, "SPEC_CIPHER_MAX_CONTENT_LENGTH", 16 * 1024),
            max_input_characters=_env_int(env, "SPEC_CIPHER_MAX_INPUT_CHARACTERS", 4000),
        )

        root = Path(
            overrides.get("project_root")
            or _env_str(env, "SPEC_CIPHER_PROJECT_ROOT", str(Path.cwd()))
        )
        release_raw = overrides.get("release_path") or _env_str(env, "SPEC_CIPHER_RELEASE_PATH", "")
        release_path = Path(release_raw) if release_raw else None
        sources_raw = overrides.get("sources_dir") or _env_str(
            env, "SPEC_CIPHER_BIS_SOURCE_DIR", ""
        )
        sources_dir = Path(sources_raw) if sources_raw else None

        paths = overrides.get("paths") or PathSettings.resolve(
            root=root,
            release_path=release_path,
            sources_dir=sources_dir,
        )

        retrieval = overrides.get("retrieval") or RetrievalSettings(
            embedding_model=_env_str(env, "SPEC_CIPHER_EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=_env_str(env, "SPEC_CIPHER_EMBEDDING_DEVICE", "auto").lower(),
            allow_cpu_fallback=_env_bool(env, "SPEC_CIPHER_ALLOW_CPU_FALLBACK", True),
            reranker_enabled=_env_bool(env, "SPEC_CIPHER_RERANKER_ENABLED", False),
            reranker_model=_env_str(env, "SPEC_CIPHER_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            reranker_device=_env_str(env, "SPEC_CIPHER_RERANKER_DEVICE", "auto").lower(),
            reranker_allow_cpu_fallback=_env_bool(
                env, "SPEC_CIPHER_RERANKER_ALLOW_CPU_FALLBACK", True
            ),
            batch_size=_env_int(env, "SPEC_CIPHER_RETRIEVAL_BATCH_SIZE", 16),
            top_k=_env_int(env, "SPEC_CIPHER_RETRIEVAL_TOP_K", 500),
            final_top_k=_env_int(env, "SPEC_CIPHER_FINAL_TOP_K", 5),
            hf_offline=_env_bool(env, "SPEC_CIPHER_HF_OFFLINE", False),
            allow_model_download=_env_bool(env, "SPEC_CIPHER_ALLOW_MODEL_DOWNLOAD", False),
        )

        explanation = overrides.get("explanation") or ExplanationSettings(
            enabled=_env_bool(env, "SPEC_CIPHER_EXPLANATION_ENABLED", True),
            ollama_host=_env_str(
                env, "SPEC_CIPHER_OLLAMA_HOST", "http://localhost:11434", "QWEN_OLLAMA_HOST"
            ).rstrip("/"),
            model_name=_env_str(
                env, "SPEC_CIPHER_OLLAMA_MODEL", "qwen3-4b-thinking:latest", "QWEN_MODEL_NAME"
            ),
            timeout_seconds=_env_float(
                env, "SPEC_CIPHER_OLLAMA_TIMEOUT_SECONDS", 180.0, "QWEN_TIMEOUT_SECONDS"
            ),
            keep_alive=_env_int(env, "SPEC_CIPHER_OLLAMA_KEEP_ALIVE", 0, "QWEN_KEEP_ALIVE"),
            think=_env_bool(env, "SPEC_CIPHER_OLLAMA_THINK", False, "QWEN_THINK"),
        )

        decision = overrides.get("decision") or DecisionSettings(
            minimum_applicability=_env_float(env, "SPEC_CIPHER_MINIMUM_APPLICABILITY", 0.20),
            minimum_final_score=_env_float(env, "SPEC_CIPHER_MINIMUM_FINAL_SCORE", 0.20),
            evidence_limit=_env_int(env, "SPEC_CIPHER_EVIDENCE_LIMIT", 3),
            relationship_depth=_env_int(env, "SPEC_CIPHER_RELATIONSHIP_DEPTH", 1),
            final_top_k=_env_int(env, "SPEC_CIPHER_FINAL_TOP_K", 5),
        )

        return cls(
            server=server,
            paths=paths,
            retrieval=retrieval,
            explanation=explanation,
            decision=decision,
        )

    def apply_side_effects(self) -> None:
        """Apply required process-level environment flags (e.g. HuggingFace offline mode)."""
        if self.retrieval.hf_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
