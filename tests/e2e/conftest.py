"""Root fixtures and configuration for SpecCipher E2E test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from flask import Flask

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# DIRECT IMPORTS FROM REAL SPECCIPHER MONOLITH (Zero Fixture Facade)
from speccipher.api.app import create_app
from speccipher.application.use_cases.hybrid_retrieval import HybridRetrieval
from speccipher.config.settings import Settings
from speccipher.domain import (
    Candidate,
    Confidence,
    ConfidenceLevel,
    DocumentChunk,
    LogicalDocument,
    Recommendation,
    RelationshipRole,
    Section,
    SourceFile,
    StandardId,
    StandardRelationship,
    StandardStub,
)
from speccipher.retrieval.dense.faiss_index import FaissVectorIndex
from speccipher.retrieval.lexical.fts5 import FTS5LexicalIndex
from speccipher.services.container import ApplicationContainer

from tests.e2e.fixtures.mock_services import MockEncoder, MockOllamaAdapter
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify this directory's workflow tests outside the default fast suite."""
    e2e_root = Path(__file__).parent.resolve()
    marker = pytest.mark.e2e
    for item in items:
        if Path(str(item.path)).resolve().is_relative_to(e2e_root):
            item.add_marker(marker)


@pytest.fixture
def release_builder(tmp_path: Path) -> SyntheticCorpusReleaseBuilder:
    """Fixture providing a fresh SyntheticCorpusReleaseBuilder."""
    return SyntheticCorpusReleaseBuilder(tmp_path)


@pytest.fixture
def valid_release(release_builder: SyntheticCorpusReleaseBuilder) -> Path:
    """Fixture providing a verified, complete Schema 2.0 corpus release."""
    return release_builder.build_valid_release()


@pytest.fixture
def corrupt_release(release_builder: SyntheticCorpusReleaseBuilder) -> Path:
    """Fixture providing a release with an invalid catalog.sqlite checksum."""
    return release_builder.build_corrupt_checksum_release()


@pytest.fixture
def container(valid_release: Path):
    """Mount a real synthetic release with production retrieval adapters and a fake encoder."""
    settings = Settings.from_env(release_path=valid_release, embedding_device="cpu")
    cntr = ApplicationContainer(settings, valid_release)
    vector_index = FaissVectorIndex(
        valid_release / "vectors.faiss",
        valid_release / "chunk_ids.json",
        MockEncoder(dim=1024),
        cntr.catalog,
        expected_dimension=1024,
    )
    cntr.set_hybrid_retrieval(
        HybridRetrieval(
            vector_index=vector_index,
            lexical_index=FTS5LexicalIndex(cntr.catalog),
        )
    )
    yield cntr
    cntr.close()


@pytest.fixture
def app(container: ApplicationContainer) -> Flask:
    """Fixture providing a genuine Flask application instance from create_app."""
    application = create_app(container)
    application.config.update({"TESTING": True})
    return application


@pytest.fixture
def client(app: Flask):
    """Fixture providing a Flask test client for HTTP requests."""
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Fixture providing a deterministic mock embedding encoder."""
    return MockEncoder(dim=1024)


@pytest.fixture
def mock_ollama() -> MockOllamaAdapter:
    """Fixture providing a mock Ollama adapter."""
    return MockOllamaAdapter(mode="online")
