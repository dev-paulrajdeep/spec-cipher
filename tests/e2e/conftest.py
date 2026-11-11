"""Root fixtures and configuration for SpecCipher E2E test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.e2e.fixtures.adapters import (
    ApplicationContainer,
    Settings,
    create_app,
)
from tests.e2e.fixtures.mock_services import MockEncoder, MockOllamaAdapter
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder


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
    """Fixture providing an ApplicationContainer mounted to valid_release."""
    settings = Settings(release_dir=valid_release, embedding_device="cpu")
    cntr = ApplicationContainer(settings, valid_release)
    yield cntr
    cntr.close()


@pytest.fixture
def app(container: ApplicationContainer):
    """Fixture providing a Flask application instance."""
    application = create_app(container)
    application.config.update({"TESTING": True})
    return application


@pytest.fixture
def client(app):
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
