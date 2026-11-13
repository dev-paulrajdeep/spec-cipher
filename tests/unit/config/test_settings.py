from __future__ import annotations

import pytest

from speccipher.config.settings import RetrievalSettings, Settings


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"reranker_enabled": 1}, "reranker_enabled"),
        ({"reranker_model": ""}, "reranker model name"),
        ({"reranker_device": "tpu"}, "reranker_device"),
        ({"reranker_batch_size": 0}, "reranker_batch_size"),
        ({"reranker_max_seq_length": 0}, "reranker_max_seq_length"),
        ({"reranker_use_fp16": 1}, "reranker_use_fp16"),
    ],
)
def test_retrieval_settings_validate_reranker_configuration(changes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalSettings(**changes)


def test_settings_resolve_reranker_and_model_download_policy_from_environment() -> None:
    settings = Settings.from_env(
        {
            "SPEC_CIPHER_RERANKER_ENABLED": "true",
            "SPEC_CIPHER_RERANKER_BATCH_SIZE": "8",
            "SPEC_CIPHER_RERANKER_MAX_SEQ_LENGTH": "1024",
            "SPEC_CIPHER_ALLOW_MODEL_DOWNLOAD": "true",
            "SPEC_CIPHER_EMBEDDING_MODEL_VERSION": "revision-1",
            "SPEC_CIPHER_EMBEDDING_DIMENSION": "1024",
            "SPEC_CIPHER_EMBEDDING_USE_FP16": "false",
            "SPEC_CIPHER_EMBEDDING_MAX_SEQ_LENGTH": "4096",
            "SPEC_CIPHER_NORMALIZE_EMBEDDINGS": "false",
            "SPEC_CIPHER_SEQUENTIAL_MODEL_LIFECYCLE": "true",
        }
    )

    assert settings.retrieval.reranker_enabled is True
    assert settings.retrieval.reranker_batch_size == 8
    assert settings.retrieval.reranker_max_seq_length == 1024
    assert settings.retrieval.allow_model_download is True
    assert settings.retrieval.embedding_model_version == "revision-1"
    assert settings.retrieval.embedding_dimension == 1024
    assert settings.retrieval.embedding_use_fp16 is False
    assert settings.retrieval.max_seq_length == 4096
    assert settings.retrieval.normalize_embeddings is False
    assert settings.retrieval.sequential_model_lifecycle is True
