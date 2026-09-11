from __future__ import annotations

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from retrieval.config import RetrievalConfig
from retrieval.embeddings import SentenceTransformerEncoder


class FakeSentenceTransformer:
    instances: list["FakeSentenceTransformer"] = []
    fail_on_cuda_init = False
    fail_on_cuda_encode = False

    def __init__(self, model_name: str, device: str, model_kwargs=None, local_files_only=False) -> None:
        if device == "cuda" and self.fail_on_cuda_init:
            raise RuntimeError("simulated CUDA initialization failure")
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 512
        self.encode_calls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        if self.device == "cuda" and self.fail_on_cuda_encode:
            raise RuntimeError("simulated CUDA inference failure")
        self.encode_calls.append({"texts": texts, **kwargs})
        return np.asarray([[3.0, 4.0, 0.0] for _ in texts], dtype=np.float32)


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, cuda_available: bool) -> None:
    FakeSentenceTransformer.instances.clear()
    FakeSentenceTransformer.fail_on_cuda_init = False
    FakeSentenceTransformer.fail_on_cuda_encode = False
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch.float16 = object()
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", sentence_transformers)


def test_bge_defaults_and_validation() -> None:
    config = RetrievalConfig()
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.normalize_embeddings is True
    assert config.embedding_device == "auto"
    with pytest.raises(ValueError, match="embedding_device"):
        RetrievalConfig(embedding_device="tpu")
    with pytest.raises(ValueError, match="max_seq_length"):
        RetrievalConfig(max_seq_length=0)


def test_encoder_is_lazy_and_supports_single_batch_empty_and_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=True)
    encoder = SentenceTransformerEncoder(
        "BAAI/bge-m3", batch_size=2, device="auto", max_seq_length=256, embedding_dimension=3
    )

    assert FakeSentenceTransformer.instances == []
    assert encoder.encode_batch([]).shape == (0, 3)
    assert FakeSentenceTransformer.instances == []

    single = encoder.encode_text("query")
    batch = encoder.encode_batch(["one", "two"])
    assert single.shape == (3,)
    assert batch.shape == (2, 3)
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0)
    assert np.allclose(single, batch[0])
    model = FakeSentenceTransformer.instances[0]
    assert model.device == "cuda"
    assert model.max_seq_length == 256
    assert model.encode_calls[0]["normalize_embeddings"] is True


def test_encoder_cpu_fallback_is_automatic(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=False)
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="cuda", allow_cpu_fallback=True)
    encoder.encode_text("query")
    assert FakeSentenceTransformer.instances[-1].device == "cpu"


def test_explicit_cuda_without_fallback_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=False)
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="cuda", allow_cpu_fallback=False)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        encoder.encode_text("query")


def test_encoder_falls_back_after_cuda_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=True)
    FakeSentenceTransformer.fail_on_cuda_init = True
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="auto", embedding_dimension=3, allow_cpu_fallback=True)

    vector = encoder.encode_text("query")

    assert vector.shape == (3,)
    assert FakeSentenceTransformer.instances[-1].device == "cpu"


def test_encoder_falls_back_after_cuda_inference_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=True)
    FakeSentenceTransformer.fail_on_cuda_encode = True
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="auto", embedding_dimension=3, allow_cpu_fallback=True)

    vector = encoder.encode_text("query")

    assert vector.shape == (3,)
    assert FakeSentenceTransformer.instances[-1].device == "cpu"


def test_encoder_rejects_model_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_runtime(monkeypatch, cuda_available=False)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        SentenceTransformerEncoder("BAAI/bge-m3", device="cpu", embedding_dimension=1024).dimension


def test_missing_embedding_runtime_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="cpu")
    with pytest.raises(RuntimeError, match="PyTorch and sentence-transformers"):
        encoder.encode_text("query")
