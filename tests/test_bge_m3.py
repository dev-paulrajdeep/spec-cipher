import torch
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

print("\n" + "=" * 60)
print("LOADING BGE-M3")
print("=" * 60)

model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cuda",
)

print("BGE-M3 loaded successfully")

print("\n" + "=" * 60)
print("MODEL INFO")
print("=" * 60)

print("Embedding dimension:", model.get_sentence_embedding_dimension())

print("\n" + "=" * 60)
print("GPU MEMORY")
print("=" * 60)

print(
    f"Allocated: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print(
    f"Reserved:  "
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)

print("\n" + "=" * 60)
print("EMBEDDING TEST")
print("=" * 60)

texts = [
    "Reinforced concrete water retaining structures",
    "Concrete structures for retaining liquids",
    "Electrical installation requirements",
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_tensor=True,
)

print("Shape:", embeddings.shape)
print("Dtype:", embeddings.dtype)
print("Device:", embeddings.device)

print("\n" + "=" * 60)
print("SIMILARITY TEST")
print("=" * 60)

similarity = embeddings @ embeddings.T

print(similarity)

print("\nBGE-M3 GPU SMOKE TEST PASSED")