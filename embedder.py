"""
embedder.py

Thin wrapper around a local BGE embedding model (via sentence-transformers).

BGE models expect specific instruction prefixes for best retrieval quality:
  - Documents/chunks being stored:  encode as-is (no prefix needed for bge-base/large)
  - Queries at search time:         prefix with "Represent this sentence for
                                     searching relevant passages: "
This module only handles the document side (embedding chunks for storage).
Use a matching query-embedding helper at retrieval time in your retriever code.
"""

from sentence_transformers import SentenceTransformer

# Swap to "BAAI/bge-large-en-v1.5" if you want higher quality at the cost
# of speed/memory. bge-base is a good default for a student project.
MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the model once and reuse it across calls."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of chunk texts. Returns a list of float vectors,
    same order as input."""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalized vectors -> cosine similarity via dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query. BGE recommends a different prefix for
    queries vs documents to improve retrieval quality."""
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()