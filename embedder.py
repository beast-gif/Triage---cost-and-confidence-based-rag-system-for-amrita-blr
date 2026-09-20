"""
embedder.py

Thin wrapper around a local BGE embedding model (via sentence-transformers).

BGE models expect specific instruction prefixes for best retrieval quality:
  - Documents/chunks being stored:  encode as-is (no prefix needed for bge-base/large)
  - Queries at search time:         prefix with "Represent this sentence for
                                     searching relevant passages: "

embed_texts() handles the document side (ingestion), embed_query() the query
side (retrieval). The prefix is the only difference between them, and getting
it wrong costs retrieval quality silently — the vectors are still valid, just
worse.

THREAD SAFETY
-------------
get_model() is called from worker threads. confidence.score_query() runs
_retrieve_and_rerank and _retrieve_uploads concurrently via asyncio.to_thread,
and both reach embed_query() -> get_model(). See the double-checked lock below
for why a bare `if _model is None` is not enough.
"""

import threading

from sentence_transformers import SentenceTransformer

# Swap to "BAAI/bge-large-en-v1.5" if you want higher quality at the cost
# of speed/memory. bge-base is a good default for a student project.
MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model = None

# Guards the load, NOT the model. SentenceTransformer.encode() is safe to call
# from several threads at once; it is only construction that must happen once.
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """
    Lazy-load the model once and reuse it across calls, including across
    threads.

    WHY THE LOCK
    ------------
    The obvious version has a race:

        if _model is None:
            _model = SentenceTransformer(MODEL_NAME)   # takes seconds

    Loading reads several hundred MB from disk and releases the GIL while it
    does. So thread A can enter the branch and still be loading when thread B
    tests `_model is None` — which is still true, because A has not assigned
    yet. Both load. The second assignment wins and the first copy is eventually
    collected, but the time is spent twice and, briefly, so is ~440MB.

    That is exactly the shape of this pipeline: score_query() runs web and
    upload retrieval in two threads, both of which call embed_query().

    It shows up in the CLI, where nothing is pre-loaded — the giveaway is
    "Loading embedding model" printed more than once in a single run. The
    FastAPI server pre-loads in its lifespan handler, so requests take the fast
    path below and never race. This makes the CLI behave the same way, which
    matters because the CLI is where timings get measured.

    Same pattern as CHROMA_INIT_LOCK in store.py, for the same reason.
    """
    # Must precede any use of the name below — Python rejects a `global`
    # declaration that comes after the variable is read in the same function.
    global _model

    # Fast path: no lock once loaded. Reading a module global is atomic under
    # the GIL, so this needs no synchronisation of its own — and taking the
    # lock on every embed would serialise concurrent retrieval for nothing.
    if _model is not None:
        return _model

    with _model_lock:
        # Re-check inside the lock. Another thread may have finished loading
        # while this one waited, in which case there is nothing left to do.
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