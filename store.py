"""
store.py

Thin wrapper around a persistent local Chroma collection. Handles
upserting chunks (with their embeddings + metadata) and deleting by
chunk_id. Kept separate from embedder.py so sync.py can control exactly
which chunks get embedded (only new/changed ones) vs just deleted
(no embedding needed to delete).
"""

import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "triage_admissions"

# Chunk fields that become the Chroma id and document respectively, so they
# must NOT be duplicated into the metadata dict.
RESERVED_KEYS = {"chunk_id", "content"}

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _clean_metadata(chunk: dict) -> dict:
    """
    Pass every chunk field through to Chroma metadata except the reserved ones.

    This used to hardcode five keys (source_url, title, heading, chunk_type,
    token_estimate). extractor.py now also attaches designation fields, and a
    hardcoded list would silently drop them on every upsert. Generic
    passthrough means any field added to a chunk dict flows through
    automatically.

    Chroma accepts only scalars (str / int / float / bool) and rejects None.
    """
    metadata = {}
    for key, value in chunk.items():
        if key in RESERVED_KEYS:
            continue
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """
    chunks: list of chunk dicts (must have chunk_id + content; every other
            field is written through to metadata automatically)
    embeddings: list of float vectors, same order/length as chunks
    """
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )
    collection = get_collection()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[_clean_metadata(c) for c in chunks],
    )


def update_metadata(chunk_ids: list[str], metadatas: list[dict]):
    """
    Update metadata WITHOUT touching stored embeddings.

    Passing no embeddings to Chroma's update() leaves the existing vectors
    intact. Because chunk_ids are content hashes, unchanged chunk text keeps
    the same id, so metadata can be backfilled onto the whole collection with
    zero re-embedding. Used by backfill_designations.py.
    """
    if not chunk_ids:
        return
    if len(chunk_ids) != len(metadatas):
        raise ValueError(
            f"ids/metadatas length mismatch: {len(chunk_ids)} vs {len(metadatas)}"
        )
    get_collection().update(ids=chunk_ids, metadatas=metadatas)


def delete_chunks(chunk_ids: list[str]):
    if not chunk_ids:
        return
    collection = get_collection()
    collection.delete(ids=chunk_ids)


def count() -> int:
    return get_collection().count()