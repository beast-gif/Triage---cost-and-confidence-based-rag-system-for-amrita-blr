"""
upload_store.py — a SECOND Chroma collection, for admin-uploaded documents.

WHY A SEPARATE COLLECTION
-------------------------
1. DELETION IS TRIVIAL. Removing a document is one filtered delete. In a shared
   collection it would mean hunting chunk_ids by prefix and hoping nothing else
   matched.

2. sync.py CANNOT REACH IT. The purge in sync.py walks the manifest and deletes
   every URL it did not scrape this run. Uploads are never scraped, so in a
   shared collection they would be swept away on every sync — the same bug that
   deleted the entire faculty corpus when the purge ran before pass 2.
   A different collection makes that structurally impossible rather than
   something to remember.

3. CONFIDENCE CAN BE CALIBRATED SEPARATELY. The web-store constants were fitted
   to scraped prose. A calendar table is a different kind of text and the
   reranker will score it differently. Separate stores mean separate pools, and
   sep_signal only means anything when the pool is internally comparable — that
   is what broke on filtered pools and had to be worked around.

NO MANIFEST, NO CONTENT HASH
----------------------------
The manifest exists to diff scraped pages across runs: a page might change, so
you hash its content to notice. An uploaded file does not change — if it does,
the admin uploads a new one and deletes the old. So chunk ids are sequential
and readable:

    upload:{doc_id}:{index}

Deleting a document is `where={"doc_id": doc_id}`.
"""

import json
from datetime import datetime, timezone

import chromadb

DB_PATH = "chroma_db"                  # same directory, different collection
COLLECTION_NAME = "triage_uploads"
REGISTRY_FILE = "uploads_registry.json"

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
    Everything except the id and the document body becomes metadata.

    Generic passthrough for the same reason store.py uses it: a hardcoded field
    list silently drops any field added later, and the failure is invisible
    until something downstream stops filtering correctly.
    """
    metadata = {}
    for key, value in chunk.items():
        if key in {"chunk_id", "content"}:
            continue
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


# ---------------------------------------------------------------------------
# registry — what the admin panel lists
# ---------------------------------------------------------------------------
def _load_registry():
    try:
        with open(REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def list_documents():
    """Uploaded documents, newest first."""
    registry = _load_registry()
    return sorted(registry.values(), key=lambda d: d["uploaded_at"], reverse=True)


def document_exists(doc_id):
    return doc_id in _load_registry()


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------
def add_document(chunks, embeddings, doc_id, title, filename):
    """
    chunks:     from doc_parser.parse_document()
    embeddings: same order and length as chunks

    Replaces any existing document with this doc_id, so re-uploading a corrected
    file is safe rather than leaving both versions in the store.
    """
    if not chunks:
        raise ValueError("no chunks to add — the parser found no text")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )

    if document_exists(doc_id):
        delete_document(doc_id)

    get_collection().upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[_clean_metadata(c) for c in chunks],
    )

    registry = _load_registry()
    registry[doc_id] = {
        "doc_id": doc_id,
        "title": title,
        "filename": filename,
        "chunks": len(chunks),
        "pages": max((c.get("page") or 1) for c in chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_registry(registry)

    return registry[doc_id]


def delete_document(doc_id):
    """Remove every chunk of one document, and its registry entry."""
    get_collection().delete(where={"doc_id": doc_id})

    registry = _load_registry()
    removed = registry.pop(doc_id, None)
    _save_registry(registry)
    return removed


def count():
    return get_collection().count()


if __name__ == "__main__":
    print(f"collection : {COLLECTION_NAME}")
    print(f"chunks     : {count()}")
    docs = list_documents()
    print(f"documents  : {len(docs)}")
    for d in docs:
        print(f"  {d['doc_id']:<28} {d['chunks']:>4} chunks  "
              f"{d['pages']:>3}p  {d['title']}")