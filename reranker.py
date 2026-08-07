"""
reranker.py

A cross-encoder reranker. Unlike embeddings (query and chunk encoded
SEPARATELY, then compared by distance), a cross-encoder reads the query
and each candidate chunk TOGETHER and outputs a genuine relevance score.
This is what catches cases like "Chairperson of ECE" vs a chunk that only
says "Board of Studies Member" — textually related, but not the same
claim — something embedding-distance alone can't reliably tell apart.

Usage pattern:
  1. Retrieve a WIDER net from Chroma (e.g. top 15-20 by embedding distance)
  2. Rerank those candidates with this module
  3. Keep only the top 3-5 after reranking for confidence scoring / final answer
"""

from sentence_transformers import CrossEncoder
import math

MODEL_NAME = "BAAI/bge-reranker-base"

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {MODEL_NAME} ...")
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates: list of dicts, each must have a "content" key (the chunk
                text). Other keys (source_url, heading, etc.) are passed
                through untouched.
    Returns candidates re-sorted by cross-encoder relevance, trimmed to
    top_k, each with an added "rerank_score" key (raw logit) and
    "rerank_score_normalized" key (sigmoid, 0-1 range).
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    raw_scores = model.predict(pairs)

    for c, score in zip(candidates, raw_scores):
        c["rerank_score"] = float(score)
        c["rerank_score_normalized"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def rerank_confidence(reranked: list[dict]) -> dict:
    """
    Same shape as retrieval_confidence() in confidence.py, but built from
    reranker scores (higher = better) instead of Chroma distances
    (lower = better). Lets confidence.py swap in reranked results without
    changing its combination logic.
    """
    if not reranked:
        return {"similarity_score": 0.0, "gap_score": 0.0, "retrieval_confidence": 0.0}

    top1 = reranked[0]["rerank_score_normalized"]
    top2 = reranked[1]["rerank_score_normalized"] if len(reranked) > 1 else top1

    similarity_score = top1  # already 0-1 from sigmoid
    gap = top1 - top2
    gap_score = min(1.0, gap / 0.1)  # same heuristic scale as before; tune empirically

    return {
        "similarity_score": round(similarity_score, 4),
        "gap_score": round(gap_score, 4),
        "retrieval_confidence": round(0.6 * similarity_score + 0.4 * gap_score, 4),
    }


if __name__ == "__main__":
    import sys
    from embedder import embed_query
    from store import get_collection

    query = " ".join(sys.argv[1:]) or "Chairperson of Electronics and communication"

    collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=15)

    candidates = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        candidates.append({
            "content": doc,
            "source_url": meta["source_url"],
            "heading": meta["heading"],
        })

    print(f'Query: "{query}"')
    print(f"Retrieved {len(candidates)} candidates from Chroma, reranking...\n")

    top = rerank(query, candidates, top_k=5)

    for i, c in enumerate(top):
        print(f"[{i+1}] rerank_score: {c['rerank_score']:.4f} (normalized: {c['rerank_score_normalized']:.4f})")
        print(f"    source: {c['source_url']}")
        print(f"    heading: {c['heading']}")
        print(f"    preview: {c['content'][:200]}")
        print()