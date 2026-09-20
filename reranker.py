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

THREAD SAFETY
-------------
get_reranker() is called from worker threads: confidence.score_query() runs
_retrieve_and_rerank and _retrieve_uploads concurrently via asyncio.to_thread
and both reach rerank(). See the double-checked lock below.
"""

import math
import os
import threading

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

_reranker = None

# Guards the load, NOT the model. predict() is safe to call from several
# threads; it is only construction that must happen once.
_reranker_lock = threading.Lock()

_threads_configured = False


def _container_cpu_limit() -> int | None:
    """
    How many CPUs this process may actually use, or None if unconstrained.

    os.cpu_count() reports the HOST's cores, not the container's share. On a
    2-vCPU Railway instance running on a 64-core host it returns 64, and
    torch.set_num_threads(64) then spawns 64 threads fighting over 2 cores —
    slower than leaving it alone. Reading the cgroup quota is the only way to
    see the real limit from inside.
    """
    try:  # cgroup v2 — "<quota> <period>", or "max <period>" when unlimited
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass

    try:  # cgroup v1 — quota of -1 means unlimited
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, int(quota / period))
    except Exception:
        pass

    return None  # not containerised, or no limit set


def configure_torch_threads() -> int:
    """
    Give torch every core it is allowed to use, once, and report the number.

    WHY THIS EXISTS
    ---------------
    Measured on the dev laptop (4 logical cores), reranking 15 pairs:

        1 thread   26.89s
        2 threads  21.61s     <- torch's own default here
        4 threads  16.02s     1.35x faster than the default

    Torch was defaulting to 2 on a 4-core machine — it picks physical cores,
    which is usually right for BLAS work but was not right here. The scores are
    bit-identical at every thread count; this is pure throughput, with no
    recalibration implied. It is the only free speedup available.

    Override with TORCH_NUM_THREADS if the automatic choice is wrong. Worth
    trying on the deployed instance: hyperthreads helped on this laptop, and on
    other hardware they sometimes do not.

    Safe to call more than once; it only acts the first time.
    """
    global _threads_configured
    if _threads_configured:
        import torch
        return torch.get_num_threads()

    import torch

    override = os.getenv("TORCH_NUM_THREADS")
    if override:
        n = max(1, int(override))
        source = "TORCH_NUM_THREADS"
    else:
        limit = _container_cpu_limit()
        n = max(1, limit or os.cpu_count() or 1)
        source = "cgroup quota" if limit else "os.cpu_count()"

    torch.set_num_threads(n)
    _threads_configured = True
    print(f"torch threads: {n} (from {source})")
    return n


def get_reranker() -> CrossEncoder:
    """
    Lazy-load once, including across threads.

    WHY THE LOCK
    ------------
    The plain `if _reranker is None` version races. Loading reads ~1.1GB from
    disk and releases the GIL while it does, so a second thread can test the
    same condition before the first has assigned, and both load. Two copies get
    built, one is thrown away, and the time is spent twice.

    The giveaway is "Loading reranker model" printed more than once in a single
    run — which is what the CLI probes were doing, inflating the timings
    measured there.

    app.py's lifespan pre-loads before serving, so the FastAPI server takes the
    fast path below and never races. This makes the CLI behave the same way, so
    numbers measured there mean something.

    Same pattern as CHROMA_INIT_LOCK in store.py.
    """
    # Must precede any use of the name — Python rejects a `global` declaration
    # that comes after the variable is read in the same function.
    global _reranker

    # Fast path: no lock once loaded. Taking it on every rerank would serialise
    # the two concurrent retrieval threads against each other for nothing.
    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        # Re-check: another thread may have finished while this one waited.
        if _reranker is None:
            # Before the model exists, so the setting is in place for the very
            # first predict() rather than one query too late.
            configure_torch_threads()
            print(f"Loading reranker model: {MODEL_NAME} ...")
            _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def _sigmoid(x: float) -> float:
    """
    UNUSED — and it must stay that way. Do not apply this to predict() output.

    CrossEncoder.predict() ALREADY applies a sigmoid for bge-reranker-base.
    Applying a second one squashes every score into [0.50, 0.73], which makes
    the HIGH confidence band unreachable and silently breaks the calibration
    in confidence_V2.py. That was a real bug here, found by measurement.

    Kept only so the history is legible. If you are reading this wondering
    whether scores need normalising: they do not.
    """
    return 1 / (1 + math.exp(-x))


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates: list of dicts, each must have a "content" key (the chunk
                text). Other keys (source_url, heading, etc.) are passed
                through untouched.

    Returns the candidates re-sorted by cross-encoder relevance and trimmed to
    top_k. Each one gains a "rerank_score" key.

    "rerank_score_normalized" is also set, to the SAME value — predict()
    already returns a 0-1 score for this model, so there is nothing to
    normalise. The two keys are identical and the second is retained only
    because older callers read it.

    CAREFUL — THIS MUTATES `candidates` IN PLACE, AND THAT IS LOAD-BEARING
    ---------------------------------------------------------------------
    The scores are written onto the caller's dicts, so every candidate carries
    its score afterwards, not just the top_k that come back. confidence.py
    depends on exactly that:

        reranked   = rerank(rerank_query, candidates, top_k=top_k)
        all_scores = [c["rerank_score"] for c in candidates]   # ALL of them

    sep_signal() measures the top hit against the REJECTED pool, so it needs
    the scores of the candidates that did not survive the trim. Rewriting this
    function to be pure — copying the dicts instead of mutating them — would
    leave that list-comprehension raising KeyError, or worse, silently reading
    a stale pool. Leave the mutation in place.
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
    SUPERSEDED — nothing in the live pipeline calls this.

    This is the original confidence heuristic: a hand-picked 0.1 gap scale and
    a 0.6/0.4 weighting. confidence_V2.py replaced it with constants fitted to
    calibration_data.json, and confidence.py imports only rerank() from this
    module.

    Kept because the before/after is worth having for the report — the
    arbitrary constants here are exactly what the calibrated version exists to
    replace. Do not wire it back in.
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
    import time

    from embedder import embed_query
    from store import get_collection

    query = " ".join(sys.argv[1:]) or "Chairperson of Electronics and communication"

    # Load both models BEFORE timing anything. Without this the numbers below
    # include model loading, which is what made the CLI report 30-60s for work
    # the warm server does in a fraction of that.
    get_reranker()
    get_collection()

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

    started = time.time()
    top = rerank(query, candidates, top_k=5)
    elapsed = time.time() - started

    for i, c in enumerate(top):
        print(f"[{i+1}] rerank_score: {c['rerank_score']:.4f} (normalized: {c['rerank_score_normalized']:.4f})")
        print(f"    source: {c['source_url']}")
        print(f"    heading: {c['heading']}")
        print(f"    preview: {c['content'][:200]}")
        print()

    # The number worth quoting: warm cross-encoder cost for this pool size.
    print(f"rerank of {len(candidates)} pairs took {elapsed:.2f}s "
          f"({elapsed / len(candidates) * 1000:.0f}ms per pair)")