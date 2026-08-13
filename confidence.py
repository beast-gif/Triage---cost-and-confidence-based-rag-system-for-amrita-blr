"""
confidence.py

    query -> route selection -> Chroma retrieval -> rerank
          -> 5-model ensemble vote -> calibrated confidence

Scoring MATH lives in confidence_v2.py, fitted to calibration_data.json.

TWO-STAGE DEPARTMENT-HEAD RETRIEVAL
-----------------------------------
Only three departments list a head of their own:

    ECE     -> Dr. T. K. Ramesh       (HOD)
    EEE     -> Dr. Vidya H. A.        (Chairperson)
    English -> Dr. Smita Sail         (Head of Department)

For the rest, the SCHOOL PRINCIPAL is the correct answer:

    Mechanical        -> Prof. Sriram Devanathan   (Principal, School of Engineering)
    CSE / AI / AI&DS  -> Dr. Gopalakrishnan E. A.  (Principal, School of Computing)

    stage 1  filter is_department_head=True; accept only if a hit belongs to
             the department that was actually asked about
    stage 2  otherwise filter designation='principal', appending the school
             name to the rerank query so the cross-encoder can tell the two
             Principals apart

THREE GUARDS, each from an observed failure:

  * The filter only engages when a KNOWN department is named. 'who is the
    chairperson' with no department cannot be verified against
    departments_match(), so it goes to plain search rather than confidently
    returning whichever head ranks highest.

  * No Principal fallback for an unrecognised department. Otherwise a query
    about a department that does not exist would return a Principal with high
    confidence.

  * Filtered pools are scored with filtered=True. sep_signal has no rejected
    set to measure against once a filter has removed the rejects — see
    confidence_v2.set_confidence().
"""

import asyncio
import time
from collections import Counter
from query_expand import expand_query

from confidence_V2 import UPLOAD_PROFILE, WEB_PROFILE, compute_confidence_v2
from designation import (
    SCHOOL_NAMES,
    department_in_query,
    departments_match,
    school_for_department,
    wants_department_head,
)
from embedder import embed_query
from ensemble import get_ensemble_votes
from reranker import rerank
from store import get_collection
from upload_store import count as upload_count
from upload_store import get_collection as get_upload_collection
from spell_fix import correct_query

RETRIEVE_N = 15
TOP_K = 5

# The upload store is small (one calendar is 9 chunks), so retrieve everything
# rather than a top-N slice. There is no meaningful vector-search cost at this
# size, and a truncated pool would hide chunks the reranker should judge.
UPLOAD_N = 15

# A metadata-filtered pool is small by construction (6 head chunks, 4 principal
# chunks today), so ask for plenty. Otherwise Chroma's vector index — not the
# filter — decides who makes the pool.
FILTERED_N = 50


def ensemble_agreement(votes: list[str], expected: int = 5) -> dict:
    """
    Fraction of the EXPECTED voters that picked the winning category.

    Denominator is `expected`, not len(votes): classify_llms_concurrent() drops
    providers that raise, so two surviving deterministic classifiers agreeing
    would otherwise report 1.0 — perfect confidence from a collapsed ensemble.
    """
    if not votes:
        return {"majority_label": "general_info", "agreement_score": 0.0,
                "votes_received": 0}

    counts = Counter(votes)
    label, n = counts.most_common(1)[0]
    return {
        "majority_label": label,
        "agreement_score": round(n / max(expected, len(votes)), 4),
        "votes_received": len(votes),
    }


def retrieve(query: str, n: int = RETRIEVE_N, where: dict | None = None):
    """Chroma vector search, optionally filtered by metadata."""
    results = get_collection().query(
        query_embeddings=[embed_query(query)], n_results=n, where=where
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "content": doc,
            "source_url": meta.get("source_url", ""),
            "heading": meta.get("heading", ""),
            "designation": meta.get("designation", ""),
            "department": meta.get("department", ""),
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def retrieve_for(query: str):
    """
    Returns (candidates, rerank_query, route, filtered).

    rerank_query may differ from the user's query: the Principal fallback
    appends the school name so the cross-encoder can separate the Engineering
    Principal from the Computing one. Getting that wrong is a wrong answer,
    not a near miss — there are two different people.
    """
    plain = (retrieve(query), query, None, False)

    if not wants_department_head(query):
        return plain

    dept = department_in_query(query)
    if not dept:
        # Cannot verify a filtered result against the department asked about.
        return plain

    # --- stage 1: does this department list its own head? ---
    #
    # NARROW the pool to the department asked about, rather than retrieving all
    # heads and merely checking afterwards that one of them matches.
    #
    # WHY: measured on a 3-head pool, the right person came SECOND twice —
    #     "who is the chairperson of EEE"          -> Vidya at #2, rank=0.9947
    #     "who is the head of the English department" -> Smita at #2, rank=0.7186
    # On a filtered pool rank_confidence measures how DECISIVELY #1 beat #2,
    # not whether #1 is correct. Both queries were therefore confidently wrong.
    # Filtering by department removes the wrong people from the pool entirely,
    # so there is nothing left to rank first.
    heads = retrieve(query, n=FILTERED_N, where={"is_department_head": True})
    own = [c for c in heads if departments_match(dept, c["department"])]
    if own:
        return own, query, "department_heads", True

    # --- stage 2: fall back to the school Principal ---
    school = school_for_department(dept)
    if not school:
        # Unknown department — do NOT hand back a Principal for a department
        # we cannot place. Plain search will score it low, which is correct.
        return plain

    principals = retrieve(query, n=FILTERED_N, where={"designation": "principal"})
    if not principals:
        return plain

    rerank_query = f"{query} {SCHOOL_NAMES[school]}"
    return principals, rerank_query, f"principal_{school}", True


def _retrieve_and_rerank(query: str, top_k: int):
    """
    The whole blocking half of the pipeline, in one call so it can be handed to
    a worker thread. Returns (candidates, reranked, all_scores, route, filtered).
    """
    candidates, rerank_query, route, filtered = retrieve_for(query)
    if not candidates:
        return [], [], [], route, filtered

    reranked = rerank(rerank_query, candidates, top_k=top_k)
    all_scores = [c["rerank_score"] for c in candidates]
    return candidates, reranked, all_scores, route, filtered


def _retrieve_uploads(query: str, top_k: int):
    """
    Search the admin-uploaded documents. Returns (reranked, all_scores).

    Separate from the web path on purpose — see UPLOAD_PROFILE in
    confidence_v2. The two stores are scored with different constants and rely
    on different signals, and sep_signal only means anything inside a pool that
    is internally comparable.
    """
    if upload_count() == 0:
        return [], []

    results = get_upload_collection().query(
        query_embeddings=[embed_query(query)], n_results=UPLOAD_N
    )
    if not results["ids"] or not results["ids"][0]:
        return [], []

    candidates = [
        {
            "content": doc,
            "source_url": f"{meta.get('title', 'document')} (page {meta.get('page', '?')})",
            "heading": meta.get("title", ""),
            "designation": "",
            "doc_id": meta.get("doc_id", ""),
            "page": meta.get("page", ""),
            "source_type": "upload",
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    reranked = rerank(query, candidates, top_k=top_k)
    return reranked, [c["rerank_score"] for c in candidates]


BAND_RANK = {"low": 0, "medium": 1, "high": 2}


async def score_query(query: str, top_k: int = TOP_K) -> dict:
    """
    Use THIS from FastAPI; __main__ wraps it in asyncio.run().

    THREE THINGS RUN CONCURRENTLY
    -----------------------------
    web retrieval, upload retrieval, and the ensemble vote. They are
    independent: retrieve_for() picks its route from the query text alone
    (wants_department_head / department_in_query), never from the ensemble's
    category label, and the two stores never consult each other.

    HOW THE TWO STORES ARE COMPARED
    -------------------------------
    Each is scored against its OWN profile — see WEB_PROFILE and UPLOAD_PROFILE
    in confidence_v2. That is not a convenience; the stores rely on different
    signals. On the web store sep_signal is essential ("who is the HOD of ECE"
    scores 0.0873 absolute and only separation rescues it). On the upload store
    sep_signal inverts, because every chunk is a month of the same calendar and
    the noise floor is meaningless.

    Because the two scores come from different calibrations, they are NOT
    directly comparable as raw numbers. So the comparison is by BAND first,
    with the raw score only breaking ties within a band. On a tie the upload
    store wins: a hand-curated document an admin deliberately added is more
    authoritative than a scraped page.
    """
    query, _ = correct_query(query)
    query = expand_query(query)
    t0 = time.time()

    async def _timed_web():
        started = time.time()
        out = await asyncio.to_thread(_retrieve_and_rerank, query, top_k)
        print(f"[timing] web retrieval   = {time.time() - started:.2f}s")
        return out

    async def _timed_uploads():
        started = time.time()
        out = await asyncio.to_thread(_retrieve_uploads, query, top_k)
        print(f"[timing] upload retrieval= {time.time() - started:.2f}s")
        return out

    async def _timed_ensemble():
        started = time.time()
        out = await get_ensemble_votes(query)
        print(f"[timing] ensemble        = {time.time() - started:.2f}s")
        return out

    web_result, upload_result, votes = await asyncio.gather(
        _timed_web(), _timed_uploads(), _timed_ensemble(),
        return_exceptions=True,
    )
    print(f"[timing] scoring total   = {time.time() - t0:.2f}s")

    if isinstance(web_result, BaseException):
        raise web_result
    if isinstance(upload_result, BaseException):
        print(f"[WARN] upload retrieval failed: {upload_result}")
        upload_result = ([], [])
    if isinstance(votes, BaseException):
        print(f"[WARN] ensemble failed: {votes}")
        votes = []

    candidates, web_chunks, web_scores, route, filtered = web_result
    upload_chunks, upload_scores = upload_result
    ensemble = ensemble_agreement(votes)
    agreement = ensemble["agreement_score"]

    # --- score each store against its own calibration ---
    web = None
    if web_scores:
        web = compute_confidence_v2(web_scores, agreement, top_k=top_k,
                                    filtered=filtered, profile=WEB_PROFILE)

    uploads = None
    if upload_scores:
        uploads = compute_confidence_v2(upload_scores, agreement, top_k=top_k,
                                        filtered=False, profile=UPLOAD_PROFILE)

    if not web and not uploads:
        return {
            "final_confidence": 0.0, "band": "low", "route": route,
            "source": None, "routing_label": ensemble["majority_label"],
            "generator_mode": "disambiguate", "retrieval_details": {},
            "ensemble_details": ensemble, "chunks": [],
            "candidates_retrieved": 0,
            "alternative": None,
            "top_result": {"source_url": None, "heading": None,
                           "content_preview": None},
            "note": "no candidates retrieved from either store",
        }

    # --- pick the winner: band first, raw score only within a band ---
    def strength(scored):
        return BAND_RANK[scored["band"]] if scored else -1

    if strength(uploads) > strength(web):
        winner, chunks, source = uploads, upload_chunks, "upload"
    elif strength(web) > strength(uploads):
        winner, chunks, source = web, web_chunks, "web"
    else:
        # Same band. Upload wins ties — an admin put that document there
        # deliberately, which is a stronger signal of authority than a page
        # that happened to be scraped.
        if uploads and web and uploads["final"] >= web["final"] * 0.9:
            winner, chunks, source = uploads, upload_chunks, "upload"
        else:
            winner, chunks, source = web, web_chunks, "web"

    loser = web if source == "upload" else uploads

    return {
        "final_confidence": winner["final"],
        "band": winner["band"],
        "generator_mode": winner["generator_mode"],
        "source": source,
        "route": route if source == "web" else "uploaded_documents",
        "routing_label": ensemble["majority_label"],
        "retrieval_details": winner,
        "ensemble_details": ensemble,
        "candidates_retrieved": len(chunks),
        # What the other store would have said, so a wrong pick is debuggable
        # rather than invisible.
        "alternative": {
            "source": "web" if source == "upload" else "upload",
            "band": loser["band"],
            "final": loser["final"],
        } if loser else None,
        "chunks": [
            {
                "source_url": c["source_url"],
                "heading": c["heading"],
                "designation": c.get("designation", ""),
                "rerank_score": round(c["rerank_score"], 4),
                "content": c["content"],
            }
            for c in chunks
        ],
        "top_result": {
            "source_url": chunks[0]["source_url"] if chunks else None,
            "heading": chunks[0]["heading"] if chunks else None,
            "content_preview": chunks[0]["content"][:200] if chunks else None,
        },
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "who is the HOD of ECE"
    result = asyncio.run(score_query(query))

    d = result["retrieval_details"]
    e = result["ensemble_details"]

    print(f'\nQuery: "{query}"\n')
    print("=" * 68)
    print(f"FINAL CONFIDENCE: {result['final_confidence']}  ({result['band'].upper()})")
    print(f"Answered from   : {result.get('source') or 'nothing'}")
    print(f"Generator mode  : {result['generator_mode']}")
    print(f"Retrieval route : {result.get('route') or '(plain vector search)'}")
    print(f"Routing label   : {result['routing_label']}")
    alt = result.get("alternative")
    if alt:
        print(f"Other store     : {alt['source']} would have said "
              f"{alt['final']} ({alt['band']})")
    print("=" * 68)

    if d:
        print(f"\n  abs_signal      : {d['abs_signal']}")
        if d.get("filtered_pool"):
            print(f"  sep_signal      : n/a  (filtered pool — no rejected set)")
            print(f"  set_confidence  : {d['set_confidence']}   = max(abs, rank)")
        else:
            print(f"  sep_signal      : {d['sep_signal']}")
            print(f"  set_confidence  : {d['set_confidence']}   = max(abs, sep)")
        print(f"  rank_confidence : {d['rank_confidence']}")
    print(f"  agreement       : {e.get('agreement_score')}   "
          f"[{e.get('votes_received', 0)}/5 voters]")
    print(f"  candidates      : {result.get('candidates_retrieved', 0)}")

    print("\n--- top chunks ---")
    for i, c in enumerate(result.get("chunks", []), start=1):
        print(f"\n#{i} score={c['rerank_score']:.4f}  desig={c['designation'] or '-'}")
        print(f"   {c['source_url']}")
        print(f"   {' '.join(c['content'].split())[:140]}")