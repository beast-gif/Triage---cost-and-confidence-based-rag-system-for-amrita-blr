"""
probe_uploads.py — what does the reranker actually score on uploaded documents?

WHY THIS MATTERS
----------------
confidence_v2.py's constants were fitted to scraped web prose:

    ABS_MIDPOINT = 0.85     top-1 above this = objectively strong
    SEP_MIDPOINT = 1.20     16x above the noise floor = stands out

A calendar table is a different kind of text — pipe-separated rows, dense
dates, repeated column headers. There is no reason to assume the cross-encoder
scores it in the same range, and every one of those constants only means
something within a comparable pool.

So before wiring the upload store into retrieval: look at the numbers.

Read-only.

    python probe_uploads.py
    python probe_uploads.py "when is Deepavali"
"""

import sys

from embedder import embed_query
from reranker import get_reranker
from upload_store import count, get_collection

# Labelled: True = the calendar can answer this, False = it cannot.
#
# The negatives are the important half. A threshold sits in the GAP between the
# two groups, so with only positives there is nothing to place it against —
# which is how ABS_MIDPOINT ended up needing correction on the web store.
QUERIES = [
    # --- the calendar should answer these ---
    ("when do classes commence for UG-S3", True),
    ("when is the mid semester exam", True),
    ("what holidays are there in October", True),
    ("when does the end semester exam start", True),
    ("when is Deepavali", True),
    ("when does the semester vacation start", True),
    ("is 15 August a holiday", True),
    ("when is the last instruction day", True),

    # --- it cannot ---
    ("who is the HOD of ECE", False),
    ("what is the fee for btech ECE", False),
    ("what are the hostel facilities", False),
    ("what is the placement record for computing", False),
    ("what is the eligibility for btech admission", False),
    ("who is the principal of the school of computing", False),
]


def probe(query, ce, show=3):
    result = get_collection().query(
        query_embeddings=[embed_query(query)], n_results=20
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    if not docs:
        return None

    scores = [float(x) for x in ce.predict([(query, d) for d in docs])]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    top = [scores[i] for i in order]
    rest = top[5:] or top[1:]
    floor = sum(rest) / len(rest)
    ratio = (top[0] + 1e-4) / (floor + 1e-4)

    return {
        "query": query,
        "top1": top[0],
        "top2": top[1] if len(top) > 1 else 0.0,
        "floor": floor,
        "ratio": ratio,
        "pool": len(docs),
        "hits": [(scores[i], metas[i].get("page", "?"),
                  " ".join(docs[i].split()).split(":", 1)[-1].strip()[:80])
                 for i in order[:show]],
    }


def report(results):
    """Where does a threshold actually go?"""
    good = [r for r in results if r["answerable"]]
    bad = [r for r in results if not r["answerable"]]

    print("\n" + "=" * 74)
    print("THRESHOLD ANALYSIS")
    print("=" * 74)

    for label, key in (("top1 (absolute score)", "top1"),
                       ("ratio (top1 / noise floor)", "ratio")):
        g = sorted(r[key] for r in good)
        b = sorted(r[key] for r in bad)
        print(f"\n{label}")
        print(f"  answerable : {min(g):.4f} .. {max(g):.4f}")
        print(f"  not answerable : {min(b):.4f} .. {max(b):.4f}")
        gap = min(g) - max(b)
        if gap > 0:
            print(f"  SEPARATED — suggested threshold {(min(g) + max(b)) / 2:.4f}"
                  f"  (margin {gap:.4f})")
        else:
            print(f"  OVERLAP — no threshold works on this signal alone")
            for r in good:
                if r[key] <= max(b):
                    print(f"    answerable but low: {r[key]:.4f}  {r['query']}")

    print("\n" + "-" * 74)
    print("web store for comparison: ABS_MIDPOINT 0.85, SEP_MIDPOINT 1.20 (=16x)")


if __name__ == "__main__":
    print(f"upload store: {count()} chunks")
    ce = get_reranker()

    if len(sys.argv) > 1:
        r = probe(" ".join(sys.argv[1:]), ce)
        if r:
            print(f"\n{r['query']}")
            print(f"  top1={r['top1']:.4f}  floor={r['floor']:.4f}  "
                  f"ratio={r['ratio']:.1f}x  pool={r['pool']}")
            for score, page, body in r["hits"]:
                print(f"    {score:.4f}  p{page}  {body}")
        sys.exit(0)

    results = []
    for query, answerable in QUERIES:
        r = probe(query, ce)
        if not r:
            continue
        r["answerable"] = answerable
        results.append(r)
        mark = "YES" if answerable else "no "
        print(f"\n[{mark}] {query}")
        print(f"      top1={r['top1']:.4f}  ratio={r['ratio']:.1f}x")
        for score, page, body in r["hits"][:2]:
            print(f"      {score:.4f}  p{page}  {body}")

    report(results)