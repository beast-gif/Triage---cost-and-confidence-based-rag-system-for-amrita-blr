"""
calibrate.py — measure the real score distribution so the constants in
confidence_v2.py stay fitted rather than guessed.

WHAT CHANGED SINCE v1
---------------------
1. CORRECTED LABEL. v1 marked "who is the HOD of mechanical engineering" as
   unanswerable and used it as one of three negatives. It IS answerable —
   Mechanical has no departmental head, so the School of Engineering Principal
   (Prof. Sriram Devanathan) is the correct answer. SEP_MIDPOINT was therefore
   partly fitted to a wrong label.

2. FILTERED POOLS MEASURED SEPARATELY. Metadata-filtered retrieval removes the
   rejected chunks, so sep_signal has no noise floor to measure against and
   rank_confidence replaces it. Those two paths need their own statistics —
   pooling them would blur both.

3. 25 QUERIES instead of 12, with 6 negatives instead of 3. Thresholds sit in
   the gap between positive and negative distributions, so the negatives are
   what actually locate them.

This runs the SAME routing logic as confidence.py, so what it measures is what
production does — not an idealised version of it.

Read-only. Embeds queries and reranks. Writes calibration_data.json.

Usage:
    python calibrate.py
    python calibrate.py --verbose
"""

import json
import math
import statistics
import sys

from confidence_V2 import abs_signal, rank_confidence, sep_signal
from confidence import retrieve_for
from reranker import rerank

TOP_K = 5
OUT_FILE = "calibration_data.json"
VERBOSE = "--verbose" in sys.argv
SEP = "=" * 82

# ---------------------------------------------------------------------------
# labelled query set
#
# expect = substring of the source_url the CORRECT chunk should come from,
#          or None when the answer is genuinely not in the corpus.
#
# Scope note: this chatbot covers Amrita Bengaluru only. That makes
# out-of-scope queries MORE likely from real users, not less — a student
# asking about Coimbatore does not know the scope, and the system has to say
# so rather than return the nearest Bengaluru page.
# ---------------------------------------------------------------------------
QUERIES = [
    # --- fees / curriculum / admissions ---------------------------------
    ("what is the fee for btech ECE", "btech-electronics-and-communication"),
    ("how much does btech computer science cost", "btech-computer-science"),
    ("what is the curriculum for btech mechanical", "btech-mechanical"),
    ("what subjects are taught in ECE", "btech-electronics-and-communication"),
    ("what is the eligibility for btech admission", "/program/"),
    ("what entrance exam do I need for btech", "/program/"),
    ("is there a scholarship for toppers", "/program/"),

    # --- faculty: departments WITH their own head -----------------------
    ("who is the HOD of ECE", "tk-ramesh"),
    ("who is the chairperson of electronics and communication", "tk-ramesh"),
    ("who is the chairperson of EEE", "vidya-h-a"),
    ("who is the head of the English department", "s-smita"),

    # --- faculty: departments where the PRINCIPAL is the head -----------
    # Mechanical and the Computing-school departments do not list their own
    # head. v1 wrongly labelled the first of these unanswerable.
    ("who is the HOD of mechanical engineering", "sriram"),
    ("who is the chairman of mechanical engineering", "sriram"),
    ("who is the head of computer science", "ea-gopalakrishnan"),
    ("who is the head of artificial intelligence", "ea-gopalakrishnan"),

    # --- faculty: deputy roles (must NOT hit the head filter) -----------
    ("who is the vice chairperson of ECE", "vinodhini"),
    ("who is the vice chairperson of mechanical", "chittawadigi"),

    # --- campus / facilities / placements -------------------------------
    ("what are the hostel facilities in Bengaluru campus", "/hostel/"),
    ("what sports facilities are available", "/sports/"),
    ("tell me about the library", "/library/"),
    ("what is the placement record for computing", "/placements/"),

    # --- NOT in the corpus ----------------------------------------------
    ("what is the fee for MBA", None),
    ("what is the hostel fee in Coimbatore campus", None),
    ("how do I apply for a PhD", None),
    ("who is the vice chancellor of the university", None),
    ("what is the fee for the Amritapuri campus", None),
]


def main():
    records = []

    for query, expect in QUERIES:
        candidates, rerank_query, route, filtered = retrieve_for(query)
        if not candidates:
            print(f"[SKIP] no candidates: {query}")
            continue

        rerank(rerank_query, candidates, top_k=TOP_K)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        scores = [c["rerank_score"] for c in ranked]

        hit_rank = None
        if expect:
            for pos, c in enumerate(ranked, start=1):
                if expect in c["source_url"]:
                    hit_rank = pos
                    break

        rec = {
            "query": query,
            "expect": expect,
            "route": route,
            "filtered": filtered,
            "hit_rank": hit_rank,
            "pool_size": len(scores),
            "scores": scores,
            "abs": abs_signal(scores),
            "sep": sep_signal(scores, TOP_K),
            "rank": rank_confidence(scores, TOP_K),
            "ranked": [
                {"score": c["rerank_score"], "source_url": c["source_url"],
                 "preview": " ".join(c["content"].split())[:90]}
                for c in ranked[:5]
            ],
        }
        records.append(rec)

        status = "n/a" if expect is None else (
            f"#{hit_rank}" if hit_rank else "MISSED")
        print(f"\n{SEP}")
        print(f"{query}")
        print(f"  route={route or 'plain':<22} pool={len(scores):<3} found={status}")
        print(f"  abs={rec['abs']:.4f}  sep={rec['sep']:.4f}  rank={rec['rank']:.4f}")
        if VERBOSE:
            for i, c in enumerate(rec["ranked"], start=1):
                print(f"    #{i} {c['score']:+.4f}  {c['source_url'][-48:]}")
                print(f"        {c['preview']}")

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    answerable = [r for r in records if r["expect"]]
    negatives = [r for r in records if not r["expect"]]
    found = [r for r in answerable if r["hit_rank"]]
    missed = [r for r in answerable if not r["hit_rank"]]

    print(f"\n\n{SEP}")
    print("CALIBRATION SUMMARY")
    print(SEP)
    print(f"\nanswerable  : {len(answerable)}   "
          f"rank1={sum(1 for r in found if r['hit_rank'] == 1)}  "
          f"top5={sum(1 for r in found if r['hit_rank'] <= 5)}  "
          f"missed={len(missed)}")
    print(f"negatives   : {len(negatives)}")
    for r in missed:
        print(f"    MISSED: {r['query']}")

    def report(label, subset, signal_key):
        """Print the distribution of one signal across a subset."""
        good = [r[signal_key] for r in subset
                if r["expect"] and r["hit_rank"] and r["hit_rank"] <= 2]
        bad = [r[signal_key] for r in subset if not r["expect"]]
        if not good or not bad:
            print(f"\n  {label}: not enough data "
                  f"({len(good)} good, {len(bad)} negative)")
            return
        print(f"\n  {label}")
        print(f"    good (rank 1-2) : min={min(good):.4f}  "
              f"median={statistics.median(good):.4f}  max={max(good):.4f}")
        print(f"    negatives       : min={min(bad):.4f}  "
              f"median={statistics.median(bad):.4f}  max={max(bad):.4f}")
        gap = min(good) - max(bad)
        print(f"    separated?      : {'YES' if gap > 0 else 'NO — OVERLAP'}"
              f"   (margin {gap:+.4f})")
        if gap > 0:
            print(f"    suggested threshold: {(min(good) + max(bad)) / 2:.4f}")

    unfiltered = [r for r in records if not r["filtered"]]
    filtered = [r for r in records if r["filtered"]]

    print(f"\n{SEP}")
    print(f"UNFILTERED POOLS  (n={len(unfiltered)})  -> uses max(abs, sep)")
    print(SEP)
    report("abs_signal", unfiltered, "abs")
    report("sep_signal", unfiltered, "sep")

    print(f"\n{SEP}")
    print(f"FILTERED POOLS  (n={len(filtered)})  -> uses max(abs, rank)")
    print(SEP)
    if filtered:
        for r in filtered:
            tag = "OK " if r["hit_rank"] == 1 else "??? "
            print(f"  {tag} {r['query'][:46]:<46} pool={r['pool_size']:<3} "
                  f"rank={r['rank']:.4f}")
        print("\n  (negatives rarely reach a filtered pool — the department")
        print("   guard sends unrecognised departments to plain search, so")
        print("   filtered thresholds are validated by rank order, not by")
        print("   separation from negatives.)")
    else:
        print("  none — check that head queries are routing correctly")

    # raw log-ratios, for re-deriving SEP_MIDPOINT by hand if needed
    print(f"\n{SEP}")
    print("RAW log10(top1 / mean(rest)) — unfiltered pools only")
    print(SEP)
    for r in sorted(unfiltered, key=lambda x: x["sep"]):
        ordered = sorted(r["scores"], reverse=True)
        rest = ordered[TOP_K:] or ordered[1:]
        floor = sum(rest) / len(rest)
        ratio = (ordered[0] + 1e-4) / (floor + 1e-4)
        label = "NEGATIVE" if not r["expect"] else f"rank {r['hit_rank']}"
        print(f"  {math.log10(ratio):6.3f}   {label:<10}  {r['query'][:46]}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nraw data written to {OUT_FILE}")


if __name__ == "__main__":
    main()