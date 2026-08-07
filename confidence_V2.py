"""
confidence_v2.py — confidence scoring derived from measured reranker behaviour.

Every constant here was fitted to calibration_data.json (12 labelled queries,
9 answerable + 3 not). Nothing is hand-picked.

FINDING 1 — SIGMOID WAS BEING APPLIED TWICE
-------------------------------------------
CrossEncoder.predict() for bge-reranker-base ALREADY applies sigmoid. Every
value in the calibration data is in [0, 1]. reranker.py then did:

    rerank_score_normalized = _sigmoid(score)     # sigmoid of a sigmoid

sigmoid([0,1]) = [0.50, 0.73]. That is the entire compression that made
similarity_score behave like a constant and capped final confidence at ~0.73.
Use the raw predict() score directly. Do not sigmoid it again.

FINDING 2 — ABSOLUTE SCORE IS NOT COMPARABLE ACROSS QUERIES
------------------------------------------------------------
    "who is the HOD of ECE"        top1 = 0.0873   CORRECT, rank #1
    "eligibility for btech"        top1 = 0.9956   CORRECT, rank #1
    "hostel fee Coimbatore"        top1 = 0.0213   nothing in corpus

The reranker calibrates differently for faculty text than for fee tables. Any
threshold accepting the 0.0873 correct answer also accepts the 0.0213 wrong
one. Measured overlap: correct chunks ranged 0.0022-0.9998, irrelevant ones
0.0001-0.6151. No single threshold separates them.

FINDING 3 — SEPARATION WORKS, EXCEPT WHEN EVERYTHING IS RIGHT
--------------------------------------------------------------
    query                    top1     mean(rest)   ratio
    HOD of ECE      CORRECT  0.0873     0.0012      75x
    hostel          CORRECT  0.9998     0.0095     105x
    sports          CORRECT  0.9970     0.0027     374x
    eligibility     CORRECT  0.9956     0.7191     1.4x   <-- low!
    HOD mechanical  WRONG    0.6151     0.0800      7.7x
    MBA fee         WRONG    0.0919     0.0092      9.9x
    Coimbatore      WRONG    0.0213     0.0038      5.6x

Ratio catches all three failures (all under 10x) but condemns the eligibility
query, where EIGHT chunks are correct because every program page repeats the
same eligibility text. Low separation is ambiguous: everything equally wrong,
or everything equally right. Absolute score disambiguates.

    score = max(abs_signal, sep_signal)

Either signal alone is sufficient. Requiring both is what broke it.

KNOWN LIMITATION
----------------
"what is the placement record for computing" returns a publications chunk at
top1=0.1098 with a 75x ratio — indistinguishable in shape from the correct HOD
query (top1=0.0873, 76x). Retrieval failed; the scores cannot reveal it. No
confidence formula built on reranker scores alone can catch this case, and the
report should say so rather than claim otherwise.

Self-test:  python confidence_v2.py
"""

import math

TOP_K = 5

# --- absolute term -------------------------------------------------------
# "is the top hit objectively strong, regardless of context?"
# Midpoint 0.85 sits above every unanswerable query's top1 (max was 0.6151)
# and below the redundant-answer queries (0.9632, 0.9956).
ABS_MIDPOINT = 0.85
ABS_TEMP = 0.05

# --- separation term -----------------------------------------------------
# "does the top hit tower over THIS query's own noise floor?"
# Measured log10(top1 / mean(rest)):
#     answerable, found well : 1.41, 1.88, 1.19, 1.21, 2.02, 2.57
#     unanswerable           : 0.89, 1.00, 0.75
# Midpoint 1.2 sits in the gap.
SEP_MIDPOINT = 1.20
SEP_TEMP = 0.20
SEP_EPS = 1e-4          # guard against a zero noise floor

# --- rank term -----------------------------------------------------------
# Observed top1-top2 gaps: min 0.0003, median 0.0231, max 0.3424.
# T=0.05 maps the median gap to ~0.43.
RANK_TEMP = 0.05

# --- ensemble ------------------------------------------------------------
# The ensemble classifies the QUERY TEXT and never sees the retrieved chunks,
# so it cannot be evidence that retrieval succeeded — only a discount when the
# query itself is ambiguous. Multiplicative, capped at a 30% reduction.
ENSEMBLE_FLOOR = 0.70
ENSEMBLE_SPAN = 0.30

# --- bands ---------------------------------------------------------------
# Fitted so all 3 unanswerable queries land LOW (0.10, 0.17, 0.27) and
# well-retrieved answerable ones land HIGH (0.91-1.00).
BAND_HIGH = 0.70
BAND_MEDIUM = 0.35

# assert vs hedge for the answer generator
RANK_ASSERT_THRESHOLD = 0.30


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# the two signals
# ---------------------------------------------------------------------------
def abs_signal(scores):
    """How strong is the top hit on its own terms?"""
    if not scores:
        return 0.0
    top1 = max(scores)
    return _sigmoid((top1 - ABS_MIDPOINT) / ABS_TEMP)


def sep_signal(scores, top_k=TOP_K):
    """
    How far does the top hit stand above this query's own noise floor?

    Uses log10 of the ratio because the ratios span two orders of magnitude
    (1.4x to 374x); a linear scale would saturate immediately.
    """
    if len(scores) < 2:
        return 0.0
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0]
    rest = ordered[top_k:] or ordered[1:]

    floor = sum(rest) / len(rest)
    ratio = (top1 + SEP_EPS) / (floor + SEP_EPS)
    if ratio <= 0:
        return 0.0
    return _sigmoid((math.log10(ratio) - SEP_MIDPOINT) / SEP_TEMP)


def set_confidence(scores, top_k=TOP_K, filtered=False):
    """
    "Is the answer inside the retrieved set?"

        unfiltered pool:  max(abs_signal, sep_signal)
        filtered pool:    max(abs_signal, rank_confidence)

    WHY FILTERED POOLS NEED A DIFFERENT SIGNAL
    ------------------------------------------
    sep_signal measures the winner against the REJECTED chunks. A metadata
    filter removes the rejects before scoring, so there is no noise floor left
    to measure against.

    Observed live on "who is the chairman of mechanical engineering", filtered
    to designation='principal':

        pool = [Sriram 0.3862, Gopalakrishnan 0.1832]     only 2 candidates
        floor falls back to the runner-up = 0.1832
        ratio = 2.1x  ->  sep_signal = 0.012  ->  LOW

    The correct answer was ranked #1, and the score said LOW — because the
    "noise floor" was the other Principal, a legitimate chunk rather than
    noise. The filter worked so well it destroyed the signal.

    On a filtered pool the meaningful question is not "is there signal above
    noise" — the filter already guaranteed relevance — but "did the filter
    single out one clear winner". That is rank_confidence, which scored 0.9994
    on the same query.
    """
    if not scores:
        return 0.0
    if filtered:
        return max(abs_signal(scores), rank_confidence(scores, top_k))
    return max(abs_signal(scores), sep_signal(scores, top_k))


def rank_confidence(scores, top_k=TOP_K):
    """
    "Do I know WHICH candidate is the answer?"

        tanh((top1 - top2) / 0.05)

    Feeds the generator's assert-vs-hedge decision, NOT the band. A low value
    with a high set_confidence means several chunks are equally good — which is
    correct for the eligibility query, where eight pages carry the same text.
    """
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores, reverse=True)[:top_k]
    return math.tanh((ordered[0] - ordered[1]) / RANK_TEMP)


def band(score):
    if score >= BAND_HIGH:
        return "high"
    if score >= BAND_MEDIUM:
        return "medium"
    return "low"


def compute_confidence_v2(scores, ensemble_agreement, top_k=TOP_K, filtered=False):
    """
    scores:   RAW CrossEncoder.predict() values for the full candidate pool.
              Do NOT sigmoid these first — predict() already did.
    filtered: True when the pool came from a metadata-filtered retrieval, in
              which case sep_signal is undefined (no rejected set exists) and
              rank_confidence takes its place. See set_confidence().
    """
    sc = set_confidence(scores, top_k, filtered)
    rc = rank_confidence(scores, top_k)
    final = sc * (ENSEMBLE_FLOOR + ENSEMBLE_SPAN * ensemble_agreement)

    return {
        "set_confidence": round(sc, 4),
        "rank_confidence": round(rc, 4),
        "abs_signal": round(abs_signal(scores), 4),
        "sep_signal": round(sep_signal(scores, top_k), 4) if not filtered else None,
        "filtered_pool": filtered,
        "ensemble_agreement": round(ensemble_agreement, 4),
        "final": round(final, 4),
        "band": band(final),
        "generator_mode": "assert" if rc >= RANK_ASSERT_THRESHOLD else "disambiguate",
    }


# ---------------------------------------------------------------------------
# self-test — real scores from all 12 calibration queries
# ---------------------------------------------------------------------------
CALIBRATION = [
    ("fee for btech ECE", 2, [
        0.9909, 0.9652, 0.8097, 0.6579, 0.6067, 0.0917, 0.0855, 0.0683,
        0.0526, 0.0426, 0.0145, 0.0128, 0.0087, 0.0073, 0.0019]),
    ("curriculum for btech mechanical", 2, [
        0.9632, 0.9427, 0.9371, 0.9308, 0.9188, 0.9154, 0.8971, 0.8803,
        0.8685, 0.8581, 0.8308, 0.7220, 0.6738, 0.3324, 0.2883]),
    ("eligibility for btech admission", 1, [
        0.9956, 0.9950, 0.9940, 0.9940, 0.9940, 0.9940, 0.9940, 0.9914,
        0.9353, 0.7603, 0.7444, 0.6723, 0.5859, 0.4489, 0.0645]),
    ("who is the HOD of ECE", 1, [
        0.0873, 0.0092, 0.0086, 0.0084, 0.0072, 0.0045, 0.0026, 0.0015,
        0.0013, 0.0006, 0.0003, 0.0003, 0.0002, 0.0002, 0.0001]),
    ("who is the head of the English department", 7, [
        0.5610, 0.2186, 0.1638, 0.1282, 0.1271, 0.0956, 0.0613, 0.0420,
        0.0366, 0.0353, 0.0338, 0.0249, 0.0164, 0.0143, 0.0061]),
    ("who is the vice chairperson of ECE", 2, [
        0.5352, 0.3861, 0.3779, 0.0755, 0.0704, 0.0687, 0.0582, 0.0447,
        0.0390, 0.0340, 0.0250, 0.0207, 0.0202, 0.0124, 0.0088]),
    ("hostel facilities Bengaluru", 1, [
        0.9998, 0.9971, 0.7539, 0.1601, 0.0549, 0.0488, 0.0118, 0.0095,
        0.0079, 0.0075, 0.0033, 0.0029, 0.0017, 0.0011, 0.0007]),
    ("placement record for computing", 7, [
        0.1098, 0.0633, 0.0125, 0.0081, 0.0032, 0.0024, 0.0022, 0.0021,
        0.0020, 0.0014, 0.0014, 0.0012, 0.0011, 0.0007, 0.0003]),
    ("what sports facilities are available", 1, [
        0.9970, 0.9967, 0.1032, 0.0191, 0.0189, 0.0178, 0.0025, 0.0020,
        0.0015, 0.0013, 0.0010, 0.0008, 0.0003, 0.0001, 0.0000]),
    # --- unanswerable ---
    ("who is the HOD of mechanical engineering", None, [
        0.6151, 0.3906, 0.3220, 0.2821, 0.2319, 0.1685, 0.1677, 0.0923,
        0.0921, 0.0851, 0.0823, 0.0598, 0.0480, 0.0037, 0.0003]),
    ("what is the fee for MBA", None, [
        0.0919, 0.0806, 0.0728, 0.0531, 0.0270, 0.0266, 0.0180, 0.0140,
        0.0135, 0.0086, 0.0068, 0.0033, 0.0010, 0.0004, 0.0003]),
    ("hostel fee in Coimbatore campus", None, [
        0.0213, 0.0193, 0.0171, 0.0131, 0.0123, 0.0107, 0.0076, 0.0074,
        0.0036, 0.0035, 0.0026, 0.0017, 0.0004, 0.0002, 0.0001]),
]

if __name__ == "__main__":
    print("=" * 88)
    print("confidence_v2 self-test — real reranker scores, 12 calibration queries")
    print("=" * 88)
    print(f"{'query':<42} {'rank':>5} {'abs':>7} {'sep':>7} {'set':>7} {'band':>7}")
    print("-" * 88)

    answerable_good, answerable_bad, unanswerable = [], [], []

    for query, hit_rank, scores in CALIBRATION:
        r = compute_confidence_v2(scores, 1.0)
        rank_str = str(hit_rank) if hit_rank else "none"
        print(f"{query[:42]:<42} {rank_str:>5} {r['abs_signal']:>7.3f} "
              f"{r['sep_signal']:>7.3f} {r['set_confidence']:>7.3f} {r['band']:>7}")

        if hit_rank is None:
            unanswerable.append(r["final"])
        elif hit_rank <= 2:
            answerable_good.append(r["final"])
        else:
            answerable_bad.append(r["final"])

    print("\n" + "=" * 88)
    print("SEPARATION CHECK")
    print("=" * 88)
    print(f"  answerable, found at rank 1-2 : "
          f"min={min(answerable_good):.3f}  max={max(answerable_good):.3f}")
    print(f"  answerable, found at rank 7   : "
          f"{[round(x, 3) for x in answerable_bad]}")
    print(f"  UNANSWERABLE                  : "
          f"min={min(unanswerable):.3f}  max={max(unanswerable):.3f}")

    clean = min(answerable_good) > max(unanswerable)
    print(f"\n  good answers separated from non-answers? {'YES' if clean else 'NO'}")
    if clean:
        print(f"  margin: {min(answerable_good) - max(unanswerable):.3f}")

    print("\n  NOTE: 'placement record for computing' scores high despite")
    print("  retrieving the wrong chunk. Its score shape (top1=0.1098, 75x")
    print("  separation) is indistinguishable from the correct HOD query")
    print("  (top1=0.0873, 76x). This failure is invisible to any formula")
    print("  built on reranker scores alone.")