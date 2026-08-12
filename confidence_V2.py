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

# --- per-store calibration profiles -------------------------------------
# The two stores need DIFFERENT constants, and more than that, they rely on
# DIFFERENT SIGNALS. Both were measured, not chosen.
#
# WEB STORE (25 labelled queries, 6 negatives)
#     Scraped prose. abs_signal alone fails: "who is the HOD of ECE" retrieves
#     correctly at top1=0.0873, which no absolute threshold can accept without
#     also accepting noise. sep_signal carries it — that chunk sits 73x above
#     its background.
#
# UPLOAD STORE (14 labelled queries, 6 negatives)
#     Every chunk is a month of the same calendar, sharing column headers, the
#     school name and the word "Holiday". Measured:
#
#         top1   answerable 0.1646-0.9557   not answerable 0.0000-0.0101
#                SEPARATED, margin 0.1546
#         ratio  answerable 5.2-927x        not answerable 1.0-18.3x
#                OVERLAP — "what holidays are there in October" scored 5.2x
#                while genuinely being the right answer
#
#     So sep_signal INVERTS here. A homogeneous document set has no meaningful
#     noise floor: "when is Deepavali" scored 927x not because 0.2532 is a
#     strong match but because every other chunk scored 0.0005.
#
#     Hence use_sep=False and a low absolute midpoint. 0.06 rather than the
#     midpoint of the measured gap (0.0874) because the binding constraint is
#     the October query at 0.1646 — on this store a false refusal costs more
#     than a weak answer, since the web store can still outrank it.
WEB_PROFILE = {
    "name": "web",
    "abs_midpoint": 0.85,
    "abs_temp": 0.05,
    "sep_midpoint": 1.20,
    "sep_temp": 0.20,
    "use_sep": True,
}

UPLOAD_PROFILE = {
    "name": "upload",
    "abs_midpoint": 0.06,
    "abs_temp": 0.03,
    "sep_midpoint": None,
    "sep_temp": None,
    "use_sep": False,
}

SEP_EPS = 1e-4          # guard against a zero noise floor

# Observed top1-top2 gaps on the web store: min 0.0003, median 0.0231,
# max 0.3424. T=0.05 maps the median gap to ~0.43.
RANK_TEMP = 0.05

# The ensemble classifies the QUERY TEXT and never sees the retrieved chunks,
# so it cannot be evidence that retrieval succeeded — only a discount when the
# query itself is ambiguous. Multiplicative, capped at a 30% reduction.
ENSEMBLE_FLOOR = 0.70
ENSEMBLE_SPAN = 0.30

# Fitted so all unanswerable web queries land LOW (0.09-0.26) and
# well-retrieved ones land HIGH (0.91-1.00).
BAND_HIGH = 0.70
BAND_MEDIUM = 0.35

# assert vs hedge for the answer generator
RANK_ASSERT_THRESHOLD = 0.30


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# the two signals
# ---------------------------------------------------------------------------
def abs_signal(scores, profile=WEB_PROFILE):
    """How strong is the top hit on its own terms?"""
    if not scores:
        return 0.0
    top1 = max(scores)
    return _sigmoid((top1 - profile["abs_midpoint"]) / profile["abs_temp"])


def sep_signal(scores, top_k=TOP_K, profile=WEB_PROFILE):
    """
    How far does the top hit stand above this query's own noise floor?

    Uses log10 of the ratio because the ratios span orders of magnitude
    (1.4x to 374x on the web store); a linear scale would saturate immediately.

    Returns 0.0 when the profile disables it — see UPLOAD_PROFILE for why a
    homogeneous document set makes this signal actively misleading.
    """
    if not profile.get("use_sep") or len(scores) < 2:
        return 0.0
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0]
    rest = ordered[top_k:] or ordered[1:]

    floor = sum(rest) / len(rest)
    ratio = (top1 + SEP_EPS) / (floor + SEP_EPS)
    if ratio <= 0:
        return 0.0
    return _sigmoid(
        (math.log10(ratio) - profile["sep_midpoint"]) / profile["sep_temp"]
    )


def set_confidence(scores, top_k=TOP_K, filtered=False, profile=WEB_PROFILE):
    """
    "Is the answer inside the retrieved set?"

        sep enabled, unfiltered : max(abs_signal, sep_signal)
        sep enabled, filtered   : max(abs_signal, rank_confidence)
        sep disabled            : abs_signal alone

    WHY max() AND NOT A WEIGHTED SUM (web store)
    --------------------------------------------
    Two correct answers had mirror-image profiles:

        "eligibility for btech"   abs 0.95   sep 0.01
        "who is the HOD of ECE"   abs 0.00   sep 0.96

    The first has eight equally-correct chunks (every program page repeats the
    same eligibility text) so nothing separates. The second scores feebly in
    absolute terms but sits 73x above its background. Averaging scores both
    around 0.48 and flags both uncertain.

    WHY FILTERED POOLS NEED rank_confidence INSTEAD
    -----------------------------------------------
    sep_signal measures the winner against the REJECTED chunks. A metadata
    filter removes the rejects before scoring, so there is no noise floor left.
    Observed on "who is the chairman of mechanical engineering", filtered to
    designation='principal':

        pool = [Sriram 0.3862, Gopalakrishnan 0.1832]
        floor falls back to the runner-up, ratio 2.1x, sep_signal 0.012 -> LOW

    The correct answer was rank #1 and the score said LOW, because the "noise
    floor" was the other Principal — a legitimate chunk. On a filtered pool the
    meaningful question is whether the filter singled out one clear winner.
    """
    if not scores:
        return 0.0

    absolute = abs_signal(scores, profile)

    if not profile.get("use_sep"):
        return absolute
    if filtered:
        return max(absolute, rank_confidence(scores, top_k))
    return max(absolute, sep_signal(scores, top_k, profile))


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


def compute_confidence_v2(scores, ensemble_agreement, top_k=TOP_K,
                          filtered=False, profile=WEB_PROFILE):
    """
    scores:   RAW CrossEncoder.predict() values for the full candidate pool.
              Do NOT sigmoid these first — predict() already did.
    filtered: True when the pool came from a metadata-filtered retrieval, in
              which case sep_signal has no rejected set to measure against.
    profile:  WEB_PROFILE or UPLOAD_PROFILE. The two stores were calibrated
              separately and rely on different signals; passing the wrong one
              produces confident nonsense rather than an error.
    """
    sc = set_confidence(scores, top_k, filtered, profile)
    rc = rank_confidence(scores, top_k)
    final = sc * (ENSEMBLE_FLOOR + ENSEMBLE_SPAN * ensemble_agreement)

    uses_sep = profile.get("use_sep") and not filtered

    return {
        "store": profile.get("name"),
        "set_confidence": round(sc, 4),
        "rank_confidence": round(rc, 4),
        "abs_signal": round(abs_signal(scores, profile), 4),
        "sep_signal": round(sep_signal(scores, top_k, profile), 4) if uses_sep else None,
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

# Upload store — 14 labelled queries against the academic calendar. Only top1
# matters for this profile (use_sep=False), so the pools are abbreviated.
UPLOAD_CALIBRATION = [
    ("classes commence for UG-S3", True, [0.7418, 0.6886, 0.4577, 0.3709]),
    ("mid semester exam", True, [0.6908, 0.3308, 0.2763, 0.2749]),
    ("holidays in October", True, [0.1646, 0.0775, 0.0527, 0.0523]),
    ("end semester exam start", True, [0.9557, 0.8555, 0.1070, 0.0469]),
    ("when is Deepavali", True, [0.2532, 0.0005, 0.0004, 0.0003]),
    ("semester vacation start", True, [0.2208, 0.0373, 0.0200, 0.0100]),
    ("is 15 August a holiday", True, [0.1828, 0.0207, 0.0100, 0.0050]),
    ("last instruction day", True, [0.5562, 0.0041, 0.0020, 0.0010]),
    ("who is the HOD of ECE", False, [0.0001, 0.0001, 0.0001, 0.0000]),
    ("fee for btech ECE", False, [0.0000, 0.0000, 0.0000, 0.0000]),
    ("hostel facilities", False, [0.0000, 0.0000, 0.0000, 0.0000]),
    ("placement record for computing", False, [0.0021, 0.0012, 0.0008, 0.0005]),
    ("eligibility for btech admission", False, [0.0010, 0.0004, 0.0003, 0.0002]),
    ("principal of the school of computing", False, [0.0101, 0.0078, 0.0050, 0.0030]),
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

    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 88)
    print("UPLOAD PROFILE — academic calendar, 14 queries, sep_signal DISABLED")
    print("=" * 88)
    print(f"{'query':<40} {'expect':>7} {'top1':>8} {'set':>7} {'band':>7}")
    print("-" * 88)

    up_good, up_bad = [], []
    for query, answerable, scores in UPLOAD_CALIBRATION:
        r = compute_confidence_v2(scores, 1.0, profile=UPLOAD_PROFILE)
        print(f"{query[:40]:<40} {('yes' if answerable else 'no'):>7} "
              f"{max(scores):>8.4f} {r['set_confidence']:>7.3f} {r['band']:>7}")
        (up_good if answerable else up_bad).append(r["final"])

    print(f"\n  answerable   : {min(up_good):.3f} .. {max(up_good):.3f}")
    print(f"  NOT answerable: {min(up_bad):.3f} .. {max(up_bad):.3f}")
    clean_up = min(up_good) > max(up_bad)
    print(f"  separated? {'YES' if clean_up else 'NO'}"
          + (f"   margin {min(up_good) - max(up_bad):.3f}" if clean_up else ""))

    print("\n  Note the signals SWAP between stores. On the web store")
    print("  sep_signal is essential — 'who is the HOD of ECE' scores 0.0873")
    print("  absolute and only separation rescues it. On the upload store")
    print("  sep_signal inverts: every chunk is a month of the same calendar,")
    print("  so 'when is Deepavali' scored 927x not because 0.2532 is strong")
    print("  but because everything else scored 0.0005.")