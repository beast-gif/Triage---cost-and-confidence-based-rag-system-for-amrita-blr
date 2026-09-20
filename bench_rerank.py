"""
bench_rerank.py — why is the cross-encoder taking 1027ms per pair?

READ-ONLY. Queries Chroma and runs the reranker. Nothing is embedded, written
or re-indexed, so this costs nothing but CPU time.

    python bench_rerank.py
    python bench_rerank.py "what is the fee for btech ECE"

Measured on the real pool, because a synthetic benchmark would use short
strings and miss the thing most likely to be responsible: sequence length.

WHAT IT TESTS
-------------
1. How many threads torch is actually using. If it is 1 on a multi-core
   machine, that alone is a 4-8x penalty and costs nothing to fix.

2. How long the candidate chunks are, in TOKENS. Cross-encoder cost grows with
   sequence length — roughly linearly through the feed-forward layers and
   quadratically through attention — so 512-token pairs cost far more than the
   ~100-token pairs people assume when they quote "30ms per pair".

3. What shortening max_length actually buys, and what it costs in SCORE
   FIDELITY. This is the part that matters: the confidence constants in
   confidence_V2.py (abs_midpoint 0.85, sep_midpoint 1.20) are fitted to the
   scores this reranker produces at its current settings. A faster setting that
   moves the scores is not free — it means re-running calibrate.py.

   So every timing below is printed next to what it did to the top-1 score and
   the ranking. Read both columns before changing anything.
"""

import os
import sys
import time

import torch
from sentence_transformers import CrossEncoder

from embedder import embed_query
from reranker import MODEL_NAME
from store import get_collection

N_CANDIDATES = 15          # matches RETRIEVE_N in confidence.py
MAX_LENGTHS = [512, 384, 256, 128]
THREAD_COUNTS = sorted({1, 2, 4, os.cpu_count() or 4})


def fetch_candidates(query: str) -> list[str]:
    """The same pool confidence.retrieve() would build. Read-only."""
    results = get_collection().query(
        query_embeddings=[embed_query(query)], n_results=N_CANDIDATES
    )
    if not results["ids"] or not results["ids"][0]:
        sys.exit("No candidates returned — is chroma_db populated?")
    return results["documents"][0]


def describe(scores) -> tuple[float, int, float]:
    """(top-1 score, index of top-1, top1 / mean of the rest).

    Those three are what the confidence math actually consumes: abs_signal
    reads the top score, sep_signal reads the top against the rejected pool.
    If they hold steady, a setting is safe. If they move, the calibration moved
    with them.
    """
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best = ranked[0]
    rest = [scores[i] for i in ranked[1:]]
    mean_rest = sum(rest) / len(rest) if rest else 0.0
    ratio = scores[best] / mean_rest if mean_rest > 1e-9 else float("inf")
    return float(scores[best]), best, ratio


def main() -> None:
    query = " ".join(sys.argv[1:]) or "who is the HOD of ECE"

    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  torch                : {torch.__version__}")
    print(f"  os.cpu_count()       : {os.cpu_count()}")
    print(f"  torch.get_num_threads: {torch.get_num_threads()}")
    print(f"  interop threads      : {torch.get_num_interop_threads()}")
    print(f"  OMP_NUM_THREADS      : {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"  MKL_NUM_THREADS      : {os.environ.get('MKL_NUM_THREADS', '(unset)')}")

    if torch.get_num_threads() == 1 and (os.cpu_count() or 1) > 1:
        print("\n  >> torch is single-threaded on a multi-core machine.")
        print("     That is very likely most of your 1027ms/pair, and the fix")
        print("     is free: torch.set_num_threads(N). See the sweep below.")

    print(f'\nQuery: "{query}"')
    docs = fetch_candidates(query)

    model = CrossEncoder(MODEL_NAME)
    tokenizer = model.tokenizer
    baseline_max_length = getattr(model, "max_length", None)

    # --- how long are these pairs, really? ---
    lengths = [
        len(tokenizer(query, doc, truncation=False)["input_ids"]) for doc in docs
    ]
    lengths.sort()
    print("\n" + "=" * 72)
    print("SEQUENCE LENGTHS (tokens per query+chunk pair, before truncation)")
    print("=" * 72)
    print(f"  candidates      : {len(lengths)}")
    print(f"  shortest        : {lengths[0]}")
    print(f"  median          : {lengths[len(lengths) // 2]}")
    print(f"  longest         : {lengths[-1]}")
    print(f"  model max_length: {baseline_max_length}")
    over = sum(1 for n in lengths if n > (baseline_max_length or 512))
    print(f"  over the cap    : {over} of {len(lengths)} (these get truncated)")
    print("\n  Batches pad to the longest member, so if even one chunk is long,")
    print("  every pair in the batch costs as if it were that long.")

    pairs = [(query, doc) for doc in docs]

    # Warm up. The first predict() call initialises kernels and would otherwise
    # be charged to whichever setting happened to run first.
    model.predict(pairs)

    # --- 1. thread sweep: pure speed, scores untouched ---
    print("\n" + "=" * 72)
    print("THREAD SWEEP  (scores are IDENTICAL — threading changes nothing but speed)")
    print("=" * 72)
    original_threads = torch.get_num_threads()
    print(f"  {'threads':>8}  {'total':>8}  {'per pair':>9}  {'vs now':>7}")
    baseline_thread_time = None
    for n in THREAD_COUNTS:
        try:
            torch.set_num_threads(n)
        except Exception as exc:
            print(f"  {n:>8}  (could not set: {exc})")
            continue
        started = time.perf_counter()
        model.predict(pairs)
        elapsed = time.perf_counter() - started
        if n == original_threads:
            baseline_thread_time = elapsed
        speedup = (
            f"{baseline_thread_time / elapsed:.2f}x"
            if baseline_thread_time else "-"
        )
        print(f"  {n:>8}  {elapsed:>7.2f}s  {elapsed / len(pairs) * 1000:>8.0f}ms"
              f"  {speedup:>7}")
    torch.set_num_threads(original_threads)

    # --- 2. max_length sweep: speed AGAINST score fidelity ---
    print("\n" + "=" * 72)
    print("MAX_LENGTH SWEEP  (faster, but watch the score columns)")
    print("=" * 72)
    print(f"  {'max_len':>7}  {'total':>8}  {'per pair':>9}  {'top1 score':>11}"
          f"  {'top1/mean':>10}  {'top1 moved':>11}")

    reference = None
    timings = []
    for max_length in MAX_LENGTHS:
        model.max_length = max_length
        started = time.perf_counter()
        scores = model.predict(pairs)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)

        top_score, top_index, ratio = describe(scores)
        if reference is None:
            reference = (top_score, top_index, ratio)
            moved = "baseline"
        else:
            moved = "NO" if top_index == reference[1] else "YES — different chunk"

        print(f"  {max_length:>7}  {elapsed:>7.2f}s  {elapsed / len(pairs) * 1000:>8.0f}ms"
              f"  {top_score:>11.4f}  {ratio:>10.1f}  {moved:>11}")

    if baseline_max_length is not None:
        model.max_length = baseline_max_length

    # If setting the attribute had no effect, every row above is the same run.
    spread = (max(timings) - min(timings)) / max(timings) if timings else 0
    if spread < 0.05:
        print("\n  >> All max_length rows timed within 5% of each other.")
        print("     Your sentence-transformers version probably ignores a")
        print("     max_length set after construction — the numbers above are")
        print("     four runs of the same setting. Pass it to the constructor")
        print("     instead: CrossEncoder(MODEL_NAME, max_length=N).")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  The THREAD sweep is free money — identical scores, no")
    print("  recalibration. If a higher thread count is faster, set it in")
    print("  reranker.py and you are done.")
    print()
    print("  The MAX_LENGTH sweep is a trade. 'top1 moved: NO' with a steady")
    print("  top1 score means truncation is not changing the answer FOR THIS")
    print("  QUERY — it says nothing about the other 40. Before adopting a")
    print("  shorter max_length, re-run calibrate.py and check the bands still")
    print("  land where confidence_V2.py expects them.")


if __name__ == "__main__":
    main()