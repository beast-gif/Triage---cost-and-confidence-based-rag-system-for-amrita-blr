"""
spell_fix.py — correct typos against the corpus's OWN vocabulary.

THE PROBLEM
-----------
A query about badminton returned nothing when the word was misspelled, and the
right answer when it was not. The reranker has no notion of "close enough" —
a typo is simply a different token.

WHY NOT A NORMAL SPELL CHECKER
------------------------------
A dictionary-based corrector would destroy exactly the terms this corpus
depends on. None of these are English words:

    Amrita, Vidyapeetham, Shikshak, Amritapuri, Vinodhini, Chittawadigi,
    Gopalakrishnan, Ramesh, Bengaluru, Aarohan

A general checker "corrects" them into nonsense, breaking the faculty and
campus queries that currently work.

THE APPROACH
------------
Build the dictionary FROM the corpus. Every word appearing in any chunk — web
or uploaded — is by definition spelled correctly for this system's purposes.
Then a query token is corrected only when it is ABSENT from that vocabulary
but close to something in it:

    "badmintan"  not in vocab, 1 edit from "badminton" (in vocab)  -> corrected
    "Vinodhini"  already in vocab                                  -> untouched
    "asdfgh"     not in vocab, nothing close                       -> untouched

The key property: a correction can only ever produce a word the corpus already
contains. It cannot invent anything.

WHAT THIS CANNOT DO
-------------------
Context. "when is the fee dew" leaves "dew" alone, because "dew" is a real
token that may well appear somewhere in the corpus. Only an LLM rewrite would
catch that, at the cost of a call per query.

Self-test:  python spell_fix.py
"""

import json
import os
import re
from collections import Counter

VOCAB_FILE = "corpus_vocab.json"

# Words shorter than this are skipped. At 3-4 characters almost everything is
# within one edit of something else — "fee"/"see", "sem"/"see", "lab"/"lap" —
# so correcting them does more harm than good.
MIN_LENGTH = 5

# Similarity threshold, 0-100, using NORMALIZED EDIT DISTANCE.
#
# The default rapidfuzz scorer (WRatio) is wrong for this job: it includes a
# partial-ratio component that rewards a short string appearing INSIDE a longer
# one, with almost no length penalty. Measured live:
#
#     "badmintan" -> "adm"      (should be "badminton")
#     "asdfgh"    -> "asd"      (should be untouched)
#
# Both are substring matches, not typos. A typo is roughly the SAME LENGTH as
# the word intended, so edit distance is the right measure and fragments must
# be rejected outright.
SCORE_CUTOFF = 82

# A candidate whose length differs from the query token by more than this is
# rejected before scoring. "badmintan" (9) vs "adm" (3) fails on length alone.
MAX_LENGTH_DELTA = 2

# A word must appear at least this often in the corpus to be a correction
# target. Scraped pages contain their own typos and OCR debris; a one-off
# garbage token should not become something queries get corrected TO.
MIN_FREQUENCY = 2

_vocab = None
_word_re = re.compile(r"[a-z]{2,}")


def build_vocabulary(save=True):
    """
    Scan every chunk in both stores and collect the words.

    Called once at startup and after any sync or upload — the vocabulary is
    only as current as the corpus it was built from, so a newly uploaded
    document's terms are unknown until this is rerun.
    """
    counts = Counter()

    for module in ("store", "upload_store"):
        try:
            mod = __import__(module)
            data = mod.get_collection().get(include=["documents"])
            for doc in data["documents"] or []:
                counts.update(_word_re.findall(doc.lower()))
        except Exception as exc:
            print(f"[WARN] could not read {module}: {exc}")

    vocab = sorted(w for w, n in counts.items()
                   if n >= MIN_FREQUENCY and len(w) >= 3)

    if save:
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab, f)

    return vocab


def get_vocabulary(rebuild=False):
    """Cached vocabulary, loaded from disk or rebuilt from the collections."""
    global _vocab
    if _vocab is not None and not rebuild:
        return _vocab

    if not rebuild and os.path.exists(VOCAB_FILE):
        try:
            with open(VOCAB_FILE, encoding="utf-8") as f:
                _vocab = json.load(f)
                return _vocab
        except (json.JSONDecodeError, OSError):
            pass

    _vocab = build_vocabulary()
    return _vocab


def correct_query(query, vocab=None):
    """
    Fix typos in a query. Returns (corrected_query, [(before, after), ...]).

    Only tokens that are (a) long enough, (b) absent from the vocabulary, and
    (c) close to a similarly-sized vocabulary word get changed. Everything else
    passes through untouched — including proper nouns, which are in the
    vocabulary precisely because they appear in the corpus.

    Candidates are filtered BY LENGTH before scoring. Without that guard,
    substring matches win: "badmintan" scored higher against "adm" than against
    "badminton", because rapidfuzz's default scorer rewards a short string
    appearing inside a longer one.
    """
    if not query:
        return query, []

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return query, []          # not installed: degrade to doing nothing

    vocab = vocab if vocab is not None else get_vocabulary()
    if not vocab:
        return query, []

    vocab_set = set(vocab)

    # Bucket by length once, so each lookup only scores plausible candidates.
    # Also makes correction far faster than scanning 16,000 words per token.
    by_length = {}
    for word in vocab:
        by_length.setdefault(len(word), []).append(word)

    changes = []

    def fix(match):
        word = match.group(0)
        lower = word.lower()

        if len(lower) < MIN_LENGTH or lower in vocab_set:
            return word

        candidates = []
        for length in range(len(lower) - MAX_LENGTH_DELTA,
                            len(lower) + MAX_LENGTH_DELTA + 1):
            candidates.extend(by_length.get(length, ()))
        if not candidates:
            return word

        # fuzz.ratio is normalized edit distance — no partial/substring
        # component, so a fragment cannot outrank a genuine near-match.
        hit = process.extractOne(
            lower, candidates, scorer=fuzz.ratio, score_cutoff=SCORE_CUTOFF
        )
        if not hit:
            return word

        replacement = hit[0]
        changes.append((word, replacement))

        # Preserve the original capitalisation shape.
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement

    corrected = re.sub(r"[A-Za-z]+", fix, query)
    return corrected, changes


if __name__ == "__main__":
    print("building vocabulary from both stores...")
    vocab = build_vocabulary()
    print(f"  {len(vocab)} distinct words\n")

    for probe in ("badminton", "semester", "amrita", "vidyapeetham",
                  "chittawadigi", "gymnasium", "deepavali"):
        print(f"  {'in vocab' if probe in vocab else 'MISSING ':<10} {probe}")

    TESTS = [
        "is there a badmintan court",
        "is there a badminton court",       # already correct
        "what is the fee for btech electronis",
        "who is the chairpersn of ECE",
        "when is the semster exam",
        "who is Dr Vinodhini",              # proper noun, must be untouched
        "what is the elgibility criteria",
        "asdfgh qwerty",                    # nonsense, must be untouched
    ]

    print("\n" + "=" * 74)
    print("correction")
    print("=" * 74)
    for query in TESTS:
        fixed, changes = correct_query(query, vocab)
        print(f"\n  in  : {query}")
        if changes:
            print(f"  out : {fixed}")
            for before, after in changes:
                print(f"        {before} -> {after}")
        else:
            print("  out : (unchanged)")

    print("\n" + "=" * 74)
    print("CHECKS")
    print("=" * 74)
    checks = [
        # Both of these failed on the first version: WRatio's partial-ratio
        # component let a short fragment outrank the real word.
        ("'badmintan' corrects to badminton, not a fragment",
         correct_query("badmintan court", vocab)[0].startswith("badminton")),
        ("nonsense is left alone, not turned into a fragment",
         correct_query("asdfgh qwerty", vocab)[0] == "asdfgh qwerty"),
        ("proper nouns in the corpus are untouched",
         correct_query("who is Dr Vinodhini", vocab)[1] == []),
        ("correctly spelled queries are untouched",
         correct_query("is there a badminton court", vocab)[1] == []),
        ("a real typo still gets fixed",
         "eligibility" in correct_query("elgibility criteria", vocab)[0]),
    ]
    for description, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {description}")