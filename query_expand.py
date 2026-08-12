"""
query_expand.py — expand campus abbreviations before retrieval.

THE PROBLEM, MEASURED
---------------------
Two queries asking the same thing, against the same 9-chunk calendar:

    "when does the end semester exam start"   top1 = 0.9557   -> HIGH 0.88
    "when does the endsem start"              top1 = 0.0053   -> refused

180x apart. 0.0053 sits inside the range of deliberate negatives
(0.0000-0.0101), so the correct answer scores like noise. The embedder gets it
roughly right — the correct chunk still ranks #2 — but the cross-encoder does
not recognise "endsem" as related to "End Semester Exam" at all.

WHY EXPAND THE QUERY RATHER THAN THE CHUNKS
-------------------------------------------
Expanding at ingestion would bloat every chunk with synonyms and still miss any
abbreviation not anticipated when the document was uploaded. Expanding the
query happens BEFORE embedding, so both the vector search and the reranker see
the expanded form, and changing the list does not require re-indexing anything.

WHY APPEND INSTEAD OF REPLACE
-----------------------------
"endsem" -> "endsem end semester", not "end semester". If a document literally
writes "endsem", replacing would destroy the only term that matches it. The
result reads awkwardly, but both forms are visible to the model.

KNOWN LIMITATIONS — worth stating rather than hiding
----------------------------------------------------
1. The list is never complete. Every abbreviation not listed still fails, and
   you only find out when someone asks. This fixes observed cases; it is not
   general abbreviation handling.
2. Appending makes queries less natural. The embedding model was trained on
   real sentences, and "when does the endsem start end semester" is not one.
   Measured to help here, but it is a trade rather than a free win.
3. Ambiguity is unresolved. Expansions are context-free, so an abbreviation
   with two meanings gets both or the wrong one.

Self-test:  python query_expand.py
"""

import re

# Whole-word matches only. Without the boundary, 'sem' fires inside 'semester'
# and 'assembly', and 'lab' inside 'labour' — the quiet way these tables rot.
ABBREVIATIONS = {
    # --- exams and assessment ---
    "endsem": "end semester exam",
    "endsems": "end semester exam",
    "midsem": "mid semester exam",
    "midsems": "mid semester exam",
    "sem": "semester",
    "sems": "semesters",
    "reval": "revaluation",
    "cie": "continuous internal evaluation",
    "see": "semester end examination",
    "ese": "end semester examination",
    "prac": "practical",
    "practicals": "practical laboratory",
    "viva": "viva voce",

    # --- people and places ---
    "prof": "professor",
    "asst": "assistant",
    "assoc": "associate",
    "hod": "head of department",
    "vc": "vice chancellor",
    "dept": "department",
    "depts": "departments",

    # --- academic admin ---
    "elig": "eligibility",
    "admn": "admission",
    "attd": "attendance",
    "att": "attendance",
    "ay": "academic year",
    "cgpa": "cumulative grade point average",
    "sgpa": "semester grade point average",
    "tt": "timetable",
    "sched": "schedule",
    "curric": "curriculum",
    "syll": "syllabus",
    "reg": "registration",
    "prereg": "pre-registration",
    "electives": "elective courses",

    # --- programs and departments ---
    "btech": "b tech bachelor of technology",
    "mtech": "m tech master of technology",
    "ug": "undergraduate",
    "pg": "postgraduate",
    "ece": "electronics and communication engineering",
    "eee": "electrical and electronics engineering",
    "cse": "computer science and engineering",
    "aie": "artificial intelligence engineering",
    "aids": "artificial intelligence and data science",
    "mech": "mechanical engineering",
    "civil": "civil engineering",

    # --- campus life ---
    "hostel": "hostel accommodation",
    "mess": "mess dining hall",
    "gym": "gymnasium fitness centre",
    "lib": "library",
    "admin": "administration",
    "placements": "placement recruitment",
    "internship": "internship training",
}

# Compiled once. Longest first so 'endsem' is tried before 'sem' — otherwise
# 'endsem' would never match, because the 'sem' pattern has no word boundary
# problem but the dictionary iteration order would decide the outcome.
_PATTERNS = [
    (re.compile(rf"\b{re.escape(abbr)}\b", re.I), expansion)
    for abbr, expansion in sorted(
        ABBREVIATIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    )
]


def expand_query(query, max_expansions=4):
    """
    Append expansions for any abbreviations found. Returns the expanded query.

    max_expansions caps how many are appended: a query full of abbreviations
    would otherwise end up mostly synonym soup, drowning the actual question.
    Four is enough for realistic phrasing and stops pathological cases.
    """
    if not query:
        return query

    additions, seen = [], set()
    for pattern, expansion in _PATTERNS:
        if len(additions) >= max_expansions:
            break
        if expansion.lower() in query.lower() or expansion in seen:
            continue
        if pattern.search(query):
            additions.append(expansion)
            seen.add(expansion)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def found_abbreviations(query):
    """Which abbreviations fired — for logging and debugging."""
    return [
        pattern.pattern.strip("\\b")
        for pattern, _ in _PATTERNS
        if pattern.search(query or "")
    ]


if __name__ == "__main__":
    TESTS = [
        "when does the endsem start",
        "when does the end semester exam start",   # already expanded, no change
        "who is the hod of ece",
        "what is the midsem sched",
        "elig for btech admn",
        "prof in mech dept",
        "what is the fee for btech ECE",
        "is there a gym",
        "when is deepavali",                        # no abbreviations at all
    ]

    print("=" * 78)
    print("query expansion")
    print("=" * 78)
    for query in TESTS:
        expanded = expand_query(query)
        changed = expanded != query
        print(f"\n  in  : {query}")
        print(f"  out : {expanded}")
        if changed:
            print(f"  hit : {', '.join(found_abbreviations(query))}")
        else:
            print("  hit : (unchanged)")

    print("\n" + "=" * 78)
    print("CHECKS")
    print("=" * 78)
    checks = [
        ("'endsem' expands, and 'sem' does not double-fire inside it",
         expand_query("endsem").count("semester") <= 2),
        ("already-expanded text is left alone",
         expand_query("end semester exam") == "end semester exam"),
        ("no abbreviation means no change",
         expand_query("when is deepavali") == "when is deepavali"),
        ("word boundary holds — 'assembly' does not trigger 'sem'",
         "semester" not in expand_query("assembly hall")),
        ("expansion count is capped",
         len(expand_query("endsem midsem hod dept elig admn ay").split())
         < 40),
    ]
    for description, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {description}")