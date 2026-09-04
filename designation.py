"""
designation.py — normalize faculty role titles into a controlled vocabulary.

WHY THIS EXISTS
---------------
Diagnostic finding on the query "Chairperson of Electronics and communication":

    #1  Dr. M. Vinodhini    Vice-Chairperson, Electronics and Communication...
    #2  Dr. T. K. Ramesh    HOD, Department of Electronics and Communication...   <- GOLD
    #15 Dr. Vidya H. A.     Chairperson, Department of Electrical and Electronics...

The ECE department labels its head "HOD". The EEE department labels its head
"Chairperson". Same site, same faculty template, inconsistent vocabulary.

So a user asking for the "Chairperson of ECE" has NO strong lexical anchor in
the correct chunk, while the Vice-Chairperson chunk leads with the query's
exact key term. The reranker was not confused — it correctly matched the only
ECE chunk that says "Chairperson" near the front. This is a data-layer bug
(bug #5), not a ranking bug.

Fix: parse the designation out of the text, map it through an alias table to a
controlled vocabulary, and store it as queryable Chroma metadata. Role queries
then stop depending on which word a particular department happened to use.

TWO PARSING TRAPS this module handles:
  1. Substring collision. "Vice-Chairperson" CONTAINS "Chairperson";
     "Associate Professor" CONTAINS "Professor". A naive keyword scan tags the
     deputy as the head. Resolved by greedy longest-non-overlapping matching.
  2. Multi-role chunks. "HOD, ... | Professor, ..." must resolve to the SENIOR
     role, not the first or last one. Resolved by seniority ranks.

Self-test:  python designation.py
"""

import re

# ---------------------------------------------------------------------------
# controlled vocabulary
# ---------------------------------------------------------------------------
# (regex, normalized_value, seniority_rank)
#
# Order in this list does NOT decide precedence — overlap is resolved by span
# length and precedence by rank — but keeping specific variants above their
# own substrings makes the table easier to read and audit.
ALIASES = [
    # --- deputy roles: MUST be able to out-span their own base term ---
    (r"vice[\s\-]*chair(?:person|man|)\b", "deputy_head", 90),
    (r"deputy\s+(?:hod|head\s+of\s+(?:the\s+)?department)\b", "deputy_head", 90),
    (r"associate\s+dean\b", "deputy_head", 90),

    # --- department head: the whole point of this module ---
    (r"chair(?:person|man)\b", "head_of_department", 100),
    (r"\bhod\b", "head_of_department", 100),
    (r"head\s+of\s+(?:the\s+)?department\b", "head_of_department", 100),

    # --- school / institute leadership ---
    (r"\bdean\b", "dean", 95),
    (r"\bdirector\b", "director", 95),
    (r"\bprincipal\b", "principal", 95),

    # --- academic ranks (specific before general) ---
    (r"assistant\s+professor\b", "assistant_professor", 40),
    (r"associate\s+professor\b", "associate_professor", 50),
    (r"\bprofessor\s+emeritus\b", "professor_emeritus", 55),
    (r"\bprofessor\b", "professor", 60),
    (r"\blecturer\b", "lecturer", 30),

    # --- non-teaching ---
    (r"executive\s+assistant\b", "staff", 10),
    (r"\b(?:lab\s+)?technician\b", "staff", 10),
    (r"administrative\s+(?:officer|assistant)\b", "staff", 10),
]

HEAD_ROLES = {"head_of_department"}

_COMPILED = [(re.compile(p, re.I), norm, rank) for p, norm, rank in ALIASES]


# ---------------------------------------------------------------------------
# core parsing
# ---------------------------------------------------------------------------
def find_designations(text):
    """
    Return every designation found, as a list of dicts, with overlaps resolved.

    Greedy longest-non-overlapping: collect all candidate spans, sort by
    (earliest start, then longest span), and let each accepted span block any
    later span that overlaps it.

    This is what stops "Vice-Chairperson" from also registering as
    "Chairperson": the vice- span starts earlier AND is longer, so it wins the
    position and blocks the inner match.
    """
    if not text:
        return []

    spans = []
    for rx, norm, rank in _COMPILED:
        for m in rx.finditer(text):
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "normalized": norm,
                "rank": rank,
                "raw": m.group(0),
            })

    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

    accepted, occupied = [], []
    for s in spans:
        overlaps = any(
            not (s["end"] <= o_start or s["start"] >= o_end)
            for o_start, o_end in occupied
        )
        if overlaps:
            continue
        occupied.append((s["start"], s["end"]))
        accepted.append(s)

    return accepted


def primary_designation(text):
    """
    The single most senior designation in the chunk, or None.

    Ramesh  : 'HOD ... | Professor ...'            -> head_of_department (100 > 60)
    Vinodhini: 'Vice-Chairperson ... | Asst Prof'  -> deputy_head        (90 > 40)
    """
    found = find_designations(text)
    if not found:
        return None
    return max(found, key=lambda s: (s["rank"], -s["start"]))


# ---------------------------------------------------------------------------
# chunk provenance
# ---------------------------------------------------------------------------
def parse_chunk_id(chunk_id):
    """
    chunk_id format is '{url}::{chunk_type}::{hash}', so provenance is
    recoverable from the id alone — no dependence on what store.py happens to
    copy into Chroma metadata.

    Returns (source_url, chunk_type). Either may be None if the id is
    malformed.
    """
    if not chunk_id or "::" not in chunk_id:
        return None, None
    parts = chunk_id.rsplit("::", 2)
    if len(parts) != 3:
        return None, None
    return parts[0], parts[1]


FACULTY_URL_MARKER = "/faculty/"
PERSON_CHUNK_TYPES = {"faculty_card"}

# A designation only counts as THIS CHUNK'S designation if it appears near the
# start, right after a name. Measured over the 11 chunks the URL gate let
# through as department heads:
#
#   6 true records   -> designation at offset 15..25
#   5 false positives-> designation buried deep in the body
#
# The false positives were Achievements / Talks Delivered / Publications
# chunks off real faculty profile pages, e.g.
#   'Talks Delivered * Recent Trends in E-Mobility ... HOD ...'
#   'Parameter Tuning For Improved Dynamic Response ... Chairperson ...'
#
# The instructive one was
#   'Dr. Amudha J. Professor, School of Computing ... chairperson ...'
# which DOES start with a name, but max(rank) reached hundreds of characters
# past 'Professor' to grab a 'chairperson' mention. Taking the highest-ranked
# designation ANYWHERE in the chunk was the bug; ranking only among leading
# designations gives her 'professor', which is what her chunk actually says.
#
# This also matches the site's own convention: Ramesh's card reads
# 'Chairperson Head of Department Professor' — senior role first.
LEADING_WINDOW = 150
_NAME_PATTERN = re.compile(r"\b(?:dr|mr|mrs|ms|prof|shri|smt)\.?\s", re.I)


def leading_designations(text):
    """
    Designations that belong to the PERSON this chunk is about, as opposed to
    role words mentioned anywhere in their bio, publications or talks.

    Requires both:
      1. a name honorific (Dr./Prof./Ms./...) inside the leading window
      2. the designation after that name, still inside the window

    NOTE: find_designations() stays unfiltered on purpose — infer_route() in
    confidence.py runs it on QUERIES, which are short and contain no 'Dr.'.
    """
    name_match = _NAME_PATTERN.search(text[:LEADING_WINDOW])
    if not name_match:
        return []
    return [
        d for d in find_designations(text)
        if name_match.start() < d["start"] < LEADING_WINDOW
    ]


def is_person_record(source_url=None, chunk_type=None):
    """
    Should this chunk be parsed for designations at all?

    WHY THIS GATE EXISTS
    --------------------
    The first dry run over the live collection tagged these as department heads:

        "grievances box : a grievances box"
        "and steering committee chair of conference dr. j. amudha"
        "cse. ms. meena belwal ms. juhi r. srivastava ms. akanchha tiwari"

    University pages are full of committee chairs, conveners and ex-officio
    members. Head-role terms are NOT rare in general prose, so scanning every
    chunk produces false positives that would corrupt any metadata filter.

    Provenance is a far more reliable signal than text heuristics: a chunk is a
    person record if it came off a faculty profile URL or was extracted as a
    faculty card.
    """
    if chunk_type in PERSON_CHUNK_TYPES:
        return True
    if source_url and FACULTY_URL_MARKER in source_url.lower():
        return True
    return False


# ---------------------------------------------------------------------------
# department (best-effort, secondary, fail-closed)
# ---------------------------------------------------------------------------
_DEPT_STRIP_LEADING_OF = re.compile(r"^\s*of\s+(?:the\s+)?", re.I)
_DEPT_STRIP_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:department|dept\.?|dep\.?|school)\s+of\s+", re.I
)
_DEPT_STRIP_SUFFIX = re.compile(r"\s+engineering\s*$", re.I)
# 'Computer Science and Engineering' -> strip 'Engineering' -> 'computer science
# and' -> a dangling conjunction. Observed live as
# 'of the department of computer science and'.
_DEPT_STRIP_CONJUNCTION = re.compile(r"\s+(?:and|&|,)\s*$", re.I)

# Same department, two spellings — the department-level equivalent of the
# HOD/Chairperson problem. Observed live: 'eee' and 'electrical and
# electronics' both appeared as distinct heads' departments.
DEPARTMENT_ALIASES = {
    "eee": "electrical and electronics",
    "ece": "electronics and communication",
    "cse": "computer science",
    "cs": "computer science",
    "computer science and engineering": "computer science",
    "ai": "artificial intelligence",
    "aie": "artificial intelligence",
    "me": "mechanical",
    "mech": "mechanical",
    "ce": "civil",
}

# Where a department name ENDS. Commas alone are not enough: extractor.py
# builds faculty cards with card_tag.get_text(" ", strip=True), which joins on
# SPACES — the commas were markup boundaries and are simply not in the text.
# That is why the live run produced
#   'head of department professor profile: https://.../tk-ramesh/'
# as Ramesh's department: there was no comma to split on, so it ran to the end.
_DEPT_STOP = re.compile(
    r"[,|;]"
    r"|\bschool\s+of\b"
    r"|\bprofile\s*:"
    r"|\bqualification\s*:"
    r"|\bemail\b|\bph\s*:"
    r"|\bamrita\b"
    r"|\b(?:bengaluru|bangalore|coimbatore|amritapuri|chennai|kochi|mysuru|nagercoil|faridabad)\b",
    re.I,
)

_DEPT_MAX_WORDS = 7
_DEPT_REJECT = re.compile(r"https?://|[:*]|\d{4}", re.I)


def _valid_department(s):
    """
    Fail closed. An empty department field is honest; a wrong one silently
    poisons every filter built on top of it.
    """
    if not s:
        return False
    if len(s.split()) > _DEPT_MAX_WORDS:
        return False
    if _DEPT_REJECT.search(s):
        return False
    if not re.search(r"[a-z]", s):
        return False
    return True


def normalize_department(raw):
    """
    'Department of Electronics and Communication'      -> 'electronics and communication'
    'Electronics and Communication Engineering'        -> 'electronics and communication'
    'of the Department of Computer Science and Engg'   -> 'computer science'
    'EEE'                                              -> 'electrical and electronics'

    Strip order matters:
      1. leading 'of the'   (the designation regex ends before it, as in
                             'Chairperson OF THE Department of ...')
      2. 'department of'
      3. trailing 'engineering'
      4. dangling conjunction left behind by step 3
      5. alias canonicalization

    Returns None (not garbage) if the result fails validation.
    """
    if not raw:
        return None
    s = " ".join(raw.split())
    s = _DEPT_STRIP_LEADING_OF.sub("", s)
    s = _DEPT_STRIP_PREFIX.sub("", s)
    s = _DEPT_STRIP_SUFFIX.sub("", s)
    s = _DEPT_STRIP_CONJUNCTION.sub("", s)
    s = s.strip(" ,|-.").lower()
    s = DEPARTMENT_ALIASES.get(s, s)
    return s if _valid_department(s) else None


def _truncate_at_designation(segment):
    """
    If the text following a designation is ANOTHER designation, there is no
    department there. Observed live as department values of 'professor',
    'head of department professor' and 'assistant professor (sl. gd.)'.
    """
    found = find_designations(segment)
    if not found:
        return segment
    first = min(found, key=lambda s: s["start"])
    return segment[:first["start"]]


def department_for(text, desig=None):
    """
    Take the text immediately following the designation, up to the first
    structural boundary.

        'Dr. T. K. Ramesh HOD, Department of Electronics and Communication, School...'
                              ^ lstrip punctuation      ^ stop at comma

        'Dr. T. K. Ramesh HOD Department of Electronics and Communication School of...'
                             ^ lstrip                  ^ stop at 'School of'
    """
    desig = desig or primary_designation(text)
    if not desig:
        return None

    tail = text[desig["end"]:desig["end"] + 200]
    # Strip leading punctuation FIRST, or a comma sitting immediately after the
    # designation matches _DEPT_STOP at position 0 and yields an empty segment.
    tail = tail.lstrip(" \t\n\r-–—:,|")

    stop = _DEPT_STOP.search(tail)
    segment = tail[:stop.start()] if stop else tail
    segment = _truncate_at_designation(segment)
    return normalize_department(segment)


# ---------------------------------------------------------------------------
# the metadata Chroma will store
# ---------------------------------------------------------------------------
def designation_metadata(text, source_url=None, chunk_type=None):
    """
    Chroma metadata values must be SCALARS (str / int / float / bool) — lists
    are rejected — so the full role set is stored as a pipe-joined string.

    Returns keys that are always present, so the schema stays uniform across
    all chunks and `where` filters behave predictably:

        designation        'head_of_department' | '' if none
        designation_raw    'HOD'                | ''
        designations_all   'head_of_department|professor'   (ALPHABETICAL,
                           not seniority-ordered — read `designation` for that)
        is_department_head True / False
        department         'electronics and communication' | ''

    If source_url/chunk_type are supplied and the chunk is not a person record,
    all fields come back empty. Pass both. Omitting them falls back to parsing
    everything, which the live dry run showed produces false positives.
    """
    if (source_url is not None or chunk_type is not None) and not is_person_record(
        source_url, chunk_type
    ):
        return EMPTY_DESIGNATION.copy()

    found = leading_designations(text)
    if not found:
        return EMPTY_DESIGNATION.copy()

    primary = max(found, key=lambda s: (s["rank"], -s["start"]))
    all_norm = sorted({s["normalized"] for s in found})
    dept = department_for(text, primary)

    return {
        "designation": primary["normalized"],
        "designation_raw": primary["raw"],
        "designations_all": "|".join(all_norm),
        "is_department_head": primary["normalized"] in HEAD_ROLES,
        "department": dept or "",
    }


EMPTY_DESIGNATION = {
    "designation": "",
    "designation_raw": "",
    "designations_all": "",
    "is_department_head": False,
    "department": "",
}


# ---------------------------------------------------------------------------
# QUERY-SIDE helpers — used by confidence.py, not by extraction
# ---------------------------------------------------------------------------
# Which school each department belongs to.
#
# WHY THIS MATTERS
# ----------------
# Not every department lists its own head. Measured on the live corpus, only
# three do:
#     ECE      -> Dr. T. K. Ramesh      (HOD)
#     EEE      -> Dr. Vidya H. A.       (Chairperson)
#     English  -> Dr. Smita Sail        (Head of Department)
#
# For the rest, the SCHOOL PRINCIPAL is the answer:
#     Mechanical          -> Prof. Sriram Devanathan  (Principal, School of Engineering)
#     CSE / AI / AI&DS    -> Dr. Gopalakrishnan E. A. (Principal, School of Computing)
#
# There are TWO different Principals, so returning the wrong one is a wrong
# answer, not a near miss. The department decides which school, and the school
# decides which Principal.
DEPARTMENT_SCHOOL = {
    # School of Engineering
    "electronics and communication": "engineering",
    "ece": "engineering",
    "electrical and electronics": "engineering",
    "eee": "engineering",
    "electrical and computer": "engineering",
    "mechanical": "engineering",
    "english": "engineering",
    "robotics": "engineering",
    # School of Computing
    "computer science and engineering": "computing",
    "computer science": "computing",
    "cse": "computing",
    "artificial intelligence and data science": "computing",
    "artificial intelligence": "computing",
    "data science": "computing",
    "computing": "computing",
    "aids": "computing",
    "ai": "computing",
}

SCHOOL_NAMES = {
    "engineering": "School of Engineering",
    "computing": "School of Computing",
}

# Which SCHOOL a query is asking about, for principal queries. Distinct from
# DEPARTMENT_SCHOOL above: that maps a department to its school, this matches
# school names appearing directly in a query.
#
# Longest first. 'robotics and ai' MUST be tested before 'ai', because it
# belongs to Engineering despite containing "AI" — a naive substring match
# sends it to the Computing principal, which is the wrong person.
SCHOOL_PATTERNS = [
    # --- Computing: the AI-related schools share one principal ---
    (r"school\s+of\s+artificial\s+intelligence", "computing"),
    (r"school\s+of\s+computing", "computing"),
    (r"\bartificial\s+intelligence\s+and\s+data\s+science\b", "computing"),
    (r"\bcomputer\s+science\b", "computing"),
    (r"\bcomputing\b", "computing"),
    (r"\bcse\b", "computing"),
    (r"\baids\b", "computing"),

    # --- Engineering ---
    (r"school\s+of\s+engineering", "engineering"),
    (r"\bengineering\s+school\b", "engineering"),
]

_SCHOOL_PATTERNS = [(re.compile(p, re.I), school) for p, school in SCHOOL_PATTERNS]

# 'robotics and ai' is an ENGINEERING program whose name contains "AI". Checked
# before the school patterns so it cannot be dragged into Computing.
_ENGINEERING_TRAPS = re.compile(
    r"\brobotics(\s+and\s+ai)?\b|\bmechanical\b|\belectrical\b|"
    r"\belectronics\b|\bcivil\b|\benglish\b|\bchemistry\b|\bphysics\b",
    re.I,
)

_PRINCIPAL_RE = re.compile(r"\bprincipal\b", re.I)

# Bare "who is the principal" with no school named. There are two, so this has
# to resolve somehow — Engineering is the larger school and the default.
DEFAULT_PRINCIPAL_SCHOOL = "engineering"


def wants_principal(query):
    """Is this query asking who a PRINCIPAL is?"""
    return bool(_PRINCIPAL_RE.search(query or ""))


def school_in_query(query):
    """
    Which school's principal is being asked about.

    Returns 'engineering' or 'computing' — never None, because a bare
    "who is the principal" still has to resolve to somebody, and Engineering is
    the default.

    Engineering traps are checked FIRST: "principal of robotics and ai" names an
    Engineering program, and matching "ai" would otherwise route it to the
    Computing principal. Two different people, so that is a wrong answer rather
    than a near miss.
    """
    if not query:
        return DEFAULT_PRINCIPAL_SCHOOL

    if _ENGINEERING_TRAPS.search(query):
        return "engineering"

    for pattern, school in _SCHOOL_PATTERNS:
        if pattern.search(query):
            return school

    return DEFAULT_PRINCIPAL_SCHOOL

# Longest phrases first, so 'computer science and engineering' wins over
# 'computer science', and 'artificial intelligence and data science' over 'ai'.
_DEPT_QUERY_PATTERNS = sorted(
    DEPARTMENT_SCHOOL.keys(), key=len, reverse=True
)


def department_in_query(query):
    """
    Which department is this query asking about? Returns a normalized name or
    None. Word-boundary matched so short aliases like 'ai' do not fire inside
    'maintenance'.
    """
    if not query:
        return None
    for phrase in _DEPT_QUERY_PATTERNS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", query, re.I):
            return phrase
    return None


def school_for_department(department):
    """'mechanical' -> 'engineering';  'cse' -> 'computing';  unknown -> None."""
    if not department:
        return None
    return DEPARTMENT_SCHOOL.get(department.lower())


# Query phrasings that mean "who runs this department" but that the chunk-side
# alias table misses. ALIASES needs 'head of DEPARTMENT' because chunk text
# writes the full title, but a user types 'head of computer science'.
_QUERY_HEAD_PATTERNS = [
    re.compile(r"\bhead\s+of\b", re.I),
    re.compile(r"\bwho\s+heads\b", re.I),
    re.compile(r"\bheaded\s+by\b", re.I),
    re.compile(r"\bin\s+charge\s+of\b", re.I),
]


def wants_department_head(query):
    """
    Is this query asking WHO HEADS a department?

    Reuses the chunk-side alias table so 'HOD', 'Chairperson', 'Chairman' and
    'Head of Department' all trigger, plus the query-only patterns above for
    phrasings like 'head of computer science'.

    A deputy role in the query VETOES this. 'Vice-Chairperson of ECE' must not
    filter to department heads — that filter would exclude the very person
    being asked about.
    """
    found = find_designations(query or "")

    if any(f["normalized"] == "deputy_head" for f in found):
        return False

    if any(f["normalized"] in HEAD_ROLES for f in found):
        return True

    return any(rx.search(query or "") for rx in _QUERY_HEAD_PATTERNS)


def departments_match(query_dept, chunk_dept):
    """
    Loose match between the department named in a query and the one stored on a
    chunk. Substring either way, because the stored form is normalized
    ('electronics and communication') while a query may use an alias ('ECE').
    """
    if not query_dept or not chunk_dept:
        return False
    a, b = query_dept.lower().strip(), chunk_dept.lower().strip()
    if a == b:
        return True
    # resolve aliases through the canonical department table
    a_canon = DEPARTMENT_ALIASES.get(a, a)
    b_canon = DEPARTMENT_ALIASES.get(b, b)
    return a_canon == b_canon or a_canon in b_canon or b_canon in a_canon


# ---------------------------------------------------------------------------
# self-test on real chunk text from the diagnostic
# ---------------------------------------------------------------------------
SAMPLES = [
    ("GOLD (ECE head, says 'HOD')",
     "Dr. T. K. Ramesh HOD, Department of Electronics and Communication, "
     "School of Engineering, Bengaluru | Professor, Department of Electronics "
     "and Communication, School of Engineering, Bengaluru"),

    ("GOLD as a faculty CARD (no commas — get_text(' ') joins on spaces)",
     "Faculty Dr. T. K. Ramesh HOD Department of Electronics and Communication "
     "School of Engineering Bengaluru Professor "
     "Profile: https://www.amrita.edu/faculty/tk-ramesh/"),

    ("DISTRACTOR (ECE deputy, says 'Vice-Chairperson')",
     "Dr. M. Vinodhini Vice-Chairperson, Electronics and Communication "
     "Engineering, School of Engineering, Bengaluru | Assistant Professor "
     "(Sl. Gd.), Electronics and Communication Engineering"),

    ("EEE head, says 'Chairperson'",
     "Dr. Vidya H. A. Chairperson, Department of Electrical and Electronics "
     "Engineering, School of Engineering, Bengaluru | Professor, Department "
     "of Electrical and Electronics Engineering"),

    ("plain professor",
     "Dr. Dhanesh G. Kurup Professor, Department of Electronics and "
     "Communication, School of Engineering, Bengaluru Qualification: B.Tech., M.E, Ph.D"),

    ("non-teaching staff",
     "Faculty B. H. Bhagyavathi Executive Assistant, Electronics and "
     "Communication Engineering, School of Engineering, Bengaluru"),

    ("no designation at all (program overview page)",
     "Overview B. Tech. in Electronics and Communication Engineering is a four "
     "year professional undergraduate program offered by Amrita School of Engineering"),

    # --- false positives observed in the live dry run over 6057 chunks ---
    ("FALSE POSITIVE: committee prose",
     "The library committee has faculty-representatives and student-nominees "
     "as members; the librarian is the ex-officio convener of the committee. "
     "Chairperson of the grievances box : a grievances box"),

    ("FALSE POSITIVE: conference listing",
     "Organising and steering committee chair of conference Dr. J. Amudha, "
     "Chairperson from January 2019- July 2020"),
]

PERSON_URL = "https://www.amrita.edu/faculty/tk-ramesh/"
PROSE_URL = "https://www.amrita.edu/school/engineering/bengaluru/library/"

if __name__ == "__main__":
    print("=" * 78)
    print("designation.py self-test — real chunk text from the ECE diagnostic")
    print("=" * 78)

    for label, text in SAMPLES:
        md = designation_metadata(text)
        print(f"\n{label}")
        print(f"  designation      : {md['designation'] or '(none)'}")
        print(f"  designation_raw  : {md['designation_raw'] or '(none)'}")
        print(f"  designations_all : {md['designations_all'] or '(none)'}")
        print(f"  is_department_head: {md['is_department_head']}")
        print(f"  department       : {md['department'] or '(none)'}")

    print("\n" + "=" * 78)
    print("KEY ASSERTIONS")
    print("=" * 78)
    ramesh = designation_metadata(SAMPLES[0][1], PERSON_URL, "text")
    ramesh_card = designation_metadata(SAMPLES[1][1], PERSON_URL, "faculty_card")
    vinodhini = designation_metadata(SAMPLES[2][1], PERSON_URL, "text")
    vidya = designation_metadata(SAMPLES[3][1], PERSON_URL, "text")

    committee = designation_metadata(SAMPLES[7][1], PROSE_URL, "text")
    conference = designation_metadata(SAMPLES[8][1], PROSE_URL, "text")

    checks = [
        ("Ramesh 'HOD' normalizes to head_of_department",
         ramesh["designation"] == "head_of_department"),
        ("Vidya 'Chairperson' normalizes to the SAME value",
         vidya["designation"] == "head_of_department"),
        ("Vinodhini 'Vice-Chairperson' does NOT become head",
         vinodhini["designation"] == "deputy_head"),
        ("Ramesh outranks his own 'Professor' second role",
         ramesh["designation"] != "professor"),
        ("Ramesh and Vinodhini share a department string",
         ramesh["department"] == vinodhini["department"] != ""),
        ("EEE head is a DIFFERENT department from ECE head",
         vidya["department"] != ramesh["department"]),
        # --- regressions from the live 6057-chunk dry run ---
        ("CARD form (no commas) gets a clean department, not a URL",
         ramesh_card["department"] == "electronics and communication"),
        ("CARD and profile-page forms agree",
         ramesh_card["department"] == ramesh["department"]),
        ("committee prose is GATED OUT (not a faculty URL)",
         committee["is_department_head"] is False),
        ("conference listing is GATED OUT",
         conference["is_department_head"] is False),
        ("ungated committee prose still yields NO garbage department",
         not designation_metadata(SAMPLES[7][1])["department"]),
        ("ungated conference listing yields NO garbage department",
         not designation_metadata(SAMPLES[8][1])["department"]),
        # --- department bugs found in the 11-head dry run ---
        ("job title after designation -> EMPTY, not 'professor'",
         normalize_department("Professor Profile: https://x") is None),
        ("'head of department professor' -> EMPTY",
         department_for("Dr. X Chairperson Head of Department Professor "
                        "Profile: https://x") is None),
        ("'of the Department of Computer Science and Engineering' -> 'computer science'",
         normalize_department("of the Department of Computer Science and Engineering")
         == "computer science"),
        ("'EEE' canonicalizes to the full EEE department name",
         normalize_department("EEE") == "electrical and electronics"),
        ("'Dept. of EEE' and 'Electrical and Electronics Engineering' agree",
         normalize_department("Dept. of EEE")
         == normalize_department("Electrical and Electronics Engineering")),
        # --- false positives from the 11-head list ---
        ("Achievements prose is NOT a head",
         designation_metadata(
             "Achievements * Leading a Thrust Area Group on Artificial "
             "Intelligence for health and well-being. Chairperson of the group.",
             PERSON_URL, "text")["is_department_head"] is False),
        ("Talks Delivered prose is NOT a head",
         designation_metadata(
             'Talks Delivered * Recent Trends in E-Mobility", as a resource '
             "person in the Webinar organized by HOD and Thrust Area Group",
             PERSON_URL, "text")["is_department_head"] is False),
        ("publication title is NOT a head",
         designation_metadata(
             "Parameter Tuning For Improved Dynamic Response of Indirect Stator "
             "Flux Oriented Induction Motor Drives Cite this Research "
             "Publication : Dr. Rashmi M. R. Chairperson",
             PERSON_URL, "text")["is_department_head"] is False),
        ("Amudha's Professor chunk resolves to professor, not head",
         designation_metadata(
             "Dr. Amudha J. Professor, School of Computing, Bengaluru "
             "Qualification: Ph.D j_amudha@blr.amrita.edu ORCID ID Google "
             "Scholar Profile Scopus Author ID Research Interests " + "x " * 40
             + " chairperson of something",
             PERSON_URL, "text")["designation"] == "professor"),
        ("real card record still tagged head",
         designation_metadata(
             "Faculty Dr. T. K. Ramesh Chairperson Head of Department Professor "
             "Profile: https://www.amrita.edu/faculty/tk-ramesh/",
             PERSON_URL, "faculty_card")["is_department_head"] is True),
    ]
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    print(f"\n  ECE department string -> '{ramesh['department']}'")
    print(f"  ECE card department   -> '{ramesh_card['department']}'")
    print(f"  EEE department string -> '{vidya['department']}'")

    print("\n  parse_chunk_id sanity:")
    demo = "https://www.amrita.edu/faculty/m-vinodhini/::text::172787ebd566e62d"
    print(f"    {parse_chunk_id(demo)}")