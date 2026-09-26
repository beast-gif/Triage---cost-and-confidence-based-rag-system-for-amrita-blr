"""
diag_rewrite.py — what does the rewriter actually produce?

Reproduces the exact exchange that failed: a CSE question answered, then the
follow-up "in ece department". Costs one gpt-4o-mini call per case.

    python diag_rewrite.py
"""

from rewrite import needs_rewrite, rewrite_query

# The shape conversations.get_history_for_rewrite() returns: {role, content},
# oldest first, last REWRITE_TURNS exchanges only.
HISTORY = [
    {"role": "user",
     "content": "faculty working on image processing in cse department"},
    {"role": "assistant",
     "content": "The faculty working on image processing in the Department of "
                "Computer Science and Engineering at the Bengaluru campus "
                "includes Dr. Nidhin Prabhakar T. V. and Dr. Tripty Singh."},
]

FOLLOW_UPS = [
    "in ece department",
    "in mechanical department",
    "what about ECE",
]

print("history given to the rewriter:")
for m in HISTORY:
    print(f"  {m['role']:9} {m['content'][:88]}")
print()

for q in FOLLOW_UPS:
    fires = needs_rewrite(q)
    rewritten, changed = rewrite_query(q, HISTORY)
    print("=" * 76)
    print(f"  asked      : {q!r}")
    print(f"  needs_rewrite: {fires}")
    print(f"  rewritten  : {rewritten!r}")
    print(f"  changed    : {changed}")

    # The three things that decide whether the new route can fire.
    from designation import department_in_query
    from confidence import wants_faculty_in_department, _department_tag

    dept = department_in_query(rewritten)
    print(f"  department_in_query      : {dept}")
    print(f"  wants_faculty_in_department: {wants_faculty_in_department(rewritten)}")
    if dept:
        print(f"  -> would search tag       : {_department_tag(dept)}")