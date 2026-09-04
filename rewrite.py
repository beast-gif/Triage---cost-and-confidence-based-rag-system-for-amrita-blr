"""
rewrite.py — turn a follow-up question into a standalone one.

THE PROBLEM
-----------
Retrieval only ever sees the query string. So:

    turn 1  "who is the HOD of ECE"   -> works
    turn 2  "what is his email"       -> embeds a vector about emails and
                                         pronouns, with nothing identifying
                                         Ramesh. Retrieval fails on a question
                                         the system could easily answer.

WHY REWRITE RATHER THAN CONCATENATE OR PASS HISTORY DOWNSTREAM
--------------------------------------------------------------
Concatenating the previous question pollutes the embedding with terms the user
did not ask about, and every confidence constant was fitted on clean
single-intent queries.

Passing history to the GENERATOR only does not help either: retrieval still
fails, so the generator never sees a chunk about Ramesh to work from.

Rewriting produces a standalone query that goes through the existing pipeline
completely unchanged — routing, both stores, confidence, all still calibrated.

WHY IT RUNS BEFORE EVERYTHING ELSE
----------------------------------
retrieve_for() picks its route from the QUERY TEXT. "what about EEE" routes
nowhere; "who is the chairperson of EEE" routes to department_heads. So the
rewrite has to happen before routing, not after.

WHY CONDITIONAL
---------------
Most queries are standalone and need no rewrite. Calling an LLM on every turn
would add latency to a pipeline that is already slow, and a bad rewrite is
worse than none — turning "what about fees" into "what are Ramesh's fees"
breaks a query the user could have gotten right by rephrasing. So a cheap
text check gates the call.

Self-test:  python rewrite.py
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

REWRITE_MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 80
HISTORY_TURNS = 2          # exchanges of context handed to the rewriter

# Signals that a query depends on what came before.
DEPENDENT_PATTERNS = [
    # pronouns with no antecedent in the query itself
    r"\b(he|him|his|she|her|hers|they|them|their|it|its)\b",
    # continuation openers
    r"^\s*(and|also|what about|how about|ok|okay|then|so)\b",
    r"^\s*(what|who|when|where|how|why)\s+about\b",
    # bare demonstratives
    r"\b(that one|this one|the same|those|these)\b",
    # elliptical follow-ups: "and for EEE?", "for mechanical?"
    r"^\s*(and\s+)?for\s+\w+\s*\??$",
]

_DEPENDENT = [re.compile(p, re.I) for p in DEPENDENT_PATTERNS]

# A query this short is usually a fragment rather than a question.
MIN_STANDALONE_WORDS = 3

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone question that makes sense without the conversation.

Rules:
- Replace pronouns and references with the actual names or terms from the conversation.
- Keep the user's intent EXACTLY. Do not answer, expand, or add detail they did not ask for.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question. No preamble, no quotes.

Conversation:
{history}

Latest question: {query}

Standalone question:"""


def needs_rewrite(query):
    """
    Cheap check: does this query depend on earlier context?

    Deliberately errs toward False. A missed rewrite means a follow-up fails
    and the user rephrases; an unnecessary rewrite risks mangling a query that
    was already fine.
    """
    query = (query or "").strip()
    if not query:
        return False

    if any(p.search(query) for p in _DEPENDENT):
        return True

    # A two-word fragment like "for mechanical?" carries no question of its own.
    if len(query.split()) < MIN_STANDALONE_WORDS:
        return True

    return False


def _as_text(content):
    """
    Message content -> plain string.

    Gradio 6 allows content to be a LIST of blocks rather than a string, to
    support multimodal messages. Passing one straight into re.sub raises
    "expected string or bytes-like object, got 'list'".
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return " ".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def _format_history(history, turns=HISTORY_TURNS):
    """
    Gradio-style history -> plain text.

    Accepts either the messages format ([{role, content}, ...]) or a list of
    (user, assistant) pairs, since the two Gradio versions differ.

    Assistant turns are truncated: the rewriter needs the NAMES an answer
    mentioned, not its full text, and a 600-token answer would dominate the
    prompt.
    """
    if not history:
        return ""

    pairs = []
    if isinstance(history[0], dict):
        user_msg = None
        for message in history:
            role = message.get("role")
            text = _as_text(message.get("content"))
            if role == "user":
                user_msg = text
            elif role == "assistant" and user_msg is not None:
                pairs.append((user_msg, text))
                user_msg = None
    else:
        pairs = [(_as_text(u), _as_text(a)) for u, a in history if u is not None]

    lines = []
    for user, assistant in pairs[-turns:]:
        lines.append(f"User: {user}")
        # Strip markdown images and the confidence badge div — neither helps
        # the rewriter and both eat tokens.
        clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", assistant)
        clean = re.sub(r"<div.*?</div>", "", clean, flags=re.S)
        clean = " ".join(clean.split())[:300]
        lines.append(f"Assistant: {clean}")
    return "\n".join(lines)


def rewrite_query(query, history=None):
    """
    Returns (query_for_retrieval, was_rewritten).

    The ORIGINAL query still goes to the generator — the rewrite exists to fix
    retrieval, and echoing a machine-rephrased question back at the user reads
    badly.
    """
    if not history or not needs_rewrite(query):
        return query, False

    context = _format_history(history)
    if not context:
        return query, False

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{
                "role": "user",
                "content": REWRITE_PROMPT.format(history=context, query=query),
            }],
            max_tokens=MAX_TOKENS,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
    except Exception as exc:
        print(f"[WARN] rewrite failed, using original: {exc}")
        return query, False

    # Guard against a runaway rewrite. A model that returns a paragraph, or
    # something unrelated to what was asked, is worse than no rewrite at all —
    # it breaks a query the user could have fixed by rephrasing.
    if not rewritten or len(rewritten.split()) > 40:
        return query, False

    print(f"[rewrite] {query!r} -> {rewritten!r}")
    return rewritten, True


if __name__ == "__main__":
    CHECKS = [
        ("who is the HOD of ECE", False),
        ("what is the fee for btech ECE", False),
        ("when does the end semester exam start", False),
        ("what is his email", True),
        ("what about EEE", True),
        ("and her research interests", True),
        ("for mechanical?", True),
        ("tell me about that one", True),
        ("is there a gym", False),
    ]

    print("=" * 66)
    print("needs_rewrite — does this query depend on earlier context?")
    print("=" * 66)
    for query, expected in CHECKS:
        got = needs_rewrite(query)
        mark = "PASS" if got == expected else "FAIL"
        print(f"  [{mark}] {str(got):<5} {query}")

    HISTORY = [
        {"role": "user", "content": "who is the HOD of ECE"},
        {"role": "assistant",
         "content": "The HOD of Electronics and Communication is "
                    "Dr. T. K. Ramesh [1]."},
    ]

    print("\n" + "=" * 66)
    print("rewrite_query — live LLM call")
    print("=" * 66)
    for query in ("what is his email", "what about EEE", "who is the principal"):
        out, changed = rewrite_query(query, HISTORY)
        print(f"\n  in  : {query}")
        print(f"  out : {out}")
        print(f"  {'rewritten' if changed else 'unchanged — no LLM call'}")