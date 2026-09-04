"""
generator.py — turns retrieved chunks into an answer with citations.

CONFIDENCE-AWARE, NOT JUST CONFIDENCE-REPORTING
-----------------------------------------------
The confidence score is not decoration. It changes what happens:

    band = low          -> NO LLM CALL AT ALL. Return a refusal.
    mode = assert       -> answer from the top chunk, cite one source
    mode = disambiguate -> synthesise across chunks, cite several

Skipping the LLM on low confidence is the "cost-aware" half of the project:
a query the retriever already failed on cannot be rescued by spending money on
generation, and calling anyway is how RAG systems produce fluent nonsense.

THE SEMANTIC GUARD
------------------
Two failures measured during calibration are invisible to the confidence score:

    "who is the vice chancellor"     -> a chunk MENTIONS the title but names
                                        nobody. Scored 0.998.
    "placement record for computing" -> returned a publications page at rank 1.

In both cases the retrieved text is lexically right and semantically wrong. No
formula over reranker scores can see that. The LLM can — it reads the chunk and
notices it does not contain an answer. So rule 3 gives it an explicit escape
hatch, and that escape hatch is the ONLY thing catching this class of error.
Do not remove it to make the bot sound more confident.

THE `declined` FLAG
-------------------
When rule 3 fires the model returns a refusal — but the result still carried
high confidence, a full chunk list and citations, because nothing downstream
knew a refusal had happened. That surfaced as a faculty photograph and a source
link rendered above "I don't have that information."

So the refusal is detected and reported. Callers check `declined` rather than
inferring from the band, which stays HIGH precisely because retrieval
succeeded — it was the CONTENT that failed to answer the question.

Usage:
    python generator.py "what is the fee for btech ECE"
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Tried in order; the first that responds wins. OpenRouter model IDs drift —
# verify against https://openrouter.ai/models if all three start failing.
MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

MAX_CHUNK_CHARS = 1400   # caps cost; the longest observed chunk was ~935 words
MAX_TOKENS = 600

REFUSAL = (
    "I don't have information about that in the Amrita Bengaluru admissions "
    "pages I have access to. I can help with fees, curriculum, eligibility, "
    "faculty, placements, hostel and campus facilities for B.Tech programs at "
    "the Bengaluru campus."
)

# Phrasings rule 3 actually produces. Matching on text is brittle — a model
# that says "that is not something these pages establish" slips through — but
# it covers what the current prompt yields, and a miss costs a stray photo
# rather than a wrong answer.
DECLINE_MARKERS = (
    "don't have that information",
    "do not have that information",
    "don't have information",
    "do not have information",
    "doesn't have that information",
    "does not have that information",
    "does not state",
    "does not specify",
    "does not mention",
    "couldn't find",
    "could not find",
    "isn't in the",
    "is not in the",
    "no information about",
)

BASE_RULES = """You are an admissions assistant for Amrita Vishwa Vidyapeetham, Bengaluru campus.

Rules:
1. Answer ONLY from the numbered context below. Never use outside knowledge.
2. Cite the sources you used as [1], [2] etc., inline, right after the claim.
3. If the context mentions the topic but does NOT actually state the answer,
   say you don't have that information. Do not infer, guess, or fill gaps.
   Example: if asked who the Vice Chancellor is and the context only mentions
   the title without naming a person, say you don't have it.
4. If the context is about a different CAMPUS (Coimbatore, Amritapuri,
   Chennai) or a different PROGRAM than the one asked about, say so rather
   than substituting it. This does NOT apply to granularity: if the question
   asks about a hostel or department facility and the context describes it as
   a campus-wide facility, that IS the answer — give it, and note the scope.
5. Be concise. No preamble, no "based on the context".
6. Tables appear as pipe-separated text. Read them carefully and quote the
   specific figure asked for rather than dumping the whole table."""

ASSERT_MODE = """
7. One source clearly matches this question. Answer directly from it."""

DISAMBIGUATE_MODE = """
7. Several sources matched about equally well. Either synthesise them if they
   agree, or state the alternatives explicitly if they differ. Do not silently
   pick one."""


def _declined(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in DECLINE_MARKERS)


def _build_context(chunks):
    """Number the chunks so the model can cite them, and cap their length."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        text = " ".join(c["content"].split())[:MAX_CHUNK_CHARS]
        parts.append(f"[{i}] Source: {c['source_url']}\n{text}")
    return "\n\n".join(parts)


def _call_llm(prompt):
    """Try each model in order. Returns (text, model_used) or (None, None)."""
    import time

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    for model in MODELS:
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            print(f"[timing] generation      = {time.time() - started:.2f}s  ({model})")
            return response.choices[0].message.content.strip(), model
        except Exception as exc:
            print(f"[WARN] {model} failed after "
                  f"{time.time() - started:.2f}s: {exc}")
    return None, None


def generate_answer(query, result):
    """
    query  : the user's question
    result : the dict returned by confidence.score_query()

    Returns the same shape regardless of path, so callers never branch on
    whether the LLM ran.
    """
    band = result.get("band", "low")
    mode = result.get("generator_mode", "disambiguate")
    chunks = result.get("chunks", [])

    # --- cost-aware short circuit ---
    if band == "low" or not chunks:
        return {
            "answer": REFUSAL,
            "sources": [],
            "band": band,
            "generator_mode": mode,
            "llm_called": False,
            "model": None,
            "declined": True,
            "reason": "confidence below threshold — generation skipped",
        }

    prompt = (
        BASE_RULES
        + (ASSERT_MODE if mode == "assert" else DISAMBIGUATE_MODE)
        + f"\n\nContext:\n\n{_build_context(chunks)}"
        + f"\n\nQuestion: {query}\n\nAnswer:"
    )

    answer, model = _call_llm(prompt)
    if answer is None:
        return {
            "answer": "I couldn't reach the answer service just now. "
                      "Please try again in a moment.",
            "sources": [], "band": band, "generator_mode": mode,
            "llm_called": True, "model": None, "declined": True,
            "reason": "all generation models failed",
        }

    declined = _declined(answer)

    # Map [n] citations back to URLs, preserving the order the model used them.
    # Suppressed on a refusal: a model saying "the context does not state who
    # that is [1]" is naming the chunk it CHECKED, not a source for an answer
    # it did not give.
    used, seen = [], set()
    for i, c in enumerate(chunks, start=1):
        if f"[{i}]" in answer and c["source_url"] not in seen:
            seen.add(c["source_url"])
            used.append(c["source_url"])

    return {
        "answer": answer,
        "sources": [] if declined else used,
        "band": band,
        "generator_mode": mode,
        "llm_called": True,
        "model": model,
        "declined": declined,
        "reason": None,
    }


async def answer_query(query):
    """Full pipeline: retrieve -> score -> generate."""
    from confidence import score_query

    result = await score_query(query)
    generated = generate_answer(query, result)
    generated["confidence"] = result.get("final_confidence")
    generated["route"] = result.get("route")
    generated["retrieval"] = result.get("retrieval_details", {})
    # kept so callers (repl.py, /chat) can show what was retrieved without
    # re-running the whole pipeline
    generated["chunks"] = result.get("chunks", [])
    return generated


if __name__ == "__main__":
    import asyncio
    import sys

    query = " ".join(sys.argv[1:]) or "what is the fee for btech ECE"
    out = asyncio.run(answer_query(query))

    print(f'\nQ: {query}')
    print("=" * 68)
    print(out["answer"])
    print("=" * 68)

    if out["sources"]:
        print("\nSources:")
        for url in out["sources"]:
            print(f"  - {url}")

    print(f"\n  confidence : {out['confidence']}  ({out['band']})")
    print(f"  mode       : {out['generator_mode']}")
    print(f"  route      : {out['route'] or 'plain vector search'}")
    print(f"  declined   : {out.get('declined')}")
    print(f"  LLM called : {out['llm_called']}"
          + (f"  ({out['model']})" if out["model"] else ""))
    if out["reason"]:
        print(f"  reason     : {out['reason']}")