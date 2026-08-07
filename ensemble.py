"""
ensemble.py

The 5-model ensemble classifier. Each "model" independently predicts a
CATEGORY label for the incoming query (which also doubles as the routing
decision, until hierarchical/graph retrieval exist — for now everything
routes to fine-grained regardless of label, but the label is what
confidence scoring and future routing logic will use).

Composition (see earlier discussion on why not all 5 are LLM calls):
  1. Keyword/rule-based classifier      -> free, instant
  2. Embedding-similarity classifier    -> free, near-instant (reuses BGE)
  3-5. Three LLM calls, different prompts -> run CONCURRENTLY (asyncio.gather),
       not sequentially, so total latency ~= 1 call, not 3

All 5 votes get combined with ensemble_agreement() from confidence.py.
"""

import os
import asyncio
from dotenv import load_dotenv
from embedder import embed_query, get_model

load_dotenv()  # reads .env in the current directory and sets os.environ vars

# Known categories for this admissions chatbot. Extend this list as you
# discover more query types during testing.
CATEGORIES = [
    "fees",
    "admissions_eligibility",
    "curriculum",
    "faculty",
    "placements",
    "hostel_campus_life",
    "achievements_research",
    "publications",
    "events",
    "facilities",
    "rankings_accreditation",
    "contact_info",
    "international_admissions",
    "general_info",
]

CLASSIFY_PROMPT = (
    "Classify this query into EXACTLY ONE category from this list: {categories}. "
    "Query: \"{query}\"\n"
    "Respond with ONLY the category name, nothing else."
)


def _parse_category(raw_output: str) -> str:
    """Extract a valid category name from a model's raw text response."""
    last_line = raw_output.strip().splitlines()[-1].strip().lower()
    for cat in CATEGORIES:
        if cat in last_line:
            return cat
    return "general_info"  # fallback if parsing fails


# ---------------------------------------------------------------------
# Model 1: Keyword/rule-based classifier
# ---------------------------------------------------------------------
KEYWORD_MAP = {
    "fees": ["fee", "fees", "tuition", "cost", "scholarship", "payment"],
    "admissions_eligibility": ["eligibility", "admission", "cutoff", "entrance", "apply", "jee"],
    "curriculum": ["curriculum", "syllabus", "subjects", "credits", "semester"],
    "faculty": ["faculty", "professor", "dr.", "hod", "chairperson", "principal"],
    "placements": ["placement", "package", "salary", "recruiter", "job"],
    "hostel_campus_life": ["hostel", "campus", "library", "mess", "accommodation"],
    "achievements_research": ["achievement", "award", "research", "publication", "paper"],
}


def classify_keyword(query: str) -> str:
    query_lower = query.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_info"


# ---------------------------------------------------------------------
# Model 2: Embedding-similarity classifier
# ---------------------------------------------------------------------
CATEGORY_DESCRIPTIONS = {
    "fees": "questions about tuition fees, scholarship costs, payment structure",
    "admissions_eligibility": "questions about admission eligibility, entrance exams, cutoffs",
    "curriculum": "questions about course curriculum, subjects, credits, semester structure",
    "faculty": "questions about professors, faculty members, department heads",
    "placements": "questions about job placements, packages, recruiters",
    "hostel_campus_life": "questions about hostel, campus facilities, library, accommodation",
    "achievements_research": "questions about student and faculty achievements, research, publications",
    "general_info": "general questions about the university or programs",
}

_category_embeddings = None


def _get_category_embeddings():
    global _category_embeddings
    if _category_embeddings is None:
        model = get_model()
        texts = list(CATEGORY_DESCRIPTIONS.values())
        vectors = model.encode(texts, normalize_embeddings=True)
        _category_embeddings = dict(zip(CATEGORY_DESCRIPTIONS.keys(), vectors))
    return _category_embeddings


def classify_embedding(query: str) -> str:
    import numpy as np
    query_vec = embed_query(query)
    cat_embeddings = _get_category_embeddings()

    best_cat, best_score = None, -1
    for cat, cat_vec in cat_embeddings.items():
        score = float(np.dot(query_vec, cat_vec))  # cosine sim (vectors are normalized)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat


# ---------------------------------------------------------------------
# Models 3-5: three DIFFERENT model providers (genuine ensemble diversity,
# not just prompt variants on one model). Each is synchronous under the
# hood, so we wrap with asyncio.to_thread so all 3 can still run
# CONCURRENTLY via asyncio.gather — total latency ~= slowest single call.
# ---------------------------------------------------------------------

def _classify_via_openrouter_sync(query: str, model: str) -> str:
    """
    OpenRouter (https://openrouter.ai) gives one API key + one prepaid
    balance that can call many different providers' models (OpenAI,
    Google, xAI, Anthropic, Meta, etc). Since it's OpenAI-SDK-compatible,
    we just point the OpenAI client at OpenRouter's base_url instead of
    OpenAI's — same code path for all 3 "different providers."
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CATEGORIES), query=query)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
    )
    return _parse_category(response.choices[0].message.content)


def _classify_openai_sync(query: str) -> str:
    return _classify_via_openrouter_sync(query, model="openai/gpt-4o-mini")


def _classify_gemini_sync(query: str) -> str:
    return _classify_via_openrouter_sync(query, model="google/gemini-3.1-flash-lite")


def _classify_grok_sync(query: str) -> str:
    return _classify_via_openrouter_sync(query, model="x-ai/grok-4.3")


async def classify_llms_concurrent(query: str) -> list[str]:
    """Runs OpenAI, Gemini, and Grok classifiers CONCURRENTLY. Each is a
    blocking SDK call wrapped in asyncio.to_thread so they overlap instead
    of running one after another."""
    tasks = [
        asyncio.to_thread(_classify_openai_sync, query),
        asyncio.to_thread(_classify_gemini_sync, query),
        asyncio.to_thread(_classify_grok_sync, query),
    ]
    # return_exceptions=True so one provider failing (rate limit, bad key,
    # etc.) doesn't crash the whole ensemble — we just drop that vote.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    votes = []
    labels = ["openai", "gemini", "grok"]
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            print(f"[WARN] {label} classifier failed: {result}")
        else:
            votes.append(result)
    return votes


# ---------------------------------------------------------------------
# The full ensemble: runs all 5, returns the raw vote list
# ---------------------------------------------------------------------
async def get_ensemble_votes(query: str) -> list[str]:
    vote_keyword = classify_keyword(query)
    vote_embedding = classify_embedding(query)
    llm_votes = await classify_llms_concurrent(query)  # up to 3 votes, run in parallel

    return [vote_keyword, vote_embedding] + llm_votes


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "what is the fee for btech ECE"
    votes = asyncio.run(get_ensemble_votes(query))
    print(f'Query: "{query}"')
    print(f"Votes: {votes}")