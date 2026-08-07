---
title: Triage
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Triage

A confidence- and cost-aware RAG chatbot for **Amrita Vishwa Vidyapeetham,
Bengaluru** admissions.

Most RAG systems always return something. Triage measures whether the retrieval
actually succeeded, and refuses — without calling the generation model at all —
when it did not.

## Setup

Add `OPENROUTER_API_KEY` under **Settings → Variables and secrets**. Without it
retrieval and scoring still work, but generation will fail.

On first boot the vector store is empty. Open the **Data** tab and click
**Run sync now**. It scrapes 26 seed pages, discovers a few hundred faculty
profile URLs from the faculty card grids, and embeds everything — roughly
10–15 minutes. After that a sync runs automatically every Sunday at 03:00 IST.

## How confidence is computed

Two signals, combined with `max()` rather than a weighted average:

| signal | question |
|---|---|
| `abs_signal` | is the top chunk's reranker score high on its own? |
| `sep_signal` | does it tower over *this query's* own noise floor? |

Either alone is sufficient. That matters because correct answers come in two
shapes, measured on real queries:

```
"eligibility for btech"   abs 0.95   sep 0.01     top score 0.9956
"who is the HOD of ECE"   abs 0.00   sep 0.96     top score 0.0873
```

Both are correct at rank 1. The first has eight equally-correct chunks (every
program page repeats the same eligibility text), so nothing separates. The
second scores feebly in absolute terms but sits 73× above its background. A
weighted average would score both around 0.48 and flag both uncertain.

The ensemble's category vote multiplies the result by `0.7 + 0.3 × agreement`,
so it can only ever discount — it reads the question, never the retrieved
chunks, and cannot be evidence that retrieval worked.

All constants were fitted to a labelled set of 25 queries, 6 of them
deliberately unanswerable.

## Retrieval routing

| route | when |
|---|---|
| plain vector search | default |
| `department_heads` | a head-of-department query naming a department that lists one |
| `principal_<school>` | that department has no head of its own — the school Principal answers |

The fallback exists because departments label their heads inconsistently: ECE
says "HOD", EEE says "Chairperson", English says "Head of Department", and
Mechanical and the Computing-school departments list none, deferring to their
school Principal. There are two different Principals, so the school is resolved
from the department before the fallback runs.

## Known limitations

Two measured failures where the retrieved text is lexically right and
semantically wrong:

- **"who is the vice chancellor"** — a chunk mentions the title but names
  nobody. Scores 0.998.
- **"placement record for computing"** — returns a publications page at rank 1.

No formula over reranker scores can detect these; the score cannot tell that a
strongly-matching chunk fails to contain an answer. The generation prompt's
refusal rule is what catches them, and on the vice-chancellor query it does —
correctly noting that only a *Pro* Vice-Chancellor is named.

## API

Gradio serves the interface at `/`. The JSON API remains available:

| endpoint | |
|---|---|
| `POST /chat` | full pipeline, may call the generation model |
| `POST /score` | retrieval and confidence only — never calls the LLM |
| `GET /health` | model and chunk-count status |
| `GET /sync/status` | last sync, next run, department-head coverage |
| `POST /sync/run` | trigger a refresh |
| `GET /docs` | Swagger UI |

## Stack

Crawl4AI · BeautifulSoup · SQLite manifest · BGE embeddings ·
`bge-reranker-base` cross-encoder · ChromaDB · OpenRouter · FastAPI ·
Gradio · APScheduler