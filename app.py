"""
app.py — FastAPI wrapper around Triage.

    uvicorn app:app --reload
    http://127.0.0.1:8000/docs

WHY A LIFESPAN HANDLER
----------------------
BAAI/bge-base-en-v1.5 and BAAI/bge-reranker-base take ~10s to load and cache in
module-level globals. In the CLI that cost was paid on every invocation. Here
they load ONCE at startup, before the server accepts traffic, so no user ever
eats the cold start.

WHY THE ENDPOINTS ARE SPLIT
---------------------------
    /chat    full pipeline, may call the generation LLM (costs money)
    /score   retrieval + confidence only, never calls the LLM (free)

/score exists because most debugging does not need generation. It also makes
the cost-aware behaviour inspectable: you can see WHY a query was refused
without paying to be refused.

ENDPOINTS
    GET  /health   models loaded? how many chunks?
    POST /chat     {"query": "..."} -> answer + citations + confidence
    POST /score    {"query": "..."} -> confidence + chunks, no generation
    GET  /docs     interactive Swagger UI
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MAX_QUERY_CHARS = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load models before accepting traffic ---
    print("loading models...")
    started = time.time()
    from embedder import get_model
    from reranker import get_reranker
    from store import count

    get_model()
    get_reranker()
    app.state.chunk_count = count()
    app.state.ready_in = round(time.time() - started, 1)
    print(f"ready in {app.state.ready_in}s — {app.state.chunk_count} chunks")

    # --- weekly data refresh, in-process so it works on any platform ---
    from scheduler import start_scheduler
    app.state.scheduler = start_scheduler()

    yield

    # --- shutdown ---
    if getattr(app.state, "scheduler", None):
        app.state.scheduler.shutdown(wait=False)
    print("shutting down")


app = FastAPI(
    title="Triage",
    description="Confidence- and cost-aware RAG chatbot for Amrita Vishwa "
                "Vidyapeetham, Bengaluru admissions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Wide open for development. Before deploying, replace "*" with the actual
# frontend origin — a public API that any site can call is a bill waiting to
# happen, since /chat spends money per request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=MAX_QUERY_CHARS,
                       examples=["what is the fee for btech ECE"])


class Source(BaseModel):
    url: str
    heading: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    band: str
    generator_mode: str
    route: str | None
    llm_called: bool
    model: str | None
    elapsed_seconds: float


class ScoreResponse(BaseModel):
    query: str
    confidence: float
    band: str
    generator_mode: str
    route: str | None
    signals: dict
    chunks: list[dict]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
# NOTE: there is deliberately no GET "/" route here. gradio_app.py mounts the
# chat UI at "/", and a FastAPI route registered on the same path wins over the
# mount — which showed up as the browser returning this file's JSON instead of
# the interface. Run `uvicorn gradio_app:app` to get both; running
# `uvicorn app:app` gives the API alone and 404s at "/".


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks": getattr(app.state, "chunk_count", None),
        "startup_seconds": getattr(app.state, "ready_in", None),
    }


@app.get("/sync/status")
async def sync_status():
    """
    When the data was last refreshed, and whether department-head coverage
    still looks sane. `health.warnings` is the field to watch: a department
    dropping to zero heads usually means the site changed a job title the
    ALIASES table does not recognise.
    """
    from scheduler import STATUS, next_run_time

    return {
        **STATUS,
        "next_run": next_run_time(getattr(app.state, "scheduler", None)),
    }


@app.post("/sync/run")
async def sync_run():
    """
    Trigger a refresh now. Takes several minutes — Pass 2 scrapes a few hundred
    faculty profile pages — and runs on a worker thread so /chat stays
    responsive. Returns 409 if a sync is already in progress.
    """
    from scheduler import run_sync_background

    result = await run_sync_background(triggered_by="manual")
    if result.get("status") == "busy":
        raise HTTPException(status_code=409, detail=result["detail"])
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result["detail"])

    app.state.chunk_count = result.get("chunks", app.state.chunk_count)
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Full pipeline. Note that a low-confidence query returns llm_called=False —
    generation is skipped rather than attempted, so the refusal is free.
    """
    from generator import answer_query

    started = time.time()
    try:
        out = await answer_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}")

    return ChatResponse(
        answer=out["answer"],
        sources=out.get("sources", []),
        confidence=out.get("confidence") or 0.0,
        band=out.get("band", "low"),
        generator_mode=out.get("generator_mode", "disambiguate"),
        route=out.get("route"),
        llm_called=out.get("llm_called", False),
        model=out.get("model"),
        elapsed_seconds=round(time.time() - started, 2),
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: ChatRequest):
    """Retrieval + confidence only. Never calls the generation LLM."""
    from confidence import score_query

    started = time.time()
    try:
        result = await score_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scoring error: {exc}")

    return ScoreResponse(
        query=request.query,
        confidence=result.get("final_confidence", 0.0),
        band=result.get("band", "low"),
        generator_mode=result.get("generator_mode", "disambiguate"),
        route=result.get("route"),
        signals=result.get("retrieval_details", {}),
        chunks=[
            {
                "source_url": c["source_url"],
                "heading": c.get("heading"),
                "designation": c.get("designation"),
                "rerank_score": c["rerank_score"],
                "preview": " ".join(c["content"].split())[:300],
            }
            for c in result.get("chunks", [])
        ],
        elapsed_seconds=round(time.time() - started, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)