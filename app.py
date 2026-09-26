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

CONVERSATIONS
-------------
/chat takes an OPTIONAL conversation_id and always returns one. History is
loaded server-side from that id rather than uploaded by the client, so the
client holds nothing but the id and a conversation survives a page reload.

Omitting the id starts a new conversation, which means an existing frontend
that knows nothing about this keeps working unchanged — it just has no memory
until it starts sending the id back.

PROGRESS
--------
/chat and /chat/stream run the IDENTICAL pipeline and persist identically —
/chat/stream just reports which phase it is in as it goes. The answer text
itself is not streamed; it arrives whole in the final `done` event. That is
deliberate: the wait is 30-60s of retrieval and reranking, and only a few
seconds of generation, so streaming tokens would leave the long part of the
wait exactly as silent as it is now.

The four stages are the honest ones. The reranker is a single blocking
predict() call with no internal callback, so "searching" cannot be subdivided
without inventing progress that does not exist. The client animates within a
stage instead of pretending to measure it.

ENDPOINTS
    GET  /health                models loaded? how many chunks?
    POST /chat                  answer + citations + confidence + photos
    POST /chat/stream           the same, as SSE, with stage events first
    POST /score                 confidence + chunks, no generation
    GET  /conversations         recent chats, for a sidebar
    GET  /conversations/{id}    full transcript, for resuming
    DELETE /conversations/{id}  forget one
    GET  /docs                  interactive Swagger UI
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MAX_QUERY_CHARS = 500

# Seconds of silence before a comment frame goes out. Proxies and load
# balancers close an idle connection, and this pipeline is idle-looking for
# the whole 30-60s of the searching stage.
SSE_KEEPALIVE_SECONDS = 15


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

    # Creates conversations.db if it does not exist. Cheap and idempotent.
    import conversations
    conversations.init()

    # --- weekly data refresh, in-process so it works on any platform ---
    #
    # Sunday 03:00 Asia/Kolkata — the timezone is pinned in scheduler.py, so
    # it means 03:00 IST wherever the host's clock is set.
    #
    # NOTE FOR ANY FUTURE DEPLOYMENT: this assumes a host that stays awake and
    # keeps its disk. On a container with ephemeral storage the job still runs,
    # scraping several hundred faculty pages for minutes, and then loses the
    # updated chroma_db at the next restart — spending the CPU for nothing.
    # If this is ever deployed somewhere like that, gate this line behind an
    # env flag rather than leaving it on.
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
    version="1.1.0",
    lifespan=lifespan,
)

# Origins allowed to call this API, comma-separated, from the environment:
#
#     ALLOWED_ORIGINS=https://triage.vercel.app,https://triage-git-main-x.vercel.app
#
# Defaults to "*" so `npm run dev` against a local backend needs no setup. In
# deployment SET IT. /chat spends money on every request, and a public API any
# site may call is someone else's free OpenRouter credit — paid for with your
# key.
#
# Vercel gives each deployment its own preview URL, so add the ones you
# actually use. A regex would cover them all at once, but allow_origin_regex
# with a loose pattern is how "*" comes back through the side door.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=MAX_QUERY_CHARS,
                       examples=["what is the fee for btech ECE"])
    # Omit to start a new conversation. The response returns the id to send
    # back on later turns.
    conversation_id: str | None = Field(default=None, max_length=64)


class Photo(BaseModel):
    url: str
    name: str
    source_url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    photos: list[Photo]
    confidence: float
    band: str
    generator_mode: str
    route: str | None
    llm_called: bool
    declined: bool
    model: str | None
    elapsed_seconds: float
    conversation_id: str


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
# helpers
# ---------------------------------------------------------------------------
def _name_from_url(url: str) -> str:
    """
    Display name from a faculty profile slug: /faculty/tk-ramesh/ -> T. K. Ramesh

    Chunks carry source_url and photo_url but no name field. Adding one would
    mean changing the extractor, re-syncing and backfilling ~4,900 chunks; the
    slug is already here. Segments of one or two letters are treated as
    initials, which is right far more often than not on this site.
    """
    slug = (url or "").rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        return ""
    parts = []
    for piece in slug.split("-"):
        if not piece:
            continue
        if len(piece) <= 2 and piece.isalpha():
            parts.append(". ".join(c.upper() for c in piece) + ".")
        else:
            parts.append(piece.capitalize())
    return " ".join(parts)


def _photos_for(out: dict) -> list[dict]:
    """
    One photo per faculty member the answer CITED.

    Returned as structured data rather than rendered markup, because the
    Gradio version built a markdown string and a React client would have to
    parse it back apart. The client decides layout, sizing and whether to
    suppress repeats.

    Nothing on a refusal, in either form: band LOW (the LLM was never called,
    but chunks are still populated) or `declined` (the LLM WAS called and the
    prompt's rule 3 fired — retrieval succeeded, the content did not answer).
    Without this guard a faculty face appears above "I don't have that
    information."
    """
    if (out.get("declined") or out.get("band") == "low"
            or not out.get("llm_called")):
        return []

    answer = out.get("answer") or ""
    photos, seen = [], set()

    # Citation [n] maps to chunks[n-1] — the numbering _build_context used
    # when the prompt was assembled.
    for i, chunk in enumerate(out.get("chunks") or [], start=1):
        if f"[{i}]" not in answer:
            continue
        url = (chunk.get("photo_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        source = chunk.get("source_url") or ""
        photos.append({
            "url": url,
            "name": _name_from_url(source),
            "source_url": source,
        })
    return photos


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
    import conversations

    return {
        "status": "ok",
        "chunks": getattr(app.state, "chunk_count", None),
        "startup_seconds": getattr(app.state, "ready_in", None),
        "conversations": conversations.stats(),
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

    # next_run_time() is only called when a scheduler exists. It always does
    # today, but startup can fail, and an endpoint that 500s while reporting on
    # the health of something else is the wrong failure.
    scheduler = getattr(app.state, "scheduler", None)

    return {
        **STATUS,
        "enabled": scheduler is not None,
        "next_run": next_run_time(scheduler) if scheduler else None,
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


def _resolve_conversation(conversation_id: str | None) -> str:
    """
    Validate an incoming id or mint a new one.

    Kept separate from _answer_and_persist so /chat/stream can run it BEFORE
    the response starts. Once a StreamingResponse has begun the status line is
    already 200 and a 404 can no longer be sent — the client would get a dead
    id reported inside the stream body, which no fetch error handler catches.
    """
    import conversations

    if conversation_id:
        if not conversations.exists(conversation_id):
            raise HTTPException(
                status_code=404,
                detail="conversation not found or expired — start a new one",
            )
        return conversation_id
    return conversations.create_conversation()


async def _answer_and_persist(query: str, conversation_id: str,
                              on_stage=None) -> ChatResponse:
    """
    The whole chat turn: history -> pipeline -> photos -> persist.

    Shared verbatim by /chat and /chat/stream so the streaming variant cannot
    drift into answering or recording differently from the plain one. The only
    difference between them is that one passes an on_stage callback.
    """
    import conversations
    from generator import answer_query

    history = conversations.get_history_for_rewrite(conversation_id)

    started = time.time()
    try:
        out = await answer_query(query, history=history, on_stage=on_stage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}")

    photos = _photos_for(out)

    response = ChatResponse(
        answer=out["answer"],
        sources=out.get("sources", []),
        photos=photos,
        confidence=out.get("confidence") or 0.0,
        band=out.get("band", "low"),
        generator_mode=out.get("generator_mode", "disambiguate"),
        route=out.get("route"),
        llm_called=out.get("llm_called", False),
        declined=out.get("declined", False),
        model=out.get("model"),
        elapsed_seconds=round(time.time() - started, 2),
        conversation_id=conversation_id,
    )

    # Persist AFTER answering, so a pipeline failure does not leave a question
    # recorded with no reply. The assistant message carries its display
    # metadata, which is what lets a resumed chat render with its badges and
    # photos instead of as a plain transcript.
    conversations.add_message(conversation_id, "user", query)
    conversations.add_message(
        conversation_id, "assistant", response.answer,
        meta={
            "band": response.band,
            "confidence": response.confidence,
            "route": response.route,
            "sources": response.sources,
            "photos": photos,
            "llm_called": response.llm_called,
            "declined": response.declined,
            "model": response.model,
        },
    )

    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Full pipeline. A low-confidence query returns llm_called=false — generation
    is skipped rather than attempted, so the refusal is free.

    An expired or unknown conversation_id returns 404 so the client can clear
    its stored id and start fresh. Conversations are deleted after 30 days of
    inactivity, silently.
    """
    conversation_id = _resolve_conversation(request.conversation_id)
    return await _answer_and_persist(request.query, conversation_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Same answer as /chat, delivered as server-sent events.

    Event shapes, one JSON object per `data:` line:

        {"type": "stage", "stage": "searching"}
        {"type": "done",  "data": { ...the full ChatResponse... }}
        {"type": "error", "status": 500, "detail": "..."}

    Stages arrive in order: understanding -> searching -> scoring ->
    generating. A `done` or `error` event always ends the stream.

    POST, not GET, so the browser's EventSource cannot be used — the client
    reads this with fetch() and a ReadableStream reader instead. That is the
    right trade: the request carries a body (query + conversation_id), and
    smuggling a 500-character question through a query string to satisfy
    EventSource would be worse.
    """
    conversation_id = _resolve_conversation(request.conversation_id)

    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(name: str) -> None:
        # Sync, and only ever called from the pipeline coroutine itself —
        # never from inside an asyncio.to_thread worker — so put_nowait is
        # safe and no thread-safe handoff is needed. Unbounded queue, so this
        # cannot block or raise QueueFull.
        queue.put_nowait({"type": "stage", "stage": name})

    async def run() -> None:
        try:
            response = await _answer_and_persist(
                request.query, conversation_id, on_stage=on_stage
            )
            await queue.put({"type": "done", "data": response.model_dump()})
        except HTTPException as exc:
            await queue.put({"type": "error", "status": exc.status_code,
                             "detail": exc.detail})
        except Exception as exc:
            await queue.put({"type": "error", "status": 500,
                             "detail": f"pipeline error: {exc}"})
        finally:
            await queue.put(None)  # sentinel: close the stream

    async def events():
        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # comment frame, ignored by clients
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # The browser tab closed or the client aborted mid-answer. Without
            # this the pipeline runs to completion — and bills for a generation
            # — with nobody left to read it.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which holds every
            # stage event until the whole stream ends — the exact opposite of
            # the point. Harmless when there is no nginx in front.
            "X-Accel-Buffering": "no",
        },
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
                "photo_url": c.get("photo_url", ""),
                "rerank_score": c["rerank_score"],
                "preview": " ".join(c["content"].split())[:300],
            }
            for c in result.get("chunks", [])
        ],
        elapsed_seconds=round(time.time() - started, 2),
    )


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------
@app.get("/conversations")
async def list_conversations(limit: int = 50):
    """Recent chats, newest first, with the opening question as a preview."""
    import conversations

    return {"conversations": conversations.list_conversations(limit=limit)}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Full transcript for resuming. Each assistant message carries the `meta` it
    was answered with, so the client can re-render badges and photos rather
    than showing a bare text log.
    """
    import conversations

    if not conversations.exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="conversation not found or expired",
        )

    return {
        "conversation_id": conversation_id,
        "messages": conversations.get_messages(conversation_id),
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    import conversations

    if not conversations.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": conversation_id}


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------
# Imported last, and at the bottom, because whatsapp.py calls back into
# _answer_and_persist above. That import lives inside the function rather than
# at its module scope, so there is no cycle — but keeping the include here
# makes the direction of the dependency obvious.
#
# The routes are inert until the WHATSAPP_* variables are set: the webhook
# rejects every request without WHATSAPP_APP_SECRET, deliberately.
import whatsapp  # noqa: E402

app.include_router(whatsapp.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)