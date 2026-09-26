"""
whatsapp.py — WhatsApp Cloud API front end for Triage.

    app.include_router(whatsapp.router)

Written against Meta's WhatsApp Cloud API rather than Twilio. Meta gives you a
free test number the moment you create an app, usable with up to five verified
recipients and no business verification — enough to build and demo against,
and it is the same API you would use in production. Twilio would mean a
different payload shape, a different signature scheme and a different send
call; the pipeline half of this file would be unchanged.

WHY THIS IS NOT LIKE /chat
--------------------------
The browser opens a connection, holds it for 30-60s, and reads stage events
off an SSE stream. WhatsApp cannot do that. Meta POSTs a message to this
webhook and expects an acknowledgement within SECONDS; the answer goes back
later as a completely separate outbound API call. The two halves are linked
only by a phone number.

That single fact produces most of what follows:

  * the handler returns 200 BEFORE running the pipeline, because a 40-second
    ACK makes Meta retry the delivery — and a retried question is a second
    full run of retrieval and generation, billed again
  * retries mean the same message id can arrive twice anyway, so ids are
    deduplicated
  * there is no progress bar, so a "looking that up" message is sent first,
    purely to show the thing is alive

ENVIRONMENT
-----------
    WHATSAPP_VERIFY_TOKEN   any string; must match what you type into Meta's
                            webhook setup form
    WHATSAPP_APP_SECRET     from the Meta app dashboard; used to verify that a
                            request genuinely came from Meta
    WHATSAPP_TOKEN          access token for sending
    WHATSAPP_PHONE_ID       the phone number id to send from
    WHATSAPP_PHONE_SALT     salt for hashing phone numbers before storage
    WHATSAPP_RATE_LIMIT     messages per hour per number (default 10)

Without WHATSAPP_APP_SECRET the webhook refuses every request. That is
deliberate — see _verify_signature.
"""

import hashlib
import hmac
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()

GRAPH_VERSION = "v21.0"

# WhatsApp rejects a text body longer than this.
MAX_BODY_CHARS = 4096

# Remembered message ids, for retry deduplication. Bounded so it cannot grow
# without limit. IN MEMORY ONLY: a restart forgets them, and a second worker
# process would keep its own set. Fine for one uvicorn worker, which is what
# this runs as; if that ever changes, move it into SQLite alongside the
# contact table below.
_SEEN_MAX = 2000
_seen_ids: deque[str] = deque(maxlen=_SEEN_MAX)
_seen_set: set[str] = set()
_seen_lock = threading.Lock()

# Sliding-window rate limit, per phone number.
_RATE_WINDOW_SECONDS = 3600
_rate_log: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()

CONTACTS_DB = os.getenv("CONVERSATIONS_DB", "conversations.db")


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def _hash_phone(phone: str) -> str:
    """
    Stable pseudonym for a phone number.

    The raw number is needed to send a reply, but only for the seconds that
    takes — it never reaches disk. What gets stored is this hash, so the
    database maps an opaque token to a conversation rather than holding a list
    of prospective students' phone numbers next to everything they asked.

    The salt matters. Phone numbers come from a space small enough to brute
    force: an unsalted hash of an Indian mobile number falls to a laptop in
    minutes, which would make the hashing decorative. Set WHATSAPP_PHONE_SALT
    to something random and keep it out of the repo.
    """
    salt = os.getenv("WHATSAPP_PHONE_SALT", "")
    return hashlib.sha256(f"{salt}:{phone}".encode()).hexdigest()[:32]


def _init_contacts() -> None:
    with sqlite3.connect(CONTACTS_DB) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS whatsapp_contacts (
                phone_hash      TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
        """)


def _conversation_for(phone: str) -> str:
    """
    The conversation id for this number, creating one on first contact.

    This is where WhatsApp inherits the follow-up handling already built for
    the web app: because the id is stable per number, get_history_for_rewrite()
    returns real history and "what about EEE" gets rewritten into a standalone
    question exactly as it does in the browser.

    A conversation that has expired (30 days idle, deleted by conversations.py)
    is replaced rather than resurrected, so a returning user simply starts
    fresh instead of hitting a dead id.
    """
    import conversations

    _init_contacts()
    key = _hash_phone(phone)

    with sqlite3.connect(CONTACTS_DB) as db:
        row = db.execute(
            "SELECT conversation_id FROM whatsapp_contacts WHERE phone_hash = ?",
            (key,),
        ).fetchone()

    if row and conversations.exists(row[0]):
        return row[0]

    conversation_id = conversations.create_conversation()
    with sqlite3.connect(CONTACTS_DB) as db:
        db.execute(
            "INSERT INTO whatsapp_contacts (phone_hash, conversation_id, created_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(phone_hash) DO UPDATE SET conversation_id = excluded.conversation_id",
            (key, conversation_id),
        )
    return conversation_id


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _verify_signature(raw_body: bytes, header: str | None) -> bool:
    """
    Confirm the request came from Meta.

    This endpoint spends money: every accepted message runs three ensemble
    classifiers and usually a generation call. Unverified, anyone who finds the
    URL can POST fabricated messages and bill them to the OpenRouter key.

    Meta signs the RAW body with HMAC-SHA256 under the app secret. The raw
    bytes matter — re-serialising the parsed JSON changes whitespace and key
    order, and the signature no longer matches.

    Fails CLOSED when the secret is unset. An endpoint that silently accepts
    everything because a variable is missing is the failure mode worth
    preventing, even though it means local testing needs the secret set too.
    """
    secret = os.getenv("WHATSAPP_APP_SECRET")
    if not secret:
        print("[whatsapp] WHATSAPP_APP_SECRET unset — rejecting webhook")
        return False
    if not header or not header.startswith("sha256="):
        return False

    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    # compare_digest, not ==, so the comparison does not leak the signature
    # one byte at a time through its timing.
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def _already_seen(message_id: str) -> bool:
    """
    True if this message id has been handled.

    Meta delivers at least once. If an ACK is slow or lost the same message
    arrives again, and without this the question is answered — and paid for —
    twice, with the user receiving two identical replies.
    """
    with _seen_lock:
        if message_id in _seen_set:
            return True
        if len(_seen_ids) == _SEEN_MAX:
            _seen_set.discard(_seen_ids[0])   # about to be evicted by maxlen
        _seen_ids.append(message_id)
        _seen_set.add(message_id)
        return False


def _rate_limited(phone: str) -> bool:
    """
    True if this number is over its hourly allowance.

    The web app has no equivalent because nobody knows the URL. A WhatsApp
    number gets forwarded into group chats, and each message costs real LLM
    calls. This caps the damage from one enthusiastic or malicious sender
    without affecting anyone else.
    """
    limit = int(os.getenv("WHATSAPP_RATE_LIMIT", "10"))
    now = time.time()
    with _rate_lock:
        log = _rate_log[phone]
        while log and now - log[0] > _RATE_WINDOW_SECONDS:
            log.popleft()
        if len(log) >= limit:
            return True
        log.append(now)
        return False


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def to_whatsapp_text(answer: str, sources: list[str], band: str,
                     declined: bool) -> str:
    """
    Markdown -> WhatsApp.

    WhatsApp has its own small markup: *bold*, _italic_, ~strike~, ```mono```.
    It renders nothing else, so the generator's markdown arrives as literal
    punctuation — "**Dr. Ramesh**" shows the asterisks, and "[1]" shows the
    brackets.

    The band is included for MEDIUM answers. In the browser every reply carries
    a confidence badge, so a hedged answer looks hedged. Stripped to plain
    text, a 0.42 and a 0.98 are indistinguishable — the honesty the whole
    project is built around disappears at exactly the point it matters. HIGH
    needs no marker and LOW never reaches here, since generation is skipped.
    """
    text = answer

    text = re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", text, flags=re.MULTILINE)  # headings

    # Bold and italic must be done together, in this order.
    #
    # Markdown's **bold** becomes WhatsApp's *bold*, and markdown's *italic*
    # becomes _italic_. Run naively one after the other and the second rule
    # eats the output of the first: **Ramesh** turns into *Ramesh*, which the
    # italic rule then "converts" to _Ramesh_ — bold silently becomes italic.
    #
    # So bold is parked on a sentinel that no rule matches, italics are
    # converted, and bold is restored last.
    text = re.sub(r"\*\*(.+?)\*\*", "\x00\\1\x00", text)
    text = re.sub(r"(?<!\w)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)", r"_\1_", text)
    text = text.replace("\x00", "*")

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)  # links

    # [ \t]* rather than \s*: \s matches newlines, so "\n\n- item" had its
    # blank line swallowed and lists ran into the preceding paragraph.
    text = re.sub(r"^[ \t]*[-*][ \t]+", "• ", text, flags=re.MULTILINE)   # bullets

    text = re.sub(r"`([^`]+)`", r"```\1```", text)                        # inline code

    # Citation markers. They index a numbered context the reader cannot see,
    # so they are noise here — the URLs go at the bottom instead.
    text = re.sub(r"\s*\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if band == "medium" and not declined:
        text += "\n\n_This one I'm less sure about — worth confirming._"

    if sources and not declined:
        text += "\n\n*Source:*" if len(sources) == 1 else "\n\n*Sources:*"
        for url in sources[:3]:
            text += f"\n{url}"

    if len(text) > MAX_BODY_CHARS:
        text = text[: MAX_BODY_CHARS - 20].rstrip() + "\n\n…(truncated)"
    return text


# ---------------------------------------------------------------------------
# sending
# ---------------------------------------------------------------------------
async def _send_text(to: str, body: str) -> None:
    """
    Outbound message.

    No template needed: this always replies to something the user just sent, so
    it falls inside WhatsApp's 24-hour customer service window where free-form
    text is allowed. Messaging someone who has not written first is a different
    thing entirely and needs a pre-approved template.
    """
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("WHATSAPP_PHONE_ID")
    if not token or not phone_id:
        print("[whatsapp] WHATSAPP_TOKEN or WHATSAPP_PHONE_ID unset — cannot send")
        return

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{phone_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        # Link previews off: an answer citing three amrita.edu pages otherwise
        # renders as a wall of cards.
        "text": {"preview_url": False, "body": body},
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code >= 400:
            print(f"[whatsapp] send failed {response.status_code}: {response.text}")
    except Exception as exc:
        # Never raise: this runs in a background task with nobody to catch it,
        # and a failed send must not take the worker down.
        print(f"[whatsapp] send error: {exc}")


# ---------------------------------------------------------------------------
# the pipeline half
# ---------------------------------------------------------------------------
async def _answer_and_reply(phone: str, question: str) -> None:
    """
    Runs AFTER the webhook has already returned 200. Nothing here can affect
    the HTTP response Meta saw, which is the entire point.
    """
    # Imported here rather than at module scope: app.py includes this router,
    # so a top-level import would be circular. By call time both modules exist.
    from app import _answer_and_persist

    # The stand-in for the progress bar. Without it the user stares at a silent
    # thread for up to a minute and assumes the number is dead. It is also
    # honest — the work really has started.
    await _send_text(phone, "Looking that up…")

    try:
        conversation_id = _conversation_for(phone)
        response = await _answer_and_persist(question, conversation_id)
    except Exception as exc:
        print(f"[whatsapp] pipeline error: {exc}")
        await _send_text(
            phone,
            "Something went wrong answering that. Please try again in a moment.",
        )
        return

    await _send_text(
        phone,
        to_whatsapp_text(
            response.answer, response.sources, response.band, response.declined
        ),
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("/whatsapp/webhook")
async def verify_webhook(request: Request):
    """
    One-time handshake. On registering the URL, Meta GETs it with a challenge
    and expects the challenge echoed back as plain text — not JSON, which is
    why this does not return a dict.
    """
    params = request.query_params
    if (params.get("hub.mode") == "subscribe"
            and params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN")):
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("verification failed", status_code=403)


@router.post("/whatsapp/webhook")
async def receive_webhook(
    request: Request,
    background: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
):
    """
    Accept a message and get out of the way.

    Returns 200 in every case that is not a bad signature — including
    unparseable payloads. A non-200 makes Meta retry, and retrying a payload
    this server cannot read will never succeed; it just repeats forever.
    """
    raw = await request.body()

    if not _verify_signature(raw, x_hub_signature_256):
        # The one case worth rejecting. 403, not 200: a forged request should
        # not be acknowledged as if it were fine.
        return PlainTextResponse("bad signature", status_code=403)

    try:
        payload = await request.json()
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])
    except Exception:
        # Status updates (delivered, read) arrive on this same webhook and have
        # no "messages" key. Routine, not an error.
        return PlainTextResponse("ok")

    for message in messages:
        phone = message.get("from")
        message_id = message.get("id", "")
        if not phone or not message_id:
            continue

        if _already_seen(message_id):
            print(f"[whatsapp] duplicate {message_id}, ignoring")
            continue

        if message.get("type") != "text":
            background.add_task(
                _send_text, phone,
                "I can only read text messages at the moment. "
                "Try typing your question.",
            )
            continue

        if _rate_limited(phone):
            background.add_task(
                _send_text, phone,
                "That's a lot of questions in one go — give it an hour "
                "and I'll be back.",
            )
            continue

        question = (message.get("text", {}).get("body") or "").strip()
        if not question:
            continue

        # Queued, not awaited. The response below goes out first.
        background.add_task(_answer_and_reply, phone, question)

    return PlainTextResponse("ok")