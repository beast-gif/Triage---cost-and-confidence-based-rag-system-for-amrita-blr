"""
conversations.py — persistent chat history, so a user can resume later.

WHY IN THE BACKEND AND NOT THE GRADIO LAYER
-------------------------------------------
Conversation memory currently lives entirely in Gradio: the chat component
holds the message list and hands it to every callback. That works, but it is
frontend state — swap Gradio for React and it vanishes, and the JSON API has
no memory at all today.

Putting it behind /chat means whatever frontend comes next just calls the
endpoint, and a conversation can be resumed from a different page load.

WHAT IS STORED AND WHY
----------------------
Not just the text. Each message keeps its band, confidence, route, sources,
photos and declined flag, because a resumed conversation should LOOK like the
original — badges and faculty photos included — rather than reading as a
plain transcript.

EXPIRY
------
Conversations untouched for RETENTION_DAYS are deleted. Silent: nothing warns
the user beforehand, and a request carrying a dead id gets a 404 so the client
can clear its stored id and start fresh.

Cleanup runs from the weekly APScheduler job rather than on every request —
deleting on read would make an ordinary question pay for housekeeping.

KNOWN LIMITS, worth stating rather than discovering
---------------------------------------------------
* Same browser only. The id lives in the client's localStorage; clear it or
  switch devices and the conversation is unreachable, though still in the
  database. Cross-device resume needs real identity.
* No privacy boundary. Anyone holding an id can read that conversation. Fine
  for a demo; not fine for anything real, since students may ask things they
  would rather not have readable by id guessing.

    python conversations.py            # stats
    python conversations.py --cleanup  # delete expired
"""

import json
import secrets
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone

DB_PATH = "conversations.db"
RETENTION_DAYS = 30

# How many exchanges the query rewriter gets. Matches rewrite.HISTORY_TURNS —
# more risks resolving a pronoun against a stale entity.
REWRITE_TURNS = 2

# SQLite allows one writer at a time. Requests are concurrent, so writes are
# serialised here rather than left to surface as "database is locked".
_write_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL lets readers proceed while a write is in progress, which matters once
    # retrieval and a history read overlap.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init():
    with _write_lock, _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id             TEXT PRIMARY KEY,
                created_at     TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                -- Display metadata, so a resumed chat renders with its badges
                -- and photos instead of as a bare transcript.
                meta            TEXT,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversations_active
                ON conversations(last_active_at);
        """)


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_conversation():
    """New conversation id. token_urlsafe, not a counter — an id IS the access
    key, so it must not be guessable by incrementing."""
    conversation_id = f"c_{secrets.token_urlsafe(12)}"
    now = _now()
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, created_at, last_active_at) "
            "VALUES (?, ?, ?)",
            (conversation_id, now, now),
        )
    return conversation_id


def exists(conversation_id):
    if not conversation_id:
        return False
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def add_message(conversation_id, role, content, meta=None):
    """Append a message and bump last_active_at, which drives expiry."""
    now = _now()
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, created_at, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, now,
             json.dumps(meta) if meta else None),
        )
        conn.execute(
            "UPDATE conversations SET last_active_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def get_messages(conversation_id, limit=None):
    """
    Full transcript, oldest first, each with its display metadata parsed back
    out of JSON.
    """
    query = ("SELECT role, content, created_at, meta FROM messages "
             "WHERE conversation_id = ? ORDER BY id")
    params = [conversation_id]
    if limit:
        # Take the LAST `limit` rows, then restore chronological order.
        query = (f"SELECT * FROM ({query} DESC LIMIT ?) ORDER BY created_at")
        params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
            "meta": json.loads(r["meta"]) if r["meta"] else None,
        }
        for r in rows
    ]


def get_history_for_rewrite(conversation_id, turns=REWRITE_TURNS):
    """
    The shape rewrite.rewrite_query() expects: a list of
    {role, content} dicts, most recent exchanges only.

    turns * 2 because each exchange is a user message plus an assistant one.
    """
    if not conversation_id:
        return []
    messages = get_messages(conversation_id, limit=turns * 2)
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def list_conversations(limit=50):
    """Most recently active first, with a preview for a sidebar."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT c.id, c.created_at, c.last_active_at,
                   COUNT(m.id) AS message_count,
                   (SELECT content FROM messages
                    WHERE conversation_id = c.id AND role = 'user'
                    ORDER BY id LIMIT 1) AS first_question
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.last_active_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(r) for r in rows]


def delete_conversation(conversation_id):
    with _write_lock, _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?",
                     (conversation_id,))
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?",
                              (conversation_id,))
    return cursor.rowcount > 0


def cleanup_expired(days=RETENTION_DAYS):
    """
    Delete conversations untouched for `days`. Returns how many went.

    Called from the weekly sync job, not per request — housekeeping should not
    be charged to whoever happens to ask the next question.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with _write_lock, _connect() as conn:
        stale = [r["id"] for r in conn.execute(
            "SELECT id FROM conversations WHERE last_active_at < ?", (cutoff,)
        ).fetchall()]

        if stale:
            marks = ",".join("?" * len(stale))
            conn.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({marks})",
                stale,
            )
            conn.execute(
                f"DELETE FROM conversations WHERE id IN ({marks})", stale
            )

    return len(stale)


def stats():
    with _connect() as conn:
        row = conn.execute("""
            SELECT (SELECT COUNT(*) FROM conversations) AS conversations,
                   (SELECT COUNT(*) FROM messages)      AS messages,
                   (SELECT MIN(created_at) FROM conversations) AS oldest,
                   (SELECT MAX(last_active_at) FROM conversations) AS newest
        """).fetchone()
    return dict(row)


init()


if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        removed = cleanup_expired()
        print(f"deleted {removed} conversation(s) inactive for "
              f"{RETENTION_DAYS}+ days")

    s = stats()
    print(f"\nconversations : {s['conversations']}")
    print(f"messages      : {s['messages']}")
    print(f"oldest        : {s['oldest'] or '-'}")
    print(f"last active   : {s['newest'] or '-'}")
    print(f"retention     : {RETENTION_DAYS} days")

    recent = list_conversations(limit=5)
    if recent:
        print("\nmost recent:")
        for c in recent:
            question = (c["first_question"] or "")[:46]
            print(f"  {c['id']}  {c['message_count']:>3} msgs  "
                  f"{c['last_active_at'][:19]}  {question}")