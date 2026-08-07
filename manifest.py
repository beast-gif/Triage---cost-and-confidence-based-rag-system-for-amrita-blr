"""
manifest.py

Tracks which chunk_ids currently exist for each source_url. Since our
chunk_id already encodes a content hash (see extractor.py), we don't need
a separate hash column — just the set of chunk_ids per URL is enough to
diff "what changed" between scrapes.

Diff logic (used by sync.py):
    new_ids = set of chunk_ids produced by THIS run for a URL
    old_ids = set of chunk_ids stored in the manifest for that URL

    to_add    = new_ids - old_ids   -> new or changed content, needs embedding
    to_delete = old_ids - new_ids   -> stale content, remove from vector store
    unchanged = new_ids & old_ids   -> skip, nothing to do

A URL that existed in the manifest but wasn't scraped this run at all
(e.g. the page was deleted from the site) has ALL its chunk_ids deleted.
"""

import sqlite3
import json
from datetime import datetime, timezone


class Manifest:
    def __init__(self, path: str = "manifest.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                chunk_ids TEXT NOT NULL,   -- JSON list
                last_synced TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get_chunk_ids(self, url: str) -> set[str]:
        """Returns the set of chunk_ids currently recorded for this URL.
        Empty set if the URL isn't in the manifest yet (brand new page)."""
        row = self.conn.execute(
            "SELECT chunk_ids FROM pages WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return set()
        return set(json.loads(row[0]))

    def set_chunk_ids(self, url: str, chunk_ids: set[str]):
        """Overwrite the recorded chunk_ids for a URL after a successful sync."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO pages (url, chunk_ids, last_synced)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                chunk_ids = excluded.chunk_ids,
                last_synced = excluded.last_synced
        """, (url, json.dumps(sorted(chunk_ids)), now))
        self.conn.commit()

    def remove_url(self, url: str):
        """Remove a URL entirely from the manifest (page was deleted)."""
        self.conn.execute("DELETE FROM pages WHERE url = ?", (url,))
        self.conn.commit()

    def all_urls(self) -> list[str]:
        return [row[0] for row in self.conn.execute("SELECT url FROM pages")]

    def close(self):
        self.conn.close()