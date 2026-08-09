"""
sync.py

THIS is the command you run to keep your vector store up to date.

    python sync.py

What it does, per URL in seed_urls.json:
  1. Scrape the page fresh (Crawl4AI)
  2. Extract clean chunks (extractor.py) -> each chunk_id encodes a
     content hash, so identical content always produces the same chunk_id
  3. Compare the new chunk_ids against what's recorded in the manifest
     for that URL:
       - new_ids - old_ids   -> new/changed chunks -> embed + upsert
       - old_ids - new_ids   -> stale chunks        -> delete from Chroma
       - new_ids & old_ids   -> unchanged           -> skip entirely
  4. Update the manifest with the new chunk_id set for that URL

TWO PASSES
  Pass 1  the URLs in seed_urls.json
  Pass 2  faculty profile URLs discovered from the .fc-item card grids
          during pass 1 — these are never in seed_urls.json

THE PURGE ORDERING BUG (fixed)
------------------------------
PURGE_MISSING_PAGES deletes any manifest URL that was not reached this run.
It used to run BETWEEN pass 1 and pass 2 — but every faculty profile URL is
only reached in pass 2, so at that moment none of them were in `seen_urls`
and all of them looked like deleted pages.

Observed live: a single run purged the entire faculty corpus —

    [REMOVED] .../faculty/s-smita/          purged 13 chunks
    [REMOVED] .../faculty/rg-chittawadigi/  purged 61 chunks
    [REMOVED] .../faculty/rashmi-m-r/       purged 73 chunks
    ... and dozens more

Pass 2 then re-scraped and re-embedded all of them, so the data survived, but
the run cost thousands of needless embeddings and left the store briefly
missing every faculty member. The purge now runs AFTER pass 2, once
`seen_urls` actually contains everything that was reached.
"""

import asyncio
import json

from Scrap_crawl4ai import scrape_all
from extractor import extract_chunks
from manifest import Manifest
from embedder import embed_texts
from store import upsert_chunks, delete_chunks, count

SEED_URLS_FILE = "seed_urls.json"

# If True: a URL that was in the manifest before but wasn't successfully
# scraped this run gets ALL its chunks deleted (treats it as "page removed").
#
# KEEP THIS FALSE unless you are deliberately cleaning up removed pages. With
# it on, ONE page failing to scrape — a timeout, a transient 503 — silently
# deletes that page's chunks even though the page is fine.
PURGE_MISSING_PAGES = False

# If True: writes each scraped page's raw HTML to raw_html/ so you can
# inspect page structure later (e.g. to debug extraction issues) without
# re-scraping. Off by default since sync.py is meant to be stateless
# beyond the manifest + Chroma — turn on only when you need to debug.
SAVE_RAW_HTML = False
RAW_HTML_DIR = "raw_html"


def save_html_to_disk(url: str, html: str, out_dir: str = RAW_HTML_DIR):
    """Writes one page's raw HTML to disk, named after its URL."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_")
    path = os.path.join(out_dir, f"{safe_name}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def process_page(result, manifest, seen_urls, collect_profiles=None):
    """
    Scrape result -> extract -> diff against manifest -> embed only what changed.

    Shared by both passes; they differed only in whether they collected
    discovered profile URLs, which was enough duplication for the two copies to
    drift apart.

    Returns (added, deleted, unchanged).
    """
    url = result["url"]

    if not result["success"]:
        print(f"[SKIP] {url} — scrape failed: {result.get('error')}")
        return 0, 0, 0

    seen_urls.add(url)

    if SAVE_RAW_HTML:
        save_html_to_disk(url, result["html"])

    chunks, page_profile_urls = extract_chunks(
        html=result["html"],
        source_url=url,
        page_title=result["title"],
    )
    if collect_profiles is not None:
        collect_profiles.update(page_profile_urls)

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    new_ids = set(chunks_by_id.keys())
    old_ids = manifest.get_chunk_ids(url)

    to_add_ids = new_ids - old_ids
    to_delete_ids = old_ids - new_ids
    unchanged_count = len(new_ids & old_ids)

    if to_add_ids:
        to_add_chunks = [chunks_by_id[cid] for cid in to_add_ids]
        embeddings = embed_texts([c["content"] for c in to_add_chunks])
        upsert_chunks(to_add_chunks, embeddings)

    if to_delete_ids:
        delete_chunks(list(to_delete_ids))

    manifest.set_chunk_ids(url, new_ids)

    bits = []
    if to_add_ids:
        bits.append(f"+{len(to_add_ids)} new/changed")
    if to_delete_ids:
        bits.append(f"-{len(to_delete_ids)} stale")
    if unchanged_count:
        bits.append(f"={unchanged_count} unchanged")
    print(f"[OK] {url} — {', '.join(bits) if bits else 'no chunks'}")

    return len(to_add_ids), len(to_delete_ids), unchanged_count


async def sync():
    with open(SEED_URLS_FILE, encoding="utf-8") as f:
        urls = json.load(f)

    manifest = Manifest()
    seen_urls: set[str] = set()
    discovered_profile_urls: set[str] = set()

    total_added = total_deleted = total_unchanged = 0

    # --- PASS 1: the seed URLs ---
    print(f"Pass 1 — scraping {len(urls)} seed URLs...")
    for result in await scrape_all(urls):
        a, d, u = process_page(result, manifest, seen_urls,
                               collect_profiles=discovered_profile_urls)
        total_added += a
        total_deleted += d
        total_unchanged += u

    # --- PASS 2: faculty profiles discovered in pass 1 ---
    # These are never in seed_urls.json; the only reason we know they exist is
    # the <a href> inside each .fc-item faculty card.
    if discovered_profile_urls:
        profile_list = sorted(discovered_profile_urls)
        print(f"\nPass 2 — {len(profile_list)} faculty profile URLs discovered...")
        for result in await scrape_all(profile_list):
            a, d, u = process_page(result, manifest, seen_urls)
            total_added += a
            total_deleted += d
            total_unchanged += u

    # --- purge, AFTER both passes ---
    # seen_urls only contains every reachable page once pass 2 has finished.
    # Running this earlier deleted the entire faculty corpus every time.
    if PURGE_MISSING_PAGES:
        print("\nPurging pages no longer present...")
        for old_url in manifest.all_urls():
            if old_url not in seen_urls:
                stale_ids = manifest.get_chunk_ids(old_url)
                delete_chunks(list(stale_ids))
                manifest.remove_url(old_url)
                total_deleted += len(stale_ids)
                print(f"[REMOVED] {old_url} — purged {len(stale_ids)} chunks")

    manifest.close()

    print("\n" + "=" * 50)
    print("SYNC SUMMARY")
    print("=" * 50)
    print(f"Pages reached        : {len(seen_urls)}")
    print(f"Chunks added/updated : {total_added}")
    print(f"Chunks deleted       : {total_deleted}")
    print(f"Chunks unchanged     : {total_unchanged}")
    print(f"Total in vector store: {count()}")
    if PURGE_MISSING_PAGES:
        print("\nNOTE: purge was ON. Set PURGE_MISSING_PAGES = False for "
              "routine syncs — one failed scrape deletes a live page's chunks.")


if __name__ == "__main__":
    asyncio.run(sync())