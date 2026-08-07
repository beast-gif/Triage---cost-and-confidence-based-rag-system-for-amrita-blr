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

Pages that were in the manifest before but aren't in seed_urls.json
this run (or fail to scrape) are left alone by default -- see the
PURGE_MISSING_PAGES flag below if you want deleted pages auto-removed.
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
# If False: skipped URLs are left untouched (safer default while you're
# still testing scraping reliability).
PURGE_MISSING_PAGES = True

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


async def sync():
    with open(SEED_URLS_FILE, encoding="utf-8") as f:
        urls = json.load(f)

    print(f"Scraping {len(urls)} URLs...")
    scrape_results = await scrape_all(urls)

    manifest = Manifest()
    seen_urls = set()
    discovered_profile_urls: set[str] = set()

    total_added = 0
    total_deleted = 0
    total_unchanged = 0

    for result in scrape_results:
        url = result["url"]

        if not result["success"]:
            print(f"[SKIP] {url} — scrape failed: {result.get('error')}")
            continue

        seen_urls.add(url)

        if SAVE_RAW_HTML:
            save_html_to_disk(url, result["html"])

        chunks, page_profile_urls = extract_chunks(
            html=result["html"],
            source_url=url,
            page_title=result["title"],
        )
        discovered_profile_urls.update(page_profile_urls)
        chunks_by_id = {c["chunk_id"]: c for c in chunks}
        new_ids = set(chunks_by_id.keys())
        old_ids = manifest.get_chunk_ids(url)

        to_add_ids = new_ids - old_ids
        to_delete_ids = old_ids - new_ids
        unchanged_count = len(new_ids & old_ids)

        # Embed + upsert only the new/changed chunks
        if to_add_ids:
            to_add_chunks = [chunks_by_id[cid] for cid in to_add_ids]
            embeddings = embed_texts([c["content"] for c in to_add_chunks])
            upsert_chunks(to_add_chunks, embeddings)

        # Delete stale chunks whose content no longer exists on the page
        if to_delete_ids:
            delete_chunks(list(to_delete_ids))

        manifest.set_chunk_ids(url, new_ids)

        total_added += len(to_add_ids)
        total_deleted += len(to_delete_ids)
        total_unchanged += unchanged_count

        status_bits = []
        if to_add_ids:
            status_bits.append(f"+{len(to_add_ids)} new/changed")
        if to_delete_ids:
            status_bits.append(f"-{len(to_delete_ids)} stale")
        if unchanged_count:
            status_bits.append(f"={unchanged_count} unchanged")
        status = ", ".join(status_bits) if status_bits else "no chunks"
        print(f"[OK] {url} — {status}")

    # Handle pages that disappeared from seed_urls.json / failed to scrape
    if PURGE_MISSING_PAGES:
        for old_url in manifest.all_urls():
            if old_url not in seen_urls:
                stale_ids = manifest.get_chunk_ids(old_url)
                delete_chunks(list(stale_ids))
                manifest.remove_url(old_url)
                total_deleted += len(stale_ids)
                print(f"[REMOVED] {old_url} — page no longer present, purged {len(stale_ids)} chunks")

    # PASS 2: scrape the faculty profile URLs discovered while processing
    # the main seed_urls.json pages above. These aren't in seed_urls.json
    # themselves — we only know they exist because we found the link
    # inside each faculty card. Same scrape -> extract -> diff -> embed
    # logic applies, just for this dynamically-discovered URL set.
    if discovered_profile_urls:
        profile_url_list = sorted(discovered_profile_urls)
        print(f"\nDiscovered {len(profile_url_list)} faculty profile URLs — scraping...")
        profile_scrape_results = await scrape_all(profile_url_list)

        for result in profile_scrape_results:
            url = result["url"]

            if not result["success"]:
                print(f"[SKIP] {url} — scrape failed: {result.get('error')}")
                continue

            if SAVE_RAW_HTML:
                save_html_to_disk(url, result["html"])

            chunks, _ = extract_chunks(
                html=result["html"],
                source_url=url,
                page_title=result["title"],
            )
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

            total_added += len(to_add_ids)
            total_deleted += len(to_delete_ids)
            total_unchanged += unchanged_count

            status_bits = []
            if to_add_ids:
                status_bits.append(f"+{len(to_add_ids)} new/changed")
            if to_delete_ids:
                status_bits.append(f"-{len(to_delete_ids)} stale")
            if unchanged_count:
                status_bits.append(f"={unchanged_count} unchanged")
            status = ", ".join(status_bits) if status_bits else "no chunks"
            print(f"[OK] {url} — {status}")

    manifest.close()

    print("\n" + "=" * 50)
    print("SYNC SUMMARY")
    print("=" * 50)
    print(f"Chunks added/updated : {total_added}")
    print(f"Chunks deleted       : {total_deleted}")
    print(f"Chunks unchanged     : {total_unchanged}")
    print(f"Total in vector store: {count()}")


if __name__ == "__main__":
    asyncio.run(sync())