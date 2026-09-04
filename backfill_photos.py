"""
backfill_photos.py — attach faculty photo URLs to chunks already in Chroma.

WHY A BACKFILL AND NOT JUST A RE-SYNC
-------------------------------------
photo_url is METADATA, but chunk_id is a hash of CONTENT only. So re-running
sync.py produces chunks with identical ids, the manifest diff reports
"unchanged", and the upsert is skipped entirely. The new metadata never lands.

That is delta-sync working correctly — it is what makes a no-change sync cost
zero embeddings — it just cannot help here. Forcing the issue by hashing
photo_url too would re-embed all ~5,900 chunks.

So instead: scrape ONLY the faculty listing pages (5 pages, ~20s), harvest
profile_url -> photo_url from the cards, then update metadata in place with
collection.update(). No embeddings are recomputed; the stored vectors are
untouched.

WHY THE PHOTO HAS TO REACH PROFILE CHUNKS TOO
---------------------------------------------
A faculty member's photo appears only on their department's listing-page card,
never on their own profile page. But a question about someone's research
retrieves a PROFILE chunk, not the card. If photos only lived on cards, most
faculty answers would have no image.

    python backfill_photos.py            # dry run, writes nothing
    python backfill_photos.py --apply
"""

import asyncio
import json
import sys
from collections import Counter

from bs4 import BeautifulSoup
from urllib.parse import urljoin

from Scrap_crawl4ai import scrape_all
from store import get_collection

SEED_URLS_FILE = "seed_urls.json"
APPLY = "--apply" in sys.argv
BATCH = 200
SEP = "=" * 76


async def harvest_photos():
    """
    profile_url -> photo_url, scraped from the faculty listing pages only.

    Deliberately not the whole seed list: the cards live on /faculty/ listing
    pages, and scraping the other 21 pages would add minutes for nothing.
    """
    with open(SEED_URLS_FILE, encoding="utf-8") as f:
        urls = json.load(f)

    faculty_pages = [u for u in urls if "/faculty/" in u]
    print(f"scraping {len(faculty_pages)} faculty listing pages...")

    photos = {}
    for result in await scrape_all(faculty_pages):
        if not result["success"]:
            print(f"  [SKIP] {result['url']} — {result.get('error')}")
            continue

        soup = BeautifulSoup(result["html"], "html.parser")
        found = 0
        for card in soup.select(".fc-item"):
            img = card.find("img", src=True)
            link = card.find("a", href=True)
            if not (img and link):
                continue

            profile = urljoin(result["url"], link["href"])
            photo = urljoin(result["url"], img["src"])
            if profile.startswith(("http://", "https://")) and \
               photo.startswith(("http://", "https://")):
                photos[profile] = photo
                found += 1

        print(f"  {found:>3} photos — {result['url']}")

    return photos


def main():
    photos = asyncio.run(harvest_photos())
    print(f"\n{len(photos)} faculty photos harvested\n")

    if not photos:
        print("nothing to backfill — check the .fc-item selector still matches")
        return

    print(SEP)
    print(f"backfill_photos — {'APPLY (will write)' if APPLY else 'DRY RUN'}")
    print(SEP)

    col = get_collection()
    everything = col.get(include=["metadatas"])
    ids = everything["ids"]
    metas = everything["metadatas"] or [{}] * len(ids)

    updated_ids, updated_metas = [], []
    per_person = Counter()

    for cid, meta in zip(ids, metas):
        meta = meta or {}
        source = meta.get("source_url", "")
        photo = photos.get(source)

        # Listing-page chunks: the card's own photo cannot be matched by URL,
        # because every card on a page shares that page's source_url. Those are
        # handled by the extractor on the next sync; here we only fill in the
        # PROFILE pages, which is where the gap actually hurts.
        if not photo or meta.get("photo_url") == photo:
            continue

        merged = dict(meta)
        merged["photo_url"] = photo
        updated_ids.append(cid)
        updated_metas.append(merged)
        per_person[source] += 1

    print(f"chunks in collection      : {len(ids)}")
    print(f"faculty profiles matched  : {len(per_person)}")
    print(f"chunks needing photo_url  : {len(updated_ids)}")

    print("\n--- sample ---")
    for source, n in list(per_person.items())[:6]:
        print(f"  {n:>3} chunks  {source}")
        print(f"           -> {photos[source]}")

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return

    print("\nwriting metadata (no embeddings passed => vectors untouched)...")
    for i in range(0, len(updated_ids), BATCH):
        col.update(ids=updated_ids[i:i + BATCH],
                   metadatas=updated_metas[i:i + BATCH])
        print(f"  {min(i + BATCH, len(updated_ids))}/{len(updated_ids)}")

    with_photo = sum(
        1 for m in (col.get(include=["metadatas"])["metadatas"] or [])
        if (m or {}).get("photo_url")
    )
    print(f"\ndone. {with_photo} chunks now carry a photo_url.")


if __name__ == "__main__":
    main()