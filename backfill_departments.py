"""
backfill_departments.py — tag faculty chunks with the department they belong to.

    python backfill_departments.py            # dry run, writes nothing
    python backfill_departments.py --apply

WHY THIS IS NEEDED
------------------
designation_metadata() only sets `department` when a chunk has a recognisable
job title at the start of it:

    found = leading_designations(text)
    if not found:
        return EMPTY_DESIGNATION.copy()     # department = ""

A profile chunk about someone's research interests or publications has no
leading designation, so it gets department = "". Which means the department is
missing from exactly the chunks that a question like "teachers working on image
processing in ECE" needs to filter — the research ones. The field is populated
on the chunk that says "HOD, Department of ..." and empty on the twenty around
it.

AND WHY THE LISTING PAGE IS THE RIGHT SOURCE
--------------------------------------------
department_for() INFERS the department from the chunk's own words. That works
when the text spells it out and fails silently when it does not.

The listing page does not have to infer anything. The ECE faculty page IS ECE —
every card on it belongs to that department by construction. So the department
is read from the page URL and carried across to the profile, which is the same
manoeuvre backfill_photos.py performs for photographs, for the same reason: the
listing page knows something the profile page never states.

WHY A BACKFILL RATHER THAN A RE-SYNC
------------------------------------
chunk_id hashes CONTENT only. Re-running sync.py produces identical ids, the
manifest diff reports "unchanged", and the upsert is skipped — so new metadata
never lands. That is delta-sync working correctly; it just cannot help here.
This scrapes 5 pages, then calls collection.update() with metadata alone. No
embeddings are recomputed and no stored vector is touched.

A NEW KEY, NOT AN OVERWRITE
---------------------------
This writes `department_canon` and leaves the existing `department` alone.
The department_heads route is tuned against the current values and works; there
is no reason to disturb it. The new key is always a CANONICAL name, which is
what makes an exact Chroma `where` filter possible — `department` holds free
text scraped from prose and cannot be matched exactly.
"""

import asyncio
import json
import sys
from collections import Counter
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from designation import DEPARTMENT_ALIASES
from Scrap_crawl4ai import scrape_all
from store import get_collection

SEED_URLS_FILE = "seed_urls.json"
APPLY = "--apply" in sys.argv
BATCH = 200
SEP = "=" * 76

# Path segments that are a campus, not a department. The Bengaluru School of
# Computing page is /school/computing/bengaluru/faculty/, so a naive "segment
# before /faculty/" would tag every computing lecturer as "bengaluru".
CAMPUS_SLUGS = {
    "bengaluru", "bangalore", "coimbatore", "amritapuri",
    "chennai", "kochi", "mysuru", "nagercoil", "faridabad",
}


def department_from_listing_url(url: str) -> str | None:
    """
    'https://.../bengaluru/electronics-and-communication/faculty/'
        -> 'electronics and communication'
    'https://.../school/computing/bengaluru/faculty/'
        -> 'computing'

    Walks back from /faculty/ and takes the first segment that is not a campus
    name. Resolved through DEPARTMENT_ALIASES so the stored value matches what
    department_in_query() produces for 'ECE'.
    """
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if "faculty" not in parts:
        return None

    for segment in reversed(parts[: parts.index("faculty")]):
        if segment in CAMPUS_SLUGS:
            continue
        if segment == "school":
            break
        name = segment.replace("-", " ").lower()
        return DEPARTMENT_ALIASES.get(name, name)
    return None


async def harvest_departments() -> dict[str, str]:
    """profile_url -> canonical department, from the faculty listing pages."""
    with open(SEED_URLS_FILE, encoding="utf-8") as f:
        urls = json.load(f)

    # Only the listing pages. The other 21 seeds have no .fc-item cards, and
    # scraping them would add minutes for nothing.
    faculty_pages = [u for u in urls if "/faculty/" in u]
    print(f"scraping {len(faculty_pages)} faculty listing pages...\n")

    departments: dict[str, str] = {}
    listing_departments: dict[str, str] = {}

    for result in await scrape_all(faculty_pages):
        if not result["success"]:
            print(f"  [SKIP] {result['url']} — {result.get('error')}")
            continue

        dept = department_from_listing_url(result["url"])
        if not dept:
            print(f"  [SKIP] could not read a department from {result['url']}")
            continue

        listing_departments[result["url"]] = dept

        soup = BeautifulSoup(result["html"], "html.parser")
        found = 0
        for card in soup.select(".fc-item"):
            link = card.find("a", href=True)
            if not link:
                continue
            profile = urljoin(result["url"], link["href"])
            if profile.startswith(("http://", "https://")):
                departments[profile] = dept
                found += 1

        print(f"  {found:>3} profiles -> {dept:<28} {result['url']}")

    # Chunks of the listing page itself belong to that department too — every
    # card on it does, by construction.
    departments.update(listing_departments)
    return departments


async def run(apply: bool = False) -> None:
    departments = await harvest_departments()
    print(f"\n{len(departments)} URLs mapped to a department\n")

    if not departments:
        print("nothing to backfill — check the .fc-item selector still matches")
        return

    print(SEP)
    print(f"backfill_departments — {'APPLY (will write)' if apply else 'DRY RUN'}")
    print(SEP)

    col = get_collection()
    everything = col.get(include=["metadatas"])
    ids = everything["ids"]
    metas = everything["metadatas"] or [{}] * len(ids)

    updated_ids, updated_metas = [], []
    per_dept = Counter()
    already = 0

    for cid, meta in zip(ids, metas):
        meta = meta or {}
        dept = departments.get(meta.get("source_url", ""))
        if not dept:
            continue
        if meta.get("department_canon") == dept:
            already += 1
            continue

        merged = dict(meta)
        merged["department_canon"] = dept
        updated_ids.append(cid)
        updated_metas.append(merged)
        per_dept[dept] += 1

    print(f"chunks in collection          : {len(ids)}")
    print(f"already tagged correctly      : {already}")
    print(f"chunks needing department_canon: {len(updated_ids)}")

    print("\n--- chunks per department ---")
    for dept, n in per_dept.most_common():
        print(f"  {n:>5}  {dept}")

    # The number worth looking at. If a department comes back with a handful of
    # chunks, either the listing page changed or the profile URLs on it no
    # longer match the source_url stored at scrape time — and a department
    # filter over four chunks will quietly answer badly rather than fail.
    thin = [d for d, n in per_dept.items() if n < 20]
    if thin:
        print(f"\n  WARNING — suspiciously few chunks for: {', '.join(thin)}")

    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return

    print("\nwriting metadata (no embeddings passed => vectors untouched)...")
    for i in range(0, len(updated_ids), BATCH):
        col.update(ids=updated_ids[i:i + BATCH],
                   metadatas=updated_metas[i:i + BATCH])
        print(f"  {min(i + BATCH, len(updated_ids))}/{len(updated_ids)}")

    tagged = sum(
        1 for m in (col.get(include=["metadatas"])["metadatas"] or [])
        if (m or {}).get("department_canon")
    )
    print(f"\ndone. {tagged} chunks now carry a department_canon.")


def main() -> None:
    """CLI entry point. sync.py awaits run() directly instead, because
    asyncio.run() cannot be called from inside a running event loop."""
    asyncio.run(run(apply=APPLY))


if __name__ == "__main__":
    main()