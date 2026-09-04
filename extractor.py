"""
extractor.py

Turns raw scraped HTML into clean, content-addressed chunks.

Key ideas:
- Walks the HTML linearly and starts a NEW chunk whenever it hits a new
  heading (h1-h4) OR a new list item that looks like a distinct entry
  (e.g. starts with a bold date marker like "September 2013").
- This fixes the "5 unrelated events under one heading" problem, where
  a single heading gets inherited by everything below it.
- Each chunk gets a chunk_id built from a HASH of its own content, not
  its position index. This makes chunk identity stable across re-scrapes
  (needed for the delta-sync / incremental update system).
"""

from bs4 import BeautifulSoup, Tag
import hashlib
import re
from urllib.parse import urljoin
from designation import designation_metadata


def hash_text(text: str, length: int = 16) -> str:
    """Short stable fingerprint of a piece of text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:length]


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Swap for tiktoken if you want exact counts."""
    return int(len(text.split()) * 1.3)


def table_to_text(table_tag: Tag) -> str:
    """
    Converts an HTML <table> into readable pipe-separated text so its data
    survives extraction instead of being silently dropped (our walk only
    handled headings/paragraphs/list items — tables need explicit handling
    since <tr>/<td>/<th> don't match any of those).

    Example output:
        Fee Type | Fee Slab | Amritapuri | Bengaluru | Coimbatore
        Scholarship Fees | 1 | 1,25,000 | 1,25,000 | 1,50,000
        Regular Fees | 4 | 3,50,000 | 3,50,000 | 4,25,000
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        cell_texts = [c.get_text(" ", strip=True) for c in cells]
        cell_texts = [c for c in cell_texts if c]  # drop empty cells
        if cell_texts:
            rows.append(" | ".join(cell_texts))
    return "\n".join(rows)


def is_new_entry_marker(li_tag: Tag) -> bool:
    """
    Detects if this <li> starts a NEW logical entry vs continuing the
    previous one. Tuned for patterns like:
        <li><strong>September 2013</strong> ... </li>
    Adjust the regex/logic here if your source HTML uses a different
    pattern to mark new entries (e.g. different date formats, or a
    different tag like <em> instead of <strong>).
    """
    first_strong = li_tag.find(["strong", "b"])
    if first_strong:
        text = first_strong.get_text(strip=True)
        if re.match(
            r"^\s*(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}",
            text,
        ):
            return True
    return False


def extract_chunks(
    html: str,
    source_url: str,
    page_title: str,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """
    Main entry point. Takes raw HTML for ONE page and returns:

        (chunks, discovered_profile_urls, discovered_photos)

    Faculty photos come from whichever page the chunk is on: a listing card
    reads its own <img>, a profile page reads its header image. An earlier
    version carried the card photo across from pass 1 to pass 2, on the
    assumption that profile pages had no photo of their own — they do, and
    checking 12 of them found the header image present every time. The
    carry-forward is gone.

    discovered_photos is still returned because sync.py logs how many were
    found, and because it costs nothing to keep the map available.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Strip out known non-content containers BEFORE walking, so nav/menu
    # text never gets swept into a chunk. Found by inspecting this site's
    # actual HTML structure:
    #   - top navbar lives inside .mobile-menu-wrap
    #   - department sidebar menu lives inside .tab-bar.sch
    NOISE_SELECTORS = [
        ".mobile-menu-wrap",   # top navbar (Home / About / Programs / ...)
        ".secondary-nav",      # department sidebar wrapper (most reliable)
        ".tab_bar",            # department sidebar (Main Page / Faculty / ...) — note: underscore, not hyphen
        "#facilities-sec",     # "Facilities at a glance" footer widget on program pages
        ".cnt-facility-wrap",  # same footer widget, alternate selector
        "#why-amrita-sec",     # "Why Amrita" + "Overall Rankings" marketing block on program pages
        # NOTE: .elig-wrap (admission eligibility criteria) was previously excluded here
        # by mistake — it repeats across program pages because eligibility genuinely IS
        # shared across BTech programs, not because it's nav/footer junk. Removing it
        # broke retrieval for eligibility queries. Keep this content in.
        "#block-info",         # "Downloads" links widget
        "nav",
        "header",
        "footer",
        ".breadcrumb",
    ]
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Scope to main content area. Adjust this selector to match your
    # site's actual structure (inspect the HTML to find the right one).
    content_root = soup.select_one("main, article, .content, #content") or soup

    # A faculty PROFILE page carries the person's photo in its own header —
    # <div class="fac-inner-col fac-inner-left"><img class="wp-post-image">.
    # Read it here so a profile page is self-sufficient, rather than depending
    # on sync.py handing across the photo harvested from the listing card.
    #
    # Searched on `soup` rather than `content_root` because the header sits
    # outside whatever main/article wrapper the page uses.
    #
    # Verified identical to the card photo for Dr. T. K. Ramesh, so this is
    # about robustness, not a different image.
    profile_photo = ""
    profile_img = soup.select_one(
        ".fac-inner-left img[src], img.wp-post-image[src]"
    )
    if profile_img:
        resolved_profile = urljoin(source_url, profile_img["src"])
        if resolved_profile.startswith(("http://", "https://")):
            profile_photo = resolved_profile

    raw_chunks = []
    current_heading = None
    current_buffer: list[str] = []

    def flush_buffer():
        if not current_buffer:
            return
        text = "\n".join(current_buffer).strip()
        if not text:
            return
        heading_text = current_heading or "General"
        full_text = f"{heading_text}\n\n{text}" if current_heading else text
        raw_chunks.append({
            "heading": current_heading,
            "content": full_text,
            "chunk_type": "text",
        })
        current_buffer.clear()

    # Tables are handled separately from the linear walk below (table
    # internals — tr/td/th — don't match any of h1-h4/li/p, so without
    # this they'd be silently skipped entirely, dropping real data like
    # fee structures). We process all tables up front and remove them
    # from the tree so the main walk below doesn't see their contents.
    for table_tag in content_root.find_all("table"):
        # Find nearest preceding heading for context (best-effort: look
        # for a heading in earlier siblings/ancestors, fall back to None)
        heading_el = table_tag.find_previous(["h1", "h2", "h3", "h4"])
        table_heading = heading_el.get_text(strip=True) if heading_el else None

        table_text = table_to_text(table_tag)
        if table_text.strip():
            heading_text = table_heading or "General"
            # Natural-language framing sentence, so the embedding model has
            # real prose to match against queries like "what is the fee for X"
            # or "curriculum for X" — raw pipe-separated table text alone
            # embeds poorly against natural language questions.
            intro = (
                f"The following table shows {heading_text.lower()} details "
                f"for {page_title}, including the data below:"
            )
            full_text = f"{heading_text}\n\n{intro}\n\n{table_text}"
            raw_chunks.append({
                "heading": table_heading,
                "content": full_text,
                "chunk_type": "table",
            })
        table_tag.decompose()  # remove so the walk below doesn't re-process its cells

    # Faculty listing pages use a card-grid layout (name in <h6>, role in
    # plain divs/spans) that our h1-h4/li/p walker below doesn't see at
    # all — this was producing "0 chunks" for every faculty page. Treat
    # each faculty card as one self-contained chunk instead, same pattern
    # as tables: extract full text, remove from tree, skip in main walk.
    FACULTY_CARD_SELECTORS = [".fc-item"]  # found by inspecting the site's faculty grid
    discovered_profile_urls: list[str] = []
    # profile_url -> photo_url, handed to pass 2 by sync.py. get_text() throws
    # <img> away, so without capturing it here the URL is lost before chunking.
    discovered_photos: dict[str, str] = {}

    for card_selector in FACULTY_CARD_SELECTORS:
        for card_tag in content_root.select(card_selector):
            card_text = card_tag.get_text(" ", strip=True)
            card_text = re.sub(r"\s+", " ", card_text).strip()
            if not card_text:
                continue

            # Photo sits in a plain src on this site — no lazy-loading
            # placeholder to work around. The alt attribute carries the
            # person's name, which is a useful cross-check that the image
            # belongs to the card it was found in.
            img_tag = card_tag.find("img", src=True)
            card_photo = ""
            if img_tag:
                resolved_img = urljoin(source_url, img_tag["src"])
                if resolved_img.startswith(("http://", "https://")):
                    card_photo = resolved_img

            # Capture the profile page link if this card wraps one, so the
            # chatbot can point users to the individual faculty profile
            # page (bio, publications, research interests, etc.). Some
            # cards use javascript:void(0) or similar non-navigable hrefs
            # (e.g. modal-trigger cards instead of real page links) —
            # only keep genuine http(s) URLs.
            link_tag = card_tag.find("a", href=True)
            profile_url = None
            if link_tag:
                resolved = urljoin(source_url, link_tag["href"])
                if resolved.startswith("http://") or resolved.startswith("https://"):
                    profile_url = resolved

            if profile_url:
                card_text = f"{card_text} Profile: {profile_url}"
                discovered_profile_urls.append(profile_url)
                if card_photo:
                    discovered_photos[profile_url] = card_photo

            full_text = f"Faculty\n\n{card_text}"
            raw_chunks.append({
                "heading": "Faculty",
                "content": full_text,
                "chunk_type": "faculty_card",
                "photo_url": card_photo,
            })
            card_tag.decompose()  # remove so the main walk below doesn't re-process it

    for el in content_root.descendants:
        if isinstance(el, Tag) and el.name in ("h1", "h2", "h3", "h4"):
            flush_buffer()
            current_heading = el.get_text(strip=True)

        elif isinstance(el, Tag) and el.name == "li":
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if is_new_entry_marker(el):
                flush_buffer()  # close out the previous entry
            current_buffer.append(f"* {text}")

        elif isinstance(el, Tag) and el.name == "p":
            text = el.get_text(" ", strip=True)
            if text:
                current_buffer.append(text)

    flush_buffer()  # flush whatever's left at the end

    # Attach stable, content-based chunk_ids
    final_chunks = []
    for c in raw_chunks:
        content = c["content"]
        chunk_hash = hash_text(content)
        chunk_type = c.get("chunk_type", "text")
        final_chunks.append({
            "chunk_id": f"{source_url}::{chunk_type}::{chunk_hash}",
            "source_url": source_url,
            "title": page_title,
            "heading": c["heading"],
            "chunk_type": chunk_type,
            "content": content,
            "token_estimate": estimate_tokens(content),
            # A card chunk uses its own card image; every other chunk on a
            # profile page uses that page's header image.
            "photo_url": c.get("photo_url") or profile_photo or "",
            **designation_metadata(content, source_url, chunk_type),
        })

    return final_chunks, discovered_profile_urls, discovered_photos