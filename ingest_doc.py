"""
ingest_doc.py — parse, embed and store one uploaded document.

    python ingest_doc.py calender.pdf --title "Academic Calendar 2026-27"
    python ingest_doc.py --list
    python ingest_doc.py --delete academic-calendar-2026-27

WHY THE TITLE MATTERS MORE THAN IT LOOKS
----------------------------------------
Every chunk is prefixed "From {title}, page {n}:". That prefix is the only
context a chunk carries once it is split away from its neighbours.

Measured on the academic calendar: part 2 of the July page held

    29-Jul | Wed | H | Guru Poornima (Holiday)

with no year anywhere in the chunk. Titling the upload "Academic Calendar
2026-27" rather than letting it default to the filename puts the year in every
single chunk header, at zero parsing cost. Prefer a descriptive title.

This is the CLI. The admin site will call the same functions.
"""

import os
import re
import sys

from doc_parser import SUPPORTED, parse_document
from embedder import embed_texts
from upload_store import add_document, count, delete_document, list_documents


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def ingest(path, title=None, doc_id=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported type '{ext}' — {', '.join(sorted(SUPPORTED))}")

    filename = os.path.basename(path)
    title = title or os.path.splitext(filename)[0]
    doc_id = doc_id or slugify(title)

    print(f"parsing {filename}...")
    chunks = parse_document(path, doc_id=doc_id, title=title)
    if not chunks:
        raise ValueError(
            "the parser found no text — if this is a scanned PDF it needs OCR, "
            "which is not supported"
        )

    sizes = [len(c["content"]) for c in chunks]
    print(f"  {len(chunks)} chunks, {min(sizes)}-{max(sizes)} chars")

    print("embedding...")
    embeddings = embed_texts([c["content"] for c in chunks])

    record = add_document(chunks, embeddings, doc_id, title, filename)
    print(f"\nstored as '{doc_id}'")
    print(f"  title  : {record['title']}")
    print(f"  chunks : {record['chunks']} across {record['pages']} page(s)")
    print(f"  store  : {count()} chunks total in the upload collection")
    return record


def show_list():
    docs = list_documents()
    if not docs:
        print("no documents uploaded")
        return
    print(f"{len(docs)} document(s), {count()} chunks total\n")
    for d in docs:
        print(f"  {d['doc_id']}")
        print(f"    {d['title']}  —  {d['chunks']} chunks, {d['pages']} page(s)")
        print(f"    {d['filename']}, uploaded {d['uploaded_at'][:19]}Z\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--list" in sys.argv:
        show_list()
        sys.exit(0)

    if "--delete" in sys.argv:
        if not args:
            print("usage: python ingest_doc.py --delete <doc_id>")
            sys.exit(1)
        removed = delete_document(args[0])
        if removed:
            print(f"deleted '{args[0]}' ({removed['chunks']} chunks)")
        else:
            print(f"no document with id '{args[0]}'")
            show_list()
        sys.exit(0)

    if not args:
        print(__doc__)
        sys.exit(1)

    title = None
    if "--title" in sys.argv:
        i = sys.argv.index("--title")
        if i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
            if title in args:
                args.remove(title)

    try:
        ingest(args[0], title=title)
    except Exception as exc:
        print(f"\nfailed: {exc}")
        sys.exit(1)