"""
backfill_designations.py — write designation metadata onto existing Chroma chunks.

NO RE-SCRAPE. NO RE-EMBEDDING.
------------------------------
The designation string ("HOD", "Vice-Chairperson") is already inside the chunk
TEXT that Chroma is storing. So this reads documents out of the collection,
parses them with designation.py, and writes metadata back via
collection.update(ids=..., metadatas=...).

Because chunk_ids are content hashes and the chunk text is unchanged, the ids
still match exactly and the existing embeddings stay valid. Passing no
embeddings to update() leaves the stored vectors untouched. This is design
decision #1 from the handoff paying off — worth putting in the report as a
measured result rather than a claimed benefit.

Cost: a few seconds of local CPU. No API calls. No network.

WHY DRY-RUN BY DEFAULT
----------------------
This still WRITES to the live chroma_db/. Run it without --apply first, read
the summary, and only then commit.

Usage:
    python backfill_designations.py                 # dry run, writes nothing
    python backfill_designations.py --apply         # commit metadata
    python backfill_designations.py --apply --verify  # commit, then re-run the ECE query
"""

import sys
from collections import Counter

from designation import designation_metadata, parse_chunk_id
from store import get_collection

APPLY = "--apply" in sys.argv
VERIFY = "--verify" in sys.argv
BATCH = 200

VERIFY_QUERY = "Chairperson of Electronics and communication"

SEP = "=" * 78


def url_of(meta):
    meta = meta or {}
    for key in ("source_url", "url", "page_url", "source"):
        if meta.get(key):
            return meta[key]
    return "(no url)"


# ---------------------------------------------------------------------------
# read + parse
# ---------------------------------------------------------------------------
print(SEP)
print(f"backfill_designations — {'APPLY (will write)' if APPLY else 'DRY RUN (no writes)'}")
print(SEP)

col = get_collection()
everything = col.get(include=["documents", "metadatas"])

ids = everything["ids"]
docs = everything["documents"]
metas = everything["metadatas"] or [{}] * len(ids)

print(f"chunks in collection: {len(ids)}\n")

updated_ids, updated_metas = [], []
counts = Counter()
type_counts = Counter()
heads = []
gated_in = 0

for cid, doc, old_meta in zip(ids, docs, metas):
    # Provenance comes from the chunk_id itself ('{url}::{type}::{hash}'),
    # so this does not depend on what store.py copies into Chroma metadata.
    source_url, chunk_type = parse_chunk_id(cid)
    type_counts[chunk_type or "(unparsed)"] += 1

    new_fields = designation_metadata(doc, source_url, chunk_type)
    counts[new_fields["designation"] or "(none)"] += 1
    if new_fields["designation"]:
        gated_in += 1

    if new_fields["is_department_head"]:
        heads.append((cid, new_fields, doc))

    # MERGE, never replace: Chroma's update() overwrites the whole metadata
    # dict for an id, so anything already there (source_url, chunk type, page
    # title) must be carried forward explicitly or it is silently lost.
    merged = dict(old_meta or {})
    merged.update(new_fields)

    if merged != (old_meta or {}):
        updated_ids.append(cid)
        updated_metas.append(merged)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
print("--- chunk types in collection ---")
for value, n in type_counts.most_common():
    print(f"  {n:>6}  {value}")

print(f"\n--- designation distribution (gate let {gated_in}/{len(ids)} through) ---")
for value, n in counts.most_common():
    print(f"  {n:>6}  {value}")

print(f"\n--- department heads found ({len(heads)}) ---")
for cid, fields, doc in heads:
    print(f"\n  raw title : {fields['designation_raw']}")
    print(f"  department: {fields['department'] or '(EMPTY — failed validation)'}")
    print(f"  url       : {parse_chunk_id(cid)[0]}")
    print(f"  text      : {' '.join(doc.split())[:110]}")

depts = Counter(f["department"] for _, f, _ in heads if f["department"])
blank_depts = sum(1 for _, f, _ in heads if not f["department"])
if depts:
    print("\n--- distinct department strings among heads ---")
    print("    (near-duplicates mean normalize_department needs another rule)")
    for d, n in depts.most_common():
        print(f"  {n:>3}  {d}")
print(f"\n  heads with EMPTY department: {blank_depts}/{len(heads)}")
print("    (empty is fail-closed and safe — garbage would poison the filter)")

print(f"\nchunks whose metadata would change: {len(updated_ids)}")

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
if not APPLY:
    print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    sys.exit(0)

print("\nwriting metadata (no embeddings passed => stored vectors untouched)...")
for i in range(0, len(updated_ids), BATCH):
    batch_ids = updated_ids[i:i + BATCH]
    batch_metas = updated_metas[i:i + BATCH]
    col.update(ids=batch_ids, metadatas=batch_metas)
    print(f"  {min(i + BATCH, len(updated_ids))}/{len(updated_ids)}")

print(f"\ndone. {len(updated_ids)} chunks updated. Collection count: {col.count()}")

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
if not VERIFY:
    print("\n(pass --verify to re-run the ECE query against the new metadata)")
    sys.exit(0)

print("\n" + SEP)
print("VERIFY — filtered retrieval vs unfiltered")
print(SEP)

from embedder import embed_query  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

qvec = embed_query(VERIFY_QUERY)
qvec = qvec.tolist() if hasattr(qvec, "tolist") else list(qvec)
ce = CrossEncoder("BAAI/bge-reranker-base")


def show(title, where=None):
    res = col.query(
        query_embeddings=[qvec],
        n_results=15,
        where=where,
        include=["documents", "metadatas"],
    )
    cand_docs = res["documents"][0]
    cand_metas = res["metadatas"][0]
    if not cand_docs:
        print(f"\n{title}\n  (no candidates matched the filter)")
        return

    logits = [float(x) for x in ce.predict([(VERIFY_QUERY, d) for d in cand_docs])]
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)

    print(f"\n{title}  ({len(cand_docs)} candidates)")
    for pos, i in enumerate(order[:5], start=1):
        m = cand_metas[i]
        print(f"  #{pos} logit={logits[i]:+.4f}  desig={m.get('designation', '?'):<20}"
              f" {' '.join(cand_docs[i].split())[:70]}")


show("BEFORE — unfiltered (what you have today)")
show("AFTER  — filtered to is_department_head=True",
     where={"is_department_head": True})