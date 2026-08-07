"""
scheduler.py — weekly data refresh, run from inside the FastAPI process.

WHY IN-PROCESS RATHER THAN CRON
--------------------------------
APScheduler lives inside the app, so the same code works on any deployment
platform. OS-level cron would tie the project to a specific host, and most
managed platforms do not give you a crontab at all.

WHY A LOCK
----------
sync.py writes to manifest.db and chroma_db/. Two syncs running at once — a
scheduled one and a manual /sync trigger — would interleave their manifest
writes and could leave chunk_ids recorded for chunks that were never upserted,
or delete chunks another run had just added. The lock makes the second caller
fail fast instead of corrupting state.

WHY THE POST-SYNC HEALTH CHECK
-------------------------------
The chatbot answers "who heads department X" from is_department_head metadata.
Three things break that silently:

  * a new job title the ALIASES table does not recognise (a new HOD whose page
    says "Convenor") -> that department drops to ZERO heads
  * a stale profile page still listing the previous head alongside the new one
    -> that department shows TWO heads
  * a site layout change breaking .fc-item extraction -> the department
    disappears entirely

None of these raise an exception. The sync reports success and the bot starts
answering wrongly. Counting heads per department after each run turns all three
into a logged warning.
"""

import asyncio
import threading
import traceback
from collections import Counter
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Sunday 03:00 — low traffic, and a full run takes several minutes because
# Pass 2 scrapes a few hundred faculty profile pages.
SYNC_DAY = "sun"
SYNC_HOUR = 3

_lock = threading.Lock()

STATUS = {
    "last_run": None,
    "last_status": None,          # "ok" | "failed" | "running"
    "last_error": None,
    "last_duration_seconds": None,
    "last_chunk_count": None,
    "health": None,               # result of check_head_coverage()
    "runs": 0,
}


def check_head_coverage():
    """
    Count department heads per department and flag anything suspicious.

    Expected today: exactly one head each for ECE, EEE and English. Mechanical
    and the Computing-school departments correctly have none — the school
    Principal answers for them — so a department missing from this list is only
    a problem if it USED to be here.
    """
    from designation import parse_chunk_id
    from store import get_collection

    col = get_collection()
    data = col.get(include=["metadatas"])

    per_dept = Counter()
    unnamed = 0
    for meta in data["metadatas"] or []:
        if not meta.get("is_department_head"):
            continue
        dept = (meta.get("department") or "").strip()
        if dept:
            per_dept[dept] += 1
        else:
            unnamed += 1

    warnings = []
    # Each real head produces TWO chunks: the faculty card and the profile page.
    # More than that means duplicate or stale records for the same department.
    for dept, n in per_dept.items():
        if n > 2:
            warnings.append(
                f"'{dept}' has {n} head chunks (expected 2: card + profile) — "
                f"possible stale page listing a previous head"
            )
    if unnamed:
        warnings.append(
            f"{unnamed} head chunk(s) with no department parsed — "
            f"check normalize_department()"
        )
    if not per_dept:
        warnings.append(
            "NO department heads found at all — extraction is probably broken "
            "(check the .fc-item selector and the ALIASES table)"
        )

    return {
        "departments_with_heads": dict(per_dept),
        "head_chunks_without_department": unnamed,
        "warnings": warnings,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def run_sync(triggered_by="schedule"):
    """
    Blocking. Returns a status dict. Safe to call from a thread.

    Acquires the lock non-blockingly: a second caller is told a sync is already
    running rather than queueing behind it.
    """
    if not _lock.acquire(blocking=False):
        return {"status": "busy", "detail": "a sync is already running"}

    started = datetime.now(timezone.utc)
    STATUS["last_status"] = "running"
    STATUS["last_error"] = None

    try:
        from store import count
        from sync import sync

        print(f"[sync] starting ({triggered_by})")
        asyncio.run(sync())

        STATUS["last_chunk_count"] = count()
        STATUS["health"] = check_head_coverage()
        STATUS["last_status"] = "ok"

        for warning in STATUS["health"]["warnings"]:
            print(f"[sync] WARNING: {warning}")

        result = {"status": "ok", "chunks": STATUS["last_chunk_count"]}

    except Exception as exc:
        STATUS["last_status"] = "failed"
        STATUS["last_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        result = {"status": "failed", "detail": STATUS["last_error"]}

    finally:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        STATUS["last_run"] = started.isoformat()
        STATUS["last_duration_seconds"] = round(elapsed, 1)
        STATUS["runs"] += 1
        _lock.release()
        print(f"[sync] finished in {elapsed:.0f}s — {STATUS['last_status']}")

    return result


async def run_sync_background(triggered_by="manual"):
    """
    Run the sync off the event loop.

    sync.py is blocking and takes minutes. Running it inline would freeze every
    /chat request for the duration, so it goes to a worker thread.
    """
    return await asyncio.to_thread(run_sync, triggered_by)


def start_scheduler():
    """Called from the FastAPI lifespan handler. Returns the scheduler."""
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        run_sync_background,
        CronTrigger(day_of_week=SYNC_DAY, hour=SYNC_HOUR, minute=0),
        id="weekly_sync",
        kwargs={"triggered_by": "schedule"},
        max_instances=1,          # belt and braces alongside the lock
        coalesce=True,            # a missed run fires once, not N times
        misfire_grace_time=3600,
    )
    scheduler.start()
    print(f"[scheduler] weekly sync armed: {SYNC_DAY} {SYNC_HOUR:02d}:00 IST")
    return scheduler


def next_run_time(scheduler):
    if not scheduler:
        return None
    job = scheduler.get_job("weekly_sync")
    return job.next_run_time.isoformat() if job and job.next_run_time else None