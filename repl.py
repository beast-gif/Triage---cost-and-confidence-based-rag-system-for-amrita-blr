"""
repl.py — interactive shell for Triage. Loads the models ONCE.

WHY THIS EXISTS
---------------
Every `python generator.py "..."` starts a fresh process, and both models cache
in module-level globals that die with it. So each CLI query pays ~10s of
startup to load BAAI/bge-base-en-v1.5 and BAAI/bge-reranker-base again.

This keeps one process alive: models load on the first query, every query after
that is instant. Structurally it is the FastAPI app minus the HTTP layer, so
whatever works here will work there.

Commands:
    <question>     ask normally
    /why           full confidence breakdown for the last answer
    /chunks        the retrieved chunks for the last answer
    /raw <query>   retrieval + scoring only, no LLM call (free)
    /warm          preload models without asking anything
    /help
    /exit

Usage:
    python repl.py
"""

import asyncio
import sys
import time

SEP = "=" * 72


def warm_up():
    """Force both models to load, so the first real query is not the slow one."""
    print("loading models...", flush=True)
    started = time.time()
    from embedder import get_model
    from reranker import get_reranker
    get_model()
    get_reranker()
    print(f"ready in {time.time() - started:.1f}s")


def show_answer(out):
    print(f"\n{SEP}")
    print(out["answer"])
    print(SEP)

    if out.get("sources"):
        print("\nSources:")
        for url in out["sources"]:
            print(f"  - {url}")

    bits = [
        f"{out['band']} {out['confidence']}",
        out.get("route") or "plain",
        f"mode={out['generator_mode']}",
    ]
    if not out["llm_called"]:
        bits.append("LLM skipped")
    elif out.get("model"):
        bits.append(out["model"].split("/")[-1])
    print(f"\n[{'  |  '.join(bits)}]")


def show_why(out):
    """The confidence breakdown — what drove the band, and which signal did it."""
    d = out.get("retrieval") or {}
    if not d:
        print("  no scoring details (nothing retrieved)")
        return

    print(f"\n  final           : {out['confidence']}  ({out['band']})")
    print(f"  route           : {out.get('route') or 'plain vector search'}")
    print(f"  abs_signal      : {d.get('abs_signal')}   "
          f"(top hit strong on its own?)")
    if d.get("filtered_pool"):
        print(f"  sep_signal      : n/a  (filtered pool — no rejected set)")
        print(f"  set_confidence  : {d.get('set_confidence')}   = max(abs, rank)")
    else:
        print(f"  sep_signal      : {d.get('sep_signal')}   "
              f"(towers over the noise floor?)")
        print(f"  set_confidence  : {d.get('set_confidence')}   = max(abs, sep)")
    print(f"  rank_confidence : {d.get('rank_confidence')}   (know WHICH chunk?)")
    print(f"  agreement       : {d.get('ensemble_agreement')}")

    carrier = "abs" if (d.get("abs_signal") or 0) >= (
        d.get("sep_signal") or d.get("rank_confidence") or 0) else "sep/rank"
    print(f"\n  -> carried by the {carrier} signal")


def show_chunks(out):
    chunks = out.get("chunks") or []
    if not chunks:
        print("  no chunks retrieved")
        return
    for i, c in enumerate(chunks, start=1):
        print(f"\n  [{i}] score={c['rerank_score']:.4f}  "
              f"desig={c.get('designation') or '-'}")
        print(f"      {c['source_url']}")
        print(f"      {' '.join(c['content'].split())[:160]}")


def run_raw(query):
    """Retrieval + scoring only. No generation, so no API cost."""
    from confidence import score_query
    started = time.time()
    result = asyncio.run(score_query(query))
    elapsed = time.time() - started

    print(f"\n  {result['final_confidence']}  ({result['band']})   "
          f"route={result.get('route') or 'plain'}   "
          f"mode={result['generator_mode']}   {elapsed:.1f}s")
    for i, c in enumerate(result.get("chunks", []), start=1):
        print(f"    #{i} {c['rerank_score']:.4f}  {c['source_url'][-52:]}")
    return result


HELP = """
  <question>     ask normally
  /why           confidence breakdown for the last answer
  /chunks        retrieved chunks for the last answer
  /raw <query>   retrieval + scoring only, no LLM call (free)
  /warm          preload models
  /help          this
  /exit          quit
"""


def main():
    from generator import answer_query

    print("Triage — Amrita Bengaluru admissions assistant")
    print("type /help for commands, /exit to quit")

    last = None

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in ("/exit", "/quit"):
            break

        if line == "/help":
            print(HELP)
            continue

        if line == "/warm":
            warm_up()
            continue

        if line == "/why":
            if last:
                show_why(last)
            else:
                print("  nothing asked yet")
            continue

        if line == "/chunks":
            if last:
                show_chunks(last)
            else:
                print("  nothing asked yet")
            continue

        if line.startswith("/raw "):
            run_raw(line[5:].strip())
            continue

        if line.startswith("/"):
            print(f"  unknown command: {line}   (/help)")
            continue

        # --- a real question ---
        try:
            started = time.time()
            last = asyncio.run(answer_query(line))
            # answer_query returns the generated dict; keep the chunks around
            # for /chunks by pulling them off the scoring result it embedded
            show_answer(last)
            print(f"[{time.time() - started:.1f}s]")
        except Exception as exc:
            print(f"\n  error: {exc}")

    print("bye")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)