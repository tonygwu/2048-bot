#!/usr/bin/env python3
"""
populate_cache.py — Seed or refresh the score_board transposition table.

Modes
─────
--recompute
    Read every board state already stored in the DB (any version), recompute
    score_board with the current SCORE_BOARD_VERSION, and write the results back.
    Use this after bumping the version to pre-populate the new version's rows
    from boards that were already discovered.

--generate N  [--depth D]
    Simulate N game positions (no browser needed) by running the expectimax bot
    in self-play at the given depth (default 3) and collecting board states.
    All newly computed states are written to the DB under SCORE_BOARD_VERSION.

Usage examples
──────────────
  # After bumping SCORE_BOARD_VERSION, refresh the cache for all known boards:
  .venv/bin/python populate_cache.py --recompute

  # Generate 500 board positions at depth 3 and cache them:
  .venv/bin/python populate_cache.py --generate 500

  # Generate 200 positions at depth 4 (slower but higher-quality boards):
  .venv/bin/python populate_cache.py --generate 200 --depth 4

  # Show what's in the DB:
  .venv/bin/python populate_cache.py --list
"""

import argparse
import multiprocessing as mp
import os
import random
import time

import cache as db
from strategy import (
    SCORE_BOARD_VERSION,
    DIRECTIONS,
    apply_move,
    score_board,
    best_action,
    board_to_bb,
    drain_new_entries,
    load_trans_table,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_start() -> list[list[int]]:
    """Return a fresh 4×4 board with two random starting tiles (2 or 4)."""
    board = [[0] * 4 for _ in range(4)]
    cells = [(r, c) for r in range(4) for c in range(4)]
    for r, c in random.sample(cells, 2):
        board[r][c] = random.choice([2, 2, 4])   # 2/3 chance of 2
    return board


def _place_random_tile(board: list[list[int]]) -> list[list[int]]:
    """Place a 2 (90%) or 4 (10%) in a random empty cell.  Returns new board."""
    empties = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empties:
        return board
    r, c = random.choice(empties)
    nb = [row[:] for row in board]
    nb[r][c] = 2 if random.random() < 0.9 else 4
    return nb


def _is_game_over(board: list[list[int]]) -> bool:
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            return False
    return True


# ── Recompute mode ────────────────────────────────────────────────────────────

def _score_state(task: tuple[int, int, int]) -> tuple[tuple[int, int, int], float]:
    """Worker helper: score one (board_bb, swap_uses, delete_uses) state."""
    bb, su, du = task
    board = db.decode_board(bb)
    powers = {"swap": su, "delete": du}
    score = score_board(board, powers)
    key = (board_to_bb(board), su, du)
    return key, score


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def cmd_recompute(
    target_version: str,
    workers: int,
    write_chunk: int,
    progress_every: int,
    only_missing_current: bool,
    limit: int | None,
    offset: int,
) -> None:
    """Recompute score_board for all states in the DB under SCORE_BOARD_VERSION."""
    if not target_version:
        raise ValueError("target_version must be non-empty")
    tasks = db.get_recompute_states(
        current_version=target_version,
        only_missing_current=only_missing_current,
        limit=limit,
        offset=offset,
    )
    if not tasks:
        print("DB is empty — nothing to recompute.")
        return
    total = len(tasks)
    workers = max(0, workers)
    if workers == 0:
        workers = max(1, (os.cpu_count() or 1) - 1)
    write_chunk = max(1, write_chunk)
    progress_every = max(1, progress_every)

    mode = "missing-current only" if only_missing_current else "all states"
    limit_str = "none" if limit is None else f"{limit:,}"
    print(
        f"Recomputing {total:,} unique states as version {target_version!r} "
        f"(mode={mode}, offset={offset:,}, limit={limit_str}, workers={workers}, "
        f"write_chunk={write_chunk:,}, progress_every={progress_every:,})…"
    )

    t0 = time.perf_counter()
    written_total = 0
    entries: dict[tuple[int, int, int], float] = {}
    if workers == 1:
        iterator = (_score_state(task) for task in tasks)
    else:
        chunksize = max(64, min(2048, total // (workers * 50) or 64))
        pool = mp.Pool(processes=workers)
        iterator = pool.imap_unordered(_score_state, tasks, chunksize=chunksize)

    processed = 0
    try:
        for key, score in iterator:
            processed += 1
            entries[key] = score

            if len(entries) >= write_chunk:
                written_total += db.save_entries(entries, target_version)
                entries.clear()

            if processed % progress_every == 0 or processed == total:
                elapsed = time.perf_counter() - t0
                rate = processed / elapsed if elapsed > 0 else 0.0
                remain = total - processed
                eta = _fmt_eta(remain / rate) if rate > 0 else "?"
                print(
                    f"  {processed:,}/{total:,} ({processed / total * 100:.1f}%)  "
                    f"{rate:,.0f} states/s  elapsed={elapsed:.1f}s  ETA={eta}",
                    flush=True,
                )

        if entries:
            written_total += db.save_entries(entries, target_version)
            entries.clear()
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    elapsed = time.perf_counter() - t0
    rate = written_total / elapsed if elapsed > 0 else 0.0
    print(f"  Written {written_total:,} rows in {elapsed:.2f}s  ({rate:,.0f} states/s)")


# ── Generate mode ─────────────────────────────────────────────────────────────

def cmd_generate(n_positions: int, depth: int) -> None:
    """Play N self-play moves at the given depth, collecting unique board states."""
    # Pre-load existing cache so generate benefits from already-cached scores
    existing = db.load_version(SCORE_BOARD_VERSION)
    if existing:
        load_trans_table(existing)
        print(f"Loaded {len(existing):,} cached states from DB.")

    print(f"Generating {n_positions:,} positions at depth {depth}…")
    t0 = time.perf_counter()

    collected = 0
    moves_total = 0
    game = 1

    while collected < n_positions:
        board = _random_start()
        powers = {}   # no power-ups in self-play simulation

        while collected < n_positions:
            if _is_game_over(board):
                break

            # score_board caches internally; best_action triggers many calls
            action = best_action(board, powers={}, depth=depth)
            if action is None:
                break

            collected += 1
            moves_total += 1

            if action[0] == "move":
                nb, _, changed = apply_move(board, action[1])
                if not changed:
                    break
                board = _place_random_tile(nb)
            else:
                break   # power-up actions not handled in headless sim

            if collected % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {collected:,}/{n_positions:,} positions  "
                      f"({elapsed:.1f}s, game {game})")

        game += 1

    elapsed = time.perf_counter() - t0
    new_entries = drain_new_entries()
    written = db.save_entries(new_entries, SCORE_BOARD_VERSION)
    print(f"\nDone: {collected:,} positions in {elapsed:.1f}s  "
          f"({moves_total} moves, {game - 1} games)")
    print(f"  Wrote {written:,} new DB entries  "
          f"(total _TRANS_TABLE size: {len(new_entries) + len(existing):,})")


# ── List mode ─────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    versions = db.list_versions()
    if not versions:
        print("DB is empty (or not yet created).")
        return
    print(f"{'Version':<12}  {'Rows':>10}")
    print("-" * 26)
    for v in sorted(versions):
        marker = "  ← current" if v == SCORE_BOARD_VERSION else ""
        print(f"{v!r:<12}  {versions[v]:>10,}{marker}")
    print(f"\nTotal rows: {sum(versions.values()):,}")
    print(f"DB path: {db.DB_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate the score_board cache")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recompute", action="store_true",
                       help="Recompute all DB states with current SCORE_BOARD_VERSION")
    group.add_argument("--generate", type=int, metavar="N",
                       help="Simulate N game positions and cache score_board results")
    group.add_argument("--list", action="store_true",
                       help="Show row counts per version in the DB")
    parser.add_argument("--depth", type=int, default=3,
                        help="Expectimax depth for --generate (default: 3)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes for --recompute (0=auto, 1=single-process)")
    parser.add_argument("--write-chunk", type=int, default=50_000,
                        help="Rows per DB write batch in --recompute (default: 50000)")
    parser.add_argument("--progress-every", type=int, default=50_000,
                        help="Print progress every N states in --recompute (default: 50000)")
    parser.add_argument("--only-missing-current", action="store_true",
                        help="In --recompute, process only states missing the current version")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max unique states to recompute (supports batching)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many ordered unique states before recomputing")
    parser.add_argument("--target-version", type=str, default=SCORE_BOARD_VERSION,
                        help="Version label to write during --recompute (default: SCORE_BOARD_VERSION)")
    args = parser.parse_args()

    if args.recompute:
        cmd_recompute(
            args.target_version.strip(),
            args.workers,
            args.write_chunk,
            args.progress_every,
            args.only_missing_current,
            args.limit,
            args.offset,
        )
    elif args.generate is not None:
        cmd_generate(args.generate, args.depth)
    elif args.list:
        cmd_list()


if __name__ == "__main__":
    main()
