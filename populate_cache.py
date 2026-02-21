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

def cmd_recompute() -> None:
    """Recompute score_board for all states in the DB under SCORE_BOARD_VERSION."""
    states = db.get_all_states()   # all versions
    if not states:
        print("DB is empty — nothing to recompute.")
        return

    # Deduplicate by (board_bb, swap_uses, delete_uses) — keep one per key
    unique: dict[tuple, tuple] = {}
    for bb, su, du, _ver, _sc in states:
        unique[(bb, su, du)] = (bb, su, du)

    print(f"Recomputing {len(unique):,} unique states as version {SCORE_BOARD_VERSION!r}…")

    t0 = time.perf_counter()
    entries = {}
    for bb, su, du in unique.values():
        board = db.decode_board(bb)
        powers = {"swap": su, "delete": du}
        score = score_board(board, powers)
        key = (board_to_bb(board), su, du)
        entries[key] = score

    elapsed = time.perf_counter() - t0
    written = db.save_entries(entries, SCORE_BOARD_VERSION)
    print(f"  Written {written:,} rows in {elapsed:.2f}s  "
          f"({written / elapsed:.0f} states/s)")


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
    args = parser.parse_args()

    if args.recompute:
        cmd_recompute()
    elif args.generate is not None:
        cmd_generate(args.generate, args.depth)
    elif args.list:
        cmd_list()


if __name__ == "__main__":
    main()
