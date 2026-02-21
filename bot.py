"""
bot.py — Main loop for the 2048 bot.

Usage:
  .venv/bin/python bot.py              # visible browser, auto depth
  .venv/bin/python bot.py --headless   # headless, auto depth
  .venv/bin/python bot.py --depth 5    # fixed depth 5
  .venv/bin/python bot.py --depth auto # explicit auto (default)
  .venv/bin/python bot.py --games 5    # play 5 games then stop
"""

import asyncio
import argparse
import time

from game import (
    launch_browser, read_state,
    execute_move, execute_undo, execute_undo_on_gameover, execute_swap, execute_delete,
    dismiss_win_overlay, new_game, print_board,
)
from strategy import (
    best_action,
    auto_depth,
    SCORE_BOARD_VERSION,
    load_trans_table,
    drain_new_entries,
    get_trans_stats,
    reset_trans_stats,
)
import cache as db

TARGET_TILE = 16384   # stop once this tile is reached


async def play_one_game(page, depth_arg, game_num: int) -> dict:
    """Play a single game to completion. Returns stats dict.

    depth_arg: int for fixed depth, or None for auto.
    """
    auto = depth_arg is None
    label = "auto" if auto else str(depth_arg)
    print(f"\n{'='*50}")
    print(f"  Game {game_num} — Expectimax depth={label}  (target: {TARGET_TILE})")
    print(f"{'='*50}")

    move_count = 0
    t_start = time.time()
    last_print = -1
    last_depth = None

    prev_board = None
    stuck_count = 0
    stuck_recoveries = 0
    best_score_seen = 0
    powers_used = {"undo": 0, "swap": 0, "delete": 0}
    win_overlay_dismissed = False  # only dismiss once; DOM element persists hidden after dismissal

    while True:
        state = await read_state(page)

        # Track best score (score briefly shows 0 when win overlay is visible)
        if state.score > best_score_seen:
            best_score_seen = state.score

        max_tile = max(v for row in state.board for v in row)

        # ── Goal reached ──────────────────────────────────────────────────────
        if max_tile >= TARGET_TILE:
            elapsed = time.time() - t_start
            print(f"\n{'*'*50}")
            print(f"  *** GOAL ACHIEVED: {max_tile} tile in {move_count} moves! ***")
            print(f"{'*'*50}")
            print("Final board:")
            print_board(state)
            return {
                "game": game_num,
                "score": best_score_seen,
                "best": state.best,
                "moves": move_count,
                "max_tile": max_tile,
                "elapsed": elapsed,
                "powers_used": dict(powers_used),
            }

        # ── True game over ────────────────────────────────────────────────────
        if state.over:
            # If undos remain, use one to resume rather than ending the game.
            # After undoing, best_action re-evaluates from the previous board
            # state and may choose a power-up or a different direction.
            if state.powers.get("undo", 0) > 0:
                print(
                    f"\n[Move {move_count}]  Game over — {state.powers['undo']} undo(s) left. "
                    f"Undoing to keep going!"
                )
                print_board(state)
                board_before_undo = [row[:] for row in state.board]
                await execute_undo_on_gameover(page)
                powers_used["undo"] += 1
                # Give the game a moment to settle after the undo animation
                await asyncio.sleep(0.4)
                state_after = await read_state(page)
                if state_after.over and state_after.board == board_before_undo:
                    # Undo didn't change anything — overlay may have failed to click.
                    # Don't loop forever; treat as real game over.
                    print(f"\n[Move {move_count}]  Undo had no effect — treating as final game over.")
                    state = state_after
                else:
                    stuck_count = 0
                    prev_board = None
                    continue

            elapsed = time.time() - t_start
            print(f"\nGame over after {move_count} moves ({elapsed:.1f}s)")
            print("Final board:")
            print_board(state)
            return {
                "game": game_num,
                "score": best_score_seen,
                "best": state.best,
                "moves": move_count,
                "max_tile": max_tile,
                "elapsed": elapsed,
                "powers_used": dict(powers_used),
            }

        # ── Win overlay (2048+) — dismiss and keep playing ────────────────────
        # Guard: only dismiss once. The "You win" DOM element persists hidden
        # after dismissal, so state.won stays True forever without this flag.
        if state.won and not win_overlay_dismissed:
            print(f"\n[Move {move_count}]  *** {max_tile} tile — dismissing win overlay ***")
            await dismiss_win_overlay(page)
            win_overlay_dismissed = True
            continue

        # ── Compute depth for this move ───────────────────────────────────────
        depth = auto_depth(state.board) if auto else depth_arg

        # Announce depth changes (auto mode)
        if auto and depth != last_depth:
            if last_depth is not None:
                print(f"\n[Move {move_count}]  depth bump: {last_depth} → {depth}  (max tile: {max_tile})")
            last_depth = depth

        # ── Choose best action (move or power-up) ─────────────────────────────
        t0 = time.time()
        action = best_action(state.board, state.powers, depth=depth)
        think_ms = (time.time() - t0) * 1000

        if action is None:
            print("No valid action — game over.")
            break

        action_type = action[0]

        # ── Print status every 25 moves ───────────────────────────────────────
        if move_count % 25 == 0 and move_count != last_print:
            last_print = move_count
            pu = state.powers
            print(
                f"\n[Move {move_count}]  {action_type}:{action[1]}  depth={depth}  "
                f"powers=U{pu.get('undo',0)}/S{pu.get('swap',0)}/D{pu.get('delete',0)}  "
                f"(think: {think_ms:.0f}ms)"
            )
            print_board(state)

        # ── Execute the action ────────────────────────────────────────────────
        if action_type == "move":
            await execute_move(page, action[1])

        elif action_type == "swap":
            _, r1, c1, r2, c2 = action
            v1, v2 = state.board[r1][c1], state.board[r2][c2]
            print(
                f"\n[Move {move_count}]  *** SWAP {v1}@({r1},{c1}) <-> {v2}@({r2},{c2}) ***"
                f"  depth={depth}  (think: {think_ms:.0f}ms)"
            )
            print_board(state)
            await execute_swap(page, r1, c1, r2, c2)
            powers_used["swap"] += 1

        elif action_type == "delete":
            _, value, row, col = action
            print(
                f"\n[Move {move_count}]  *** DELETE all {value}-tiles ***"
                f"  depth={depth}  (think: {think_ms:.0f}ms)"
            )
            print_board(state)
            await execute_delete(page, row, col)
            powers_used["delete"] += 1

        # ── Stuck-board guard ─────────────────────────────────────────────────
        if state.board == prev_board and action_type == "move":
            stuck_count += 1
            if stuck_count >= 5:
                if stuck_recoveries < 2:
                    stuck_recoveries += 1
                    stuck_count = 0
                    print(
                        f"\nBoard looked stuck for 5 moves — attempting recovery "
                        f"({stuck_recoveries}/2) by refocusing canvas."
                    )
                    try:
                        await page.keyboard.press("Escape")
                        await page.locator("canvas").first.click(timeout=500)
                    except Exception:
                        pass
                    await asyncio.sleep(0.12)
                    continue
                print(f"\nBoard stuck after recovery attempts — ending game.")
                break
        else:
            stuck_count = 0
        prev_board = [row[:] for row in state.board]

        move_count += 1

    # Fallback return if loop breaks unexpectedly
    state = await read_state(page)
    return {
        "game": game_num,
        "score": max(best_score_seen, state.score),
        "best": state.best,
        "moves": move_count,
        "max_tile": max(v for row in state.board for v in row),
        "elapsed": time.time() - t_start,
        "powers_used": dict(powers_used),
    }


async def run_bot(headless: bool, depth_arg, num_games: int) -> None:
    label = "auto" if depth_arg is None else str(depth_arg)
    print(f"Launching 2048 bot  (depth={label}, games={num_games}, headless={headless})")

    # ── Load transposition table from DB ──────────────────────────────────────
    print(f"Cache DB path: {db.DB_PATH}")
    try:
        versions = db.list_versions()
        print(f"Cache rows for current version {SCORE_BOARD_VERSION!r}: "
              f"{versions.get(SCORE_BOARD_VERSION, 0):,}")
    except Exception as e:
        print(f"Could not read cache version counts from DB: {e}")

    t_load = time.time()
    cached = db.load_version(SCORE_BOARD_VERSION)
    if cached:
        load_trans_table(cached)
        print(f"Loaded {len(cached):,} cached score_board entries  "
              f"(version={SCORE_BOARD_VERSION!r}, {(time.time()-t_load)*1000:.0f}ms)")
    else:
        print(f"No cached entries found for version={SCORE_BOARD_VERSION!r}  "
              f"(run populate_cache.py --generate N to seed)")

    pw, browser, page = await launch_browser(headless=headless)

    all_stats = []
    try:
        for game_num in range(1, num_games + 1):
            reset_trans_stats()
            stats = await play_one_game(page, depth_arg, game_num)

            # ── Print cache stats for this game ───────────────────────────────
            ts = get_trans_stats()
            total_lookups = ts["hits"] + ts["misses"]
            hit_rate = ts["hits"] / total_lookups * 100 if total_lookups else 0.0
            print(f"\nCache stats (game {game_num}): "
                  f"hits={ts['hits']:,}  misses={ts['misses']:,}  "
                  f"hit_rate={hit_rate:.1f}%  table_size={len(cached) + ts['misses']:,}")

            # ── Flush new entries to DB ────────────────────────────────────────
            new_entries = drain_new_entries()
            if new_entries:
                written = db.save_entries(new_entries, SCORE_BOARD_VERSION)
                print(f"  Flushed {written:,} new entries to DB.")
                cached.update(new_entries)   # keep local count accurate

            stats["cache_hits"]   = ts["hits"]
            stats["cache_misses"] = ts["misses"]
            all_stats.append(stats)

            if game_num < num_games:
                print(f"\nStarting new game in 2s…")
                await asyncio.sleep(2)
                await new_game(page)
                # Wait for fresh board
                await asyncio.sleep(0.8)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Summary — {num_games} game(s)")
        print(f"{'='*60}")
        print(f"{'Game':>5}  {'Score':>7}  {'Best':>7}  {'MaxTile':>8}  "
              f"{'Moves':>6}  {'CacheHit%':>9}  Powers(S/D)")
        print("-" * 68)
        for s in all_stats:
            pu = s.get("powers_used", {})
            total = s["cache_hits"] + s["cache_misses"]
            hr = s["cache_hits"] / total * 100 if total else 0.0
            print(
                f"{s['game']:>5}  {s['score']:>7}  {s['best']:>7}  "
                f"{s['max_tile']:>8}  {s['moves']:>6}  "
                f"{hr:>8.1f}%  "
                f"S={pu.get('swap',0)} D={pu.get('delete',0)}"
            )

        if not headless:
            print("\nKeeping browser open for 25s so you can see the final state…")
            await asyncio.sleep(25)

    finally:
        await browser.close()
        await pw.stop()


def main():
    parser = argparse.ArgumentParser(description="2048 bot using Expectimax AI")
    parser.add_argument("--headless", action="store_true", help="Run without visible browser")
    parser.add_argument("--depth", default="auto",
                        help="Expectimax search depth: integer for fixed, 'auto' for board-state adaptive depth (default: auto)")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play (default 1)")
    args = parser.parse_args()

    if args.depth == "auto":
        depth_arg = None
    else:
        try:
            depth_arg = int(args.depth)
        except ValueError:
            parser.error(f"--depth must be an integer or 'auto', got: {args.depth!r}")

    asyncio.run(run_bot(
        headless=args.headless,
        depth_arg=depth_arg,
        num_games=args.games,
    ))


if __name__ == "__main__":
    main()
