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
    _expectimax,
    apply_delete,
    apply_move,
    apply_swap,
    score_board,
    best_action,
    auto_depth,
    SCORE_BOARD_VERSION,
    load_trans_table,
    drain_new_entries,
    get_trans_stats,
    reset_trans_stats,
)
import cache as db
from undo_policy import analyze_undo, best_fallback_move, projected_action_eval

TARGET_TILE = 16384   # stop once this tile is reached


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    pos = (len(s) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _depth_stats(depths: list[int]) -> dict:
    if not depths:
        return {
            "depth_mean": 0.0,
            "depth_p25": 0.0,
            "depth_p50": 0.0,
            "depth_p75": 0.0,
        }
    return {
        "depth_mean": sum(depths) / len(depths),
        "depth_p25": _percentile(depths, 0.25),
        "depth_p50": _percentile(depths, 0.50),
        "depth_p75": _percentile(depths, 0.75),
    }


def _cache_bucket_key(max_tile: int) -> int:
    """Bucket key for cache stats by current max tile (0 = <512)."""
    if max_tile < 512:
        return 0
    return max(512, 1 << (max_tile.bit_length() - 1))


def _cache_bucket_label(key: int) -> str:
    if key == 0:
        return "<512"
    return f"{key}-{(key * 2) - 1}"


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
    blocked_action_once = None
    best_score_seen = 0
    powers_used = {"undo": 0, "swap": 0, "delete": 0}
    depth_samples: list[int] = []
    cache_by_tile_bucket: dict[int, dict[str, int]] = {}
    win_overlay_dismissed = False  # only dismiss once; DOM element persists hidden after dismissal
    win_overlay_retry_count = 0

    async def try_undo_recovery(state, reason: str) -> bool:
        """Attempt to recover from terminal position by spending one Undo."""
        undos = state.powers.get("undo", 0)
        if undos <= 0:
            return False
        print(
            f"\n[Move {move_count}]  {reason} — {undos} undo(s) left. "
            f"Undoing to keep going!"
        )
        print_board(state)
        board_before_undo = [row[:] for row in state.board]
        await execute_undo_on_gameover(page)
        powers_used["undo"] += 1
        # Give the game a moment to settle after the undo animation.
        await asyncio.sleep(0.4)
        state_after = await read_state(page)
        if state_after.board == board_before_undo:
            # One extra settle/read avoids consuming multiple undos on a stale
            # frame if the overlay animation finishes slightly later.
            await asyncio.sleep(0.25)
            state_after = await read_state(page)
        if state_after.board == board_before_undo:
            print(f"\n[Move {move_count}]  Undo had no effect — treating as final game over.")
            return False
        return True

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
                "cache_by_tile_bucket": dict(cache_by_tile_bucket),
                **_depth_stats(depth_samples),
            }

        # ── True game over ────────────────────────────────────────────────────
        if state.over:
            if await try_undo_recovery(state, "Game over"):
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
                "cache_by_tile_bucket": dict(cache_by_tile_bucket),
                **_depth_stats(depth_samples),
            }

        # ── Win overlay (2048+) — dismiss and keep playing ────────────────────
        # Some site variants may not expose a reliable `won` flag in worker/DOM.
        # Once 2048 is present, proactively try to click "Keep Going".
        if max_tile >= 2048 and not win_overlay_dismissed:
            print(f"\n[Move {move_count}]  *** {max_tile} tile — dismissing win overlay ***")
            clicked = await dismiss_win_overlay(page)
            await asyncio.sleep(0.15)
            state_after_dismiss = await read_state(page)
            if state_after_dismiss.won:
                win_overlay_retry_count += 1
                print(f"[Move {move_count}]  Win overlay still present; will retry dismissal.")
                if win_overlay_retry_count >= 8:
                    print(f"[Move {move_count}]  Win overlay retry cap reached; continuing anyway.")
                    win_overlay_dismissed = True
            elif clicked:
                win_overlay_dismissed = True
                win_overlay_retry_count = 0
            else:
                # No overlay signal and no click target found; keep trying briefly.
                win_overlay_retry_count += 1
                if win_overlay_retry_count >= 3:
                    win_overlay_dismissed = True
            continue

        # ── Compute depth for this move ───────────────────────────────────────
        depth = auto_depth(state.board) if auto else depth_arg
        depth_samples.append(depth)

        # Announce depth changes (auto mode)
        if auto and depth != last_depth:
            if last_depth is not None:
                print(f"\n[Move {move_count}]  depth bump: {last_depth} → {depth}  (max tile: {max_tile})")
            last_depth = depth

        # ── Choose best action (move or power-up) ─────────────────────────────
        ts_before = get_trans_stats()
        t0 = time.time()
        action = best_action(state.board, state.powers, depth=depth)
        think_ms = (time.time() - t0) * 1000
        ts_after = get_trans_stats()
        delta_hits = ts_after["hits"] - ts_before["hits"]
        delta_misses = ts_after["misses"] - ts_before["misses"]
        bkey = _cache_bucket_key(max_tile)
        bucket = cache_by_tile_bucket.setdefault(bkey, {"hits": 0, "misses": 0})
        bucket["hits"] += max(0, delta_hits)
        bucket["misses"] += max(0, delta_misses)

        if action is None:
            # Some terminal boards are detected by strategy one iteration before
            # the overlay-backed `state.over` flag updates; still allow Undo.
            refreshed = await read_state(page)
            if await try_undo_recovery(refreshed, "No valid action"):
                stuck_count = 0
                prev_board = None
                continue
            print("No valid action — game over.")
            break

        if blocked_action_once is not None:
            if action == blocked_action_once:
                blocked_direction = blocked_action_once[1] if blocked_action_once[0] == "move" else None
                fallback = best_fallback_move(
                    board=state.board,
                    powers=state.powers,
                    depth=depth,
                    blocked_direction=blocked_direction,
                    apply_move_fn=apply_move,
                    score_board_fn=score_board,
                    expectimax_fn=_expectimax,
                )
                if fallback is not None:
                    print(
                        f"\n[Move {move_count}]  Avoiding repeated action after Undo: "
                        f"{blocked_action_once} -> {fallback}"
                    )
                    action = fallback
            blocked_action_once = None

        action_type = action[0]
        planned_eval = projected_action_eval(
            state.board,
            state.powers,
            action,
            score_board_fn=score_board,
            apply_move_fn=apply_move,
            apply_swap_fn=apply_swap,
            apply_delete_fn=apply_delete,
        )

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
        try:
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
        except Exception as exc:
            print(
                f"\n[Move {move_count}]  Action execution failed "
                f"({type(exc).__name__}): {exc}"
            )
            # Recover focus and continue loop so one transient UI failure
            # does not terminate the whole run before summary output.
            try:
                await page.keyboard.press("Escape")
                await page.locator("canvas").first.click(timeout=500)
            except Exception:
                pass
            await asyncio.sleep(0.15)
            continue

        post_action_state = await read_state(page)
        undo_decision = analyze_undo(
            board_before=state.board,
            powers_before=state.powers,
            board_after=post_action_state.board,
            powers_after=post_action_state.powers,
            planned_eval=planned_eval,
            score_board_fn=score_board,
            apply_move_fn=apply_move,
        )
        if undo_decision.should_undo:
            reasons = ",".join(undo_decision.reasons)
            print(
                f"\n[Move {move_count}]  *** UNDO trigger ({reasons}) *** "
                f"drop={undo_decision.eval_drop:.1f}/{undo_decision.drop_trigger:.1f} "
                f"gap={undo_decision.plan_gap:.1f}/{undo_decision.gap_trigger:.1f} "
                f"pressure={undo_decision.pressure:.2f}"
            )
            board_after_action = [row[:] for row in post_action_state.board]
            if post_action_state.over:
                await execute_undo_on_gameover(page)
            else:
                await execute_undo(page)
            await asyncio.sleep(0.35)
            state_after_undo = await read_state(page)
            if state_after_undo.board != board_after_action:
                powers_used["undo"] += 1
                blocked_action_once = action
                move_count += 2  # count both the bad action and the undo
                stuck_count = 0
                prev_board = None
                continue
            print(f"[Move {move_count}]  Undo trigger fired, but undo had no effect.")

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
        "cache_by_tile_bucket": dict(cache_by_tile_bucket),
        **_depth_stats(depth_samples),
    }


async def run_bot(headless: bool, depth_arg, num_games: int) -> None:
    label = "auto" if depth_arg is None else str(depth_arg)
    print(f"Launching 2048 bot  (depth={label}, games={num_games}, headless={headless})")

    # ── Load transposition table from DB ──────────────────────────────────────
    print(f"Cache DB path: {db.DB_PATH}")
    version_rows = None
    try:
        versions = db.list_versions()
        version_rows = versions.get(SCORE_BOARD_VERSION, 0)
        print(f"Cache rows for current version {SCORE_BOARD_VERSION!r}: {version_rows:,}")
    except Exception as e:
        print(f"Could not read cache version counts from DB: {e}")

    def _print_load_progress(loaded: int, total: int) -> None:
        if total > 0:
            width = 34
            ratio = min(1.0, loaded / total)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\rLoading cache: [{bar}] {loaded:,}/{total:,} ({ratio*100:5.1f}%)",
                end="",
                flush=True,
            )
        else:
            print(f"\rLoading cache: {loaded:,} rows", end="", flush=True)

    t_load = time.time()
    cached = db.load_version(
        SCORE_BOARD_VERSION,
        progress_cb=_print_load_progress,
        total_rows=version_rows,
    )
    print()
    if cached:
        load_trans_table(cached)
        print(f"Loaded {len(cached):,} cached score_board entries  "
              f"(version={SCORE_BOARD_VERSION!r}, {(time.time()-t_load)*1000:.0f}ms)")
    else:
        print(f"No cached entries found for version={SCORE_BOARD_VERSION!r}  "
              f"(run populate_cache.py --generate N to seed)")

    pw, browser, page = await launch_browser(headless=headless)

    all_stats = []
    pending_db_flushes: list[tuple[int, dict, int, int, float]] = []
    try:
        for game_num in range(1, num_games + 1):
            reset_trans_stats()
            try:
                stats = await play_one_game(page, depth_arg, game_num)
            except Exception as exc:
                print(
                    f"\nGame {game_num} aborted with exception "
                    f"({type(exc).__name__}): {exc}"
                )
                try:
                    state = await read_state(page)
                    fallback_max = max(v for row in state.board for v in row)
                    stats = {
                        "game": game_num,
                        "score": state.score,
                        "best": state.best,
                        "moves": 0,
                        "max_tile": fallback_max,
                        "elapsed": 0.0,
                        "powers_used": {"undo": 0, "swap": 0, "delete": 0},
                        "cache_by_tile_bucket": {},
                        **_depth_stats([]),
                    }
                except Exception:
                    stats = {
                        "game": game_num,
                        "score": 0,
                        "best": 0,
                        "moves": 0,
                        "max_tile": 0,
                        "elapsed": 0.0,
                        "powers_used": {"undo": 0, "swap": 0, "delete": 0},
                        "cache_by_tile_bucket": {},
                        **_depth_stats([]),
                    }

            # ── Collect cache stats for this game ─────────────────────────────
            ts = get_trans_stats()
            total_lookups = ts["hits"] + ts["misses"]
            hit_rate = ts["hits"] / total_lookups * 100 if total_lookups else 0.0
            # Unique additions can be lower than misses because the same missing
            # key may be evaluated more than once in a game.
            new_entries = drain_new_entries()
            unique_additions = sum(1 for k in new_entries if k not in cached)
            pending_db_flushes.append(
                (
                    game_num,
                    new_entries,
                    ts["hits"],
                    ts["misses"],
                    hit_rate,
                )
            )
            if new_entries:
                cached.update(new_entries)   # keep local count accurate in memory

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
              f"{'Moves':>6}  {'CacheHit%':>9}  {'Depth mean/p25/p50/p75':>25}  Powers(U/S/D)")
        print("-" * 96)
        for s in all_stats:
            pu = s.get("powers_used", {})
            total = s["cache_hits"] + s["cache_misses"]
            hr = s["cache_hits"] / total * 100 if total else 0.0
            depth_tuple = (
                f"{s.get('depth_mean', 0.0):.2f}/"
                f"{s.get('depth_p25', 0.0):.2f}/"
                f"{s.get('depth_p50', 0.0):.2f}/"
                f"{s.get('depth_p75', 0.0):.2f}"
            )
            print(
                f"{s['game']:>5}  {s['score']:>7}  {s['best']:>7}  "
                f"{s['max_tile']:>8}  {s['moves']:>6}  "
                f"{hr:>8.1f}%  {depth_tuple:>25}  "
                f"U={pu.get('undo',0)} S={pu.get('swap',0)} D={pu.get('delete',0)}"
            )

        print("\nCacheHit% by max-tile bucket:")
        for s in all_stats:
            print(f"  Game {s['game']}:")
            buckets = s.get("cache_by_tile_bucket", {})
            if not buckets:
                print("    (no cache lookups)")
                continue
            for key in sorted(buckets.keys()):
                hs = buckets[key]["hits"]
                ms = buckets[key]["misses"]
                total = hs + ms
                rate = hs * 100.0 / total if total else 0.0
                print(
                    f"    {_cache_bucket_label(key):>10}: "
                    f"{rate:5.1f}%  (hits={hs:,}, misses={ms:,})"
                )

        # ── Persist cache updates after summary output ────────────────────────
        total_flush_rows = 0
        total_flush_seconds = 0.0
        if pending_db_flushes:
            print("\nPersisting cache updates to DB...")
        for game_num, new_entries, hits, misses, hit_rate in pending_db_flushes:
            print(
                f"\nCache stats (game {game_num}): "
                f"hits={hits:,}  misses={misses:,}  hit_rate={hit_rate:.1f}%  "
                f"unique_new={len(new_entries):,}"
            )
            if not new_entries:
                print("  No new entries to flush.")
                continue

            def _print_flush_progress(done: int, total: int) -> None:
                width = 28
                ratio = min(1.0, done / total) if total else 1.0
                filled = int(width * ratio)
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"\r  Flushing DB: [{bar}] {done:,}/{total:,} ({ratio*100:5.1f}%)",
                    end="",
                    flush=True,
                )

            t_flush = time.time()
            written = db.save_entries(
                new_entries,
                SCORE_BOARD_VERSION,
                progress_cb=_print_flush_progress,
            )
            flush_seconds = time.time() - t_flush
            total_flush_rows += written
            total_flush_seconds += flush_seconds
            print()
            print(
                f"  Flushed {written:,} new entries to DB in {flush_seconds:.1f}s. "
                f"(table_size={len(cached):,})"
            )
        if total_flush_rows > 0:
            print(
                f"\nDB flush complete: {total_flush_rows:,} rows written in "
                f"{total_flush_seconds:.1f}s total."
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
