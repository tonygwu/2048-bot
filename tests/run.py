#!/usr/bin/env python3
"""
Test harness for 2048 strategy — run the bot against fixture board states
without launching a browser.

Usage:
  .venv/bin/python tests/run.py                            # list available boards
  .venv/bin/python tests/run.py mid_game                   # run 10 moves
  .venv/bin/python tests/run.py mid_game --moves 20
  .venv/bin/python tests/run.py late_game --depth 5
  .venv/bin/python tests/run.py mid_game --seed 42         # reproducible tile spawns
  .venv/bin/python tests/run.py jammed --scores            # show eval score per direction
  .venv/bin/python tests/run.py swap_test --peek           # show first action only, then stop
  .venv/bin/python tests/run.py mid_game --no-random       # skip random tile placement
  .venv/bin/python tests/run.py late_game --depth 2 --moves 80 --episodes 20 --write-to-cache
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Allow importing from the project root regardless of where the script is run from.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cache as db
from sim_utils import count_created_tile, place_random_tile, recharge_delete_uses
from strategy import (
    DIRECTIONS,
    SEARCH_CACHE_VERSION,
    SCORE_BOARD_VERSION,
    apply_delete,
    apply_move,
    apply_swap,
    auto_depth,
    best_action,
    drain_new_entries,
    drain_search_new_entries,
    is_game_over,
    score_board,
    _expectimax,
)
from undo_policy import analyze_undo, best_fallback_move, projected_action_eval


# ── Lightweight board display (no playwright dependency) ──────────────────────

def print_board(board: list[list[int]], score: int = 0, powers: dict | None = None) -> None:
    pu = powers or {}
    pu_str = f"  U={pu.get('undo',0)} S={pu.get('swap',0)} D={pu.get('delete',0)}" if pu else ""
    print(f"Score: {score}{pu_str}")
    print("+------+------+------+------+")
    for row in board:
        cells = "".join(f"{v:^6}" if v else "      " for v in row)
        print(f"|{cells[0:6]}|{cells[6:12]}|{cells[12:18]}|{cells[18:24]}|")
        print("+------+------+------+------+")

BOARDS_DIR = Path(__file__).parent / "boards"


# ── Board loading ─────────────────────────────────────────────────────────────

def resolve_board_path(name: str) -> Path:
    """Resolve a board name or path to an existing .json file."""
    # Try as a literal path first.
    p = Path(name)
    if p.exists():
        return p
    # Try relative to BOARDS_DIR, with/without .json suffix.
    for candidate in [BOARDS_DIR / name, BOARDS_DIR / (name + ".json")]:
        if candidate.exists():
            return candidate
    return p   # will cause a clean FileNotFoundError later


def load_fixture(name: str) -> dict:
    path = resolve_board_path(name)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: board not found: {name!r}")
        print(f"\nAvailable boards:")
        list_boards()
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}")
        sys.exit(1)
    # Basic validation
    board = data.get("board")
    if not isinstance(board, list) or len(board) != 4 or any(len(r) != 4 for r in board):
        print(f"Error: 'board' must be a 4×4 list in {path}")
        sys.exit(1)
    return data


def list_boards() -> None:
    boards = sorted(BOARDS_DIR.glob("*.json"))
    if not boards:
        print("  (no boards found in tests/boards/)")
        return
    for b in boards:
        try:
            with open(b) as f:
                data = json.load(f)
            max_tile = max(v for row in data["board"] for v in row)
            score    = data.get("score", 0)
            desc     = data.get("description", "")
            powers   = data.get("powers", {})
            pu_str   = ""
            if any(powers.get(k, 0) > 0 for k in ("undo", "swap", "delete")):
                pu_str = "  powers=" + "/".join(
                    f"{k[0].upper()}{powers[k]}" for k in ("undo", "swap", "delete") if powers.get(k, 0) > 0
                )
            print(f"  {b.stem:<20}  max={max_tile:>5}  score={score:>6}{pu_str}  {desc}")
        except Exception as e:
            print(f"  {b.stem}  (parse error: {e})")


def direction_scores(board: list[list[int]], depth: int, powers: dict | None = None) -> list[tuple[str, float | None]]:
    """Return (direction, expectimax_value) for each direction, None if the move is invalid."""
    result = []
    for d in DIRECTIONS:
        nb, delta, changed = apply_move(board, d)
        if not changed:
            result.append((d, None))
            continue
        val = _expectimax(nb, max(0, depth - 1), False, powers=powers) + delta
        result.append((d, val))
    return result


def print_direction_scores(board: list[list[int]], depth: int, powers: dict | None = None) -> None:
    scores = direction_scores(board, depth, powers=powers)
    valid = [(d, v) for d, v in scores if v is not None]
    best_d = max(valid, key=lambda x: x[1])[0] if valid else None
    print("  Direction scores:")
    for d, val in scores:
        if val is None:
            print(f"    {d:<6}:  (no change)")
        else:
            marker = "  ◀ best" if d == best_d else ""
            print(f"    {d:<6}: {val:>12.1f}{marker}")


def apply_action(
    board: list[list[int]],
    score: int,
    powers: dict,
    action: tuple,
    *,
    rng: random.Random,
    no_random: bool,
) -> tuple[list[list[int]], int, dict]:
    """Apply one action in the local simulator and return updated state."""
    action_type = action[0]
    next_board = [row[:] for row in board]
    next_score = int(score)
    next_powers = dict(powers)

    if action_type == "move":
        _, direction = action
        created_512 = count_created_tile(next_board, direction, 512)
        moved, delta, changed = apply_move(next_board, direction)
        if not changed:
            raise ValueError(f"Invalid move action (no board change): {action!r}")
        next_board = moved
        next_score += delta
        next_powers = recharge_delete_uses(next_powers, created_512)
    elif action_type == "swap":
        _, r1, c1, r2, c2 = action
        next_board = apply_swap(next_board, r1, c1, r2, c2)
        next_powers["swap"] = max(0, next_powers.get("swap", 0) - 1)
    elif action_type == "delete":
        _, value, _, _ = action
        next_board = apply_delete(next_board, value)
        next_powers["delete"] = max(0, next_powers.get("delete", 0) - 1)
    else:
        raise ValueError(f"Unknown action type: {action_type!r}")

    if not no_random:
        next_board = place_random_tile(next_board, rng)
    return next_board, next_score, next_powers


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(
    fixture: dict,
    num_moves: int,
    fixed_depth: int | None,
    rng: random.Random,
    show_scores: bool,
    peek: bool,
    no_random: bool,
    *,
    write_to_cache: bool = False,
    cache_target_version: str = SCORE_BOARD_VERSION,
    search_cache_target_version: str = SEARCH_CACHE_VERSION,
    cache_write_chunk: int = 50_000,
    cache_flush_every_moves: int = 100,
    episode_index: int = 1,
    episodes_total: int = 1,
) -> dict:
    board  = [row[:] for row in fixture["board"]]
    score  = fixture.get("score", 0)
    powers = dict(fixture.get("powers", {}))

    name = fixture.get("name", "unnamed")
    desc = fixture.get("description", "")
    cache_target_version = cache_target_version.strip() or SCORE_BOARD_VERSION
    search_cache_target_version = search_cache_target_version.strip() or SEARCH_CACHE_VERSION
    cache_write_chunk = max(1, int(cache_write_chunk))
    cache_flush_every_moves = max(1, int(cache_flush_every_moves))
    cache_written_eval_total = 0
    cache_written_search_total = 0

    def flush_cache_entries(reason: str) -> int:
        nonlocal cache_written_eval_total, cache_written_search_total
        if not write_to_cache:
            return 0
        new_eval_entries = drain_new_entries()
        new_search_entries = drain_search_new_entries()
        if not new_eval_entries and not new_search_entries:
            return 0
        written_eval = 0
        written_search = 0
        if new_eval_entries:
            written_eval = db.save_entries(
                new_eval_entries,
                cache_target_version,
                batch_size=cache_write_chunk,
            )
            cache_written_eval_total += written_eval
        if new_search_entries:
            written_search = db.save_search_entries(
                new_search_entries,
                eval_version=cache_target_version,
                search_version=search_cache_target_version,
                batch_size=cache_write_chunk,
            )
            cache_written_search_total += written_search
        written_total = written_eval + written_search
        print(
            f"  [cache] flushed eval={written_eval:,} search={written_search:,} ({reason}, "
            f"eval_total={cache_written_eval_total:,}, search_total={cache_written_search_total:,}, "
            f"eval_version={cache_target_version!r}, search_version={search_cache_target_version!r})"
        )
        return written_total

    print(f"\n{'═'*54}")
    if episodes_total > 1:
        print(f"  Episode: {episode_index}/{episodes_total}")
    print(f"  Board: {name}")
    if desc:
        print(f"  {desc}")
    print(f"{'═'*54}")
    if write_to_cache:
        print(
            f"  Cache write: enabled  eval_version={cache_target_version!r}  "
            f"search_version={search_cache_target_version!r}  "
            f"flush_every_moves={cache_flush_every_moves}  write_chunk={cache_write_chunk:,}"
        )

    max_tile = max(v for row in board for v in row)
    depth    = fixed_depth if fixed_depth is not None else auto_depth(board)

    print(f"\nInitial state  (score={score}, max={max_tile}, depth={depth})")
    print_board(board, score=score, powers=powers)
    print(f"  Static eval: {score_board(board, powers):.1f}")

    if is_game_over(board):
        print("\nThis board is already game-over — no moves possible.")
        flush_cache_entries("game-over-initial")
        total_written = cache_written_eval_total + cache_written_search_total
        return {
            "cache_written": total_written,
            "cache_written_eval": cache_written_eval_total,
            "cache_written_search": cache_written_search_total,
            "moves_played": 0,
        }

    action_num = 0
    blocked_action_once = None
    while action_num < num_moves:
        max_tile = max(v for row in board for v in row)
        depth    = fixed_depth if fixed_depth is not None else auto_depth(board)
        move_num = action_num + 1

        print(f"\n── Move {move_num}  (depth={depth}, max_tile={max_tile}) {'─'*30}")

        if show_scores:
            print_direction_scores(board, depth, powers=powers)

        board_before = [row[:] for row in board]
        powers_before = dict(powers)
        score_before = score

        t0     = time.perf_counter()
        action = best_action(board, powers, depth=depth)
        think  = (time.perf_counter() - t0) * 1000

        if action is None:
            print("  No valid action — game over.")
            break

        if blocked_action_once is not None and action == blocked_action_once:
            blocked_direction = blocked_action_once[1] if blocked_action_once[0] == "move" else None
            fallback = best_fallback_move(
                board=board,
                powers=powers,
                depth=depth,
                blocked_direction=blocked_direction,
                apply_move_fn=apply_move,
                score_board_fn=score_board,
                expectimax_fn=_expectimax,
            )
            if fallback is not None:
                print(f"  (undo retry) avoid repeating {blocked_action_once} -> {fallback}")
                action = fallback
        blocked_action_once = None

        action_type = action[0]
        planned_eval = projected_action_eval(
            board_before,
            powers_before,
            action,
            score_board_fn=score_board,
            apply_move_fn=apply_move,
            apply_swap_fn=apply_swap,
            apply_delete_fn=apply_delete,
        )

        if action_type == "move":
            direction         = action[1]
            created_512 = count_created_tile(board, direction, 512)
            _, delta, _ = apply_move(board, direction)
            extra = f"  (+D{created_512})" if created_512 > 0 else ""
            print(f"  → move {direction}  (+{delta} pts){extra}  [{think:.0f} ms]")

        elif action_type == "swap":
            _, r1, c1, r2, c2 = action
            v1, v2 = board[r1][c1], board[r2][c2]
            print(f"  → SWAP {v1}@({r1},{c1}) ↔ {v2}@({r2},{c2})  [{think:.0f} ms]")

        elif action_type == "delete":
            _, value, row, col = action
            print(f"  → DELETE all {value}-tiles  [{think:.0f} ms]")
        else:
            print(f"  Unknown action type: {action_type!r}")
            break

        try:
            board, score, powers = apply_action(
                board,
                score,
                powers,
                action,
                rng=rng,
                no_random=no_random,
            )
        except ValueError as exc:
            print(f"  Action failed: {exc}")
            break

        action_num += 1

        print_board(board, score=score, powers=powers)
        eval_after = score_board(board, powers)
        print(f"  Static eval: {eval_after:.1f}")

        undo_decision = analyze_undo(
            board_before=board_before,
            powers_before=powers_before,
            board_after=board,
            powers_after=powers,
            planned_eval=planned_eval,
            score_board_fn=score_board,
            apply_move_fn=apply_move,
        )
        if undo_decision.should_undo and action_num < num_moves:
            powers = dict(powers_before)
            powers["undo"] = max(0, powers.get("undo", 0) - 1)
            board = [row[:] for row in board_before]
            score = score_before
            blocked_action_once = action
            action_num += 1
            reasons = ",".join(undo_decision.reasons)
            print(
                f"  → UNDO ({reasons}) "
                f"drop={undo_decision.eval_drop:.1f}/{undo_decision.drop_trigger:.1f} "
                f"gap={undo_decision.plan_gap:.1f}/{undo_decision.gap_trigger:.1f} "
                f"drop%={undo_decision.eval_drop_ratio*100:.1f} "
                f"gap%={undo_decision.plan_gap_ratio*100:.1f}"
            )
            print_board(board, score=score, powers=powers)
            print(f"  Static eval after undo: {score_board(board, powers):.1f}")

        if action_num % cache_flush_every_moves == 0:
            flush_cache_entries(f"after {action_num} action(s)")

        if is_game_over(board):
            print("\n  Game over — no moves remain.")
            break

        if peek:
            print("\n(--peek: stopping after first move)")
            break

    max_tile = max(v for row in board for v in row)
    flush_cache_entries("final")
    print(f"\nFinal  score={score}  max_tile={max_tile}")
    if write_to_cache:
        print(
            "Cache write summary: "
            f"eval={cache_written_eval_total:,} (version={cache_target_version!r})  "
            f"search={cache_written_search_total:,} (version={search_cache_target_version!r})"
        )
    total_written = cache_written_eval_total + cache_written_search_total
    return {
        "cache_written": total_written,
        "cache_written_eval": cache_written_eval_total,
        "cache_written_search": cache_written_search_total,
        "moves_played": action_num,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the 2048 strategy against fixture board states (no browser).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  .venv/bin/python tests/run.py                       # list all boards
  .venv/bin/python tests/run.py mid_game              # run 10 moves with auto depth
  .venv/bin/python tests/run.py late_game --depth 5 --moves 20
  .venv/bin/python tests/run.py jammed --scores       # show eval score for each direction
  .venv/bin/python tests/run.py swap_test --peek      # show only the first action
  .venv/bin/python tests/run.py mid_game --seed 42    # reproducible random tile spawns
  .venv/bin/python tests/run.py mid_game --no-random  # skip tile spawning (pure decision trace)
  .venv/bin/python tests/run.py late_game --depth 2 --moves 80 --episodes 20 --write-to-cache
""",
    )
    parser.add_argument(
        "board", nargs="?",
        help="Board fixture name or path. Omit to list all available fixtures.",
    )
    parser.add_argument(
        "--moves", "-n", type=int, default=10,
        help="Number of moves to simulate (default: 10)",
    )
    parser.add_argument(
        "--depth", "-d", type=str, default="auto",
        help="Search depth: integer or 'auto' to follow live board-state adaptive depth (default: auto)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for tile placement (default: random each run)",
    )
    parser.add_argument(
        "--scores", action="store_true",
        help="Before each move, print the expectimax score for every direction",
    )
    parser.add_argument(
        "--peek", action="store_true",
        help="Compute and display only the first action, then stop",
    )
    parser.add_argument(
        "--no-random", dest="no_random", action="store_true",
        help="Do not place random tiles after moves (trace decisions without noise)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Run this fixture repeatedly to explore more board states (default: 1)",
    )
    parser.add_argument(
        "--write-to-cache", action="store_true",
        help="Flush newly evaluated states from this run into SQLite cache",
    )
    parser.add_argument(
        "--cache-target-version", type=str, default=SCORE_BOARD_VERSION,
        help="Version label for cache writes (default: SCORE_BOARD_VERSION)",
    )
    parser.add_argument(
        "--search-cache-target-version", type=str, default=SEARCH_CACHE_VERSION,
        help="Search cache version label for cache writes (default: SEARCH_CACHE_VERSION)",
    )
    parser.add_argument(
        "--cache-write-chunk", type=int, default=50_000,
        help="Batch size for SQLite writes when --write-to-cache is enabled (default: 50000)",
    )
    parser.add_argument(
        "--cache-flush-every-moves", type=int, default=100,
        help="Flush pending cache entries every N actions (default: 100)",
    )
    args = parser.parse_args()
    print(f"Using SCORE_BOARD_VERSION={SCORE_BOARD_VERSION!r}")
    print(f"Using SEARCH_CACHE_VERSION={SEARCH_CACHE_VERSION!r}")

    if args.board is None:
        print(f"Available boards in {BOARDS_DIR.relative_to(ROOT)}:")
        list_boards()
        return

    if args.depth == "auto":
        fixed_depth = None
    else:
        try:
            fixed_depth = int(args.depth)
            if fixed_depth < 1:
                raise ValueError
        except ValueError:
            parser.error(f"--depth must be a positive integer or 'auto', got: {args.depth!r}")

    if args.episodes < 1:
        parser.error(f"--episodes must be >= 1, got: {args.episodes!r}")
    if args.cache_write_chunk < 1:
        parser.error(f"--cache-write-chunk must be >= 1, got: {args.cache_write_chunk!r}")
    if args.cache_flush_every_moves < 1:
        parser.error(f"--cache-flush-every-moves must be >= 1, got: {args.cache_flush_every_moves!r}")

    fixture = load_fixture(args.board)
    rng     = random.Random(args.seed)

    if args.seed is not None:
        print(f"(random seed: {args.seed})")

    total_written = 0
    total_written_eval = 0
    total_written_search = 0
    total_actions = 0
    for episode in range(1, args.episodes + 1):
        result = run(
            fixture=fixture,
            num_moves=args.moves,
            fixed_depth=fixed_depth,
            rng=rng,
            show_scores=args.scores,
            peek=args.peek,
            no_random=args.no_random,
            write_to_cache=args.write_to_cache,
            cache_target_version=args.cache_target_version,
            search_cache_target_version=args.search_cache_target_version,
            cache_write_chunk=args.cache_write_chunk,
            cache_flush_every_moves=args.cache_flush_every_moves,
            episode_index=episode,
            episodes_total=args.episodes,
        )
        total_written += int(result.get("cache_written", 0))
        total_written_eval += int(result.get("cache_written_eval", 0))
        total_written_search += int(result.get("cache_written_search", 0))
        total_actions += int(result.get("moves_played", 0))

    if args.episodes > 1:
        print(
            f"\nEpisodes summary: episodes={args.episodes}  "
            f"moves_played={total_actions:,}  cache_written={total_written:,}  "
            f"(eval={total_written_eval:,}, search={total_written_search:,})"
        )


if __name__ == "__main__":
    main()
