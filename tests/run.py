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

from sim_utils import place_random_tile
from strategy import (
    DIRECTIONS,
    apply_delete,
    apply_move,
    apply_swap,
    auto_depth,
    best_action,
    is_game_over,
    score_board,
    _expectimax,
)


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


def direction_scores(board: list[list[int]], depth: int) -> list[tuple[str, float | None]]:
    """Return (direction, expectimax_value) for each direction, None if the move is invalid."""
    result = []
    for d in DIRECTIONS:
        nb, delta, changed = apply_move(board, d)
        if not changed:
            result.append((d, None))
            continue
        val = _expectimax(nb, max(0, depth - 1), False) + delta
        result.append((d, val))
    return result


def print_direction_scores(board: list[list[int]], depth: int) -> None:
    scores = direction_scores(board, depth)
    valid = [(d, v) for d, v in scores if v is not None]
    best_d = max(valid, key=lambda x: x[1])[0] if valid else None
    print("  Direction scores:")
    for d, val in scores:
        if val is None:
            print(f"    {d:<6}:  (no change)")
        else:
            marker = "  ◀ best" if d == best_d else ""
            print(f"    {d:<6}: {val:>12.1f}{marker}")


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(
    fixture: dict,
    num_moves: int,
    fixed_depth: int | None,
    rng: random.Random,
    show_scores: bool,
    peek: bool,
    no_random: bool,
) -> None:
    board  = [row[:] for row in fixture["board"]]
    score  = fixture.get("score", 0)
    powers = dict(fixture.get("powers", {}))

    name = fixture.get("name", "unnamed")
    desc = fixture.get("description", "")

    print(f"\n{'═'*54}")
    print(f"  Board: {name}")
    if desc:
        print(f"  {desc}")
    print(f"{'═'*54}")

    max_tile = max(v for row in board for v in row)
    depth    = fixed_depth if fixed_depth is not None else auto_depth(board)

    print(f"\nInitial state  (score={score}, max={max_tile}, depth={depth})")
    print_board(board, score=score, powers=powers)
    print(f"  Static eval: {score_board(board):.1f}")

    if is_game_over(board):
        print("\nThis board is already game-over — no moves possible.")
        return

    for move_num in range(1, num_moves + 1):
        max_tile = max(v for row in board for v in row)
        depth    = fixed_depth if fixed_depth is not None else auto_depth(board)

        print(f"\n── Move {move_num}  (depth={depth}, max_tile={max_tile}) {'─'*30}")

        if show_scores:
            print_direction_scores(board, depth)

        t0     = time.perf_counter()
        action = best_action(board, powers, depth=depth)
        think  = (time.perf_counter() - t0) * 1000

        if action is None:
            print("  No valid action — game over.")
            break

        action_type = action[0]

        if action_type == "move":
            direction         = action[1]
            new_board, delta, changed = apply_move(board, direction)
            score            += delta
            print(f"  → move {direction}  (+{delta} pts)  [{think:.0f} ms]")
            if not changed:
                print("  (board didn't change — bug?)")
                break
            board = new_board
            if not no_random:
                board = place_random_tile(board, rng)

        elif action_type == "swap":
            _, r1, c1, r2, c2 = action
            v1, v2 = board[r1][c1], board[r2][c2]
            print(f"  → SWAP {v1}@({r1},{c1}) ↔ {v2}@({r2},{c2})  [{think:.0f} ms]")
            board = apply_swap(board, r1, c1, r2, c2)
            powers = dict(powers)
            powers["swap"] = max(0, powers.get("swap", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)

        elif action_type == "delete":
            _, value, row, col = action
            print(f"  → DELETE all {value}-tiles  [{think:.0f} ms]")
            board = apply_delete(board, value)
            powers = dict(powers)
            powers["delete"] = max(0, powers.get("delete", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)

        print_board(board, score=score, powers=powers)
        print(f"  Static eval: {score_board(board):.1f}")

        if is_game_over(board):
            print("\n  Game over — no moves remain.")
            break

        if peek:
            print("\n(--peek: stopping after first move)")
            break

    max_tile = max(v for row in board for v in row)
    print(f"\nFinal  score={score}  max_tile={max_tile}")


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
    args = parser.parse_args()

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

    fixture = load_fixture(args.board)
    rng     = random.Random(args.seed)

    if args.seed is not None:
        print(f"(random seed: {args.seed})")

    run(
        fixture=fixture,
        num_moves=args.moves,
        fixed_depth=fixed_depth,
        rng=rng,
        show_scores=args.scores,
        peek=args.peek,
        no_random=args.no_random,
    )


if __name__ == "__main__":
    main()
