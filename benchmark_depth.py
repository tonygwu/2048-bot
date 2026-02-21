"""
benchmark_depth.py — Calibrate the adaptive depth heuristic for bot.py.

Plays fixed board positions at depths 2–5, averages over multiple random seeds,
and reports how much extra depth improves the final score_board, and how much
it costs in wall-clock time.

Run:
    .venv/bin/python benchmark_depth.py

The output drives depth-schedule tuning decisions in bot.py.
"""

import argparse
import random
import time

from sim_utils import place_random_tile
from strategy import (
    apply_move,
    best_action,
    is_game_over,
    score_board,
    _roughness,
    _monotonicity,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_N_MOVES = 25     # moves per simulation run
DEFAULT_N_SEEDS = 5      # random seeds for tile placement
DEFAULT_DEPTHS = [2, 3, 4, 5]

# ── Board Fixtures ─────────────────────────────────────────────────────────────
# 8 boards spanning (empty_cells, roughness, max_tile) space.
# All are legal 2048 positions (tiles are powers of 2, row sums are sane).

BOARD_STATES: dict[str, dict] = {

    # 12 empty, max=64. Very early game, anything goes.
    "early_open": {
        "desc": "12 empty, max=64, very open",
        "board": [
            [ 64,  0,  0,  0],
            [ 32, 16,  0,  0],
            [  8,  0,  0,  0],
            [  0,  0,  0,  0],
        ],
    },

    # 8 empty, max=512, perfect snake ordering. Moves are obvious.
    "mid_open": {
        "desc": "8 empty, max=512, clean monotone",
        "board": [
            [512, 256, 128, 64],
            [ 32,  16,   0,  0],
            [  0,   0,   0,  0],
            [  0,   0,   0,  0],
        ],
    },

    # 6 empty, max=512, scattered layout — needs surgery.
    "mid_messy": {
        "desc": "6 empty, max=512, scattered",
        "board": [
            [512,  32,  64,  0],
            [ 16, 256,   4,  0],
            [  8,  32,   2,  0],
            [  4,   0,   0,  0],
        ],
    },

    # 5 empty, max=4096, well-ordered snake. Current auto_depth → 5.
    "late_clean": {
        "desc": "5 empty, max=4096, monotone snake",
        "board": [
            [4096, 2048, 1024, 512],
            [ 128,  256,  512, 256],
            [  64,   32,   16,   0],
            [   0,    0,    0,   0],
        ],
    },

    # 4 empty, max=2048, poor monotonicity. The hardest mid-game regime.
    "late_messy": {
        "desc": "4 empty, max=2048, scrambled",
        "board": [
            [2048,  64, 512, 1024],
            [ 256,  32, 128,   16],
            [   8,  64,   4,    0],
            [   0,   0,   0,    2],
        ],
    },

    # 3 empty, max=1024. Crisis: cheap to search deep, critical to get right.
    "crisis_low_max": {
        "desc": "3 empty, max=1024, crisis",
        "board": [
            [1024, 512, 256, 128],
            [  64,  32,  16,   8],
            [   4,   2,   4,   2],
            [   0,   0,   0,   2],
        ],
    },

    # 1 empty, max=4096. Real board from the game crash at move 3630.
    "crisis_high_max": {
        "desc": "1 empty, max=4096, near-death",
        "board": [
            [   8,    2, 4096, 2048],
            [ 256,    8,  128, 1024],
            [   8,   64,   16,    8],
            [   4,    2,   32,    0],
        ],
    },

    # 10 empty, max=8192. Open board with monster tile.
    # Current auto_depth returns depth 6 (341ms/move). Is that warranted?
    "big_tile_open": {
        "desc": "10 empty, max=8192, very open",
        "board": [
            [8192, 4096, 2048, 1024],
            [ 512,  256,    0,    0],
            [   0,    0,    0,    0],
            [   0,    0,    0,    0],
        ],
    },
}


def play_from(board: list[list[int]], depth: int, n_moves: int, seed: int) -> tuple[int, float]:
    """
    Play up to n_moves moves from board at the given depth.
    Returns (moves_survived, final_score_board).
    """
    rng = random.Random(seed)
    b = [row[:] for row in board]
    survived = 0
    for _ in range(n_moves):
        if is_game_over(b):
            break
        action = best_action(b, powers={}, depth=depth)
        if action is None:
            break
        if action[0] == "move":
            nb, _, changed = apply_move(b, action[1])
            if not changed:
                break
            b = place_random_tile(nb, rng)
        survived += 1
    return survived, score_board(b)


def board_stats(board: list[list[int]]) -> dict:
    n_empty = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_tile = max(board[r][c] for r in range(4) for c in range(4))
    rough = _roughness(board)
    mono = _monotonicity(board)
    return {
        "empty": n_empty, "max_tile": max_tile,
        "roughness": rough, "monotonicity": mono,
    }


# ── Main benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(n_moves: int, n_seeds: int, depths: list[int]) -> None:
    print("=" * 74)
    print(f"  benchmark_depth.py  —  N_MOVES={n_moves}  N_SEEDS={n_seeds}  DEPTHS={depths}")
    print("=" * 74)

    # Board properties header
    print()
    print(f"  {'Board':<20}  {'empty':>5}  {'max':>5}  {'rough':>6}  {'mono':>7}  {'score0':>10}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*10}")
    for name, spec in BOARD_STATES.items():
        s = board_stats(spec["board"])
        print(
            f"  {name:<20}  {s['empty']:>5}  {s['max_tile']:>5}  "
            f"{s['roughness']:>6.1f}  {s['monotonicity']:>7.1f}  "
            f"{score_board(spec['board']):>10.0f}"
        )

    # Run simulations
    # results[name][depth] = {"avg_score", "avg_moves", "total_time"}
    results: dict[str, dict[int, dict]] = {}

    print()
    print(f"  {'Board':<20}  {'D':>2}  {'avg_score':>10}  {'avg_moves':>9}  {'ms/move':>8}")
    print(f"  {'-'*20}  {'-'*2}  {'-'*10}  {'-'*9}  {'-'*8}")

    for name, spec in BOARD_STATES.items():
        results[name] = {}
        for depth in depths:
            t0 = time.perf_counter()
            runs = [play_from(spec["board"], depth, n_moves, seed=s) for s in range(n_seeds)]
            elapsed = time.perf_counter() - t0

            avg_moves = sum(r[0] for r in runs) / n_seeds
            avg_score = sum(r[1] for r in runs) / n_seeds
            ms_per_move = elapsed / (n_seeds * max(avg_moves, 0.01)) * 1000

            results[name][depth] = {
                "avg_score": avg_score,
                "avg_moves": avg_moves,
                "total_time": elapsed,
                "ms_per_move": ms_per_move,
            }
            print(
                f"  {name:<20}  {depth:>2}  {avg_score:>10.0f}  "
                f"{avg_moves:>9.1f}  {ms_per_move:>7.1f}ms"
            )
        print()

    # Improvement table: % change D → D+1
    depth_pairs = list(zip(depths, depths[1:]))

    print()
    print("  Depth improvement (% Δ avg_score going D → D+1).  * = |Δ| ≥ 5%")
    hdr = f"  {'Board':<20}  {'empty':>5}  {'rough':>5}"
    for d_lo, d_hi in depth_pairs:
        hdr += f"  {d_lo}→{d_hi}:Δ%".rjust(9)
    print(hdr)
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}" + "  " + "  ".join(["-"*7]*len(depth_pairs)))

    for name, spec in BOARD_STATES.items():
        s = board_stats(spec["board"])
        row = f"  {name:<20}  {s['empty']:>5}  {s['roughness']:>5.1f}"
        for d_lo, d_hi in depth_pairs:
            lo = results[name][d_lo]["avg_score"]
            hi = results[name][d_hi]["avg_score"]
            pct = (hi - lo) / abs(lo) * 100 if lo != 0 else 0.0
            star = "*" if abs(pct) >= 5 else " "
            row += f"  {pct:>+6.1f}%{star}"
        print(row)

    # ms/move table
    print()
    print("  ms per move at each depth:")
    hdr2 = f"  {'Board':<20}"
    for d in depths:
        hdr2 += f"  D={d}".rjust(10)
    print(hdr2)
    print(f"  {'-'*20}" + "  " + "  ".join(["-"*8]*len(depths)))

    for name in BOARD_STATES:
        row = f"  {name:<20}"
        for d in depths:
            ms = results[name][d]["ms_per_move"]
            row += f"  {ms:>7.1f}ms"
        print(row)

    # Final summary: which boards benefit from depth 4→5?
    if 4 in depths and 5 in depths:
        print()
        print("  Summary — depth 4→5 cost vs gain:")
        print(f"  {'Board':<20}  {'4→5 Δ%':>8}  {'D=5 ms/mv':>10}  {'verdict'}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*30}")

        for name in BOARD_STATES:
            r4 = results[name][4]
            r5 = results[name][5]
            pct = (r5["avg_score"] - r4["avg_score"]) / abs(r4["avg_score"]) * 100 if r4["avg_score"] else 0
            ms5 = r5["ms_per_move"]

            if ms5 < 50 and pct >= 3.0:
                verdict = "USE depth 5 (cheap + real gain)"
            elif ms5 < 50 and pct < 1.0:
                verdict = "depth 4 OK (cheap, no gain)"
            elif ms5 >= 80 and pct < 3.0:
                verdict = "STAY at 4 (expensive, no gain)"
            elif ms5 >= 80 and pct >= 5.0:
                verdict = "BORDERLINE (expensive but gains)"
            else:
                verdict = "neutral"
            print(
                f"  {name:<20}  {pct:>+7.1f}%  {ms5:>9.1f}ms  {verdict}"
            )


def _parse_depths(raw: str) -> list[int]:
    depths = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(depths) < 2:
        raise ValueError("--depths must contain at least two depths")
    if any(d <= 0 for d in depths):
        raise ValueError("--depths values must be positive integers")
    return depths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark expectimax depth tradeoffs.")
    parser.add_argument("--moves", type=int, default=DEFAULT_N_MOVES, help=f"Moves per simulation (default: {DEFAULT_N_MOVES})")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS, help=f"Random seeds per board/depth (default: {DEFAULT_N_SEEDS})")
    parser.add_argument(
        "--depths",
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help=f"Comma-separated depth list (default: {','.join(str(d) for d in DEFAULT_DEPTHS)})",
    )
    args = parser.parse_args()

    if args.moves <= 0:
        parser.error("--moves must be > 0")
    if args.seeds <= 0:
        parser.error("--seeds must be > 0")
    try:
        depths = _parse_depths(args.depths)
    except ValueError as e:
        parser.error(str(e))

    run_benchmark(args.moves, args.seeds, depths)


if __name__ == "__main__":
    main()
