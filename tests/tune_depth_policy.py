#!/usr/bin/env python3
"""
Tune adaptive depth policy weights with local fixture rollouts.

This script does not launch a browser. It simulates from fixture states,
runs best_action for each move, and evaluates final static board quality.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from strategy import (  # noqa: E402
    DIRECTIONS,
    apply_delete,
    apply_move,
    apply_swap,
    best_action,
    empty_cells,
    score_board,
)


BOARDS_DIR = ROOT / "tests" / "boards"
DEFAULT_FIXTURES = [
    "early_game",
    "mid_game",
    "late_game",
    "jammed",
    "corner_trap",
    "swap_test",
    "high_tile_sparse",
    "messy_lowmax",
]


@dataclass(frozen=True)
class Params:
    base: float
    w_max: float
    w_full: float
    w_block: float
    w_rough: float


def load_fixture(name: str) -> dict:
    p = BOARDS_DIR / f"{name}.json"
    with open(p) as f:
        return json.load(f)


def roughness(board: list[list[int]]) -> float:
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v <= 0:
                continue
            lv = math.log2(v)
            if c + 1 < 4 and board[r][c + 1] > 0:
                s += abs(lv - math.log2(board[r][c + 1]))
            if r + 1 < 4 and board[r + 1][c] > 0:
                s += abs(lv - math.log2(board[r + 1][c]))
    return s


def valid_moves(board: list[list[int]]) -> int:
    return sum(1 for d in DIRECTIONS if apply_move(board, d)[2])


def choose_depth(board: list[list[int]], p: Params) -> int:
    empties = len(empty_cells(board))
    max_tile = max(max(row) for row in board)
    max_log = math.log2(max_tile) if max_tile > 0 else 0.0
    fullness = (16 - empties) / 16.0

    vm = valid_moves(board)
    blocked = (4 - vm) / 3.0
    rough_n = min(1.0, roughness(board) / 36.0)

    score = (
        p.base
        + p.w_max * max_log
        + p.w_full * fullness
        + p.w_block * blocked
        + p.w_rough * rough_n
    )

    if empties >= 8:
        score -= 0.60
    elif empties <= 2:
        score += 0.55

    if vm <= 2:
        score += 0.40

    return max(2, min(6, int(round(score))))


def place_random_tile(board: list[list[int]], rng: random.Random) -> list[list[int]]:
    empties = empty_cells(board)
    if not empties:
        return board
    r, c = rng.choice(empties)
    v = 2 if rng.random() < 0.9 else 4
    nb = [row[:] for row in board]
    nb[r][c] = v
    return nb


def rollout(fixture: dict, moves: int, seed: int, params: Params) -> tuple[float, float, float]:
    board = [row[:] for row in fixture["board"]]
    powers = dict(fixture.get("powers", {}))
    score = fixture.get("score", 0)
    rng = random.Random(seed)

    depth_samples: list[int] = []
    think_ms = 0.0
    for _ in range(moves):
        depth = choose_depth(board, params)
        depth_samples.append(depth)
        t0 = time.perf_counter()
        action = best_action(board, powers, depth=depth)
        think_ms += (time.perf_counter() - t0) * 1000
        if action is None:
            break
        kind = action[0]
        if kind == "move":
            _, d = action
            board, delta, changed = apply_move(board, d)
            if not changed:
                break
            score += delta
            board = place_random_tile(board, rng)
        elif kind == "swap":
            _, r1, c1, r2, c2 = action
            board = apply_swap(board, r1, c1, r2, c2)
            powers["swap"] = max(0, powers.get("swap", 0) - 1)
            board = place_random_tile(board, rng)
        elif kind == "delete":
            _, value, _row, _col = action
            board = apply_delete(board, value)
            powers["delete"] = max(0, powers.get("delete", 0) - 1)
            board = place_random_tile(board, rng)

    final_eval = score_board(board, powers) + 0.02 * score
    avg_depth = statistics.mean(depth_samples) if depth_samples else 0.0
    return final_eval, think_ms, avg_depth


def baseline_depth(board: list[list[int]]) -> int:
    max_tile = max(max(row) for row in board)
    if max_tile < 512:
        return 2
    if max_tile < 2048:
        return 3
    if max_tile < 4096:
        return 4
    if max_tile < 8192:
        return 5
    return 6


def rollout_baseline(fixture: dict, moves: int, seed: int) -> tuple[float, float, float]:
    # Evaluate the legacy schedule in the same rollout machinery.
    p = Params(0.0, 0.0, 0.0, 0.0, 0.0)

    def choose(board: list[list[int]]) -> int:
        return baseline_depth(board)

    board = [row[:] for row in fixture["board"]]
    powers = dict(fixture.get("powers", {}))
    score = fixture.get("score", 0)
    rng = random.Random(seed)

    depth_samples: list[int] = []
    think_ms = 0.0
    for _ in range(moves):
        depth = choose(board)
        depth_samples.append(depth)
        t0 = time.perf_counter()
        action = best_action(board, powers, depth=depth)
        think_ms += (time.perf_counter() - t0) * 1000
        if action is None:
            break
        kind = action[0]
        if kind == "move":
            _, d = action
            board, delta, changed = apply_move(board, d)
            if not changed:
                break
            score += delta
            board = place_random_tile(board, rng)
        elif kind == "swap":
            _, r1, c1, r2, c2 = action
            board = apply_swap(board, r1, c1, r2, c2)
            powers["swap"] = max(0, powers.get("swap", 0) - 1)
            board = place_random_tile(board, rng)
        elif kind == "delete":
            _, value, _row, _col = action
            board = apply_delete(board, value)
            powers["delete"] = max(0, powers.get("delete", 0) - 1)
            board = place_random_tile(board, rng)

    final_eval = score_board(board, powers) + 0.02 * score
    avg_depth = statistics.mean(depth_samples) if depth_samples else 0.0
    return final_eval, think_ms, avg_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune depth policy weights over fixture rollouts.")
    parser.add_argument("--moves", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    fixtures = [load_fixture(name) for name in DEFAULT_FIXTURES]

    base_values = [0.70, 1.00, 1.30]
    w_max_values = [0.18, 0.22, 0.26]
    w_full_values = [0.85, 1.10, 1.35]
    w_block_values = [0.60, 0.85, 1.10]
    w_rough_values = [0.25, 0.45, 0.65]

    candidates = [
        Params(*vals)
        for vals in itertools.product(
            base_values, w_max_values, w_full_values, w_block_values, w_rough_values
        )
    ]

    baseline_eval = 0.0
    baseline_think = 0.0
    baseline_depth = 0.0
    for fx in fixtures:
        e, t, d = rollout_baseline(fx, moves=args.moves, seed=args.seed)
        baseline_eval += e
        baseline_think += t
        baseline_depth += d
    baseline_depth /= len(fixtures)

    ranked: list[tuple[float, Params, float, float, float]] = []
    for idx, p in enumerate(candidates, start=1):
        total_eval = 0.0
        total_think = 0.0
        total_depth = 0.0
        for fx in fixtures:
            e, t, d = rollout(fx, moves=args.moves, seed=args.seed, params=p)
            total_eval += e
            total_think += t
            total_depth += d
        avg_depth = total_depth / len(fixtures)
        eval_ratio = total_eval / baseline_eval if baseline_eval else 0.0
        think_ratio = total_think / baseline_think if baseline_think else 0.0
        # Higher is better. Bias toward quality while mildly penalizing runtime.
        utility = eval_ratio - 0.32 * max(0.0, think_ratio - 1.0)
        ranked.append((utility, p, eval_ratio, think_ratio, avg_depth))
        if idx % 20 == 0:
            print(f"evaluated {idx}/{len(candidates)} candidates...")

    ranked.sort(key=lambda x: x[0], reverse=True)
    print(
        f"Baseline (legacy max-tile schedule): eval={baseline_eval:.1f} "
        f"think_ms={baseline_think:.1f} avg_depth={baseline_depth:.2f}"
    )
    print("\nTop candidates:")
    print("| Rank | base | w_max | w_full | w_block | w_rough | Eval vs base | Think vs base | Avg depth | Utility |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, (u, p, er, tr, ad) in enumerate(ranked[: args.top], start=1):
        print(
            f"| {i} | {p.base:.2f} | {p.w_max:.2f} | {p.w_full:.2f} | {p.w_block:.2f} | {p.w_rough:.2f} | "
            f"{er:.4f}x | {tr:.4f}x | {ad:.2f} | {u:.4f} |"
        )

    best = ranked[0]
    _, p, er, tr, ad = best
    print("\nSuggested params:")
    print(
        f"base={p.base:.2f}, w_max={p.w_max:.2f}, w_full={p.w_full:.2f}, "
        f"w_block={p.w_block:.2f}, w_rough={p.w_rough:.2f}"
    )
    print(
        f"expected eval_ratio={er:.4f}x, think_ratio={tr:.4f}x, avg_depth={ad:.2f}"
    )


if __name__ == "__main__":
    main()
