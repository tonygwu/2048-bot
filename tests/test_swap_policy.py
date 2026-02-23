#!/usr/bin/env python3
"""Regression tests for swap candidate selection and value."""

import json
import unittest
from pathlib import Path

import strategy_search
from strategy import DIRECTIONS, apply_move, apply_swap, best_action, normalize_powers, score_board
from strategy_search import _expectimax


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


def _best_swap_exhaustive(
    board: list[list[int]], powers_after: dict, depth: int
) -> tuple[tuple[int, int, int, int] | None, float]:
    positions = [(r, c) for r in range(4) for c in range(4) if board[r][c] > 0]
    best_pair: tuple[int, int, int, int] | None = None
    best_val = float("-inf")
    for i in range(len(positions)):
        r1, c1 = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2 = positions[j]
            if board[r1][c1] == board[r2][c2]:
                continue
            nb = apply_swap(board, r1, c1, r2, c2)
            val = _expectimax(nb, depth - 1, False, powers_after)
            if val > best_val:
                best_val = val
                best_pair = (r1, c1, r2, c2)
    return best_pair, best_val


class TestSwapPolicy(unittest.TestCase):
    def test_live_board_prefers_move_over_optional_swap(self) -> None:
        board, powers = _load_board("swap_low_tile_live_rescue")
        action = best_action(board, powers, depth=3)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move")

    def test_no_move_board_chooses_low_tile_rescue_swap(self) -> None:
        board, powers = _load_board("swap_low_tile_rescue")
        for direction in DIRECTIONS:
            with self.subTest(direction=direction):
                self.assertFalse(apply_move(board, direction)[2])

        action = best_action(board, powers, depth=3)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "swap")
        _, r1, c1, r2, c2 = action
        self.assertEqual({(r1, c1), (r2, c2)}, {(0, 1), (3, 0)})

    def test_swap_shortlist_keeps_exhaustive_best_pair(self) -> None:
        board, powers = _load_board("swap_test")
        powers = normalize_powers(powers)
        powers_after = {**powers, "swap": max(0, powers["swap"] - 1)}
        best_pair, _ = _best_swap_exhaustive(board, powers_after, depth=4)
        self.assertIsNotNone(best_pair)

        shortlist = strategy_search._swap_candidate_pairs(board, powers_after)
        self.assertIn(best_pair, shortlist)

    def test_obvious_swap_substantially_increases_static_eval(self) -> None:
        board, powers = _load_board("swap_low_tile_rescue")
        base = score_board(board, powers)
        improved = apply_swap(board, 0, 1, 3, 0)
        after = score_board(improved, {"undo": 0, "swap": 0, "delete": 0})
        self.assertGreater(after - base, 5000.0)


if __name__ == "__main__":
    unittest.main()
