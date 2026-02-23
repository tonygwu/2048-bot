#!/usr/bin/env python3
"""Regression tests for delete power-up candidate selection and scoring."""

import json
import unittest
from pathlib import Path

from strategy import apply_delete, best_action, score_board
from strategy_search import _delete_candidates


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestDeletePolicy(unittest.TestCase):
    def test_candidates_include_singleton_mid_values_on_jammed_late_boards(self) -> None:
        board_128, _ = _load_board("delete_singleton_128_blocker")
        cands_128 = _delete_candidates(board_128, empties=1, max_tile=1024)
        self.assertIn(128, cands_128)

        board_256, _ = _load_board("delete_singleton_256_blocker")
        cands_256 = _delete_candidates(board_256, empties=1, max_tile=4096)
        self.assertIn(256, cands_256)

    def test_candidates_keep_anchor_scale_singletons_out(self) -> None:
        board, _ = _load_board("delete_singleton_256_blocker")
        cands = _delete_candidates(board, empties=1, max_tile=4096)
        self.assertNotIn(4096, cands)

    def test_best_action_prefers_move_over_singleton_delete_candidates(self) -> None:
        board_128, powers_128 = _load_board("delete_singleton_128_blocker")
        action_128 = best_action(board_128, powers_128, depth=4)
        self.assertIsNotNone(action_128)
        self.assertEqual(action_128[0], "move")

        board_256, powers_256 = _load_board("delete_singleton_256_blocker")
        action_256 = best_action(board_256, powers_256, depth=4)
        self.assertIsNotNone(action_256)
        self.assertEqual(action_256[0], "move")

    def test_sparse_8192_delete_trap_can_prefer_non_delete(self) -> None:
        board, powers = _load_board("promotion_8192_delete_undo_trap_b")
        action = best_action(board, powers, depth=4)
        self.assertIsNotNone(action)
        self.assertNotEqual(action[0], "delete")

    def test_obvious_delete_choices_raise_static_eval(self) -> None:
        base_powers = {"undo": 0, "swap": 0, "delete": 0}

        jammed_board, _ = _load_board("jammed")
        jammed_gain = score_board(apply_delete(jammed_board, 4), base_powers) - score_board(jammed_board, base_powers)
        self.assertGreater(jammed_gain, 1000.0)

        high_board, _ = _load_board("powerup_delete_high_value")
        high_gain = score_board(apply_delete(high_board, 128), base_powers) - score_board(high_board, base_powers)
        self.assertGreater(high_gain, 1200.0)


if __name__ == "__main__":
    unittest.main()
