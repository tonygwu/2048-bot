#!/usr/bin/env python3
"""Regression tests for strategy action APIs."""

import json
import unittest
from pathlib import Path

from strategy import (
    DeleteAction,
    MoveAction,
    SwapAction,
    action_from_tuple,
    action_to_tuple,
    best_action,
    best_action_obj,
)


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestStrategyActions(unittest.TestCase):
    def test_action_tuple_roundtrip(self) -> None:
        actions = [
            MoveAction(direction="right"),
            SwapAction(r1=0, c1=0, r2=1, c2=1),
            DeleteAction(value=4, row=2, col=3),
            None,
        ]
        for action in actions:
            with self.subTest(action=action):
                self.assertEqual(action_from_tuple(action_to_tuple(action)), action)

    def test_best_action_backcompat_matches_typed_api(self) -> None:
        fixtures = ["early_game", "mid_game", "late_game", "jammed", "swap_test", "corner_trap"]
        for name in fixtures:
            board, powers = _load_board(name)
            with self.subTest(fixture=name):
                self.assertEqual(
                    best_action(board, powers, depth=4),
                    action_to_tuple(best_action_obj(board, powers, depth=4)),
                )

    def test_regression_actions_on_core_fixtures(self) -> None:
        expected = {
            "early_game": ("move", "right"),
            "mid_game": ("move", "right"),
            "late_game": ("move", "right"),
            "jammed": ("delete", 4, 1, 3),
            "swap_test": ("move", "right"),
            "corner_trap": ("move", "down"),
        }
        for name, action in expected.items():
            board, powers = _load_board(name)
            with self.subTest(fixture=name):
                self.assertEqual(best_action(board, powers, depth=4), action)


if __name__ == "__main__":
    unittest.main()
