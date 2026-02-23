#!/usr/bin/env python3
"""Regression tests for late-stage power-up spend margin shaping."""

import json
import unittest
from pathlib import Path

import strategy_search as search_mod


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestPowerupMarginPolicy(unittest.TestCase):
    def test_late_stage_low_runner_up_relaxes_margin(self) -> None:
        # max=8192, second=1024 -> promotion stage, runner-up is behind.
        late_board = [
            [8192, 1024, 256, 0],
            [128, 64, 32, 0],
            [16, 8, 4, 0],
            [2, 0, 0, 0],
        ]
        # max=8192, second=4096 -> runner-up already healthy.
        advanced_board = [
            [8192, 4096, 256, 0],
            [128, 64, 32, 0],
            [16, 8, 4, 0],
            [2, 0, 0, 0],
        ]
        late_margin = search_mod._required_powerup_advantage(late_board, uses_left=2, kind="swap")
        advanced_margin = search_mod._required_powerup_advantage(advanced_board, uses_left=2, kind="swap")
        self.assertLess(late_margin, advanced_margin)

    def test_last_use_requires_higher_margin_than_first_use(self) -> None:
        board, _ = _load_board("late_8192_swap_double_spend_trap")
        swap_last_use = search_mod._required_powerup_advantage(board, uses_left=1, kind="swap")
        swap_two_uses = search_mod._required_powerup_advantage(board, uses_left=2, kind="swap")
        self.assertGreater(swap_last_use, swap_two_uses)

        delete_last_use = search_mod._required_powerup_advantage(board, uses_left=1, kind="delete")
        delete_two_uses = search_mod._required_powerup_advantage(board, uses_left=2, kind="delete")
        self.assertGreater(delete_last_use, delete_two_uses)


if __name__ == "__main__":
    unittest.main()
