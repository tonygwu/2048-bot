#!/usr/bin/env python3
"""Regression tests for late-stage power-up spend margin shaping."""

import unittest

import strategy_search as search_mod


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


if __name__ == "__main__":
    unittest.main()
