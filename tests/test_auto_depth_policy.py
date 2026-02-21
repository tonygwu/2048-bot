#!/usr/bin/env python3
"""Regression tests for adaptive depth policy over fixture boards."""

import json
import unittest
from pathlib import Path

from strategy import auto_depth


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> list[list[int]]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"]


class TestAutoDepthPolicy(unittest.TestCase):
    def test_fixture_depth_regression(self) -> None:
        expected = {
            "early_game": 2,
            "mid_game": 3,
            "swap_test": 3,
            "corner_trap": 4,
            "messy_lowmax": 4,
            "high_tile_sparse": 4,
            "jammed": 5,
            "late_game": 6,
        }
        for name, depth in expected.items():
            with self.subTest(fixture=name):
                self.assertEqual(auto_depth(_load_board(name)), depth)


if __name__ == "__main__":
    unittest.main()
