#!/usr/bin/env python3
"""Tests for expectimax search-node transposition cache behavior."""

import json
import unittest
from pathlib import Path

from strategy import _expectimax, get_search_trans_stats, reset_search_trans_cache


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestStrategySearchCache(unittest.TestCase):
    def test_repeated_same_state_hits_search_cache(self) -> None:
        board, powers = _load_board("mid_game")
        reset_search_trans_cache()
        _expectimax([row[:] for row in board], depth=3, is_max=True, powers=powers)
        first = get_search_trans_stats()
        _expectimax([row[:] for row in board], depth=3, is_max=True, powers=powers)
        second = get_search_trans_stats()
        self.assertGreater(second["hits"], first["hits"])
        self.assertEqual(second["size"], first["size"])

    def test_depth_is_part_of_search_cache_key(self) -> None:
        board, powers = _load_board("mid_game")
        reset_search_trans_cache()
        _expectimax([row[:] for row in board], depth=2, is_max=True, powers=powers)
        after_d2 = get_search_trans_stats()
        _expectimax([row[:] for row in board], depth=3, is_max=True, powers=powers)
        after_d3 = get_search_trans_stats()
        self.assertGreater(after_d3["size"], after_d2["size"])
        self.assertGreater(after_d3["misses"], after_d2["misses"])


if __name__ == "__main__":
    unittest.main()
