#!/usr/bin/env python3
"""Tests for strategy evaluation decomposition and cache behavior."""

import json
import unittest
from pathlib import Path

import strategy_eval
from strategy import (
    EvalWeights,
    extract_eval_features,
    get_trans_stats,
    load_trans_table,
    normalize_powers,
    reset_trans_stats,
    score_board,
    score_from_features,
)
from transposition_cache import TranspositionCache


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestStrategyEval(unittest.TestCase):
    def test_score_from_features_matches_score_board(self) -> None:
        board, powers = _load_board("mid_game")
        features = extract_eval_features(board, powers)
        self.assertAlmostEqual(score_from_features(features), score_board(board, powers), places=9)

    def test_zero_weights_zero_out_eval(self) -> None:
        board, powers = _load_board("late_game")
        features = extract_eval_features(board, powers)
        zero = EvalWeights(
            empty=0.0,
            gradient=0.0,
            monotonicity=0.0,
            roughness=0.0,
            merge_potential=0.0,
            max_tile_log=0.0,
            sum_log=0.0,
            mobility=0.0,
            corner=0.0,
            powerup=0.0,
        )
        self.assertEqual(score_from_features(features, zero), 0.0)

    def test_score_cache_hit_miss_accounting(self) -> None:
        board, powers = _load_board("corner_trap")
        reset_trans_stats()
        _ = score_board(board, powers)
        _ = score_board(board, powers)
        stats = get_trans_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hits"], 1)

    def test_oversized_preload_preserves_table(self) -> None:
        board, powers = _load_board("mid_game")
        original_cache = strategy_eval._TRANS_CACHE
        try:
            test_cache = TranspositionCache(cap=2)
            strategy_eval._TRANS_CACHE = test_cache
            load_trans_table({(1, 0, 0): 1.0, (2, 0, 0): 2.0, (3, 0, 0): 3.0})
            before = len(test_cache.table)
            _ = score_board(board, powers)
            self.assertEqual(len(test_cache.table), before)
            self.assertGreaterEqual(len(test_cache.new_entries), 1)
        finally:
            strategy_eval._TRANS_CACHE = original_cache

    def test_normalize_powers_caps_uses_to_two(self) -> None:
        self.assertEqual(
            normalize_powers({"undo": 7, "swap": 5, "delete": 3}),
            {"undo": 2, "swap": 2, "delete": 2},
        )
        self.assertEqual(
            normalize_powers({"undo": -1, "swap": -9, "delete": 1}),
            {"undo": 0, "swap": 0, "delete": 1},
        )

    def test_score_board_cache_key_clamps_power_counts(self) -> None:
        board, _ = _load_board("mid_game")
        original_cache = strategy_eval._TRANS_CACHE
        try:
            test_cache = TranspositionCache(cap=100)
            strategy_eval._TRANS_CACHE = test_cache
            v1 = score_board(board, {"swap": 5, "delete": 9})
            v2 = score_board(board, {"swap": 2, "delete": 2})
            self.assertAlmostEqual(v1, v2, places=9)
            self.assertEqual(len(test_cache.table), 1)
            stats = get_trans_stats()
            self.assertEqual(stats["misses"], 1)
            self.assertGreaterEqual(stats["hits"], 1)
        finally:
            strategy_eval._TRANS_CACHE = original_cache


if __name__ == "__main__":
    unittest.main()
